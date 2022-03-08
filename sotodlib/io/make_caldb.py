import os
import h5py
import logging
import numpy as np
import datetime as dt

import sotodlib
from sotodlib.core import Context, metadata, axisman_io
import sotodlib.io.metadata as io_meta

from sotodlib.core.metadata import ManifestDb, ManifestScheme

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _MakeCalDb:
    """A class where I try to make it easier to interact with the sotodlib metadata
    scheme for the special types of CalDbs that contain calculations derived per 
    observation and per detector (readout_id). Each class calculates and saves data for 
    n >= 1 fields, where fields are the names of the data to be loaded into axis managers.
    
    The classes inheriting from this one will define different saving and loading schemes
    for the different types of calculated data. Options:
    
    (1) ObsCalDb -- data is number per field, per obs_id, per detector
    (2) IntervalCalDb -- data is N numbers per field, per obs_id, per detector. N is the 
        same for all detectors. The intervals will be definited by sample numbers so that 
        they could be split by file, frames, or unit time.
    (3) SampleCalDb -- data is n_i numbers per field, per obs_id, per detector. n_i is 
        different for all detectors. The numbers will be indexed to specific sample numbers.
        (Will there be a special type of these for RangesMatrices?)
    
    Actual calculators will inherit from one of those three options. ObsCalDb, IntervalCalDb, and
    SampleCalDb will each define 
    """
    def __init__(self, name,
                 context,
                 db_file=None, 
                 version=None, 
                 prefix=None,
                 h5prefix=None,
                 field_names=None,
                 manifest_args=None):
        """
        Arguments
        -----------
        name : string
            base name that will be used in database / hdf5 file naming
        context : string or loaded context object. used to access loaders and obsfiledb
        prefix : place to start creating database if db_file is None
        h5prefix : place to start creating h5 files if h5 files is None
        field_names : string, list of strings, or None
            names to be used in the result set creation. Must match length of 
            results returned from the calc function. If none, name will be used
        manifest_args : dict or none. Information to add to the Manifest Database for
            each observation.
        """
        self.name = name
        if isinstance(context, str):
            self.context = Context(context)
        else:
            self.context = context
            
        if db_file is None:
            if version is None:
                now = dt.datetime.now()
                version= dt.datetime.strftime( now, "v%y%m%d")
            if prefix is None:
                prefix = ''
            db_file = os.path.join(prefix, f"{self.name}_{version}.sqlite")
            if h5prefix is None:
                h5prefix = os.path.join(prefix, f"{self.name}_{version}")
            logger.warning(f"Will create database at {db_file}")
        else:
            if prefix is None:
                prefix = os.path.split(db_file)[0]
        
        if manifest_args is None:
            manifest_args = {}
        self.manifest_args = manifest_args
        
        if not os.path.exists(db_file):
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('dataset')   
            for k in self.manifest_args:
                scheme.add_data_field(k)
            self.db = ManifestDb(scheme=scheme)
        else:
            self.db = ManifestDb.from_file(db_file)
            
        self.db_file = db_file
        h5files = sorted(self.db.get_files())
        
        if len(h5files) == 0:
            if h5prefix is None:
                self.h5prefix = os.path.join(os.path.split(db_file)[0], name)
            else:
                self.h5prefix = h5prefix
            self.h5current = os.path.split(db_file)[1].split('.')[0]+'_000.h5'
        else:
            self.h5prefix = os.path.split(h5files[0])[0]
            self.h5current = os.path.split(h5files[-1])[1]
        
        if field_names is None:
            self.field_names = [self.name]
        elif isinstance(field_names, str):
            self.field_names = [field_names]
        else:
            self.field_names = field_names
        
        
        
    def __repr__(self):
        return f"CalcCalDb {self.name}"
        
    def calc(self, obs_id, samples=None, *args ):
        """Function written over by child classes, takes in an obs_id and set of 
        sample ranges and returns calculated products on those ranges.
        
        Arguments
        ----------
        obs_id : obs_id
        samples : a list of sample ranges to run and return these calculations
            Ex: [(0, 2000), (2000, 4000), ...]
        
        Returns :
        -----------
        ids : [n_dets]
        samples : None or samples as described in argument
            used when sample range determined within function
        result : [n_dets, len(self.field_list), ...]
            exact result formatting is different for different CalDb Makers
        """
        raise NotImplementedError("calc function must be defined by child classes")
    
    def write_h5_data(self, h5_file, dataset, samples, 
                      ids, result, attrs, overwrite=False ):
        
        raise NotImplementedError("write_h5_data function must be defined by child classes")

    def save(self, obs_id, samples, ids, result, overwrite=False, new_h5=False,
             save_attrs={}):
   
        if new_h5:
            raise NotImplementedError("I don't know how to make a new file")

        if not os.path.exists(self.h5prefix):
            os.makedirs(self.h5prefix)
        
        attrs = {
            'sotodlib.version':sotodlib.__version__,
            'save_time': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        attrs.update(save_attrs)
        
        dataset = f"{obs_id}"
        h5_file = os.path.join(self.h5prefix, self.h5current)
        
        args = {
            'obs:obs_id': obs_id, 
            'dataset': dataset,
        }
        
        add_to_manifest = True
        if self.db.match(args) is not None:
            if not overwrite:
                raise ValueError(f"Entry {args} in ManifestDb already")
            else:
                ## we are overwriting the h5 data but not changing the 
                ## manifest db entry
                add_to_manifest = False
        
        self.write_h5_data(h5_file, dataset, samples, ids, result, attrs, overwrite=overwrite)
        
        if add_to_manifest:
            args.update(self.manifest_args)
            self.db.add_entry(args, filename=h5_file)
            self.db.to_file(self.db_file)
        
class Make_ObsCalDb(_MakeCalDb):
    """This CalDb maker is designed for data that is some N numbers per detector per 
    observation. These N numbers should all apply for the entire observation. An 
    example of this could be a 1/f noise calculator that returns white noise, 
    spectral index, and knee frequency per detector per observation.
    """
    def __init__(self, name, context, **kwargs): 
        super().__init__(name, context, **kwargs)
    
    def write_h5_data(self, h5_file, dataset, samples, 
                      ids, result, attrs, overwrite=False ):
        
        if len(samples) != 1:
            raise ValueError("Expect len(samples)==1 where samples=[(0,obs_n_samps)]"
                            f"received {samples}")
        attrs['n_samps'] = samples[0][1]
        
        rs = metadata.ResultSet(keys=['dets:readout_id', *self.field_names])
        
        if len(self.field_names)==1:
            for d,det in enumerate( ids ):
                rs.rows.append( (det, result[d]) )
        else:
            for d,det in enumerate( ids ):
                rs.rows.append( (det, *result[d]) )
        
        io_meta.write_dataset(rs, h5_file, 
                              dataset, 
                              attrs=attrs, 
                              overwrite=overwrite, )
        
    def calc(self, obs_id, samples=None):
        if samples is not None:
            raise ValueError(f"{type(self).__name__} does not accept samples argument")
        
        flist = self.context.obsfiledb.get_files(obs_id)
        nsamps = [flist[dset][-1][2] for dset in flist]
        if not np.all(nsamps != nsamps[0]):
            raise ValueError("nsamps different for different detsets.")
        return [(0,nsamps[0])]

class Make_SampleCalDb(_MakeCalDb):
    """This CalDb maker is designed for data that is indexed along select samples of 
    an observation. An example of this might be a glitch detection algorithm that returns
    detected glitches at certain samples and their significance at that sample.
    """
    def __init__(self, name, context, **kwargs):
        super().__init__(name, context, **kwargs)
        
    def calc(self, obs_id, samples=None):
        if samples is not None:
            raise ValueError(f"{type(self).__name__} does not accept samples argument")
        
        flist = self.context.obsfiledb.get_files(obs_id)
        nsamps = [flist[dset][-1][2] for dset in flist]
        if not np.all(nsamps != nsamps[0]):
            raise ValueError("nsamps different for different detsets.")
        return [(0,nsamps[0])]
            
    def write_h5_data( self, h5_file, dataset, samples, 
                      ids, result, attrs, overwrite=False):
        if len(ids) != len(result):
            raise ValueError("Expect ids and result to be the same length."
                             f" Received {len(ids)} and {len(result)} instead")
        
        if len(samples) != 1:
            raise ValueError("Expect len(samples)==1 where samples=[(0,obs_n_samps)]"
                            f"received {samples}")
            
        if isinstance(result, list):
            for r in result:
                if len(r) != len(self.field_names):
                    raise ValueError(f"Expect result[item] to have {len(self.field_names)} items."
                                    f"Received {len(r)} instead.")
        elif isinstance(result, RangesMatrix):
            raise NotImplementedError("I don't know how to save RangesMatrix yet")
        else:
            raise ValueError("Result not formatted for hdf5 saving")

        f = h5py.File(h5_file, "a")
        if dataset in f and not overwrite:
            raise ValueError(f"{dataset} already saved in {h5_file}")
        elif dataset in f: 
            del f[dataset]

        f.create_group(dataset)
        for k in attrs:
            f[dataset].attrs[k] = attrs[k]
        f[dataset].attrs['n_samps'] = samples[0][1]
        f[dataset].create_dataset('dets:readout_id', data=axisman_io._retype_for_write(ids) )

        for i, field in enumerate(self.field_names):
            ends = []
            vals = []
            for r in result:
                if len(ends)==0:
                    ends.append(len(r[i]))
                else:
                    ends.append(ends[-1] + len(r[i]))
                vals.append(r[i])
            vals = np.hstack(vals)
            f[dataset].create_group(field)
            f[dataset][field].create_dataset("ends", data=ends)
            f[dataset][field].create_dataset("vals", data=vals)
        
        f.close()