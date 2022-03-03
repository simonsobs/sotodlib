import os
import datetime as dt
import logging

import sotodlib
from sotodlib.core import Context, metadata
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
        manifest_args : dict or none. Information to add to the Manifest information for
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
            
        if not os.path.exists(db_file):
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('dataset')   
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
        
        if manifest_args is None:
            manifest_args = {}
        self.manifest_args = manifest_args
        
    def __repr__(self):
        return f"CalcCalDb {self.name}"
        
    def calc(self, obs_id, samples=None, *args ):
        """Function written over by child classes, takes in an obs_id and set of 
        sample ranges and returns calculated products on those ranges.
        
        Arguments
        ----------
        obs_id : obs_id
        samples : a range of samples to run and return these calculations
        
        Returns :
        -----------
        ids : [n_dets]
        result : [n_dets, len(samples)]
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
        
        attrs = {'sotodlib.version':sotodlib.__version__}
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
    
    def __init__(self, name, context, **kwargs): 
        super().__init__(name, context, **kwargs)
    
    def write_h5_data(self, h5_file, dataset, samples, 
                      ids, result, attrs, overwrite=False ):
        
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
            