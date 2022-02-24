import os
import logging

import sotodlib
from sotodlib.core import metadata
import sotodlib.io.metadata as io_meta

from sotodlib.core.metadata import ManifestDb, ManifestScheme

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _MakeCalDb:
    """A class where I try to make it easier to interact with the sotodlib metadata
    scheme for the special types of CalDbs that will interact exactly by obs_id, file, 
    or frame.
    """
    def __init__(self, name, 
                 db_file=None, 
                 version=None, 
                 context=None, 
                 prefix=None,
                 h5prefix=None):
        """
        prefix: place to start creating database if db_file is None
        h5prefix : place to start creating h5 files if h5 files is None
        """
        self.name = name
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

    def __repr__(self):
        return f"CalcCalDb {self.name}"
        
    def calc(self, obs_id, samples=None ):
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
    
    def save(self, obs_id, samples, ids, result, overwrite=False, new_h5=False,
             save_attrs={}):
   
        if new_h5:
            raise NotImplementedError("I don't know how to make a new file")
        rs = metadata.ResultSet(keys=['dets:readout_id', self.name])
        for d,det in enumerate( ids ):
            rs.rows.append( (det, result[d]) )
    
        if not os.path.exists(self.h5prefix):
            os.makedirs(self.h5prefix)
        
        attrs = {'sotodlib.version':sotodlib.__version__}
        attrs.update(save_attrs)
        dataset = f"{obs_id}"
        h5_file = os.path.join(self.h5prefix, self.h5current)
        
        io_meta.write_dataset(rs, h5_file, 
                              dataset, attrs=attrs, overwrite=overwrite, )

        self.db.add_entry({'obs:obs_id': obs_id, 
                           'dataset': dataset},
                           filename=h5_file)
        self.db.to_file(self.db_file)
        
class Make_ObsCalDb(_MakeCalDb):
    
    def __init__(self, name, **kwargs): 
        super().__init__(name, **kwargs)
        
    def calc(self, obs_id, samples=None):
        if samples is not None:
            raise ValueError(f"{type(self).__name__} does not accept samples argument")
            