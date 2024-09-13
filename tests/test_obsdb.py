import unittest
import shutil
import tempfile
import numpy as np
from sotodlib.core import metadata

import os
import time

# from ._helpers import mpi_multi

def get_example():
    # Create a new Db and add two columns.
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(['timestamp float', 'hwp_speed float', 'drift string'])

    # Add 10 rows.
    for i in range(10):
        tags = []
        if i == 6:
            tags.append('cryo_problem')
        if i > 7:
            tags.append('planet')
        else:
            tags.append('cmb_survey')
        obsdb.update_obs(f'myobs{i}', {'timestamp': 1900000000. + i * 100,
                                       'hwp_speed': 2.0,
                                       'drift': 'rising' if i%2 else 'setting'},
                         tags=tags)
    return obsdb
    
def make_extdb_pwv(dir_name, file_name):
    # create a extenal manifestdb for pwv, and return temporal filepath to the database
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match('obs:obs_id')
    scheme.add_data_field('pwv')
    extdb_pwv = metadata.ManifestDb(map_file=os.path.join(dir_name, file_name),
                                    scheme=scheme)
    for i in range(10):
        if i in [3, 8]:
            pwv = np.random.uniform(2.1, 3)
        else:
            pwv = np.random.uniform(0, 2)
        entry_dict = {'obs:obs_id': f'myobs{i}', 
                     'pwv': pwv}
        extdb_pwv.add_entry(entry_dict)
    return os.path.join(dir_name, file_name)

def make_extdb_coverage(dir_name, file_name):
    # create a extenal manifestdb for source coverage, and return temporal filepath to the database
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match('obs:obs_id')
    scheme.add_data_field('coverage')
    extdb_coverage = metadata.ManifestDb(map_file=os.path.join(dir_name, file_name),
                                         scheme=scheme)
    for i in range(10):
        if i in [3, 8]:
            coverage = 'jupiter:ws0,jupiter:ws1,saturn:ws0'
        else:
            coverage = 'jupiter:ws1,jupiter:ws2,saturn:ws0'
        entry_dict = {'obs:obs_id': f'myobs{i}', 
                      'coverage': coverage}
        extdb_coverage.add_entry(entry_dict)
    return os.path.join(dir_name, file_name)

# @unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestObsDb(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        pass
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_smoke(self):
        """Basic functionality."""
        db = get_example()
        all_obs = db.query()
        db.get(all_obs[0])
        db.get(all_obs[0]['obs_id'])
        db.query('timestamp > 0')
        db.query('timestamp > 0', tags=['cryo_problem=1'])

    def test_query(self):
        db = get_example()
        r0 = db.query('drift == "rising"')
        r1 = db.query('drift == "setting"')
        print(r0, len(r0))
        print(r1, len(r1))
        self.assertGreater(len(r0), 0)
        self.assertEqual(len(r0) + len(r1), len(db))
        
    def test_query_extension(self):
        db = get_example()
        extdb_path = make_extdb_pwv(dir_name=self.test_dir, file_name='pwv.sqlite')
        r0 = db.query('drift == "rising"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['pwv<2.0'],
                                         'params_list': ['pwv']}]
                   )
        r1 = db.query('drift == "setting"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['pwv<2.0'],
                                         'params_list': ['pwv']}]
                   )
        r2 = db.query('drift == "rising"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['pwv>=2.0'],
                                         'params_list': ['pwv']}]
                   )
        r3 = db.query('drift == "setting"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['pwv>=2.0'],
                                         'params_list': ['pwv']}]
                   )
        print(r0, len(r0))
        print(r1, len(r1))
        print(r2, len(r2))
        print(r3, len(r3))
        self.assertEqual(len(r0)+len(r1)+len(r2)+len(r3),
                         len(db))
        self.assertTrue('pwv' in r0.keys)

    def test_query_extension_coverage(self):
        db = get_example()
        extdb_path = make_extdb_coverage(dir_name=self.test_dir, file_name='coverage.sqlite')
        r0 = db.query('drift == "rising"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['jupiter:ws0 in coverage'],
                                         'params_list': ['coverage']}]
                   )
        r1 = db.query('drift == "setting"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['jupiter:ws0 in coverage'],
                                         'params_list': ['coverage']}]
                   )
        r2 = db.query('drift == "rising"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['jupiter:ws2 in coverage'],
                                         'params_list': ['coverage']}]
                   )
        r3 = db.query('drift == "setting"', 
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['jupiter:ws2 in coverage'],
                                         'params_list': ['coverage']}]
                   )
        r4 = db.query(
                    subdbs_info_list = [{'filepath': extdb_path,
                                         'query_list': ['jupiter:ws1 in coverage'],
                                         'params_list': ['coverage']}]
                   )
        print(r0, len(r0))
        print(r1, len(r1))
        print(r2, len(r2))
        print(r3, len(r3))
        print(r4, len(r4))
        self.assertEqual(len(r0)+len(r1)+len(r2)+len(r3),
                         len(db))
        self.assertEqual(len(r4), len(db))
        self.assertTrue('coverage' in r0.keys)
        

    def test_tags(self):
        db = get_example()
        r0 = db.query(tags=['planet=1', 'cryo_problem', 'not_a_tag'])
        r1 = db.query(tags=['planet=0', 'cryo_problem', 'not_a_tag'])
        self.assertGreater(len(r0), 0)
        self.assertEqual(len(r0) + len(r1), len(db))
        for k in ['planet', 'cryo_problem', 'not_a_tag']:
            self.assertTrue(k in r0.keys)
            self.assertTrue(k in r1.keys)

    def test_io(self):
        """Check to_file and from_file."""
        db0 = get_example()
        dump_list = [(f'test.sqlite', None),
                     (f'test.txt', 'dump'),
                     (f'test.gz', None)]
        # Save.
        for fn, fmt in dump_list:
            print(f'Writing {fn}')
            db0.to_file(fn, fmt=fmt)
            print('  -- output has size {}'.format(os.path.getsize(fn)))
            t0 = time.time()
            db1 = metadata.ObsDb.from_file(fn, fmt=fmt)
            dt = time.time() - t0
            self.assertEqual(len(db1.query()), len(db0.query()))
            print('  -- removing.')
            os.remove(fn)

    def test_info(self):
        """Check the .info method."""
        db0 = get_example()
        db0.info()


if __name__ == '__main__':
    unittest.main()
