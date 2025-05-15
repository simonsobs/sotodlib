import unittest
from sotodlib.core import metadata

import os
import time

from ._helpers import mpi_multi


def get_example():
    # Create a new Db keyed by obs_id and add two columns.
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

def get_owb_example():
    # Create a new Db keyed by obsid, wafer_slot, bandpass, and add two columns.
    obsdb = metadata.ObsDb(wafer_info=('wafer_slot', 'bandpass'))
    obsdb.add_obs_columns(['timestamp float', 'data1 int', 'data2 int'])

    wafer_slots = ['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6']
    band_passes = ['f090', 'f150']
    data1 = 0
    data2 = 10000
    for i in range(3):
        if i == 2:
            tags = ['max_was_here']
        else:
            tags = []
        for w in range(7):
            for b in range(2):
                obsdb.update_obs(f'myobs{i}', wafer_info=(wafer_slots[w], band_passes[b]), 
                                 data = {'timestamp': 1900000000. + i * 100,
                                         'data1':data1, 'data2':data2}, tags=tags)
                data1+=1
                data2+=1
    return obsdb

@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestObsDb(unittest.TestCase):

    def setUp(self):
        pass

    def test_smoke(self):
        """Basic functionality."""
        db = get_example()
        all_obs = db.query()
        db.get(all_obs[0]['obs_id'])
        db.query('timestamp > 0')
        db.query('timestamp > 0', tags=['cryo_problem=1'])

    def test_smoke_owb(self):
        """Basic functionality."""
        db = get_owb_example()
        all_obs = db.query()
        q1 = db.get(all_obs[0]['obs_id'], wafer_info=('ws3', 'f090'))
        self.assertEqual(q1['data1'], 6)
        rs = db.query('timestamp > 0', tags=['max_was_here=1'])
        self.assertEqual(len(rs), 14)

    def test_query(self):
        db = get_example()
        r0 = db.query("drift == 'rising'")
        r1 = db.query("drift == 'setting'")
        self.assertGreater(len(r0), 0)
        self.assertEqual(len(r0) + len(r1), len(db))

    def test_owb_query(self):
        """Test querying with wafer_info keys."""
        db = get_owb_example()
        # Query for a specific wafer_slot and bandpass
        results = db.query("timestamp > 0 and wafer_slot =='ws3' and bandpass == 'f090'")
        self.assertGreater(len(results), 0)
        for row in results:
            self.assertEqual(row['wafer_slot'], 'ws3')
            self.assertEqual(row['bandpass'], 'f090')

    def test_tags(self):
        db = get_example()
        r0 = db.query(tags=['planet=1', 'cryo_problem', 'not_a_tag'])
        r1 = db.query(tags=['planet=0', 'cryo_problem', 'not_a_tag'])
        self.assertGreater(len(r0), 0)
        self.assertEqual(len(r0) + len(r1), len(db))
        for k in ['planet', 'cryo_problem', 'not_a_tag']:
            self.assertTrue(k in r0.keys)
            self.assertTrue(k in r1.keys)

    def test_owb_tags(self):
        """Test tags with wafer_info keys."""
        db = get_owb_example()
        db.update_obs('myobs0', wafer_info=('ws3', 'f090'), tags=['test_tag'])
        result = db.get('myobs0', wafer_info=('ws3', 'f090'), tags=True)
        self.assertIn('test_tag', result['tags'])
        result_other = db.get('myobs0', wafer_info=('ws4', 'f090'), tags=True)
        self.assertNotIn('test_tag', result_other['tags'])

    def test_owb_tag_deletion(self):
        """Test deleting tags for specific wafer_info keys."""
        db = get_owb_example()
        db.update_obs('myobs0', wafer_info=('ws3', 'f090'), tags=['delete_me'])
        db.update_obs('myobs0', wafer_info=('ws3', 'f090'), tags=['!delete_me'])
        result = db.get('myobs0', wafer_info=('ws3', 'f090'), tags=True)
        self.assertNotIn('delete_me', result.get('tags', []))
        db.update_obs('myobs0', wafer_info=('ws4', 'f090'), tags=['delete_me'])
        result_other = db.get('myobs0', wafer_info=('ws4', 'f090'), tags=True)
        self.assertIn('delete_me', result_other['tags'])

    def test_linked_query(self):
        obs_db = get_example()
        owb_db = get_owb_example()
        query_res = obs_db.query_linked_dbs(owb_db, 'obs_id == "myobs2"')
        self.assertIsInstance(query_res, list)
        self.assertEqual(len(query_res), 1)
        self.assertIsInstance(query_res[0], tuple)
        self.assertEqual(len(query_res[0]), 2)
        self.assertEqual(query_res[0][0]['drift'], "setting")
        self.assertEqual(len(query_res[0][1]), 14)

    def test_owb_primary_linked_query(self):
        """Test linked queries with wafer_info keys."""
        obs_db = get_example()
        owb_db = get_owb_example()
        query_res = obs_db.query_linked_dbs(owb_db, 'obs_id == "myobs0"', wafer_info=("ws3", "f090"))
        self.assertIsInstance(query_res, list)
        self.assertGreater(len(query_res), 0)
        for primary, linked in query_res:
            self.assertEqual(primary['obs_id'], 'myobs0')
            for row in linked:
                self.assertEqual(row['wafer_slot'], 'ws3')
                self.assertEqual(row['bandpass'], 'f090')

    def test_owb_update(self):
        """
        Test updating data for specific wafer_info keys.
        Tests all 4 possible ways to update data.
        """
        db = get_owb_example()
        # Method 1
        db.update_obs('myobs0', wafer_info=('ws3', 'f090'), data={'data1': 999})
        result = db.get('myobs0', wafer_info=('ws3', 'f090'))
        self.assertEqual(result['data1'], 999)
        result_other = db.get('myobs0', wafer_info=('ws4', 'f090'))
        self.assertNotEqual(result_other['data1'], 999)
        # Method 2
        db.update_obs('myobs0', wafer_info={'wafer_slot':'ws3', 'bandpass':'f150'},
                      data={'data1': 998})
        result_m2 = db.get('myobs0', wafer_info={'wafer_slot':'ws3', 'bandpass':'f150'})
        self.assertEqual(result_m2['data1'], 998)
        # Method 3
        db.update_obs({'obs_id':'myobs0', 'wafer_slot':'ws5', 'bandpass':'f090'},
                      data={'data1': 997})
        result_m3 = db.get({'obs_id':'myobs0', 'wafer_slot':'ws5', 'bandpass':'f090'})
        self.assertEqual(result_m3['data1'], 997)
        # Method 4
        db.update_obs(('myobs0', 'ws5', 'f150'),
                      data={'data1': 996})
        result_m4 = db.get(('myobs0', 'ws5', 'f150'))
        self.assertEqual(result_m4['data1'], 996)

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
