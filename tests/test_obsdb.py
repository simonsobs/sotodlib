import unittest
from sotodlib.core import metadata

import os
import time

from ._helpers import mpi_multi


def get_example(stuff_missing=False):
    # Create a new Db and add two columns.
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(['timestamp float', 'hwp_speed float', 'drift string'])

    # Add 10 rows.
    for i in range(10):
        if stuff_missing and i in [2, 4]:
            continue
        tags = []
        if i == 6:
            tags.append('cryo_problem')
        if i > 7:
            tags.append('planet')
        elif not stuff_missing:
            tags.append('cmb_survey')
        obsdb.update_obs(f'myobs{i}', {'timestamp': 1900000000. + i * 100,
                                       'hwp_speed': 2.0,
                                       'drift': 'rising' if i%2 else 'setting'},
                         tags=tags)
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

    def test_query(self):
        db = get_example()
        r0 = db.query("drift == 'rising'")
        r1 = db.query("drift == 'setting'")
        self.assertGreater(len(r0), 0)
        self.assertEqual(len(r0) + len(r1), len(db))

        # Grumbles about double quotes?
        with self.assertWarns(UserWarning):
            r1 = db.query('drift == "setting"')

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

    def test_diff_patch(self):
        """Use diff/patch to update one obsdb to match another."""
        db0 = get_example(stuff_missing=True)
        db1 = get_example()
        diff = metadata.obsdb.diff_obsdbs(db0, db1)
        assert diff['different']
        assert diff['patchable']

        metadata.obsdb.patch_obsdb(diff['patch_data'], db0)
        diff = metadata.obsdb.diff_obsdbs(db0, db1)
        assert not diff['different']


if __name__ == '__main__':
    unittest.main()
