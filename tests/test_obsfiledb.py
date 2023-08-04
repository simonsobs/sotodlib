import unittest
import os
import shutil
import tempfile

from sotodlib.core import metadata

from ._helpers import create_outdir, mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestObsFileDB(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test_obsfiledb.sqlite'
        self.test_datatree = 'test_datatree'
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_simple_db(self, n_obs=2, n_detsets=2, n_dets=3, n_segs=3,
                      prepath=''):
        # Create database in RAM
        db = metadata.ObsFileDb()
        obs_ids = ['obs%i' % i for i in range(n_obs)]
        detsets = ['group%i' % i for i in range(n_detsets)]
        for di, d in enumerate(detsets):
            db.add_detset(d, ['det%i_%i' % (di, i) for i in range(n_dets)])
        for obs_id in obs_ids:
            for detset in detsets:
                for file_index in range(n_segs):
                    sample_index = file_index * 1000
                    filename = f'{prepath}{obs_id}_{detset}_{sample_index:04d}.g3'
                    db.add_obsfile(filename, obs_id, detset, sample_index)
        db.prefix = self.test_dir
        return db

    def test_000_basic(self):
        # Get example database
        n_obs, n_detsets = 4, 3
        db = self.get_simple_db(n_obs, n_detsets)
        for fmt in ['sqlite', 'dump', 'gz']:
            # Write database to disk.
            db.to_file(self.test_filename, fmt=fmt)

            # Load it up again.
            db2 = metadata.ObsFileDb.from_file(self.test_filename, fmt=fmt)

            # Check.
            assert (sorted(db.get_obs()) == sorted(db2.get_obs()))
            obs_id = db.get_obs()[0]
            assert (sorted(db.get_detsets(obs_id)) ==
                    sorted(db2.get_detsets(obs_id)))
            assert (db.get_detsets('not an obs') == [])

            # CLI
            if fmt == 'sqlite':
                metadata.obsfiledb.main([self.test_filename, 'files'])

    def test_010_remove(self):
        db = self.get_simple_db(4, 3)
        n0 = len(db.get_obs())
        obs_id = db.get_obs()[0]
        detsets = db.get_detsets(obs_id)

        # Drop obs_id.
        db.drop_obs(obs_id)
        n1 = len(db.get_obs())
        self.assertEqual(n0 - 1, n1)

        # Drop detset
        n0 = len(db.verify()['raw'])
        db.drop_detset(detsets[0])
        n1 = len(db.verify()['raw'])
        self.assertEqual(n1 * len(detsets),
                         n0 * (len(detsets) - 1))

    def test_020_update(self):
        n_detsets, n_segs = 4, 3
        db = self.get_simple_db(n_detsets=n_detsets, n_segs=n_segs)
        # Create all the missing files.
        for row in db.verify()['raw']:
            present, filename = row[:2]
            open(filename, 'w').close()
        # Rescan.
        results = db.verify()
        assert(all([r[0] for r in results['raw']]))
        # Remove a file and check that database can clean itself up.
        os.remove(results['raw'][0][1])
        results2 = db.verify()
        assert(results2['raw'][0][0] is False)
        db2 = db.copy()
        db2.drop_incomplete()

        self.assertEqual(db2.get_obs(), db.get_obs())
        self.assertEqual(len(db2.verify()['raw']),
                         len(db.verify()['raw']) - n_segs)

    def test_030_prefix(self):
        db = self.get_simple_db()
        # Create all the missing files.
        for row in db.verify()['raw']:
            present, filename = row[:2]
            open(filename, 'w').close()
        # Quick check we've done that properly...
        results = db.verify()
        assert(all([r[0] for r in results['raw']]))

        # Absolute paths should override db.prefix.
        db = self.get_simple_db(prepath=self.test_dir + '/')
        db.prefix = '/whatever'
        results = db.verify()
        assert(all([r[0] for r in results['raw']]))

    def test_040_lookups(self):
        db = self.get_simple_db(prepath='somewhere/else/')
        target = 'somewhere/else/obs0_group0_0000.g3'
        for item, result in enumerate([
                db.lookup_file(target, resolve_paths=False),
                db.lookup_file(self.test_dir + '/' + target),
                db.lookup_file(self.test_dir + '/' + target, resolve_paths=True),
                db.lookup_file('x/' + target, prefix='x/'),
                db.lookup_file('notx/../x/' + target, prefix='x/'),
                ]):
            self.assertEqual(result['obs_id'], 'obs0', 'Subcase #%i' % item)

        assert(db.lookup_file('obsXYZ_.g3', fail_ok=True) is None)
        with self.assertRaises(RuntimeError):
            db.lookup_file('obsXYZ_.g3')
        with self.assertRaises(RuntimeError):
            db.lookup_file('notx/../x/' + target, resolve_paths=False)


if __name__ == '__main__':
    unittest.main()
