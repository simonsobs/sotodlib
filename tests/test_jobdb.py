import unittest
import os
import tempfile


from sotodlib.site_pipeline import jobdb

class TestBasic(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_00_smoke(self):
        # Create some jobs
        jdb = jobdb.JobManager(sqlite_file=':memory:')

        jdb.create_job('jclass1', {'obs_id': '123455'})
        jdb.create_job('jclass1', {'obs_id': '123456'}, jstate='done')
        jdb.create_job('jclass1', {'obs_id': '123457'})
        with self.assertRaises(jobdb.JobNotUniqueError):
            jdb.create_job('jclass1', {'obs_id': '123456'})
        jdb.create_job('jclass2', {'obs_id': '123456'})

        # Counting
        jobs_to_do = jdb.get_jobs(jclass='jclass1', jstate='open')
        self.assertEqual(len(jobs_to_do), 2)

        # Locking
        j = jobs_to_do[0]
        job = jdb.lock(j.id)
        with self.assertRaises(jobdb.JobLockedError):
            job = jdb.lock(j.id)
        jdb.unlock(job)
        
        job = jdb.lock(j.id)
        jdb.unlock(job.id, merge=False)

        # State write-back
        for row in jobs_to_do:
            print(f'Finishing {row.id} ...')
            with jdb.locked(row) as job:
                job.jstate = 'done'

        jobs_to_do = jdb.get_jobs(jclass='jclass1', jstate='open')
        self.assertEqual(len(jobs_to_do), 0)

        self.assertNotEqual(len(jdb.get_jobs(jclass='jclass1', jstate='all')), 0)

        # Deleting
        jobs_to_delete = jdb.get_jobs(jclass='jclass1', jstate='done')
        for j in jobs_to_delete:
            jdb.remove_job(j.id)

        self.assertEqual(len(jdb.get_jobs(jclass='jclass1', jstate='all')), 0)

    def test_10_report(self):
        db_file = os.path.join(self.tempdir.name, 'test_10.sqlite')
        jdb = jobdb.JobManager(sqlite_file=db_file)

        jdb.create_job('jclass1', {'obs_id': '123455'})
        jdb.create_job('jclass1', {'obs_id': '123456'}, jstate='done')
        jdb.create_job('jclass1', {'obs_id': '123457'})
        with self.assertRaises(jobdb.JobNotUniqueError):
            jdb.create_job('jclass1', {'obs_id': '123456'})
        jdb.create_job('jclass2', {'obs_id': '123456'})

        print()
        jobdb.cli(['--sqlite-file', db_file, 'select'])

    def test_20_locks(self):
        db_file = os.path.join(self.tempdir.name, 'test_20.sqlite')
        jdb = jobdb.JobManager(sqlite_file=db_file)

        jdb.create_job('jclass1', {'obs_id': '123455'})
        jdb.create_job('jclass1', {'obs_id': '123456'})
        jdb.create_job('jclass1', {'obs_id': '123457'})

        jobs = jdb.get_jobs(jclass='jclass1', jstate='open')
        with jdb.locked(jobs[0].id):
            with self.assertRaises(jobdb.JobLockedError):
                jdb.lock(jobs[0].id)
            with jdb.locked(jobs[1].id):
                pass
            jdb.clear_locks('all')
            jdb.lock(jobs[0].id)

        with self.assertRaises(jobdb.JobNotOwnedError):
            with jdb.locked(jobs, count=10) as jobs:
                # Simulate another entity stealing a lock.
                jx = jdb.lock(jobs[1].id, owner='xyz', force=True)

    def test_30_resource(self):
        jdb = jobdb.JobManager(sqlite_file=':memory:')

        kls = 'resource1'
        jdb.create_job(kls, {'channel': 'c1'})
        jdb.create_job(kls, {'channel': 'c2'})
        jdb.create_job(kls, {'channel': 'c3'})
        jdb.create_job(kls, {'channel': 'c4'})

        r = jdb.get_resource(kls)
        assert r is not None
        del r

        rs = []
        for i in range(5):
            r = jdb.get_resource(kls)
            assert (r is not None) ^ (i >= 4)
            rs.append(r)
        del rs
        #jdb.clear
        rs1 = jdb.get_resource(kls, n=3)
        assert len(rs1) == 3
        rs2 = jdb.get_resource(kls, n=3)
        assert len(rs2) == 1
        del rs1, rs2

        rs = jdb.get_resource(kls, tags={'channel': 'c2'}, n=4)
        assert len(rs) == 1
