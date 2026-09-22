import unittest
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from sotodlib.site_pipeline import jobdb

from ._helpers import mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
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

        # Locking many jobs at once
        jobs = jdb.lock(jobs_to_do)
        self.assertEqual(jdb.lock(jobs), [])
        jdb.unlock(jobs, merge=False)

        # State write-back
        for row in jobs_to_do:
            print(f'Finishing {row.id} ...')
            with jdb.locked(row) as job:
                job.jstate = 'done'

        jobs_to_do = jdb.get_jobs(jclass='jclass1', jstate='open')
        self.assertEqual(len(jobs_to_do), 0)

        self.assertNotEqual(len(jdb.get_jobs(jclass='jclass1', jstate='all')), 0)

        # Deleting one job
        jobs_to_delete = jdb.get_jobs(jclass='jclass1', jstate='done')
        jdb.remove_jobs(jobs_to_delete[0].id)
        self.assertEqual(
            len(jdb.get_jobs(jclass='jclass1', jstate='all')),
            len(jobs_to_delete) - 1
        )

        # Deleting many jobs
        jobs_to_delete = jdb.get_jobs(jclass='jclass1', jstate='done')
        jdb.remove_jobs(jobs_to_delete)
        self.assertEqual(len(jdb.get_jobs(jclass='jclass1', jstate='all')), 0)

        # Create-and-operate
        j = jdb.create_job('jclass2', {'obs_id': '123455'})
        with jdb.locked(j) as job:
            job.mark_visited()

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

        # Test locking many at once
        with jdb.locked(jobs, count=len(jobs)):
            for j in jobs:
                with self.assertRaises(jobdb.JobLockedError):
                    jdb.lock(j.id)
            jdb.clear_locks('all')
            jdb.lock(jobs)

        with self.assertRaises(jobdb.JobUnlockError) as error:
            with jdb.locked(jobs, count=10) as jobs:
                # Simulate another entity stealing a lock.
                jdb.lock(jobs[1].id, owner='xyz', force=True)
        self.assertEqual(error.exception.failures[0][0], jobs[1].id)

    def test_25_bulk_operations(self):
        db_file = os.path.join(self.tempdir.name, 'test_25.sqlite')
        jdb = jobdb.JobManager(sqlite_file=db_file)
        jobs = [
            jdb.create_job('jclass1', {'obs_id': str(index)})
            for index in range(4)
        ]

        # List results preserve input order and remain lists for one item.
        locked = jdb.lock([jobs[2], jobs[0]], count=2)
        self.assertEqual([job.id for job in locked], [jobs[2].id, jobs[0].id])
        self.assertIsInstance(jdb.lock([jobs[1]], count=1), list)
        jdb.clear_locks('all')

        # Valid merges succeed even if other jobs in the batch fail ownership
        # checks, and the aggregate error identifies each failed row.
        locked = jdb.lock(jobs[:3])
        locked[2].jstate = jobdb.JState.done
        jdb.lock(locked[0], owner='other', force=True)
        jdb.unlock(locked[1].id, merge=False)
        with self.assertRaises(jobdb.JobUnlockError) as error:
            jdb.unlock(locked)
        self.assertEqual(
            {job_id for job_id, _ in error.exception.failures},
            {locked[0].id, locked[1].id},
        )
        updated = jdb.get_jobs(job_id=[locked[2].id])[0]
        self.assertEqual(updated.jstate, jobdb.JState.done)
        jdb.clear_locks('all')

        # Partial deletion reports its counts but still deletes eligible rows.
        jdb.lock(jobs[0])
        with self.assertRaises(jobdb.JobNotDeletedError) as error:
            jdb.remove_jobs([jobs[0], jobs[1]], check_locked=True)
        self.assertEqual((error.exception.deleted, error.exception.requested),
                         (1, 2))
        jdb.clear_locks('all')

    def test_26_concurrent_bulk_locking(self):
        db_file = os.path.join(self.tempdir.name, 'test_26.sqlite')
        manager = jobdb.JobManager(sqlite_file=db_file)
        job_ids = [
            manager.create_job('work', {'index': str(index)}).id
            for index in range(20)
        ]

        def acquire(owner):
            db = jobdb.JobManager(sqlite_file=db_file)
            return {
                job.id for job in db.lock(job_ids, owner=owner, count=10)
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(acquire, ('first', 'second'))

        self.assertEqual(len(first), 10)
        self.assertEqual(len(second), 10)
        self.assertTrue(first.isdisjoint(second))
        self.assertEqual(first | second, set(job_ids))

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
