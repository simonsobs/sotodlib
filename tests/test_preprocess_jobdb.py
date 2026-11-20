import os
import tempfile
import unittest

from sotodlib.preprocess import preprocess_util
from sotodlib.site_pipeline import jobdb

from ._helpers import mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestPreprocessJobDb(unittest.TestCase):
    def test_config_compatibility(self):
        self.assertEqual(
            preprocess_util.get_jobdb_config({"jobdb": "jobs.sqlite"}),
            ("jobs.sqlite", 1),
        )
        self.assertEqual(
            preprocess_util.get_jobdb_config({
                "jobdb": {"path": "jobs.sqlite", "batch_size": 100},
            }),
            ("jobs.sqlite", 100),
        )
        self.assertEqual(
            preprocess_util.get_jobdb_config({}),
            (None, 1),
        )
        with self.assertRaises(ValueError):
            preprocess_util.get_jobdb_config({
                "jobdb": {"path": "jobs.sqlite", "batch_size": 0},
            })

    def test_batch_updates_match_job_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manager = jobdb.JobManager(
                sqlite_file=os.path.join(tempdir, "jobs.sqlite")
            )
            jobs = [
                manager.create_job(
                    "init", {"obs:obs_id": str(index), "error": None}
                )
                for index in range(3)
            ]
            updates = [
                {"job": jobs[2], "jstate": jobdb.JState.failed,
                 "error": "third-error"},
                {"job": jobs[0], "jstate": jobdb.JState.done,
                 "error": None},
                {"job": jobs[1], "jstate": jobdb.JState.failed,
                 "error": "second-error"},
            ]

            preprocess_util.update_jobdb(manager, updates)

            result = {job.id: job for job in manager.get_jobs(jclass="init")}
            self.assertEqual(result[jobs[0].id].jstate, jobdb.JState.done)
            self.assertIsNone(result[jobs[0].id].tags["error"])
            self.assertEqual(result[jobs[1].id].tags["error"], "second-error")
            self.assertEqual(result[jobs[2].id].tags["error"], "third-error")

    def test_batch_update_requires_every_lock(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manager = jobdb.JobManager(
                sqlite_file=os.path.join(tempdir, "jobs.sqlite")
            )
            jobs = [
                manager.create_job(
                    "init", {"obs:obs_id": str(index), "error": None}
                )
                for index in range(2)
            ]
            manager.lock(jobs[1], owner="other")
            updates = [
                {"job": job, "jstate": jobdb.JState.done, "error": None}
                for job in jobs
            ]

            with self.assertRaises(jobdb.JobLockedError):
                preprocess_util.update_jobdb(manager, updates)

            result = manager.get_jobs(jclass="init")
            self.assertTrue(all(job.jstate == jobdb.JState.open for job in result))
            self.assertIsNone(result[0].lock)
            self.assertEqual(result[1].lock_owner, "other")


if __name__ == "__main__":
    unittest.main()
