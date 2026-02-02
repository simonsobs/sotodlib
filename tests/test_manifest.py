import os
import shutil
import tempfile
import unittest

from sotodlib.core.metadata.manifest import (
    DbBatchManager,
    ManifestDb,
    ManifestScheme,
    MultiDbBatchManager,
)


class TestDbBatchManager(unittest.TestCase):
    """Test the DbBatchManager class."""

    def test_batch_manager_basic(self):
        """Test basic functionality of the batch manager."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test.db')

        try:
            # Create a ManifestDb instance first
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db = ManifestDb(db_path, scheme=scheme)

            # Test basic initialization with existing db
            batch_size = 10

            manager = DbBatchManager(db, batch_size=batch_size)

            # Test context manager
            with manager:
                self.assertIsNotNone(manager.db)
                self.assertEqual(manager.batch_counter, 0)
                self.assertEqual(manager.batch_size, batch_size)

            # Test that the manager doesn't close the db after context
            self.assertIsNotNone(manager.db)

            # Clean up the db
            db.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_batch_manager_add_entry(self):
        """Test adding entries with the batch manager."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test.db')

        try:
            # Create a ManifestDb instance first
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db = ManifestDb(db_path, scheme=scheme)

            batch_size = 5  # Small batch size for testing

            with DbBatchManager(db, batch_size=batch_size) as manager:
                # Test adding entries
                for i in range(8):  # Add more than batch_size entries
                    params = {
                        'obs:obs_id': f'test_obs_{i}',
                        'dets:detset': f'detset_{i % 3}',
                        'dataset': f'dataset_{i}',
                    }
                    manager.add_entry(params, f'file_{i}.h5')

                # Check that batch counter is correct (should be 3 after 8 entries with batch_size=5)
                self.assertEqual(manager.batch_counter, 3)

                # Verify entries were added
                results = manager.db.inspect({})
                self.assertEqual(len(results), 8)

            # Clean up the db
            db.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestMultiDbBatchManager(unittest.TestCase):
    """Test the MultiDbBatchManager class."""

    def test_multi_manager_basic(self):
        """Test basic functionality of the multi-database batch manager."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path1 = os.path.join(temp_dir, 'test1.db')
        db_path2 = os.path.join(temp_dir, 'test2.db')

        try:
            # Create ManifestDb instances
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db1 = ManifestDb(db_path1, scheme=scheme)
            db2 = ManifestDb(db_path2, scheme=scheme)

            batch_size = 10

            # Test with single batch size for all databases
            multi_manager = MultiDbBatchManager([db1, db2], batch_size=batch_size)

            with multi_manager as (mgr1, mgr2):
                self.assertIsNotNone(mgr1.db)
                self.assertIsNotNone(mgr2.db)
                self.assertEqual(mgr1.batch_size, batch_size)
                self.assertEqual(mgr2.batch_size, batch_size)
                self.assertEqual(mgr1.batch_counter, 0)
                self.assertEqual(mgr2.batch_counter, 0)

            # Clean up the databases
            db1.conn.close()
            db2.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_multi_manager_different_batch_sizes(self):
        """Test multi-database manager with different batch sizes."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path1 = os.path.join(temp_dir, 'test1.db')
        db_path2 = os.path.join(temp_dir, 'test2.db')
        db_path3 = os.path.join(temp_dir, 'test3.db')

        try:
            # Create ManifestDb instances
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db1 = ManifestDb(db_path1, scheme=scheme)
            db2 = ManifestDb(db_path2, scheme=scheme)
            db3 = ManifestDb(db_path3, scheme=scheme)

            batch_sizes = [5, 10, 15]

            with MultiDbBatchManager([db1, db2, db3], batch_size=batch_sizes) as managers:
                mgr1, mgr2, mgr3 = managers
                self.assertEqual(mgr1.batch_size, 5)
                self.assertEqual(mgr2.batch_size, 10)
                self.assertEqual(mgr3.batch_size, 15)

            # Clean up the databases
            db1.conn.close()
            db2.conn.close()
            db3.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_multi_manager_add_entries(self):
        """Test adding entries to multiple databases."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path1 = os.path.join(temp_dir, 'test1.db')
        db_path2 = os.path.join(temp_dir, 'test2.db')

        try:
            # Create ManifestDb instances
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db1 = ManifestDb(db_path1, scheme=scheme)
            db2 = ManifestDb(db_path2, scheme=scheme)

            batch_sizes = [3, 5]

            with MultiDbBatchManager([db1, db2], batch_size=batch_sizes) as (mgr1, mgr2):
                # Add entries to first database
                for i in range(7):
                    params = {
                        'obs:obs_id': f'db1_obs_{i}',
                        'dets:detset': f'detset_{i % 2}',
                        'dataset': f'dataset_{i}',
                    }
                    mgr1.add_entry(params, f'file1_{i}.h5')

                # Add entries to second database
                for i in range(8):
                    params = {
                        'obs:obs_id': f'db2_obs_{i}',
                        'dets:detset': f'detset_{i % 3}',
                        'dataset': f'dataset_{i}',
                    }
                    mgr2.add_entry(params, f'file2_{i}.h5')

                # Check batch counters (7 % 3 = 1, 8 % 5 = 3)
                self.assertEqual(mgr1.batch_counter, 1)
                self.assertEqual(mgr2.batch_counter, 3)

            # Verify entries were added to both databases
            results1 = db1.inspect({})
            results2 = db2.inspect({})
            self.assertEqual(len(results1), 7)
            self.assertEqual(len(results2), 8)

            # Clean up the databases
            db1.conn.close()
            db2.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_multi_manager_invalid_batch_sizes(self):
        """Test that invalid batch_size list raises ValueError."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path1 = os.path.join(temp_dir, 'test1.db')
        db_path2 = os.path.join(temp_dir, 'test2.db')

        try:
            # Create ManifestDb instances
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db1 = ManifestDb(db_path1, scheme=scheme)
            db2 = ManifestDb(db_path2, scheme=scheme)

            # Should raise ValueError because batch_sizes length doesn't match databases length
            with self.assertRaises(ValueError):
                MultiDbBatchManager([db1, db2], batch_size=[5, 10, 15])

            # Clean up the databases
            db1.conn.close()
            db2.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_multi_manager_force_commit_all(self):
        """Test force_commit_all commits all pending operations."""

        # Create a temporary directory for the test
        temp_dir = tempfile.mkdtemp()
        db_path1 = os.path.join(temp_dir, 'test1.db')
        db_path2 = os.path.join(temp_dir, 'test2.db')

        try:
            # Create ManifestDb instances
            scheme = ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')

            db1 = ManifestDb(db_path1, scheme=scheme)
            db2 = ManifestDb(db_path2, scheme=scheme)

            multi_manager = MultiDbBatchManager([db1, db2], batch_size=100)

            with multi_manager as (mgr1, mgr2):
                # Add a few entries (less than batch size)
                for i in range(3):
                    params1 = {
                        'obs:obs_id': f'db1_obs_{i}',
                        'dets:detset': f'detset_{i}',
                        'dataset': f'dataset_{i}',
                    }
                    mgr1.add_entry(params1, f'file1_{i}.h5')

                    params2 = {
                        'obs:obs_id': f'db2_obs_{i}',
                        'dets:detset': f'detset_{i}',
                        'dataset': f'dataset_{i}',
                    }
                    mgr2.add_entry(params2, f'file2_{i}.h5')

                # Verify batch counters before exit
                self.assertEqual(mgr1.batch_counter, 3)
                self.assertEqual(mgr2.batch_counter, 3)

            # After context exit, force_commit_all should have been called
            # Verify all entries were committed
            results1 = db1.inspect({})
            results2 = db2.inspect({})
            self.assertEqual(len(results1), 3)
            self.assertEqual(len(results2), 3)

            # Clean up the databases
            db1.conn.close()
            db2.conn.close()

        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
