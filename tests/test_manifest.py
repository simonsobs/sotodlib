import os
import shutil
import tempfile
import unittest

from sotodlib.core.metadata.manifest import ManifestDb, ManifestDbBatchManager, ManifestScheme


class TestManifestDbBatchManager(unittest.TestCase):
    """Test the ManifestDbBatchManager class."""

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

            manager = ManifestDbBatchManager(db, batch_size=batch_size)

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

            with ManifestDbBatchManager(db, batch_size=batch_size) as manager:
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


if __name__ == '__main__':
    unittest.main()
