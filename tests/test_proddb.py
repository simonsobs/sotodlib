import unittest

from sotodlib.core import metadata

from ._helpers import create_outdir, mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestProdDb(unittest.TestCase):
    def setUp(self):
        # Init.
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('array')
        scheme.add_range_match('time', dtype='float')
        scheme.add_data_field('also_data')
        self.scheme = scheme
        self.manifest = metadata.ManifestDb(scheme=scheme)

    def test_basic(self):
        manifest = self.manifest
        # Check file creation.
        file_id = manifest._get_file_id('test')
        assert(file_id is None) # New file_id should not be created.
        file_id = manifest._get_file_id('test', True)
        assert(file_id is not None) # New file_id should be created.
        assert(file_id == manifest._get_file_id('test')) # This file should now exist.

        # Add an indexed entry, with a new file attached.
        manifest.add_entry({'array': 'pa3', 'time': (12000., 13000.),
                            'also_data': 'wafered'},
                           'test2',create=True)

        for bad_time in [11999, 13000]:
            assert manifest.match({'array': 'pa3', 'time': bad_time}) is None
        for good_time in [12000, 12123]:
            assert manifest.match({'array': 'pa3',
                                   'time': good_time})['filename'] == 'test2'
        assert manifest.match({'array': 'pa2',
                               'time': 12000}) is None # Array does not match.

    def test_schema(self):
        print('\nCONSTRUCTED   :', self.scheme.cols)
        print('\nRECONSTRUCTED :', self.manifest.scheme.cols)
        assert (self.manifest.scheme.cols == self.scheme.cols)


if __name__ == '__main__':
    unittest.main()

