# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Demonstrate construction of some simple metadata structures.  This
includes HDF5 IO helper routines, and the ObsDb/DetDb resolution and
association system used in Context/SuperLoader.

"""

import unittest
import tempfile

from sotodlib.core import metadata
from sotodlib.io.metadata import ResultSetHdfLoader, write_dataset, _decode_array

import os
import h5py


class MetadataTest(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_000_support(self):
        """Test some numpy-HDF5 conversion support functions.

        """
        rs = metadata.ResultSet(keys=['a_string', 'a_float', 'a_bad_string', 'a_bad_float'])
        rs.rows.append(('hello', 1.2, 'yuck', 1.3))
        aru = rs.asarray(hdf_compat=True)
        self.assertTrue(aru.dtype['a_string'].char == 'S')
        # Conversion code.
        arx = _decode_array(aru, key_map={
            'a_string': 'another_string',
            'a_float': 'another_float',
            'a_bad_string': None,
            'a_bad_float': None,
            })
        self.assertCountEqual(arx.dtype.names, ['another_string', 'another_float'])
        self.assertEqual(arx['another_string'].dtype.char, 'U')

    def test_001_hdf(self):
        """Test metadata write/read to HDF5 datasets

        """
        hdf_fn = os.path.join(self.tempdir.name, '_test_000_hdf.h5')

        # The reason we're here today is that this things works but is
        # going to be removed.
        loader = ResultSetHdfLoader()
        test_obs_id = 'testobs_1234'

        # Create an hdf5 dataset which is a structured array with only the
        # 'timeconst' column, containing the single fixed value.  Since there
        # are no columns with names prefixed by 'dets:' or 'obs:', this value
        # will be broadcast to all observations and detectors that access it.
        TGOOD = 1e-3
        rs = metadata.ResultSet(keys=['timeconst'])
        rs.append({'timeconst': TGOOD})
        with h5py.File(hdf_fn, 'a') as fout:
            # Simple one...
            write_dataset(rs, fout, 'timeconst_1ms', overwrite=True)

        # Simple look-up:
        req = {'filename': hdf_fn,
               'obs:obs_id': test_obs_id,
               'dataset': 'timeconst_1ms'}
        data = loader.from_loadspec(req)
        self.assertCountEqual(data['timeconst'], [TGOOD])

    def test_010_dbs(self):
        """Test metadata detdb/obsdb resolution system

        This tests one of the more complicated cases:

        - The ManifestDb includes restrictions on dets:band, so f090
          is to be loaded from one dataset and f150 is to be loaded
          from another.

        - The two datasets both provide values for f090 and f150, so
          the code has to know to ignore the ones that weren't asked
          for.

        """
        hdf_fn = os.path.join(self.tempdir.name, '_test_010_dbs.h5')
        mandb_fn = os.path.join(self.tempdir.name, '_test_010_dbs.sqlite')

        # Add two datasets to the HDF file.  They are called
        # "timeconst_early" and "timeconst_late" but there is no
        # specific time range associated with each.  Each dataset
        # contains a value for bands f090 and f150.  The "early" set
        # has TBAD for f150 and the "late" set has TBAD for f090.
        T090, T150, TBAD = 90e-3, 150e-3, 1e0
        with h5py.File(hdf_fn, 'a') as fout:
            # First test.
            for label, tau1, tau2 in [('early', T090, TBAD),
                                      ('late', TBAD, T150)]:
                rs = metadata.ResultSet(keys=['dets:band', 'timeconst'])
                rs.append({'dets:band': 'f090', 'timeconst': tau1})
                rs.append({'dets:band': 'f150', 'timeconst': tau2})
                write_dataset(rs, fout, 'timeconst_%s' % label, overwrite=True)

        # To match the early/late example we need DetDb and ObsDb.
        detdb = metadata.DetDb()
        detdb.create_table('base', ["`band` str", "`polcode` str"])
        detdb.add_props('base', 'det1', band='f090', polcode='A')
        detdb.add_props('base', 'det2', band='f090', polcode='B')
        detdb.add_props('base', 'det3', band='f150', polcode='A')
        detdb.add_props('base', 'det4', band='f150', polcode='B')

        obsdb = metadata.ObsDb()
        t_pivot = 2000010000
        obsdb.add_obs_columns(['timestamp float'])
        obsdb.update_obs('obs_00', {'timestamp': t_pivot - 10000})
        obsdb.update_obs('obs_01', {'timestamp': t_pivot + 10000})

        # Test 1 -- ManifestDb and Stored datasets both have "band" rules.
        scheme = metadata.ManifestScheme() \
                         .add_range_match('obs:timestamp') \
                         .add_data_field('dets:band') \
                         .add_data_field('dataset')
        mandb = metadata.ManifestDb(scheme=scheme)
        for band, this_pivot in [('f090', t_pivot + 1e6),
                                 ('f150', t_pivot - 1e6)]:
            mandb.add_entry({'dataset': 'timeconst_early',
                             'dets:band': band,
                             'obs:timestamp': (0, this_pivot)},
                            filename=hdf_fn)
            mandb.add_entry({'dataset': 'timeconst_late',
                             'dets:band': band,
                             'obs:timestamp': (this_pivot, 4e9)},
                            filename=hdf_fn)
        mandb.to_file(mandb_fn)

        # The SuperLoader is where the logic lives to combine multiple
        # results and pull out the right information in the right
        # order.  It should leave us with no TBAD values.
        loader = metadata.SuperLoader(obsdb=obsdb, detdb=detdb)
        spec_list = [
            {'db': mandb_fn,
             'name': 'tau&timeconst'}
        ]
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00'})
        self.assertCountEqual(mtod['tau'], [T090, T090, T150, T150])

        # Test 2: ManifestDb specifies polcode, which crosses with
        # dataset band.
        scheme = metadata.ManifestScheme() \
                         .add_range_match('obs:timestamp') \
                         .add_data_field('dets:polcode') \
                         .add_data_field('dataset')
        mandb = metadata.ManifestDb(scheme=scheme)
        for polcode, this_pivot in [('A', t_pivot + 1e6),
                                    ('B', t_pivot - 1e6)]:
            mandb.add_entry({'dataset': 'timeconst_early',
                             'dets:polcode': polcode,
                             'obs:timestamp': (0, this_pivot)},
                            filename=hdf_fn)
            mandb.add_entry({'dataset': 'timeconst_late',
                             'dets:polcode': polcode,
                             'obs:timestamp': (this_pivot, 4e9)},
                            filename=hdf_fn)
        mandb.to_file(mandb_fn)

        # Now we expect only f090 A and f150 B to resolve to non-bad vals.
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00'})
        self.assertCountEqual(mtod['tau'], [T090, TBAD, TBAD, T150])


if __name__ == '__main__':
    unittest.main()
