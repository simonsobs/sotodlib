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
import sqlite3

from ._helpers import mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
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
        self.assertEqual(list(arx.dtype.names), ['another_string', 'another_float'])
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
        self.assertEqual(list(data['timeconst']), [TGOOD])

    def test_010_manifest_basics(self):
        """Test that you can create a ManifestScheme and ManifestDb and add
        records but not duplicates unless you need to.

        """
        scheme = metadata.ManifestScheme() \
                         .add_range_match('obs:timestamp')
        mandb = metadata.ManifestDb(scheme=scheme)
        mandb.add_entry({'obs:timestamp': (0, 1e9)}, 'a.h5')

        # Duplicate index prevented?
        with self.assertRaises(sqlite3.IntegrityError):
            mandb.add_entry({'obs:timestamp': (0, 1e9)}, 'b.h5')

        # Replace accepted?
        mandb.add_entry({'obs:timestamp': (0, 1e9)}, 'c.h5',
                        replace=True)

        # Returns the single correct value?
        self.assertEqual(
            'c.h5', mandb.match({'obs:timestamp': 100})['filename'])

    def test_020_db_resolution(self):
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
        detdb.create_table('base', ["`readout_id` str", "`band` str", "`polcode` str"])
        detdb.add_props('base', 'det1', readout_id='det1', band='f090', polcode='A')
        detdb.add_props('base', 'det2', readout_id='det2', band='f090', polcode='B')
        detdb.add_props('base', 'det3', readout_id='det3', band='f150', polcode='A')
        detdb.add_props('base', 'det4', readout_id='det4', band='f150', polcode='B')

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
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00'}, detdb.props())
        self.assertEqual(list(mtod['tau']), [T090, T090, T150, T150])

        # Make sure that also plays well with det specs.
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00', 'dets:band': 'f090'},
                           detdb.props())
        self.assertEqual(list(mtod['tau']), [T090, T090])
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00', 'dets:band': 'f150'},
                           detdb.props())
        self.assertEqual(list(mtod['tau']), [T150, T150])

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
        # Make sure you reinit the loader, to avoid cached dbs.
        loader = metadata.SuperLoader(obsdb=obsdb, detdb=detdb)
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00'}, detdb.props())
        self.assertEqual(list(mtod['tau']), [T090, TBAD, TBAD, T150])

        # Make sure that also plays well with det specs.
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00', 'dets:band': 'f090'},
                           detdb.props())
        self.assertEqual(list(mtod['tau']), [T090, TBAD])
        mtod = loader.load(spec_list, {'obs:obs_id': 'obs_00', 'dets:band': 'f150'},
                           detdb.props())
        self.assertEqual(list(mtod['tau']), [TBAD, T150])

    def test_030_manipulation(self):
        """Test some helpers for inspecting and updating ManifestDbs.

        """
        scheme = metadata.ManifestScheme() \
                         .add_range_match('obs:timestamp') \
                         .add_exact_match('wafer') \
                         .add_data_field('dets:band') \
                         .add_data_field('dataset')

        mandb = metadata.ManifestDb(scheme=scheme)

        mandb.add_entry({'dets:band': 'f150',
                         'wafer': 'A',
                         'obs:timestamp': (1200000000, 1300000000),
                         'dataset': 'early'}, filename='x')
        mandb.add_entry({'dets:band': 'f150',
                         'wafer': 'A',
                         'obs:timestamp': (1200000000, 1300000001),
                         'dataset': 'early'}, filename='x')
        mandb.add_entry({'dets:band': 'f220',
                         'wafer': 'A',
                         'obs:timestamp': (1300000000, 1400000001),
                         'dataset': 'early'}, filename='y')

        # Does .get_entries work?
        entries = mandb.get_entries(['obs:timestamp__lo', 'obs:timestamp__hi'])
        print(entries)

        # Does .inspect work properly?
        entries = mandb.inspect({'wafer': 'A'})
        self.assertEqual(len(entries), 3)
        self.assertIn('wafer', entries[0])

        entries = mandb.inspect({'dets:band': 'f150'})
        self.assertEqual(len(entries), 2)
        entries = mandb.inspect({'filename': 'x'})
        self.assertEqual(len(entries), 2)

        # Modify an entry
        entries[1]['wafer'] = 'B'
        entries[1]['dets:band'] = 'f090'
        entries[1].pop('filename')
        mandb.update_entry(entries[1])
        entries = mandb.inspect({'wafer': 'A'})
        self.assertEqual(len(entries), 2)

        # Delete an entry
        entries = mandb.inspect({'dets:band': 'f220'})
        mandb.remove_entry(entries[0])
        ## check file unreg'd
        c = mandb.conn.execute("select count(id) from files where name='y'")
        self.assertEqual(c.fetchall()[0][0], 0)

        # Delete another entry
        entries = mandb.inspect()
        mandb.remove_entry(entries[0])
        ## check file not unreg'd (because it's used twice)
        c = mandb.conn.execute("select count(id) from files where name='x'")
        self.assertEqual(1, c.fetchone()[0])


if __name__ == '__main__':
    unittest.main()
