# Copyright (c) 2022 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Test Context class, including metadata and TOD loading.

"""

import unittest
import tempfile

from sotodlib import core
from sotodlib.core import metadata, Context, OBSLOADER_REGISTRY
from sotodlib.io.metadata import ResultSetHdfLoader, write_dataset, _decode_array
import so3g

import numpy as np
import os
import h5py
import sqlite3
import yaml

from ._helpers import mpi_multi

MINIMAL_CONTEXT = {
    'tags': {},
    'imports': [],
    'metadata': [],
    }


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class ContextTest(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _open_new(self, name, mode='w'):
        full = os.path.join(self.tempdir.name, name)
        if mode == 'create':
            open(full, 'w').close()
            fout = None
        else:
            fout = open(full, mode)
        return full, fout

    def _write_context(self, d):
        name, fout = self._open_new('context.yaml')
        fout.write(yaml.dump(d))
        fout.close()
        return name

    def test_000_smoke(self):
        """Test basic instantiation of Context and basic DBs from files."""
        ctx_file = self._write_context(MINIMAL_CONTEXT)
        ctx = Context(ctx_file)
        metadata.cli.main(args=['context', ctx_file])
        del ctx, ctx_file

        # Does context populate obsdb, detdb, obsfiledb from file?
        for key, cls in [
                ('obsdb', metadata.ObsDb),
                ('detdb', metadata.DetDb),
                ('obsfiledb', metadata.ObsFileDb),
        ]:
            cd = MINIMAL_CONTEXT.copy()
            db = cls()
            name, _ = self._open_new('db.sqlite', 'create')
            db.to_file(name)
            cd[key] = name
            ctx_file = self._write_context(cd)
            ctx = Context(ctx_file)
            self.assertIsInstance(getattr(ctx, key), cls,
                                  msg=f"Instantiating '{key}'")
            metadata.cli.main(args=['context', ctx_file])

    def test_010_cli(self):
        def save_db(src, k, suffix=''):
            if src.get(k) is None:
                return
            if isinstance(src[k], str):
                return
            db_file, _ = self._open_new(k + suffix, 'create')
            src[k].to_file(db_file)
            src[k] = db_file

        dataset_sim = DatasetSim()
        ctx = dataset_sim.get_context()
        for _db in ['obsfiledb', 'obsdb', 'detdb']:
            print(ctx)
            save_db(ctx, _db)
        for i, entry in enumerate(ctx.get('metadata', [])):
            save_db(entry, 'db', str(i))

        ctx_file = self._write_context(dict(ctx))
        metadata.cli.main(args=['context', ctx_file])

    def test_100_loads(self):
        dataset_sim = DatasetSim()

        # Test detector resolutions ...
        ctx = dataset_sim.get_context()
        n = len(dataset_sim.dets)
        obs_id = dataset_sim.obss['obs_id'][1]
        for selection, count in [
                (['read05'], 1),
                (np.array(['read05']), 1),
                (metadata.ResultSet(['readout_id'], [['read05']]), 1),
                ({'dets:readout_id': ['read05']}, 1),
                ({'dets:readout_id': np.array(['read05'])}, 1),
                ({'dets:detset': 'neard'}, 4),
                ({'dets:detset': ['neard']}, 4),
                ({'dets:detset': ['neard', 'fard']}, 8),
                ({'dets:detset': ['neard'], 'dets:readout_id': ['read00', 'read05']}, 1),
                ({'dets:band': ['f090']}, 4),
                ({'dets:band': ['f090'], 'dets:detset': ['neard']}, 2),
                ({'dets:det_id': ['NO_MATCH']}, 3),
        ]:
            meta = ctx.get_meta(obs_id, dets=selection)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")
            self.assertTrue('cal' in meta)
            self.assertTrue('flags' in meta)
            self.assertTrue('freeform' in meta)
            self.assertTrue('samps_only' in meta)

        # And tolerance of the detsets argument ...
        for selection, count in [
                ('neard', 4),
                (['neard'], 4),
                (np.array(['neard', 'fard']), 8),
        ]:
            meta = ctx.get_meta(obs_id, detsets=selection)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")
            self.assertTrue('cal' in meta)
            self.assertTrue('flags' in meta)
            self.assertTrue('freeform' in meta)
            self.assertTrue('samps_only' in meta)

        # And without detdb nor metadata
        ctx = dataset_sim.get_context(with_detdb=False)
        for selection, count in [
                ({'dets:detset': ['neard']}, 4),
        ]:
            meta = ctx.get_meta(obs_id, dets=selection)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")

        # Using free tags
        ctx = dataset_sim.get_context()
        for _id, _tags, count in [
                (obs_id, ['f090'], 4),
                (obs_id + ':f090', [], 4),
        ]:
            meta = ctx.get_meta(_id, free_tags=_tags)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")

        # And with obs_id as a dict.
        meta = ctx.get_meta({'obs_id': obs_id})
        self.assertEqual(meta.dets.count, 8)

        # You know, the kind of dict you get from obsdb.
        meta = ctx.get_meta(ctx.obsdb.get()[0])
        self.assertEqual(meta.dets.count, 8)

        # Check check mode
        checks = ctx.get_meta(obs_id, check=True)
        self.assertIsInstance(checks, list)

        # Check det_info mode
        det_info = ctx.get_det_info(obs_id)
        self.assertIsInstance(det_info, metadata.ResultSet)
        self.assertEqual(len(det_info), n)

        # Check tolerant mode
        ctx = dataset_sim.get_context(with_bad_metadata=True)
        with self.assertRaises(Exception):
            ctx.get_meta(obs_id)
        ctx.get_meta(obs_id, ignore_missing=True)

        # Check det_info with acceptable partial coverage.
        ctx = dataset_sim.get_context(with_incomplete_det_info='trim')
        meta = ctx.get_meta(obs_id)
        self.assertTrue('newcal' in meta.det_info)
        self.assertLess(meta.dets.count, 8)

        ctx = dataset_sim.get_context(with_incomplete_det_info='skip')
        meta = ctx.get_meta(obs_id)
        self.assertFalse('newcal' in meta.det_info)
        self.assertEqual(meta.dets.count, 8)

        ctx = dataset_sim.get_context(with_incomplete_det_info='trim',
                                      with_dependent_metadata='trim')
        meta = ctx.get_meta(obs_id)
        self.assertTrue('depends' in meta)

        # ... but skip should fail if dependent metadata requires what
        # it is providing.
        for dep in ['fail', 'skip', 'trim']:
            # arguably this could be asked to not raise, for dep =
            # skip or trim.
            ctx = dataset_sim.get_context(with_incomplete_det_info='skip',
                                          with_dependent_metadata=dep)
            with self.assertRaises(metadata.loader.IncompleteDetInfoError):
                ctx.get_meta(obs_id)

        # Check det_info with unacceptable partial coverage.
        ctx = dataset_sim.get_context(with_incomplete_det_info='fail')
        with self.assertRaises(metadata.loader.IncompleteMetadataError):
            ctx.get_meta(obs_id)

        # Manual overrides.
        meta = ctx.get_meta(obs_id, on_missing={'newcal': 'trim'})
        self.assertTrue('newcal' in meta.det_info)
        self.assertLess(meta.dets.count, 8)

        meta = ctx.get_meta(obs_id, on_missing={'newcal': 'skip'})
        self.assertFalse('newcal' in meta.det_info)
        self.assertEqual(meta.dets.count, 8)

        # Check metadata with acceptable partial coverage.
        ctx = dataset_sim.get_context(with_incomplete_metadata='trim')
        meta = ctx.get_meta(obs_id)
        self.assertTrue('othercal' in meta)
        self.assertLess(meta.dets.count, 8)

        dataset_sim = DatasetSim()
        obs_id = dataset_sim.obss['obs_id'][1]

        ctx = dataset_sim.get_context(with_incomplete_metadata='skip')
        meta = ctx.get_meta(obs_id)
        self.assertEqual(meta.dets.count, 8)
        self.assertFalse('othercal' in meta)

        # Check metadata with unacceptable partial coverage.
        ctx = dataset_sim.get_context(with_incomplete_metadata='fail')
        with self.assertRaises(metadata.loader.IncompleteMetadataError):
            ctx.get_meta(obs_id)

        # Manual overrides.
        meta = ctx.get_meta(obs_id, on_missing={'othercal': 'trim'})
        self.assertTrue('othercal' in meta)
        self.assertLess(meta.dets.count, 8)

        meta = ctx.get_meta(obs_id, on_missing={'othercal': 'skip'})
        self.assertEqual(meta.dets.count, 8)
        self.assertFalse('othercal' in meta)

        # Nothing wrong with good old obs 13 though
        ctx.get_meta('obs_number_13')

    def test_110_more_loads(self):
        dataset_sim = DatasetSim()
        n_det, n_samp = dataset_sim.det_count, dataset_sim.sample_count
        obs_id = dataset_sim.obss['obs_id'][1]

        # Test some TOD loading.
        ctx = dataset_sim.get_context()
        tod = ctx.get_obs(obs_id)
        self.assertEqual(tod.signal.shape, (n_det, n_samp))

        # ... metadata reconciliation check
        self.assertTrue(np.all(tod.cal > 0))
        for band, f in zip(tod.det_info['band'], tod.flags):
            # The f090 dets should have 0 flag intervals; f150 have 1
            self.assertEqual(len(f.ranges()), int(band == 'f150'))
        # Check if NO_MATCH det_id seemed to broadcast propertly ...
        self.assertEqual(list(tod.det_info['det_param'] == -1),
                         list(dataset_sim.dets['det_id'] == 'NO_MATCH'))
        for di, ds, AB1, AB2 in zip(tod.det_info.det_id, tod.det_info.detset,
                                    tod.XY, tod.focal_plane2.AB):
            self.assertEqual(AB1, len(ds) * (di == 'NO_MATCH'))
            self.assertEqual(AB2, len(ds) * (di == 'NO_MATCH'))

        tod = ctx.get_obs(obs_id + ':f090')
        self.assertEqual(tod.signal.shape, (n_det // 2, n_samp))

        tod = ctx.get_obs(obs_id, free_tags=['f150'])
        self.assertEqual(tod.signal.shape, (n_det // 2, n_samp))

        tod = ctx.get_obs(obs_id, samples=(10, n_samp // 2))
        self.assertEqual(tod.signal.shape, (n_det, n_samp // 2 - 10))

        # Loading via filename
        tod = ctx.get_obs(filename='obs_number_11_neard.txt')
        self.assertEqual(tod.signal.shape, (n_det // 2, n_samp))

        # Loading via prepopulated & modified meta
        meta = ctx.get_meta(obs_id)
        meta.restrict('dets', meta.dets.vals[:5])
        tod = ctx.get_obs(obs_id=meta)
        self.assertEqual(tod.signal.shape, (5, n_samp))
        tod = ctx.get_obs(meta=meta)
        self.assertEqual(tod.signal.shape, (5, n_samp))

        meta = ctx.get_meta(obs_id, samples=(10,90))
        meta.restrict('dets', meta.dets.vals[:5])
        tod = ctx.get_obs(meta)
        self.assertEqual(tod.signal.shape, (5, 80))

        det_info = ctx.get_det_info(obs_id)
        det_info = det_info.subset(rows=det_info['band'] == 'f090')
        tod = ctx.get_obs(obs_id, dets=det_info)
        self.assertEqual(tod.signal.shape, (n_det // 2, n_samp))

    def test_120_load_fields(self):
        dataset_sim = DatasetSim()
        obs_id = dataset_sim.obss['obs_id'][1]
        ctx = dataset_sim.get_context(with_axisman_ondisk=True)
        tod = ctx.get_obs(obs_id)
        self.assertCountEqual(tod.ondisk._fields.keys(), ['disk1', 'subaman'])
        self.assertCountEqual(tod.ondisk.subaman._fields.keys(), ['disk2'])

    def test_load_fields_resultset(self):
        dataset_sim = DatasetSim()
        obs_id = dataset_sim.obss['obs_id'][1]
        ctx = dataset_sim.get_context(with_resultset_ondisk=True)
        tod = ctx.get_obs(obs_id)
        # Make sure that 'band1' was loaded  into det_info
        self.assertTrue('band1' in tod.det_info._fields)
        # Make sure that 'band1' laods the same informaton as 'band'
        self.assertTrue((tod.det_info.band1 == tod.det_info.band).all())
        # Make sure that only 'pol_code' was loaded into ondisc_resultset
        self.assertCountEqual(tod.ondisk_resultset._fields.keys(), ['pol_code'])
        self.assertTrue((tod.det_info.pol_code == tod.ondisk_resultset.pol_code).all())

    def test_200_load_metadata(self):
        """Test the simple metadata load wrapper."""
        dataset_sim = DatasetSim()
        obs_id = dataset_sim.obss['obs_id'][1]

        ctx = dataset_sim.get_context()
        tod = ctx.get_meta(obs_id)
        for spec in ctx['metadata']:
            item = metadata.loader.load_metadata(tod, spec)
            assert(item is not None)
            item = metadata.loader.load_metadata(tod, spec, unpack=True)
            assert(item is not None)

    def test_300_clean_concat(self):
        """Test we can dodge concat failures."""
        dataset_sim = DatasetSim()
        obs_id = dataset_sim.obss['obs_id'][1]

        ctx = dataset_sim.get_context(with_inconcatable=True)
        m_entry = ctx['metadata'][0]
        orig_unpack = m_entry['unpack']

        # The discrepant fields should cause this to fail.
        with self.assertRaises(ValueError):
            tod = ctx.get_meta(obs_id)

        m_entry['drop_fields'] = ['discrepant_s', 'discrepant_v']
        tod = ctx.get_meta(obs_id)

        m_entry['drop_fields'] = ['discrepant_*']
        tod = ctx.get_meta(obs_id)

        m_entry['drop_fields'] = 'discrepant_*'
        tod = ctx.get_meta(obs_id)


class DatasetSim:
    """Provide in-RAM Context objects and tod/metadata loader functions
    for Context behavior testing.

    """
    def __init__(self):
        self.dets = metadata.ResultSet(
            ['readout_id', 'band', 'pol_code', 'x', 'y', 'detset', 'det_id', 'det_param'],
            [('read00', 'f090', 'A', 0.0, 0.0, 'neard', 'det00', 120.),
             ('read01', 'f090', 'B', 0.0, 0.0, 'neard', 'det01', 121.),
             ('read02', 'f150', 'A', 0.0, 0.0, 'neard', 'NO_MATCH', -1.),
             ('read03', 'f150', 'B', 0.0, 0.0, 'neard', 'NO_MATCH', -1.),
             ('read04', 'f090', 'A', 1.0, 0.0, 'fard',  'det04', 124.), 
             ('read05', 'f090', 'B', 1.0, 0.0, 'fard',  'det05', 125.),
             ('read06', 'f150', 'A', 1.0, 0.0, 'fard',  'NO_MATCH', -1.),
             ('read07', 'f150', 'B', 1.0, 0.0, 'fard',  'det07', 127.),
            ])

        self.obss = metadata.ResultSet(
            ['obs_id', 'timestamp', 'type', 'target'],
            [('obs_number_11', 1600010000., 'planet', 'uranus'),
             ('obs_number_12', 1600020000., 'survey', 'the_cmb'),
             ('obs_number_13', 1600030000., 'survey', 'the_cmb'),
            ])

        self.det_count = len(self.dets)
        self.sample_count = 100

        class _TestML(metadata.LoaderInterface):
            def from_loadspec(_self, load_params, **load_kwargs):
                return self.metadata_loader(load_params, **load_kwargs)

        OBSLOADER_REGISTRY['unittest_loader'] = self.tod_loader
        metadata.SuperLoader.register_metadata('unittest_loader', _TestML)

    def get_context(self, with_detdb=True, with_metadata=True,
                    with_bad_metadata=False, with_incomplete_det_info=False,
                    with_dependent_metadata=False,
                    with_incomplete_metadata=False,
                    with_inconcatable=False,
                    with_axisman_ondisk=False,
                    with_resultset_ondisk=False,
                    on_missing='trim'):
        """Args:
          with_detdb: if False, no detdb is included.
          with_metadata: if False, no metadata are included.
          with_bad_metadata: if True, an entry that refers to a
            non-existant sqlite database is included.
          with_incomplete_det_info: if True, include det_info entries
            that are missing some dets, or a complete detset.
          with_incomplete_metadata: if True, include entries that are
            missing some dets, or a complete detset.

        """
        detdb = metadata.DetDb()
        detdb.create_table('base', ['readout_id string',
                                    'pol_code string',
                                    'x float',
                                    'y float'])
        save_for_later = ['band', 'detset', 'det_id', 'det_param']
        for row in self.dets.subset(keys=[k for k in self.dets.keys
                                          if k not in save_for_later]):
            detdb.add_props('base', row['readout_id'], **row)

        obsdb = metadata.ObsDb()
        obsdb.add_obs_columns([f'{k} string' for k in self.obss.keys
                               if k not in ['obs_id', 'timestamp']])
        for row in self.obss:
            obsdb.update_obs(row['obs_id'], data=row)

        obsfiledb = metadata.ObsFileDb()
        for ds in ['neard', 'fard']:
            obsfiledb.add_detset(ds, self.dets['readout_id'][self.dets['detset'] == ds])
            for obs_id in self.obss['obs_id']:
                obsfiledb.add_obsfile(f'{obs_id}_{ds}.txt', obs_id, ds,
                                      0, self.sample_count)

        ctx = Context(data=MINIMAL_CONTEXT)
        ctx.obsdb = obsdb
        ctx.obsfiledb = obsfiledb
        if with_detdb:
            ctx.detdb = detdb
        ctx.reload(['loader'])
        ctx.update({
            'obs_loader_type': 'unittest_loader',
            'obs_colon_tags': ['band'],
        })

        if not with_metadata:
            return ctx

        class _ManifestDb(metadata.ManifestDb):
            def __init__(self, filename='unnamed', **kwargs):
                super().__init__(**kwargs)
                self._filename = filename
            def __repr__(self):
                return f'ManifestDb[{self._filename}]'

        def _db_single_dataset(filename):
            # Many dbs map all obs to a single file, based on
            # timestamp I guess.
            scheme = metadata.ManifestScheme() \
                             .add_data_field('loader') \
                             .add_range_match('obs:timestamp')
            db = _ManifestDb(scheme=scheme, filename=filename)
            db.add_entry(
                {'obs:timestamp': [0, 2e9], 'loader': 'unittest_loader'},
                filename)
            return db

        def _db_multi_dataset(filename, detsets=['neard', 'fard']):
            # ... while others are to a single file but by detset, using dataset=detset data arg.
            scheme = metadata.ManifestScheme() \
                      .add_data_field('dataset') \
                      .add_exact_match('dets:detset') \
                      .add_data_field('loader')
            db = _ManifestDb(scheme=scheme, filename=filename)
            for detset in detsets:
                db.add_entry(
                    {'loader': 'unittest_loader',
                     'dataset': detset,
                     'dets:detset': detset,
                     }, filename)
            return db

        # metadata: abscal.h5
        _scheme = metadata.ManifestScheme() \
                  .add_range_match('obs:timestamp') \
                  .add_data_field('loader') \
                  .add_data_field('dataset') \
                  .add_data_field('dets:band')
        abscal_db = _ManifestDb(scheme=_scheme, filename='abscal.h5')
        for band in ['f090', 'f150']:
            abscal_db.add_entry(
                {'obs:timestamp': [0, 2e9],
                 'loader': 'unittest_loader',
                 'dataset': f'mostly_{band}',
                 'dets:band': band
                },
                'abscal.h5')

        # metadata: some_flags.g3
        _scheme = metadata.ManifestScheme() \
                  .add_range_match('obs:timestamp') \
                  .add_data_field('loader') \
                  .add_data_field('frame_index', 'int') \
                  .add_data_field('dets:band')
        flags_db = _ManifestDb(scheme=_scheme, filename='some_flags.g3')
        for frame, band in enumerate(['f090', 'f150', 'f220']):
            flags_db.add_entry(
                {'obs:timestamp': [0, 2e9],
                 'loader': 'unittest_loader',
                 'frame_index': frame,
                 'dets:band': band
                 }, 'some_flags.g3')

        # metadata into context.
        ctx['metadata'] = [
            {'db': _db_single_dataset('bands.h5'),
             'det_info': True},
            {'db': _db_single_dataset('det_id.h5'),
             'det_info': True},
            {'db': abscal_db,
             'name': 'cal&abscal',
             'on_missing': on_missing},
            {'db': flags_db,
             'name': 'flags&',
             'on_missing': on_missing},
            {'db': _db_multi_dataset('some_detset_info.h5'),
             'name': 'focal_plane',
             'on_missing': on_missing},
            {'db': _db_multi_dataset('more_detset_info.h5'),
             'label': 'focal_plane2',
             'unpack': [
                 'focal_plane2',
                 'XY&AB',
                 ],
             'on_missing': on_missing},
            {'db': _db_single_dataset('freeform_info.h5'),
             'label': 'freeform',
             'unpack': 'freeform'},
            {'db': _db_single_dataset('samps_only.h5'),
             'label': 'samps_only',
             'unpack': 'samps_only'},
            {'db': _db_single_dataset('det_param.h5'),
             'det_info': True},
            {'db': _db_multi_dataset('detinfo_multimatch.h5'),
             'det_info': True},
            {'db': _db_multi_dataset('detinfo_nomatch.h5'),
             'det_info': True},
        ]

        if with_bad_metadata:
            # This entry is intended to cause a lookup failure.
            ctx['metadata'].insert(0, {
                'db': 'not-a-file.sqlite',
                'name': 'important_info&',
            })

        if with_incomplete_det_info:
            # det_info -- incomplete dataset; for key obs only a few
            ## dets will have field defined.
            ctx['metadata'].insert(0, {
                'db': _db_single_dataset('incomplete_det_info.h5'),
                'det_info': True,
                'on_missing': with_incomplete_det_info,
                'label': 'newcal',
            })

            if with_dependent_metadata:
                # metadata -- loads in association with det_info
                # fields defined through incomplete_det_info.
                ctx['metadata'].insert(1, {
                    'db': _db_single_dataset('dependent_metadata.h5'),
                    'on_missing': with_dependent_metadata,
                    'name': 'depends',
                })

        if with_incomplete_metadata:
            # metadata: incomplete_metadata.h5
            ## This is an incomplete dataset.
            _scheme = metadata.ManifestScheme() \
                      .add_exact_match('obs:obs_id') \
                      .add_data_field('loader')
            bad_meta_db = metadata.ManifestDb(scheme=_scheme)
            bad_meta_db.add_entry(
                {'obs:obs_id': 'obs_number_12', 'loader': 'unittest_loader'},
                 'incomplete_metadata.h5'
            )
            bad_meta_db.add_entry(
                {'obs:obs_id': 'obs_number_13', 'loader': 'unittest_loader'},
                 'incomplete_metadata.h5'
            )
            # This entry has incomplete metadata, which will cause a
            # failure, a trim, or omission of the product depending on
            # the value of with_incomplete_metadata.
            ctx['metadata'].insert(0, {
                'db': bad_meta_db,
                'name': 'othercal',
                'on_missing': with_incomplete_metadata,
                'label': 'othercal',
            })

        if with_inconcatable:
            _scheme = metadata.ManifestScheme() \
                      .add_exact_match('obs:obs_id') \
                      .add_exact_match('dets:detset') \
                      .add_data_field('loader')
            inconcat_db = metadata.ManifestDb(scheme=_scheme)
            for detset in ['neard', 'fard']:
                inconcat_db.add_entry(
                    {'obs:obs_id': 'obs_number_12',
                     'dets:detset': detset,
                     'loader': 'unittest_loader'},
                     'inconcat_metadata.h5'
                )
            ctx['metadata'].insert(0, {
                'db': inconcat_db,
                'unpack': ['inconcat&per_det'],
                'label': 'inconcat',
            })

        if with_axisman_ondisk:
            # Note this uses standard HDF5 loader, not our in-RAM loader.
            _scheme = metadata.ManifestScheme() \
                .add_exact_match('obs:obs_id') \
                .add_data_field('dataset')
            ondisk_db = metadata.ManifestDb(scheme=_scheme)
            ondisk_db._tempdir = tempfile.TemporaryDirectory()
            filename = os.path.join(ondisk_db._tempdir.name,
                                    'ondisk_metadata.h5')
            ondisk_db.add_entry(
                {'obs:obs_id': 'obs_number_12',
                 'dataset': 'xyz'},
                filename)
            ctx['metadata'].insert(0, {
                'db': ondisk_db,
                'unpack': 'ondisk',
                'load_fields': ['disk1', 'subaman.disk2', 'subaman.disk3'],
            })
            # Write the result.
            output = core.AxisManager(
                core.LabelAxis('dets', self.dets['readout_id']),
                core.OffsetAxis('samps', self.sample_count, 0))
            output.wrap('subaman', core.AxisManager(output.dets))
            for i in [1, 2]:
                output.wrap_new(f'disk{i}', shape=('samps',))
                output['subaman'].wrap_new(f'disk{i}', shape=('dets',))
            output.save(filename, 'xyz')

        if with_resultset_ondisk:
            _scheme = metadata.ManifestScheme() \
                .add_exact_match('obs:obs_id') \
                .add_data_field('dataset')
            ondisk_db = metadata.ManifestDb(scheme=_scheme)
            ondisk_db._tempdir = tempfile.TemporaryDirectory()
            filename = os.path.join(ondisk_db._tempdir.name,
                                    'ondisk_resultset_metadata.h5')
            write_dataset(self.dets, filename, 'obs_number_12')
            ondisk_db.add_entry(
                {'obs:obs_id': 'obs_number_12',
                 'dataset': 'obs_number_12'},
                filename)
            ctx['metadata'].insert(0, {
                'db': ondisk_db,
                'det_info': True,
                'load_fields': [{'band': 'dets:band1'}, {'readout_id': 'dets:readout_id'}],
            })

            ctx['metadata'].insert(0, {
                'db': ondisk_db,
                'unpack': 'ondisk_resultset',
                'load_fields': [{'readout_id': 'dets:readout_id'}, 'pol_code'],
            })

        return ctx


    def metadata_loader(self, kw, **load_kwargs):
        # For Superloader.
        filename = os.path.split(kw['filename'])[1]
        if filename == 'bands.h5':
            rs = self.dets.subset(keys=['readout_id', 'band'])
            rs.keys = ['dets:' + k for k in rs.keys]
            return rs
        elif filename == 'det_id.h5':
            rs = self.dets.subset(keys=['readout_id', 'det_id'])
            rs.keys = ['dets:' + k for k in rs.keys]
            return rs
        elif filename == 'det_param.h5':
            rs = self.dets.subset(keys=['det_id', 'det_param'])
            # Keep only 1 row with 'NO_MATCH'
            while sum(rs['det_id'] == 'NO_MATCH') > 1:
                rs.rows.pop(list(rs['det_id']).index('NO_MATCH'))
            rs.keys = ['dets:' + k for k in rs.keys]
            return rs
        elif filename == 'abscal.h5':
            rs = metadata.ResultSet(['dets:band', 'abscal'])
            if kw['dataset'] == 'mostly_f090':
                rs.append({'dets:band': 'f090',
                           'abscal': 90.})
                rs.append({'dets:band': 'f150',
                           'abscal': -1.})
            else:
                rs.append({'dets:band': 'f090',
                           'abscal': -1.})
                rs.append({'dets:band': 'f150',
                           'abscal': 150.})
            return rs
        elif filename == 'some_flags.g3':
            output = core.AxisManager(
                core.LabelAxis('dets', self.dets['readout_id']),
                core.OffsetAxis('samps', self.sample_count, 0))
            if kw['frame_index'] == 0:
                # Frame 0 declares zeros for all dets, but is
                # restricted by the ManifestDb to only apply to f090.
                output.wrap('flags', so3g.proj.RangesMatrix.zeros(output.shape),
                            [(0, 'dets'), (1, 'samps')])
            elif kw['frame_index'] == 1:
                # Frame 1 declares ones for only the f150 dets (and is
                # marked to apply to f150 in the ManifestDb).
                output.wrap('flags', so3g.proj.RangesMatrix.ones(output.shape),
                            [(0, 'dets'), (1, 'samps')])
                output.restrict('dets',
                                self.dets['readout_id'][self.dets['band'] == 'f150'])
            else:
                # Frame 2 is marked in the ManifestDb as being for
                # f220; the det_info preprocessing should prevent this
                # from ever getting requested.
                raise RuntimeError('metadata system asked for f220 data')
            return output
        elif filename == 'freeform_info.h5':
            output = core.AxisManager()
            output.wrap('number1', 1.)
            output.wrap('numbers', np.array([1,2,3]))
            return output
        elif filename == 'samps_only.h5':
            output = core.AxisManager(
                core.OffsetAxis('samps', self.sample_count, 0))
            output.wrap_new('encoder_something', shape=('samps', ))[:] = \
                np.arange(self.sample_count)
            return output
        elif filename == 'some_detset_info.h5':
            rs = metadata.ResultSet(['dets:readout_id', 'x', 'y'])
            for row in self.dets.subset(rows=self.dets['detset'] == kw['dets:detset']):
                rs.append({'dets:readout_id': row['readout_id'],
                           'x': 100., 'y': 102.})
            return rs
        elif filename == 'more_detset_info.h5':
            # Here, match against det_id and makes sure there's only
            # one NO_MATCH entry _per detset_.  AB will be 0 unless
            # NO_MATCH, in which case it's the len of the detset name.
            rs = metadata.ResultSet(['dets:det_id', 'AB'])
            for row in self.dets.subset(rows=self.dets['detset'] == kw['dets:detset']):
                rs.append({'dets:det_id': row['det_id'],
                           'AB': len(kw['dets:detset']) * (row['det_id'] == 'NO_MATCH')})
            while sum(rs['dets:det_id'] == 'NO_MATCH') > 1:
                rs.rows.pop(list(rs['dets:det_id']).index('NO_MATCH'))
            return rs
        elif filename == 'detinfo_multimatch.h5':
            # This is to test whether det_info fields (bp_code) can be
            # populated based on matching against multiple other
            # det_info fields (band, pol_code).
            rs = metadata.ResultSet(
                ['dets:band', 'dets:pol_code', 'dets:bp_code'], [
                    ['f090', 'A', 'f090-A'],
                    ['f090', 'B', 'f090-B'],
                    ['f150', 'A', 'f150-A'],
                    ['f150', 'B', 'f150-B'],
                ])
            return rs
        elif filename == 'detinfo_nomatch.h5':
            # This is to test that det_info can reconcile against
            # multiple det_id=NO_MATCH entries.  So at least one of
            # the detsets should have multiple det_id=NO_MATCH
            # entries.
            farness = 100. if kw['dataset'] == 'fard' else 0.001
            rs = metadata.ResultSet(['dets:det_id', 'dets:farness'])
            for row in self.dets.subset(rows=self.dets['detset'] == kw['dataset']):
                rs.append({'dets:det_id': row['det_id'],
                           'dets:farness': farness})
            return rs
        elif filename == 'incomplete_det_info.h5':
            rs = metadata.ResultSet(['dets:readout_id', 'dets:newcal'])
            for row in self.dets.subset(rows=[0,1,2]):
                rs.append({'dets:readout_id': row['readout_id'],
                           'dets:newcal':20})
            return rs
        elif filename == 'dependent_metadata.h5':
            rs = metadata.ResultSet(['dets:newcal', 'newcal_number'])
            rs.append({'dets:newcal': 20,
                       'newcal_number': 'twenty'})
            return rs
        elif filename == 'incomplete_metadata.h5':
            rs = metadata.ResultSet([
                'obs:obs_id',
                'dets:readout_id',
                'othercal'
            ])
            if kw['obs:obs_id']=='obs_number_12':
                for row in self.dets.subset(rows=[0,1,2]):
                    rs.append({'obs:obs_id': 'obs_number_12',
                               'dets:readout_id': row['readout_id'],
                               'othercal':40})
            elif kw['obs:obs_id'] =='obs_number_13':
                for row in self.dets:
                    rs.append({'obs:obs_id': 'obs_number_13',
                               'dets:readout_id': row['readout_id'],
                               'othercal':40})
            return rs

        elif filename == 'inconcat_metadata.h5':
            ds = self.dets.subset(rows=self.dets['detset'] == kw['dets:detset'])
            output = core.AxisManager(
                core.LabelAxis('dets', ds['readout_id']),
                core.OffsetAxis('samps', self.sample_count))
            # These are ok ...
            output.wrap_new('per_det', ('dets',), dtype=int)[:] = \
                np.arange(len(ds))
            output.wrap_new('duplicated', (100,))[:] = 100.
            # But put a nan in there ...
            output['duplicated'][10] = np.nan
            output.wrap('same_nan', np.nan)
            # And these are not concatenable ...
            discrepancy = int(kw['dets:detset'] == 'neard')
            output.wrap('discrepant_s', discrepancy)
            output.wrap_new('discrepant_v', (100,))[:] = discrepancy
            return output

        raise ValueError(f'metadata request for "{filename}"')

    def tod_loader(self, obsfiledb, obs_id, dets=None, prefix=None,
                   samples=None, no_signal=None,
                   **kwargs):
        # For Context.get_obs
        if samples is None:
            samples = 0, None
        samples = list(samples)
        if samples[1] is None:
            samples[1] = self.sample_count
        samples = [s if s >= 0 else self.sample_count + s
                   for s in samples]

        tod = core.AxisManager(core.LabelAxis('dets', dets),
                               core.OffsetAxis('samps', samples[1] - samples[0], samples[0]))
        tod.wrap_new('signal', ('dets', 'samps'))
        return tod


if __name__ == '__main__':
    unittest.main()
