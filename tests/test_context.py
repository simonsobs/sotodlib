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
        ctx_file = self._write_context(MINIMAL_CONTEXT)
        ctx = Context(ctx_file)

    def test_001_dbs(self):
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
            self.assertIsInstance(getattr(ctx, key), cls)
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
                ({'dets:det_id': ['NO_MATCH']}, 2),
        ]:
            meta = ctx.get_meta(obs_id, dets=selection)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")
            self.assertTrue('cal' in meta)
            self.assertTrue('flags' in meta)

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


class DatasetSim:
    """Provide in-RAM Context objects and tod/metadata loader functions
    for Context behavior testing.

    """
    def __init__(self):
        self.dets = metadata.ResultSet(
            ['readout_id', 'band', 'pol_code', 'x', 'y', 'detset', 'det_id', 'det_param'],
            [('read00', 'f090', 'A', 0.0, 0.0, 'neard', 'det00', 120.),
             ('read01', 'f090', 'B', 0.0, 0.0, 'neard', 'det01', 121.),
             ('read02', 'f150', 'A', 0.0, 0.0, 'neard', 'det02', 122.),
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
            def from_loadspec(_self, load_params):
                return self.metadata_loader(load_params)

        OBSLOADER_REGISTRY['unittest_loader'] = self.tod_loader
        metadata.SuperLoader.register_metadata('unittest_loader', _TestML)

    def get_context(self, with_detdb=True, with_metadata=True,
                    with_bad_metadata=False):
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

        # metadata: bands.h5
        _scheme = metadata.ManifestScheme() \
                  .add_data_field('loader') \
                  .add_range_match('obs:timestamp')
        bands_db = metadata.ManifestDb(scheme=_scheme)
        bands_db.add_entry(
            {'obs:timestamp': [0, 2e9], 'loader': 'unittest_loader'},
            'bands.h5')

        # metadata: det_id.h5
        _scheme = metadata.ManifestScheme() \
                  .add_data_field('loader') \
                  .add_range_match('obs:timestamp')
        det_id_db = metadata.ManifestDb(scheme=_scheme)
        det_id_db.add_entry(
            {'obs:timestamp': [0, 2e9], 'loader': 'unittest_loader'},
            'det_id.h5')

        # metadata: det_param.h5
        _scheme = metadata.ManifestScheme() \
                  .add_data_field('loader') \
                  .add_range_match('obs:timestamp')
        det_par_db = metadata.ManifestDb(scheme=_scheme)
        det_par_db.add_entry(
            {'obs:timestamp': [0, 2e9], 'loader': 'unittest_loader'},
            'det_param.h5')

        # metadata: abscals.h5
        _scheme = metadata.ManifestScheme() \
                  .add_range_match('obs:timestamp') \
                  .add_data_field('loader') \
                  .add_data_field('dataset') \
                  .add_data_field('dets:band')
        abscal_db = metadata.ManifestDb(scheme=_scheme)
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
        flags_db = metadata.ManifestDb(scheme=_scheme)
        for frame, band in enumerate(['f090', 'f150', 'f220']):
            flags_db.add_entry(
                {'obs:timestamp': [0, 2e9],
                 'loader': 'unittest_loader',
                 'frame_index': frame,
                 'dets:band': band
                 }, 'some_flags.g3')

        # metadata: some_detset_info.h5
        ## This matches purely based on dets:* properties.
        _scheme = metadata.ManifestScheme() \
                  .add_data_field('dataset') \
                  .add_exact_match('dets:detset') \
                  .add_data_field('loader')
        info_db = metadata.ManifestDb(scheme=_scheme)
        for detset in ['neard', 'fard']:
            info_db.add_entry(
                {'loader': 'unittest_loader',
                 'dataset': detset,
                 'dets:detset': detset,
                 }, 'some_detset_info.h5')

        # metadata into context.
        ctx['metadata'] = [
            {'db': bands_db,
             'det_info': True,
            },
            {'db': det_id_db,
             'det_info': True,
            },
            {'db': abscal_db,
             'name': 'cal&abscal'},
            {'db': flags_db,
             'name': 'flags&'},
            {'db': info_db,
             'name': 'focal_plane'},
            {'db': det_par_db,
             'det_info': True,
             'multi': True,
            },
        ]

        if with_bad_metadata:
            # This entry is intended to cause a lookup failure.
            ctx['metadata'].insert(0, {
                'db': 'not-a-file.sqlite',
                'name': 'important_info&',
            })

        return ctx

    def metadata_loader(self, kw):
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
        elif filename == 'some_detset_info.h5':
            rs = metadata.ResultSet(['dets:readout_id', 'x', 'y'])
            for row in self.dets.subset(rows=self.dets['detset'] == kw['dets:detset']):
                rs.append({'dets:readout_id': row['readout_id'],
                           'x': 100., 'y': 102.})
            return rs
        else:
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
