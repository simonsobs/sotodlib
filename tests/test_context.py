# Copyright (c) 2022 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Test Context class, including metadata and TOD loading.

"""

import unittest
import tempfile

from sotodlib import core
from sotodlib.core import metadata, Context, OBSLOADER_REGISTRY
from sotodlib.io.metadata import ResultSetHdfLoader, write_dataset, _decode_array

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
            ctx = Context(self._write_context(cd))
            self.assertIsInstance(getattr(ctx, key), cls)

    def test_100_loads(self):
        dataset_sim = DatasetSim()

        # Test detector resolutions ...
        ctx = dataset_sim.get_context()
        n = len(dataset_sim.dets)
        obs_id = dataset_sim.obss['obs_id'][1]
        for selection, count in [
                ({'dets:readout_id': ['det05']}, 1),
                ({'dets:readout_id': np.array(['det05'])}, 1),
                ({'dets:detset': 'neard'}, 4),
                ({'dets:detset': ['neard']}, 4),
                ({'dets:detset': ['neard', 'fard']}, 8),
                ({'dets:detset': ['neard'], 'dets:readout_id': ['det00', 'det05']}, 1),
                ({'dets:readout_id': ['xx']}, 0),
                ({'dets:band': ['f090']}, 4),
                ({'dets:band': ['f090'], 'dets:detset': ['neard']}, 2),
        ]:
            meta = ctx.get_meta(obs_id, dets=selection)
            self.assertEqual(meta.dets.count, count, msg=f"{selection}")

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
        meta = ctx.get_meta({'obs:obs_id': obs_id})
        self.assertEqual(meta.dets.count, 8)

        # Check check mode
        checks = ctx.get_meta(obs_id, check=True)
        self.assertIsInstance(checks, list)

        # Check det_info mode
        det_info = ctx.get_det_info(obs_id)
        self.assertIsInstance(det_info, metadata.ResultSet)
        self.assertEqual(len(det_info), n)

    def test_110_more_loads(self):
        dataset_sim = DatasetSim()
        n_det, n_samp = dataset_sim.det_count, dataset_sim.sample_count
        obs_id = dataset_sim.obss['obs_id'][1]

        # Test some TOD loading.
        ctx = dataset_sim.get_context()
        tod = ctx.get_obs(obs_id)
        self.assertEqual(tod.signal.shape, (n_det, n_samp))

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
            ['readout_id', 'band', 'pol_code', 'x', 'y'],
            [('det00', 'f090', 'A', 0.0, 0.0),
             ('det01', 'f090', 'B', 0.0, 0.0),
             ('det02', 'f150', 'A', 0.0, 0.0),
             ('det03', 'f150', 'B', 0.0, 0.0),
             ('det04', 'f090', 'A', 1.0, 0.0),
             ('det05', 'f090', 'B', 1.0, 0.0),
             ('det06', 'f150', 'A', 1.0, 0.0),
             ('det07', 'f150', 'B', 1.0, 0.0)])

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

    def get_context(self, with_detdb=True, with_metadata=True):
        detdb = metadata.DetDb()
        detdb.create_table('base', ['readout_id string',
                                    'pol_code string',
                                    'x float',
                                    'y float'])
        save_for_later = ['band']
        for row in self.dets.subset(keys=[k for k in self.dets.keys
                                          if k not in save_for_later]):
            detdb.add_props('base', row['readout_id'], **row)

        obsdb = metadata.ObsDb()
        obsdb.add_obs_columns([f'{k} string' for k in self.obss.keys
                               if k not in ['obs_id', 'timestamp']])
        for row in self.obss:
            obsdb.update_obs(row['obs_id'], data=row)

        obsfiledb = metadata.ObsFileDb()
        obsfiledb.add_detset('neard', self.dets['readout_id'][:len(self.dets)//2])
        obsfiledb.add_detset('fard',  self.dets['readout_id'][len(self.dets)//2:])
        for obs_id in self.obss['obs_id']:
            obsfiledb.add_obsfile(f'{obs_id}_neard.txt', obs_id, 'neard',
                                  0, self.sample_count)
            obsfiledb.add_obsfile(f'{obs_id}_fard.txt', obs_id, 'fard',
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

        # metadata: bands.h5
        _scheme = metadata.ManifestScheme() \
                  .add_data_field('loader') \
                  .add_range_match('obs:timestamp')
        bands_db = metadata.ManifestDb(scheme=_scheme)
        bands_db.add_entry({'obs:timestamp': [0, 2e9], 'loader': 'unittest_loader'}, 'bands.h5')

        if with_metadata:
            ctx['metadata'] = [
                {'db': bands_db,
                 'det_info': True,
                 'dets_key': 'readout_id'},
            ]

        return ctx

    def metadata_loader(self, kw):
        # For Superloader.
        filename = os.path.split(kw['filename'])[1]
        if filename == 'bands.h5':
            rs = self.dets.subset(keys=['readout_id', 'band'])
            rs.keys = ['dets:' + k for k in rs.keys]
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
