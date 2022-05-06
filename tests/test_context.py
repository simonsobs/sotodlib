# Copyright (c) 2022 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Test Context class, including metadata and TOD loading.

"""

import unittest
import tempfile

from sotodlib.core import metadata, Context
from sotodlib.io.metadata import ResultSetHdfLoader, write_dataset, _decode_array

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

    def test_010_metadata(self):
        obs_list = ['obs%i' % i for i in range(2)]
        det_list = ['det%i' % i for i in range(4)]
        ctx = Context(self._write_context(MINIMAL_CONTEXT))

        ctx.obsdb = metadata.ObsDb()
        for i, obs_id in enumerate(obs_list):
            ctx.obsdb.update_obs(obs_id, data={'timestamp': 1680000000 + i*600})

        ctx.detdb = metadata.DetDb()
        ctx.detdb.create_table('base', [
            "`index` integer",
            "`band` string",
        ])
        for i, d in enumerate(det_list):
            ctx.detdb.add_props('base', d,
                                index=i,
                                band={0: 'f090', 1: 'f150'}[i % 2])

        req = ctx.get_obs('obs0', logic_only=True)
        self.assertEqual(len(req['dets']), len(det_list))
        req = ctx.get_obs('obs0', logic_only=True, dets=['det0'])
        self.assertEqual(len(req['dets']), 1)
        req = ctx.get_obs('obs0', logic_only=True, dets={'band': 'f090'})
        self.assertEqual(len(req['dets']), len(det_list) // 2)


if __name__ == '__main__':
    unittest.main()
