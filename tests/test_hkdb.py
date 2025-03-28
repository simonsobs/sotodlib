import unittest
import tempfile

from sotodlib.io import hkdb

import os
import so3g
import numpy as np
import time
from spt3g import core


def write_hk(filepath):
    """
    Writes an HK g3 file with data from an agent `test_agent`, feed `test_feed`,
    and field `test_field`.

    Returns
    ---------
    t0: float
        start time of generated data.
    t1: float
        End time of generated data.
    nsamps: int
        Number of samples in timestream
    """
    hksess = so3g.hk.HKSessionHelper()

    writer = core.G3Writer(filepath)

    frames = []
    frames.append(hksess.session_frame())
    frames.append(hksess.status_frame())

    t0 = time.time()
    t1 = t0 + 10 * 60
    sample_rate = 10 # Hz
    frame_len = 60
    times = np.arange(t0, t1, 1/sample_rate)
    data = np.zeros(len(times), dtype=float)
    frame_idxs = (times - t0) // frame_len

    prov_address = 'site.test_agent.feeds.test_feed'
    prov_session_id = int(time.time())

    for fr_idx in np.unique(frame_idxs):
        m = frame_idxs == fr_idx
        frame = hksess.data_frame(0)
        tsmap = core.G3TimesampleMap()
        tsmap.times = core.G3VectorTime(times[m] * core.G3Units.s)
        tsmap['test_field'] = core.G3VectorDouble(data[m].tolist())
        frame['block_names'] = core.G3VectorString(['test_block'])
        frame['blocks'].append(tsmap)
        frame['address'] = core.G3String(prov_address)
        frame['provider_session_id'] = core.G3Int(prov_session_id)
        frames.append(frame)

    for f in frames:
        writer(f)
    writer(core.G3Frame(core.G3FrameType.EndProcessing))
    return t0, t1, len(times)


class HkDbTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_hkdb(self):
        hk_root = os.path.join(self.tempdir.name, 'hk')
        hkdb_path = os.path.join(self.tempdir.name, 'hk.db')
        hk_file = os.path.join(hk_root, '12345_hk_test', 'hk.g3')
        os.makedirs(os.path.dirname(hk_file), exist_ok=True)
        hkcfg = hkdb.HkConfig(
            hk_root=hk_root,
            db_file=hkdb_path,
            echo_db=True,
            aliases={
                'test': 'test_agent.test_feed.test_field'
            }
        )
        t0, t1, nsamp = write_hk(hk_file)
        hkdb.update_index_all(hkcfg)
        load_spec = hkdb.LoadSpec(
            cfg=hkcfg,
            fields=['test'],
            start=t0-1, end=t1+1,
        )
        res = hkdb.load_hk(load_spec)

        self.assertEqual(len(res.test[0]), nsamp)

        feeds = hkdb.get_feed_list(load_spec)
        self.assertEqual(feeds, ['test_agent.test_feed.*'])
