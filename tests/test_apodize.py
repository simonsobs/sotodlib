import unittest

import numpy as np
import so3g

from sotodlib import core
from sotodlib.tod_ops.apodize import apodize_cosine


def _make_aman(ndets=2, nsamps=100000):
    """
    Build a minimal AxisManager with ``dets`` and ``samps`` axes and an
    optional 2D ``signal`` field. ``signal`` defaults to ones.
    """
    dets = [f'det{i}' for i in range(ndets)]
    aman = core.AxisManager(
        core.LabelAxis('dets', dets),
        core.OffsetAxis('samps', nsamps),
    )
    signal = np.ones((ndets, nsamps), dtype='float32')
    aman.wrap('signal', signal, [(0, 'dets'), (1, 'samps')])
    aman.wrap('flags', core.AxisManager())
    return aman


class TestApodizeSignal(unittest.TestCase):
    """Tests for ``sotodlib.tod_ops.apodize.apodize_cosine``."""

    def test_apodize_edge(self):
        ndets, nsamps, apodize_samps = 2, 100000, 2000

        aman = _make_aman(ndets, nsamps)
        apodize_cosine(aman)
        np.testing.assert_allclose(aman.signal[:, 0],  0.0, atol=1e-6)
        np.testing.assert_allclose(aman.signal[:, -1], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            aman.signal[:, apodize_samps:-apodize_samps], 1.0, atol=1e-6)
        self.assertEqual(aman.signal.dtype, np.float32)

    def test_apodize_flags(self):
        ndets, nsamps, apodize_samps = 2, 10000, 200
        aman = _make_aman(ndets, nsamps)
        mask = np.zeros((ndets, nsamps), dtype=bool)
        mask[0, 1000:2000] = True
        flags = so3g.proj.RangesMatrix.from_mask(mask)
        apodize_cosine(aman, flags=flags, apodize_samps=apodize_samps)

        np.testing.assert_allclose(aman.signal[0, 1000:2000], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            aman.signal[0, :1000-apodize_samps], 1.0, atol=1e-6)
        np.testing.assert_allclose(
            aman.signal[0, 2000+apodize_samps:], 1.0, atol=1e-6)
        np.testing.assert_allclose(aman.signal[1, 1000:2000], 1.0, atol=1e-6)
        self.assertEqual(aman.signal.dtype, np.float32)
