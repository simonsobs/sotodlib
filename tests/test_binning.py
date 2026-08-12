import unittest

import numpy as np
import so3g

from sotodlib import core
from sotodlib.tod_ops.binning import bin_signal


def _make_aman(ndets=3, nsamps=1000, signal=None):
    """
    Build a minimal AxisManager with ``dets`` and ``samps`` axes and an
    optional 2D ``signal`` field. ``signal`` defaults to zeros.
    """
    dets = [f'det{i}' for i in range(ndets)]
    aman = core.AxisManager(
        core.LabelAxis('dets', dets),
        core.OffsetAxis('samps', nsamps),
    )
    if signal is None:
        signal = np.zeros((ndets, nsamps), dtype='float32')
    aman.wrap('signal', signal, [(0, 'dets'), (1, 'samps')])
    return aman


class TestBinSignal(unittest.TestCase):
    """Tests for ``sotodlib.tod_ops.binning.bin_signal``."""

    def test_100_2d_binning(self):
        """ test 2d binning """
        ndets, nsamps, nbins = 2, 500, 10
        v = 7.
        aman = _make_aman(ndets, nsamps)
        signal = np.full((ndets, nsamps), v)
        bin_by = np.linspace(-1, 1, nsamps)
        out = bin_signal(aman, bin_by=bin_by, signal=None)
        out = bin_signal(aman, bin_by=bin_by, signal=None, bins=nbins)
        out = bin_signal(aman, bin_by=bin_by, signal='signal', bins=nbins)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, bins=nbins)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, range=(0, 1),
                         bins=nbins)

        # shape
        self.assertEqual(out['binned_signal'].shape, (ndets, nbins))
        self.assertEqual(out['binned_signal_sigma'].shape, (ndets, nbins))
        self.assertEqual(out['bin_counts'].shape, (ndets, nbins))
        self.assertEqual(out['bin_centers'].shape, (nbins,))
        self.assertEqual(out['bin_edges'].shape, (nbins + 1,))

        # numerical
        ok = ~np.isnan(out['binned_signal'])
        np.testing.assert_allclose(out['binned_signal'][ok], v)
        np.testing.assert_allclose(
            out['binned_signal_sigma'][ok], 0.0, atol=1e-12)

        # flags
        flags = so3g.proj.RangesMatrix.zeros((ndets, nsamps))
        out = bin_signal(aman, bin_by=bin_by, flags=flags)
        flags = so3g.proj.Ranges(nsamps)
        out = bin_signal(aman, bin_by=bin_by, flags=flags)

    def test_200_1d_binning(self):
        """ test 1d binning """
        ndets, nsamps, nbins = 2, 500, 10
        v = 7.
        aman = _make_aman(ndets, nsamps)
        signal = np.full(nsamps, v)
        bin_by = np.linspace(-1, 1, nsamps)
        out = bin_signal(aman, bin_by=bin_by, signal=None)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, bins=nbins)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, range=(0, 1),
                         bins=nbins)

        # shape
        self.assertEqual(out['binned_signal'].shape, (nbins,))
        self.assertEqual(out['binned_signal_sigma'].shape, (nbins,))
        self.assertEqual(out['bin_counts'].shape, (nbins,))

        # numerical
        ok = ~np.isnan(out['binned_signal'])
        np.testing.assert_allclose(out['binned_signal'][ok], v)
        np.testing.assert_allclose(
            out['binned_signal_sigma'][ok], 0.0, atol=1e-12)

        # flags
        flags = so3g.proj.RangesMatrix.zeros((ndets, nsamps))
        with self.assertRaises(ValueError):
            out = bin_signal(aman, bin_by=bin_by, signal=signal, flags=flags)
        flags = so3g.proj.Ranges(nsamps)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, flags=flags)

    def test_300_out_of_range_sample_robustness(self):
        """ test robustness to values in out of binning range """
        ndets, nsamps, nbins = 2, 500, 10
        v = 7.
        aman = _make_aman(ndets, nsamps)
        signal = np.full((ndets, nsamps), v)
        bin_by = np.linspace(-1, 1, nsamps)

        # inject outlier in out of range
        signal[:, bin_by < 0] = 1e10
        out = bin_signal(aman, bin_by=bin_by, signal=signal, range=(0, 1),
                         bins=nbins)

        # numerical
        ok = ~np.isnan(out['binned_signal'])
        np.testing.assert_allclose(out['binned_signal'][ok], v)
        np.testing.assert_allclose(
            out['binned_signal_sigma'][ok], 0.0, atol=1e-12)

    def test_400_arbitrary_length_of_signal(self):
        """ test binning arbitrary length of signal which has consistent
        shape as bin_by
        """
        ndets, nsamps, nbins = 2, 500, 10
        signal_nsamps = 200
        v = 7.
        aman = _make_aman(ndets, nsamps)

        # 1d
        signal = np.full(signal_nsamps, v)
        bin_by = np.linspace(-1, 1, signal_nsamps)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, bins=nbins)

        self.assertEqual(out['binned_signal'].shape, (nbins,))
        self.assertEqual(out['binned_signal_sigma'].shape, (nbins,))
        self.assertEqual(out['bin_counts'].shape, (nbins,))

        # 2d
        signal = np.full((ndets, signal_nsamps), v)
        bin_by = np.linspace(-1, 1, signal_nsamps)
        out = bin_signal(aman, bin_by=bin_by, signal=signal, bins=nbins)

        self.assertEqual(out['binned_signal'].shape, (ndets, nbins))
        self.assertEqual(out['binned_signal_sigma'].shape, (ndets, nbins))
        self.assertEqual(out['bin_counts'].shape, (ndets, nbins))
        self.assertEqual(out['bin_centers'].shape, (nbins,))
        self.assertEqual(out['bin_edges'].shape, (nbins + 1,))
