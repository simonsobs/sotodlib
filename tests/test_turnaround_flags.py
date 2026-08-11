import unittest
import numpy as np

import so3g
from sotodlib import core
from sotodlib.tod_ops.flags import get_turnaround_flags


def _make_aman_with_sinusoidal_scan(
    nsamps=8000,
    fs=200,
    scan_freq=0.05,
    az_amp_deg=20.,
    ndets=3
):
    """
    Build a minimal AxisManager whose ``boresight.az`` is a smooth
    sinusoid — 2 * scan_freq * duration full cycles, so it contains
    several turnarounds.
    """
    dets = [f'det{i}' for i in range(ndets)]
    aman = core.AxisManager(
        core.LabelAxis('dets', dets),
        core.OffsetAxis('samps', nsamps),
    )
    t = np.arange(nsamps) / fs
    aman.wrap('timestamps', t, [(0, 'samps')])

    boresight = core.AxisManager(core.OffsetAxis('samps', nsamps))
    az_rad = np.deg2rad(az_amp_deg) * np.sin(2 * np.pi * scan_freq * t)
    boresight.wrap('az', az_rad, [(0, 'samps')])
    aman.wrap('boresight', boresight)

    aman.wrap('signal',
              np.zeros((ndets, nsamps), dtype='float32'),
              [(0, 'dets'), (1, 'samps')])
    aman.wrap('flags', core.FlagManager.for_tod(aman, 'dets', 'samps'))
    return aman


class TestGetTurnaroundFlags(unittest.TestCase):
    """Tests for ``sotodlib.tod_ops.flags.get_turnaround_flags``."""

    def test_100_returns_three(self):
        """az method now returns (ta, left, right), matching scanspeed."""
        for method in ('scanspeed', 'az'):
            aman = _make_aman_with_sinusoidal_scan()
            ta, left, right = get_turnaround_flags(
                aman, method=method,
                merge=False, merge_lr=False, merge_subscans=False)
            self.assertGreater(ta.mask().sum(), 0)
            self.assertGreater(left.mask().sum(), 0)
            self.assertGreater(right.mask().sum(), 0)

    def test_200_left_right(self):
        """Left and right scan flags never overlap."""
        for method in ('scanspeed', 'az'):
            aman = _make_aman_with_sinusoidal_scan()
            ta, left, right = get_turnaround_flags(
                aman, method=method,
                merge=False, merge_lr=False, merge_subscans=False)
            m_ta = ta.mask()[0]  if ta.mask().ndim  == 2 else ta.mask()
            m_l = left.mask()[0] if left.mask().ndim == 2 else left.mask()
            m_r = right.mask()[0] if right.mask().ndim == 2 else right.mask()
            self.assertFalse((m_l & m_r).any(),
                             msg=f'left/right overlap in {method}')
            # sign match
            az = aman.boresight.az
            daz = np.diff(az, prepend=az[0])
            self.assertTrue((daz[m_r & ~m_ta] >= 0).all())
            self.assertTrue((daz[m_l & ~m_ta] < 0).all())

    def test_300_scanspeed_scalar_vs_tuple_buffer(self):
        """Scalar t_buffer=X gives the same result as tuple (X/2, X/2)."""
        aman = _make_aman_with_sinusoidal_scan()
        ta1, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=2.0, az_buffer=None,
            merge=False, merge_lr=False, merge_subscans=False)
        ta2, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=(1.0, 1.0), az_buffer=None,
            merge=False, merge_lr=False, merge_subscans=False)
        np.testing.assert_array_equal(ta1.mask(), ta2.mask())
        ta1, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=None, az_buffer=2.0,
            merge=False, merge_lr=False, merge_subscans=False)
        ta2, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=None, az_buffer=(1.0, 1.0),
            merge=False, merge_lr=False, merge_subscans=False)
        np.testing.assert_array_equal(ta1.mask(), ta2.mask())

    def test_310_scanspeed_asymmetric_buffer_widens_one_side(self):
        """A larger t_after than t_before produces asymmetric flag around peaks."""
        aman = _make_aman_with_sinusoidal_scan()
        # Symmetric (0.5 s each side)
        ta_sym, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=(0.5, 0.5),
            merge=False, merge_lr=False, merge_subscans=False)
        # Asymmetric: 0.1 s before, 1.0 s after -> more samples after peak
        ta_asy, *_ = get_turnaround_flags(
            aman, method='scanspeed', t_buffer=(0.1, 1.0),
            merge=False, merge_lr=False, merge_subscans=False)
        # Total number of flagged samples differs
        self.assertNotEqual(ta_sym.mask().sum(), ta_asy.mask().sum())

    def test_400_az_buffer_extends_flag(self):
        """Non-zero az_buffer flags more samples than no buffer."""
        aman = _make_aman_with_sinusoidal_scan()
        ta_none, *_ = get_turnaround_flags(
            aman, method='az', qlim=1, az_buffer=None,
            merge=False, merge_lr=False, merge_subscans=False)
        ta_buf, *_ = get_turnaround_flags(
            aman, method='az', qlim=1, az_buffer=1.0,
            merge=False, merge_lr=False, merge_subscans=False)
        self.assertGreater(ta_buf.mask().sum(), ta_none.mask().sum())

    def test_410_az_buffer_asymmetric_differs(self):
        """Asymmetric az_buffer produces different flag than symmetric of the same total."""
        aman = _make_aman_with_sinusoidal_scan()
        b = 1.0   # 1 deg total (each side 0.5)
        ta_sym, *_ = get_turnaround_flags(
            aman, method='az', az_buffer=b,
            merge=False, merge_lr=False, merge_subscans=False)
        ta_asym, *_ = get_turnaround_flags(
            aman, method='az', az_buffer=(0.0, b),   # buffer only after
            merge=False, merge_lr=False, merge_subscans=False)
        self.assertNotEqual(ta_sym.mask().sum(), ta_asym.mask().sum())

    def test_500_scanspeed_truncate(self):
        """Truncate unstable scan in scanspeed method"""
        nsamps = 8000
        aman = _make_aman_with_sinusoidal_scan(nsamps=nsamps, scan_freq=0.05)
        _aman = _make_aman_with_sinusoidal_scan(nsamps=nsamps, scan_freq=0.20)
        # inject faster scan at the beginning
        # the results are merged correctly and truccated
        aman.boresight.az[:50] = _aman.boresight.az[:50]
        _ = get_turnaround_flags(
            aman, method='scanspeed', truncate=True,
            merge=True, merge_lr=True, merge_subscans=True)
        self.assertTrue(aman.samps.count < nsamps)


if __name__ == '__main__':
    unittest.main()
