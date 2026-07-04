"""Tests of samps-axis downsampling (sotodlib.preprocess.downsample).

Only the core ``down_sample_aman`` function and its config helpers are
tested here; the preprocess layer-boundary integration is separate.

Intended location: tests/test_downsample.py
"""
import unittest
import numpy as np

from so3g.proj import Ranges, RangesMatrix

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from sotodlib import core
from sotodlib.tod_ops.downsample import down_sample_aman 
from sotodlib.preprocess.preprocess_util import (
    parse_downsample_cfg, downsample_cfg_aman,
    check_saved_downsample)


NSAMP = 1000
OFFSET = 25
FACTOR = 10
DETS = ['det0', 'det1', 'det2']
# local indices of kept samples (absolute index % FACTOR == 0)
START = (-OFFSET) % FACTOR
IDX = np.arange(START, NSAMP, FACTOR)


def make_tod(n=NSAMP, ofs=OFFSET):
    tod = core.AxisManager(
        core.LabelAxis('dets', DETS),
        core.OffsetAxis('samps', n, ofs, 'obs_ref'))
    # timestamps encode the absolute sample index
    tod.wrap('timestamps', np.arange(ofs, ofs + n, dtype=float),
             [(0, 'samps')])
    tod.wrap('signal',
             np.arange(len(DETS) * n, dtype=float).reshape(len(DETS), n),
             [(0, 'dets'), (1, 'samps')])
    return tod


class TestDownsample(unittest.TestCase):

    # Axis math and the absolute-grid convention.

    def test_100_axis(self):
        out = down_sample_aman(make_tod(), FACTOR)
        self.assertEqual(out.samps.count, len(IDX))
        # output offset is ceil(OFFSET / FACTOR) on the downsampled grid
        self.assertEqual(out.samps.offset, -(-OFFSET // FACTOR))
        self.assertEqual(out.samps.origin_tag, 'obs_ref')
        # every kept sample sits on the absolute grid ...
        np.testing.assert_array_equal(out.timestamps % FACTOR,
                                      np.zeros(out.samps.count))
        # ... at the absolute position implied by the new offset
        np.testing.assert_array_equal(
            out.timestamps,
            (out.samps.offset + np.arange(out.samps.count)) * FACTOR)

    def test_110_arrays(self):
        tod = make_tod()
        cube = np.arange(len(DETS) * 2 * NSAMP,
                         dtype=float).reshape(len(DETS), 2, NSAMP)
        tod.wrap('cube', cube, [(0, 'dets'), (2, 'samps')])
        tod.wrap('per_det', np.ones(len(DETS)), [(0, 'dets')])
        tod.wrap('scalar', 42)

        out = down_sample_aman(tod, FACTOR)
        np.testing.assert_array_equal(out.signal, tod.signal[:, IDX])
        np.testing.assert_array_equal(out.cube, tod.cube[:, :, IDX])
        np.testing.assert_array_equal(out.per_det, tod.per_det)
        self.assertEqual(out.scalar, 42)
        # input is not modified, and copies are decoupled
        self.assertEqual(tod.samps.count, NSAMP)
        out.per_det[0] = -1.
        self.assertNotEqual(tod.per_det[0], -1.)

    def test_120_child(self):
        # wrapped AxisManagers are downsampled recursively
        tod = make_tod()
        child = core.AxisManager(core.LabelAxis('dets', DETS),
                                 core.OffsetAxis('samps', NSAMP, OFFSET,
                                                 'obs_ref'))
        child.wrap('x', np.arange(OFFSET, OFFSET + NSAMP, dtype=float),
                   [(0, 'samps')])
        tod.wrap('preprocess', child)
        out = down_sample_aman(tod, FACTOR)
        self.assertEqual(out.preprocess.samps.count, len(IDX))
        np.testing.assert_array_equal(out.preprocess.x, out.timestamps)

    def test_130_grid_consistency(self):
        # The kept samples do not depend on how the data was trimmed:
        # downsampling a restricted aman lands on the same absolute grid.
        full = down_sample_aman(make_tod(), FACTOR)
        trimmed = make_tod()
        trimmed.restrict('samps', (OFFSET + 17, OFFSET + NSAMP - 23))
        tds = down_sample_aman(trimmed, FACTOR)
        common = np.intersect1d(full.timestamps, tds.timestamps)
        self.assertEqual(len(common), tds.samps.count)
        i0 = np.searchsorted(full.timestamps, tds.timestamps[0])
        np.testing.assert_array_equal(
            full.signal[:, i0:i0 + tds.samps.count], tds.signal)
        self.assertEqual(tds.timestamps[0], tds.samps.offset * FACTOR)

    # Flag types: pooled with 'any within block', never lost.

    def test_200_ranges(self):
        tod = make_tod()
        p_local = START + 3 * FACTOR + 4     # a dropped sample in block 3
        m = np.zeros(NSAMP, bool)
        m[p_local] = True
        tod.wrap('turnaround', Ranges.from_mask(m), [(0, 'samps')])
        rows = [Ranges.from_mask(m if d == 'det1' else np.zeros(NSAMP, bool))
                for d in DETS]
        tod.wrap('glitches', RangesMatrix(rows),
                 [(0, 'dets'), (1, 'samps')])
        out = down_sample_aman(tod, FACTOR)
        self.assertTrue(out.turnaround.mask()[3])
        self.assertEqual(out.turnaround.mask().sum(), 1)
        self.assertTrue(out.glitches[1].mask()[3])
        self.assertFalse(out.glitches[0].mask().any())

    def test_210_bool_ndarray(self):
        tod = make_tod()
        flags = np.zeros((len(DETS), NSAMP), bool)
        flags[1, START + 4 * FACTOR + 2] = True   # dropped sample, block 4
        tod.wrap('flags', flags, [(0, 'dets'), (1, 'samps')])
        out = down_sample_aman(tod, FACTOR)
        self.assertTrue(out.flags[1, 4])
        self.assertEqual(out.flags.sum(), 1)
        self.assertEqual(out.flags.dtype, np.dtype(bool))

    def test_220_sparse(self):
        # csr is flag-like: max-pooled within blocks (any, for bool)
        tod = make_tod()
        dense = np.zeros((len(DETS), NSAMP))
        dense[0, START + 2 * FACTOR] = 7.       # kept sample, block 2
        dense[2, START + 2 * FACTOR + 1] = 5.   # dropped sample, block 2
        dense[2, START + 2 * FACTOR + 3] = 4.   # same block: merged, max
        tod.wrap('sparse', csr_array(dense), [(0, 'dets'), (1, 'samps')])
        bmat = np.zeros((len(DETS), NSAMP), bool)
        bmat[1, START + 4 * FACTOR + 2] = True
        tod.wrap('sparse_flags', csr_array(bmat), [(0, 'dets'), (1, 'samps')])

        out = down_sample_aman(tod, FACTOR)
        blocks = np.append(IDX, NSAMP)
        expect = np.array([[row[a:b].max()
                            for a, b in zip(blocks[:-1], blocks[1:])]
                           for row in dense])
        np.testing.assert_array_equal(np.asarray(out.sparse.todense()),
                                      expect)
        self.assertEqual(out.sparse[2, 2], 5.)
        pooled = np.asarray(out.sparse_flags.todense())
        self.assertEqual(pooled.dtype, np.dtype(bool))
        self.assertTrue(pooled[1, 4])
        self.assertEqual(pooled.sum(), 1)

    # Methods.

    def test_300_mean(self):
        tod = make_tod()
        out = down_sample_aman(tod, FACTOR, method='mean')
        blocks = np.append(IDX, NSAMP)
        expect_ts = np.array([tod.timestamps[a:b].mean()
                              for a, b in zip(blocks[:-1], blocks[1:])])
        np.testing.assert_allclose(out.timestamps, expect_ts)
        expect_sig = np.array([[row[a:b].mean()
                                for a, b in zip(blocks[:-1], blocks[1:])]
                               for row in tod.signal])
        np.testing.assert_allclose(out.signal, expect_sig)

    def test_310_bad_method(self):
        with self.assertRaises(ValueError):
            down_sample_aman(make_tod(), FACTOR, method='fft')

    # Config helpers and archive validation.

    def test_400_cfg(self):
        with self.assertRaises(ValueError):
            parse_downsample_cfg({'factor': 1})
        with self.assertRaises(ValueError):
            parse_downsample_cfg({'factor': 10, 'methd': 'slice'})
        self.assertEqual(parse_downsample_cfg({'factor': 10}),
                         (10, 'slice'))

        rec = downsample_cfg_aman({'factor': 10})
        self.assertEqual(rec['factor'], 10)
        self.assertEqual(rec['phase'], 'absolute')

        pre = core.AxisManager()
        pre.wrap('downsample_cfg', rec)
        check_saved_downsample(pre, {'factor': 10})
        with self.assertRaises(ValueError):
            check_saved_downsample(pre, {'factor': 20})

    def test_410_cfg_samps(self):
        # the samps-axis consistency check catches grid mismatches
        tds = down_sample_aman(make_tod(), FACTOR)
        rec = downsample_cfg_aman({'factor': FACTOR})
        good = core.AxisManager(core.OffsetAxis(
            'samps', tds.samps.count, tds.samps.offset))
        good.wrap('downsample_cfg', rec)
        check_saved_downsample(good, {'factor': FACTOR}, samps=tds.samps)
        bad = core.AxisManager(core.OffsetAxis(
            'samps', tds.samps.count, tds.samps.offset + 1))
        bad.wrap('downsample_cfg', rec.copy())
        with self.assertRaises(ValueError):
            check_saved_downsample(bad, {'factor': FACTOR},
                                   samps=tds.samps)

    # Other axis situations.

    def test_500_other_axes(self):
        am = core.AxisManager(core.IndexAxis('samps', 100))
        am.wrap('y', np.arange(100.), [(0, 'samps')])
        out = down_sample_aman(am, FACTOR)
        self.assertEqual(out.samps.count, 10)
        np.testing.assert_array_equal(out.y, np.arange(0., 100, FACTOR))

        am = core.AxisManager(core.LabelAxis('dets', DETS))
        am.wrap('z', np.ones(len(DETS)), [(0, 'dets')])
        out = down_sample_aman(am, FACTOR)
        np.testing.assert_array_equal(out.z, am.z)


if __name__ == '__main__':
    unittest.main()
