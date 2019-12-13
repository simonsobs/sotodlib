import unittest

import numpy as np
from sotodlib import core


class TestAxisManager(unittest.TestCase):

    # Basic behavior of each axis type.

    def test_100_index(self):
        a1 = np.zeros(100)
        a1[10] = 1.
        aman = core.AxisManager(core.IndexAxis('samps', len(a1)))
        aman.wrap('a1', a1, [(0, 'samps')])
        aman.restrict('samps', (10, 30))
        self.assertNotEqual(aman.a1[0], 0.)
        self.assertEqual(len(aman.a1), 20)

    def test_110_offset(self):
        a1 = np.zeros(100)
        # Place the mark at index 10, and offset 15 -- so the mark is
        # at absolute index 25.
        a1[10] = 1.
        aman = core.AxisManager(
            core.OffsetAxis('samps', len(a1), 15))
        aman.wrap('a1', a1, [(0, 'samps')])
        aman.restrict('samps', (25, 30))
        self.assertNotEqual(aman.a1[0], 0.)
        self.assertEqual(len(aman.a1), 5)

    def test_120_label(self):
        dets = ['det0', 'det1', 'det2']
        a1 = np.zeros(len(dets))
        a1[1] = 1.
        aman = core.AxisManager(core.LabelAxis('dets', dets))
        aman.wrap('a1', a1, [(0, 'dets')])
        aman.restrict('dets', ['det1'])
        self.assertNotEqual(aman.a1[0], 0.)

    # Multi-dimensional restrictions.

    def test_200_multid(self):
        dets = ['det0', 'det1', 'det2']
        a1 = np.zeros((len(dets), len(dets)))
        a1[2, 2] = 1.
        aman = core.AxisManager(core.LabelAxis('dets', dets))
        aman.wrap('a1', a1, [(0, 'dets'), (1, 'dets')])
        aman.restrict('dets', ['det1', 'det2'])
        self.assertEqual(aman.a1.shape, (2, 2))
        self.assertNotEqual(aman.a1[1, 1], 0.)

    def test_300_restrict(self):
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 5000
        aman = core.AxisManager(
            core.LabelAxis('dets', dets),
            core.OffsetAxis('samps', n, ofs))
        # Super-correlation matrix.
        a1 = np.zeros((len(dets), len(dets), n, n))
        a1[1, 1, 20, 21] = 1.
        aman.wrap('a1', a1, [(0, 'dets'), (1, 'dets'),
                             (2, 'samps'), (3, 'samps')])
        aman.restrict('dets', ['det1']).restrict('samps', (20 + ofs, 30 + ofs))
        self.assertEqual(aman.shape, (1, 10))
        self.assertEqual(aman.a1.shape, (1, 1, 10, 10))
        self.assertNotEqual(aman.a1[0, 0, 0, 1], 0.)

    # wrap of AxisManager, merge.

    def test_400_child(self):
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        aman = core.AxisManager(
            core.LabelAxis('dets', dets),
            core.OffsetAxis('samps', n, ofs))
        child = core.AxisManager(
            core.LabelAxis('dets', dets + ['det3']),
            core.OffsetAxis('samps', n, ofs - n//2))
        aman.wrap('child', child)
        self.assertEqual(aman.shape, (3, n//2))
        self.assertEqual(aman._axes['samps'].offset, ofs)

    def test_410_merge(self):
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        aman = core.AxisManager(
            core.LabelAxis('dets', dets),
            core.OffsetAxis('samps', n, ofs))
        coparent = core.AxisManager(
            core.LabelAxis('dets', dets + ['det3']),
            core.OffsetAxis('samps', n, ofs - n//2))\
            .wrap('x', np.arange(n), [(0, 'samps')])
        aman.merge(coparent)
        self.assertEqual(aman.shape, (3, n//2))
        self.assertEqual(aman._axes['samps'].offset, ofs)
        self.assertEqual(aman.x[0], n//2)

    def test_900_everything(self):
        tod = core.AxisManager(
            core.LabelAxis('dets', list('abcdef')),
            core.OffsetAxis('samps', 1000))
        cal = core.AxisManager(
            core.LabelAxis('dets', list('feghij')))
        cuts = core.AxisManager(
            core.OffsetAxis('samps', 800, 100))
        tod.wrap('data', np.ones(tod.shape, 'float32'), )
        cal.wrap('cal', np.linspace(.9, 1.2, 6), [(0, 'dets')])
        cuts.wrap('cuts', np.ones(cuts.shape, 'int32'), [(0, 'samps')])
        tod.merge(cal, cuts)
        self.assertEqual(tod.shape, (2, 800))


if __name__ == '__main__':
    unittest.main()
