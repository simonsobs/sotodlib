import unittest

import numpy as np
from sotodlib import core
import so3g


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

    def test_130_not_inplace(self):
        a1 = np.zeros(100)
        a1[10] = 1.
        aman = core.AxisManager(core.IndexAxis('samps', len(a1)))
        aman.wrap('a1', a1, [(0, 'samps')])
        aman.wrap('a2', 1)

        # This should return a separate thing.
        rman = aman.restrict('samps', (10, 30), in_place=False)
        #self.assertNotEqual(aman.a1[0], 0.)
        self.assertEqual(len(aman.a1), 100)
        self.assertEqual(len(rman.a1), 20)
        self.assertNotEqual(aman.a1[10], 0.)
        self.assertNotEqual(rman.a1[0], 0.)

    def test_140_restrict_axes(self):
        dets = ['det0', 'det1', 'det2']
        a1 = np.zeros((len(dets), 100))
        a1[1, 10] = 1.
        aman = core.AxisManager(core.LabelAxis('dets', dets),
                                core.OffsetAxis('samps', a1.shape[1]))
        aman.wrap('a1', a1, [(0, 'dets'), (1, 'samps')])
        aman.wrap('a2', 1)

        r_axes = {'dets': core.LabelAxis('dets', dets[1:2]),
                  'samps': core.OffsetAxis('samps', 20, 10)}
        # Not-in-place...
        rman = aman.restrict_axes(r_axes, in_place=False)
        self.assertEqual(aman.a1.shape, (3, 100))
        self.assertEqual(rman.a1.shape, (1, 20))
        self.assertNotEqual(aman.a1[1, 10], 0.)
        self.assertNotEqual(rman.a1[0, 0], 0.)
        # In-place.
        aman.restrict_axes(r_axes, in_place=True)
        self.assertEqual(aman.a1.shape, (1, 20))
        self.assertNotEqual(aman.a1[0, 0], 0.)

    def test_150_wrap_new(self):
        dets = ['det0', 'det1', 'det2']
        a1 = np.zeros((len(dets), 100))
        a1[1, 10] = 1.
        aman = core.AxisManager(core.LabelAxis('dets', dets),
                                core.OffsetAxis('samps', a1.shape[1]))
        x = aman.wrap_new('x', shape=('dets', 'samps'))
        y = aman.wrap_new('y', shape=('dets', 'samps'), dtype='float32')
        self.assertEqual(aman.x.shape, aman.y.shape)
        if hasattr(so3g.proj.RangesMatrix, 'zeros'):
            # Jan 2021 -- some so3g might not have this method yet...
            f = aman.wrap_new('f', shape=('dets', 'samps'),
                              cls=so3g.proj.RangesMatrix.zeros)
            self.assertEqual(aman.x.shape, aman.f.shape)

    def test_160_scalars(self):
        aman = core.AxisManager(core.LabelAxis('dets', ['a', 'b']),
                                core.OffsetAxis('samps', 100))

        # Accept trivially promoted scalars
        aman.wrap('x', 12)
        aman.wrap('z', 'hello')

        # Don't just let people wrap any old thing.
        with self.assertRaises(ValueError):
            aman.wrap('a_dict', {'a': 123})

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

    def test_401_restrict(self):
        # Test AxisManager.restrict when it has AxisManager members.
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        for in_place in [True, False]:
            aman = core.AxisManager(
                core.LabelAxis('dets', dets),
                core.OffsetAxis('samps', n, ofs))
            child = core.AxisManager(aman.dets, aman.samps)
            child2 = core.AxisManager(
                core.LabelAxis('not_dets', ['x', 'y', 'z']))
            aman.wrap('child', child)
            aman.wrap('rebel_child', child2)
            aout = aman.restrict('dets', ['det1'], in_place=in_place)
            msg = f'Note restrict was with in_place={in_place}'
            self.assertTrue(aout is aman or not in_place, msg=msg)
            self.assertEqual(aout['child'].shape, (1, n), msg=msg)
            self.assertIn('rebel_child', aout, msg=msg)
            self.assertEqual(aout['rebel_child'].shape, (3,), msg=msg)

    def test_402_restrict_axes(self):
        # Test AxisManager.restrict_axes when it has AxisManager members.
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        for in_place in [True, False]:
            aman = core.AxisManager(
                core.LabelAxis('dets', dets),
                core.OffsetAxis('samps', n, ofs))
            child = core.AxisManager(aman.dets, aman.samps)
            child2 = core.AxisManager(
                core.LabelAxis('not_dets', ['x', 'y', 'z']))
            aman.wrap('child', child)
            aman.wrap('rebel_child', child2)
            new_dets = core.LabelAxis('dets', ['det1'])
            aout = aman.restrict_axes([new_dets], in_place=in_place)
            msg = f'Note restrict was with in_place={in_place}'
            self.assertTrue(aout is aman or not in_place, msg=msg)
            self.assertEqual(aout['child'].shape, (1, n), msg=msg)
            self.assertIn('rebel_child', aout, msg=msg)
            self.assertEqual(aout['rebel_child'].shape, (3,), msg=msg)

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

class TestFlagManager(unittest.TestCase):
    
    def test_100_inheritance(self):
        tod = core.AxisManager(
            core.LabelAxis('dets', list('abcdef')),
            core.OffsetAxis('samps', 1000))
        flags = core.FlagManager.for_tod(tod, 'dets', 'samps')
        tod.wrap('flags', flags)
        self.assertTrue( type(tod.flags) == core.FlagManager )

if __name__ == '__main__':
    unittest.main()
