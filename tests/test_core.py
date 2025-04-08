import unittest
import tempfile
import os

import numpy as np
import astropy.units as u
from sotodlib import core
import sotodlib.core.axisman_util as amutil
import so3g

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

class TestAxisManager(unittest.TestCase):

    # Basic behavior of each axis type.

    def test_100_index(self):
        a1 = np.zeros(100)
        a1[10] = 1.
        aman = core.AxisManager(core.IndexAxis('samps', len(a1)))
        aman.wrap('a1', a1, [(0, 'samps')])
        # Don't let people wrap the same field twice
        with self.assertRaises(ValueError):
            aman.wrap('a1', 2*a1, [(0, 'samps')])
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

        # Don't let people use non string labels
        dets_int = [0, 1, 2]
        with self.assertRaises(TypeError):
            aman = core.AxisManager(core.LabelAxis('dets', dets_int))

    def test_130_not_inplace(self):
        a1 = np.zeros(100)
        a1[10] = 1.
        aman = core.AxisManager(core.IndexAxis('samps', len(a1)))
        aman.wrap('a1', a1, [(0, 'samps')])
        aman.wrap('a2', 1)

        # This should return a separate thing.
        rman = aman.restrict('samps', (10, 30), in_place=False)
        # self.assertNotEqual(aman.a1[0], 0.)
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

        # Check that numpy int/float types are unpacked.
        aman.wrap('a', np.int32(12))
        aman.wrap('b', np.float32(12.))
        aman.wrap('c', np.str_('twelve'))
        self.assertNotIsInstance(aman['a'], np.integer)
        self.assertNotIsInstance(aman['b'], np.floating)
        self.assertNotIsInstance(aman['c'], np.str_)

        # Don't let people wrap the same scalar twice
        with self.assertRaises(ValueError):
            aman.wrap('x', 13)

        # Don't just let people wrap any old thing.
        with self.assertRaises(AttributeError):
            aman.wrap('a_dict', {'a': 123})
        with self.assertRaises(ValueError):
            aman.wrap('square_root', 1j)

        # Make sure AxisManager with scalar can be copied
        aman_copy = aman.copy()
        self.assertEqual(aman['x'], aman_copy['x'])

    def test_170_concat(self):
        # AxisManagers with shape (2, 100) and (3, 100)...
        detsA = ['det0', 'det1']
        detsB = ['det2', 'det3', 'det4']
        nsamps = 100
        amanA = core.AxisManager(core.LabelAxis('dets', detsA),
                                 core.OffsetAxis('samps', nsamps))
        amanB = core.AxisManager(core.LabelAxis('dets', detsB),
                                 core.OffsetAxis('samps', nsamps))

        # Empty AxisManagers ...
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')
        self.assertEqual(aman.dets.count, len(detsA) + len(detsB))

        # Concat arrays?
        amanA.wrap_new('signal', shape=('dets', 'samps'))
        amanB.wrap_new('signal', shape=('dets', 'samps'))
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')
        self.assertEqual(aman.signal.shape[0], len(detsA) + len(detsB))

        # Even if one is empty?
        amanX = amanA.restrict('dets', [])
        amanY = amanB.copy()
        aman = core.AxisManager.concatenate([amanX, amanY])
        self.assertEqual(aman.signal.shape[0], amanY.signal.shape[0])

        # or both are empty?
        amanY = amanB.restrict('dets', [])
        aman = core.AxisManager.concatenate([amanX, amanY])
        self.assertEqual(aman.signal.shape[0], 0)

        # or with sparse arrays?
        amanAA = core.AxisManager(core.LabelAxis('dets', detsA),
                                 core.OffsetAxis('samps', nsamps))
        amanBB = core.AxisManager(core.LabelAxis('dets', detsB),
                                 core.OffsetAxis('samps', nsamps))
        amanAA.wrap('sparse', csr_array( ((8,3), ([0,1], [1,21])), 
                                      shape=(amanAA.dets.count, amanAA.samps.count)),
                   [(0,'dets'),(1,'samps')])
        amanBB.wrap('sparse', csr_array( ((8,3), ([0,1], [2,54])), 
                                      shape=(amanBB.dets.count, amanBB.samps.count)),
                    [(0,'dets'),(1,'samps')])
        aman = core.AxisManager.concatenate([amanAA, amanBB])
        self.assertEqual(aman.dets.count, len(detsA) + len(detsB))

        # Handling of arrays++ that do not share the axis?
        for t in [amanA, amanB]:
            # Vectors, with nan.
            t.wrap_new('azimuth', shape=('samps',))[:] = 1.
            t.azimuth[3] = float('nan')

            # Scalars
            t.wrap("ans", 42)
            t.wrap("nans", float('nan'))

            # Include strings, because array_equal is picky on that...
            t.wrap('tersely', 'yo')
            t.wrap_new('greetings', shape=('samps',), dtype='S4')[:] = b'hi'
            t.wrap_new('free_greetings', shape=(4, ), dtype='U')[:] = 'hello'

            # And RangesMatrix
            t.wrap('rmat', so3g.proj.RangesMatrix.ones((5, 4)))


        amanB_backup, amanB = amanB, None

        # ... other_fields="exact"
        amanB = amanB_backup.copy()
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')

        # ... other_fields="exact"
        amanB = amanB_backup.copy()
        amanB.azimuth[:] = 2.
        with self.assertRaises(ValueError):
            aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')

        # RangesMatrix fail?
        amanB = amanB_backup.copy()
        amanB.rmat = so3g.proj.RangesMatrix.zeros(amanA.rmat.shape)
        with self.assertRaises(ValueError):
            aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')

        # ... other_fields="exact" and arrays of different shapes
        amanB = amanB_backup.copy()
        amanB.move("azimuth", None)
        amanB.wrap("azimuth", np.array([43,5,2,3]))
        with self.assertRaises(ValueError):
            aman = core.AxisManager.concatenate([amanA, amanB], axis='dets')

        # ... other_fields="fail"
        amanB = amanB_backup.copy()
        amanB.move("azimuth",None)
        amanB.wrap_new('azimuth', shape=('samps',))[:] = 2.
        with self.assertRaises(ValueError):
            aman = core.AxisManager.concatenate([amanA, amanB], axis='dets',
                                               other_fields='fail')
        amanB.azimuth[:] = 1.
        with self.assertRaises(ValueError):
            aman = core.AxisManager.concatenate([amanA, amanB], axis='dets',
                                               other_fields='fail')

        # ... other_fields="drop"
        amanB = amanB_backup.copy()
        amanB.azimuth[:] = 2.
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets',
                                            other_fields="drop")
        self.assertNotIn('azimuth', aman)

        # ... other_fields="first"
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets',
                                            other_fields='first')
        self.assertIn('azimuth', aman)
        self.assertSequenceEqual(aman._assignments['azimuth'], ['samps'])
        self.assertEqual(aman.azimuth[0], 1.)

        # Handling of child AxisManagers
        fpA = core.AxisManager(amanA.dets.copy())
        fpB = core.AxisManager(amanB.dets.copy())
        fpA.wrap_new('x', shape=('dets',))
        fpB.wrap_new('x', shape=('dets',))
        amanA.wrap('fp', fpA)
        amanB.wrap('fp', fpB)
        aman = core.AxisManager.concatenate([amanA, amanB], axis='dets',
                                            other_fields='drop')
        self.assertSequenceEqual(aman.fp.shape, (aman.dets.count, ))

        # Loop checking
        a = amanA.copy()
        b = amanB.copy()
        a.wrap('b', b)
        with self.assertRaises(AssertionError):
            a.wrap('a', a)
        with self.assertRaises(AssertionError):
            a.b.wrap('a', a)
        # This is allowed because a.b is not the same as b.  Maybe it
        # should be... but that will be a deliberate API change.
        b.wrap('a', a)
        self.assertNotIn('a', a.b)

    def test_180_overwrite(self):
        dets = ['det0', 'det1', 'det2']
        a1 = np.zeros((len(dets), 100))
        a1[1, 10] = 1.
        aman = core.AxisManager(core.LabelAxis('dets', dets),
                                core.OffsetAxis('samps', a1.shape[1]))
        aman.wrap('a1', a1, [(0, 'dets'), (1, 'samps')])
        a2 = np.zeros((len(dets), 100))
        a2[2, 11] = 1.
        aman.wrap('a1', a2, [(0, 'dets'), (1, 'samps')],
                  overwrite=True)
        self.assertNotEqual(aman.a1[2,11], 0)
        self.assertNotEqual(aman.a1[1,10], 1.)

    def test_190_get_set(self):
        dets = ["det0", "det1", "det2"]
        n, ofs = 1000, 0
        aman = core.AxisManager(
            core.LabelAxis("dets", dets), core.OffsetAxis("samps", n, ofs)
        )
        child = core.AxisManager(
            core.LabelAxis("dets", dets + ["det3"]),
            core.OffsetAxis("samps", n, ofs - n // 2),
        )

        child2 = core.AxisManager(
            core.LabelAxis("dets2", ["det4", "det5"]),
            core.OffsetAxis("samps", n, ofs - n // 2),
        )
        child2.wrap("tod", np.zeros((2, 1000)))
        aman.wrap("child", child)
        aman["child"].wrap("child2", child2)
        self.assertEqual(aman["child.child2.dets2"].count, 2)
        self.assertEqual(aman["child.dets"].name, "dets")
        np.testing.assert_array_equal(
            aman["child.child2.dets2"].vals, np.array(["det4", "det5"])
        )
        self.assertEqual(aman["child.child2.samps"].count, n // 2)
        self.assertEqual(aman["child.child2.samps"].offset, 0)
        self.assertEqual(
            aman["child.child2.samps"].count, aman.child.child2.samps.count
        )
        self.assertEqual(
            aman["child.child2.samps"].offset, aman.child.child2.samps.offset
        )

        np.testing.assert_array_equal(aman["child.child2.tod"], np.zeros((2, 1000)))

        with self.assertRaises(KeyError):
            aman["child2"]

        with self.assertRaises(AttributeError):
            aman["child.dets.an_extra_layer"]

        self.assertIn("child.dets", aman)
        self.assertIn("child.dets2", aman)  # I am not sure why this is true
        self.assertNotIn("child.child2.someentry", aman)
        self.assertNotIn("child.child2.someentry.someotherentry", aman)

        with self.assertRaises(ValueError):
            aman["child"] = child2

        new_tods = np.ones((2, 500))
        aman.child.child2.tod = new_tods
        np.testing.assert_array_equal(aman["child.child2.tod"], np.ones((2, 500)))
        np.testing.assert_array_equal(aman.child.child2.tod, np.ones((2, 500)))

        new_tods = np.ones((2, 1500))
        aman["child.child2.tod"] = new_tods
        np.testing.assert_array_equal(aman["child.child2.tod"], np.ones((2, 1500)))
        np.testing.assert_array_equal(aman.child.child2.tod, np.ones((2, 1500)))

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
        # See that aman.child and child are different...
        self.assertEqual(aman.child.dets.count, 3)
        self.assertEqual(child.dets.count, 4)

        # But you can tell it it's ok.
        child2 = core.AxisManager(
            core.LabelAxis('dets', dets + ['det3']),
            core.OffsetAxis('samps', n, ofs - n//2))
        aman.wrap('child2', child2, restrict_in_place=True)
        self.assertEqual(aman.child2.dets.count, 3)
        self.assertEqual(child2.dets.count, 3)

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

    def test_403_restrict_slices(self):
        # Test in_place is false for slices of arrays
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        aman = core.AxisManager(
                core.LabelAxis('dets', dets),
                core.OffsetAxis('samps', n, ofs)
        )
        aman.wrap_new('test', ('dets','samps'))
        rman = aman.restrict('samps', (0,500), in_place=False)
        rman.test += 5
        self.assertNotEqual( aman.test[0,0], rman.test[0,0])

    def test_410_merge(self):
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        for in_place in [False, True]:
            aman = core.AxisManager(
                core.LabelAxis('dets', dets),
                core.OffsetAxis('samps', n, ofs))
            coparent = core.AxisManager(
                core.LabelAxis('dets', dets + ['det3']),
                core.OffsetAxis('samps', n, ofs - n//2))\
                .wrap('x', np.arange(n), [(0, 'samps')])
            if in_place:
                kw = {'restrict_in_place': True}
            else:
                kw = {}
            aman.merge(coparent, **kw)
            self.assertEqual(aman.shape, (3, n//2))
            self.assertEqual(aman._axes['samps'].offset, ofs)
            self.assertEqual(aman.x[0], n//2)

            if in_place:
                # Check coparent was modified
                self.assertEqual(coparent.shape, aman.shape)
            else:
                # Check coparent was not modified in place
                self.assertEqual(coparent.shape, (4, n))


    def test_500_io(self):
        # Test save/load HDF5
        dets = ['det0', 'det1', 'det2']
        n, ofs = 1000, 0
        aman = core.AxisManager(
            core.LabelAxis('dets', dets),
            core.OffsetAxis('samps', n, ofs),
            core.IndexAxis('indexaxis', 12))
        # Make sure this has axes, scalars, a string array ...
        aman.wrap_new('test1', ('dets', 'samps'), dtype='float32')
        aman.wrap_new('flag1', shape=('dets', 'samps'),
                      cls=so3g.proj.RangesMatrix.zeros)
        aman.wrap('scalar', 8)
        aman.wrap('test_str', np.array(['a', 'b', 'cd']))
        aman.wrap('flags', core.FlagManager.for_tod(aman, 'dets', 'samps'))

        aman.wrap('a', np.int32(12))
        aman.wrap('b', np.float32(12.))
        aman.wrap('c', np.str_('twelve'))
        aman.wrap('d', np.bool_(False))

        aman.wrap('rangesint32', so3g.proj.Ranges.from_array(
                  np.array([[0, 1], [10, 20]]).astype(np.int32), aman.samps.count))
        aman.wrap('sparse', csr_array( ((8,3), ([0,1], [1,54])), 
                                      shape=(aman.dets.count, aman.samps.count)))

        aman.wrap('quantity', np.ones(5) << u.m)
        aman.wrap('quantity2', (np.ones(1) << u.m)[0])

        aman.wrap('subaman', core.AxisManager(
            aman.dets, core.LabelAxis('bands', ['f090', 'f150'])))
        aman['subaman'].wrap_new('subtest1', shape=[('dets', 'bands')])
        aman['subaman'].wrap_new('subtest2', shape=(100,))

        # Make sure the saving / clobbering / readback logic works
        # equally for simple group name, root group, None->root group.
        for dataset in ['path/to/my_axisman', '/', None]:
            with tempfile.TemporaryDirectory() as tempdir:
                filename = os.path.join(tempdir, 'test.h5')
                aman.save(filename, dataset)
                # Overwrite
                aman.save(filename, dataset, overwrite=True)
                # Compress and Overwrite
                aman.save(filename, dataset, overwrite=True, compression='gzip')
                # Refuse to overwrite
                with self.assertRaises(RuntimeError):
                    aman.save(filename, dataset)
                # Read back.
                aman2 = aman.load(filename, dataset)
            # Check what was read back.
            # This is not a very satisfying comparison ... support for ==
            # should be required for all AxisManager members!
            for k in aman._fields.keys():
                self.assertEqual(aman[k].__class__, aman2[k].__class__)
                if hasattr(aman[k], 'shape'):
                    self.assertEqual(aman[k].shape, aman2[k].shape)
                else:
                    self.assertEqual(aman[k], aman2[k])  # scalar

        # Test field subset load
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, 'test.h5')
            aman.save(filename, dataset)
            aman2 = aman.load(filename, dataset, fields=['test1', 'flags'])
            aman3 = aman.load(filename, dataset, fields=['test1', 'subaman'])
            aman4 = aman.load(filename, dataset, fields=['test1', 'subaman.subtest1'])
        for target, keys, subkeys in [
                (aman2, ['test1', 'flags'], None),
                (aman3, ['test1', 'subaman'], ['subtest1', 'subtest2']),
                (aman4, ['test1', 'subaman'], ['subtest1']),
        ]:
            self.assertCountEqual(target._fields.keys(),
                                  keys)
            if subkeys:
                self.assertCountEqual(target['subaman']._fields.keys(),
                                      subkeys)

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


class TestUtil(unittest.TestCase):
    def test_coindices(self):
        x = ['d', 'a', 'g', 'e', 'b']
        y = ['x', 'b', 'o', 'd', 'q']
        z, i0, i1 = core.util.get_coindices(x, y)
        the_answers = (['d', 'b'], [0, 4], [3, 1])
        self.assertEqual(list(z),  the_answers[0])
        self.assertEqual(list(i0), the_answers[1])
        self.assertEqual(list(i1), the_answers[2])

    def test_multi_index(self):
        x = ['x', 'y', 'c', 'a']
        y = ['a', 'a', 'b', 'c', 'a', 'u']
        ix = core.util.get_multi_index(x, y)
        the_answer = np.array([x.index(_y) if _y in x else -1
                               for _y in y])
        self.assertEqual(list(ix), list(the_answer))


class TestAxisManagerUtil(unittest.TestCase):
    def test_restrict_times(self):
        am = core.AxisManager()
        nsamps = 100
        am.wrap('timestamps', np.arange(nsamps),
                [(0, core.OffsetAxis('samps', nsamps))])
        am2 = amutil.restrict_to_times(am, 10, 20)
        self.assertEqual(len(am2.timestamps),10)

    def test_restrict_times_raises(self):
        am = core.AxisManager()
        nsamps = 100
        am.wrap('timestamps', np.arange(nsamps),
                [(0, core.OffsetAxis('samps', nsamps))])
        with self.assertRaises(amutil.RestrictionException):
            amutil.restrict_to_times(am, 200, 300)

    def test_dict_to_am(self):
        d = {
            'hi': 10,
            'nested': {
                'abcd': np.array([0, 1]),
                'test': 'answer',
            }
        }
        am = amutil.dict_to_am(d)
        self.assertEqual(am.nested.test, 'answer')


if __name__ == '__main__':
    unittest.main()
