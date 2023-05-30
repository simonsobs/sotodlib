import unittest
from sotodlib.core import metadata

import os
import time
import numpy as np

from ._helpers import mpi_multi

# Global Announcement: I know, but I hate slow tests.
example = None


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestDetDb(unittest.TestCase):
    def setUp(self):
        global example
        if example is None:
            print('Creating example database...')
            example = metadata.get_example('DetDb')

    def test_smoke(self):
        """Basic functionality."""
        db = example.copy()

        print('Test 1: how many dets have array_code=HF1?')
        X = db.dets(props={'base.array_code': 'HF1'})
        print('  Answer: %i' % len(X))
        print('  The first few are:')
        for x in X[:5]:
            print('    ' + str(x))
        print()

        print('Test 2: Get (array, wafer) for a bunch of dets.')
        u2 = db.props(X, props=['base.array_code', 'wafer_code'])
        pairs = list(u2.distinct())
        print('  Distinct pairs:')
        for p in pairs:
            print('    ' + str(p))
        print()
        assert(len(pairs) == 3)

    def test_resultset(self):
        """Test that ResultSet objects have required behaviors.

        """
        db0 = example.copy()
        dets = db0.dets()
        props = db0.props(props=['base.array_code', 'base.wafer_code'])
        combos = props.distinct()

        assert isinstance(dets, metadata.ResultSet)
        assert isinstance(props, metadata.ResultSet)
        assert isinstance(combos, metadata.ResultSet)

        assert isinstance(combos[0], dict)
        assert isinstance(list(combos)[0], dict)

        # Test distinct coverage.
        n0 = len(db0.dets(props=combos))
        n1 = 0
        for c in combos:
            subd = db0.dets(props=c)
            n1 += len(subd)
        assert(n0 == n1)

        # Check operators...
        assert isinstance(combos[:2] + combos[2:], metadata.ResultSet)
        with self.assertRaises(TypeError):
            combos + [1, 2, 3]
        with self.assertRaises(ValueError):
            combos + dets

        # Check indexing
        # ... with int
        self.assertIsInstance(dets[1], dict)
        self.assertIsInstance(dets[int(1)], dict)
        self.assertIsInstance(dets[np.arange(3)[1]], dict)

        # ... with slice
        dets_subset = dets[1:3]
        self.assertIsInstance(dets_subset, metadata.ResultSet)
        self.assertEqual(len(dets_subset), 2)

        # ... Check that string index returns field
        names = dets['name']
        self.assertIsInstance(names, np.ndarray)
        self.assertEqual(len(names), len(dets))

    def test_io(self):
        """Check to_file and from_file."""

        db0 = example.copy()
        dump_list = [(f'test.sqlite', None),
                     (f'test.txt', 'dump'),
                     (f'test.gz', None)]
        # Save.
        for fn, fmt in dump_list:
            print(f'Writing {fn}')
            db0.to_file(fn, fmt=fmt)
            print('  -- output has size {}'.format(os.path.getsize(fn)))
            t0 = time.time()
            db1 = metadata.DetDb.from_file(fn, fmt=fmt)
            dt = time.time() - t0
            print('  -- read-back {} rows in {} seconds.'.format(
                len(db1.dets()), dt))
            print('  -- removing.')
            os.remove(fn)


if __name__ == '__main__':
    unittest.main()
