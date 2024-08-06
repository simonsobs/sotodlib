# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np
import pylab as pl
import scipy.signal

from sotodlib import core, tod_ops
from sotodlib.preprocess.pcore import _expand

from numpy.testing import assert_array_equal


from ._helpers import mpi_multi
from .test_tod_ops import get_tod

from so3g.proj import Ranges, RangesMatrix
from scipy.sparse import csr_array

class TestExpand(unittest.TestCase):

    def test_100_expand(self):
        aman = get_tod(sig_type='trendy',ndets=5, nsamps=1000)
        full = core.AxisManager( aman.dets, aman.samps)
        proc_aman = core.AxisManager( aman.dets, aman.samps)
        
        # wrap a dets array with floats
        proc_aman.wrap('arr1', np.mean( aman.signal, axis=1), [(0,'dets')],) 
        # wrap a dets array with ints
        proc_aman.wrap(
            'arr2', np.round( np.min(aman.signal, axis=1)).astype('int32'),
            [(0,'dets')],
        ) 
        # wrap a dets,samps array to make sure it works 
        proc_aman.wrap('arr3', aman.signal*23, [(0,'dets'), (1,'samps')],) 

        # wrap a RangesMatrix
        proc_aman.wrap(
            'flag1', RangesMatrix.zeros( (aman.dets.count, aman.samps.count)),
            [(0,'dets'), (1,'samps')],
        ) 
        proc_aman.flag1[0].add_interval( 100,700 )
        proc_aman.flag1[3].add_interval( 110,800 )
        proc_aman.flag1[4].add_interval( 0,200 )

        # wrap a Ranges
        proc_aman.wrap('flag2', Ranges( aman.samps.count ),[(0,'samps')],) 
        proc_aman.flag2.add_interval( 450,700 )

        # wrap a csr_array
        proc_aman.wrap_new('sparse_thing', shape=('dets', 'samps'))[:] = \
            1.* (np.random.uniform(size=(aman.dets.count, aman.samps.count)) > .05)
        proc_aman.wrap('csr_thing', csr_array(proc_aman['sparse_thing']),
            [(0,'dets'), (1,'samps')])

        ## test same size expansion
        out = _expand( proc_aman, full)
        assert_array_equal( out.flag1[0].ranges()[0] , [100,700] )
        assert_array_equal( out.sparse_thing, out.csr_thing.toarray() )

        ## test with detectors cut
        aman.restrict( 'dets', aman.dets.vals[3:5])
        proc_aman.restrict( 'dets', aman.dets.vals)
        out = _expand( proc_aman, full)
        self.assertTrue( proc_aman.dets.vals[0] == out.dets.vals[3])
        assert_array_equal( out.flag1[3].ranges()[0] , [110,800] )
        assert_array_equal( out.sparse_thing, out.csr_thing.toarray() )

        aman.restrict( 'samps', (300,None))
        proc_aman.restrict( 'samps', (300,None))
        out = _expand( proc_aman, full)
        assert_array_equal( proc_aman.flag1[0].ranges()[0] , [0,500] )
        assert_array_equal( out.flag1[3].ranges()[0] , [300,800] )
        assert_array_equal( out.sparse_thing, out.csr_thing.toarray() )

        ## test with a wrapped axis manager
        dummy = core.AxisManager(
            aman.dets, 
        )
        dummy.wrap( 'blah', np.median( aman.signal, axis=1),
            [(0,'dets')],)
        proc_aman.wrap('dummy', dummy)
        out = _expand( proc_aman, full)
        
        ## check value mask
        self.assertTrue( len(out.valid[0].ranges()) == 0 )
        assert_array_equal( out.valid[3].ranges()[0], [300,1000] )

        ## test with extra axis
        pain = core.AxisManager(
            aman.dets, aman.samps, core.LabelAxis('ouch', ['x','y']),
        )
        pain.wrap_new('wammy', ('ouch', 'dets', 'samps'))
        proc_aman.wrap('pain', pain)
        out = _expand( proc_aman, full)

        ## test with no detectors
        proc_aman.restrict('dets', [])
        out = _expand( proc_aman, full)
        assert_array_equal( out.sparse_thing, out.csr_thing.toarray() )



if __name__ == '__main__':
    unittest.main()
