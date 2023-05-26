# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np
import pylab as pl
import scipy.signal

from numpy.testing import assert_array_equal, assert_allclose

from sotodlib import core, tod_ops
import so3g

from ._helpers import mpi_multi


SAMPLE_FREQ_HZ = 100.

def get_tod(sig_type='trendy'):
    tod = core.AxisManager(core.LabelAxis('dets', ['a', 'b', 'c']),
                           core.IndexAxis('samps', 1000))
    tod.wrap_new('signal', ('dets', 'samps'), dtype='float32')
    tod.wrap_new('timestamps', ('samps',))[:] = (
        np.arange(tod.samps.count) / SAMPLE_FREQ_HZ)
    if sig_type == 'zero':
        pass
    elif sig_type == 'trendy':
        x = np.linspace(0, 1., tod.samps.count)
        tod.signal[:] = [(i+1) + (i+1)**2 * x for i in range(tod.dets.count)]
    elif sig_type == 'white':
        tod.signal = np.random.normal(size=tod.shape)
    elif sig_type == 'red':
        tod.signal = np.random.normal(size=tod.shape)
        tod.signal[:] = np.cumsum(tod.signal, axis=1)
    else:
        raise RuntimeError(f'sig_type={sig_type}?')
    return tod

class FactorsTest(unittest.TestCase):
    def test_inf(self):
        f = tod_ops.fft_ops.find_inferior_integer
        self.assertEqual(f(257), 256)
        self.assertEqual(f(28), 28)
        self.assertEqual(f(2**2 * 7**8 + 1), 2**2 * 7**8)

    def test_sup(self):
        f = tod_ops.fft_ops.find_superior_integer
        self.assertEqual(f(255), 256)
        self.assertEqual(f(28), 28)
        self.assertEqual(f(2**2 * 7**8 - 1), 2**2 * 7**8)

class PcaTest(unittest.TestCase):
    """Test the pca module."""
    def test_basic(self):
        tod = get_tod('trendy')
        amps0 = tod.signal.max(axis=1) - tod.signal.min(axis=1)
        modes = tod_ops.pca.get_trends(tod, remove=True)
        amps1 = tod.signal.max(axis=1) - tod.signal.min(axis=1)
        print(f'Amplitudes from {amps0} to {amps1}.')
        self.assertTrue(np.all(amps1 < amps0 * 1e-6))

    def test_detrend(self):
        tod = get_tod('trendy')
        tod.wrap('sig1d', tod.signal[0], [(0, 'samps')])
        tod.wrap('sig1e', tod.signal[:,0], [(0, 'dets')])
        tod.wrap('sig3d', tod.signal[None], [(1, 'dets'), (2, 'samps')])
        tod_ops.detrend_tod(tod, signal_name='signal', in_place=False, wrap_name='detrended')
        self.assertTrue( np.all(np.mean(tod.signal,axis=-1) != np.mean(tod.detrended, axis=-1)))

        dx = tod_ops.detrend_tod(tod, signal_name='signal', in_place=False, wrap_name=None)
        self.assertTrue( np.all(np.mean(tod.detrended,axis=-1) == np.mean(dx,axis=-1)))
        
        tod_ops.detrend_tod(tod)
        tod_ops.detrend_tod(tod, signal_name='sig1d')
        tod_ops.detrend_tod(tod, signal_name='sig1e', axis_name='dets')
        tod_ops.detrend_tod(tod, signal_name='sig3d')
        tod_ops.detrend_tod(tod, signal_name='sig3d', axis_name='dets')
        with self.assertRaises(ValueError):
            tod_ops.detrend_tod(tod, signal_name='sig1e')

class GapFillTest(unittest.TestCase):
    def test_basic(self):
        """Test linear fill on simple linear data."""
        tod = get_tod('zero')
        # Latest so3g has RangesMatrix.zeros ... use that next time
        flags = so3g.proj.RangesMatrix([so3g.proj.Ranges(tod.samps.count)
                                        for i in range(tod.dets.count)])
        # For detector 1, identify a short gap; make a linear signal
        # that we know exacly how to gap-fill.
        flags[1].add_interval(10, 20)
        gap_mask = flags[1].mask()
        sig = np.linspace(100., 156., tod.signal.shape[1])
        sentinel = 1.7  # Place-holder for bad samples
        atol = 1e-5  # Note float32 dynamic range!

        # Note _method=None should become 'fast' if accelerated
        # routine is available...
        for order in [1,2]:
            for _method in ['slow', None]:
                # Setup signal.
                tod.signal[1] = sig * ~gap_mask
                tod.signal[1][gap_mask] = sentinel

                # Fill with model, inplace.  (Return origisamples in ex.)
                ex = tod_ops.get_gap_fill(tod, flags=flags, swap=True, _method=_method)
                assert_allclose(sig[gap_mask], tod.signal[1][gap_mask], atol=atol)
                # ... check "extraction" has bad samples
                assert_allclose(ex[1].data, sentinel, atol=atol)

                # Reset signal.
                tod.signal[1] = sig * ~gap_mask
                tod.signal[1][gap_mask] = sentinel

                # Compute fill samples and return in ex, leaving tod unmodified.
                ex = tod_ops.get_gap_fill(tod, flags=flags, order=order,
                                          swap=False, _method=_method)
                assert_allclose(tod.signal[1][gap_mask], sentinel)
                # ... check "extraction" has model values.
                assert_allclose(ex[1].data, sig[gap_mask], atol=atol)

class FilterTest(unittest.TestCase):
    def test_basic(self):
        """Test that fourier filters reduce RMS of white noise."""
        tod = get_tod('white')
        tod.wrap('sig1d', tod.signal[0], [(0, 'samps')])
        sigma0 = tod.signal.std(axis=1)
        f0 = SAMPLE_FREQ_HZ
        fc = f0 / 4

        # A simple IIR filter
        b, a = scipy.signal.butter(4, fc, fs=f0)
        iir_params = core.AxisManager()
        iir_params.wrap('a', a)
        iir_params.wrap('b', b)
        iir_params.wrap('fscale', 1)

        for filt in [
                tod_ops.filters.high_pass_butter4(fc),
                tod_ops.filters.high_pass_sine2(fc),
                tod_ops.filters.low_pass_butter4(fc),
                tod_ops.filters.low_pass_sine2(fc),
                tod_ops.filters.gaussian_filter(fc, f_sigma=f0 / 10),
                tod_ops.filters.gaussian_filter(0, f_sigma=f0 / 10),
                tod_ops.filters.iir_filter(iir_params=iir_params),
                tod_ops.filters.iir_filter(
                    a=iir_params.a, b=iir_params.b, fscale=iir_params.fscale),
                tod_ops.filters.identity_filter(),
        ]:
            f = np.fft.fftfreq(tod.samps.count) * f0
            y = filt(f, tod)
            sig_filt = tod_ops.fourier_filter(tod, filt)
            sigma1 = sig_filt.std(axis=1)
            print(f'Filter takes sigma from {sigma0} to {sigma1}')
            if not isinstance(filt, tod_ops.filters.identity_filter):
                self.assertTrue(np.all(sigma1 < sigma0))

        # Check 1d
        sig1f = tod_ops.fourier_filter(tod, filt, signal_name='sig1d',
                                       detrend='linear')
        self.assertEqual(sig1f.shape, tod['sig1d'].shape)


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class JumpfindTest(unittest.TestCase):
    def test_jumpfinder(self):
        """Test that jumpfinder finds jumps in white noise."""
        np.random.seed(0)
        tod = get_tod('white')
        sig_jumps = tod.signal[0]
        jump_locs = np.array([200, 400, 700])
        sig_jumps[jump_locs[0]:] += 10
        sig_jumps[jump_locs[1]:] -= 13
        sig_jumps[jump_locs[2]:] -= 8

        tod.wrap('sig_jumps', sig_jumps, [(0, 'samps')])

        # Find jumps with TV filtering
        jumps_tv = tod_ops.jumps.find_jumps(tod, signal=tod.sig_jumps, min_size=5)
        jumps_tv = jumps_tv.ranges().flatten()

        # Find jumps with gaussian filtering
        jumps_gauss = tod_ops.jumps.find_jumps(tod, signal=tod.sig_jumps,
                                               jumpfinder=tod_ops.jumps.jumpfinder_gaussian, min_size=5)
        jumps_gauss = jumps_gauss.ranges().flatten()

        # Remove double counted jumps and round to remove uncertainty
        jumps_tv = np.unique(np.round(jumps_tv, -2))
        jumps_gauss = np.unique(np.round(jumps_gauss, -2))

        # Check that both methods agree
        self.assertEqual(len(jumps_tv), len(jumps_gauss))
        self.assertTrue(np.all(np.abs(jumps_tv - jumps_gauss) == 0))

        # Check that they agree with the input
        self.assertEqual(len(jump_locs), len(jumps_tv))
        self.assertTrue(np.all(np.abs(jumps_tv - jump_locs) == 0))

if __name__ == '__main__':
    unittest.main()
