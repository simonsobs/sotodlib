# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np
import pylab as pl

from numpy.testing import assert_array_equal, assert_allclose

from sotodlib import core, tod_ops
import so3g

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
        tod_ops.detrend_data(tod)
        tod_ops.detrend_data(tod, signal_name='sig1d')
        tod_ops.detrend_data(tod, signal_name='sig1e', axis_name='dets')
        tod_ops.detrend_data(tod, signal_name='sig3d')
        tod_ops.detrend_data(tod, signal_name='sig3d', axis_name='dets')
        with self.assertRaises(ValueError):
            tod_ops.detrend_data(tod, signal_name='sig1e')

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
        for filt in [
                tod_ops.filters.high_pass_butter4(fc),
                tod_ops.filters.high_pass_sine2(fc),
                tod_ops.filters.low_pass_butter4(fc),
                tod_ops.filters.low_pass_sine2(fc),
                tod_ops.filters.gaussian_filter(fc, f_sigma=f0 / 10),
                tod_ops.filters.gaussian_filter(0, f_sigma=f0 / 10),
        ]:
            f = np.fft.fftfreq(tod.samps.count) * f0
            y = filt(f, tod)
            sig_filt = tod_ops.fourier_filter(tod, filt)
            sigma1 = sig_filt.std(axis=1)
            self.assertTrue(np.all(sigma1 < sigma0))
            print(f'Filter takes sigma from {sigma0} to {sigma1}')

        # Check 1d
        sig1f = tod_ops.fourier_filter(tod, filt, signal_name='sig1d',
                                       detrend='linear')
        self.assertEqual(sig1f.shape, tod['sig1d'].shape)

class JumpfindTest(unittest.TestCase):
    def test_jumpfinder(self):
        """Test that jumpfinder finds jumps in white noise."""
        tod = get_tod('white')
        sig_jumps = 100*tod.signal[0]
        jump_locs = np.array([200, 400, 700])
        sig_jumps[jump_locs[0]:] += 150
        sig_jumps[jump_locs[1]:] -= 100
        sig_jumps[jump_locs[2]:] -= 200
        tod.wrap('sig_jumps', sig_jumps, [(0, 'samps')])

        jumps = tod_ops.jumpfind(tod, signal_name='sig_jumps')

        self.assertEqual(len(jump_locs), len(jumps))
        self.assertTrue(np.all(np.abs(jumps - jump_locs) < 20))

if __name__ == '__main__':
    unittest.main()
