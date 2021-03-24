# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np
import pylab as pl

from sotodlib import core, tod_ops

SAMPLE_FREQ_HZ = 100.

def get_tod(sig_type='trendy'):
    tod = core.AxisManager(core.LabelAxis('dets', ['a', 'b', 'c']),
                           core.IndexAxis('samps', 1000))
    tod.wrap_new('signal', ('dets', 'samps'), dtype='float32')
    tod.wrap_new('timestamps', ('samps',))[:] = (
        np.arange(tod.samps.count) / SAMPLE_FREQ_HZ)
    if sig_type == 'trendy':
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

if __name__ == '__main__':
    unittest.main()
