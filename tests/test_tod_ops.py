# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np
import pylab as pl
import scipy.signal

from numpy.testing import assert_array_equal, assert_allclose

from sotodlib import core, tod_ops, sim_flags
import so3g

from ._helpers import mpi_multi


SAMPLE_FREQ_HZ = 100.

def get_tod(sig_type='trendy', ndets=3, nsamps=1000):
    tod = core.AxisManager(
        core.LabelAxis('dets', ['det%i' % i for i in range(ndets)]),
        core.OffsetAxis('samps', nsamps)
    )
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


def get_glitchy_tod(ts, noise_amp=0, ndets=2, npoly=3, poly_coeffs=None):
    """Returns axis manager to test fill_glitches"""
    fake_signal = np.zeros((ndets, len(ts)))
    input_sig = np.zeros((ndets, len(ts)))
    if poly_coeffs is None:
        poly_coeffs = np.random.uniform(0.5, 1.6, npoly)*1e-1
    poly_sig = np.polyval(poly_coeffs, ts-np.mean(ts))
    for nd in range(ndets):
        input_sig[nd] = poly_sig
        fake_signal[nd] = poly_sig
        noise = np.random.normal(0, noise_amp, size=len(ts))
        fake_signal[nd] += noise

    dets = ['det%i' % i for i in range(ndets)]

    tod_fake = core.AxisManager(core.LabelAxis('dets', vals=dets),
                                core.OffsetAxis('samps', count=len(ts)))
    tod_fake.wrap('timestamps', ts, axis_map=[(0, 'samps')])
    tod_fake.wrap('signal', np.atleast_2d(fake_signal),
                  axis_map=[(0, 'dets'), (1, 'samps')])
    tod_fake.wrap('inputsignal', np.atleast_2d(input_sig),
                  axis_map=[(0, 'dets'), (1, 'samps')])
    flgs = core.AxisManager()
    tod_fake.wrap('flags', flgs)
    params = {'n_glitches': 10, 'sig_n_glitch': 10, 'h_glitch': 10,
              'sig_h_glitch': 2}
    sim_flags.add_random_glitches(tod_fake, params=params, signal='signal',
                                  flag='glitches', overwrite=False)
    return tod_fake


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

    def test_pca(self):
        tod = get_tod('white')
        x = tod.timestamps / tod.timestamps[-1]
        comp0 = x**2 - x
        comp1 = x**3 - .2
        comp0, comp1 = [c / np.std(c) for c in [comp0, comp1]]
        tod.signal += comp0 * 100
        tod.signal[1] += comp1 * 50
        mod = tod_ops.pca.get_pca_model(tod, n_modes=2)
        tod_ops.pca.add_model(tod, mod, scale=-1)
        assert (tod.signal.std(axis=1) < 1.).all()

        # With a glitch
        tod = get_tod('white')
        tod.signal += comp0 * 100
        tod.signal[1][10] = 1e6
        mask = np.ones(tod.dets.count, bool)
        mask[1] = False

        # The glitch dominates this PCA, and white noise is not recovered.
        pca = tod_ops.pca.get_pca(tod)
        mod = tod_ops.pca.get_pca_model(tod, pca=pca, n_modes=1)
        sig1 = tod_ops.pca.add_model(tod, mod, scale=-1, signal=tod.signal.copy())
        assert (sig1[mask].std(axis=1) > 10.).any()

        # Excluding det with glitch successfully cleans other dets.
        pca = tod_ops.pca.get_pca(tod, mask=mask)
        mod = tod_ops.pca.get_pca_model(tod, pca=pca, n_modes=1)
        sig2 = tod_ops.pca.add_model(tod, mod, scale=-1, signal=tod.signal.copy())
        assert (sig2[mask].std(axis=1) < 1.).all()

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

    def test_detrend_inplace(self):
        tod = get_tod('trendy')
        sig_id = id(tod.signal)

        for method in ["linear", "mean", "median"]:
            tod_ops.detrend_tod(tod, method=method, in_place=True)
            self.assertEqual(sig_id, id(tod.signal))

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
    
    def test_fillglitches(self):
        """Tests fill glitches wrapper function"""
        ts = np.arange(0, 1*60, 1/200)
        aman = get_glitchy_tod(ts, ndets=100)
        # test poly fill
        up, mg = False, False
        glitch_filled = tod_ops.gapfill.fill_glitches(aman, use_pca=up,
                                                      wrap=mg)
        self.assertTrue(np.max(np.abs(glitch_filled-aman.inputsignal)) < 1e-3)

        # test pca fill
        up, mg = True, False
        glitch_filled = tod_ops.gapfill.fill_glitches(aman, use_pca=up,
                                                      wrap=mg)
        print(np.max(np.abs(glitch_filled-aman.inputsignal)))

        # test wrap new field
        up, mg = False, True
        glitch_filled = tod_ops.gapfill.fill_glitches(aman, use_pca=up,
                                                      wrap=mg)
        self.assertTrue('gap_filled' in aman._assignments)

class FilterTest(unittest.TestCase):
    def test_basic(self):
        """Test that fourier filters reduce RMS of white noise."""
        tod = get_tod('white')
        tod.wrap('sig1d', tod.signal[0], [(0, 'samps')])
        sigma0 = tod.signal.std(axis=1)
        f0 = SAMPLE_FREQ_HZ
        fc = f0 / 4

        def wrap_iir(N, wn, fs=f0):
            b, a = scipy.signal.butter(N, wn, fs=fs)
            iir_params = core.AxisManager()
            iir_params.wrap('a', a)
            iir_params.wrap('b', b)
            iir_params.wrap('fscale', 1)
            return iir_params

        # A simple IIR filter
        iir_params = wrap_iir(4, fc)

        # Per-wafer IIR filter params (ok if uniform)
        iir_params_multi = core.AxisManager()
        iir_params_multi.wrap('wafer1', wrap_iir(4, fc))
        iir_params_multi.wrap('wafer2', wrap_iir(4, fc))

        for filt in [
                tod_ops.filters.high_pass_butter4(fc),
                tod_ops.filters.high_pass_sine2(fc),
                tod_ops.filters.low_pass_butter4(fc),
                tod_ops.filters.low_pass_sine2(fc),
                tod_ops.filters.gaussian_filter(fc, f_sigma=f0 / 10),
                tod_ops.filters.gaussian_filter(0, f_sigma=f0 / 10),
                tod_ops.filters.iir_filter(iir_params=iir_params),
                tod_ops.filters.iir_filter(iir_params=iir_params_multi),
                tod_ops.filters.iir_filter(
                    a=iir_params.a, b=iir_params.b, fscale=iir_params.fscale),
                tod_ops.filters.iir_filter(
                    iir_params=dict(iir_params._fields.items())),
                tod_ops.filters.identity_filter(),
        ]:
            f = np.fft.fftfreq(tod.samps.count) * f0
            y = filt(f, tod)
            sig_filt = tod_ops.fourier_filter(tod, filt)
            sigma1 = sig_filt.std(axis=1)
            print(f'Filter takes sigma from {sigma0} to {sigma1}')
            if not isinstance(filt, tod_ops.filters.identity_filter):
                self.assertTrue(np.all(sigma1 < sigma0))

        # Confirm fail if not uniform per-wafer
        iir_params_multi.wrap('wafer3', wrap_iir(6, fc))
        filt = tod_ops.filters.iir_filter(iir_params=iir_params_multi)
        with self.assertRaises(ValueError):
            y = filt(f, tod)

        # Check 1d
        filt = tod_ops.filters.high_pass_butter4(fc)
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

        # Find jumps without filtering
        jumps_nf, _ = tod_ops.jumps.find_jumps(tod, signal=tod.sig_jumps, min_size=5)
        jumps_nf = jumps_nf.ranges().flatten()
        
        # Find jumps with TV filtering
        jumps_tv, _ = tod_ops.jumps.find_jumps(tod, signal=tod.sig_jumps, tv_weight=.5, min_size=5)
        jumps_tv = jumps_tv.ranges().flatten()

        # Find jumps with gaussian filtering
        jumps_gauss, _ = tod_ops.jumps.find_jumps(tod, signal=tod.sig_jumps, gaussian_width=.5, min_size=5)
        jumps_gauss = jumps_gauss.ranges().flatten()

        # Remove double counted jumps and round to remove uncertainty
        jumps_nf = np.unique(np.round(jumps_nf, -2))
        jumps_tv = np.unique(np.round(jumps_tv, -2))
        jumps_gauss = np.unique(np.round(jumps_gauss, -2))

        # Check that all methods agree
        self.assertEqual(len(jumps_tv), len(jumps_gauss))
        self.assertTrue(np.all(np.abs(jumps_tv - jumps_gauss) == 0))
        self.assertEqual(len(jumps_nf), len(jumps_gauss))
        self.assertTrue(np.all(np.abs(jumps_nf - jumps_gauss) == 0))

        # Check that they agree with the input
        self.assertEqual(len(jump_locs), len(jumps_nf))
        self.assertTrue(np.all(np.abs(jumps_nf - jump_locs) == 0))

        # Check height
        jumps_msk = np.zeros_like(sig_jumps, dtype=bool)
        jumps_msk[jumps_nf] = True
        heights = tod_ops.jumps.estimate_heights(sig_jumps, jumps_msk)
        heights = heights[heights.nonzero()].ravel()
        self.assertTrue(np.all(np.abs(np.array([10, -13, -8]) - np.round(heights)) < 3))


class FFTTest(unittest.TestCase):
    def test_psd(self):
        tod = get_tod("white")
        f, Pxx = tod_ops.fft_ops.calc_psd(tod, nperseg=256)
        self.assertEqual(len(f), 129) # nperseg/2 + 1
        f, Pxx = tod_ops.fft_ops.calc_psd(tod, freq_spacing=.1)
        self.assertEqual(np.round(np.median(np.diff(f)), 1), .1)

if __name__ == '__main__':
    unittest.main()
