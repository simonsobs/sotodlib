""" Test psd calculation
"""


import unittest
import numpy as np
from numpy.fft import rfftfreq, irfft

from sotodlib import core
from sotodlib.tod_ops import detrend_tod
from sotodlib.tod_ops.flags import get_turnaround_flags
from sotodlib.tod_ops.fft_ops import (
    calc_psd, calc_wn, fit_noise_model, noise_model)

from .test_azss import get_scan

TOL_BIAS = 0.005


class PSDTest(unittest.TestCase):
    def test_psd_fit(self):
        fs = 200.
        dets = core.LabelAxis('dets', [f'det{di:003}' for di in range(20)])
        nsamps = 200 * 3600

        aman = core.AxisManager(dets)
        ndets = aman.dets.count

        white_noise_amp_input = 50 + np.random.randn(ndets)  # W/sqrt{Hz}
        fknee_input = 1 + 0.1 * np.random.randn(ndets)
        alpha_input = 3 + 0.2 * np.random.randn(ndets)

        freqs = rfftfreq(nsamps, d=1/fs)
        params = [white_noise_amp_input[:, np.newaxis],
                  fknee_input[:, np.newaxis],
                  alpha_input[:, np.newaxis]]
        pxx_input = noise_model(freqs, params)

        pxx_input[:, 0] = 0

        T = nsamps/fs
        ft_amps = np.sqrt(pxx_input * T * fs**2 / 2)

        ft_phases = np.random.uniform(0, 2 * np.pi, size=ft_amps.shape)
        ft_coefs = ft_amps * np.exp(1.0j * ft_phases)
        realized_noise = irfft(ft_coefs)
        timestamps = 1700000000 + np.arange(0, realized_noise.shape[1])/fs
        aman.add_axis(core.OffsetAxis('samps', len(timestamps)))
        aman.wrap('timestamps', timestamps, [(0, 'samps')])
        aman.wrap('signal', realized_noise, [(0, 'dets'), (1, 'samps')])

        detrend_tod(aman)
        freqs_output, Pxx_output = calc_psd(aman, nperseg=200*100)
        fit_result = fit_noise_model(aman, wn_est=50, fknee_est=1.0,
                                     alpha_est=3.3, lowf=0.05,
                                     f_max=5, binning=True,
                                     psdargs={'nperseg': 200*1000})
        wnl_fit = fit_result.fit[:, 0]
        fk_fit = fit_result.fit[:, 1]
        alpha_fit = fit_result.fit[:, 2]

        self.assertTrue(np.abs(np.median(white_noise_amp_input - wnl_fit)) < 1)
        self.assertTrue(np.abs(np.median(fknee_input - fk_fit)) < 0.1)
        self.assertTrue(np.abs(np.median(alpha_input - alpha_fit)) < 0.1)

    def test_wn_debias(self):
        # prep
        timestamps, az = get_scan(
            n_scans=20, scan_accel=0.25, scanrate=0.5, az0=0, az1=40)

        nsamps = len(timestamps)
        ndets = 100
        np.random.seed(0)
        signal = np.random.normal(0, 1, size=(ndets, nsamps))

        dets = [f"det{i}" for i in range(ndets)]
        aman = core.AxisManager(
                core.LabelAxis("dets", dets),
                core.IndexAxis("samps", nsamps)
        )
        aman.wrap("timestamps", timestamps, [(0, "samps")])
        aman.wrap("signal", signal, [(0, "dets"), (1, "samps")])
        boresight = core.AxisManager(aman.samps)
        boresight.wrap("az", az, [(0, "samps")])
        aman.wrap('boresight', boresight)
        aman.wrap('flags', core.AxisManager(aman.dets, aman.samps))
        get_turnaround_flags(aman)

        # test default arguments, this is biased
        calc_psd(aman, merge=True, nperseg=2**18)
        wn = calc_wn(aman)
        ratio = np.average(wn) / np.sqrt(np.average(aman.Pxx))
        self.assertTrue(abs(ratio - 1) > TOL_BIAS)
        # test debias, full_output=True, noverlap=0
        freqs, Pxx, nseg = calc_psd(aman, merge=False, full_output=True,
                                    noverlap=0, nperseg=2**18)
        wn = calc_wn(aman, Pxx, freqs, nseg)
        ratio = np.average(wn) / np.sqrt(np.average(Pxx))
        self.assertAlmostEqual(ratio, 1, delta=TOL_BIAS)
        # test quarter nperseg
        freqs, Pxx, nseg = calc_psd(aman, merge=False, full_output=True,
                                    noverlap=0, nperseg=2**16)
        wn = calc_wn(aman, Pxx, freqs, nseg)
        ratio = np.average(wn) / np.sqrt(np.average(Pxx))
        self.assertAlmostEqual(ratio, 1, delta=TOL_BIAS)
        # test defulat nperseg
        freqs, Pxx, nseg = calc_psd(aman, merge=False, full_output=True,
                                    noverlap=0)
        wn = calc_wn(aman, Pxx, freqs, nseg)
        ratio = np.average(wn) / np.sqrt(np.average(Pxx))
        self.assertAlmostEqual(ratio, 1, delta=TOL_BIAS)
        # test subscan
        freqs, Pxx, nseg = calc_psd(aman, merge=False, full_output=True,
                                    noverlap=0, subscan=True)
        wn = calc_wn(aman, Pxx, freqs, nseg)
        ratio = np.average(wn) / np.sqrt(np.average(Pxx))
        self.assertAlmostEqual(ratio, 1, delta=TOL_BIAS)
