# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check sss routines.

"""

import unittest
import numpy as np
from numpy.polynomial import legendre as L

from sotodlib import core
from sotodlib.tod_ops import sss


def get_scan(n_scans=33, scan_accel=0.025, scanrate=0.025,
             az0=2, az1=2.25, fs=200):
    """
    Returns timestamps and az positions for a CES scan.
    """
    t_scan = (az1-az0)/scanrate
    t_turnaround = 2*scanrate/scan_accel

    t_tot = np.asarray([])
    d_tot = np.asarray([])

    t_scan = np.arange(0, t_scan+1/fs, 1/fs)
    t_turn = np.arange(0, t_turnaround+1/fs, 1/fs)
    ds_right = az0 + t_scan*scanrate
    dt_right = az1 + t_turn * scanrate - 0.5 * scan_accel * t_turn ** 2
    ds_left = az1 - t_scan*scanrate
    dt_left = az0 - t_turn * scanrate + 0.5 * scan_accel * t_turn ** 2

    for i in range(n_scans):
        # right scan
        if i == 0:
            t_tot = np.concatenate((t_tot, t_scan))
        else:
            t_tot = np.concatenate((t_tot, t_tot[-1]+t_scan))
        d_tot = np.concatenate((d_tot, ds_right))
        # right turnaround
        t_tot = np.concatenate((t_tot, t_tot[-1]+t_turn))
        d_tot = np.concatenate((d_tot, dt_right))
        # left scan
        t_tot = np.concatenate((t_tot, t_tot[-1]+t_scan))
        d_tot = np.concatenate((d_tot, ds_left))
        # right turnaround
        t_tot = np.concatenate((t_tot, t_tot[-1]+t_turn))
        d_tot = np.concatenate((d_tot, dt_left))
    return t_tot, d_tot


def make_fake_sss_tod(nmodes=20, noise_amp=1, n_scans=10,
                      ndets=2, input_coeffs=None):
    """
    Makes an axis manager with azimuth synchronous signal
    in it, populated via legendre polynomials plus gaussian noise.
    """
    ts, azpoint = get_scan(n_scans=n_scans)
    az_min, az_max = np.min(azpoint), np.max(azpoint)
    x = ( 2*azpoint - (az_min+az_max) ) / (az_max - az_min)

    fake_signal = np.zeros((ndets, len(ts)))
    if input_coeffs is None:
        input_coeffs = np.random.uniform(-10, 11, size=(ndets, nmodes+1))
    for nd in range(ndets):
        fake_signal[nd] += L.legval(x, input_coeffs[nd])
        noise = np.random.normal(0, noise_amp, size=len(ts))
        fake_signal[nd] += noise

    dets = ['det%i' % i for i in range(ndets)]
    mode_names = []
    for mode in range(nmodes+1):
        mode_names.append(f'legendre{mode}')

    tod_fake = core.AxisManager(core.LabelAxis('dets', vals=dets),
                                core.OffsetAxis('samps', count=len(ts)),
                                core.LabelAxis('modes', vals=mode_names))
    point = core.AxisManager(core.OffsetAxis('samps', count=len(ts)))
    tod_fake.wrap('boresight', point)
    tod_fake.boresight.wrap('az', azpoint, axis_map=[(0, 'samps')])
    tod_fake.wrap('timestamps', ts, axis_map=[(0, 'samps')])
    tod_fake.wrap('signal', np.atleast_2d(fake_signal),
                  axis_map=[(0, 'dets'), (1, 'samps')])
    tod_fake.wrap('input_coeffs', np.atleast_2d(input_coeffs),
                  axis_map=[(0, 'dets'), (1, 'modes')])
    return tod_fake


def get_coeff_metric(tod):
    """
    Evaluates fit is working by comparing coefficients in to out.
    """
    print(tod.input_coeffs[0])
    print(tod.sss_stats.coeffs[0])
    outmetric_num = (tod.sss_stats.coeffs - tod.input_coeffs)**2
    outmetric_denom = (tod.input_coeffs)**2
    return np.max(100*(outmetric_num/outmetric_denom))


class SssTest(unittest.TestCase):
    "Test the SSS fitting functions"
    def test_fit(self):
        tod = make_fake_sss_tod(noise_amp=0)
        sss_stats, model_sig_tod = sss.get_sss(tod, method='fit', nmodes=20, range=None, bins=10000)
        ommax = get_coeff_metric(tod)
        print(ommax)
        self.assertTrue(ommax < 1.0)

if __name__ == '__main__':
    unittest.main()
