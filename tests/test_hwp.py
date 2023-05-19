# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np

from sotodlib import core
from sotodlib.hwp import hwp

INPUT_COEFFS = np.asarray(
                [[14.4086668, 10.57294344, 11.18964606, 12.27585103,
                 11.27456133, 11.7841095, 12.87044498, 13.25598597,
                 12.56675035, 12.06382477, 14.72865531, 12.62258284,
                 11.43530194, 11.92264568, 14.71371932, 11.8379756],
                [10.32748604, 11.82000625, 14.96949369, 14.18303644,
                 13.45385166, 12.81443907, 13.65328508, 13.78361639,
                 11.3964662, 13.06204199, 10.35769728, 10.20522442,
                 14.79385459, 11.24546604, 10.27690037, 10.04351162]])


def make_fake_tod(ts=np.arange(0, 60, 1/200), modes=np.arange(8)+1,
                  mult_fact=12.5*np.ones(8), ndets=2, input_coeffs=None):
    hwp_angle = (2*np.pi*2*ts) % (2*np.pi)
    fake_signal = np.zeros((ndets, len(hwp_angle)))
    if input_coeffs is None:
        input_coeffs = np.zeros((ndets, 2*len(mult_fact)))
    for nd in range(ndets):
        for i, md in enumerate(modes):
            if input_coeffs is None:
                input_coeffs[nd, 2*i] = mult_fact[i]*np.random.rand()
                input_coeffs[nd, 2*i+1] = mult_fact[i]*np.random.rand()
            fake_signal[nd] += input_coeffs[nd, 2*i]*np.sin(md*hwp_angle)
            fake_signal[nd] += input_coeffs[nd, 2*i + 1]*np.cos(md*hwp_angle)

    dets = ['det%i' % i for i in range(ndets)]
    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                           core.OffsetAxis('samps', count=len(hwp_angle)),
                           core.LabelAxis('modes', vals=mode_names))
    tod.wrap('timestamps', ts, axis_map=[(0, 'samps')])
    tod.wrap('hwp_angle', hwp_angle, axis_map=[(0, 'samps')])
    tod.wrap('signal', np.atleast_2d(fake_signal),
             axis_map=[(0, 'dets'), (1, 'samps')])
    tod.wrap('input_coeffs', np.atleast_2d(input_coeffs),
             axis_map=[(0, 'dets'), (1, 'modes')])
    return tod

def get_coeff_metric(tod):
    nm = tod.modes.count//2
    nd = tod.dets.count
    outmetric_num = np.zeros((nd, nm))
    outmetric_denom = np.zeros((nd, nm))
    for i in range(nm):
        outmetric_num[:, i] = (tod.hwpss_stats.coeffs[:, 2*i] - \
                               tod.input_coeffs[: , 2*i])**2
        outmetric_num[:, i] += (tod.hwpss_stats.coeffs[:, 2*i+1] - \
                                tod.input_coeffs[: , 2*i+1])**2
        outmetric_denom[:, i] = (tod.input_coeffs[:, 2*i])**2
        outmetric_denom[:, i] += (tod.input_coeffs[:, 2*i+1])**2
    return np.max(100*(outmetric_num/outmetric_denom))



class HwpssTest(unittest.TestCase):
    "Test the HWPSS fitting functions"
    def test_linregbin(self):
        lr, bn = True, True
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn,
                           bins=200)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)

    def test_linregnobin(self):
        lr, bn = True, False
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)

    def test_fitbin(self):
        lr, bn = False, True
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn, bins=200)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)
        # Checks that returned covariance matrix from fit is finite.
        # When not using wrapper function in lambda function we get
        # infinite covariance matrix returned. So this is a warning
        # to not change that even though it looks wrong.
        self.assertFalse(False in np.isfinite(tod.hwpss_stats.covars))

    def test_fitnobin(self):
        lr, bn = False, False
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        with self.assertRaises(ValueError):
            _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn, modes=[2, 4])

if __name__ == '__main__':
    unittest.main()
