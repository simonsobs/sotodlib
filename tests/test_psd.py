""" Test psd calculation
"""


import unittest
import numpy as np

from sotodlib import core
from sotodlib.tod_ops.flags import get_turnaround_flags
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn

from test_azss import get_scan


class PSDTest(unittest.TestCase):
    def test_wn_debias(self):
        # prepare
        timestamps, az = get_scan(
            n_scans=20, scan_accel=0.25, scanrate=0.5, az0=0, az1=40)

        nsamps = len(timestamps)
        ndets = 100
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

        # test
        calc_psd(aman, full_output=True, merge=True, subscan=True)
        calc_wn(aman)
