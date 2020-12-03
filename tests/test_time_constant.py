# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast data loading.
"""
from glob import glob
import sys
import os

import numpy as np
from numpy.testing import (
    assert_equal, assert_array_almost_equal, assert_array_equal, assert_allclose,
)

from unittest import TestCase

from ._helpers import create_outdir

from sotodlib.sim_hardware import get_example, sim_telescope_detectors

from sotodlib.toast.time_constant import OpTimeConst


toast_available = None
toast_import_error = None
if toast_available is None:
    try:
        import toast
        from toast.mpi import MPI
        from toast.todmap import TODGround
        from toast.tod import AnalyticNoise
        toast_available = True
    except ImportError as e:
        toast_import_error = e
        toast_available = False


class TimeConstantTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print(
                "toast cannot be imported ({})- skipping unit test".format(toast_import_error),
                flush=True,
            )
            return

        self.outdir = None
        if MPI.COMM_WORLD.rank == 0:
            self.outdir = create_outdir(fixture_name)
        self.outdir = MPI.COMM_WORLD.bcast(self.outdir, root=0)

        toastcomm = toast.Comm()
        self.data = toast.Data(toastcomm)

        # Focalplane
        hwfull = get_example()
        dets = sim_telescope_detectors(hwfull, "SAT4")
        hwfull.data["detectors"] = dets
        hw = hwfull.select(
            match={"wafer_slot": "w42", "band": "f030", "pixel": "00[01]"})
        # print(hw.data["detectors"], flush=True)
        detquats = {k: v["quat"] for k, v in hw.data["detectors"].items()}

        # Samples per observation
        self.totsamp = 10000

        # Scan properties
        self.site_lon = '-67:47:10'
        self.site_lat = '-22:57:30'
        self.site_alt = 5200.
        self.coord = 'C'
        self.azmin = 45
        self.azmax = 55
        self.el = 60
        self.scanrate = 1.0
        self.scan_accel = 0.1
        self.CES_start = None

        # Noise properties
        self.rate = 100.0
        self.NET = 1e-3  # 1 mK NET
        self.epsilon = 0.0
        self.fmin = 1.0e-5
        self.alpha = 1.0
        self.fknee = 0.05

        for ob in range(3):
            ftime = (self.totsamp / self.rate) * ob + 1564015655.88
            tod = TODGround(
                self.data.comm.comm_group,
                detquats,
                self.totsamp,
                detranks=self.data.comm.group_size,
                firsttime=ftime,
                rate=self.rate,
                site_lon=self.site_lon,
                site_lat=self.site_lat,
                site_alt=self.site_alt,
                azmin=self.azmin,
                azmax=self.azmax,
                el=self.el,
                coord=self.coord,
                scanrate=self.scanrate,
                scan_accel=self.scan_accel,
                CES_start=self.CES_start)

            # Analytic noise model
            detnames = list(detquats.keys())
            drate = {x: self.rate for x in detnames}
            dfmin = {x: self.fmin for x in detnames}
            dfknee = {x: self.fknee for x in detnames}
            dalpha = {x: self.alpha for x in detnames}
            dnet = {x: self.NET for x in detnames}
            nse = AnalyticNoise(
                rate=drate,
                fmin=dfmin,
                detectors=detnames,
                fknee=dfknee,
                alpha=dalpha,
                NET=dnet
            )

            # Single observation
            obs = dict()
            obs["tod"] = tod
            obs["noise"] = nse
            obs["id"] = 12345
            obs["intervals"] = tod.subscans
            obs["site"] = "SimonsObs"
            obs["telescope"] = "SAT4"
            obs["site_id"] = 1
            obs["telescope_id"] = 4
            obs["fpradius"] = 5.0
            obs["start_time"] = ftime
            obs["altitude"] = self.site_alt
            obs["name"] = "test_{:02}".format(ob)

            # Add a focalplane dictionary with just the detector index
            focalplane = {}
            for idet, det in enumerate(detnames):
                focalplane[det] = {"index" : idet}
            obs["focalplane"] = focalplane

            # Add the observation to the dataset
            self.data.obs.append(obs)

        nse = toast.tod.OpSimNoise(out="signal", realization=0)
        nse.exec(self.data)

        return

    def test_convolve(self):
        if not toast_available:
            return

        tod = self.data.obs[0]["tod"]
        times = tod.local_times()
        ind = slice(times.size // 4, times.size // 4 * 3)
        det = tod.local_dets[0]

        initial = tod.local_signal(det).copy()

        tau = 1.0

        tauop = OpTimeConst(name="signal", inverse=False, tau=tau)

        tauop.exec(self.data)

        convolved = tod.local_signal(det).copy()

        tauop = OpTimeConst(name="signal", inverse=True, tau=tau)

        tauop.exec(self.data)

        deconvolved = tod.local_signal(det).copy()

        # Check that convolution reduces the noise RMS

        self.assertGreater(np.std(initial[ind]), 1e1 * np.std(convolved[ind]))

        # Check that de-convolved TOD is closer to the input

        self.assertLess(
            np.std((initial - deconvolved)[ind]),
            1e-1 * np.std((initial - convolved)[ind])
        )

        return

    def test_convolve_with_error(self):
        if not toast_available:
            return

        tod = self.data.obs[0]["tod"]
        times = tod.local_times()
        ind = slice(times.size // 4, times.size // 4 * 3)
        det = tod.local_dets[0]

        initial = tod.local_signal(det).copy()

        tau = 1.0

        tauop = OpTimeConst(name="signal", inverse=False, tau=tau)

        tauop.exec(self.data)

        convolved = tod.local_signal(det).copy()

        tauop = OpTimeConst(name="signal", inverse=True, tau=tau)

        tauop.exec(self.data)

        deconvolved = tod.local_signal(det).copy()

        tod.local_signal(det)[:] = convolved

        tauop = OpTimeConst(name="signal", inverse=True, tau=tau, tau_sigma=0.1)

        tauop.exec(self.data)

        deconvolved_with_error = tod.local_signal(det).copy()

        # Check that error-free de-convolved TOD is closer to the input

        self.assertLess(
            np.std((initial - deconvolved)[ind]),
            np.std((initial - deconvolved_with_error)[ind])
        )

        return
