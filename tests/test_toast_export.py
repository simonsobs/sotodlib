# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast data export.
"""

import os

from unittest import TestCase

from ._helpers import create_outdir

from sotodlib.hardware.config import get_example

from sotodlib.hardware.sim import sim_telescope_detectors

from sotodlib.data.toast import ToastExport

from spt3g import core as core3g


toast_available = None
if toast_available is None:
    try:
        import toast
        from toast.mpi import MPI
        from toast.tod import TODGround
        from toast.tod import AnalyticNoise
        toast_available = True
    except ImportError:
        toast_available = False


class ToastExportTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(fixture_name)
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        toastcomm = toast.Comm()
        self.data = toast.Data(toastcomm)

        # Focalplane
        hwfull = get_example()
        dets = sim_telescope_detectors(hwfull, "SAT3")
        hwfull.data["detectors"] = dets
        hw = hwfull.select(
            match={"wafer": "42", "band": "LF1", "pixel": "000"})
        print(hw.data["detectors"], flush=True)
        detquats = {k: v["quat"] for k, v in hw.data["detectors"].items()}

        # Samples per observation
        self.totsamp = 10000

        # Pixelization
        nside = 512
        self.sim_nside = nside
        self.map_nside = nside

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
        self.NET = 5.0
        self.epsilon = 0.0
        self.fmin = 1.0e-5
        self.alpha = 1.0
        self.fknee = 0.05

        # Create a ground based TOD
        tod = TODGround(
            self.data.comm.comm_group,
            detquats,
            self.totsamp,
            detranks=self.data.comm.group_size,
            firsttime=0.0,
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
        obs["telescope"] = "SAT3"
        obs["site_id"] = 1
        obs["telescope_id"] = 4
        obs["fpradius"] = 5.0
        obs["start_time"] = 0
        obs["altitude"] = self.site_alt
        obs["name"] = "test"

        # Add the observation to the dataset
        self.data.obs.append(obs)
        return

    def test_dump(self):
        if not toast_available:
            return

        # Simulate some noise into multiple cache prefixes.  This is used
        # to test the export of multiple timestream flavors.
        nse = toast.tod.OpSimNoise(out="signal", realization=0)
        nse.exec(self.data)
        nse = toast.tod.OpSimNoise(out="component1", realization=0)
        nse.exec(self.data)
        nse = toast.tod.OpSimNoise(out="component2", realization=0)
        nse.exec(self.data)

        tod = self.data.obs[0]["tod"]

        # Dump to disk
        dumper = ToastExport(
            self.outdir,
            prefix="sat3",
            use_intervals=True,
            cache_name="signal",
            mask_flag_common=tod.TURNAROUND,
            filesize=500000,
            units=core3g.G3TimestreamUnits.Tcmb)
        dumper.exec(self.data)

        # Inspect the dumped frames
        for root, dirs, files in os.walk(self.outdir):
            files = sorted(files)
            for f in files:
                path = os.path.join(root, f)
                print("file {}".format(path), flush=True)
                gf = core3g.G3File(path)
                for frm in gf:
                    print(frm, flush=True)
                    # if frm.type == core3g.G3FrameType.Scan:
                    #     common = frm.get("flags_common")
                    #     print(common.array(), flush=True)
                    #     print(frm.get("flags"))
                    #     print(frm.get("signal"), flush=True)

        return
