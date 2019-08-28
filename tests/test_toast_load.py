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

from sotodlib.hardware.config import get_example

from sotodlib.hardware.sim import sim_telescope_detectors

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g


toast_available = None
if toast_available is None:
    try:
        import toast
        from toast.mpi import MPI
        from toast.tod import TODGround
        from toast.tod import AnalyticNoise
        from sotodlib.data.toast_export import ToastExport
        from sotodlib.data.toast_load import load_data
        toast_available = True
    except ImportError:
        toast_available = False


class ToastLoadTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        self.outdir = None
        if MPI.COMM_WORLD.rank == 0:
            self.outdir = create_outdir(fixture_name)
        self.outdir = MPI.COMM_WORLD.bcast(self.outdir, root=0)

        toastcomm = toast.Comm()
        self.data = toast.Data(toastcomm)

        # Focalplane
        hwfull = get_example()
        dets = sim_telescope_detectors(hwfull, "SAT3")
        hwfull.data["detectors"] = dets
        hw = hwfull.select(
            match={"wafer": "42", "band": "LF1", "pixel": "00[01]"})
        # print(hw.data["detectors"], flush=True)
        detquats = {k: v["quat"] for k, v in hw.data["detectors"].items()}

        # File dump size in bytes (1MB)
        self.dumpsize = 2 ** 20

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
            obs["telescope"] = "SAT3"
            obs["site_id"] = 1
            obs["telescope_id"] = 4
            obs["fpradius"] = 5.0
            obs["start_time"] = ftime
            obs["altitude"] = self.site_alt
            obs["name"] = "test_{:02}".format(ob)

            # Add the observation to the dataset
            self.data.obs.append(obs)

        # Simulate some noise into multiple cache prefixes.  This is used
        # to test the export of multiple timestream flavors.

        nse = toast.tod.OpSimNoise(out="signal", realization=0)
        nse.exec(self.data)
        nse = toast.tod.OpSimNoise(out="component1", realization=0)
        nse.exec(self.data)
        nse = toast.tod.OpSimNoise(out="component2", realization=0)
        nse.exec(self.data)


        return

    def test_load(self):
        if not toast_available:
            return

        tod = self.data.obs[0]["tod"]

        # Dump to disk
        prefix = "sat3"
        dumper = ToastExport(
            self.outdir,
            prefix=prefix,
            use_intervals=True,
            cache_name="signal",
            cache_copy=["component1", "component2"],
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize,
            units=core3g.G3TimestreamUnits.Tcmb,
            verbose=False,
        )
        dumper.exec(self.data)

        # Load the data back in

        checkdata = load_data(self.outdir, comm=self.data.comm, prefix=prefix)
        checkdata.info(sys.stdout)

        return


    def test_load_split(self):
        if not toast_available:
            return

        obs = self.data.obs[0]
        tod = obs["tod"]

        # Split the detectors into separate groups
        dets = sorted(tod.detectors)
        detgroups = {}
        for idet, det in enumerate(dets):
            detgroups["group{}".format(idet)] = [det]

        outdir = self.outdir
        prefix = "sat3"

        # Dump to disk
        dumper = ToastExport(
            outdir,
            prefix=prefix,
            use_intervals=True,
            cache_name="signal",
            cache_copy=["component1", "component2"],
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize,
            units=core3g.G3TimestreamUnits.Tcmb,
            detgroups=detgroups,
            verbose=False,
        )
        dumper.exec(self.data)

        # Count the number of resulting files
        files = glob("{}/{}/sat3_group*_00000000.g3".format(outdir, obs['name']))
        assert_equal(len(files), len(detgroups),
                     "Exported files ({}) does not match the detector "
                     "groups ({})".format(files, detgroups))

        # Load the data back in

        checkdata = load_data(outdir, comm=self.data.comm,
                              prefix="{}_{}".format(prefix, "group."),
                              all_flavors=True)

        # Verify that input and output are equal

        checktod = checkdata.obs[0]["tod"]

        times = tod.local_times()
        checktimes = checktod.local_times()
        assert_allclose(checktimes, times, rtol=1e-6, err_msg="Timestamps do not agree")

        cflags = ((tod.local_common_flags() & tod.TURNAROUND) != 0).astype(np.uint8)
        checkcflags = checktod.local_common_flags()
        assert_array_equal(checkcflags, cflags, err_msg="Common flags do not agree")

        for det in dets:
            sig0 = tod.local_signal(det)
            sig1 = tod.local_signal(det, "component1")
            sig2 = tod.local_signal(det, "component2")
            checksig0 = checktod.local_signal(det)
            checksig1 = checktod.local_signal(det, "component1")
            checksig2 = checktod.local_signal(det, "component2")
            assert_allclose(checksig0, sig0, rtol=1e-6,
                            err_msg="Signal0 does not agree")
            assert_allclose(checksig1, sig1, rtol=1e-6,
                            err_msg="Signal1 does not agree")
            assert_allclose(checksig2, sig2, rtol=1e-6,
                            err_msg="Signal2 does not agree")

            flg = tod.local_flags(det)
            checkflg = checktod.local_flags(det)
            assert_array_equal(checkflg, flg, err_msg="Flags do not agree")

        return


    def test_load_compressed(self):
        if not toast_available:
            return

        obs = self.data.obs[0]
        tod = obs["tod"]

        # We'll write the file with and without one detector
        # to measure the change in the TOD size
        dets = sorted(tod.detectors)
        detgroups = {'all_but_one' : []}
        for det in enumerate(dets[1:]):
            detgroups["all_but_one"].append(det)

        # uncompressed output

        outdir = self.outdir
        uncompressed_prefix = "sat3_uncompressed"

        dumper = ToastExport(
            outdir,
            prefix=uncompressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress=False,
            verbose=False,
        )
        dumper.exec(self.data)

        dumper = ToastExport(
            outdir,
            prefix=uncompressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress=False,
            detgroups=detgroups,
            verbose=False,
        )
        dumper.exec(self.data)

        # compressed output

        compressed_prefix = "sat3_compressed"

        dumper = ToastExport(
            outdir,
            prefix=compressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress=True,
            verbose=False,
        )
        dumper.exec(self.data)

        compressed_prefix = "sat3_compressed"

        dumper = ToastExport(
            outdir,
            prefix=compressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress=True,
            detgroups=detgroups,
            verbose=False,
        )
        dumper.exec(self.data)

        # Very high compression ratio

        very_compressed_prefix = "sat3_very_compressed"
        rmstarget = 2 ** 8

        dumper = ToastExport(
            outdir,
            prefix=very_compressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress={"rmstarget" : rmstarget},
            verbose=False,
        )
        dumper.exec(self.data)

        dumper = ToastExport(
            outdir,
            prefix=very_compressed_prefix,
            use_intervals=True,
            mask_flag_common=tod.TURNAROUND,
            filesize=self.dumpsize * 100,
            units=core3g.G3TimestreamUnits.Tcmb,
            compress={"rmstarget" : rmstarget},
            detgroups=detgroups,
            verbose=False,
        )
        dumper.exec(self.data)

        # Check that the timestreams do shrink in size

        sizes = {}
        for prefix in ["uncompressed", "uncompressed_all_but_one",
                       "compressed", "compressed_all_but_one",
                       "very_compressed", "very_compressed_all_but_one",
        ]:
            fnames =  glob("{}/{}/sat3_{}_0*.g3"
                           "".format(outdir, obs["name"], prefix))
            sizes[prefix] = 0
            for fname in fnames:
                size = os.path.getsize(fname)
                sizes[prefix] += size

        # These are the sizes of individual timestreams
        uncompressed_size = sizes["uncompressed"] - sizes["uncompressed_all_but_one"]
        compressed_size = sizes["compressed"] - sizes["compressed_all_but_one"]
        very_compressed_size = sizes["very_compressed"] - sizes["very_compressed_all_but_one"]
        ratio1 = compressed_size / uncompressed_size
        assert ratio1 < 1
        ratio2 = very_compressed_size / uncompressed_size
        assert ratio2 < ratio1

        # Load the data back in

        checkdata = load_data(outdir, comm=self.data.comm,
                              prefix=compressed_prefix)

        # Verify that input and output are equal

        checktod = checkdata.obs[0]["tod"]

        times = tod.local_times()
        checktimes = checktod.local_times()
        assert_allclose(checktimes, times, rtol=1e-6, err_msg="Timestamps do not agree")

        cflags = ((tod.local_common_flags() & tod.TURNAROUND) != 0).astype(np.uint8)
        checkcflags = checktod.local_common_flags()
        assert_array_equal(checkcflags, cflags, err_msg="Common flags do not agree")

        print("\nCompression ratio1 is {:.4f} (default)\n"
              "".format(ratio1), flush=True)
        print("\nCompression ratio2 is {:.4f} (rmstarget={})\n"
              "".format(ratio2, rmstarget), flush=True)

        for det in tod.detectors:
            sig = tod.local_signal(det)
            checksig = checktod.local_signal(det)
            assert_allclose(checksig, sig, atol=1e-5, rtol=1e-3,
                            err_msg="Compressed signal does not agree with the input")

            flg = tod.local_flags(det)
            checkflg = checktod.local_flags(det)
            assert_array_equal(checkflg, flg, err_msg="Flags do not agree")

        return
