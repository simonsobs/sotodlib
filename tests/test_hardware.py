# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test hardware model creation, dump, and load.
"""

import os

from unittest import TestCase

from collections import OrderedDict

from ._helpers import create_outdir

from sotodlib.hardware.config import Hardware, get_example

from sotodlib.hardware.sim import (sim_wafer_detectors,
                                   sim_telescope_detectors)

from sotodlib.hardware.vis import plot_detectors


class HardwareTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(fixture_name)
        self.skip_plots = False
        if "SOTODLIB_TEST_DISABLE_PLOTS" in os.environ:
            self.skip_plots = os.environ["SOTODLIB_TEST_DISABLE_PLOTS"]

    def test_config_example(self):
        outpath = os.path.join(self.outdir, "hardware_example.toml.gz")
        hw = get_example()
        hw.dump(outpath, overwrite=True, compress=True)
        hwcheck = Hardware(outpath)
        checkpath = os.path.join(self.outdir, "hardware_example_check.toml.gz")
        hwcheck.dump(checkpath, overwrite=True, compress=True)
        return

    def test_sim_wafer(self):
        hw = get_example()
        # Simulate all wafers
        for tele, teleprops in hw.data["telescopes"].items():
            platescale = teleprops["platescale"]
            fwhm = teleprops["fwhm"]
            for tube in teleprops["tubes"]:
                tubeprops = hw.data["tubes"][tube]
                for wafer in tubeprops["wafers"]:
                    outpath = os.path.join(
                        self.outdir, "wafer_{}.toml.gz".format(wafer))
                    dets = sim_wafer_detectors(hw, wafer, platescale, fwhm)
                    # replace detectors with this set for dumping
                    hw.data["detectors"] = dets
                    hw.dump(outpath, overwrite=True, compress=True)
                    if not self.skip_plots:
                        outpath = os.path.join(self.outdir,
                                               "wafer_{}.pdf".format(wafer))
                        plot_detectors(dets, outpath, labels=True)
            return

    def test_sim_telescope(self):
        hw = get_example()
        for tele, teleprops in hw.data["telescopes"].items():
            hw.data["detectors"] = sim_telescope_detectors(hw, tele)
            outpath = os.path.join(self.outdir,
                                   "telescope_{}.toml.gz".format(tele))
            hw.dump(outpath, overwrite=True, compress=True)
            if not self.skip_plots:
                outpath = os.path.join(self.outdir,
                                       "telescope_{}.pdf".format(tele))
                plot_detectors(hw.data["detectors"], outpath, labels=False)
        return

    def test_sim_full(self):
        hw = get_example()
        hw.data["detectors"] = OrderedDict()
        for tele, teleprops in hw.data["telescopes"].items():
            dets = sim_telescope_detectors(hw, tele)
            hw.data["detectors"].update(dets)
        dbpath = os.path.join(self.outdir, "hardware.toml.gz")
        hw.dump(dbpath, overwrite=True, compress=True)
        check = Hardware(dbpath)

        # Test selection of 90GHz detectors on wafers 25 and 26 which have
        # "A" polarization configuration and are located in pixels 20-29.
        wbhw = hw.select(
            match={"wafer": ["25", "26"],
                   "band": "MF.1",
                   "pol": "A",
                   "pixel": "02."})
        dbpath = os.path.join(self.outdir, "w25-26_b1_p20-29_A.toml.gz")
        wbhw.dump(dbpath, overwrite=True, compress=True)
        check = Hardware(dbpath)
        self.assertTrue(len(check.data["detectors"]) == 20)
        chkpath = os.path.join(self.outdir, "w25-26_b1_p20-29_A.txt")
        with open(chkpath, "w") as f:
            for d in check.data["detectors"]:
                f.write("{}\n".format(d))

        # Test selection of pixels on 27GHz wafer 44.
        lfhw = hw.select(
            match={"wafer": ["44"],
                   "pixel": "00."})
        dbpath = os.path.join(self.outdir, "w44_bLF1_p000-009.toml.gz")
        lfhw.dump(dbpath, overwrite=True, compress=True)
        check = Hardware(dbpath)
        self.assertTrue(len(check.data["detectors"]) == 40)
        chkpath = os.path.join(self.outdir, "w44_bLF1_p000-009.txt")
        with open(chkpath, "w") as f:
            for d in check.data["detectors"]:
                f.write("{}\n".format(d))
        return
