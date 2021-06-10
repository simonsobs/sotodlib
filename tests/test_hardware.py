# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test hardware model creation, dump, and load.
"""

import os

from unittest import TestCase

from collections import OrderedDict

from ._helpers import create_outdir

from sotodlib.core import Hardware

from sotodlib.sim_hardware import get_example

from sotodlib.sim_hardware import (sim_wafer_detectors,
                                   sim_telescope_detectors)

from sotodlib.vis_hardware import plot_detectors


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
        # Simulate some wafers
        for tele, teleprops in hw.data["telescopes"].items():
            if tele != "SAT1":
                continue
            platescale = teleprops["platescale"]
            fwhm = teleprops["fwhm"]
            for tube_slot in teleprops["tube_slots"]:
                if tube_slot != "ST1":
                    continue
                tubeprops = hw.data["tube_slots"][tube_slot]
                for wafer in tubeprops["wafer_slots"]:
                    if wafer != "w25":
                        continue
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
        fullhw = get_example()
        fullhw.data["detectors"] = sim_telescope_detectors(fullhw, "SAT1")
        hw = fullhw.select(match={"wafer_slot": ["w25",],})
        outpath = os.path.join(self.outdir, "telescope_SAT1_w25.toml.gz")
        hw.dump(outpath, overwrite=True, compress=True)
        if not self.skip_plots:
            outpath = os.path.join(self.outdir, "telescope_SAT1_w25.pdf")
            plot_detectors(hw.data["detectors"], outpath, labels=False)
        return

    def test_sim_full(self):
        hw = get_example()
        hw.data["detectors"] = OrderedDict()
        for tele, teleprops in hw.data["telescopes"].items():
            if tele != "SAT1":
                continue
            dets = sim_telescope_detectors(hw, tele)
            hw.data["detectors"].update(dets)
        dbpath = os.path.join(self.outdir, "hardware_SAT1.toml.gz")
        hw.dump(dbpath, overwrite=True, compress=True)
        check = Hardware(dbpath)

        # Test selection of 90GHz detectors on wafers 25 and 26 which have
        # "A" polarization configuration and are located in pixels 20-29.
        wbhw = hw.select(
            match={"wafer_slot": ["w25", "w26"],
                   "band": "SAT_f090",
                   "pol": "A",
                   "pixel": "02."})
        dbpath = os.path.join(self.outdir, "w25-26_p20-29_SAT_f090_A.toml.gz")
        wbhw.dump(dbpath, overwrite=True, compress=True)
        check = Hardware(dbpath)
        self.assertTrue(len(check.data["detectors"]) == 20)
        chkpath = os.path.join(self.outdir, "w25-26_p20-29_SAT_f090_A.txt")
        with open(chkpath, "w") as f:
            for d in check.data["detectors"]:
                f.write("{}\n".format(d))
        return
