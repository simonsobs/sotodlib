# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test hardware model creation, dump, and load.
"""

import os

from unittest import TestCase

from collections import OrderedDict

import toml

from ._helpers import create_outdir

from sotodlib.hardware.config import get_example

from sotodlib.hardware.config import dump as conf_dump

from sotodlib.hardware.config import load as conf_load

from sotodlib.hardware.config import select as conf_select

from sotodlib.hardware.sim import (sim_wafer_detectors,
                                   sim_telescope_detectors)

from sotodlib.hardware.vis import plot_detectors


class HardwareTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(fixture_name)

    def test_config_example(self):
        outpath = os.path.join(self.outdir, "config_example.toml.gz")
        conf = get_example()
        conf_dump(outpath, conf, overwrite=True, compress=True)
        check = conf_load(outpath)
        checkpath = os.path.join(self.outdir, "config_example_check.toml.gz")
        conf_dump(checkpath, check, overwrite=True, compress=True)
        return

    def test_sim_wafer(self):
        conf = get_example()
        for tele, teleprops in conf["telescopes"].items():
            platescale = teleprops["platescale"]
            plotdim = 150.0 * platescale
            fwhm = teleprops["fwhm"]
            for tube in teleprops["tubes"]:
                tubeprops = conf["tubes"][tube]
                for wafer in tubeprops["wafers"]:
                    outpath = os.path.join(
                        self.outdir, "wafer_{}_dets.toml.gz".format(wafer))
                    dets = sim_wafer_detectors(conf, wafer, platescale, fwhm)
                    conf_dump(outpath, dets, overwrite=True, compress=True)
                    outpath = os.path.join(self.outdir,
                                           "wafer_{}.pdf".format(wafer))
                    plot_detectors(dets, plotdim, plotdim, outpath,
                                   labels=True)
        return

    def test_sim_telescope(self):
        conf = get_example()
        for tele, teleprops in conf["telescopes"].items():
            platescale = teleprops["platescale"]
            plotdim = None
            if tele[0] == "S":
                plotdim = 400.0 * platescale
            else:
                plotdim = 900.0 * platescale
            dets = sim_telescope_detectors(conf, tele)
            outpath = os.path.join(self.outdir,
                                   "telescope_{}_dets.toml.gz".format(tele))
            conf_dump(outpath, dets, overwrite=True, compress=True)
            outpath = os.path.join(self.outdir,
                                   "telescope_{}.pdf".format(tele))
            plot_detectors(dets, plotdim, plotdim, outpath, labels=False)
        return

    def test_sim_full(self):
        conf = get_example()
        alldets = OrderedDict()
        for tele, teleprops in conf["telescopes"].items():
            dets = sim_telescope_detectors(conf, tele)
            alldets.update(dets)
        conf["detectors"] = alldets
        dbpath = os.path.join(self.outdir, "hardware.toml.gz")
        conf_dump(dbpath, conf, overwrite=True, compress=True)
        check = conf_load(dbpath)

        # Test selection of 90GHz detectors on wafers 25 and 26 which have
        # "A" polarization configuration and are located in pixels 20-29.
        wbconf = conf_select(
            conf, {"wafer": ["25", "26"],
                   "band": "MF.1",
                   "pol": "A",
                   "pixel": "02."})
        dbpath = os.path.join(self.outdir, "w25-26_b1_p20-29_A.toml.gz")
        conf_dump(dbpath, wbconf, overwrite=True, compress=True)
        check = conf_load(dbpath)
        self.assertTrue(len(check["detectors"]) == 20)
        chkpath = os.path.join(self.outdir, "w25-26_b1_p20-29_A.txt")
        with open(chkpath, "w") as f:
            for d in check["detectors"]:
                f.write("{}\n".format(d))
        return
