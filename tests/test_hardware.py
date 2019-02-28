# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test hardware model creation, dump, and load.
"""

import os

from unittest import TestCase

import toml

from ._helpers import create_outdir

from sotodlib.hardware.config import get_example

from sotodlib.hardware.sim import (sim_wafer_detectors,
                                   sim_telescope_detectors)

from sotodlib.hardware.vis import plot_detectors


class HardwareTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(fixture_name)

    def test_config_example(self):
        outpath = os.path.join(self.outdir, "config_example.toml")
        conf = get_example()
        with open(outpath, "w") as f:
            toml.dump(conf, f)

        check = None
        with open(outpath, "r") as f:
            check = toml.load(f)

        checkpath = os.path.join(self.outdir, "config_example_check.toml")
        with open(checkpath, "w") as f:
            toml.dump(check, f)

        self.assertTrue(conf == check)
        return

    # def test_sim_wafer(self):
    #     conf = get_example()
    #     for tele, teleprops in conf["telescopes"].items():
    #         platescale = teleprops["platescale"]
    #         plotdim = 150.0 * platescale
    #         fwhm = teleprops["fwhm"]
    #         for tube in teleprops["tubes"]:
    #             tubeprops = conf["tubes"][tube]
    #             for wafer in tubeprops["wafers"]:
    #                 wprops = conf["wafers"][wafer]
    #                 print("telescope {}, tube {}, wafer {}, platescale = {},"
    #                       " fwhm = {}".format(tele, tube, wafer, platescale,
    #                       fwhm[wprops["bands"][0]]), flush=True)
    #                 outpath = os.path.join(self.outdir,
    #                                        "wafer_{}.toml".format(wafer))
    #                 dets = sim_wafer_detectors(conf, wafer, platescale, fwhm)
    #                 # with open(outpath, "w") as f:
    #                 #     toml.dump(dets, f)
    #                 outpath = os.path.join(self.outdir,
    #                                        "wafer_{}.pdf".format(wafer))
    #                 plot_detectors(dets, plotdim, plotdim, outpath,
    #                                labels=True)
    #     return

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
                                   "telescope_{}.toml".format(tele))
            with open(outpath, "w") as f:
                toml.dump(dets, f)
            outpath = os.path.join(self.outdir,
                                   "telescope_{}.pdf".format(tele))
            plot_detectors(dets, plotdim, plotdim, outpath, labels=False)
        return
