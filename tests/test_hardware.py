# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test hardware model creation, dump, and load.
"""

import os

from unittest import TestCase

import toml

from ._helpers import create_outdir

from sotodlib.db.config import get_example

from sotodlib.db.hardware import (sim_wafer_detectors,)


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

    def test_sim_wafer(self):
        conf = get_example()
        for tele, teleprops in conf["telescopes"].items():
            platescale = teleprops["platescale"]
            for tube in teleprops["tubes"]:
                tubeprops = conf["tubes"][tube]
                for wafer in tubeprops["wafers"]:
                    outpath = os.path.join(self.outdir,
                                           "wafer_{}.toml".format(wafer))
                    dets = sim_wafer_detectors(conf, wafer, platescale)
                    with open(outpath, "w") as f:
                        toml.dump(dets, f)
        return
