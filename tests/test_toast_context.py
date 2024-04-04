# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast context loading.
"""

import os
import shutil
import glob
import re
import yaml

import numpy as np
import astropy.units as u

import unittest
from unittest import TestCase

# Import so3g before any other packages that import spt3g
import so3g

from ._helpers import create_outdir, simulation_test_data, close_data_and_comm


try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    from toast.observation import default_values as defaults

    toast_available = True
except ImportError:
    raise
    toast_available = False

import sotodlib


class ToastContextTest(TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_context_load(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()

        cloader = so_ops.LoadContext(
            telescope_name="SAT1",
            context=None,
            readout_ids=['w25_p000_SAT_f090_A'],
        )


if __name__ == '__main__':
    unittest.main()
