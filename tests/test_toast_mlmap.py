# Copyright (c) 2018-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast interface to ML mapmaker.
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

from sotodlib.sim_hardware import get_example

from sotodlib.sim_hardware import sim_telescope_detectors


toast_available = None
if toast_available is None:
    try:
        import toast
        from toast.mpi import MPI
        from toast.todmap import TODGround
        from toast.tod import AnalyticNoise
        from sotodlib.toast.export import ToastExport
        from sotodlib.toast.load import load_data
        toast_available = True
    except ImportError:
        toast_available = False


class ToastMLmapTest(TestCase):

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


    def test_map(self):
        if not toast_available:
            return
        pass

