# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast workflow tools.
"""

import os
import shutil
import glob
import re

import numpy as np
import astropy.units as u

from unittest import TestCase

# Import so3g before any other packages that import spt3g
import so3g

from ._helpers import create_outdir

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import sotodlib.toast.workflows as wrk
    import toast
    from toast.observation import default_values as defaults

    toast_available = True
except ImportError:
    raise
    toast_available = False


class ToastWorkflowTest(TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_workflow_config(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()
        testdir = os.path.join(self.outdir, "config")
        if world is None or world.rank == 0:
            if os.path.isdir(testdir):
                shutil.rmtree(testdir)
            os.makedirs(testdir)
        if world is not None:
            world.barrier()

        operators = list()
        wrk.setup_simulate_atmosphere_signal(operators)

        job, config, otherargs, runargs = wrk.setup_job(
            operators=operators,
            opts={
                "sim_atmosphere.enable": True,
                "sim_atmosphere.xstep": "10.0 m",
            },
        )

        self.assertTrue(job.operators.sim_atmosphere.enabled)
        self.assertTrue(job.operators.sim_atmosphere.xstep == 10.0 * u.meter)
