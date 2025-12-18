# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast context loading."""

import os
import copy

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

    def test_context_ctor(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()

        cloader = so_ops.LoadContext(
            telescope_name="SAT1",
            context=None,
            readout_ids=["w25_p000_SAT_f090_A"],
        )

    def _fake_hk(self, data):
        """Add fake housekeeping data"""
        for ob in data.obs:
            timestamps = ob.shared[defaults.times].data
            slow_times = timestamps[::10]
            ivals = np.random.randint(-100, high=100, size=len(slow_times), dtype=np.int32)
            fvals = np.random.random(size=len(slow_times))
            ob.hk = sotodlib.toast.hkmanager.HKManager(
                data.comm.comm_group, timestamps
            )
            ob.hk._internal["integers"] = (slow_times, ivals)
            ob.hk._internal["floats"] = (slow_times, fvals)

    def test_hk_hdf5(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()

        data = simulation_test_data(
            world,
            telescope_name="SAT1",
            wafer_slot=None,
            bands="SAT_f090",
            sample_rate=10.0 * u.Hz,
            thin_fp=16,
            cal_schedule=False,
        )
        self._fake_hk(data)

        orig = dict()
        for ob in data.obs:
            orig[ob.name] = copy.deepcopy(ob.hk)

        vol_path = os.path.join(self.outdir, "test_hk")
        toast.ops.SaveHDF5(
            volume=vol_path,
            detdata=[
                (defaults.det_data, {"quanta": 1.0e-10}),
                (defaults.det_flags, {"level": 6}),
            ]
        ).apply(data)

        new_data = toast.Data(comm=data.comm)
        toast.ops.LoadHDF5(volume=vol_path).apply(new_data)

        for old_ob, new_ob in zip(data.obs, new_data.obs):
            if old_ob != new_ob:
                print(f"OLD: {old_ob} not equal to")
                print(f"NEW: {new_ob}", flush=True)
                #self.assertTrue(False)

        close_data_and_comm(new_data)
        close_data_and_comm(data)



if __name__ == "__main__":
    unittest.main()
