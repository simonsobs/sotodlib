# Copyright (c) 2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Check translation between TOAST observations and AxisManagers.

"""
import os
import unittest

import astropy.units as u
import numpy as np

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast

    import toast
    import toast.ops
    from toast.observation import default_values as defaults

    import sotodlib.toast.ops as so_ops
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import (
    create_outdir,
    close_data_and_comm,
    simulation_test_data,
)


class ToastAxmgrTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    # def test_obs_info(self):
    #     if not toast_available:
    #         return
    #     world, procs, rank = toast.get_world()

    #     # Generate synthetic data
    #     data = simulation_test_data(
    #         world,
    #         telescope_name="SAT1",
    #         wafer_slot=None,
    #         bands="SAT_f090",
    #         sample_rate=10.0 * u.Hz,
    #         thin_fp=16,
    #         cal_schedule=False,
    #     )

    #     # Convert simulated naming conventions into the format that matches real data
    #     for ob in data.obs:
    #         sotoast.instrument.convert_sim_meta(ob)
    #         obs_info = sotoast.axman.extract_obs_info(ob)
    #         print(obs_info)

    #     close_data_and_comm(data)

    # def test_sim_roundtrip(self):
    #     if not toast_available:
    #         print("toast cannot be imported- skipping unit tests", flush=True)
    #         return

    #     comm, procs, rank = toast.get_world()
    #     data = simulation_test_data(
    #         comm,
    #         telescope_name="SAT1",
    #         wafer_slot="w28",
    #         bands="SAT_f090",
    #         sample_rate=10.0 * u.Hz,
    #         thin_fp=16,
    #         cal_schedule=False,
    #     )

    #     # Convert local data to AxisManagers


    #     # Convert to new toast observations


    #     # Compare


    #     close_data_and_comm(data)


if __name__ == '__main__':
    unittest.main()
