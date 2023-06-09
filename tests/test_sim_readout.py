# Copyright (c) 2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the wiregrid simulation

"""

import unittest
import numpy as np

import astropy.units as u

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    from toast.observation import default_values as defaults
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import calibration_schedule, close_data_and_comm, simulation_test_data


class SimReadoutTest(unittest.TestCase):
    def test_sim_readout(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        data = simulation_test_data(
            comm,
            telescope_name="LAT",
            wafer_slot="w17",
            bands="LAT_f090",
            sample_rate=10.0 * u.Hz,
            thin_fp=64,
            cal_schedule=False,
        )

        # Simple test just confirms that the operator functions

        so_ops.SimReadout(
            name="sim_readout",
            simulate_glitches=True,
            simulate_jumps=True,
            misidentify_bolometers=True,
            simulate_yield=True,
        ).apply(data)

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()
