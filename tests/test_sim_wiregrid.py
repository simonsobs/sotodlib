# Copyright (c) 2020 Simons Observatory.
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


class SimWireGridTest(unittest.TestCase):
    def test_sim_wiregrid(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        data = simulation_test_data(
            comm,
            telescope_name="SAT1",
            wafer_slot=None,
            bands="SAT_f090",
            sample_rate=10.0 * u.Hz,
            thin_fp=16,
            cal_schedule=True,
        )

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_azel",
            quats="quats_azel",
            boresight=defaults.boresight_azel,
        )
        weights = toast.ops.StokesWeights(
            name="weights_azel",
            weights="weights_azel",
            mode="IQU",
            detector_pointing=pointing,
        )

        so_ops.SimWireGrid(
            name="sim_wiregrid",
            detector_pointing=pointing,
            detector_weights=weights,
            wiregrid_angular_speed=10.0*u.degree/u.second,
            wiregrid_angular_acceleration=0.1*u.degree/u.second**2,
        ).apply(data)

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()
