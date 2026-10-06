# Copyright (c) 2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of SimSatellite

"""

import unittest

import astropy.units as u
import numpy as np

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import toast
    import toast.ops
    from toast.observation import default_values as defaults

    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import (calibration_schedule, close_data_and_comm,
                      simulation_test_data)

from sotodlib.toast.ops import pos_to_chi


# Project 4: Implement a unit test for SimSatellite


class SimSatellitesTest(unittest.TestCase):

    def test_sim_satellites(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        data = simulation_test_data(
            comm,
            telescope_name="SAT1",
            wafer_slot="w28",
            bands="SAT_f090",
            sample_rate=10.0 * u.Hz,
            thin_fp=1,
            cal_schedule=False,
        )

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_radec",
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        weights = toast.ops.StokesWeights(
            name="weights_radec",
            weights="weights_radec",
            mode="IQU",
            detector_pointing=pointing,
        )

        # Simple test just confirms that the operator functions

        so_ops.SimSatellites(
            name="sim_satellites", stokes_weights=weights
        ).apply(data)

        # Make sure the magnitude of the effect is not crazy

        signal = data.obs[0].detdata["signal"].data.copy()

        rms = np.std(signal)

        assert rms > .1
        assert rms < 10

        close_data_and_comm(data)


if __name__ == '__main__':
    unittest.main()
