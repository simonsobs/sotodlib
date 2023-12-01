# Copyright (c) 2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the muMUX crosstalk simulation

"""

import unittest
import numpy as np

import astropy.units as u

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    import toast.ops
    from toast.observation import default_values as defaults
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import calibration_schedule, close_data_and_comm, simulation_test_data


class SimMuMUXCrosstalkTest(unittest.TestCase):
    def test_sim_mumux_crosstalk(self):
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

        # Need to simulate atmosphere to have nonzero signal

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_azel",
            quats="quats_azel",
            boresight="boresight_azel",
        )

        toast.ops.SimAtmosphere(
            name="sim_atmosphere",
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=5 * u.m,
            ystep=5 * u.m,
            zstep=5 * u.m,
            zmax=200 * u.m,
            gain=6e-5,
            wind_dist=3000 * u.m,
            detector_pointing=pointing,
        ).apply(data)

        # Make a copy of the data for reference

        signal0 = data.obs[0].detdata["signal"].data.copy()

        # Simple test just confirms that the operator functions

        so_ops.SimMuMUXCrosstalk(
            name="sim_mumux_crosstalk",
        ).apply(data)

        # Compare signal before and after

        signal1 = data.obs[0].detdata["signal"].data.copy()

        rms0 = np.std(signal0)
        rmsdiff = np.std(signal0 - signal1)

        assert rms0 != 0
        assert rmsdiff / rms0 < 1e-3

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()
