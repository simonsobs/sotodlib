# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the TOAST intensity templates

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


class IntensityTemplateTest(unittest.TestCase):
    def test_intensity_templates(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        data = simulation_test_data(
            comm,
            telescope_name="SAT4",
            wafer_slot="w42",
            bands="SAT_f030,SAT_f040",
            sample_rate=37.0 * u.Hz,
            thin_fp=64,
            cal_schedule=False,
        )

        # Simple test just confirms that the operator functions

        default_model = toast.ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        pointing = toast.ops.PointingDetectorSimple(
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        weights = toast.ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=pointing,
            weights="weights_radec",
        )

        demod = toast.ops.Demodulate(stokes_weights=weights, in_place=True)
        demod.apply(data)

        so_ops.IntensityTemplates(
            name="intensity_templates",
            fpkeys="wafer_slot,bandcenter",
        ).apply(data)

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()
