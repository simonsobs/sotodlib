# Copyright (c) 2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Check functionality of Splits.

"""
import os
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

from ._helpers import (
    calibration_schedule,
    create_outdir,
    close_data_and_comm,
    simulation_test_data,
)

from sotodlib.toast.ops import pos_to_chi


class SplitTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_split_maps(self):
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
            thin_fp=16,
            cal_schedule=False,
        )

        pointing_radec = toast.ops.PointingDetectorSimple(
            name="det_pointing_radec",
            quats="quats_radec",
            boresight=defaults.boresight_radec,
        )
        pointing_azel = toast.ops.PointingDetectorSimple(
            name="det_pointing_azel",
            quats="quats_azel",
            boresight=defaults.boresight_azel,
        )

        noise_model = toast.ops.DefaultNoiseModel(
            name="default_model", noise_model="noise_model"
        )
        noise_model.apply(data)

        el_model = toast.ops.ElevationNoise(
            name="elevation_model",
            out_model="noise_model",
            detector_pointing=pointing_azel,
        )
        el_model.apply(data)

        weights = toast.ops.StokesWeights(
            name="weights_radec",
            weights="weights_radec",
            mode="IQU",
            detector_pointing=pointing_radec,
        )

        pixels = toast.ops.PixelsHealpix(
            name="pixels_radec",
            detector_pointing=pointing_radec,
            nside=128,
        )

        binner = toast.ops.BinMap(
            name="binner",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )

        mapmaker = toast.ops.MapMaker(
            name="mapmaker",
            binning=binner,
            map_binning=binner,
        )

        # Test that we can instantiate all the splits
        splits = so_ops.Splits(
            name="splits",
            splits=[
                "all",
                "left_going",
                "right_going",
                "outer_detectors",
                "inner_detectors",
                "polA",
                "polB",
            ],
            mapmaker=mapmaker,
            output_dir=self.outdir,
        )

        splits.apply(data)

        close_data_and_comm(data)


if __name__ == '__main__':
    unittest.main()
