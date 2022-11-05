# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the wiregrid simulation

"""

import unittest
import numpy as np

import astropy.units as u

try:
    import toast
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import toast_site, calibration_schedule, close_data_and_comm


class SimWireGridTest(unittest.TestCase):
    def test_instantiate(self):
        """Test instantiating a simulation operator."""
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        toast_comm = toast.Comm(world=comm, groupsize=procs)
        data = toast.Data(comm=toast_comm)

        focalplane = sotoast.SOFocalplane(
            hwfile=None,
            telescope="SAT1",
            sample_rate=10 * u.Hz,
            bands="SAT_f090",
            wafer_slots=None,
            tube_slots=None,
            thinfp=16,
            comm=comm,
        )

        site = toast_site()
        telescope = toast.instrument.Telescope(
            "SAT1",
            focalplane=focalplane,
            site=site,
        )

        schedule = calibration_schedule(telescope)

        sim_ground = toast.ops.SimGround(
            name="sim_ground",
            weather="atacama",
            detset_key="pixel",
            telescope=telescope,
            schedule=schedule,
        )
        sim_ground.apply(data)

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_azel",
            quats="quats_azel",
            boresight=sim_ground.boresight_azel,
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
