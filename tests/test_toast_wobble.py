# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test application of wobble correction."""

import os

import numpy as np
import astropy.units as u
from astropy.table import Column

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


class ToastWobbleTest(TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_wobble_correction(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()

        zaxis = np.array([0.0, 0.0, 1.0])

        # Generate synthetic data
        data = simulation_test_data(
            world,
            telescope_name="SAT1",
            wafer_slot=None,
            bands="SAT_f090",
            sample_rate=40.0 * u.Hz,
            thin_fp=16,
            cal_schedule=False,
        )

        raw_amp = 60.0 # arcmin
        raw_phase = np.pi / 6 # radians
        amp_name = "wobble_params:amp"
        phase_name = "wobble_params:phase"

        so3g_compare = dict()

        for ob in data.obs:
            # Add columns to focalplane table
            fp = ob.telescope.focalplane.detector_data
            n_det = len(fp)
            amp_col = Column(
                name=amp_name,
                data=raw_amp * np.ones(n_det, dtype=np.float64),
            )
            phase_col = Column(
                name=phase_name,
                data=raw_phase * np.ones(n_det, dtype=np.float64),
            )
            if amp_name not in fp.colnames:
                fp.add_column(amp_col)
            if phase_name not in fp.colnames:
                fp.add_column(phase_col)

            # Compute the sight line deflection according to so3g
            amp = raw_amp/60.*np.pi/180.0
            phase = raw_phase

            dxi = amp * np.cos(ob.shared["hwp_angle"].data - phase)
            deta = -amp * np.sin(ob.shared["hwp_angle"].data - phase)
            deflq = so3g.proj.quat.rotation_xieta(xi=dxi, eta=deta)

            times = np.copy(ob.shared["times"].data)
            lon, lat, psi = toast.qarray.to_lonlat_angles(
                ob.shared['boresight_azel'].data
            )

            az = -lon
            el = lat
            roll = psi

            sight = so3g.proj.CelestialSightLine.for_horizon(
                times, az, el, roll=roll
            )
            sight.Q = sight.Q * ~deflq
            so3g_compare[ob.uid] = toast.spt3g.from_g3_quats(sight.Q)

        so_ops.HWPWobbleCorrect().apply(data)

        for ob in data.obs:
            check_vec = toast.qarray.rotate(so3g_compare[ob.uid], zaxis)
            bore_vec = toast.qarray.rotate(ob.shared["boresight_azel"].data, zaxis)
            self.assertTrue(np.allclose(check_vec, bore_vec))

        close_data_and_comm(data)


if __name__ == "__main__":
    unittest.main()
