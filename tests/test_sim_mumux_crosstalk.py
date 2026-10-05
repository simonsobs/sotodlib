# Copyright (c) 2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the muMUX crosstalk simulation"""

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
    from sotodlib.toast.ops import detmap_available, pos_to_chi, jbolo_available

    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import calibration_schedule, close_data_and_comm, simulation_test_data


class SimMuMUXCrosstalkTest(unittest.TestCase):
    def test_sim_mumux_crosstalk(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        if not detmap_available or not jbolo_available:
            print(
                "DetMap / jbolo cannot be imported- skipping muMUX unit tests",
                flush=True,
            )
            return

        if "JBOLO_PATH" not in os.environ or "JBOLO_MODELS_PATH" not in os.environ:
            print(
                "JBOLO environment variables not set- skipping muMUX unit tests",
                flush=True,
            )
            return

        comm, procs, rank = toast.get_world()
        data = simulation_test_data(
            comm,
            telescope_name="LAT",
            wafer_slot="w17",
            bands="LAT_f090",
            sample_rate=10.0 * u.Hz,
            thin_fp=1,
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

        toast.ops.Copy(detdata=[("signal", "input")]).apply(data)

        # Simple test just confirms that the operator functions

        so_ops.SimMuMUXCrosstalk(
            detector_pointing=pointing,
        ).apply(data)

        # Compare signal before and after to make sure the magnitude
        # of the effect is not crazy

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                # We have simulated data in the turnarounds above, and have not
                # done anything else to flag samples.  So we ignore sample flags
                # in this test.
                sig_in = ob.detdata["input"][det]
                sig_out = ob.detdata["signal"][det]
                rms_in = np.std(sig_in)
                rmsdiff = np.std(sig_out - sig_in)
                if rms_in == 0:
                    print(f"{ob.name}:{det} rms_in should not be zero!", flush=True)
                    self.assertTrue(False)
                if rmsdiff / rms_in > 1.0e-2:
                    print(
                        f"{ob.name}:{det} rmsdiff / rms_in = {rmsdiff / rms_in}",
                        flush=True,
                    )
                    self.assertTrue(False)

        # Check that the crosstalk strength is as expected

        for ob in data.obs:
            dets = ob.select_local_detectors(flagmask=defaults.det_mask_invalid)
            fp = ob.telescope.focalplane
            chis = pos_to_chi(fp, dets)
            ndet = len(dets)
            nnz = len(chis)
            med = np.median(np.log10(list(chis.values())))
            if nnz <= ndet or nnz >= 2 * ndet:
                print(f"{ob.name} ndet = {ndet}", flush=True)
                print(f"{ob.name} nnz = {nnz} = {nnz / ndet} ndet", flush=True)
                self.assertTrue(False)
            if med < -3 or med > -2:
                print(f"{ob.name} median(log10(chi)) = {med}", flush=True)
                self.assertTrue(False)

        close_data_and_comm(data)


if __name__ == "__main__":
    unittest.main()
