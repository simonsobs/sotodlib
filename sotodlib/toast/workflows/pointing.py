# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np
from astropy import units as u
import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops


def setup_pointing(operators):
    """Add commandline args and operators for pointing.

    This sets up operators different types of pointing matrix
    scenarios.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    # Detector quaternion pointing
    operators.append(
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel")
    )
    operators.append(
        toast.ops.PointingDetectorSimple(name="det_pointing_radec", quats="quats_radec")
    )
    # Stokes weights
    operators.append(
        toast.ops.StokesWeights(name="weights_azel", weights="weights_azel", mode="IQU")
    )
    operators.append(toast.ops.StokesWeights(name="weights_radec", mode="IQU"))
    # Healpix pixelization
    operators.append(toast.ops.PixelsHealpix(name="pixels_healpix_radec"))
    operators.append(
        toast.ops.PixelsHealpix(name="pixels_healpix_radec_final", enabled=False)
    )
    # WCS pixelization in Az/El
    operators.append(
        toast.ops.PixelsWCS(
            name="pixels_wcs_azel",
            projection="CAR",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            submaps=1,
            auto_bounds=True,
            enabled=False,
        )
    )
    operators.append(toast.ops.PixelsWCS(name="pixels_wcs_azel_final", enabled=False))
    # WCS pixelization in RA/DEC
    operators.append(
        toast.ops.PixelsWCS(
            name="pixels_wcs_radec",
            projection="CAR",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            submaps=1,
            auto_bounds=True,
            enabled=False,
        )
    )
    operators.append(toast.ops.PixelsWCS(name="pixels_wcs_radec_final", enabled=False))


def select_pointing(job, otherargs, runargs, data):
    """Select pointing scheme and check for consistency

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()

    # Configured operators for this job
    job_ops = job.operators

    n_enabled_solve = np.sum(
        [
            job_ops.pixels_wcs_azel.enabled,
            job_ops.pixels_wcs_radec.enabled,
            job_ops.pixels_healpix_radec.enabled,
        ]
    )
    if n_enabled_solve != 1:
        msg = "Only one pixelization operator should be enabled for the solver."
        log.error(msg)
        raise RuntimeError(msg)

    n_enabled_final = np.sum(
        [
            job_ops.pixels_wcs_azel_final.enabled,
            job_ops.pixels_wcs_radec_final.enabled,
            job_ops.pixels_healpix_radec_final.enabled,
        ]
    )
    if n_enabled_final > 1:
        msg = "At most, one pixelization operator can be enabled "
        msg += "for the final binning."
        log.error(msg)
        raise RuntimeError(msg)

    # Configure Az/El and RA/DEC boresight and detector pointing and weights

    job_ops.det_pointing_azel.boresight = defaults.boresight_azel
    job_ops.det_pointing_radec.boresight = defaults.boresight_radec

    job_ops.pixels_wcs_azel.detector_pointing = job_ops.det_pointing_azel
    job_ops.pixels_wcs_radec.detector_pointing = job_ops.det_pointing_radec
    job_ops.pixels_healpix_radec.detector_pointing = job_ops.det_pointing_radec

    job_ops.pixels_wcs_azel_final.detector_pointing = job_ops.det_pointing_azel
    job_ops.pixels_wcs_radec_final.detector_pointing = job_ops.det_pointing_radec
    job_ops.pixels_healpix_radec_final.detector_pointing = job_ops.det_pointing_radec

    job_ops.weights_azel.detector_pointing = job_ops.det_pointing_azel
    job_ops.weights_radec.detector_pointing = job_ops.det_pointing_radec

    if len(data.obs) > 0 and defaults.hwp_angle in data.obs[0].shared:
        job_ops.weights_azel.hwp_angle = defaults.hwp_angle
        job_ops.weights_radec.hwp_angle = defaults.hwp_angle

    # Select Pixelization and weights for solve and final binning

    if job_ops.pixels_wcs_azel.enabled:
        job.pixels_solve = job_ops.pixels_wcs_azel
        job.weights_solve = job_ops.weights_azel
    elif job_ops.pixels_wcs_radec.enabled:
        job.pixels_solve = job_ops.pixels_wcs_radec
        job.weights_solve = job_ops.weights_radec
    else:
        job.pixels_solve = job_ops.pixels_healpix_radec
        job.weights_solve = job_ops.weights_radec
    job.weights_final = job.weights_solve

    if n_enabled_final == 0:
        # Use same as solve
        job.pixels_final = job.pixels_solve
    else:
        if job_ops.pixels_wcs_azel_final.enabled:
            job.pixels_final = job_ops.pixels_wcs_azel_final
        elif job_ops.pixels_wcs_radec_final.enabled:
            job.pixels_final = job_ops.pixels_wcs_radec_final
        else:
            job.pixels_final = job_ops.pixels_healpix_radec_final

    # Set up pointing matrices for binning operators, if they are
    # being used in this workflow.

    if hasattr(job_ops, "binner"):
        job_ops.binner.pixel_pointing = job.pixels_solve
        job_ops.binner.stokes_weights = job.weights_solve
    if hasattr(job_ops, "binner_final"):
        job_ops.binner_final.pixel_pointing = job.pixels_final
        job_ops.binner_final.stokes_weights = job.weights_final
        # If we are not using a different binner for our final binning, use the
        # same one as the solve.
        if not job_ops.binner_final.enabled:
            job_ops.binner_final = job_ops.binner
