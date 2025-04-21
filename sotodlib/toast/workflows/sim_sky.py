# Copyright (c) 2023-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated detector response to sky signal.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from ..instrument import simulated_telescope
from .job import workflow_timer


def setup_simulate_sky_map_signal(operators):
    """Add commandline args and operators for scanning from a map.

    There are pointing matrix operators associated with each scan map operator
    and these are disabled by default.  If not enabled, the scan map operator
    use the same pointing matrix as the mapmaking.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.ScanHealpixMap(name="scan_map", enabled=False))
    operators.append(toast.ops.ScanHealpixDetectorMap(name="scan_detector_map", enabled=False))
    operators.append(
        toast.ops.StokesWeights(
            name="scan_map_weights",
            mode="IQU",
            weights="weights_scan_map",
            enabled=False,
        )
    )
    operators.append(
        toast.ops.PixelsHealpix(
            name="scan_map_pixels", pixels="pixels_scan_map", enabled=False
        )
    )
    operators.append(toast.ops.ScanWCSMap(name="scan_wcs_map", enabled=False))
    operators.append(
        toast.ops.StokesWeights(
            name="scan_wcs_map_weights",
            mode="IQU",
            weights="weights_scan_map",
            enabled=False,
        )
    )
    operators.append(
        toast.ops.PixelsWCS(
            name="scan_wcs_map_pixels",
            projection="CAR",
            pixels="pixels_scan_map",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            submaps=1,
            auto_bounds=True,
            enabled=False,
        )
    )


@workflow_timer
def simulate_sky_map_signal(job, otherargs, runargs, data):
    """Scan a sky map into detector signals.

    This uses detector pointing to sample from a map into detector
    timestreams.  The maps should already be smoothed with any desired
    beam effects.

    We scan the sky with the "final" pointing model in case that is
    different from the solver pointing model.

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

    scan_healpix = job_ops.scan_map.enabled or job_ops.scan_detector_map.enabled
    scan_wcs = job_ops.scan_wcs_map.enabled
    scan_map = scan_healpix or scan_wcs

    # Check for conflicting options

    if scan_healpix and scan_wcs:
        msg = "Cannot scan from both healpix and WCS maps"
        log.error(msg)
        raise RuntimeError(msg)

    if job_ops.det_pointing_radec_sim is not job_ops.det_pointing_radec:
        if scan_healpix and not job_ops.scan_map_pixels.enabled:
            msg = "Simulation pointing is different from data reduction pointing. " \
                " You must enable scan_map_pixels"
            log.error(msg)
            raise RuntimeError(msg)
        if scan_wcs and not job_ops.scan_wcs_map_pixels.enabled:
            msg = "Simulation pointing is different from data reduction pointing. " \
                " You must enable scan_wcs_map_pixels"
            log.error(msg)
            raise RuntimeError(msg)

    # Scan signal from HEALPix maps

    for scan_op in job_ops.scan_map, job_ops.scan_detector_map:
        if scan_op.enabled:
            if job_ops.scan_map_pixels.enabled:
                # We are using a custom pointing matrix
                job_ops.scan_map_pixels.detector_pointing = job_ops.det_pointing_radec_sim
                scan_op.pixel_dist = "scan_map_pixel_dist"
                scan_op.pixel_pointing = job_ops.scan_map_pixels
            else:
                # We are using the same pointing matrix as the mapmaking
                scan_op.pixel_dist = job_ops.binner_final.pixel_dist
                scan_op.pixel_pointing = job.pixels_final
            if job_ops.scan_map_weights.enabled:
                job_ops.scan_map_weights.detector_pointing = job_ops.det_pointing_radec_sim
                scan_op.stokes_weights = job_ops.scan_map_weights
            else:
                scan_op.stokes_weights = job_ops.weights_radec
            scan_op.save_pointing = otherargs.full_pointing
            scan_op.apply(data)
            if job_ops.scan_map_pixels.enabled:
                # Clean up our custom pointing
                toast.ops.Delete(detdata=[
                    job_ops.scan_map_pixels.pixels,
                    job_ops.det_pointing_radec_sim.quats,
                ]).apply(data)
            if job_ops.scan_map_weights.enabled:
                # Clean up our custom pointing
                toast.ops.Delete(detdata=[
                    job_ops.scan_map_weights.weights,
                ]).apply(data)
            data.info()

    # Scan signal from WCS maps

    if scan_wcs:
        if job_ops.scan_wcs_map_pixels.enabled:
            # We are using a custom pointing matrix
            job_ops.scan_wcs_map_pixels.detector_pointing = job_ops.det_pointing_radec_sim
            job_ops.scan_wcs_map.pixel_dist = "scan_wcs_map_pixel_dist"
            job_ops.scan_wcs_map.pixel_pointing = job_ops.scan_wcs_map_pixels
        else:
            # We are using the same pointing matrix as the mapmaking
            job_ops.scan_wcs_map.pixel_dist = job_ops.binner_final.pixel_dist
            job_ops.scan_wcs_map.pixel_pointing = job.pixels_final
        if job_ops.scan_wcs_map_weights.enabled:
            job_ops.scan_wcs_map_weights.detector_pointing = job_ops.det_pointing_radec_sim
            job_ops.scan_wcs_map.stokes_weights = job_ops.scan_wcs_map_weights
        else:
            job_ops.scan_wcs_map.stokes_weights = job_ops.weights_radec
        job_ops.scan_wcs_map.save_pointing = otherargs.full_pointing
        job_ops.scan_wcs_map.apply(data)
        if job_ops.scan_wcs_map_pixels.enabled:
            # Clean up our custom pointing
            toast.ops.Delete(detdata=[
                job_ops.scan_wcs_map_pixels.pixels,
                job_ops.det_pointing_radec_sim.quats,
            ]).apply(data)
        if job_ops.scan_wcs_map_weights.enabled:
            # Clean up our custom pointing
            toast.ops.Delete(detdata=[
                job_ops.scan_wcs_map_weights.weights,
            ]).apply(data)


def setup_simulate_conviqt_signal(operators):
    """Add commandline args and operators for beam covolution with conviqt.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.SimConviqt(name="conviqt", enabled=False))
    operators.append(toast.ops.SimTEBConviqt(name="conviqt_teb", enabled=False))


@workflow_timer
def simulate_conviqt_signal(job, otherargs, runargs, data):
    """Use libconviqt to generate beam-convolved sky signal timestreams.

    This uses detector pointing and input beam and sky a_lm expansions
    to generate detector timestreams.

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

    if job_ops.conviqt.enabled and job_ops.sim_ground.hwp_angle is not None:
        msg = "Data has a half-wave plate.  Use conviqt_teb operator "
        msg += "instead of conviqt."
        log.error(msg)
        raise RuntimeError(msg)

    if job_ops.conviqt_teb.enabled and job_ops.sim_ground.hwp_angle is None:
        msg = "Data has no half-wave plate.  Use conviqt operator "
        msg += "instead of conviqt_teb."
        log.error(msg)
        raise RuntimeError(msg)

    if job_ops.sim_ground.hwp_angle is None:
        if job_ops.conviqt.enabled:
            job_ops.conviqt.comm = data.comm.comm_world
            job_ops.conviqt.detector_pointing = job_ops.det_pointing_radec_sim
            job_ops.conviqt.apply(data)
    else:
        if job_ops.conviqt_teb.enabled:
            job_ops.conviqt_teb.comm = data.comm.comm_world
            job_ops.conviqt_teb.detector_pointing = job_ops.det_pointing_radec_sim
            job_ops.conviqt_teb.hwp_angle = job_ops.sim_ground.hwp_angle
            job_ops.conviqt_teb.apply(data)
