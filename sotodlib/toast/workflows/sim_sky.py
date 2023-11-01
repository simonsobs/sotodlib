# Copyright (c) 2023-2023 Simons Observatory.
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

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.ScanHealpixMap(name="scan_map", enabled=False))
    operators.append(toast.ops.ScanWCSMap(name="scan_wcs_map", enabled=False))


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

    if job_ops.scan_map.enabled and job_ops.scan_wcs_map.enabled:
        msg = "Cannot scan from both healpix and WCS maps"
        log.error(msg)
        raise RuntimeError(msg)

    if job_ops.scan_map.enabled:
        job_ops.scan_map.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.scan_map.pixel_pointing = job.pixels_final
        job_ops.scan_map.stokes_weights = job_ops.weights_radec
        job_ops.scan_map.save_pointing = otherargs.full_pointing
        job_ops.scan_map.apply(data)

    if job_ops.scan_wcs_map.enabled:
        job_ops.scan_wcs_map.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.scan_wcs_map.pixel_pointing = job.pixels_final
        job_ops.scan_wcs_map.stokes_weights = job_ops.weights_radec
        job_ops.scan_wcs_map.save_pointing = otherargs.full_pointing
        job_ops.scan_wcs_map.apply(data)


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
            job_ops.conviqt.detector_pointing = job_ops.det_pointing_radec
            job_ops.conviqt.apply(data)
    else:
        if job_ops.conviqt_teb.enabled:
            job_ops.conviqt_teb.comm = data.comm.comm_world
            job_ops.conviqt_teb.detector_pointing = job_ops.det_pointing_radec
            job_ops.conviqt_teb.hwp_angle = job_ops.sim_ground.hwp_angle
            job_ops.conviqt_teb.apply(data)
