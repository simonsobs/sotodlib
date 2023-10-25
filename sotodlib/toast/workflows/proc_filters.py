# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream processing filters.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_deconvolve_detector_timeconstant(operators):
    """Add commandline args and operators for timeconstant convolution.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.TimeConstant(
            name="deconvolve_time_constant", deconvolve=True, enabled=False
        )
    )


def deconvolve_detector_timeconstant(job, otherargs, runargs, data):
    """Deconvolve the detector timeconstants.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.deconvolve_time_constant.enabled:
        log.info_rank(
            "Running detector timeconstant deconvolution...", comm=data.comm.comm_world
        )
        job_ops.deconvolve_time_constant.apply(data)
        log.info_rank(
            "Deconvolved time constant in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After deconvolving time constant"
        job_ops.mem_count.apply(data)


def setup_filter_hwpss(operators):
    """Add commandline args and operators for HWPSS filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.HWPFilter(name="hwpfilter", enabled=False))


def filter_hwpss(job, otherargs, runargs, data):
    """Filter HWP synchronous signal.

    This filters the HWPSS, as opposed to solving for it in the mapmaking
    with a template.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.hwpfilter.enabled:
        log.info_rank("Running HWPSS filtering...", comm=data.comm.comm_world)
        job_ops.hwpfilter.apply(data)
        log.info_rank("HWP-filtered in", comm=data.comm.comm_world, timer=timer)


def setup_filter_ground(operators):
    """Add commandline args and operators for ground filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.GroundFilter(name="groundfilter", enabled=False))


def filter_ground(job, otherargs, runargs, data):
    """Filter Azimuth synchronous signal.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.groundfilter.enabled:
        log.info_rank("Running ground filter...", comm=data.comm.comm_world)
        job_ops.groundfilter.apply(data)
        log.info_rank(
            "Finished ground-filtering in", comm=data.comm.comm_world, timer=timer
        )


def setup_filter_poly1d(operators):
    """Add commandline args and operators for 1D polynomial filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.PolyFilter(name="polyfilter1D"))


def filter_poly1d(job, otherargs, runargs, data):
    """Filter scans with 1D polynomial.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.polyfilter1D.enabled:
        log.info_rank("Running 1D polynomial filtering", comm=data.comm.comm_world)
        job_ops.polyfilter1D.apply(data)
        log.info_rank(
            "Finished 1D-poly-filtering in", comm=data.comm.comm_world, timer=timer
        )


def setup_filter_poly2d(operators):
    """Add commandline args and operators for 2D polynomial filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.PolyFilter2D(name="polyfilter2D", enabled=False))


def filter_poly2d(job, otherargs, runargs, data):
    """Filter data with a 2D polynomial.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.polyfilter2D.enabled:
        log.info_rank("Running 2D polynomial filtering", comm=data.comm.comm_world)
        job_ops.polyfilter2D.apply(data)
        log.info_rank(
            "Finished 2D-poly-filtering in", comm=data.comm.comm_world, timer=timer
        )


def setup_filter_common_mode(operators):
    """Add commandline args and operators for common mode filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.CommonModeFilter(name="common_mode_filter", enabled=False)
    )


def filter_common_mode(job, otherargs, runargs, data):
    """Filter data to remove common modes across the detectors.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.common_mode_filter.enabled:
        log.info_rank("Running common mode filtering...", comm=data.comm.comm_world)
        job_ops.common_mode_filter.apply(data)
        log.info_rank(
            "Finished common-mode-filtering in", comm=data.comm.comm_world, timer=timer
        )
