# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream processing filters.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_readout_filter(operators):
    """Add commandline args and operators for readout filter.
    Args:
        operators (list):  The list of operators to extend.
    Returns:
        None
    """
    operators.append(
        so_ops.ReadoutFilter(name="readout_filter", enabled=False)
    )


@workflow_timer
def apply_readout_filter(job, otherargs, runargs, data):
    """Apply readout filter.
    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.
    Returns:
        None
    """
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.readout_filter.enabled:
        job_ops.readout_filter.apply(data)


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


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.deconvolve_time_constant.enabled:
        job_ops.deconvolve_time_constant.apply(data)


def setup_filter_hwpss(operators):
    """Add commandline args and operators for HWPSS filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.HWPFilter(name="hwpfilter", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.hwpfilter.enabled:
        job_ops.hwpfilter.apply(data)


def setup_filter_ground(operators):
    """Add commandline args and operators for ground filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.GroundFilter(name="groundfilter", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.groundfilter.enabled:
        job_ops.groundfilter.apply(data)


def setup_filter_poly1d(operators):
    """Add commandline args and operators for 1D polynomial filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.PolyFilter(name="polyfilter1D", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.polyfilter1D.enabled:
        job_ops.polyfilter1D.apply(data)


def setup_filter_poly2d(operators):
    """Add commandline args and operators for 2D polynomial filters.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.PolyFilter2D(name="polyfilter2D", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.polyfilter2D.enabled:
        job_ops.polyfilter2D.apply(data)


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


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.common_mode_filter.enabled:
        job_ops.common_mode_filter.apply(data)
