# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream processing filters.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_raw_statistics(operators):
    """Add commandline args and operators for raw timestream statistics.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.Statistics(name="raw_statistics", enabled=False))


@workflow_timer
def raw_statistics(job, otherargs, runargs, data):
    """Compute timestream statistics on the raw data.

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

    if job_ops.raw_statistics.enabled:
        job_ops.raw_statistics.output_dir = otherargs.out_dir
        job_ops.raw_statistics.apply(data)


def setup_filtered_statistics(operators):
    """Add commandline args and operators for filtered timestream statistics.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.Statistics(name="filtered_statistics", enabled=False))


@workflow_timer
def filtered_statistics(job, otherargs, runargs, data):
    """Compute timestream statistics on the filtered data.

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

    if job_ops.filtered_statistics.enabled:
        job_ops.filtered_statistics.output_dir = otherargs.out_dir
        job_ops.filtered_statistics.apply(data)


def setup_hn_map(operators):
    """Add commandline args and operators for H_n map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.Hn(name="h_n", enabled=False))


@workflow_timer
def hn_map(job, otherargs, runargs, data):
    """Compute the H_n map.

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

    if job_ops.h_n.enabled:
        job_ops.h_n.pixel_pointing = job.pixels_final
        job_ops.h_n.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.h_n.output_dir = otherargs.out_dir
        job_ops.h_n.save_pointing = otherargs.full_pointing
        job_ops.h_n.apply(data)


def setup_cadence_map(operators):
    """Add commandline args and operators for the cadence map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.CadenceMap(name="cadence_map", enabled=False))


@workflow_timer
def cadence_map(job, otherargs, runargs, data):
    """Compute the cadence map.

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

    if job_ops.cadence_map.enabled:
        job_ops.cadence_map.pixel_pointing = job.pixels_final
        job_ops.cadence_map.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.cadence_map.output_dir = otherargs.out_dir
        job_ops.cadence_map.save_pointing = otherargs.full_pointing
        job_ops.cadence_map.apply(data)


def setup_crosslinking_map(operators):
    """Add commandline args and operators for the crosslinking map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.CrossLinking(name="crosslinking", enabled=False))


@workflow_timer
def crosslinking_map(job, otherargs, runargs, data):
    """Compute the crosslinking map.

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

    if job_ops.crosslinking.enabled:
        job_ops.crosslinking.pixel_pointing = job.pixels_final
        job_ops.crosslinking.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.crosslinking.output_dir = otherargs.out_dir
        job_ops.crosslinking.save_pointing = otherargs.full_pointing
        job_ops.crosslinking.apply(data)
