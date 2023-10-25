# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream processing filters.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_raw_statistics(operators):
    """Add commandline args and operators for raw timestream statistics.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.Statistics(name="raw_statistics", enabled=False))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.raw_statistics.output_dir = otherargs.out_dir

    if job_ops.raw_statistics.enabled:
        log.info_rank("Running raw statistics...", comm=data.comm.comm_world)
        job_ops.raw_statistics.apply(data)
        log.info_rank(
            "Calculated raw statistics in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After raw statistics"
        job_ops.mem_count.apply(data)


def setup_filtered_statistics(operators):
    """Add commandline args and operators for filtered timestream statistics.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.Statistics(name="filtered_statistics", enabled=False))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.filtered_statistics.output_dir = otherargs.out_dir

    if job_ops.filtered_statistics.enabled:
        log.info_rank(
            "Running statistics on filtered data...", comm=data.comm.comm_world
        )
        job_ops.filtered_statistics.apply(data)
        log.info_rank(
            "Calculated filtered statistics in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After filtered statistics"
        job_ops.mem_count.apply(data)


def setup_hn_map(operators):
    """Add commandline args and operators for H_n map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.Hn(name="h_n", enabled=False))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.h_n.pixel_pointing = job.pixels_final
    job_ops.h_n.pixel_dist = job_ops.binner_final.pixel_dist
    job_ops.h_n.output_dir = otherargs.out_dir

    if job_ops.h_n.enabled:
        log.info_rank("Running h_n calculation...", comm=data.comm.comm_world)
        job_ops.h_n.apply(data)
        log.info_rank("Calculated h_n in", comm=data.comm.comm_world, timer=timer)
        job_ops.mem_count.prefix = "After h_n map"
        job_ops.mem_count.apply(data)


def setup_cadence_map(operators):
    """Add commandline args and operators for the cadence map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.CadenceMap(name="cadence_map", enabled=False))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.cadence_map.pixel_pointing = job.pixels_final
    job_ops.cadence_map.pixel_dist = job_ops.binner_final.pixel_dist
    job_ops.cadence_map.output_dir = otherargs.out_dir

    if job_ops.cadence_map.enabled:
        log.info_rank("Running cadence map...", comm=data.comm.comm_world)
        job_ops.cadence_map.apply(data)
        log.info_rank(
            "Calculated cadence map in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After cadence map"
        job_ops.mem_count.apply(data)


def setup_crosslinking_map(operators):
    """Add commandline args and operators for the crosslinking map.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.CrossLinking(name="crosslinking", enabled=False))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.crosslinking.pixel_pointing = job.pixels_final
    job_ops.crosslinking.pixel_dist = job_ops.binner_final.pixel_dist
    job_ops.crosslinking.output_dir = otherargs.out_dir

    if job_ops.crosslinking.enabled:
        log.info_rank("Running crosslinking map...", comm=data.comm.comm_world)
        job_ops.crosslinking.apply(data)
        log.info_rank(
            "Calculated crosslinking map in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After crosslinking map"
        job_ops.mem_count.apply(data)
