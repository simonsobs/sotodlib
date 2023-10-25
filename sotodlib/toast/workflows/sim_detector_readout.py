# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated detector and readout effects.

This includes things fundamental to the detector and readout chain, including
timeconstant, noise, tuning yield, and readout systematics.

"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_simulate_detector_timeconstant(operators):
    """Add commandline args and operators for timeconstant convolution.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.TimeConstant(
            name="convolve_time_constant", deconvolve=False, enabled=False
        )
    )


def simulate_detector_timeconstant(job, otherargs, runargs, data):
    """Simulate the effects of detector timeconstants.

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

    if otherargs.realization is not None:
        job_ops.convolve_time_constant.realization = otherargs.realization

    if job_ops.convolve_time_constant.enabled:
        log.info_rank("Running time constant convolution...", comm=data.comm.comm_world)
        job_ops.convolve_time_constant.apply(data)
        log.info_rank(
            "Convolved time constant in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After applying time constant"
        job_ops.mem_count.apply(data)


def setup_simulate_detector_noise(operators):
    """Add commandline args and operators for simulating detector noise.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.SimNoise(name="sim_noise"))


def simulate_detector_noise(job, otherargs, runargs, data):
    """Simulate the intrinsic detector and readout noise.

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

    if otherargs.realization is not None:
        job_ops.sim_noise.realization = otherargs.realization

    if job_ops.sim_noise.enabled:
        log.info_rank("Running detector noise simulation...", comm=data.comm.comm_world)
        job_ops.sim_noise.apply(data)
        log.info_rank(
            "Simulated detector noise in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After simulating noise"
        job_ops.mem_count.apply(data)


def setup_simulate_readout_effects(operators):
    """Add commandline args and operators for simulating readout effects.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimReadout(name="sim_readout", enabled=False))


def simulate_readout_effects(job, otherargs, runargs, data):
    """Simulate various readout effects.

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

    if job_ops.sim_readout.enabled:
        log.info_rank(
            "Running readout systematics simulation", comm=data.comm.comm_world
        )
        job_ops.sim_readout.apply(data)
        log.info_rank(
            "Simulated readout systematics in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After simulating readout systematics"
        job_ops.mem_count.apply(data)


def setup_simulate_detector_yield(operators):
    """Add commandline args and operators for simulating detector yield.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.YieldCut(name="yield_cut", enabled=False))


def simulate_detector_yield(job, otherargs, runargs, data):
    """Simulate detector yield.

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

    if job_ops.yield_cut.enabled:
        log.info_rank("Running simulated yield cut...", comm=data.comm.comm_world)
        job_ops.yield_cut.apply(data)
        log.info_rank(
            "Applied yield flags in", comm=data.comm.comm_world, timer=timer
        )
