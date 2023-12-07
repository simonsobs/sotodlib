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
from .job import workflow_timer


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


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if otherargs.realization is not None:
        job_ops.convolve_time_constant.realization = otherargs.realization

    if job_ops.convolve_time_constant.enabled:
        job_ops.convolve_time_constant.apply(data)


def setup_simulate_detector_noise(operators):
    """Add commandline args and operators for simulating detector noise.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.SimNoise(name="sim_noise"))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if otherargs.realization is not None:
        job_ops.sim_noise.realization = otherargs.realization

    if job_ops.sim_noise.enabled:
        job_ops.sim_noise.apply(data)


def setup_simulate_readout_effects(operators):
    """Add commandline args and operators for simulating readout effects.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimReadout(name="sim_readout", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.sim_readout.enabled:
        job_ops.sim_readout.apply(data)


def setup_simulate_detector_yield(operators):
    """Add commandline args and operators for simulating detector yield.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.YieldCut(name="yield_cut", enabled=False))


@workflow_timer
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.yield_cut.enabled:
        job_ops.yield_cut.apply(data)


def setup_simulate_mumux_crosstalk(operators):
    """Add commandline args and operators for simulating nonlinear muMUX crosstalk.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimMuMUXCrosstalk(name="sim_mumux_crosstalk", enabled=False))


@workflow_timer
def simulate_mumux_crosstalk(job, otherargs, runargs, data):
    """Simulate nonlinear muMUX crosstalk.

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

    if job_ops.sim_mumux_crosstalk.enabled:
        job_ops.sim_mumux_crosstalk.apply(data)
