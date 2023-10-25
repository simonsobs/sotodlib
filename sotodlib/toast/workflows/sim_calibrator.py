# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated detector response to calibrators.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_simulate_wiregrid_signal(operators):
    """Add commandline args and operators for simulating wiregrid.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        so_ops.SimWireGrid(name="sim_wiregrid", enabled=False)
    )


def simulate_wiregrid_signal(job, otherargs, runargs, data):
    """Simulate wire grid signal.

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

    if job_ops.sim_wiregrid.enabled:
        job_ops.sim_wiregrid.detector_pointing = job_ops.det_pointing_azel
        job_ops.sim_wiregrid.detector_weights = job_ops.weights_azel
        log.info_rank("Running wiregrid simulation...", comm=data.comm.comm_world)
        job_ops.sim_wiregrid.apply(data)
        log.info_rank("Simulated wiregrid in", comm=data.comm.comm_world, timer=timer)


def setup_simulate_stimulator_signal(operators):
    """Add commandline args and operators for simulating the stimulator.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        so_ops.SimStimulator(name="sim_stimulator", enabled=False)
    )


def simulate_stimulator_signal(job, otherargs, runargs, data):
    """Simulate stimulator signal.

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

    if job_ops.sim_stimulator.enabled:
        log.info_rank("Running stimulator simulation...", comm=data.comm.comm_world)
        job_ops.sim_stimulator.apply(data)
        log.info_rank("Simulated stimulator in", comm=data.comm.comm_world, timer=timer)
