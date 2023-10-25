# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated calibration error.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_simulate_calibration_error(operators):
    """Add commandline args and operators for simulating gain error.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.GainScrambler(name="gainscrambler", enabled=False))


def simulate_calibration_error(job, otherargs, runargs, data):
    """Simulate calibration errors.

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
        job_ops.gainscrambler.realization = otherargs.realization

    if job_ops.gainscrambler.enabled:
        log.info_rank("Running simulated gain errors...", comm=data.comm.comm_world)
        job_ops.gainscrambler.apply(data)
        log.info_rank(
            "Simulated gain errors in", comm=data.comm.comm_world, timer=timer
        )
