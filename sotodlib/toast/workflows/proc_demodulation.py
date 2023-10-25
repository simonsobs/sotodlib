# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream demodulation.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_demodulate(operators):
    """Add commandline args and operators for demodulation.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.Demodulate(name="demodulate", enabled=False))


def demodulate(job, otherargs, runargs, data):
    """Run timestream demodulation.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The input data container.

    Returns:
        (Data):  The new, demodulated data.

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.demodulate.enabled:
        # The Demodulation operator is special because it returns a
        # new TOAST data object
        job_ops.demodulate.stokes_weights = job_ops.weights_radec
        job_ops.demodulate.hwp_angle = job_ops.sim_ground.hwp_angle
        log.info_rank("Running demodulation...", comm=data.comm.comm_world)
        data = job_ops.demodulate.apply(data)
        log.info_rank("Demodulated in", comm=data.comm.comm_world, timer=timer)
        demod_weights = toast.ops.StokesWeightsDemod()
        job_ops.weights_radec = demod_weights
        if hasattr(job_ops, "binner"):
            job_ops.binner.stokes_weights = demod_weights
        if hasattr(job_ops, "binner_final"):
            job_ops.binner_final.stokes_weights = demod_weights

