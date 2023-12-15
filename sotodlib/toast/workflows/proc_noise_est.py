# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Noise estimation for mapmaking.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_noise_estimation(operators):
    """Add commandline args and operators for estimating detector noise.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.NoiseEstim(
            name="noise_estim",
            out_model="noise_estim",
            lagmax=1024,
            nbin_psd=64,
            nsum=1,
            naverage=64,
            enabled=False,
        )
    )
    operators.append(
        toast.ops.FitNoiseModel(
            name="noise_estim_fit",
            out_model="noise_estim_fit",
        )
    )


@workflow_timer
def noise_estimation(job, otherargs, runargs, data):
    """Run noise estimation and create a best-fit 1/f model.

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

    # If any operators are disabled, just return
    all_enabled = job_ops.noise_estim.enabled and job_ops.noise_estim_fit.enabled
    if not all_enabled:
        log.info_rank("Noise estimation disabled", comm=data.comm.comm_world)
        return

    # Estimate noise.
    log.info_rank(
        "  Building noise estimate...", comm=data.comm.comm_world
    )
    job_ops.noise_estim.apply(data)
    log.info_rank(
        "  Finished noise estimate in", comm=data.comm.comm_world, timer=timer
    )

    # Create a fit to this noise model
    job_ops.noise_estim_fit.noise_model = job_ops.noise_estim.out_model
    log.info_rank(
        "  Running fit to noise estimate...", comm=data.comm.comm_world
    )
    job_ops.noise_estim_fit.apply(data)
    log.info_rank(
        "  Fit 1/f noise model in", comm=data.comm.comm_world, timer=timer
    )
