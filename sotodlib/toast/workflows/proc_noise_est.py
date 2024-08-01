# Copyright (c) 2023-2024 Simons Observatory.
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
    # Fitting is only performed if noise_estim is enabled
    operators.append(
        toast.ops.FitNoiseModel(
            name="noise_estim_fit",
            out_model="noise_estim_fit",
            enabled=True,
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

    if job_ops.noise_estim.enabled:
        # Estimate noise.
        log.info_rank("  Building noise estimate...", comm=data.comm.comm_world)
        job_ops.noise_estim.apply(data)
        log.info_rank(
            "  Finished noise estimate in",
            comm=data.comm.comm_world,
            timer=timer,
        )

    # Only run noise fitting if estimation is enabled
    if job_ops.noise_estim_fit.enabled:
        if not job_ops.noise_estim.enabled:
            log.info_rank(
                "Noise_estim disabled, nothing to fit",
                comm=data.comm.comm_world,
            )
        else:
            # Create a fit to this noise model
            job_ops.noise_estim_fit.noise_model = job_ops.noise_estim.out_model
            log.info_rank(
                "  Running fit to noise estimate...", comm=data.comm.comm_world
            )
            job_ops.noise_estim_fit.apply(data)
            log.info_rank(
                "  Fit 1/f noise model in",
                comm=data.comm.comm_world,
                timer=timer,
            )


def setup_diff_noise_estimation(operators):
    """Add commandline args and operators for signal difference noise estimation.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.SignalDiffNoiseModel(
            name="diff_noise_estim",
            noise_model="diff_noise_estim",
            enabled=False,
        )
    )


@workflow_timer
def diff_noise_estimation(job, otherargs, runargs, data):
    """Estimate a simple white noise model for every detector.

    This uses sample differences to estimate the white noise properties.

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

    if job_ops.diff_noise_estim.enabled:
        job_ops.diff_noise_estim.apply(data)
