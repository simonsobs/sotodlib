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


def select_current_noise_model(job, otherargs, runargs, data):
    """Select the current active noise model.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        (str):  The name of the noise model, or None if a model has
            not yet been created.

    """
    log = toast.utils.Logger.get()

    # Configured operators for this job
    job_ops = job.operators

    noise_model = None
    if (
        hasattr(job_ops, "demodulate")
        and job_ops.demodulate.enabled
        and job_ops.demod_noise_estim.enabled
        and job_ops.demod_noise_estim_fit.enabled
    ):
        # We will use the noise estimate made after demodulation
        log.info_rank("  Using demodulated noise model", comm=data.comm.comm_world)
        noise_model = job_ops.demod_noise_estim_fit.out_model
    elif hasattr(job_ops, "diff_noise_estim") and job_ops.diff_noise_estim.enabled:
        # We have a signal-diff noise estimate
        log.info_rank("  Using signal diff noise model", comm=data.comm.comm_world)
        noise_model = job_ops.diff_noise_estim.noise_model
    elif (
        hasattr(job_ops, "noise_estim")
        and job_ops.noise_estim.enabled
        and job_ops.noise_estim_fit.enabled
    ):
        # We have a noise estimate
        log.info_rank("  Using estimated noise model", comm=data.comm.comm_world)
        noise_model = job_ops.noise_estim_fit.out_model
    else:
        have_noise = True
        for ob in data.obs:
            if "noise_model" not in ob:
                have_noise = False
        if have_noise:
            log.info_rank(
                "  Using noise model from data files", comm=data.comm.comm_world
            )
            noise_model = "noise_model"
        if not have_noise:
            # No noise estimates and no external noise models in the data.
            # Do we have the simulated elevation-weighted nominal model?
            have_noise = True
            for ob in data.obs:
                if "elevation_model" not in ob:
                    have_noise = False
            if have_noise:
                log.info_rank(
                    "  Using nominal synthetic noise model", comm=data.comm.comm_world
                )
                noise_model = "elevation_model"
    return noise_model
