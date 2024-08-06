# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream demodulation.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .job import workflow_timer


def setup_demodulate(operators):
    """Add commandline args and operators for demodulation.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.Demodulate(
            name="demodulate",
            hwp_angle=defaults.hwp_angle,
            enabled=False,
        )
    )
    operators.append(
        toast.ops.NoiseEstim(
            name="demod_noise_estim",
            out_model="demod_noise",
            lagmax=1024,
            nbin_psd=64,
            nsum=1,
            naverage=64,
            enabled=False,
        )
    )
    # Fitting is only performed if demod_noise_estim is enabled
    operators.append(
        toast.ops.FitNoiseModel(
            name="demod_noise_estim_fit",
            out_model="demod_noise_fit",
            enabled=True,
        )
    )


@workflow_timer
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

    if not job_ops.demodulate.enabled:
        # Short-circuit back
        return data

    # The pre-demodulation noise model to use
    if job_ops.noise_estim.enabled and job_ops.noise_estim_fit.enabled:
        # We have a noise estimate
        log.info_rank(
            "Demodulation using estimated noise model",
            comm=data.comm.comm_world,
        )
        noise_model = job_ops.noise_estim_fit.out_model
    else:
        have_noise = True
        for ob in data.obs:
            if "noise_model" not in ob:
                have_noise = False
        if have_noise:
            log.info_rank(
                "Demodulation using noise model from data files",
                comm=data.comm.comm_world
            )
            noise_model = "noise_model"
        else:
            # Use the synthetic elevation weighted model
            noise_model = "elevation_model"

    # The Demodulation operator is special because it returns a
    # new TOAST data object
    job_ops.demodulate.stokes_weights = job_ops.weights_radec
    job_ops.demodulate.noise_model = noise_model
    new_data = job_ops.demodulate.apply(data)
    log.info_rank(
        "  Finished demodulation in",
        comm=data.comm.comm_world,
        timer=timer,
    )
    demod_weights = toast.ops.StokesWeightsDemod()
    job_ops.weights_radec = demod_weights
    if hasattr(job_ops, "binner"):
        job_ops.binner.stokes_weights = demod_weights
    if hasattr(job_ops, "binner_final"):
        job_ops.binner_final.stokes_weights = demod_weights

    if job_ops.demod_noise_estim.enabled:
        # Estimate the (mostly white) noise on the demodulated data
        log.info_rank(
            "  Building demodulated noise estimate...",
            comm=data.comm.comm_world,
        )
        job_ops.demod_noise_estim.apply(new_data)
        log.info_rank(
            "  Finished demodulated noise estimate in",
            comm=data.comm.comm_world,
            timer=timer,
        )

    # Only run noise fitting if estimation is enabled
    if job_ops.demod_noise_estim_fit.enabled:
        if not job_ops.demod_noise_estim.enabled:
            log.info_rank(
                "demod_noise_estim disabled, nothing to fit",
                comm=data.comm.comm_world,
            )
        else:
            # Create a fit to this noise model
            job_ops.demod_noise_estim_fit.noise_model = \
                job_ops.demod_noise_estim.out_model
            log.info_rank(
                "  Running fit to noise estimate...",
                comm=data.comm.comm_world,
            )
            job_ops.demod_noise_estim_fit.apply(new_data)
            log.info_rank(
                "  Fit 1/f noise model in",
                comm=data.comm.comm_world,
                timer=timer,
            )

    return new_data
