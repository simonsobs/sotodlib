# Copyright (c) 2023-2023 Simons Observatory.
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

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.demodulate.enabled:
        # The pre-demodulation noise model to use
        if job_ops.noise_estim.enabled and job_ops.noise_estim_fit.enabled:
            # We have a noise estimate
            log.info_rank("Demodulation using estimated noise model", comm=data.comm.comm_world)
            noise_model = job_ops.noise_estim_fit.out_model
        else:
            have_noise = True
            for ob in data.obs:
                if "noise_model" not in ob:
                    have_noise = False
            if have_noise:
                log.info_rank(
                    "Demodulation using noise model from data files", comm=data.comm.comm_world
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
        demod_weights = toast.ops.StokesWeightsDemod()
        job_ops.weights_radec = demod_weights
        if hasattr(job_ops, "binner"):
            job_ops.binner.stokes_weights = demod_weights
        if hasattr(job_ops, "binner_final"):
            job_ops.binner_final.stokes_weights = demod_weights
        return new_data
    else:
        return data
