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
    # This is for estimating the white noise levels after demodulation
    operators.append(
        toast.ops.NoiseEstim(
            name="demod_noise_estim",
            out_model="noise_model",
            lagmax=1,
            nbin_psd=1,
            nsum=1,
            naverage=1,
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
    # Configured operators for this job
    job_ops = job.operators

    if job_ops.demodulate.enabled:
        # The Demodulation operator is special because it returns a
        # new TOAST data object
        job_ops.demodulate.stokes_weights = job_ops.weights_radec
        data = job_ops.demodulate.apply(data)
        demod_weights = toast.ops.StokesWeightsDemod()
        job_ops.weights_radec = demod_weights
        if hasattr(job_ops, "binner"):
            job_ops.binner.stokes_weights = demod_weights
        if hasattr(job_ops, "binner_final"):
            job_ops.binner_final.stokes_weights = demod_weights
        job_ops.demod_noise_estim.apply(data)

