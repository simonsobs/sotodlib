# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream flagging operations.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_flag_sso(operators):
    """Add commandline args and operators for SSO flagging.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.FlagSSO(name="flag_sso", enabled=False))


@workflow_timer
def flag_sso(job, otherargs, runargs, data):
    """Flag solarsystem objects.

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

    if job_ops.flag_sso.enabled:
        job_ops.flag_sso.detector_pointing = job_ops.det_pointing_azel
        job_ops.flag_sso.apply(data)


def setup_flag_noise_outliers(operators):
    """Add commandline args and operators for flagging detectors with bad noise.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.NoiseEstim(
            name="noise_cut",
            out_model="noise_cut",
            lagmax=1024,
            nbin_psd=64,
            nsum=1,
            naverage=64,
            enabled=False,
        )
    )
    operators.append(
        toast.ops.FitNoiseModel(
            name="noise_cut_fit",
        )
    )
    operators.append(
        toast.ops.FlagNoiseFit(
            name="noise_cut_flag",
            sigma_NET=5.0,
            sigma_fknee=5.0,
        )
    )


@workflow_timer
def flag_noise_outliers(job, otherargs, runargs, data):
    """Flag detectors with extremely bad noise properties.

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
    all_enabled = (
        job_ops.noise_cut.enabled
        and job_ops.noise_cut_fit.enabled
        and job_ops.noise_cut_flag.enabled
    )
    if not all_enabled:
        log.info_rank(
            "Noise-based cut of detectors disabled", comm=data.comm.comm_world
        )
        return

    # Estimate noise.
    log.info_rank(
        "  Running noise estimation on raw data...", comm=data.comm.comm_world
    )
    job_ops.noise_cut.apply(data)
    log.info_rank(
        "  Estimated raw data noise in", comm=data.comm.comm_world, timer=timer
    )

    # Create a fit to this noise model
    job_ops.noise_cut_fit.noise_model = job_ops.noise_cut.out_model
    log.info_rank(
        "  Running 1/f fit to estimated raw noise model", comm=data.comm.comm_world
    )
    job_ops.noise_cut_fit.apply(data)
    log.info_rank(
        "  Fit raw 1/f noise model in", comm=data.comm.comm_world, timer=timer
    )

    # Flag detector outliers
    job_ops.noise_cut_flag.noise_model = job_ops.noise_cut.out_model
    log.info_rank(
        "  Running flagging of noise model outliers...", comm=data.comm.comm_world
    )
    job_ops.noise_cut_flag.apply(data)
    log.info_rank(
        "  Flag raw noise model outliers in", comm=data.comm.comm_world, timer=timer
    )

    # Delete these temporary models
    toast.ops.Delete(
        meta=[
            job_ops.noise_cut.out_model,
        ]
    ).apply(data)
