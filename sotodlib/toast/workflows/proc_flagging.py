# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Timestream flagging operations.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_simple_jumpcorrect(operators):
    """Add commandline args and operators for jump flagging.
    Args:
        operators (list):  The list of operators to extend.
    Returns:
        None
    """
    operators.append(toast.ops.SimpleJumpCorrect(name="simple_jumpcorrect", enabled=False))


@workflow_timer
def simple_jumpcorrect(job, otherargs, runargs, data):
    """Apply simple jump correction
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

    if job_ops.simple_jumpcorrect.enabled:
        job_ops.simple_jumpcorrect.apply(data)


def setup_simple_deglitch(operators):
    """Add commandline args and operators for glitch flagging.
    Args:
        operators (list):  The list of operators to extend.
    Returns:
        None
    """
    operators.append(toast.ops.SimpleDeglitch(name="simple_deglitch", enabled=False))


@workflow_timer
def simple_deglitch(job, otherargs, runargs, data):
    """Apply simple deglitching
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

    if job_ops.simple_deglitch.enabled:
        job_ops.simple_deglitch.apply(data)


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
            out_model="noise_cut_fit",
            enabled=True,
        )
    )
    operators.append(
        toast.ops.FlagNoiseFit(
            name="noise_cut_flag",
            sigma_NET=5.0,
            sigma_fknee=5.0,
            enabled=True,
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
    job_ops.noise_cut_flag.noise_model = job_ops.noise_cut_fit.out_model
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
            job_ops.noise_cut_fit.out_model,
        ]
    ).apply(data)


def setup_flag_diff_noise_outliers(operators):
    """Add commandline args and operators for flagging white noise outliers.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.SignalDiffNoiseModel(
            name="diff_noise_cut",
            noise_model="diff_noise_cut",
            enabled=False,
        )
    )
    operators.append(
        toast.ops.FlagNoiseFit(
            name="diff_noise_cut_flag",
            sigma_NET=5.0,
            enabled=True,
        )
    )


@workflow_timer
def flag_diff_noise_outliers(job, otherargs, runargs, data):
    """Flag detectors with outlier white noise properties.

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
    all_enabled = job_ops.diff_noise_cut.enabled and job_ops.diff_noise_cut_flag.enabled
    if not all_enabled:
        log.info_rank(
            "Noise-based cut of detectors disabled", comm=data.comm.comm_world
        )
        return

    # Estimate noise.
    log.info_rank(
        "  Running noise estimation on raw data...", comm=data.comm.comm_world
    )
    job_ops.diff_noise_cut.apply(data)
    log.info_rank(
        "  Estimated raw data noise in", comm=data.comm.comm_world, timer=timer
    )

    # Flag detector outliers
    job_ops.diff_noise_cut_flag.noise_model = job_ops.diff_noise_cut.noise_model
    log.info_rank(
        "  Running flagging of noise model outliers...", comm=data.comm.comm_world
    )
    job_ops.diff_noise_cut_flag.apply(data)
    log.info_rank(
        "  Flag raw noise model outliers in", comm=data.comm.comm_world, timer=timer
    )

    # Delete these temporary models
    toast.ops.Delete(
        meta=[
            job_ops.diff_noise_cut.noise_model,
        ]
    ).apply(data)


def setup_processing_mask(operators):
    """Add commandline args and operators for processing mask flagging.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.PixelsHealpix(
            name="processing_mask_pixels",
            pixels="pixels_processing_mask",
            enabled=False,
        )
    )
    operators.append(toast.ops.ScanHealpixMask(name="processing_mask", enabled=False))


@workflow_timer
def processing_mask(job, otherargs, runargs, data):
    """Raise data processing flags based on a mask.

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

    if job_ops.processing_mask.enabled:
        if job_ops.processing_mask_pixels.enabled:
            # We are using a custom pointing matrix
            job_ops.processing_mask_pixels.detector_pointing = (
                job_ops.det_pointing_radec
            )
            job_ops.processing_mask.pixel_dist = "processing_mask_pixel_dist"
            job_ops.processing_mask.pixel_pointing = job_ops.processing_mask_pixels
        else:
            # We are using the same pointing matrix as the mapmaking
            job_ops.processing_mask.pixel_dist = job_ops.binner.pixel_dist
            job_ops.processing_mask.pixel_pointing = job.pixels_solve
        job_ops.processing_mask.save_pointing = otherargs.full_pointing
        job_ops.processing_mask.apply(data)
