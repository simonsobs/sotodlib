# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simple nominal noise models.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_simple_noise_models(operators):
    """Add commandline args and operators for simple noise models.

    These noise models use just nominal values from the focalplane
    and an elevation-weighting of the NET.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.DefaultNoiseModel(name="default_model", noise_model="noise_model")
    )
    operators.append(
        toast.ops.ElevationNoise(name="elevation_model", out_model="noise_model")
    )
    operators.append(toast.ops.CommonModeNoise(name="common_mode_noise", enabled=False))


def simple_noise_models(job, otherargs, runargs, data):
    """Generate trivial noise models.

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

    # Default model from focalplane nominal values
    if job_ops.default_model.enabled:
        log.info_rank("Running nominal noise model...", comm=data.comm.comm_world)
        job_ops.default_model.apply(data)
        log.info_rank(
            "Created default noise model in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After default noise model"
        job_ops.mem_count.apply(data)

    # Simple elevation-weighted model
    if job_ops.elevation_model.enabled:
        job_ops.elevation_model.detector_pointing = job_ops.det_pointing_azel
        log.info_rank(
            "Running elevation-weighted noise model...", comm=data.comm.comm_world
        )
        job_ops.elevation_model.apply(data)
        log.info_rank(
            "Created elevation noise model in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After elevation noise model"
        job_ops.mem_count.apply(data)

    # Add common noise modes
    if job_ops.common_mode_noise.enabled:
        log.info_rank("Running common mode noise model...", comm=data.comm.comm_world)
        job_ops.common_mode_noise.apply(data)
        log.info_rank(
            "Added common mode noise model in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After common mode noise model"
        job_ops.mem_count.apply(data)
