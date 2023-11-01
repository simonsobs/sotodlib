# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Mapmaking with the MADAM destriper.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .job import workflow_timer


def setup_mapmaker_madam(operators):
    """Add commandline args and operators for the MADAM mapmaker.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    if toast.ops.madam.available():
        operators.append(toast.ops.Madam(name="madam", enabled=False))


@workflow_timer
def mapmaker_madam(job, otherargs, runargs, data):
    """Run the MADAM mapmaker.

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

    if toast.ops.madam.available() and job_ops.madam.enabled:
        job_ops.madam.params = toast.ops.madam_params_from_mapmaker(job_ops.mapmaker)
        job_ops.madam.pixel_pointing = job.pixels_final
        job_ops.madam.stokes_weights = job.weights_final
        job_ops.madam.apply(data)
