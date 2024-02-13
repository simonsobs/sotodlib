# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Interval operations.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_az_intervals(operators):
    """Add commandline args and operators for building Azimuth intervals.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.AzimuthIntervals(name="az_intervals", enabled=False))


@workflow_timer
def create_az_intervals(job, otherargs, runargs, data):
    """Pass through pointing and create Az intervals

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

    if job_ops.az_intervals.enabled:
        job_ops.az_intervals.apply(data)
