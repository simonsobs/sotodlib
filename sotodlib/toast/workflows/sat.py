# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""SAT-specific operations.
"""

import os
import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer
from .proc_mapmaker import mapmaker_select_noise_and_binner, mapmaker_run


def setup_splits(operators):
    """Add commandline args and operators for SAT mapmaking splits.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.Splits(name="splits", enabled=False))


@workflow_timer
def splits(job, otherargs, runargs, data):
    """Apply mapmaking splits.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    job_ops = job.operators
    splits = job.operators.splits

    if splits.enabled:
        if not job_ops.mapmaker.enabled:
            raise RuntimeError(f"Cannot run Splits without MapMaker")
        mapmaker_select_noise_and_binner(job, otherargs, runargs, data)
        splits.mapmaker = job_ops.mapmaker
        splits.output_dir = otherargs.out_dir
        mapmaker_run(job, otherargs, runargs, data, splits)
