# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools specific to ACT processing.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_act_responsivity_sign(operators):
    """Add commandline args and operators for ACT responsivity sign flip.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.ActSign(name="act_sign", enabled=False))


@workflow_timer
def act_responsivity_sign(job, otherargs, runargs, data):
    """Apply sign to ACT timestreams.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    job_ops = job.operators

    if job_ops.act_sign.enabled:
        job_ops.act_sign.apply(data)

