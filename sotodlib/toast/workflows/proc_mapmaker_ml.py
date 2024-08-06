# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simultaneous filtering and binned mapmaking.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .job import workflow_timer


def setup_mapmaker_ml(operators):
    """Add commandline args and operators for the S.O. maximum likelihood mapmaker.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.MLMapmaker(name="mlmapmaker", enabled=False, comps="TQU"))


@workflow_timer
def mapmaker_ml(job, otherargs, runargs, data):
    """Run the S.O. maximum likelihood mapmaker.

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

    if job_ops.mlmapmaker.enabled:
        if data.comm.group_size != 1:
            msg = "The ML mapmaker requires the process group"
            msg += " size to be exactly one, since it uses"
            msg += " threads for parallelism."
            raise RuntimeError(msg)
        job_ops.mlmapmaker.out_dir = otherargs.out_dir
        job_ops.mlmapmaker.apply(data)

