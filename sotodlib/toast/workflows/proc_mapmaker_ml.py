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


def setup_mapmaker_ml(operators):
    """Add commandline args and operators for the S.O. maximum likelihood mapmaker.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.MLMapmaker(name="mlmapmaker", enabled=False, comps="TQU"))


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
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    job_ops.mlmapmaker.out_dir = otherargs.out_dir

    if job_ops.mlmapmaker.enabled:
        log.info_rank("Running ML mapmaker...", comm=data.comm.comm_world)
        job_ops.mlmapmaker.apply(data)
        log.info_rank(
            "Finished ML map-making in", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After ML mapmaker"
        job_ops.mem_count.apply(data)
