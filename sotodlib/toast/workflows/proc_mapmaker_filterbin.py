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


def setup_mapmaker_filterbin(operators):
    """Add commandline args, operators, and templates for TOAST FilterBin mapmaker.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.FilterBin(name="filterbin", enabled=False))


@workflow_timer
def mapmaker_filterbin(job, otherargs, runargs, data):
    """Run the TOAST FilterBin mapmaker.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.filterbin.enabled:
        job_ops.filterbin.binning = job_ops.binner_final
        job_ops.filterbin.det_data = job_ops.sim_noise.det_data
        job_ops.filterbin.output_dir = otherargs.out_dir
        if otherargs.obsmaps:
            # Map each observation separately
            timer_obs = toast.timing.Timer()
            timer_obs.start()
            group = data.comm.group
            orig_name_filterbin = job_ops.filterbin.name
            orig_comm = data.comm
            new_comm = toast.mpi.Comm(world=data.comm.comm_group)
            for iobs, obs in enumerate(data.obs):
                log.info_rank(
                    f"{group} : mapping observation {iobs + 1} / {len(data.obs)}.",
                    comm=new_comm.comm_world,
                )
                # Data object that only covers one observation
                obs_data = data.select(obs_uid=obs.uid)
                # Replace comm_world with the group communicator
                obs_data._comm = new_comm
                job_ops.filterbin.name = f"{orig_name_filterbin}_{obs.name}"
                job_ops.filterbin.reset_pix_dist = True
                job_ops.filterbin.apply(obs_data)
                log.info_rank(
                    f"{group} : Filter+binned {obs.name} in",
                    comm=new_comm.comm_world,
                    timer=timer_obs,
                )
            log.info_rank(
                f"{group} : Done mapping {len(data.obs)} observations.",
                comm=new_comm.comm_world,
            )
            data._comm = orig_comm
        else:
            job_ops.filterbin.apply(data)
