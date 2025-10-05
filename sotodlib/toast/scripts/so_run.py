#!/usr/bin/env python3

# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This workflow is a wrapper around the toast_run workflow.

This script is designed to execute toast_run on a batch of one or more
"tasks".  Each task can be a group of observations, a single observation,
or a single wafer of one observation.

For real data (or simulated timestreams starting from real data), tasks can
be specified in terms of obs_ids and detector selections used with a Context
to load data.

For purely synthetic observations, a schedule file and a telescope instrument
file should be specified.  These will be split in the same way as real data
into tasks.

"""

import argparse
import os
import re
import sys
import traceback

import numpy as np

from astropy import units as u

from pshmem import MPIBatch

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
import toast.ops
from toast.mpi import MPI, Comm
from toast.observation import default_values as defaults

from toast.scripts.toast_run import main as toast_run_main

from ...core import Context


def get_context_tasks(
    comm, out_root, context_file, obs_file, det_select, by_obs, by_wafer
):
    tasks = None
    if comm is None or comm.rank == 0:
        olist = list()
        with open(obs_file, "r") as f:
            for line in f:
                if re.match(r"^#.*", line) is None:
                    olist.append(line.rstrip())

        # Query the obsdb directly so that we only have one DB query,
        # rather than doing one query per observation.
        sort_by = ["obs_id"]
        if len(olist) > 0:
            query_str = "obs.obs_id IN ("
            for oid in olist:
                query_str += f"'{oid}',"
            query_str = query_str.rstrip(",")
            query_str += ")"
        else:
            query_str = "1"

        # Open the databases and query
        ctx = Context(context_file)
        query_result = ctx.obsdb.query(query_text=query_str, sort=sort_by)
        del ctx

        tasks = list()
        single_task = {
            "obs_ids": list(),
            "select": det_select,
            "telescope": None,
        }
        for row in query_result:
            obs_id = str(row["obs_id"])
            wafer_slots = list(row["wafer_slots_list"].split(","))
            stream_ids = list(row["stream_ids_list"].split(","))
            telescope = str(row["telescope"])
            n_samp = int(row["n_samples"])
            if by_obs:
                # Building per-obs tasks
                tasks.append(
                    {
                        "obs_id": obs_id,
                        "telescope": telescope,
                        "n_samp": n_samp,
                        "stream_ids": stream_ids,
                        "wafer_slots": wafer_slots,
                        "select": det_select,
                        "outdir": os.path.join(out_root, obs_id),
                    }
                )
            elif by_wafer:
                # Per wafer tasks
                for strm, waf in zip(stream_ids, wafer_slots):
                    tasks.append(
                        {
                            "obs_id": obs_id,
                            "telescope": telescope,
                            "n_samp": n_samp,
                            "stream_id": strm,
                            "wafer_slot": waf,
                            "select": det_select,
                            "outdir": os.path.join(out_root, f"{obs_id}-{strm}"),
                        }
                    )
            else:
                # One task
                if single_task["telescope"] is None:
                    single_task["telescope"] = telescope
                single_task["obs_ids"].append(obs_id)

        if not (by_obs or by_wafer):
            tasks.append(single_task)
    if comm is not None:
        tasks = comm.bcast(tasks, root=0)
    return tasks


def get_sim_tasks(comm, schedule_file, telescope_file):
    return None


def get_task_args(task_args, parsed, task):
    """Substitute per-task parameters into CLI arguments.

    This takes the arguments intended for toast_run and makes substitutions
    based on the properties of the current task.

    """
    subst = {
        "{out_dir}": task["outdir"],
        "{output_dir}": task["outdir"],
        "{context_file}": parsed.context_file,
        "{det_select}": parsed.det_select,
    }
    for targ in task_args:
        for sbkey, sbval in subst.items():



def main():
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    parser = argparse.ArgumentParser(
        description="Run Simons Observatory TOAST workflows"
    )

    parser.add_argument(
        "--worker_size", required=False, default=None, help="The number of procs per worker"
    )

    parser.add_argument(
        "--out_root", required=True, default=None, help="The top-level output directory"
    )

    parser.add_argument(
        "--guard_file",
        required=True,
        default=None,
        help="Per-task filename whose presence indicates task completion",
    )

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument(
        "--task_per_obs",
        action="store_true",
        default=False,
        help="Create one task per observation",
    )
    parser_group.add_argument(
        "--task_per_wafer",
        action="store_true",
        default=False,
        help="Create one task per wafer-observation",
    )

    # Options for real data

    parser.add_argument(
        "--obs_file", required=False, default=None, help="File with observation IDs"
    )

    parser.add_argument(
        "--context_file", required=False, default=None, help="Context file to use"
    )

    parser.add_argument(
        "--det_select",
        required=False,
        default=None,
        help="Det selection (as a string) common to all tasks",
    )

    # Options for pure synthetic data

    parser.add_argument(
        "--sim_telescope", required=False, default=None, help="Synthetic telescope file"
    )

    parser.add_argument(
        "--sim_schedule",
        required=False,
        default=None,
        help="Synthetic observing schedule",
    )

    # Parse just the args we are using in this wrapper
    opts = sys.argv[1:]
    args, remaining = parser.parse_known_args(args=opts)

    if args.context_file is not None:
        msg = f"Working with real data from context {args.context_file}"
        log.info_rank(msg, comm=comm)
        # Using real data loader
        if args.sim_telescope is not None or args.sim_schedule is not None:
            raise RuntimeError("If context file specified, sim parameters should be unset")
        if args.obs_file is None:
            raise RuntimeError("If using a context, must also specify obs_file")
        tasks = get_context_tasks(
            comm,
            args.out_root,
            args.context_file,
            args.obs_file,
            args.det_select,
            args.task_per_obs,
            args.task_per_wafer,
        )
    else:
        msg = f"Working with synthetic data from schedule {args.sim_schedule}"
        log.info_rank(msg, comm=comm)
        if args.sim_telescope is None or args.sim_schedule is None:
            raise RuntimeError("If using simulated observing, both telescope and schedule required")
        raise NotImplementedError("Sim case not yet implemented")
        #tasks = get_sim_tasks()

    # Make any substitutions into the remaining commandline arguments based

    if args.worker_size is None:
        if comm is None:
            worker_size = 1
        else:
            worker_size = comm.size
        msg = f"Using default worker_size {worker_size}"
        log.info_rank(msg, comm=comm)
    else:
        worker_size = args.worker_size
        msg = f"Using specified worker_size {worker_size}"
        log.info_rank(msg, comm=comm)

    # Create the batch setup
    n_task = len(tasks)
    batch = MPIBatch(comm, worker_size, n_task, debug=True)

    msg = f"Using {batch.n_worker} workers"
    log.info_rank(msg, comm=comm)

    if batch.n_worker > n_task:
        msg = f"Number of tasks ({n_task}) less than number of "
        msg += f"workers ({batch.n_worker}), some will be idle"
        log.warning_rank(msg, comm=comm)

    # Run the tasks

    task_indx = batch.INVALID
    while task_indx is not None:
        task_indx = batch.next_task()
        if task_indx is None:
            msg = f"Worker {batch.worker} has no more tasks, waiting"
            log.info_rank(msg, comm=batch.worker_comm)
            continue

        tprops = tasks[task_indx]

        if args.task_per_obs:
            task_desc = f"{tprops['telescope']}:{tprops['obs_id']}"
        elif args.task_per_wafer:
            task_desc = f"{tprops['telescope']}:{tprops['obs_id']}"
            task_desc += f":{tprops['stream_id']}"
        else:
            task_desc = "single task with all observations"

        task_args = list(remaining)

        get_task_args(task_args, args, tprops)

        os.makedirs(tprops["outdir"], exist_ok=True)

        msg = f"Worker {batch.worker} starting {task_desc}"
        log.info_rank(msg, comm=batch.worker_comm)

        try:
            toast_run_main(opts=task_args)
        except Exception as e:
            # The task failed
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            lines = [f"Proc {rank}: {x}" for x in lines]
            msg = f"Worker {batch.worker} {task_desc} FAILED: "
            msg += "".join(lines)
            log.error_rank(msg, comm=batch.worker_comm)
            if batch.worker_rank == 0:
                batch.set_task_state(task_indx, batch.FAILED)

        if batch.worker_rank == 0:
            batch.set_task_state(task_indx, batch.DONE)

    if comm is not None:
        comm.barrier()


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
