#!/usr/bin/env python3

# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script batch processes the outputs of per-observation products and
computes statistics to enable the cutting of whole observations and / or
computing relative noise weights when combining those products.
"""

import argparse
import os
import sys
import traceback

from pshmem import MPIBatch

import toast
from toast.timing import Timer
from toast.scripts.toast_map_stats import main as toast_map_stats_main


def get_tasks(
    comm, in_root, out_root, mapmaker_name, use_fits
):
    """Get the list of tasks, given the existing mapmaker outputs.

    Args:
        comm (MPI.Comm):  The world communicator.
        in_root (str):  The top-level directory of mapmaking task outputs.
        out_root (str):  The top-level output directory.
        mapmaker_name (str):  The root filename of the maps (MapMaker in workflow)

    Returns:
        (list):  The list of tasks.

    """
    log = toast.utils.Logger.get()
    tasks = None
    if comm is None or comm.rank == 0:
        tasks = list()
        itask = 0
        for root, dirs, files in os.walk(in_root):
            for dir in dirs:
                if use_fits:
                    check_map = os.path.join(root, dir, f"{mapmaker_name}_map.fits")
                    check_invcov = os.path.join(root, dir, f"{mapmaker_name}_invcov.fits")
                    check_hits = os.path.join(root, dir, f"{mapmaker_name}_hits.fits")
                else:
                    check_map = os.path.join(root, dir, f"{mapmaker_name}_map.h5")
                    check_invcov = os.path.join(root, dir, f"{mapmaker_name}_invcov.h5")
                    check_hits = os.path.join(root, dir, f"{mapmaker_name}_hits.h5")
                if not os.path.exists(check_map):
                    msg = f"Map file {check_map} does not exist, skipping unfinished"
                    msg += " task"
                    log.debug(msg)
                    continue
                if not os.path.exists(check_hits):
                    msg = f"Hits file {check_hits} does not exist, skipping unfinished"
                    msg += " task"
                    log.debug(msg)
                    continue
                if not os.path.exists(check_invcov):
                    msg = f"Inverse covariance file {check_invcov} does not exist, "
                    msg += "skipping incomplete task"
                    log.info(msg)
                    continue
                tsk = dict()
                tsk["name"] = dir
                tsk["index"] = itask
                tsk["hits_file"] = check_hits
                tsk["map_file"] = check_map
                tsk["invcov_file"] = check_invcov
                tsk["out_dir"] = os.path.join(out_root, dir)
                tsk["desc"] = f"Map statistics for {tsk['name']}"
                tasks.append(tsk)
                itask += 1
            break
    if comm is not None:
        tasks = comm.bcast(tasks, root=0)
    return tasks


def get_task_args(task_args, parsed, task):
    """Substitute per-task parameters into CLI arguments.

    This takes the arguments intended for toast_run and makes substitutions
    based on the properties of the current task.

    Args:
        task_args (list):  The list of un-parsed / remaining commandline arguments
            after parsing the options known to this script.
        parsed (SimpleNamespace):  The already-parse options known to this script.
        task (dict):  The properties of this task.

    Returns:
        (list):  The new list of commandline arguments to be passed to toast_map_stats
            for this task.

    """
    task_args = [
        "--hits_file", task["hits_file"],
        "--map_file", task["map_file"],
        "--invcov_file", task["invcov_file"],
        "--out_dir", task["out_dir"],
        "--out_log_name", "log"
    ]
    return task_args


def cleanup_states(parsed):
    """Cleanup any stale states."""
    in_state = MPIBatch.state_to_string(MPIBatch.RUNNING)
    if parsed.set_running_to_open:
        out_state = MPIBatch.state_to_string(MPIBatch.OPEN)
    elif parsed.set_running_to_failed:
        out_state = MPIBatch.state_to_string(MPIBatch.FAILED)
    else:
        # Nothing to do
        return
    for root, dirs, files in os.walk(parsed.out_root):
        for dir in dirs:
            task_dir = os.path.join(root, dir)
            state_file = os.path.join(task_dir, "state")
            with open(state_file, "r") as f:
                state_str = f.readline().rstrip()
            if state_str == in_state:
                with open(state_file, "w") as f:
                    f.write(f"{out_state}\n")
        break


def main(opts=None, comm=None):
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    rank = 0
    if comm is not None:
        rank = comm.rank

    parser = argparse.ArgumentParser(
        description="Run map statistics on an output batch directory"
    )

    parser.add_argument(
        "--in_root",
        required=True,
        type=str,
        default=None,
        help="The top-level input directory of per-observation mapmaking products",
    )

    parser.add_argument(
        "--out_root",
        required=True,
        type=str,
        default=None,
        help="The top-level output directory with collected stats",
    )

    parser.add_argument(
        "--mapmaker_name",
        required=False,
        type=str,
        default="mapmaker",
        help="The base name of the mapmaker output files.",
    )

    parser.add_argument(
        "--use_fits",
        required=False,
        action="store_true",
        default=False,
        help="Output maps are in FITS, not HDF5 format.",
    )

    cleanup_group = parser.add_mutually_exclusive_group(required=False)
    cleanup_group.add_argument(
        "--set_running_to_open",
        action="store_true",
        default=False,
        help="Mark RUNNING jobs as OPEN before starting",
    )
    cleanup_group.add_argument(
        "--set_running_to_failed",
        action="store_true",
        default=False,
        help="Mark RUNNING jobs as FAILED before starting",
    )

    # Parse just the args we are using in this wrapper
    args, remaining = parser.parse_known_args(args=opts)

    tasks = get_tasks(
        comm,
        args.in_root,
        args.out_root,
        args.mapmaker_name,
        args.use_fits,
    )

    # Cleanup any running states if needed
    cleanup_states(args)

    # Create the batch setup.  The underlying tool we are calling only
    # uses one process, so we set the worker size to 1.
    n_task = len(tasks)
    batch = MPIBatch(
        comm,
        1,
        n_task,
        task_fs_root=args.out_root,
        task_fs_names=[x["name"] for x in tasks],
        debug=False,
    )

    msg = f"Using {batch.n_worker} workers"
    log.info_rank(msg, comm=comm)

    if batch.n_worker > n_task:
        msg = f"Number of tasks ({n_task}) less than number of "
        msg += f"workers ({batch.n_worker}), some will be idle"
        log.warning_rank(msg, comm=comm)

    # Run the tasks
    task_timer = Timer()
    task_timer.start()

    task_indx = batch.INVALID
    while task_indx is not None:
        # This call will also mark the task as running.
        task_indx = batch.next_task()
        if task_indx is None:
            msg = f"Worker {batch.worker} has no more tasks, waiting"
            log.info_rank(msg, comm=batch.worker_comm)
            continue

        tprops = tasks[task_indx]
        task_desc = tprops["desc"]

        # Perform any per-task substitutions on the args passed to toast_run
        task_args = get_task_args(remaining, args, tprops)

        msg = f"Worker {batch.worker} starting {task_desc}"
        log.info_rank(msg, comm=batch.worker_comm)

        try:
            log.debug_rank(
                f"Worker {batch.worker} task {task_indx}: toast_run_main({task_args})",
                comm=batch.worker_comm,
            )
            toast_map_stats_main(opts=task_args, comm=batch.worker_comm)
            batch.set_task_state(task_indx, batch.DONE)
        except Exception as e:
            # The task failed
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            lines = [f"Proc {rank}: {x}" for x in lines]
            msg = f"Worker {batch.worker} {task_desc} FAILED: "
            msg += "".join(lines)
            log.error_rank(msg, comm=batch.worker_comm)
            batch.set_task_state(task_indx, batch.FAILED)

        msg = f"Worker {batch.worker} finished {task_desc} in"
        log.info_rank(msg, comm=batch.worker_comm, timer=task_timer)

    if comm is not None:
        comm.barrier()


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(opts=None, comm=world)


if __name__ == "__main__":
    cli()
