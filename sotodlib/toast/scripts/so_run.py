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

from pshmem import MPIBatch

import toast
from toast.timing import Timer
from toast.scripts.toast_run import main as toast_run_main

from ...core import Context


def get_context_tasks(
    comm, out_root, context_file, obs_file, dets_select, by_obs, by_wafer
):
    """Get the list of tasks, given the observations and selections.

    Args:
        comm (MPI.Comm):  The world communicator.
        out_root (str):  The top-level output directory.
        context_file (str):  The context file to use.
        obs_file (str):  The file of observation IDs to consider.
        dets_select (str):  The global (not task-specific) dets_select
            dictionary as a STRING (i.e. as it appears in CLI parsing).
        by_obs (bool):  If True, divide observation IDs into separate
            tasks.
        by_wafer (bool):  If True, divide each observation into per-wafer
            tasks.

    Returns:
        (list):  The list of tasks.

    """
    tasks = None
    if comm is None or comm.rank == 0:
        olist = list()
        with open(obs_file, "r") as f:
            for line in f:
                if re.match(r"^#.*", line) is None:
                    olist.append(line.rstrip())

        if dets_select is None:
            dets_select = dict()
        else:
            dets_select = eval(dets_select)

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
            "name": "all",
            "obs_ids": list(),
            "select": dets_select,
            "telescope": None,
            "outdir": out_root,
            "index": 0,
            "desc": "All observations",
        }
        itask = 0
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
                        "name": f"{obs_id}",
                        "obs_id": obs_id,
                        "telescope": telescope,
                        "n_samp": n_samp,
                        "stream_ids": stream_ids,
                        "wafer_slots": wafer_slots,
                        "select": dets_select,
                        "outdir": os.path.join(out_root, obs_id),
                        "index": itask,
                        "desc": f"{telescope}:{obs_id}",
                    }
                )
                itask += 1
            elif by_wafer:
                # Per wafer tasks
                for strm, waf in zip(stream_ids, wafer_slots):
                    dselect = dict(dets_select)
                    dselect["stream_id"] = [strm]
                    tasks.append(
                        {
                            "name": f"{obs_id}-{strm}",
                            "obs_id": obs_id,
                            "telescope": telescope,
                            "n_samp": n_samp,
                            "stream_id": strm,
                            "wafer_slot": waf,
                            "select": dselect,
                            "outdir": os.path.join(out_root, f"{obs_id}-{strm}"),
                            "index": itask,
                            "desc": f"{telescope}:{obs_id}:{strm}",
                        }
                    )
                    itask += 1
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


def get_sim_tasks(comm, out_root, schedule_file, telescope_file, by_obs, by_wafer):
    """Get the list of tasks, given the observations and selections.

    Args:
        comm (MPI.Comm):  The world communicator.
        out_root (str):  The top-level output directory.
        schedule_file (str):  The schedule file containing observations.
        telescope_file (str):  The synthetic instrument model.
        by_obs (bool):  If True, divide observation IDs into separate
            tasks.
        by_wafer (bool):  If True, divide each observation into per-wafer
            tasks.

    Returns:
        (list):  The list of tasks.

    """
    tasks = list()
    if comm is None or comm.rank == 0:
        telescope, _ = toast.io.load_instrument_file(telescope_file)
        tele_name = telescope.name
        full_detdata = telescope.focalplane.detector_data
        wafers = list(set(full_detdata["wafer_slot"]))

        # If we are running per-wafer, split the telescope file now.
        if by_wafer:
            waf_tele_files = dict()
            tele_wafer_dir = os.path.join(out_root, "wafer_telescopes")
            os.makedirs(tele_wafer_dir, exist_ok=True)
            for waf in wafers:
                wtele_file = os.path.join(tele_wafer_dir, f"telescope_{waf}.h5")
                rows = full_detdata["wafer_slot"] == waf
                new_detdata = full_detdata[rows]
                new_fp = toast.instrument.Focalplane(
                    detector_data=new_detdata,
                    field_of_view=telescope.focalplane.field_of_view,
                    sample_rate=telescope.focalplane.sample_rate,
                    thinfp=None,
                )
                waf_tele = toast.instrument.Telescope(
                    telescope.name,
                    uid=telescope.uid,
                    focalplane=new_fp,
                    site=telescope.site,
                )
                toast.io.save_instrument_file(wtele_file, waf_tele, None)
                waf_tele_files[waf] = wtele_file

        # Load the schedule so we can split it.
        full_schedule = toast.schedule.GroundSchedule()
        full_schedule.read(schedule_file)

        single_task = {
            "name": "all",
            "obs_ids": list(),
            "telescope": tele_name,
            "telescope_file": telescope_file,
            "schedule_file": schedule_file,
            "outdir": out_root,
            "index": 0,
            "desc": "All observations",
        }
        itask = 0
        for iscan, scan in enumerate(full_schedule.scans):
            obs_id = f"obs_{tele_name}_{int(scan.start.timestamp())}"
            # Create the per-observation schedule
            task_schedule = toast.schedule.GroundSchedule(
                scans=[scan],
                site_name=full_schedule.site_name,
                telescope_name=full_schedule.telescope_name,
                site_lat=full_schedule.site_lat,
                site_lon=full_schedule.site_lon,
                site_alt=full_schedule.site_alt,
            )

            if by_obs:
                # Building per-obs tasks
                task_dir = os.path.join(out_root, obs_id)
                os.makedirs(task_dir, exist_ok=True)
                task_schedule_file = os.path.join(task_dir, "schedule.txt")
                task_schedule.write(task_schedule_file)
                tasks.append(
                    {
                        "name": f"{obs_id}",
                        "obs_id": obs_id,
                        "telescope": tele_name,
                        "telescope_file": telescope_file,
                        "schedule_file": task_schedule_file,
                        "wafer_slots": wafers,
                        "outdir": task_dir,
                        "index": itask,
                        "desc": f"{tele_name}:{obs_id}",
                    }
                )
                itask += 1
            elif by_wafer:
                # Per wafer tasks
                for waf in wafers:
                    wtele_file = waf_tele_files[waf]
                    task_dir = os.path.join(out_root, f"{obs_id}-{waf}")
                    os.makedirs(task_dir, exist_ok=True)
                    task_schedule_file = os.path.join(task_dir, "schedule.ecsv")
                    task_schedule.write(task_schedule_file)
                    tasks.append(
                        {
                            "name": f"{obs_id}-{waf}",
                            "obs_id": obs_id,
                            "telescope": tele_name,
                            "telescope_file": wtele_file,
                            "schedule_file": task_schedule_file,
                            "wafer_slot": waf,
                            "outdir": task_dir,
                            "index": itask,
                            "desc": f"{tele_name}:{obs_id}:{waf}",
                        }
                    )
                    itask += 1
            else:
                single_task["obs_ids"].append(obs_id)

        if not (by_obs or by_wafer):
            tasks.append(single_task)

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
        (list):  The new list of commandline arguments to be passed to toast_run
            for this task.

    """
    new_args = list()
    subst = {
        "{run_dir}": task["outdir"],
        "{telescope}": task["telescope"],
    }
    if parsed.context_file is not None:
        # Real data case
        subst["{context_file}"] = parsed.context_file
        subst["{dets_select}"] = str(task["select"]).replace(" ", "")
        if parsed.task_per_obs or parsed.task_per_wafer:
            # Single obs
            subst["{samples}"] = str(task["n_samp"])
    else:
        # Synthetic data
        subst["{schedule_file}"] = task["schedule_file"]
        subst["{telescope_file}"] = task["telescope_file"]

    # Redirect the toast_run output to a logfile in the output directory.
    new_args = ["--out_dir", task["outdir"]]
    if not parsed.no_redirect:
        new_args.extend(["--out_log_name", "log"])

    if parsed.task_per_obs or parsed.task_per_wafer:
        # Single obs
        subst["{observations}"] = str([task["obs_id"]])
    else:
        # Multiple obs
        subst["{observations}"] = str(",".join(task["obs_ids"]))

    for targ in task_args:
        temp = str(targ)
        for sbkey, sbval in subst.items():
            temp = temp.replace(sbkey, sbval)
        new_args.append(temp)
    return new_args


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


def print_status(parsed):
    """Print status of tasks."""
    states = dict()
    for root, dirs, files in os.walk(parsed.out_root):
        for dir in dirs:
            task_dir = os.path.join(root, dir)
            state_file = os.path.join(task_dir, "state")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    task_state_str = f.readline().rstrip()
            else:
                task_state_str = "INVALID"
            if task_state_str not in states:
                states[task_state_str] = list()
            states[task_state_str].append(dir)
        break
    state_order = ["OPEN", "DONE", "RUNNING", "FAILED", "INVALID"]
    print(f"Tasks in {parsed.out_root}")
    print("------------------------------------------------------------", flush=True)
    for state_str in state_order:
        if state_str in states:
            total = len(states[state_str])
        else:
            total = 0
        print(f"{state_str:<8}: {total}", flush=True)
        if parsed.status_details is not None and parsed.status_details == state_str:
            # Print every task
            if state_str in states and len(states[state_str]) > 0:
                for tsk in states[state_str]:
                    print(f"  {tsk}", flush=True)
            else:
                print("  (no tasks)", flush=True)


def main(opts=None, comm=None):
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    rank = 0
    if comm is not None:
        rank = comm.rank

    parser = argparse.ArgumentParser(
        description="Run Simons Observatory TOAST workflows"
    )

    parser.add_argument(
        "--worker_size",
        required=False,
        type=int,
        default=None,
        help="The number of procs per worker",
    )

    parser.add_argument(
        "--out_root",
        required=True,
        type=str,
        default=None,
        help="The top-level output directory",
    )

    parser.add_argument(
        "--no_redirect",
        action="store_true",
        default=False,
        help="Disable worker stderr / stdout redirection",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Print task status and exit",
    )

    parser.add_argument(
        "--status_details",
        required=False,
        type=str,
        default=None,
        help="Print details of each task with this status (e.g. 'FAILED', 'RUNNING')",
    )

    cleanup_group = parser.add_mutually_exclusive_group(required=False)
    cleanup_group.add_argument(
        "--set_running_to_open",
        action="store_true",
        default=False,
        help="Mark RUNNING jobs as OPEN",
    )
    cleanup_group.add_argument(
        "--set_running_to_failed",
        action="store_true",
        default=False,
        help="Mark RUNNING jobs as FAILED",
    )

    task_group = parser.add_mutually_exclusive_group(required=False)
    task_group.add_argument(
        "--task_per_obs",
        action="store_true",
        default=False,
        help="Create one task per observation",
    )
    task_group.add_argument(
        "--task_per_wafer",
        action="store_true",
        default=False,
        help="Create one task per wafer-observation",
    )

    # Options for real data

    parser.add_argument(
        "--obs_file",
        required=False,
        type=str,
        default=None,
        help="File with observation IDs",
    )

    parser.add_argument(
        "--context_file",
        required=False,
        type=str,
        default=None,
        help="Context file to use",
    )

    parser.add_argument(
        "--dets_select",
        required=False,
        type=str,
        default=None,
        help="Det selection (as a string) common to all tasks",
    )

    # Options for pure synthetic data

    parser.add_argument(
        "--sim_telescope",
        required=False,
        type=str,
        default=None,
        help="Synthetic telescope file",
    )

    parser.add_argument(
        "--sim_schedule",
        required=False,
        type=str,
        default=None,
        help="Synthetic observing schedule",
    )

    # Parse just the args we are using in this wrapper
    args, remaining = parser.parse_known_args(args=opts)

    # If we are printing status, just do that and exit
    if args.status:
        # Cleanup any running states if needed
        cleanup_states(args)
        print_status(args)
        return

    if args.context_file is not None:
        msg = f"Working with data from context {args.context_file}"
        log.info_rank(msg, comm=comm)
        # Using real data loader
        if args.sim_telescope is not None or args.sim_schedule is not None:
            raise RuntimeError(
                "If context file specified, sim parameters should be unset"
            )
        if args.obs_file is None:
            raise RuntimeError("If using a context, must also specify obs_file")
        tasks = get_context_tasks(
            comm,
            args.out_root,
            args.context_file,
            args.obs_file,
            args.dets_select,
            args.task_per_obs,
            args.task_per_wafer,
        )
    else:
        msg = f"Working with synthetic data from schedule {args.sim_schedule}"
        log.info_rank(msg, comm=comm)
        if args.sim_telescope is None or args.sim_schedule is None:
            raise RuntimeError(
                "If using simulated observing, both telescope and schedule required"
            )
        tasks = get_sim_tasks(
            comm,
            args.out_root,
            args.sim_schedule,
            args.sim_telescope,
            args.task_per_obs,
            args.task_per_wafer,
        )

    # Cleanup any running states if needed
    cleanup_states(args)

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
    batch = MPIBatch(
        comm,
        worker_size,
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
            toast_run_main(opts=task_args, comm=batch.worker_comm)
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
