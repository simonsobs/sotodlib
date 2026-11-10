#!/usr/bin/env python3

# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
Run data I/O and conversion tests.

This script is designed to test more aspects of the bidirectional toast.Observation
to AxisManager conversion process than can be easily tested in the unit tests.

"""

import argparse
import os
import re
import sys

import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops


def create_toast_comm(comm, args):
    """Create a toast communicator.

    Args:
        comm (MPI.Comm):  The world communicator.
        args (namespace):  The parsed args.

    Returns:
        (toast.Comm):  The toast communicator.

    """
    if comm is None:
        toastcomm = toast.Comm(world=comm)
    else:
        if args.group_size is None:
            gsize = comm.size
        else:
            gsize = args.group_size
        if comm.size % gsize != 0:
            msg = f"Group size ({gsize}) does not divide evenly into the world"
            msg += f" communicator size ({comm.size}).  Using world size instead."
            print(msg, flush=True)
            gsize = comm.size
        toastcomm = toast.Comm(world=comm, groupsize=gsize)
    return toastcomm


def load_obs_ids(comm, path):
    """Load the list of observation IDs from a file

    Args:
        comm (MPI.Comm):  The world communicator.
        path (str):  The path to the file.

    Returns:
        (list):  The list of obs IDs.

    """
    olist = None
    if comm is None or comm.rank == 0:
        olist = []
        with open(path, "r") as f:
            for line in f:
                if re.match(r"^#.*", line) is None:
                    olist.append(line.rstrip())
    if comm is not None:
        olist = comm.bcast(olist, root=0)
    return olist


def create_context_data(comm, args):
    """Create data by loading from a context.

    Args:
        comm (MPI.Comm):  The world communicator.
        args (namespace):  The parsed args.

    Returns:
        (Data):  The data object

    """
    toastcomm = create_toast_comm(comm, args)
    obs_ids = load_obs_ids(comm, args.obs_file)

    if args.context_dets_select is None:
        dets_select = {}
    else:
        dets_select = eval(args.context_dets_select)

    data = toast.Data(comm=toastcomm)

    so_ops.LoadContext(
        context_file=args.context_file,
        observations=obs_ids,
        dets_select=dets_select,
    ).apply(data)

    return data


def create_hdf5_data(comm, args):
    """Create data from a TOAST HDF5 volume.

    Args:
        comm (MPI.Comm):  The world communicator.
        args (namespace):  The parsed args.

    Returns:
        (Data):  The data object

    """
    toastcomm = create_toast_comm(comm, args)
    obs_ids = load_obs_ids(comm, args.obs_file)

    data = toast.Data(comm=toastcomm)

    obs_sel = ",".join([f"'{x}'" for x in obs_ids])
    if args.hdf5_query is None:
        # Just select the sessions
        vselect = f"where session in ({obs_sel})"
    else:
        # Merge selection
        vselect = f"where session in ({obs_sel}) and {args.hdf5_query}"

    toast.ops.LoadHDF5(
        volume=args.hdf5_volume,
        volume_index=args.hdf5_index,
        det_select=args.hdf5_det_select,
        process_rows=toastcomm.group_size,
        volume_select=vselect,
    ).apply(data)

    return data


def create_sim_data(comm, args):
    """Create data from synthetic observing.

    Args:
        comm (MPI.Comm):  The world communicator.
        args (namespace):  The parsed args.

    Returns:
        (Data):  The data object

    """
    msg = "Simulated data cannot yet be read into real-data format"
    raise NotImplementedError(msg)


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
        "--group_size",
        required=False,
        type=int,
        default=None,
        help="The number of procs per group",
    )

    parser.add_argument(
        "--out_root",
        required=True,
        type=str,
        default=None,
        help="The top-level output directory",
    )

    parser.add_argument(
        "--obs_file",
        required=False,
        type=str,
        default=None,
        help="File with observation IDs",
    )

    # Context options

    parser.add_argument(
        "--context_file",
        required=False,
        type=str,
        default=None,
        help="Context file to use",
    )

    parser.add_argument(
        "--context_dets_select",
        required=False,
        type=str,
        default=None,
        help="Det selection (as a string) common to all tasks",
    )

    # HDF5 options

    parser.add_argument(
        "--hdf5_volume",
        required=False,
        type=str,
        default=None,
        help="HDF5 Toast volume",
    )

    parser.add_argument(
        "--hdf5_index",
        required=False,
        type=str,
        default="DEFAULT",
        help="HDF5 Toast volume",
    )

    parser.add_argument(
        "--hdf5_query",
        required=False,
        type=str,
        default=None,
        help="Additional query for selecting observations- merged with `obs_file`",
    )

    parser.add_argument(
        "--hdf5_det_select",
        required=False,
        type=str,
        default=None,
        help="Detector selection to apply when loading HDF5 data",
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

    args = parser.parse_args(args=opts)

    timer = toast.timing.Timer()
    timer.start()

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
        data = create_context_data(comm, args)
    elif args.hdf5_volume is not None:
        msg = f"Working with data from HDF5 volume {args.hdf5_volume}"
        log.info_rank(msg, comm=comm)
        data = create_hdf5_data(comm, args)
    elif args.sim_schedule is not None:
        msg = f"Working with synthetic data from schedule {args.sim_schedule}"
        log.info_rank(msg, comm=comm)
        if args.sim_telescope is None or args.sim_schedule is None:
            raise RuntimeError(
                "If using simulated observing, both telescope and schedule required"
            )
        data = create_sim_data(comm, args)
    else:
        msg = "You must specify exactly one of `context_file`, `hdf5_volume`"
        msg += " or `sim_schedule`"
        raise RuntimeError(msg)

    log.info_rank("Finished loading data", comm=comm, timer=timer)


    if comm is not None:
        comm.barrier()
    data.clear()
    del data


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(opts=None, comm=world)


if __name__ == "__main__":
    cli()
