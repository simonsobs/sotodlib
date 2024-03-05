#!/usr/bin/env python3

# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This workflow converts data between supported TOAST data formats.  It uses a single
process group to load data from:

- A set of HDF5 observation files

- A set of books

And then writes these out to:

- A set of HDF5 observation files

- A set of books

You can see the automatically generated command line options with:

    toast_so_convert --help

Or you can dump a config file with all the default values with:

    toast_so_convert --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import argparse
import datetime
import os

import numpy as np

from astropy import units as u

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
import toast.ops
from toast.mpi import MPI, Comm
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .. import workflows as wrk


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_convert (total)")
    timer = toast.timing.Timer()
    timer.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # This workflow is I/O dominated and does not benefit from
    # threads.  If the user has not told us to use multiple threads,
    # then just use one.

    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = 1
    log.info_rank(
        f"Executing workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads at {datetime.datetime.now()}",
        comm,
    )

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.debug_rank(f"Start of the workflow:  {mem}", comm)

    # Argument parsing and operator configuration

    parser = argparse.ArgumentParser(description="Convert SO data between formats")
    parser.add_argument(
        "--timing_root",
        required=False,
        default=None,
        help="Root filename (without extension) for logging timing info",
    )
    parser.add_argument(
        "--config_log",
        required=False,
        default=None,
        help="Write final config file to this path",
    )

    operators = list()
    wrk.setup_load_data_books(operators)
    wrk.setup_load_data_hdf5(operators)
    wrk.setup_save_data_books(operators)
    wrk.setup_save_data_hdf5(operators)

    job, config, otherargs, runargs = wrk.setup_job(parser=parser, operators=operators)

    # We enforce only one process group for this job
    group_size = procs
    if runargs.group_size is not None and runargs.group_size != group_size:
        raise RuntimeError(
            "This conversion script only works with a single process group"
        )

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Log the config that was actually used at runtime.
    if otherargs.config_log is not None:
        toast.config.dump_toml(otherargs.config_log, config, comm=comm)

    # If this is a dry run, exit
    if otherargs.dry_run:
        log.info_rank("Dry-run complete", comm=comm)
        return

    # Load data
    data = toast.Data(comm=toast_comm)
    wrk.load_data_books(job, otherargs, runargs, data)
    wrk.load_data_hdf5(job, otherargs, runargs, data)

    # Save data
    wrk.save_data_books(job, otherargs, runargs, data)
    wrk.save_data_hdf5(job, otherargs, runargs, data)

    # Collect optional timing information
    if otherargs.timing_root is not None:
        alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
        if toast_comm.world_rank == 0:
            toast.timing.dump(alltimers, otherargs.timing_root)

    log.info_rank("Conversion completed in", comm=comm, timer=timer)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
