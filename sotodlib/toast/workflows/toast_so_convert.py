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

    toast_so_convert.py --help

Or you can dump a config file with all the default values with:

    toast_so_convert.py --default_toml config.toml

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


def parse_config(operators, templates, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert SO data between formats")

    # Arguments specific to this script:
    # (none yet)

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    return config, args, jobargs


def load_data(job, args, toast_comm):
    log = toast.utils.Logger.get()
    ops = job.operators

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Load all of our HDF5 data

    if ops.in_hdf5.enabled:
        ops.mem_count.prefix = "Before HDF5 data load"
        ops.mem_count.apply(data)

        log.info_rank(f"Loading {ops.in_hdf5.files}", comm=toast_comm.comm_world)
        ops.in_hdf5.apply(data)

        log.info_rank("Loaded HDF5 data in", comm=toast_comm.comm_world, timer=timer)
        ops.mem_count.prefix = "After loading HDF5 data"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Loading HDF5 data disabled", comm=toast_comm.comm_world)

    # Load all book data

    if ops.in_books.enabled:
        ops.mem_count.prefix = "Before Book data load"
        ops.mem_count.apply(data)

        ops.in_books.apply(data)

        log.info_rank("Loaded Book data in", comm=toast_comm.comm_world, timer=timer)
        ops.mem_count.prefix = "After loading Book data"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Loading Book data disabled", comm=toast_comm.comm_world)

    if len(data.obs) == 0:
        raise RuntimeError("No input data specified!")

    return data


def convert_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Write out HDF5 data

    if ops.out_hdf5.enabled:
        ops.mem_count.prefix = "Before HDF5 data write"
        ops.mem_count.apply(data)

        ops.out_hdf5.apply(data)

        log.info_rank("Saved HDF5 data in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After saving HDF5 data"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Saving HDF5 data disabled", comm=world_comm)

    # Write out book data

    if ops.out_books.enabled:
        ops.mem_count.prefix = "Before Book data write"
        ops.mem_count.apply(data)

        ops.out_books.apply(data)

        log.info_rank("Saved Book data in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After saving Book data"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Saving Book data disabled", comm=world_comm)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_convert (total)")
    timer0 = toast.timing.Timer()
    timer0.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

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

    operators = [
        toast.ops.LoadHDF5(name="in_hdf5"),
        so_ops.LoadBooks(name="in_books", enabled=False),
        toast.ops.SaveHDF5(name="out_hdf5", enabled=False),
        so_ops.SaveBooks(name="out_books"),
        toast.ops.MemoryCounter(name="mem_count", enabled=False),
    ]

    # Parse options
    config, args, jobargs = parse_config(operators, list(), comm)

    # Instantiate our operators
    job = toast.create_from_config(config)

    # Log the config that was actually used at runtime.
    if job.operators.out_books.enabled:
        logroot = f"{job.operators.out_books.book_dir}"
    else:
        logroot = f"{job.operators.out_hdf5.volume}"
    toast.config.dump_toml(f"{logroot}_config_log.toml", config, comm=comm)

    # We enforce only one process group for this job
    group_size = comm.size
    if jobargs.group_size != group_size:
        raise RuntimeError(
            "This conversion script only works with a single process group"
        )

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Load one or more observations
    data = load_data(job, args, toast_comm)

    # Convert the data
    convert_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
        toast.timing.dump(alltimers, f"{logroot}_timing")

    log.info_rank("Workflow completed in", comm=comm, timer=timer0)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
