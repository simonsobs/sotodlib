#!/usr/bin/env python3

# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs a sequence of SAT signal simulations for computing
the transfer function.

You can see the automatically generated command line options with:

    toast_so_sat_transfer --help

Or you can dump a config file with all the default values with:

    toast_so_sat_transfer --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

The observations used for each run are controlled by the configuration of the
loading operators being used.


"""

import argparse
import datetime
import os
import sys
import traceback

import numpy as np

from astropy import units as u

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
import toast.ops
from toast.observation import default_values as defaults

from toast.mpi import MPI, Comm

from .. import ops as so_ops
from .. import workflows as wrk

# Make sure pixell uses a reliable FFT engine
import pixell.fft

pixell.fft.engine = "fftw"


def load_data(job, otherargs, runargs, data, split_dets):
    log = toast.utils.Logger.get()
    job_ops = job.operators

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    job_ops.mem_count.prefix = "Before Data Load"
    job_ops.mem_count.apply(data)

    # Set the specific detectors to load for this split.
    # FIXME: this needs small tweak to the loading operator

    # Load data from all formats
    wrk.load_data_hdf5(job, otherargs, runargs, data)
    # wrk.load_data_context(job, otherargs, runargs, data)

    if len(data.obs) == 0:
        raise RuntimeError("No input data specified!")

    job_ops.mem_count.prefix = "After Data Load"
    job_ops.mem_count.apply(data)

    wrk.select_pointing(job, otherargs, runargs, data)
    wrk.simple_noise_models(job, otherargs, runargs, data)


def simulate_signal(job, otherargs, runargs, data, signal_file):
    # Clear detector timestreams
    toast.ops.Reset(detdata=[defaults.det_data])

    # Set the input map
    job.operators.scan_map.file = signal_file

    # Scan signal map
    wrk.simulate_sky_map_signal(job, otherargs, runargs, data)


def reduce_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()

    # Predefined splits for left and right going scans
    scan_splits = [
        defaults.scan_leftright_interval,
        defaults.scan_rightleft_interval,
    ]

    wrk.flag_noise_outliers(job, otherargs, runargs, data)
    wrk.filter_hwpss(job, otherargs, runargs, data)
    wrk.demodulate(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)
    wrk.flag_sso(job, otherargs, runargs, data)

    wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)

    wrk.mapmaker_ml(job, otherargs, runargs, data)
    
    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)
    wrk.filter_common_mode(job, otherargs, runargs, data)
    
    wrk.mapmaker(job, otherargs, runargs, data)
    wrk.mapmaker_filterbin(job, otherargs, runargs, data)

    mem = toast.utils.memreport(
        msg="(whole node)", comm=data.comm.comm_world, silent=True
    )
    log.info_rank(f"After reducing data:  {mem}", data.comm.comm_world)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_sat_transfer (total)")
    timer = toast.timing.Timer()
    timer.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # If the user has not told us to use multiple threads,
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
    log.info_rank(f"Start of the workflow:  {mem}", comm)

    # Argument parsing
    parser = argparse.ArgumentParser(description="SO SAT transfer function pipeline")

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_out",
        help="The output directory",
    )
    parser.add_argument(
        "--signal_map",
        required=True,
        default=None,
        type=str,
        nargs="+",
        help="The input signal realizations",
    )
    parser.add_argument(
        "--obsmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each observation separately.",
    )
    parser.add_argument(
        "--det_splits",
        required=False,
        default=None,
        type=str,
        nargs="+",
        help="Files with lists of detectors, one per split",
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    wrk.setup_load_data_hdf5(operators)
    # wrk.setup_load_data_context(operators)

    wrk.setup_pointing(operators)
    wrk.setup_simple_noise_models(operators)

    wrk.setup_simulate_sky_map_signal(operators)

    wrk.setup_flag_noise_outliers(operators)
    wrk.setup_filter_hwpss(operators)
    wrk.setup_demodulate(operators)
    wrk.setup_noise_estimation(operators)
    wrk.setup_flag_sso(operators)

    wrk.setup_deconvolve_detector_timeconstant(operators)
    wrk.setup_mapmaker_ml(operators)
    wrk.setup_filter_ground(operators)
    wrk.setup_filter_poly1d(operators)
    wrk.setup_filter_poly2d(operators)
    wrk.setup_filter_common_mode(operators)
    wrk.setup_mapmaker(operators, templates)
    wrk.setup_mapmaker_filterbin(operators)

    job, config, otherargs, runargs = wrk.setup_job(
        parser=parser, operators=operators, templates=templates
    )

    if not otherargs.full_pointing:
        msg = "The --full_pointing option is not enabled.  "
        msg += "You should consider setting this to compute the detector"
        msg += " pointing only once."
        log.warning(msg)

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(otherargs.out_dir):
            os.makedirs(otherargs.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(otherargs.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    # Load detector splits
    if otherargs.det_splits is None:
        det_splits = {"ALL": None}
    else:
        raise NotImplementedError("det splits not yet added to loader")
        det_splits = dict()
        for split_file in otherargs.det_splits:
            slist = list()
            with open(split_file, "r") as f:
                for line in f:
                    slist.append(line.rstrip())
            det_splits[split_file] = slist

    # If this is a dry run, exit
    if otherargs.dry_run:
        log.info_rank("Dry-run complete", comm=comm)
        return
    
    # Determine the process group size
    if runargs.group_size is not None:
        msg = f"Using user-specifed process group size of {runargs.group_size}"
        log.info_rank(msg, comm=comm)
        group_size = runargs.group_size
    else:
        if job.operators.mapmaker_ml.enabled:
            msg = f"ML mapmaker is enabled, forcing process group size to 1"
            log.info_rank(msg, comm=comm)
            group_size = 1
        else:
            msg = f"Using default process group size"
            log.info_rank(msg, comm=comm)
            if comm is None:
                group_size = 1
            else:
                group_size = comm.size

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Empty data container
    data = toast.Data(comm=toast_comm)

    # Process each detector split for all realizations.
    for split_file, split_dets in det_splits.items():
        # Clear out data
        data.clear()

        # Load the data.  This will load detector data too, but that
        # is fine since we set it to zero before each realization.
        load_data(job, otherargs, runargs, data, split_dets)

        # Loop over signal realizations

        for signal_file in otherargs.signal_map:
            # Use the base filename as the output directory
            sfile = os.path.basename(signal_file)
            root = os.path.splitext(sfile)[0]
            out_real_dir = os.path.join(otherargs.out_dir, root)

            # Create the output directory for this realization
            if comm is None or comm.rank == 0:
                if not os.path.isdir(out_real_dir):
                    os.makedirs(out_real_dir)
            if comm is not None:
                comm.barrier()

            # Simulate this realization
            simulate_signal(job, otherargs, runargs, data, signal_file)
            
            # Do the data reduction
            reduce_data(job, otherargs, runargs, data)
        
    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=comm)
    if data.comm.world_rank == 0:
        out = os.path.join(otherargs.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    log.info_rank("Workflow completed in", comm=comm, timer=timer)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
