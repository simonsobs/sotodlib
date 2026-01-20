#!/usr/bin/env python3

# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs a transfer function simulation.

You can see the automatically generated command line options with:

    toast_so_transfer --help

Or you can dump a config file with all the default values with:

    toast_so_transfer --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import argparse
import datetime
import os
import sys
import traceback

import numpy as np
import yaml

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

# Make sure pixell uses a reliable FFT engine
import pixell.fft

pixell.fft.engine = "fftw"


def reduce_input(job, otherargs, runargs, data):
    """Apply cuts and compute noise model on original data."""
    job_ops = job.operators

    wrk.create_az_intervals(job, otherargs, runargs, data)
    wrk.apply_readout_filter(job, otherargs, runargs, data)
    wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)
    wrk.select_pointing(job, otherargs, runargs, data)

    # Preprocess data to get flags / cuts
    wrk.preprocess(job, otherargs, runargs, data)

    wrk.diff_noise_estimation(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)

    if job_ops.demodulate.enabled:
        save_weights = job_ops.weights_radec
        data = wrk.demodulate(job, otherargs, runargs, data)
        job_ops.weights_radec = save_weights

    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_common_mode(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)

    wrk.diff_noise_estimation(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)

    wrk.processing_mask(job, otherargs, runargs, data)
    wrk.flag_sso(job, otherargs, runargs, data)

    return data


def load_observations(job, otherargs, runargs, comm):
    """Load data and compute noise model and cuts"""
    log = toast.utils.Logger.get()
    job_ops = job.operators

    group_size = wrk.reduction_group_size(job, runargs, comm)
    toast_comm = toast.Comm(world=comm, groupsize=group_size)
    data = toast.Data(comm=toast_comm)

    # Load data from any format
    wrk.load_data_hdf5(job, otherargs, runargs, data)
    wrk.load_data_context(job, otherargs, runargs, data)

    # Perform any other operations on original data before zeroing
    data = reduce_input(job, otherargs, runargs, data)

    mem = toast.utils.memreport(
        msg="(whole node)", comm=data.comm.comm_world, silent=True
    )
    log.info_rank(f"After loading data:  {mem}", data.comm.comm_world)

    # Zero out detector data
    toast.ops.Reset(detdata=[defaults.det_data])

    return data


def reduce_realization(job, otherargs, runargs, data):
    """Apply data reduction to a single realization."""
    log = toast.utils.Logger.get()
    job_ops = job.operators
    save_weights = job_ops.weights_radec

    cleanup = False
    if otherargs.simulate_demodulated:
        # We are simulating already-demodulated data.
        if not job_ops.demodulate.enabled:
            msg = "Cannot start realizations with demodulated data, "
            msg += "since demodulation is disabled."
            raise RuntimeError(msg)
        processed_data = data
    else:
        wrk.filter_hwpss(job, otherargs, runargs, data)
        if job_ops.demodulate.enabled:
            cleanup = True
            processed_data = wrk.demodulate(job, otherargs, runargs, data)
        else:
            processed_data = data

    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_common_mode(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)

    if job.operators.splits.enabled:
        wrk.splits(job, otherargs, runargs, processed_data)
    else:
        wrk.mapmaker(job, otherargs, runargs, processed_data)
        wrk.mapmaker_filterbin(job, otherargs, runargs, processed_data)

    # Clean up demodulated data copy if we created it.
    if cleanup:
        processed_data.close()
    del processed_data

    job_ops.weights_radec = save_weights


def signal_realizations(job, otherargs, runargs, data):
    """Run minimal reduction on signal realizations for transfer functions."""
    log = toast.utils.Logger.get()
    job_ops = job.operators

    # Determine the binner being used
    if job_ops.mapmaker.enabled:
        if job_ops.binner_final.enabled:
            map_binner = job_ops.binner_final
        else:
            map_binner = job_ops.binner
    elif job_ops.filterbin.enabled:
        map_binner = job_ops.binner
    else:
        msg = "No mapmaker is enabled!"
        raise RuntimeError(msg)

    if not map_binner.full_pointing:
        log.warning_rank(
            "Mapmaking binner not using full detector pointing.  Overriding.",
            comm=data.comm.comm_world,
        )
        map_binner.full_pointing = True

    if (
        (job_ops.demodulate.enabled and job_ops.demodulate.purge)
        and not otherargs.simulate_demodulated
    ):
        log.warning_rank(
            "Demodulation configured to purge original data.  Overriding.",
            comm=data.comm.comm_world,
        )
        job_ops.demodulate.purge = False

    # Get the file for each realization
    realizations = dict()
    if data.comm.world_rank == 0:
        with open(otherargs.realizations, "r") as f:
            realizations = yaml.safe_load(f)
    if data.comm.comm_world is not None:
        realizations = data.comm.comm_world.bcast(realizations, root=0)

    # Get the base output directory
    base_output = str(otherargs.out_dir)

    # Create the pixel distribution prior to looping over realizations.
    # This might have been created by the processing chain on the
    # original data.  We clear it and recreate it with known options,
    # in particular saving all detector pointing.  Same with any detector
    # pointing generated during the initial processing on the input data.
    if map_binner.pixel_dist in data:
        del data[map_binner.pixel_dist]
    toast.ops.Delete(
        detdata=[
            map_binner.pixel_pointing.pixels,
            map_binner.stokes_weights.weights,
            map_binner.pixel_pointing.detector_pointing.quats,
        ],
    ).apply(data)

    pix_dist = toast.ops.BuildPixelDistribution(
        pixel_dist=map_binner.pixel_dist,
        pixel_pointing=map_binner.pixel_pointing,
        save_pointing=True,
    )
    pix_dist.apply(data)
    map_binner.stokes_weights.apply(data)

    for realization_indx in list(sorted(realizations.keys())):
        realization_file = realizations[realization_indx]
        msg = f"Realization {realization_indx:04d} using {realization_file}"
        log.info_rank(msg, comm=data.comm.comm_world)

        # Replace detector data with signal
        sig_scan = toast.ops.ScanHealpixMap(
            name="sig_scan",
            file=realization_file,
            zero=True,
            save_pointing=True,
            pixel_pointing=map_binner.pixel_pointing,
            stokes_weights=map_binner.stokes_weights,
        )
        sig_scan.apply(data)

        # Set the output directory for this realization
        realization_dir = os.path.join(
            base_output,
            f"{realization_indx:04d}",
        )
        otherargs.out_dir = realization_dir

        if data.comm.world_rank == 0:
            if not os.path.isdir(otherargs.out_dir):
                os.makedirs(otherargs.out_dir, exist_ok=True)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Run processing
        reduce_realization(job, otherargs, runargs, data)

    otherargs.out_dir = base_output

    mem = toast.utils.memreport(
        msg="(whole node)", comm=data.comm.comm_world, silent=True
    )
    log.info_rank(f"After all signal realizations:  {mem}", data.comm.comm_world)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_transfer (total)")
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
    parser = argparse.ArgumentParser(description="SO simulation pipeline")

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="out_sat",
        help="The output directory",
    )
    parser.add_argument(
        "--obsmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each observation separately.",
    )
    parser.add_argument(
        "--simulate_demodulated",
        required=False,
        default=False,
        action="store_true",
        help="Start each realization with demodulated / downsampled data",
    )
    parser.add_argument(
        "--realizations",
        required=False,
        default=None,
        help="YAML file containing signal realizations to process",
    )
    parser.add_argument(
        "--preprocess_copy",
        required=False,
        default=False,
        action="store_true",
        help="Perform preprocessing on a copy of the data.",
    )
    parser.add_argument(
        "--log_config",
        required=False,
        default=None,
        help="Dump out config log to this yaml file",
    )

    # FIXME:  Add an extra option here to specify a file of additional
    # per observation weight factors.

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    # Data loading
    wrk.setup_load_data_hdf5(operators)
    wrk.setup_load_data_books(operators)
    wrk.setup_load_data_context(operators)
    wrk.setup_az_intervals(operators)
    wrk.setup_diff_noise_estimation(operators)
    wrk.setup_readout_filter(operators)
    wrk.setup_deconvolve_detector_timeconstant(operators)
    wrk.setup_pointing(operators)

    # Processing that might be used.
    wrk.setup_preprocess(parser, operators)

    wrk.setup_filter_common_mode(operators)
    wrk.setup_filter_ground(operators)
    wrk.setup_filter_poly1d(operators)
    wrk.setup_filter_poly2d(operators)
    wrk.setup_noise_estimation(operators)

    wrk.setup_demodulate(operators)

    wrk.setup_processing_mask(operators)
    wrk.setup_flag_sso(operators)

    wrk.setup_splits(operators)

    wrk.setup_mapmaker(operators, templates)
    wrk.setup_mapmaker_filterbin(operators)

    job, config, otherargs, runargs = wrk.setup_job(
        parser=parser, operators=operators, templates=templates
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(otherargs.out_dir):
            os.makedirs(otherargs.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    if otherargs.log_config is not None:
        toast.config.dump_yaml(otherargs.log_config, config, comm=comm)

    # If this is a dry run, exit
    if otherargs.dry_run:
        log.info_rank("Dry-run complete", comm=comm)
        return

    # Load data and generate any noise models, cuts, and flags
    data = load_observations(job, otherargs, runargs, comm)

    # Loop over signal realizations
    signal_realizations(job, otherargs, runargs, data)

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
