#!/usr/bin/env python3

# Copyright (c) 2019-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs an SO time domain simulation.

You can see the automatically generated command line options with:

    toast_so_sim --help

Or you can dump a config file with all the default values with:

    toast_so_sim --default_toml config.toml

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


def simulate_data(job, otherargs, runargs, comm):
    log = toast.utils.Logger.get()
    job_ops = job.operators

    if job_ops.sim_ground.enabled:
        data = wrk.simulate_observing(job, otherargs, runargs, comm)
    else:
        group_size = wrk.reduction_group_size(job, runargs, comm)
        toast_comm = toast.Comm(world=comm, groupsize=group_size)
        data = toast.Data(comm=toast_comm)
        # Load data from all formats
        wrk.load_data_hdf5(job, otherargs, runargs, data)
        wrk.load_data_books(job, otherargs, runargs, data)
        wrk.load_data_context(job, otherargs, runargs, data)
        wrk.create_az_intervals(job, otherargs, runargs, data)
        # Optionally derive simple noise estimates
        wrk.act_responsivity_sign(job, otherargs, runargs, data)
        wrk.apply_readout_filter(job, otherargs, runargs, data)
        wrk.diff_noise(job, otherargs, runargs, data)
        # optionally zero out
        if otherargs.zero_loaded_data:
            toast.ops.Reset(detdata=[defaults.det_data]).apply(data)
        # Append a weather model
        wrk.append_weather_model(job, otherargs, runargs, data)

    wrk.select_pointing(job, otherargs, runargs, data)
    wrk.simple_noise_models(job, otherargs, runargs, data)
    wrk.simulate_atmosphere_signal(job, otherargs, runargs, data)

    # Shortcut if we are only caching the atmosphere.  If this job is only caching
    # (not observing) the atmosphere, then return at this point.
    if job.operators.sim_atmosphere.cache_only:
        return data

    wrk.simulate_sky_map_signal(job, otherargs, runargs, data)
    wrk.simulate_conviqt_signal(job, otherargs, runargs, data)
    wrk.simulate_scan_synchronous_signal(job, otherargs, runargs, data)
    wrk.simulate_source_signal(job, otherargs, runargs, data)
    wrk.simulate_sso_signal(job, otherargs, runargs, data)
    wrk.simulate_catalog_signal(job, otherargs, runargs, data)
    wrk.simulate_wiregrid_signal(job, otherargs, runargs, data)
    wrk.simulate_stimulator_signal(job, otherargs, runargs, data)
    wrk.simulate_detector_timeconstant(job, otherargs, runargs, data)
    wrk.simulate_mumux_crosstalk(job, otherargs, runargs, data)
    wrk.simulate_detector_noise(job, otherargs, runargs, data)
    wrk.simulate_hwpss_signal(job, otherargs, runargs, data)
    wrk.simulate_detector_yield(job, otherargs, runargs, data)
    wrk.simulate_calibration_error(job, otherargs, runargs, data)
    wrk.simulate_readout_effects(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After simulating data:  {mem}", comm)

    wrk.save_data_hdf5(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After saving data:  {mem}", comm)

    return data


def reduce_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()

    wrk.flag_noise_outliers(job, otherargs, runargs, data)
    wrk.filter_hwpss(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)
    data = wrk.demodulate(job, otherargs, runargs, data)
    wrk.processing_mask(job, otherargs, runargs, data)
    wrk.flag_sso(job, otherargs, runargs, data)
    wrk.hn_map(job, otherargs, runargs, data)
    wrk.cadence_map(job, otherargs, runargs, data)
    wrk.crosslinking_map(job, otherargs, runargs, data)
    wrk.raw_statistics(job, otherargs, runargs, data)
    wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)
    wrk.mapmaker_ml(job, otherargs, runargs, data)
    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)
    wrk.filter_common_mode(job, otherargs, runargs, data)
    wrk.mapmaker(job, otherargs, runargs, data)
    wrk.mapmaker_filterbin(job, otherargs, runargs, data)
    wrk.mapmaker_madam(job, otherargs, runargs, data)
    wrk.filtered_statistics(job, otherargs, runargs, data)

    mem = toast.utils.memreport(
        msg="(whole node)", comm=data.comm.comm_world, silent=True
    )
    log.info_rank(f"After reducing data:  {mem}", data.comm.comm_world)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_sim (total)")
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
        default="toast_out",
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
        "--intervalmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each interval separately.",
    )
    parser.add_argument(
        "--zero_loaded_data",
        required=False,
        default=False,
        action="store_true",
        help="Zero out detector data loaded from disk",
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    # Loading data from disk is disabled by default
    wrk.setup_load_data_hdf5(operators)
    wrk.setup_load_data_books(operators)
    wrk.setup_load_data_context(operators)
    wrk.setup_act_responsivity_sign(operators)

    # Simulated observing is enabled by default
    wrk.setup_simulate_observing(parser, operators)

    wrk.setup_pointing(operators)
    wrk.setup_az_intervals(operators)
    wrk.setup_simple_noise_models(operators)
    wrk.setup_weather_model(operators)
    wrk.setup_readout_filter(operators)
    wrk.setup_diff_noise(operators)
    wrk.setup_simulate_atmosphere_signal(operators)
    wrk.setup_simulate_sky_map_signal(operators)
    wrk.setup_simulate_conviqt_signal(operators)
    wrk.setup_simulate_scan_synchronous_signal(operators)
    wrk.setup_simulate_source_signal(operators)
    wrk.setup_simulate_sso_signal(operators)
    wrk.setup_simulate_catalog_signal(operators)
    wrk.setup_simulate_wiregrid_signal(operators)
    wrk.setup_simulate_stimulator_signal(operators)
    wrk.setup_simulate_detector_timeconstant(operators)
    wrk.setup_simulate_mumux_crosstalk(operators)
    wrk.setup_simulate_detector_noise(operators)
    wrk.setup_simulate_hwpss_signal(operators)
    wrk.setup_simulate_detector_yield(operators)
    wrk.setup_simulate_calibration_error(operators)
    wrk.setup_simulate_readout_effects(operators)
    wrk.setup_save_data_hdf5(operators)

    wrk.setup_flag_noise_outliers(operators)
    wrk.setup_filter_hwpss(operators)
    wrk.setup_demodulate(operators)
    wrk.setup_noise_estimation(operators)
    wrk.setup_processing_mask(operators)
    wrk.setup_flag_sso(operators)
    wrk.setup_hn_map(operators)
    wrk.setup_cadence_map(operators)
    wrk.setup_crosslinking_map(operators)
    wrk.setup_raw_statistics(operators)
    wrk.setup_deconvolve_detector_timeconstant(operators)
    wrk.setup_mapmaker_ml(operators)
    wrk.setup_filter_ground(operators)
    wrk.setup_filter_poly1d(operators)
    wrk.setup_filter_poly2d(operators)
    wrk.setup_filter_common_mode(operators)
    wrk.setup_mapmaker(operators, templates)
    wrk.setup_mapmaker_filterbin(operators)
    wrk.setup_mapmaker_madam(operators)
    wrk.setup_filtered_statistics(operators)

    job, config, otherargs, runargs = wrk.setup_job(
        parser=parser, operators=operators, templates=templates
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(otherargs.out_dir):
            os.makedirs(otherargs.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(otherargs.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    # If this is a dry run, exit
    if otherargs.dry_run:
        log.info_rank("Dry-run complete", comm=comm)
        return

    data = simulate_data(job, otherargs, runargs, comm)

    if not job.operators.sim_atmosphere.cache_only:
        # Reduce the data
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
