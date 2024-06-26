#!/usr/bin/env python3

# Copyright (c) 2019-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This workflow runs basic data reduction and map-making:

- A single MPI process group is used

- All input observations are loaded and made into the output map

In particular, this script is designed for testing mapmaking techniques on data that
already exists on disk.  It does not simulate data and it is not a full workflow for
running null tests, building observation matrices, etc.

You can see the automatically generated command line options with:

    toast_so_map --help

Or you can dump a config file with all the default values with:

    toast_so_map --default_toml config.toml

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


def preprocess(job, otherargs, runargs, data):
    """Apply flags and cuts based on data quality."""
    log = toast.utils.Logger.get()
    job_ops = job.operators

    if otherargs.preprocess_copy:
        prename = "preprocess"
        toast.ops.Copy(detdata=[(defaults.det_data, prename)]).apply(data)
    else:
        prename = defaults.det_data

    def _run_pre_op(op, wrkf):
        save_op_detdata = op.det_data
        op.det_data = prename
        wrkf(job, otherargs, runargs, data)
        op.det_data = save_op_detdata

    # Ensure that we run the HWPSS filter for preprocessing, even if we
    # are not using it on the original data.
    if otherargs.preprocess_copy:
        save_hwpf_state = job_ops.hwpfilter.enabled
        job_ops.hwpfilter.enabled = True
    _run_pre_op(job_ops.hwpfilter, wrk.filter_hwpss)
    if otherargs.preprocess_copy:
        job_ops.hwpfilter.enabled = save_hwpf_state

    _run_pre_op(job_ops.simple_jumpcorrect, wrk.simple_jumpcorrect)

    _run_pre_op(job_ops.simple_deglitch, wrk.simple_deglitch)

    _run_pre_op(job_ops.diff_noise_cut, wrk.flag_diff_noise_outliers)

    _run_pre_op(job_ops.noise_cut, wrk.flag_noise_outliers)

    if otherargs.preprocess_copy:
        toast.ops.Delete(detdata=[prename,]).apply(data)
        # FIXME:  Although the glitches / jumps are flagged, the original
        # data has not been filled / corrected.  Not sure how to do this
        # without applying the hwpfilter to the original data, which we
        # might want to avoid if solving for that template in the mapmaking.


def reduce_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()
    job_ops = job.operators
    job_tmpl = job.templates

    # Preprocess data to get flags / cuts
    preprocess(job, otherargs, runargs, data)

    # Now apply preprocessing to good detectors
    wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)
    wrk.raw_statistics(job, otherargs, runargs, data)

    wrk.filter_common_mode(job, otherargs, runargs, data)
    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)
    wrk.diff_noise_estimation(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)

    data = wrk.demodulate(job, otherargs, runargs, data)

    wrk.processing_mask(job, otherargs, runargs, data)
    wrk.flag_sso(job, otherargs, runargs, data)
    wrk.hn_map(job, otherargs, runargs, data)
    wrk.cadence_map(job, otherargs, runargs, data)
    wrk.crosslinking_map(job, otherargs, runargs, data)

    if job.operators.splits.enabled:
        wrk.splits(job, otherargs, runargs, data)
    else:
        wrk.mapmaker_ml(job, otherargs, runargs, data)
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
    gt.start("toast_so_map (total)")
    timer = toast.timing.Timer()
    timer.start()

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
    log.info_rank(f"Start of the workflow:  {mem}", comm)

    parser = argparse.ArgumentParser(description="SO mapmaking pipeline")

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
        "--detmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each detector separately.",
    )
    parser.add_argument(
        "--intervalmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each interval separately.",
    )
    parser.add_argument(
        "--preprocess_copy",
        required=False,
        default=False,
        action="store_true",
        help="Perform preprocessing on a copy of the data.",
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    wrk.setup_load_or_simulate_observing(parser, operators)

    wrk.setup_simple_jumpcorrect(operators)
    wrk.setup_simple_deglitch(operators)

    wrk.setup_flag_diff_noise_outliers(operators)
    wrk.setup_flag_noise_outliers(operators)
    wrk.setup_deconvolve_detector_timeconstant(operators)
    wrk.setup_raw_statistics(operators)

    wrk.setup_filter_hwpss(operators)
    wrk.setup_filter_common_mode(operators)
    wrk.setup_filter_ground(operators)
    wrk.setup_filter_poly1d(operators)
    wrk.setup_filter_poly2d(operators)
    wrk.setup_noise_estimation(operators)

    wrk.setup_demodulate(operators)

    wrk.setup_processing_mask(operators)
    wrk.setup_flag_sso(operators)
    wrk.setup_hn_map(operators)
    wrk.setup_cadence_map(operators)
    wrk.setup_crosslinking_map(operators)

    wrk.setup_mapmaker_ml(operators)
    wrk.setup_mapmaker(operators, templates)
    wrk.setup_mapmaker_filterbin(operators)
    wrk.setup_mapmaker_madam(operators)
    wrk.setup_splits(operators)
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

    # Load data
    data = wrk.load_or_simulate_observing(job, otherargs, runargs, comm)

    # Reduce it
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
