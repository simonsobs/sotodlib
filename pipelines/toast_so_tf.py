#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

# This pipeline exists only to measure the filter&bin transfer function


import os

# TOAST must be imported before numpy to ensure the right MKL is used
import toast

import copy
from datetime import datetime
import gc
import os
import pickle
import re
import sys
import traceback

import argparse
import dateutil.parser

from toast.mpi import get_world, Comm
from toast.utils import Logger, Environment, memreport

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

import toast.pipeline_tools as toast_tools

import sotodlib.pipeline_tools as so_tools

import numpy as np

import sotodlib.hardware

import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

XAXIS, YAXIS, ZAXIS = np.eye(3)


def parse_arguments(comm):
    timer = Timer()
    timer.start()
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars="@",
    )

    toast_tools.add_dist_args(parser)
    toast_tools.add_todground_args(parser)
    toast_tools.add_pointing_args(parser)
    toast_tools.add_polyfilter_args(parser)
    toast_tools.add_groundfilter_args(parser)
    toast_tools.add_noise_args(parser)
    toast_tools.add_madam_args(parser)
    toast_tools.add_sky_map_args(parser)
    toast_tools.add_mc_args(parser)
    so_tools.add_hw_args(parser)
    so_tools.add_so_noise_args(parser)
    so_tools.add_pysm_args(parser)
    so_tools.add_export_args(parser)
    toast_tools.add_debug_args(parser)

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )

    parser.add_argument(
        "--map-prefix", required=False, default="toast", help="Output map prefix"
    )
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        sys.exit()

    if len(args.bands.split(",")) != 1:
        # Multi frequency run.  We don't support multiple copies of
        # scanned signal.
        if args.input_map:
            raise RuntimeError(
                "Multiple frequencies are not supported when scanning from a map"
            )

    if args.simulate_atmosphere and args.weather is None:
        raise RuntimeError("Cannot simulate atmosphere without a TOAST weather file")

    if comm.world_rank == 0:
        log.info("\n")
        log.info("All parameters:")
        for ag in vars(args):
            log.info("{} = {}".format(ag, getattr(args, ag)))
        log.info("\n")

    if args.group_size:
        comm = Comm(groupsize=args.group_size)

    if comm.world_rank == 0:
        if not os.path.isdir(args.outdir):
            try:
                os.makedirs(args.outdir)
            except FileExistsError:
                pass
        timer.report_clear("Parse arguments")

    return args, comm


def setup_output(args, comm, mc):
    outpath = "{}/{:08}".format(args.outdir, mc)
    if comm.world_rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_so_sim (total)")
    timer0 = Timer()
    timer0.start()

    mpiworld, procs, rank, comm = toast_tools.get_comm()

    memreport("at the beginning of the pipeline", comm.comm_world)

    args, comm = parse_arguments(comm)

    # Initialize madam parameters

    madampars = toast_tools.setup_madam(args)

    # Load and broadcast the schedule file

    schedules = toast_tools.load_schedule(args, comm)

    # Load the weather and append to schedules

    toast_tools.load_weather(args, comm, schedules)

    # load or simulate the focalplane

    detweights = so_tools.load_focalplanes(args, comm, schedules)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    data, telescope_data = so_tools.create_observations(args, comm, schedules)

    memreport("after creating observations", comm.comm_world)

    # Optionally rewrite the noise PSD:s in each observation to include
    # elevation-dependence
    so_tools.get_elevation_noise(args, comm, data)

    totalname = "total"

    # Split the communicator for day and season mapmaking

    time_comms = toast_tools.get_time_communicators(args, comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    toast_tools.expand_pointing(args, comm, data)

    memreport("after pointing", comm.comm_world)

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    for mc in range(firstmc, firstmc + nmc):

        if comm.world_rank == 0:
            log.info("Processing MC = {}".format(mc))

        # Ensure there is no stale signal in the cache
            
        toast.tod.OpCacheClear(totalname).exec(data)

        if args.pysm_model:
            if schedules is not None:
                focalplanes = [s.telescope.focalplane.detector_data for s in schedules]
            else:
                focalplanes = [telescope.focalplane.detector_data]
            so_tools.simulate_sky_signal(args, comm, data, focalplanes, totalname, mc=mc)
        else:
            toast_tools.scan_sky_signal(args, comm, data, totalname, mc=mc)

        memreport("after PySM", comm.comm_world)

        # update_atmospheric_noise_weights(args, comm, data, freq, mc)

        toast_tools.add_signal(
            args, comm, data, totalname, signalname, purge=(mc == firstmc + nmc - 1)
        )

        memreport("after adding sky", comm.comm_world)

        outpath = setup_output(args, comm, mc)

        if args.apply_polyfilter or args.apply_groundfilter:

            # Filter signal

            toast_tools.apply_polyfilter(args, comm, data, totalname)

            toast_tools.apply_groundfilter(args, comm, data, totalname)

            memreport("after filter", comm.comm_world)

            # Bin maps

            mapmaker = toast.todmap.MapMaker(
                nside=args.nside,
                nnz=3,
                name=totalname,
                outdir=outpath,
                outprefix=args.map_prefix + "_filtered",
                write_hits=False,
                zip_maps=False,
                write_wcov_inv=False,
                write_wcov=False,
                write_binned=True,
                write_destriped=False,
                write_rcond=False,
                rcond_limit=1e-3,
                baseline_length=None,
                common_flag_mask=args.common_flag_mask,
                pixels=pixels,
            )
            mapmaker.exec(data)

            memreport("after filter & bin", comm.comm_world)

    if comm.comm_world is not None:
        comm.comm_world.barrier()

    memreport("at the end of the pipeline", comm.comm_world)

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        timer.stop()
        timer.report("Gather and dump timing info")
    timer0.stop()
    if comm.world_rank == 0:
        timer0.report("toast_so_sim.py pipeline")
    return


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None and procs > 1:
            mpiworld.Abort(6)
        else:
            raise
