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

import sotodlib.toast.pipeline_tools as so_tools

import numpy as np

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

    parser.add_argument(
        "--madam",
        required=False,
        action="store_true",
        help="Use libmadam to bin the signal",
        dest="madam",
    )
    parser.add_argument(
        "--no-madam",
        required=False,
        action="store_false",
        help="Do not use libMadam to bin the signal",
        dest="madam",
    )
    parser.set_defaults(madam=False)

    parser.add_argument(
        "--madam-conserve-memory",
        required=False,
        action="store_true",
        help="Stage the Madam buffer packing",
        dest="madam_conserve_memory",
    )
    parser.add_argument(
        "--no-madam-conserve-memory",
        required=False,
        action="store_false",
        help="Do not stage the Madam buffer packing",
        dest="madam_conserve_memory",
    )
    parser.set_defaults(madam_conserve_memory=True)

    parser.add_argument(
        "--madam-allreduce",
        required=False,
        action="store_true",
        help="Use the allreduce communication pattern in Madam",
        dest="madam_allreduce",
    )
    parser.add_argument(
        "--no-madam-allreduce",
        required=False,
        action="store_false",
        help="Do not use the allreduce communication pattern in Madam",
        dest="madam_allreduce",
    )
    parser.set_defaults(madam_allreduce=False)

    parser.add_argument(
        "--madam-concatenate-messages",
        required=False,
        action="store_true",
        help="Use the alltoallv commucation pattern in Madam",
        dest="madam_concatenate_messages",
    )
    parser.add_argument(
        "--no-madam-concatenate-messages",
        required=False,
        action="store_false",
        help="Use the point-to-point communication pattern in Madam",
        dest="madam_concatenate_messages",
    )
    parser.set_defaults(madam_concatenate_messages=True)

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

    madam = None

    for mc in range(firstmc, firstmc + nmc):

        timer_mc = Timer()
        timer_mc.start()

        outpath = setup_output(args, comm, mc)
        outprefix = args.map_prefix + "_filtered"
        if args.madam:
            outmap = os.path.join(outpath, outprefix + "_bmap.fits")
        else:
            outmap = os.path.join(outpath, outprefix + "_binned.fits")

        if os.path.isfile(outmap):
            if comm.world_rank == 0:
                log.info("{} exists, skipping".format(outmap))
            continue

        if comm.world_rank == 0:
            log.info("Processing MC = {} into {}".format(mc, outmap))

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

        if args.apply_polyfilter or args.apply_groundfilter:

            # Filter signal

            toast_tools.apply_polyfilter(args, comm, data, totalname)

            toast_tools.apply_groundfilter(args, comm, data, totalname)

            memreport("after filter", comm.comm_world)

            # Bin maps

            timer_map = Timer()
            timer_map.start()

            if args.madam:
                if madam is None:
                    madampars = {}
                    madampars["temperature_only"] = False
                    for name in [
                            "kfirst",
                            "write_map",
                            "write_matrix",
                            "write_wcov",
                            "write_hits"
                    ]:
                        madampars[name] = False
                    madampars["write_binmap"] = True
                    madampars["concatenate_messages"] = True
                    madampars["allreduce"] = True
                    madampars["nside_submap"] = args.nside_submap
                    madampars["reassign_submaps"] = True
                    madampars["pixlim_map"] = 1e-2
                    madampars["pixmode_map"] = 2
                    # Instead of fixed detector weights, we'll want to use scaled noise
                    # PSD:s that include the atmospheric noise
                    madampars["radiometers"] = True
                    madampars["noise_weights_from_psd"] = True
                    madampars["nside_map"] = args.nside
                    madampars["fsample"] = args.sample_rate
                    madampars["path_output"] = outpath
                    madampars["file_root"] = outprefix
                    if args.madam_concatenate_messages:
                        # Collective communication is fast but requires memory
                        madampars["concatenate_messages"] = True
                        if args.madam_allreduce:
                            # Every process will allocate a copy of every observed submap.
                            madampars["allreduce"] = True
                        else:
                            # Every process will allocate complete send and receive buffers
                            madampars["allreduce"] = False
                    else:
                        # Slow but memory-efficient point-to-point communication.  Allocate
                        # only enough memory to communicate with one process at a time.
                        madampars["concatenate_messages"] = False
                        madampars["allreduce"] = False
                    madam = toast.todmap.OpMadam(
                        params=madampars,
                        detweights=detweights,
                        name=totalname,
                        common_flag_mask=args.common_flag_mask,
                        purge_tod=True,
                        mcmode=False,
                        conserve_memory=args.madam_conserve_memory,
                    )
                else:
                    madam.params["path_output"] = outpath
                madam.exec(data)
                del madam
                madam = None
            else:
                mapmaker = toast.todmap.OpMapMaker(
                    nside=args.nside,
                    nnz=3,
                    name=totalname,
                    outdir=outpath,
                    outprefix=outprefix + "_",
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
                )
                mapmaker.exec(data)

            if comm.world_rank == 0:
                timer_map.report_clear("Bin map")

            memreport("after filter & bin", comm.comm_world)

            if comm.world_rank == 0:
                timer_mc.report_clear("Monte Carlo iteration # {:05}".format(mc))

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
        timer.report_clear("Gather and dump timing info")

    if comm.world_rank == 0:
        timer0.report_clear("toast_so_tf.py pipeline")

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
