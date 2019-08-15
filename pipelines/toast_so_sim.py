#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

# import so3g

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


from toast.pipeline_tools import (
    add_dist_args,
    add_debug_args,
    get_time_communicators,
    get_comm,
    add_polyfilter_args,
    apply_polyfilter,
    add_groundfilter_args,
    apply_groundfilter,
    add_atmosphere_args,
    simulate_atmosphere,
    scale_atmosphere_by_frequency,
    update_atmospheric_noise_weights,
    add_noise_args,
    simulate_noise,
    # get_analytic_noise,
    add_gainscrambler_args,
    scramble_gains,
    add_pointing_args,
    expand_pointing,
    get_submaps,
    add_madam_args,
    setup_madam,
    apply_madam,
    add_sky_map_args,
    # add_pysm_args,
    scan_sky_signal,
    # simulate_sky_signal,
    add_sss_args,
    simulate_sss,
    add_signal,
    copy_signal,
    add_tidas_args,
    output_tidas,
    # add_spt3g_args,
    # output_spt3g,
    add_todground_args,
    load_schedule,
    load_weather,
    add_mc_args,
)

from sotodlib.pipeline_tools import (
    add_hw_args,
    add_so_noise_args,
    get_elevation_noise,
    add_pysm_args,
    simulate_sky_signal,
    create_observations,
    load_focalplanes,
    scale_atmosphere_by_bandpass,
    add_export_args,
    export_TOD,
)

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

    add_dist_args(parser)
    add_todground_args(parser)
    add_pointing_args(parser)
    add_polyfilter_args(parser)
    add_groundfilter_args(parser)
    add_atmosphere_args(parser)
    add_noise_args(parser)
    add_gainscrambler_args(parser)
    add_madam_args(parser, ground_data=True)
    add_sky_map_args(parser)
    add_sss_args(parser)
    add_tidas_args(parser)
    add_mc_args(parser)
    add_hw_args(parser)
    add_so_noise_args(parser)
    add_pysm_args(parser)
    add_export_args(parser)
    add_debug_args(parser)

    parser.add_argument(
        "--no-maps",
        required=False,
        default=False,
        action="store_true",
        help="Disable all mapmaking.",
    )

    parser.add_argument(
        "--import-dir",
        required=False,
        help="Directory to load TOD from",
    )
    parser.add_argument(
        "--import-obs",
        required=False,
        help="Comma-separated list of observations to load.  Default is to load "
        "all observations in --import-dir",
    )
    parser.add_argument(
        "--import-prefix",
        required=False,
        help="Prefix for TOD files to import"
    )
    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
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

    mpiworld, procs, rank, comm = get_comm()

    memreport(comm.comm_world, "at the beginning of the pipeline")

    args, comm = parse_arguments(comm)

    # Initialize madam parameters

    madampars = setup_madam(args)

    if args.import_dir is not None:
        schedules = None
        # Load existing data and optionally
        # add simulated data to it
        from sotodlib.data.toast_load import load_data
        if args.import_obs is not None:
            import_obs = args.import_obs.split(",")
        else:
            import_obs = None
        from sotodlib.pipeline_tools.hardware import get_hardware, get_focalplane
        hw, telescope, det_index = get_hardware(args, comm, verbose=True)
        focalplane = get_focalplane(args, comm, hw, det_index, verbose=True)
        detweights = focalplane.detweights
        telescope.focalplane = focalplane

        if comm.world_rank == 0:
            log.info("Loading TOD from {}".format(args.import_dir))
        timer = Timer()
        timer.start()
        data = load_data(
            args.import_dir,
            obs=import_obs,
            comm=comm,
            prefix=args.import_prefix,
            dets=hw,
            detranks=comm.group_size,
            )
        if comm.world_rank == 0:
            timer.report_clear("Load data")
        telescope_data = [("all", data)]
        site = telescope.site
        focalplane = telescope.focalplane
        for obs in data.obs:
            #obs["baselines"] = None
            obs["noise"] = focalplane.noise
            #obs["id"] = int(ces.mjdstart * 10000)
            #obs["intervals"] = tod.subscans
            obs["site"] = site.name
            obs["site_id"] = site.id
            obs["telescope"] = telescope.name
            obs["telescope_id"] = telescope.id
            obs["fpradius"] = focalplane.radius
            #obs["weather"] = site.weather
            #obs["start_time"] = ces.start_time
            obs["altitude"] = site.alt
            #obs["season"] = ces.season
            #obs["date"] = ces.start_date
            #obs["MJD"] = ces.mjdstart
            obs["focalplane"] = focalplane.detector_data
            #obs["rising"] = ces.rising
            #obs["mindist_sun"] = ces.mindist_sun
            #obs["mindist_moon"] = ces.mindist_moon
            #obs["el_sun"] = ces.el_sun
        memreport(comm.comm_world, "after imports")
        totalname = "signal"
    else:
        # Load and broadcast the schedule file

        schedules = load_schedule(args, comm)

        # Load the weather and append to schedules

        load_weather(args, comm, schedules)

        # load or simulate the focalplane

        detweights = load_focalplanes(args, comm, schedules)

        # Create the TOAST data object to match the schedule.  This will
        # include simulating the boresight pointing.

        data, telescope_data = create_observations(args, comm, schedules)

        memreport(comm.comm_world, "after creating observations")

        # Optionally rewrite the noise PSD:s in each observation to include
        # elevation-dependence
        get_elevation_noise(args, comm, data)

        totalname = "total"

    time_comms = get_time_communicators(args, comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data)

    # Only purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.export is None):
        for ob in data.obs:
            tod = ob["tod"]
            try:
                tod.free_radec_quats()
            except AttributeError:
                # These TOD objects do not have RA/Dec quaternions
                pass

    memreport(comm.comm_world, "after pointing")

    # Split the communicator for day and season mapmaking

    time_comms = get_time_communicators(comm, data)

    # Prepare auxiliary information for distributed map objects

    _, localsm, subnpix = get_submaps(args, comm, data)

    memreport(comm.comm_world, "after submaps")

    # Set up objects to take copies of the TOD at appropriate times

    if args.input_pysm_model:
        if schedules is not None:
            focalplanes = [s.telescope.focalplane.detector_data for s in schedules]
        else:
            focalplanes = [telescope.focalplane.detector_data]
        signalname = simulate_sky_signal(args, comm, data, focalplanes, subnpix, localsm)
    else:
        signalname = scan_sky_signal(args, comm, data, localsm, subnpix)

    memreport(comm.comm_world, "after PySM")

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    for mc in range(firstmc, firstmc + nmc):

        if comm.world_rank == 0:
            log.info("Processing MC = {}".format(mc))

        simulate_atmosphere(args, comm, data, mc, totalname)

        scale_atmosphere_by_bandpass(args, comm, data, totalname, mc)

        memreport(comm.comm_world, "after atmosphere")

        # update_atmospheric_noise_weights(args, comm, data, freq, mc)

        add_signal(args, comm, data, totalname, signalname, purge=(nmc == 1))

        memreport(comm.comm_world, "after adding sky")

        simulate_noise(args, comm, data, mc, totalname)

        memreport(comm.comm_world, "after simulating noise")

        simulate_sss(args, comm, data, mc, totalname)

        memreport(comm.comm_world, "after simulating SSS")

        # DEBUG begin
        """
        import matplotlib.pyplot as plt
        tod = data.obs[0]['tod']
        times = tod.local_times()
        for det in tod.local_dets:
            sig = tod.local_signal(det, totalname)
            plt.plot(times, sig, label=det)
        plt.legend(loc='best')
        fnplot = 'debug_{}.png'.format(args.madam_prefix)
        plt.savefig(fnplot)
        plt.close()
        print('DEBUG plot saved in', fnplot)
        return
        """
        # DEBUG end

        scramble_gains(args, comm, data, mc, totalname)

        if mc == firstmc:
            # For the first realization and frequency, optionally
            # export the timestream data.
            output_tidas(args, comm, data, totalname)
            # export_TOD(args, comm, data, totalname, other=[signalname])
            export_TOD(args, comm, data, totalname, schedules)

            memreport(comm.comm_world, "after export")

        if args.no_maps:
            continue

        outpath = setup_output(args, comm, mc)

        # Bin and destripe maps

        apply_madam(
            args,
            comm,
            data,
            madampars,
            outpath,
            detweights,
            totalname,
            time_comms=time_comms,
            telescope_data=telescope_data,
            first_call=(mc == firstmc),
        )

        memreport(comm.comm_world, "after madam")

        if args.apply_polyfilter or args.apply_groundfilter:

            # Filter signal

            apply_polyfilter(args, comm, data, totalname)

            apply_groundfilter(args, comm, data, totalname)

            memreport(comm.comm_world, "after filter")

            # Bin maps

            apply_madam(
                args,
                comm,
                data,
                madampars,
                outpath,
                detweights,
                totalname,
                time_comms=time_comms,
                telescope_data=telescope_data,
                first_call=False,
                extra_prefix="filtered",
                bin_only=True,
            )

            memreport(comm.comm_world, "after filter & bin")

    if comm.comm_world is not None:
        comm.comm_world.barrier()

    memreport(comm.comm_world, "at the end of the pipeline")

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
        if mpiworld is not None:
            mpiworld.Abort(6)
