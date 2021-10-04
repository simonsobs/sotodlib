#!/usr/bin/env python3

# Copyright (c) 2019-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs a pointing and crosslinking test.

You can see the automatically generated command line options with:

    crosslinking_test.py --help

Or you can dump a config file with all the default values with:

    crosslinking_test.py --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import os
import sys
import traceback
import argparse

import numpy as np

from astropy import units as u

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
import toast.ops

from toast.mpi import MPI

import sotodlib.toast.ops as so_ops


def parse_config(operators, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="SO crosslinking test")

    # Arguments specific to this script

    parser.add_argument(
        "--hardware", required=False, default=None, help="Input hardware file"
    )
    parser.add_argument(
        "--thinfp",
        required=False,
        type=int,
        help="Thin the focalplane by this factor",
    )
    parser.add_argument(
        "--bands",
        required=True,
        help="Comma-separated list of bands: LAT_f030 (27GHz), LAT_f040 (39GHz), "
        "LAT_f090 (93GHz), LAT_f150 (145GHz), "
        "LAT_f230 (225GHz), LAT_f290 (285GHz), "
        "SAT_f030 (27GHz), SAT_f040 (39GHz), "
        "SAT_f090 (93GHz), SAT_f150 (145GHz), "
        "SAT_f230 (225GHz), SAT_f290 (285GHz). "
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--telescope",
        help="Telescope to simulate: LAT, SAT1, SAT2, SAT3, SAT4.",
    )
    group.add_argument(
        "--tube_slots",
        help="Comma-separated list of optics tube slots: c1 (LAT_UHF), i5 (LAT_UHF), "
        " i6 (LAT_MF), i1 (LAT_MF), i3 (LAT_MF), i4 (LAT_MF), o6 (LAT_LF),"
        " ST1 (SAT_MF), ST2 (SAT_MF), ST3 (SAT_UHF), ST4 (SAT_LF)."
    )
    group.add_argument(
        "--wafer_slots",
        help="Comma-separated list of optics tube slots. "
    )

    parser.add_argument(
        "--sample_rate", required=False, default=10, help="Sampling rate"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="out_crosslinking",
        help="The output directory",
    )

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    return config, args, jobargs


def load_instrument_and_schedule(args, comm):
    focalplane = sotoast.SOFocalplane(
        hwfile=args.hardware,
        telescope=args.telescope,
        sample_rate=args.sample_rate * u.Hz,
        bands=args.bands,
        wafer_slots=args.wafer_slots,
        tube_slots=args.tube_slots,
        thinfp=args.thinfp,
        comm=comm,
    )

    # Load the schedule file
    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule, comm=comm)

    # FIXME : hardcode site parameters?
    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
        weather=None,
    )

    telescope = toast.instrument.Telescope(
        focalplane.telescope, focalplane=focalplane, site=site
    )
    return telescope, schedule


def job_create(config, jobargs, telescope, schedule, comm):
    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.
    group_size = toast.job_group_size(
        comm,
        jobargs,
        schedule=schedule,
        focalplane=telescope.focalplane,
        full_pointing=False,
    )
    return job, group_size


def simulate_data(job, toast_comm, telescope, schedule):
    log = toast.utils.Logger.get()
    job_ops = job.operators
    world_comm = toast_comm.comm_world

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Simulate the telescope pointing

    job_ops.sim_ground.telescope = telescope
    job_ops.sim_ground.schedule = schedule
    if job_ops.sim_ground.weather is None:
        job_ops.sim_ground.weather = telescope.site.name
    job_ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    # Apply LAT co-rotation
    job_ops.corotate_lat.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters

    job_ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Set up detector pointing in both Az/El and RA/DEC

    job_ops.det_pointing_azel.boresight = job_ops.sim_ground.boresight_azel
    job_ops.det_pointing_radec.boresight = job_ops.sim_ground.boresight_radec

    job_ops.weights_azel.detector_pointing = job_ops.det_pointing_azel
    job_ops.weights_azel.hwp_angle = job_ops.sim_ground.hwp_angle

    # Create the Elevation modulated noise model

    job_ops.elevation_model.noise_model = job_ops.default_model.noise_model
    job_ops.elevation_model.detector_pointing = job_ops.det_pointing_azel
    job_ops.elevation_model.view = job_ops.det_pointing_azel.view
    job_ops.elevation_model.apply(data)
    log.info_rank("Created elevation noise model in", comm=world_comm, timer=timer)

    # Set up the pointing.  Each pointing matrix operator requires a detector pointing
    # operator, and each binning operator requires a pointing matrix operator.
    job_ops.pixels_radec.detector_pointing = job_ops.det_pointing_radec
    job_ops.weights_radec.detector_pointing = job_ops.det_pointing_radec
    job_ops.weights_radec.hwp_angle = job_ops.sim_ground.hwp_angle
    job_ops.pixels_radec_final.detector_pointing = job_ops.det_pointing_radec

    # Simulate detector noise, just so that the TOD contains more than zeros.
    job_ops.sim_noise.noise_model = job_ops.elevation_model.out_model
    job_ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    job_ops = job.operators
    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Build the pixel distribution information, needed below
    job_ops.pixel_dist.pixel_pointing = job_ops.pixel_radec
    job_ops.pixel_dist.apply(data)

    # Various geometric factors

    job_ops.h_n.pixel_pointing = job_ops.pixels_radec
    job_ops.h_n.pixel_dist = job_ops.pixel_dist.pixel_dist
    job_ops.h_n.output_dir = args.out_dir
    job_ops.h_n.apply(data)
    log.info_rank("Calculated h_n in", comm=world_comm, timer=timer)

    job_ops.cadence_map.pixel_pointing = job_ops.pixels_radec
    job_ops.cadence_map.pixel_dist = job_ops.pixel_dist.pixel_dist
    job_ops.cadence_map.output_dir = args.out_dir
    job_ops.cadence_map.apply(data)
    log.info_rank("Calculated cadence map in", comm=world_comm, timer=timer)

    job_ops.crosslinking.pixel_pointing = job_ops.pixels_radec
    job_ops.crosslinking.pixel_dist = job_ops.pixel_dist.pixel_dist
    job_ops.crosslinking.output_dir = args.out_dir
    job_ops.crosslinking.apply(data)
    log.info_rank("Calculated crosslinking in", comm=world_comm, timer=timer)

    job_ops.mlmapmaker.apply(data)
    log.info_rank("Finished ML map-making in", comm=world_comm, timer=timer)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_sim (total)")

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    #
    # We can also set some default values here for the traits, including whether an
    # operator is disabled by default.

    operators = [
        toast.ops.SimGround(name="sim_ground", weather="atacama"),
        so_ops.CoRotator(name="corotate_lat"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ElevationNoise(
            name="elevation_model",
            out_model="el_noise_model",
        ),
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel"),
        toast.ops.StokesWeights(
            name="weights_azel", weights="weights_azel", mode="IQU"
        ),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        toast.ops.BuildPixelDistribution(name="pixel_dist"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PixelsHealpix(name="pixels_radec"),
        toast.ops.StokesWeights(name="weights_radec", mode="IQU"),
        so_ops.Hn(name="h_n"),
        toast.ops.CadenceMap(name="cadence_map"),
        toast.ops.CrossLinking(name="crosslinking"),
        so_ops.MLMapmaker(name="mlmapmaker", comps="TQU")
    ]

    # Parse options
    config, args, jobargs = parse_config(operators, list(), comm)

    # Load our instrument model and observing schedule
    telescope, schedule = load_instrument_and_schedule(args, comm)

    # Instantiate our operators and get the size of the process groups
    job, group_size = job_create(
        config, jobargs, telescope, schedule, comm
    )

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Create simulated data
    data = simulate_data(job, toast_comm, telescope, schedule)

    # Reduce the data
    reduce_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        toast.timing.dump(alltimers, out)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
