#!/usr/bin/env python3

# Copyright (c) 2019-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs an SO time domain simulation.

You can see the automatically generated command line options with:

    toast_sim_ground.py --help

Or you can dump a config file with all the default values with:

    toast_sim_ground.py --default_toml config.toml

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

import toast
import toast.ops

from toast.mpi import MPI

import sotodlib.toast as sotoast
import sotodlib.toast.ops as so_ops


def parse_config(operators, templates, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="SO simulation pipeline")

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
        default="toast_out",
        help="The output directory",
    )

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
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


def use_full_pointing(job):
    # Are we using full pointing?  We determine this from whether the binning operator
    # used in the solve has full pointing enabled and also whether madam (which
    # requires full pointing) is enabled.
    full_pointing = False
    if toast.ops.madam.available() and job.operators.madam.enabled:
        full_pointing = True
    if job.operators.binner.full_pointing:
        full_pointing = True
    return full_pointing


def job_create(config, jobargs, telescope, schedule, comm):
    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.
    full_pointing = use_full_pointing(job)
    group_size = toast.job_group_size(
        comm,
        jobargs,
        schedule=schedule,
        focalplane=telescope.focalplane,
        full_pointing=full_pointing,
    )
    return job, group_size, full_pointing


def simulate_data(job, toast_comm, telescope, schedule):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates
    world_comm = toast_comm.comm_world

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Simulate the telescope pointing

    ops.sim_ground.telescope = telescope
    ops.sim_ground.schedule = schedule
    if ops.sim_ground.weather is None:
        ops.sim_ground.weather = telescope.site.name
    ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    # Apply LAT co-rotation
    ops.corotate_lat.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Set up detector pointing in both Az/El and RA/DEC

    ops.det_pointing_azel.boresight = ops.sim_ground.boresight_azel
    ops.det_pointing_radec.boresight = ops.sim_ground.boresight_radec

    ops.det_weights_azel.detector_pointing = ops.det_pointing_azel
    ops.det_weights_azel.hwp_angle = ops.sim_ground.hwp_angle

    # Create the Elevation modulated noise model

    ops.elevation_model.noise_model = ops.default_model.noise_model
    ops.elevation_model.detector_pointing = ops.det_pointing_azel
    ops.elevation_model.view = ops.det_pointing_azel.view
    ops.elevation_model.apply(data)
    log.info_rank("Created elevation noise model in", comm=world_comm, timer=timer)

    # Set up the pointing.  Each pointing matrix operator requires a detector pointing
    # operator, and each binning operator requires a pointing matrix operator.
    ops.pointing.detector_pointing = ops.det_pointing_radec
    ops.pointing.hwp_angle = ops.sim_ground.hwp_angle
    ops.pointing_final.detector_pointing = ops.det_pointing_radec
    ops.pointing_final.hwp_angle = ops.sim_ground.hwp_angle

    ops.binner.pointing = ops.pointing

    # If we are not using a different pointing matrix for our final binning, then
    # use the same one as the solve.
    if not ops.pointing_final.enabled:
        ops.pointing_final = ops.pointing

    ops.binner_final.pointing = ops.pointing_final

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    if not ops.binner_final.enabled:
        ops.binner_final = ops.binner

    # Simulate sky signal from a map.  We scan the sky with the "final" pointing model
    # in case that is different from the solver pointing model.

    ops.scan_map.pixel_dist = ops.binner_final.pixel_dist
    ops.scan_map.pointing = ops.pointing_final
    ops.scan_map.save_pointing = use_full_pointing(job)
    ops.scan_map.apply(data)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    # Simulate atmosphere

    ops.sim_atmosphere.detector_pointing = ops.det_pointing_azel
    if ops.sim_atmosphere.polarization_fraction != 0:
        ops.sim_atmosphere.detector_weights = ops.det_weights_azel
    ops.sim_atmosphere.apply(data)
    log.info_rank("Simulated and observed atmosphere in", comm=world_comm, timer=timer)

    # Simulate Solar System Objects

    ops.sim_sso.detector_pointing = ops.det_pointing_azel
    ops.sim_sso.apply(data)
    log.info_rank(
        "Simulated and observed solar system objects",
        comm=world_comm,
        timer=timer,
    )

    # Apply a time constant

    ops.convolve_time_constant.apply(data)
    log.info_rank("Convolved time constant in", comm=world_comm, timer=timer)

    # Simulate detector noise

    ops.sim_noise.noise_model = ops.elevation_model.out_model
    ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    # Simulate HWP-synchronous signal

    #ops.sim_hwpss.detector_pointing = ops.det_pointing_azel
    ops.sim_hwpss.detector_weights = ops.det_weights_azel
    ops.sim_hwpss.apply(data)
    log.info_rank(
        "Simulated HWP-synchronous signal",
        comm=world_comm,
        timer=timer,
    )

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Flag Sun, Moon and the planets

    ops.flag_sso.detector_pointing = ops.det_pointing_azel
    ops.flag_sso.apply(data)
    log.info_rank("Flagged SSOs in", comm=world_comm, timer=timer)

    # Optional geometric factors

    ops.cadence_map.pointing = ops.pointing_final
    ops.cadence_map.pixel_dist = ops.binner_final.pixel_dist
    ops.cadence_map.output_dir = args.out_dir
    ops.cadence_map.apply(data)
    log.info_rank("Calculated cadence map in", comm=world_comm, timer=timer)

    ops.crosslinking.pointing = ops.pointing_final
    ops.crosslinking.pixel_dist = ops.binner_final.pixel_dist
    ops.crosslinking.output_dir = args.out_dir
    ops.crosslinking.apply(data)
    log.info_rank("Calculated crosslinking in", comm=world_comm, timer=timer)

    # Collect signal statistics before filtering

    ops.raw_statistics.output_dir = args.out_dir
    ops.raw_statistics.apply(data)
    log.info_rank("Calculated raw statistics in", comm=world_comm, timer=timer)

    # Deconvolve a time constant

    ops.deconvolve_time_constant.apply(data)
    log.info_rank("Deconvolved time constant in", comm=world_comm, timer=timer)

    # Apply the filter stack

    ops.groundfilter.apply(data)
    log.info_rank("Finished ground-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter1D.apply(data)
    log.info_rank("Finished 1D-poly-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter2D.apply(data)
    log.info_rank("Finished 2D-poly-filtering in", comm=world_comm, timer=timer)
    ops.common_mode_filter.apply(data)
    log.info_rank("Finished common-mode-filtering in", comm=world_comm, timer=timer)

    # Collect signal statistics after filtering

    ops.filtered_statistics.output_dir = args.out_dir
    ops.filtered_statistics.apply(data)
    log.info_rank("Calculated filtered statistics in", comm=world_comm, timer=timer)

    # The map maker requires the the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.binner.noise_model = ops.elevation_model.out_model
    ops.binner_final.noise_model = ops.elevation_model.out_model

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[tmpls.baselines])
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = ops.sim_noise.det_data
    ops.mapmaker.output_dir = args.out_dir

    ops.mapmaker.apply(data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    # Optionally run Madam
    if toast.ops.madam.available():
        ops.madam.apply(data)
        log.info_rank("Finished Madam in", comm=world_comm, timer=timer)


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
        # In the future, `det_weights_azel` may be a dedicated operator that does not
        # expand pixel numbers but just Stokes weights
        toast.ops.PointingHealpix(
            name="det_weights_azel", weights="weights_azel", mode="IQU"
        ),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        toast.ops.ScanHealpix(name="scan_map", enabled=False),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.SimAtmosphere(name="sim_atmosphere"),
        so_ops.SimSSO(name="sim_sso", enabled=False),
        so_ops.SimHWPSS(name="sim_hwpss", enabled=False),
        toast.ops.TimeConstant(
            name="convolve_time_constant", deconvolve=False, enabled=False
        ),
        toast.ops.PointingHealpix(name="pointing", mode="IQU"),
        toast.ops.FlagSSO(name="flag_sso", enabled=False),
        toast.ops.CadenceMap(name="cadence_map", enabled=False),
        toast.ops.CrossLinking(name="crosslinking", enabled=False),
        toast.ops.Statistics(name="raw_statistics", enabled=False),
        toast.ops.TimeConstant(
            name="deconvolve_time_constant", deconvolve=True, enabled=False
        ),
        toast.ops.GroundFilter(name="groundfilter", enabled=False),
        toast.ops.PolyFilter(name="polyfilter1D"),
        toast.ops.PolyFilter2D(name="polyfilter2D", enabled=False),
        toast.ops.CommonModeFilter(name="common_mode_filter", enabled=False),
        toast.ops.Statistics(name="filtered_statistics", enabled=False),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PointingHealpix(name="pointing_final", enabled=False, mode="IQU"),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
    ]
    if toast.ops.madam.available():
        operators.append(toast.ops.Madam(name="madam", enabled=False))

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # Parse options
    config, args, jobargs = parse_config(operators, templates, comm)

    # Load our instrument model and observing schedule
    telescope, schedule = load_instrument_and_schedule(args, comm)

    # Instantiate our operators and get the size of the process groups
    job, group_size, full_pointing = job_create(
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
