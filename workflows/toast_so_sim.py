#!/usr/bin/env python3

# Copyright (c) 2019-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs an SO time domain simulation.

NOTE:  This script is an example.  If you are doing a simulation for a specific
experiment, you should use a custom Focalplane class rather that the simple base class
used here.

You can see the automatically generated command line options with:

    toast_so_sim.py --help

Or you can dump a config file with all the default values with:

    toast_so_sim.py --default_toml config.toml

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

import sotodlib.toast.ops as so_ops
import sotodlib.mapmaking

# Make sure pixell uses a reliable FFT engine
import pixell.fft
pixell.fft.engine = "fftw"


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
        "--det_info_file",
        required=False,
        default=None,
        help="Input detector info file for real hardware maps",
    )
    parser.add_argument(
        "--det_info_version",
        required=False,
        default=None,
        help="Detector info file version such as 'Cv4'",
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
        default=None,
        help="Telescope to simulate: LAT, SAT1, SAT2, SAT3, SAT4.",
    )
    group.add_argument(
        "--tube_slots",
        default=None,
        help="Comma-separated list of optics tube slots: c1 (LAT_UHF), i5 (LAT_UHF), "
        " i6 (LAT_MF), i1 (LAT_MF), i3 (LAT_MF), i4 (LAT_MF), o6 (LAT_LF),"
        " ST1 (SAT_MF), ST2 (SAT_MF), ST3 (SAT_UHF), ST4 (SAT_LF)."
    )
    group.add_argument(
        "--wafer_slots",
        default=None,
        help="Comma-separated list of wafer slots. "
    )

    parser.add_argument(
        "--sample_rate", required=False, default=10, help="Sampling rate", type=float,
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

    parser.add_argument(
        "--save_books",
        required=False,
        default=False,
        action="store_true",
        help="Save simulated data to L3 book format.",
    )

    parser.add_argument(
        "--save_hdf5",
        required=False,
        default=False,
        action="store_true",
        help="Save simulated data to TOAST native HDF5 format.",
    )

    parser.add_argument(
        "--obsmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each observation separately.",
    )

    parser.add_argument(
        "--realization", required=False, help="Realization index", type=int,
    )

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    if comm is not None:
        comm.barrier()
    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    return config, args, jobargs


def load_instrument_and_schedule(args, comm):
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    telescope = sotoast.simulated_telescope(
        hwfile=args.hardware,
        det_info_file=args.det_info_file,
        det_info_version=args.det_info_version,
        telescope_name=args.telescope,
        sample_rate=args.sample_rate * u.Hz,
        bands=args.bands,
        wafer_slots=args.wafer_slots,
        tube_slots=args.tube_slots,
        thinfp=args.thinfp,
        comm=comm,
    )
    ndet = len(telescope.focalplane.detectors)
    log.info_rank(f"Loaded focalplane with {ndet} detectors in", comm=comm, timer=timer)
    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After loading focalplane:  {mem}", comm)

    # Load the schedule file
    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule, comm=comm)
    log.info_rank("Loaded schedule in", comm=comm, timer=timer)
    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After loading schedule:  {mem}", comm)

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


def select_pointing(job, args):
    """Select the pixelization scheme for both the solver and final binning."""

    ops = job.operators

    n_enabled_solve = np.sum(
        [
            ops.pixels_wcs_azel.enabled,
            ops.pixels_wcs_radec.enabled,
            ops.pixels_healpix_radec.enabled,
        ]
    )
    if n_enabled_solve != 1:
        raise RuntimeError(
            "Only one pixelization operator should be enabled for the solver."
        )

    n_enabled_final = np.sum(
        [
            ops.pixels_wcs_azel_final.enabled,
            ops.pixels_wcs_radec_final.enabled,
            ops.pixels_healpix_radec_final.enabled,
        ]
    )
    if n_enabled_final > 1:
        raise RuntimeError(
            "At most, one pixelization operator can be enabled for the final binning."
        )

    # Configure Az/El and RA/DEC boresight and detector pointing and weights

    ops.det_pointing_azel.boresight = ops.sim_ground.boresight_azel
    ops.det_pointing_radec.boresight = ops.sim_ground.boresight_radec

    ops.pixels_wcs_azel.detector_pointing = ops.det_pointing_azel
    ops.pixels_wcs_radec.detector_pointing = ops.det_pointing_radec
    ops.pixels_healpix_radec.detector_pointing = ops.det_pointing_radec

    ops.pixels_wcs_azel_final.detector_pointing = ops.det_pointing_azel
    ops.pixels_wcs_radec_final.detector_pointing = ops.det_pointing_radec
    ops.pixels_healpix_radec_final.detector_pointing = ops.det_pointing_radec

    ops.weights_azel.detector_pointing = ops.det_pointing_azel
    ops.weights_azel.hwp_angle = ops.sim_ground.hwp_angle

    ops.weights_radec.detector_pointing = ops.det_pointing_radec
    ops.weights_radec.hwp_angle = ops.sim_ground.hwp_angle

    # Select Pixelization and weights for solve and final binning

    if ops.pixels_wcs_azel.enabled:
        job.pixels_solve = ops.pixels_wcs_azel
        job.weights_solve = ops.weights_azel
    elif ops.pixels_wcs_radec.enabled:
        job.pixels_solve = ops.pixels_wcs_radec
        job.weights_solve = ops.weights_radec
    else:
        job.pixels_solve = ops.pixels_healpix_radec
        job.weights_solve = ops.weights_radec
    job.weights_final = job.weights_solve

    if n_enabled_final == 0:
        # Use same as solve
        job.pixels_final = job.pixels_solve
    else:
        if ops.pixels_wcs_azel_final.enabled:
            job.pixels_final = ops.pixels_wcs_azel_final
        elif ops.pixels_wcs_radec_final.enabled:
            job.pixels_final = ops.pixels_wcs_radec_final
        else:
            job.pixels_final = ops.pixels_healpix_radec_final


def simulate_data(job, args, toast_comm, telescope, schedule):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates
    world_comm = toast_comm.comm_world

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    ops.mem_count.prefix = "Before Simulation"
    ops.mem_count.apply(data)

    # Simulate the telescope pointing

    ops.sim_ground.telescope = telescope
    ops.sim_ground.schedule = schedule
    if ops.sim_ground.weather is None:
        ops.sim_ground.weather = telescope.site.name
    if args.realization is not None:
        ops.sim_ground.realization = args.realization
    ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After Scan Simulation"
    ops.mem_count.apply(data)

    # Apply LAT co-rotation
    ops.corotate_lat.apply(data)
    log.info_rank("Apply LAT co-rotation in", comm=world_comm, timer=timer)

    # Perturb HWP spin
    ops.perturb_hwp.apply(data)
    log.info_rank("Perturbed HWP rotation in", comm=world_comm, timer=timer)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After default noise model"
    ops.mem_count.apply(data)

    # Set up pointing, pixelization, and weights

    select_pointing(job, args)

    # Create the Elevation modulated noise model

    ops.elevation_model.detector_pointing = ops.det_pointing_azel
    ops.elevation_model.apply(data)
    log.info_rank("Created elevation noise model in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After elevation noise model"
    ops.mem_count.apply(data)

    # Add common noise modes

    ops.common_mode_noise.apply(data)
    log.info_rank("Added common mode noise model in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After common mode noise model"
    ops.mem_count.apply(data)

    # Set up pointing matrices for binning operators

    ops.binner.pixel_pointing = job.pixels_solve
    ops.binner.stokes_weights = job.weights_solve

    ops.binner_final.pixel_pointing = job.pixels_final
    ops.binner_final.stokes_weights = job.weights_final

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    if not ops.binner_final.enabled:
        ops.binner_final = ops.binner

    # Simulate atmosphere

    if args.realization is not None:
        ops.sim_atmosphere_coarse.realization = 1000000 + args.realization
        ops.sim_atmosphere.realization = args.realization

    for sim_atm in ops.sim_atmosphere_coarse, ops.sim_atmosphere:
        if not sim_atm.enabled:
            continue
        sim_atm.detector_pointing = ops.det_pointing_azel
        if sim_atm.polarization_fraction != 0:
            sim_atm.detector_weights = ops.weights_azel
        log.info_rank(f"Applying {sim_atm.name}", comm=world_comm)
        sim_atm.apply(data)
        log.info_rank(f"Applied {sim_atm.name} in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = f"After {sim_atm.name}"
        ops.mem_count.apply(data)

    # Shortcut if we are only caching the atmosphere.  If this job is only caching
    # (not observing) the atmosphere, then return at this point.
    if ops.sim_atmosphere.cache_only:
        return data

    # Simulate sky signal from a map.  We scan the sky with the "final" pointing model
    # in case that is different from the solver pointing model.

    ops.scan_map.pixel_dist = ops.binner_final.pixel_dist
    ops.scan_map.pixel_pointing = job.pixels_final
    ops.scan_map.stokes_weights = ops.weights_radec
    ops.scan_map.save_pointing = use_full_pointing(job)
    log.info_rank("Simulating sky signal", comm=world_comm)
    ops.scan_map.apply(data)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After simulating sky signal"
    ops.mem_count.apply(data)

    # Use the Conviqt operator to generate beam-convolved TOD

    if ops.sim_ground.hwp_angle is None:
        ops.conviqt.comm = world_comm
        ops.conviqt.detector_pointing = ops.det_pointing_radec
        ops.conviqt.apply(data)
    else:
        ops.conviqt_teb.comm = world_comm
        ops.conviqt_teb.detector_pointing = ops.det_pointing_radec
        ops.conviqt_teb.hwp_angle = ops.sim_ground.hwp_angle
        ops.conviqt_teb.apply(data)

    ops.mem_count.prefix = "After 4-pi beam convolution"
    ops.mem_count.apply(data)

    # Simulate scan-synchronous signal

    ops.sim_sss.detector_pointing = ops.det_pointing_azel
    if args.realization is not None:
        ops.sim_sss.realization = args.realization
    ops.sim_sss.apply(data)
    log.info_rank("Simulated Scan-synchronous signal", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After simulating scan-synchronous signal"
    ops.mem_count.apply(data)

    # Simulate Artificial Source

    ops.sim_source.detector_pointing = ops.det_pointing_azel
    ops.sim_source.focalplane = telescope.focalplane
    if ops.sim_source.polarization_fraction != 0:
        ops.sim_source.detector_weights = ops.weights_azel
    ops.sim_source.apply(data)
    log.info_rank(
        "Simulated and observed artificial source",
        comm=world_comm,
        timer=timer,
    )

    # Simulate Solar System Objects

    ops.sim_sso.detector_pointing = ops.det_pointing_azel
    ops.sim_sso.detector_weights = ops.weights_radec
    ops.sim_sso.apply(data)
    log.info_rank(
        "Simulated and observed solar system objects",
        comm=world_comm,
        timer=timer,
    )

    # Simulate Solar System Objects

    ops.sim_catalog.detector_pointing = ops.det_pointing_radec
    ops.sim_catalog.apply(data)
    log.info_rank(
        "Simulated and observed catalog",
        comm=world_comm,
        timer=timer,
    )

    # Simulate wiregrid calibration observations

    ops.sim_wiregrid.detector_pointing = ops.det_pointing_azel
    ops.sim_wiregrid.detector_weights = ops.weights_azel
    ops.sim_wiregrid.apply(data)
    log.info_rank("Simulated wiregrid in", comm=world_comm, timer=timer)

    # Simulate stimulator calibration observations

    ops.sim_stimulator.apply(data)
    log.info_rank("Simulated stimulator in", comm=world_comm, timer=timer)

    # Apply a time constant

    if args.realization is not None:
        ops.convolve_time_constant.realization = args.realization
    ops.convolve_time_constant.apply(data)
    log.info_rank("Convolved time constant in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After applying time constant"
    ops.mem_count.apply(data)

    # Simulate detector noise

    if args.realization is not None:
        ops.sim_noise.realization = args.realization
    log.info_rank("Simulating detector noise", comm=world_comm)
    ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After simulating noise"
    ops.mem_count.apply(data)

    # Simulate HWP-synchronous signal

    if ops.sim_ground.hwp_angle is not None:
        ops.sim_hwpss.hwp_angle = ops.sim_ground.hwp_angle
        ops.sim_hwpss.detector_pointing = ops.det_pointing_azel
        ops.sim_hwpss.stokes_weights = ops.weights_azel
        ops.sim_hwpss.apply(data)
        log.info_rank(
            "Simulated HWP-synchronous signal",
            comm=world_comm,
            timer=timer,
        )

    # Add calibration error

    if args.realization is not None:
        ops.gainscrambler.realization = args.realization
    ops.gainscrambler.apply(data)
    log.info_rank("Simulated gain errors in", comm=world_comm, timer=timer)

    # Add readout systematics

    ops.sim_readout.apply(data)
    log.info_rank("Simulated readout systematics in", comm=world_comm, timer=timer)

    # Optionally write out the data
    if args.out_dir is not None:
        hdf5_path = os.path.join(args.out_dir, "data")
        if toast_comm.world_rank == 0:
            if not os.path.isdir(hdf5_path):
                os.makedirs(hdf5_path)
        ops.save_hdf5.volume = hdf5_path
        ops.save_books.book_dir = os.path.join(args.out_dir, "books")
    ops.save_hdf5.apply(data)
    log.info_rank("Saved HDF5 data in", comm=world_comm, timer=timer)
    ops.save_books.apply(data)
    log.info_rank("Saved book data in", comm=world_comm, timer=timer)

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Apply demodulation,
    # optional HWP filter must apply before demodulation

    ops.hwpfilter.apply(data)
    log.info_rank("HWP-filtered in", comm=world_comm, timer=timer)

    if ops.demodulate.enabled:
        # The Demodulation operator is special because it returns a
        # new TOAST data object
        ops.demodulate.stokes_weights = ops.weights_radec
        ops.demodulate.hwp_angle = ops.sim_ground.hwp_angle
        data = ops.demodulate.apply(data)
        log.info_rank("Demodulated in", comm=world_comm, timer=timer)
        demod_weights = toast.ops.StokesWeightsDemod()
        ops.weights_radec = demod_weights
        ops.binner.stokes_weights = demod_weights
        ops.binner_final.stokes_weights = demod_weights

    # Apply noise estimation

    ops.noise_estim.detector_pointing = job.pixels_final.detector_pointing
    ops.noise_estim.pixel_pointing = job.pixels_final
    ops.noise_estim.stokes_weights = job.weights_final
    ops.noise_estim.pixel_dist = ops.binner_final.pixel_dist
    ops.noise_estim.output_dir = args.out_dir
    ops.noise_estim.apply(data)
    log.info_rank("Estimated noise in", comm=world_comm, timer=timer)

    # Flag Sun, Moon and the planets

    ops.flag_sso.detector_pointing = ops.det_pointing_azel
    log.info_rank("Flagging SSOs", comm=world_comm)
    ops.flag_sso.apply(data)
    log.info_rank("Flagged SSOs in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After flagging SSOs"
    ops.mem_count.apply(data)

    # Optional geometric factors

    ops.h_n.pixel_pointing = job.pixels_final
    ops.h_n.pixel_dist = ops.binner_final.pixel_dist
    ops.h_n.output_dir = args.out_dir
    ops.h_n.apply(data)
    log.info_rank("Calculated h_n in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After h_n map"
    ops.mem_count.apply(data)

    ops.cadence_map.pixel_pointing = job.pixels_final
    ops.cadence_map.pixel_dist = ops.binner_final.pixel_dist
    ops.cadence_map.output_dir = args.out_dir
    ops.cadence_map.apply(data)
    log.info_rank("Calculated cadence map in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After cadence map"
    ops.mem_count.apply(data)

    ops.crosslinking.pixel_pointing = job.pixels_final
    ops.crosslinking.pixel_dist = ops.binner_final.pixel_dist
    ops.crosslinking.output_dir = args.out_dir
    ops.crosslinking.apply(data)
    log.info_rank("Calculated crosslinking in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After crosslinking map"
    ops.mem_count.apply(data)

    # Collect signal statistics before filtering

    ops.raw_statistics.output_dir = args.out_dir
    ops.raw_statistics.apply(data)
    log.info_rank("Calculated raw statistics in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After raw statistics"
    ops.mem_count.apply(data)

    # Deconvolve a time constant

    ops.deconvolve_time_constant.apply(data)
    log.info_rank("Deconvolved time constant in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After deconvolving time constant"
    ops.mem_count.apply(data)

    # Run ML mapmaker

    ops.mlmapmaker.out_dir = args.out_dir
    ops.mlmapmaker.apply(data)
    log.info_rank("Finished ML map-making in", comm=world_comm, timer=timer)

    # Apply the filter stack

    log.info_rank("Filtering signal", comm=world_comm)
    ops.groundfilter.apply(data)
    log.info_rank("Finished ground-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter1D.apply(data)
    log.info_rank("Finished 1D-poly-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter2D.apply(data)
    log.info_rank("Finished 2D-poly-filtering in", comm=world_comm, timer=timer)
    ops.common_mode_filter.apply(data)
    log.info_rank("Finished common-mode-filtering in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After filtering"
    ops.mem_count.apply(data)

    # The map maker requires the the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[tmpls.baselines])
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = ops.sim_noise.det_data
    ops.mapmaker.output_dir = args.out_dir

    ops.filterbin.binning = ops.binner_final
    ops.filterbin.det_data = ops.sim_noise.det_data
    ops.filterbin.output_dir = args.out_dir

    log.info_rank("Making maps", comm=world_comm)
    if args.obsmaps:
        # Map each observation separately
        timer_obs = toast.timing.Timer()
        timer_obs.start()
        group = data.comm.group
        orig_name = ops.mapmaker.name
        orig_name_filterbin = ops.filterbin.name
        orig_comm = data.comm
        new_comm = Comm(world=data.comm.comm_group)
        for iobs, obs in enumerate(data.obs):
            log.info_rank(
                f"{group} : mapping observation {iobs + 1} / {len(data.obs)}.",
                comm=new_comm.comm_world,
            )
            # Data object that only covers one observation
            obs_data = data.select(obs_uid=obs.uid)
            # Replace comm_world with the group communicator
            obs_data._comm = new_comm
            ops.mapmaker.name = f"{orig_name}_{obs.name}"
            ops.mapmaker.reset_pix_dist = True
            ops.mapmaker.apply(obs_data)
            log.info_rank(
                f"{group} : Mapped {obs.name} in",
                comm=new_comm.comm_world,
                timer=timer_obs,
            )
            ops.filterbin.name = f"{orig_name_filterbin}_{obs.name}"
            ops.filterbin.reset_pix_dist = True
            ops.filterbin.apply(obs_data)
            log.info_rank(
                f"{group} : Filter+binned {obs.name} in",
                comm=new_comm.comm_world,
                timer=timer_obs,
            )
        log.info_rank(
            f"{group} : Done mapping {len(data.obs)} observations.",
            comm=new_comm.comm_world,
        )
        data._comm = orig_comm
    else:
        ops.mapmaker.apply(data)
        ops.filterbin.apply(data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After mapmaker"
    ops.mem_count.apply(data)

    # Optionally run Madam

    if toast.ops.madam.available() and ops.madam.enabled:
        ops.madam.params = toast.ops.madam_params_from_mapmaker(ops.mapmaker)
        ops.madam.pixel_pointing = job.pixels_final
        ops.madam.stokes_weights = ops.weights_radec
        ops.madam.apply(data)
        log.info_rank("Finished Madam in", comm=world_comm, timer=timer)

        ops.mem_count.prefix = "After Madam"
        ops.mem_count.apply(data)

    # Collect signal statistics after filtering/destriping

    ops.filtered_statistics.output_dir = args.out_dir
    ops.filtered_statistics.apply(data)
    log.info_rank("Calculated filtered statistics in", comm=world_comm, timer=timer)

    ops.mem_count.prefix = "After filtered statistics"
    ops.mem_count.apply(data)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_sim (total)")
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
    log.info_rank(f"Start of the workflow:  {mem}", comm)

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    #
    # We can also set some default values here for the traits, including whether an
    # operator is disabled by default.

    operators = [
        toast.ops.SimGround(
            name="sim_ground",
            weather="atacama",
            detset_key="pixel",
            session_split_key="tube_slot",
        ),
        so_ops.CoRotator(name="corotate_lat"),
        toast.ops.PerturbHWP(name="perturb_hwp", enabled=False),
        toast.ops.DefaultNoiseModel(name="default_model", noise_model="noise_model"),
        toast.ops.ElevationNoise(name="elevation_model", out_model="noise_model"),
        toast.ops.CommonModeNoise(name="common_mode_noise", enabled=False),
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel"),
        toast.ops.StokesWeights(
            name="weights_azel", weights="weights_azel", mode="IQU"
        ),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        toast.ops.ScanHealpixMap(name="scan_map", enabled=False),
        toast.ops.SimConviqt(name="conviqt", enabled=False),
        toast.ops.SimTEBConviqt(name="conviqt_teb", enabled=False),
        toast.ops.SimAtmosphere(
            name="sim_atmosphere_coarse",
            add_loading=False,
            lmin_center=300 * u.m,
            lmin_sigma=30 * u.m,
            lmax_center=10000 * u.m,
            lmax_sigma=1000 * u.m,
            xstep=50 * u.m,
            ystep=50 * u.m,
            zstep=50 * u.m,
            zmax=2000 * u.m,
            nelem_sim_max=30000,
            gain=6e-4,
            realization=1000000,
            wind_dist=10000 * u.m,
            enabled=False,
        ),
        toast.ops.SimAtmosphere(
            name="sim_atmosphere",
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=5 * u.m,
            ystep=5 * u.m,
            zstep=5 * u.m,
            zmax=200 * u.m,
            gain=6e-5,
            wind_dist=3000 * u.m,
            enabled=False,
        ),
        toast.ops.SimScanSynchronousSignal(name="sim_sss", enabled=False),
        so_ops.SimSSO(name="sim_sso", enabled=False),
        so_ops.SimCatalog(name="sim_catalog", enabled=False),
        so_ops.SimSource(name="sim_source", enabled=False),
        so_ops.SimHWPSS(name="sim_hwpss", enabled=False),
        so_ops.SimWireGrid(name="sim_wiregrid", enabled=False),
        so_ops.SimStimulator(name="sim_stimulator", enabled=False),
        toast.ops.TimeConstant(
            name="convolve_time_constant", deconvolve=False, enabled=False
        ),
        toast.ops.SaveHDF5(name="save_hdf5", enabled=False),
        so_ops.SaveBooks(name="save_books", enabled=False),
        toast.ops.GainScrambler(name="gainscrambler", enabled=False),
        toast.ops.SimNoise(name="sim_noise"),
        so_ops.SimReadout(name="sim_readout", enabled=False),
        toast.ops.PixelsHealpix(name="pixels_healpix_radec"),
        toast.ops.PixelsWCS(
            name="pixels_wcs_radec",
            project="CAR",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            auto_bounds=True,
            enabled=False,
        ),
        toast.ops.PixelsWCS(
            name="pixels_wcs_azel",
            project="CAR",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            auto_bounds=True,
            enabled=False,
        ),
        toast.ops.StokesWeights(name="weights_radec", mode="IQU"),
        toast.ops.Demodulate(name="demodulate", enabled=False),
        toast.ops.NoiseEstim(name="noise_estim", enabled=False),
        toast.ops.FlagSSO(name="flag_sso", enabled=False),
        so_ops.Hn(name="h_n", enabled=False),
        toast.ops.CadenceMap(name="cadence_map", enabled=False),
        toast.ops.CrossLinking(name="crosslinking", enabled=False),
        toast.ops.Statistics(name="raw_statistics", enabled=False),
        toast.ops.TimeConstant(
            name="deconvolve_time_constant", deconvolve=True, enabled=False
        ),
        toast.ops.HWPFilter(name="hwpfilter", enabled=False),
        toast.ops.GroundFilter(name="groundfilter", enabled=False),
        toast.ops.PolyFilter(name="polyfilter1D"),
        toast.ops.PolyFilter2D(name="polyfilter2D", enabled=False),
        toast.ops.CommonModeFilter(name="common_mode_filter", enabled=False),
        toast.ops.Statistics(name="filtered_statistics", enabled=False),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PixelsHealpix(name="pixels_healpix_radec_final", enabled=False),
        toast.ops.PixelsWCS(name="pixels_wcs_radec_final", enabled=False),
        toast.ops.PixelsWCS(name="pixels_wcs_azel_final", enabled=False),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
        toast.ops.FilterBin(name="filterbin", enabled=False),
        so_ops.MLMapmaker(name="mlmapmaker", enabled=False, comps="TQU"),
        toast.ops.MemoryCounter(name="mem_count", enabled=False),
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
    data = simulate_data(job, args, toast_comm, telescope, schedule)

    # Handle special case of caching the atmosphere simulations.  In this
    # case, we are not simulating timestreams or doing data reductions.

    if not job.operators.sim_atmosphere.cache_only:
        # Reduce the data
        reduce_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    log.info_rank("Workflow completed in", comm=comm, timer=timer0)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
