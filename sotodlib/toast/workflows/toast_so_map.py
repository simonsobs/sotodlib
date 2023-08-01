#!/usr/bin/env python3

# Copyright (c) 2019-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This workflow runs basic data reduction and map-making:

- A single MPI process group is used

- All input observations are loaded and made into the output map

In particular, this script is designed for testing mapmaking techniques on data that
already exists on disk.  It does not simulate data and it is not a full workflow for
running null tests, building observation matrices, etc.

You can see the automatically generated command line options with:

    toast_so_map.py --help

Or you can dump a config file with all the default values with:

    toast_so_map.py --default_toml config.toml

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

# Make sure pixell uses a reliable FFT engine
import pixell.fft

pixell.fft.engine = "fftw"


def parse_config(operators, templates, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Make maps of SO data")

    # Arguments specific to this script

    parser.add_argument(
        "--obs_hdf5",
        required=False,
        action="extend",
        nargs="*",
        help="Path to a TOAST hdf5 observation dump (can use multiple times)",
    )

    parser.add_argument(
        "--obs_book",
        required=False,
        action="extend",
        nargs="*",
        help="Path to a L3 book directory (can use multiple times)",
    )

    parser.add_argument(
        "--obs_raw",
        required=False,
        action="extend",
        nargs="*",
        help="Path to raw data directory (can use multiple times)",
    )

    parser.add_argument(
        "--band",
        required=False,
        default=None,
        help="Only use detectors from this band (e.g. LAT_f150, SAT_f040)",
    )

    parser.add_argument(
        "--wafer_slots",
        required=False,
        default=None,
        help="Comma-separated list of wafer slots to use. ",
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="output_maps",
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
            os.makedirs(args.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    return config, args, jobargs


def use_full_pointing(job):
    # Are we using full pointing?  We determine this from whether the binning operator
    # used in the solve has full pointing enabled.
    full_pointing = False
    if job.operators.binner.full_pointing:
        full_pointing = True
    return full_pointing


def job_create(config, comm):
    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)

    # For this workflow, we will just use one process group
    full_pointing = use_full_pointing(job)
    if comm is None:
        group_size = 1
    else:
        group_size = comm.size
    return job, group_size, full_pointing


def select_pointing(job, args, data):
    """Select the pixelization scheme for both the solver and final binning."""
    log = toast.utils.Logger.get()

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

    ops.det_pointing_azel.boresight = defaults.boresight_azel
    ops.det_pointing_radec.boresight = defaults.boresight_radec

    ops.pixels_wcs_azel.detector_pointing = ops.det_pointing_azel
    ops.pixels_wcs_radec.detector_pointing = ops.det_pointing_radec
    ops.pixels_healpix_radec.detector_pointing = ops.det_pointing_radec

    ops.pixels_wcs_azel_final.detector_pointing = ops.det_pointing_azel
    ops.pixels_wcs_radec_final.detector_pointing = ops.det_pointing_radec
    ops.pixels_healpix_radec_final.detector_pointing = ops.det_pointing_radec

    ops.weights_azel.detector_pointing = ops.det_pointing_azel
    ops.weights_radec.detector_pointing = ops.det_pointing_radec

    if job.has_HWP:
        ops.weights_azel.hwp_angle = defaults.hwp_angle
        ops.weights_radec.hwp_angle = defaults.hwp_angle

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
    log.info_rank(
        f"Template solve using pixelization: {job.pixels_solve.name}",
        comm=data.comm.comm_world,
    )
    log.info_rank(
        f"Template solve using weights: {job.weights_solve.name}",
        comm=data.comm.comm_world,
    )
    log.info_rank(
        f"Final binning using pixelization: {job.pixels_final.name}",
        comm=data.comm.comm_world,
    )
    log.info_rank(
        f"Final binning using weights: {job.weights_final.name}",
        comm=data.comm.comm_world,
    )


def load_data(job, args, toast_comm):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    ops.mem_count.prefix = "Before Data Load"
    ops.mem_count.apply(data)

    # Load all of our toast HDF5 datasets

    job.has_HWP = False
    for hobs in list(args.obs_hdf5):
        log.info_rank(f"Starting load of HDF5 data {hobs}", comm=data.comm.comm_group)
        ob = toast.io.load_hdf5(
            hobs,
            toast_comm,
            process_rows=toast_comm.group_size,
            meta=None,
            detdata=None,
            shared=None,
            intervals=None,
            force_serial=True,
        )
        if defaults.hwp_angle in ob.shared:
            job.has_HWP = True
        # print("boresight_radec:  ", ob.shared["boresight_radec"].data)
        # print("boresight_azel:  ", ob.shared["boresight_azel"].data)
        # print("shared_flags:  ", ob.shared["flags"].data)
        # bad = (ob.shared["flags"].data & defaults.shared_mask_invalid) != 0
        # print("shared_flags invalid:  ", np.count_nonzero(bad))
        # print("det_flags:  ", ob.detdata["flags"].data)
        # print("noise:  ", ob["noise_model"])
        data.obs.append(ob)
        log.info_rank(
            f"Finished load of HDF5 data {hobs} in",
            comm=data.comm.comm_group,
            timer=timer,
        )
        ops.mem_count.prefix = f"After Loading {hobs}"
        ops.mem_count.apply(data)

    # Load all of our book directories
    if args.obs_book is not None and len(args.obs_book) > 0:
        raise NotImplementedError("Book loading not supported until PR #183 is merged")

    # Load raw data
    if args.obs_raw is not None and len(args.obs_raw) > 0:
        raise NotImplementedError("Raw loading not supported until PR #183 is merged")

    if len(data.obs) == 0:
        raise RuntimeError("No input data specified!")

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Set up pointing, pixelization, and weights

    select_pointing(job, args, data)

    # Set up pointing matrices for binning operators

    ops.binner.pixel_pointing = job.pixels_solve
    ops.binner.stokes_weights = job.weights_solve

    ops.binner_final.pixel_pointing = job.pixels_final
    ops.binner_final.stokes_weights = job.weights_final

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    if not ops.binner_final.enabled:
        ops.binner_final = ops.binner

    # Flag Sun, Moon and the planets

    ops.flag_sso.detector_pointing = ops.det_pointing_azel
    if ops.flag_sso.enabled:
        ops.flag_sso.apply(data)
        log.info_rank("Flagged SSOs in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After flagging SSOs"
        ops.mem_count.apply(data)
    else:
        log.info_rank("SSO Flagging disabled", comm=world_comm)

    # Noise model.  If noise estimation is not enabled, and no existing noise model
    # is found, then create a fake noise model with uniform weighting.

    noise_model = None

    if ops.noise_estim.enabled:
        ops.noise_estim.detector_pointing = job.pixels_final.detector_pointing
        ops.noise_estim.pixel_pointing = job.pixels_final
        ops.noise_estim.stokes_weights = job.weights_final
        ops.noise_estim.pixel_dist = ops.binner_final.pixel_dist
        ops.noise_estim.output_dir = args.out_dir
        ops.noise_estim.apply(data)
        log.info_rank("Estimated noise in", comm=world_comm, timer=timer)

        if ops.noise_fit.enabled:
            ops.noise_fit.apply(data)
            log.info_rank("Fit noise model in", comm=world_comm, timer=timer)
            log.info_rank("Using noise model from 1/f fit", comm=world_comm)
            noise_model = ops.noise_fit.out_model
        else:
            log.info_rank("Using noise model from raw estimate", comm=world_comm)
            noise_model = ops.noise_estim.out_model
    else:
        have_noise = True
        for ob in data.obs:
            if "noise_model" not in ob:
                have_noise = False
        if have_noise:
            log.info_rank("Using noise model from data files", comm=world_comm)
            noise_model = "noise_model"
        else:
            for ob in data.obs:
                (estrate, _, _, _, _) = toast.utils.rate_from_times(
                    ob.shared[defaults.times].data
                )
                ob["fake_noise"] = toast.noise_sim.AnalyticNoise(
                    detectors=ob.all_detectors,
                    rate={x: estrate * u.Hz for x in ob.all_detectors},
                    fmin={x: 1.0e-5 * u.Hz for x in ob.all_detectors},
                    fknee={x: 0.0 * u.Hz for x in ob.all_detectors},
                    alpha={x: 1.0 for x in ob.all_detectors},
                    NET={
                        x: 1.0 * u.K * np.sqrt(1.0 * u.second) for x in ob.all_detectors
                    },
                )
            log.info_rank(
                "Using fake noise model with uniform weighting", comm=world_comm
            )
            noise_model = "fake_noise"
    ops.binner.noise_model = noise_model
    ops.binner_final.noise_model = noise_model

    # Optional geometric factors

    ops.h_n.pixel_pointing = job.pixels_final
    ops.h_n.pixel_dist = ops.binner_final.pixel_dist
    ops.h_n.noise_model = noise_model
    ops.h_n.output_dir = args.out_dir
    if ops.h_n.enabled:
        ops.h_n.apply(data)
        log.info_rank("Calculated h_n in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After h_n map"
        ops.mem_count.apply(data)
    else:
        log.info_rank("H_n map calculation disabled", comm=world_comm)

    ops.cadence_map.pixel_pointing = job.pixels_final
    ops.cadence_map.pixel_dist = ops.binner_final.pixel_dist
    ops.cadence_map.output_dir = args.out_dir
    if ops.cadence_map.enabled:
        ops.cadence_map.apply(data)
        log.info_rank("Calculated cadence map in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After cadence map"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Cadence map calculation disabled", comm=world_comm)

    ops.crosslinking.pixel_pointing = job.pixels_final
    ops.crosslinking.pixel_dist = ops.binner_final.pixel_dist
    ops.crosslinking.output_dir = args.out_dir
    if ops.crosslinking.enabled:
        ops.crosslinking.apply(data)
        log.info_rank("Calculated crosslinking in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After crosslinking map"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Crosslinking map calculation disabled", comm=world_comm)

    # Collect signal statistics before filtering

    ops.raw_statistics.output_dir = args.out_dir
    if ops.raw_statistics.enabled:
        ops.raw_statistics.apply(data)
        log.info_rank("Calculated raw statistics in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After raw statistics"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Raw statistics disabled", comm=world_comm)

    # Deconvolve a time constant

    if ops.deconvolve_time_constant.enabled:
        ops.deconvolve_time_constant.apply(data)
        log.info_rank("Deconvolved time constant in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After deconvolving time constant"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Timeconstant deconvolution disabled", comm=world_comm)

    # Run ML mapmaker

    ops.mlmapmaker.out_dir = args.out_dir
    if ops.mlmapmaker.enabled:
        ops.mlmapmaker.apply(data)
        log.info_rank("Finished ML map-making in", comm=world_comm, timer=timer)
    else:
        log.info_rank("ML map-making disabled", comm=world_comm)

    # Apply the filter stack

    log.info_rank("Begin Filtering", comm=world_comm)

    if ops.groundfilter.enabled:
        ops.groundfilter.apply(data)
        log.info_rank("Finished ground-filtering in", comm=world_comm, timer=timer)
    else:
        log.info_rank("Ground-filtering disabled", comm=world_comm)

    if ops.polyfilter1D.enabled:
        ops.polyfilter1D.apply(data)
        log.info_rank("Finished 1D-poly-filtering in", comm=world_comm, timer=timer)
    else:
        log.info_rank("1D-poly-filtering disabled", comm=world_comm)

    if ops.polyfilter2D.enabled:
        ops.polyfilter2D.apply(data)
        log.info_rank("Finished 2D-poly-filtering in", comm=world_comm, timer=timer)
    else:
        log.info_rank("2D-poly-filtering disabled", comm=world_comm)

    if ops.common_mode_filter.enabled:
        ops.common_mode_filter.apply(data)
        log.info_rank("Finished common-mode-filtering in", comm=world_comm, timer=timer)
    else:
        log.info_rank("common-mode-filtering disabled", comm=world_comm)

    ops.mem_count.prefix = "After filtering"
    ops.mem_count.apply(data)

    # The map maker requires the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.mapmaker.binning = ops.binner

    tmpls.baselines.noise_model = noise_model

    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
        templates=[tmpls.baselines,]
    )
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = defaults.det_data
    ops.mapmaker.output_dir = args.out_dir
    if ops.mapmaker.enabled:
        log.info_rank("Begin generalized destriping map-maker", comm=world_comm)
        # if not tmpls.baselines.enabled and not tmpls.fourier.enabled:
        if not tmpls.baselines.enabled:
            log.info_rank(
                "  No solver templates are enabled- only making a binned map",
                comm=world_comm,
            )
        ops.mapmaker.apply(data)
        log.info_rank("Finished generalized destriper in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After generalized destriping map-maker"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Generalized destriping map-maker disabled", comm=world_comm)

    ops.filterbin.binning = ops.binner_final
    ops.filterbin.det_data = defaults.det_data
    ops.filterbin.output_dir = args.out_dir
    if ops.filterbin.enabled:
        log.info_rank(
            "Begin simultaneous filter/bin map-maker and observation matrix",
            comm=world_comm,
        )
        ops.filterbin.apply(data)
        ops.mem_count.prefix = "After simultaneous filter/bin map-maker"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Simultaneous filter/bin map-maker disabled", comm=world_comm)

    if ops.mlmapmaker.enabled:
        log.info_rank(
            "Begin ML map-maker",
            comm=world_comm,
        )
        ops.mlmapmaker.apply(data)
        log.info_rank("Finished ML map-making in", comm=world_comm, timer=timer)
    else:
        log.info_rank("ML map-maker disabled", comm=world_comm)

    # Collect signal statistics after filtering/destriping

    ops.filtered_statistics.output_dir = args.out_dir
    if ops.filtered_statistics.enabled:
        ops.filtered_statistics.apply(data)
        log.info_rank("Calculated filtered statistics in", comm=world_comm, timer=timer)
        ops.mem_count.prefix = "After filtered statistics"
        ops.mem_count.apply(data)
    else:
        log.info_rank("Filtered statistics disabled", comm=world_comm)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_map (total)")
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
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel"),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        toast.ops.StokesWeights(
            name="weights_azel", weights="weights_azel", mode="IQU"
        ),
        toast.ops.StokesWeights(name="weights_radec", mode="IQU"),
        toast.ops.PixelsHealpix(
            name="pixels_healpix_radec",
            enabled=False,
        ),
        toast.ops.PixelsWCS(
            name="pixels_wcs_radec",
            project="CAR",
            resolution=(0.005 * u.degree, 0.005 * u.degree),
            auto_bounds=True,
            enabled=True,
        ),
        toast.ops.PixelsWCS(
            name="pixels_wcs_azel",
            project="CAR",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            auto_bounds=True,
            enabled=False,
        ),
        toast.ops.NoiseEstim(
            name="noise_estim",
            out_model="estimated_noise",
            enabled=False,
        ),
        toast.ops.FitNoiseModel(
            name="noise_fit",
            noise_model="estimated_noise",
            out_model="estimated_noise_fit",
            enabled=False,
        ),
        toast.ops.FlagSSO(name="flag_sso", enabled=False),
        so_ops.Hn(name="h_n", enabled=False),
        toast.ops.CadenceMap(name="cadence_map", enabled=False),
        toast.ops.CrossLinking(name="crosslinking", enabled=False),
        toast.ops.Statistics(name="raw_statistics", enabled=False),
        toast.ops.TimeConstant(
            name="deconvolve_time_constant", deconvolve=True, enabled=False
        ),
        toast.ops.GroundFilter(name="groundfilter", enabled=False),
        toast.ops.PolyFilter(name="polyfilter1D", enabled=False),
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

    # Templates we want to configure from the command line or a parameter file.
    templates = [
        toast.templates.Offset(name="baselines", enabled=False),
        # toast.templates.Fourier2D(name="fourier", enabled=False),
    ]

    # Parse options
    config, args, jobargs = parse_config(operators, templates, comm)

    # Instantiate our operators and get the size of the process groups
    job, group_size, full_pointing = job_create(config, comm)

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Load one or more observations
    data = load_data(job, args, toast_comm)

    # Reduce the data
    reduce_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    log.info_rank("Workflow completed in", comm=comm, timer=timer0)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
