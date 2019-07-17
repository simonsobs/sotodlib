#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

#import so3g

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
from toast.dist import distribute_uniform, Data
from toast.utils import Logger, Environment

from toast.weather import Weather
import toast

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.map import OpMadam, OpLocalPixels, DistPixels

from toast.tod import (
    AnalyticNoise,
    OpSimNoise,
    OpSimScanSynchronousSignal,
    OpPointingHpix,
    OpSimPySM,
    OpMemoryCounter,
    TODGround,
    OpSimScan,
    OpCacheCopy,
    OpGainScrambler,
    OpPolyFilter,
    OpGroundFilter,
    OpSimAtmosphere,
    atm_available_utils,
)

if atm_available_utils:
    from toast.tod.atm import (
        atm_atmospheric_loading,
        atm_absorption_coefficient,
        atm_absorption_coefficient_vec,
    )

import healpy as hp
import numpy as np
import toast.qarray as qa

import sotodlib.hardware

try:
    import pysm
    import so_pysm_models
except:
    pysm = None
    so_pysm_models = None

from toast.tod import tidas_available
if tidas_available:
    from toast.tod.tidas import OpTidasExport, TODTidas

try:
    import psutil
except:
    psutil = None


def memreport(comm=None, msg=""):
    # Gather and report the amount of allocated, free and swapped system memory
    if psutil is None:
        return
    log = Logger.get()
    vmem = psutil.virtual_memory()._asdict()
    gc.collect()
    vmem2 = psutil.virtual_memory()._asdict()
    memstr = "Memory usage {}\n".format(msg)
    for key, value in vmem.items():
        value2 = vmem2[key]
        if comm is None:
            vlist = [value]
            vlist2 = [value2]
        else:
            vlist = comm.gather(value)
            vlist2 = comm.gather(value2)
        if comm is None or comm.rank == 0:
            vlist = np.array(vlist, dtype=np.float64)
            vlist2 = np.array(vlist2, dtype=np.float64)
            if key != "percent":
                # From bytes to better units
                if np.amax(vlist) < 2 ** 20:
                    vlist /= 2 ** 10
                    vlist2 /= 2 ** 10
                    unit = "kB"
                elif np.amax(vlist) < 2 ** 30:
                    vlist /= 2 ** 20
                    vlist2 /= 2 ** 20
                    unit = "MB"
                else:
                    vlist /= 2 ** 30
                    vlist2 /= 2 ** 30
                    unit = "GB"
            else:
                unit = "% "
            if comm is None or comm.size == 1:
                memstr += "{:>12} : {:8.3f} {}\n".format(key, vlist[0], unit)
                if np.abs(vlist2[0] - vlist[0]) / vlist[0] > 1e-3:
                    memstr += "{:>12} : {:8.3f} {} (after GC)\n".format(
                        key, vlist2[0], unit
                    )
            else:
                med1 = np.median(vlist)
                memstr += (
                    "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                    "< {:8.3f} {}\n".format(
                        key,
                        np.amin(vlist),
                        unit,
                        med1,
                        np.std(vlist),
                        unit,
                        np.amax(vlist),
                        unit,
                    )
                )
                med2 = np.median(vlist2)
                if np.abs(med2 - med1) / med1 > 1e-3:
                    memstr += (
                        "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                        "< {:8.3f} {} (after GC)\n".format(
                            key,
                            np.amin(vlist2),
                            unit,
                            med2,
                            np.std(vlist2),
                            unit,
                            np.amax(vlist2),
                            unit,
                        )
                    )
    if comm is None or comm.rank == 0:
        log.info(memstr)
    if comm is not None:
        comm.Barrier()
    return


# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

XAXIS, YAXIS, ZAXIS = np.eye(3)


class CES(object):
    def __init__(self, start_time, stop_time, name, mjdstart, scan, subscan,
                 azmin, azmax, el, season, start_date,
                 rising, mindist_sun, mindist_moon, el_sun):
        self.start_time = start_time
        self.stop_time = stop_time
        self.name = name
        self.mjdstart = mjdstart
        self.scan = scan
        self.subscan = subscan
        self.azmin = azmin
        self.azmax = azmax
        self.el = el
        self.season = season
        self.start_date = start_date
        self.rising = rising
        self.mindist_sun = mindist_sun
        self.mindist_moon = mindist_moon
        self.el_sun = el_sun


class Site(object):
    def __init__(self, name, lat, lon, alt):
        self.name = name
        # Strings get interpreted correctly pyEphem.
        # Floats must be in radians
        self.lat = str(lat)
        self.lon = str(lon)
        self.alt = alt
        self.id = 0

class Telescope(object):
    def __init__(self, name):
        self.name = name
        self.id = {
            'LAT' : 0, 'SAT0' : 1, 'SAT1' : 2, 'SAT2' : 3, 'SAT3' : 4
        }[name]


def parse_arguments(comm):

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--groupsize",
        required=False,
        type=np.int,
        help="Size of a process group assigned to a CES",
    )

    parser.add_argument(
        "--timezone",
        required=False,
        type=np.int,
        default=0,
        help="Offset to apply to MJD to separate days [hours]",
    )
    parser.add_argument(
        "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
    )
    parser.add_argument(
        "--hardware", required=False, default=None,
        help="Input hardware file"
    )
    parser.add_argument(
        "--thinfp", required=False, type=np.int,
        help="Thin the focalplane by this factor"
    )
    parser.add_argument(
        "--schedule",
        required=True,
        help="Comma-separated list CES schedule files "
        "(from toast_ground_schedule.py)",
    )
    parser.add_argument(
        "--split_schedule",
        required=False,
        help='Only use a subset of the schedule.  The argument is a string '
        'of the form "[isplit],[nsplit]" and only observations that satisfy '
        'scan % nsplit == isplit are included',
    )
    parser.add_argument(
        "--weather",
        required=False,
        help="Comma-separated list of TOAST weather files for "
        "every schedule.  Repeat the same file if the "
        "schedules share observing site.",
    )
    parser.add_argument(
        "--samplerate",
        required=False,
        default=100.0,
        type=np.float,
        help="Detector sample rate (Hz)",
    )
    parser.add_argument(
        "--scanrate",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate [deg / s]",
    )
    parser.add_argument(
        "--scan_accel",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate change [deg / s^2]",
    )
    parser.add_argument(
        "--sun_angle_min",
        required=False,
        default=30.0,
        type=np.float,
        help="Minimum azimuthal distance between the Sun and the bore sight [deg]",
    )

    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        required=False,
        action="store_true",
        help="Conserve memory",
    )
    parser.add_argument(
        "--no_conserve_memory",
        dest="conserve_memory",
        required=False,
        action="store_false",
        help="Do not conserve memory",
    )
    parser.set_defaults(conserve_memory=True)

    parser.add_argument(
        "--polyorder",
        required=False,
        type=np.int,
        help="Polynomial order for the polyfilter",
    )

    parser.add_argument(
        "--groundorder", required=False, type=np.int, help="Ground template order"
    )

    parser.add_argument(
        "--gain_sigma", required=False, type=np.float, help="Gain error distribution"
    )

    parser.add_argument(
        "--hwprpm",
        required=False,
        default=0.0,
        type=np.float,
        help="The rate (in RPM) of the HWP rotation",
    )
    parser.add_argument(
        "--hwpstep",
        required=False,
        default=None,
        help="For stepped HWP, the angle in degrees of each step",
    )
    parser.add_argument(
        "--hwpsteptime",
        required=False,
        default=0.0,
        type=np.float,
        help="For stepped HWP, the time in seconds between steps",
    )

    parser.add_argument("--input_map", required=False, help="Input map for signal")
    parser.add_argument("--groundmap", required=False, help="Fixed ground template map")
    parser.add_argument(
        "--simulate_ground",
        required=False,
        default=False,
        action="store_true",
        help="Enable simulating ground pickup.",
    )
    parser.add_argument(
        "--input_pysm_model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. s3,d6,f1,a2" '
        'this pipeline also supports the SO specific models from '
        'so_pysm_models, see https://github.com/simonsobs/so_pysm_models '
        'currently the most complete PySM model for simulations is:'
        '"SO_d0,SO_s0,SO_a0,SO_f0,SO_x1_cib,SO_x1_tsz,SO_x1_ksz,SO_x1_cmb_lensed_solardipole"',

    )
    parser.add_argument(
        "--apply_beam",
        required=False,
        action="store_true",
        help="Apply beam convolution to input map with "
        "gaussian beam parameters defined in focalplane",
    )

    parser.add_argument(
        "--skip_atmosphere",
        required=False,
        default=False,
        action="store_true",
        help="Disable simulating the atmosphere.",
    )
    parser.add_argument(
        "--skip_noise",
        required=False,
        default=False,
        action="store_true",
        help="Disable simulating detector noise.",
    )
    parser.add_argument(
        "--common_mode_noise",
        required=False,
        help="String defining analytical parameters of a per-tube "
        "common mode that is co-added with every detector: "
        "'fmin[Hz],fknee[Hz],alpha,NET[K]'",
    )
    parser.add_argument(
        "--skip_bin",
        required=False,
        default=False,
        action="store_true",
        help="Disable binning the map.",
    )
    parser.add_argument(
        "--skip_hits",
        required=False,
        default=False,
        action="store_true",
        help="Do not save the 3x3 matrices and hitmaps",
    )
    parser.add_argument(
        "--skip_destripe",
        required=False,
        default=False,
        action="store_true",
        help="Do not destripe the data",
    )
    parser.add_argument(
        "--skip_daymaps",
        required=False,
        default=False,
        action="store_true",
        help="Do not bin daily maps",
    )

    parser.add_argument(
        "--atm_lmin_center",
        required=False,
        default=0.01,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale center",
    )
    parser.add_argument(
        "--atm_lmin_sigma",
        required=False,
        default=0.001,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale sigma",
    )
    parser.add_argument(
        "--atm_lmax_center",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale center",
    )
    parser.add_argument(
        "--atm_lmax_sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale sigma",
    )
    parser.add_argument(
        "--atm_gain",
        required=False,
        default=3e-5,
        type=np.float,
        help="Atmospheric gain factor.",
    )
    parser.add_argument(
        "--atm_zatm",
        required=False,
        default=40000.0,
        type=np.float,
        help="atmosphere extent for temperature profile",
    )
    parser.add_argument(
        "--atm_zmax",
        required=False,
        default=200.0,
        type=np.float,
        help="atmosphere extent for water vapor integration",
    )
    parser.add_argument(
        "--atm_xstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in X direction",
    )
    parser.add_argument(
        "--atm_ystep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Y direction",
    )
    parser.add_argument(
        "--atm_zstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Z direction",
    )
    parser.add_argument(
        "--atm_nelem_sim_max",
        required=False,
        default=1000,
        type=np.int,
        help="controls the size of the simulation slices",
    )
    parser.add_argument(
        "--atm_wind_dist",
        required=False,
        default=500.0,
        type=np.float,
        help="Maximum wind drift to simulate without discontinuity",
    )
    parser.add_argument(
        "--atm_z0_center",
        required=False,
        default=2000.0,
        type=np.float,
        help="central value of the water vapor distribution",
    )
    parser.add_argument(
        "--atm_z0_sigma",
        required=False,
        default=0.0,
        type=np.float,
        help="sigma of the water vapor distribution",
    )
    parser.add_argument(
        "--atm_T0_center",
        required=False,
        default=280.0,
        type=np.float,
        help="central value of the temperature distribution",
    )
    parser.add_argument(
        "--atm_T0_sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="sigma of the temperature distribution",
    )
    parser.add_argument(
        "--atm_cache",
        required=False,
        default="atm_cache",
        help="Atmosphere cache directory",
    )

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )
    #parser.add_argument(
    #    "--zip",
    #    required=False,
    #    default=False,
    #    action="store_true",
    #    help="Compress the output fits files",
    #)
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write diagnostics",
    )
    parser.add_argument(
        "--flush",
        required=False,
        default=False,
        action="store_true",
        help="Flush every print statement.",
    )
    parser.add_argument(
        "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
    )
    parser.add_argument(
        "--madam_prefix", required=False, default="toast", help="Output map prefix"
    )
    parser.add_argument(
        "--madam_iter_max",
        required=False,
        default=1000,
        type=np.int,
        help="Maximum number of CG iterations in Madam",
    )
    parser.add_argument(
        "--madam_nside_cross",
        required=False,
        type=np.int,
        help="Madam destriping resolution (default is nside / 2)",
    )
    parser.add_argument(
        "--madam_baseline_length",
        required=False,
        default=10000.0,
        type=np.float,
        help="Destriping baseline length (seconds)",
    )
    parser.add_argument(
        "--madam_baseline_order",
        required=False,
        default=0,
        type=np.int,
        help="Destriping baseline polynomial order",
    )
    parser.add_argument(
        "--madam_precond_width",
        required=False,
        default=1,
        type=np.int,
        help="Madam preconditioner width",
    )
    parser.add_argument(
        "--madam_noisefilter",
        required=False,
        default=False,
        action="store_true",
        help="Destripe with the noise filter enabled",
    )
    parser.add_argument(
        "--madampar", required=False, default=None, help="Madam parameter file"
    )
    parser.add_argument(
        "--no_madam_allreduce",
        required=False,
        default=False,
        action="store_true",
        help="Do not use allreduce communication in Madam",
    )
    parser.add_argument(
        "--common_flag_mask",
        required=False,
        default=1,
        type=np.uint8,
        help="Common flag mask",
    )
    parser.add_argument(
        "--MC_start",
        required=False,
        default=0,
        type=np.int,
        help="First Monte Carlo noise realization",
    )
    parser.add_argument(
        "--MC_count",
        required=False,
        default=1,
        type=np.int,
        help="Number of Monte Carlo noise realizations",
    )
    parser.add_argument(
        "--focalplane_radius",
        required=False,
        type=np.float,
        help="Override focal plane radius [deg]",
    )
    parser.add_argument(
        "--bands",
        required=True,
        help="Comma-separated list of bands: LF1 (27GHz), LF2 (39GHz), "
        "MFF1 (93GHz), MFF2 (145GHz), MFS1 (93GHz), MFS2 (145GHz), "
        "UHF1 (225GHz), UHF2 (280GHz). "
        "Length of list must equal --tubes",
    )
    parser.add_argument(
        "--tubes",
        required=True,
        help="Comma-separated list of  optics tubes: LT0 (UHF), LT1 (UHF), "
        " LT2 (MFF), LT3 (MFF), LT4 (MFS), LT5 (MFS), LT6 (LF). "
        "Length of list must equal --bands",
    )
    parser.add_argument(
        "--tidas", required=False, default=None, help="Output TIDAS export path"
    )
    parser.add_argument(
        "--export", required=False, default=None, help="Output TOD export path"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    if len(args.bands.split(",")) != 1:
        # Multi frequency run.  We don't support multiple copies of
        # scanned signal.
        if args.input_map:
            raise RuntimeError(
                "Multiple frequencies are not supported when scanning from a map"
            )

    if not args.skip_atmosphere and args.weather is None:
        raise RuntimeError("Cannot simulate atmosphere without a TOAST weather file")

    if args.tidas is not None:
        if not tidas_available:
            raise RuntimeError("TIDAS not found- cannot export")

    if comm.world_rank == 0:
        print("\nAll parameters:")
        print(args, flush=args.flush)
        print("")

    if args.groupsize:
        comm = toast.Comm(groupsize=args.groupsize)

    if comm.world_rank == 0:
        if not os.path.isdir(args.outdir):
            try:
                os.makedirs(args.outdir)
            except FileExistsError:
                pass

    return args, comm


def name2id(name, maxval=2 ** 16):
    """ Map a name into an index.

    """
    value = 0
    for c in name:
        value += ord(c)
    return value % maxval


@function_timer
def load_weather(args, comm, schedules):
    """ Load TOAST weather file(s) and attach them to the schedules.

    """
    if args.weather is None:
        return
    timer = Timer()
    timer.start()

    if comm.world_rank == 0:
        timer1 = Timer()
        weathers = []
        weatherdict = {}
        for fname in args.weather.split(","):
            if fname not in weatherdict:
                if not os.path.isfile(fname):
                    raise RuntimeError("No such weather file: {}".format(fname))
                timer1.start()
                weatherdict[fname] = Weather(fname)
                timer1.stop()
                timer1.report("Load {}".format(fname))
            weathers.append(weatherdict[fname])
    else:
        weathers = None

    weathers = comm.comm_world.bcast(weathers)
    if len(weathers) == 1 and len(schedules) > 1:
        weathers *= len(schedules)
    if len(weathers) != len(schedules):
        raise RuntimeError("Number of weathers must equal number of schedules or be 1.")

    for schedule, weather in zip(schedules, weathers):
        schedule.append(weather)

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Loading weather")
    return


def min_sso_dist(el, azmin, azmax, sso_el1, sso_az1, sso_el2, sso_az2):
    """ Return a rough minimum angular distance between the bore sight
    and a solar system object"""
    sso_vec1 = hp.dir2vec(sso_az1, sso_el1, lonlat=True)
    sso_vec2 = hp.dir2vec(sso_az2, sso_el2, lonlat=True)
    az1 = azmin
    az2 = azmax
    if az2 < az1:
        az2 += 360
    n = 100
    az = np.linspace(az1, az2, n)
    el = np.ones(n) * el
    vec = hp.dir2vec(az, el, lonlat=True)
    dist1 = np.degrees(np.arccos(np.dot(sso_vec1, vec)))
    dist2 = np.degrees(np.arccos(np.dot(sso_vec2, vec)))
    return min(np.amin(dist1), np.amin(dist2))


@function_timer
def load_schedule(args, comm):
    """ Load the observing schedule(s).

    """
    schedules = []
    timer0 = Timer()
    timer0.start()

    if comm.world_rank == 0:
        timer1 = Timer()
        isplit, nsplit = None, None
        if args.split_schedule is not None:
            isplit, nsplit = args.split_schedule.split(",")
            isplit = np.int(isplit)
            nsplit = np.int(nsplit)
            scan_counters = {}
        for fn in args.schedule.split(","):
            if not os.path.isfile(fn):
                raise RuntimeError("No such schedule file: {}".format(fn))
            timer1.start()
            with open(fn, "r") as f:
                while True:
                    line = f.readline()
                    if line.startswith("#"):
                        continue
                    (site_name, telescope, site_lat, site_lon, site_alt) = line.split()
                    site = Site(site_name, site_lat, site_lon, float(site_alt))
                    break
                all_ces = []
                for line in f:
                    if line.startswith("#"):
                        continue
                    (
                        start_date,
                        start_time,
                        stop_date,
                        stop_time,
                        mjdstart,
                        mjdstop,
                        name,
                        azmin,
                        azmax,
                        el,
                        rs,
                        sun_el1,
                        sun_az1,
                        sun_el2,
                        sun_az2,
                        moon_el1,
                        moon_az1,
                        moon_el2,
                        moon_az2,
                        moon_phase,
                        scan,
                        subscan,
                    ) = line.split()
                    if nsplit:
                        # Only accept 1 / `nsplit` of the rising and setting
                        # scans in patch `name`.  Selection is performed
                        # during the first subscan.
                        if int(subscan) == 0:
                            if name not in scan_counters:
                                scan_counters[name] = {}
                            counter = scan_counters[name]
                            # Separate counters for rising and setting scans
                            if rs not in counter:
                                counter[rs] = 0
                            else:
                                counter[rs] += 1
                            iscan = counter[rs]
                        if iscan % nsplit != isplit:
                            continue
                    start_time = start_date + " " + start_time
                    stop_time = stop_date + " " + stop_time
                    # Define season as a calendar year.  This can be
                    # changed later and could even be in the schedule file.
                    season = int(start_date.split("-")[0])
                    # Gather other useful metadata
                    mindist_sun = min_sso_dist(
                        *np.array(
                            [el, azmin, azmax, sun_el1, sun_az1,
                             sun_el2, sun_az2]).astype(np.float))
                    mindist_moon = min_sso_dist(
                        *np.array(
                            [el, azmin, azmax, moon_el1, moon_az1,
                             moon_el2, moon_az2]).astype(np.float))
                    el_sun = max(float(sun_el1), float(sun_el2))
                    try:
                        start_time = dateutil.parser.parse(start_time + " +0000")
                        stop_time = dateutil.parser.parse(stop_time + " +0000")
                    except Exception:
                        start_time = dateutil.parser.parse(start_time)
                        stop_time = dateutil.parser.parse(stop_time)
                    start_timestamp = start_time.timestamp()
                    stop_timestamp = stop_time.timestamp()
                    all_ces.append(
                        CES(
                            start_time=start_timestamp,
                            stop_time=stop_timestamp,
                            name=name,
                            mjdstart=float(mjdstart),
                            scan=int(scan),
                            subscan=int(subscan),
                            azmin=float(azmin),
                            azmax=float(azmax),
                            el=float(el),
                            season=season,
                            start_date=start_date,
                            rising=(rs.upper() == "R"),
                            mindist_sun=mindist_sun,
                            mindist_moon=mindist_moon,
                            el_sun=el_sun,
                        )
                    )
            schedules.append([site, all_ces])
            timer1.stop()
            timer1.report_clear(
                "Load {} (sub)scans in {}".format(len(all_ces), fn))

    schedules = comm.comm_world.bcast(schedules)

    timer0.stop()
    if comm.world_rank == 0:
        timer0.report("Loading schedule")
    return schedules


@function_timer
def get_focalplane_radius(args, focalplane, rmin=1.0):
    """ Find the furthest angular distance from the boresight

    """
    if args.focalplane_radius:
        return args.focalplane_radius

    cosangs = []
    for det in focalplane:
        quat = focalplane[det]["quat"]
        vec = qa.rotate(quat, ZAXIS)
        cosangs.append(np.dot(ZAXIS, vec))
    mincos = np.amin(cosangs)
    maxdist = max(np.degrees(np.arccos(mincos)), rmin)
    return maxdist * 1.001


def get_band_params(banddata):
    net = banddata["NET"] * 1e-6  # uK -> K
    fknee = banddata["fknee"] * 1e-3  # mHz -> Hz
    fmin = banddata["fmin"] * 1e-3  # mHz -> Hz
    # alpha = banddata[band]["alpha"]
    alpha = 1  # hardwire a sensible number. 3.5 is not realistic.
    A = banddata["A"]
    C = banddata["C"]
    lower = banddata["low"]  # GHz
    center = banddata["center"]  # GHz
    upper = banddata["high"]  # GHz
    return net, fknee, fmin, alpha, A, C, lower, center, upper


def get_det_params(detdata, band_net, band_fknee, band_fmin, band_alpha,
                   band_A, band_C, band_lower, band_center, band_upper):
    def get_par(key, default, scale=1):
        if key in detdata:
            return detdata[key] * scale
        else:
            return default
    net = get_par("NET", band_net, 1e-6)  # uK -> K
    fknee = get_par("fknee", band_fknee, 1e-3)  # mHz -> Hz
    fmin = get_par("fmin", band_fmin, 1e-3)  # mHz -> Hz
    alpha = get_par("alpha", band_alpha)
    alpha = 1  # hardwire a sensible number. 3.5 is not realistic.
    A = get_par("A", band_A)
    C = get_par("C", band_C)
    lower = get_par("low", band_lower)  # GHz
    center = get_par("center", band_center)  # GHz
    upper = get_par("high", band_upper)  # GHz
    center = 0.5 * (lower + upper)
    width = upper - lower
    return net, fknee, fmin, alpha, A, C, center, width


@function_timer
def load_focalplanes(args, comm, schedules):
    """ Attach a focalplane to each of the schedules.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    # Load focalplane information

    bands = args.bands.split(",")
    tubes = args.tubes.split(",")
    telescopes = []
    hwexample = sotodlib.hardware.get_example()
    for tube in tubes:
        for telescope, teledata in hwexample.data['telescopes'].items():
            if tube in teledata['tubes']:
                telescopes.append(telescope)
                break

    focalplanes = []
    if comm.world_rank == 0:
        timer1 = Timer()
        for telescope, band, tube in zip(telescopes, bands, tubes):
            timer1.start()
            if args.hardware:
                log.info("Loading hardware configuration from {}..."
                         "".format(args.hardware))
                hw = sotodlib.hardware.Hardware(args.hardware)
            else:
                log.info("Simulating default hardware configuration")
                hw = sotodlib.hardware.get_example()
                hw.data["detectors"] = sotodlib.hardware.sim_telescope_detectors(
                    hw, telescope
                )
            # Construct a running index for all detectors across all
            # telescopes for independent noise realizations
            detindex = {}
            for idet, det in enumerate(sorted(hw.data["detectors"])):
                detindex[det] = idet
            match = {"band": band}
            hw = hw.select(telescopes=None, tubes=[tube], match=match)
            if len(hw.data["detectors"]) == 0:
                raise RuntimeError(
                    "No detectors match query: telescopes={}, "
                    "tubes={}, match={}".format(telescopes, tubes, match))
            # Transfer the detector information into a TOAST dictionary
            focalplane = {}
            banddata = hw.data["bands"][band]
            (band_net, band_fknee, band_fmin, band_alpha, band_A, band_C,
             band_lower, band_center, band_upper) = get_band_params(banddata)
            for idet, (detname, detdata) in enumerate(hw.data["detectors"].items()):
                (net, fknee, fmin, alpha, A, C, center, width) = get_det_params(
                    detdata, band_net, band_fknee, band_fmin, band_alpha,
                    band_A, band_C, band_lower, band_center, band_upper)
                # DEBUG begin
                #if idet % 100 != 0:
                #    continue
                # DEBUG end
                wafer = detdata["wafer"]
                for tube, tubedata in hw.data["tubes"].items():
                    if wafer in tubedata["wafers"]:
                        break
                index = detindex[detname]
                if args.thinfp and index % args.thinfp != 0:
                    # Only accept a fraction of the detectors for
                    # testing and development
                    continue
                focalplane[detname] = {
                    "NET": net,
                    "fknee": fknee,
                    "fmin": fmin,
                    "alpha": alpha,
                    "A": A,
                    "C": C,
                    "quat": detdata["quat"],
                    "FWHM": detdata["fwhm"],
                    "freq": center,
                    "bandcenter_ghz": center,
                    "bandwidth_ghz": width,
                    "index": index,
                    "telescope": telescope,
                    "tube": tube,
                    "wafer": wafer,
                    "band": band,
                }
            focalplanes.append(focalplane)
            timer1.stop()
            timer1.report(
                "Load tele = {} tube = {} band = {} focalplane ({} detectors)"
                "".format(telescope, tube, band, len(focalplane)))
    focalplanes = comm.comm_world.bcast(focalplanes)
    telescopes = comm.comm_world.bcast(telescopes)

    if len(schedules) == 1:
        schedules *= len(focalplanes)

    if len(focalplanes) != len(schedules):
        raise RuntimeError(
            "Number of focalplanes must equal number of schedules"
        )

    detweights = {}
    for schedule, focalplane, telescope in zip(schedules, focalplanes, telescopes):
        schedule.append(focalplane)
        schedule.append(Telescope(telescope))
        for detname, detdata in focalplane.items():
            # Transfer the detector properties from the band dictionary to the detectors
            net = detdata["NET"]
            # And build a dictionary of detector weights
            detweight = 1.0 / (args.samplerate * net * net)
            if detname in detweights and detweights[detname] != detweight:
                raise RuntimeError("Detector weight for {} changes".format(detname))
            detweights[detname] = detweight

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Loading focalplane(s)")
    return detweights


@function_timer
def get_analytic_noise(args, focalplane):
    """ Create a TOAST noise object.

    Create a noise object from the 1/f noise parameters contained in the
    focalplane database.

    """
    detectors = sorted(focalplane.keys())
    fmins = {}
    fknees = {}
    alphas = {}
    NETs = {}
    rates = {}
    indices = {}
    for d in detectors:
        rates[d] = args.samplerate
        fmins[d] = focalplane[d]["fmin"]
        fknees[d] = focalplane[d]["fknee"]
        alphas[d] = focalplane[d]["alpha"]
        NETs[d] = focalplane[d]["NET"]
        indices[d] = focalplane[d]["index"]

    if args.common_mode_noise:
        # Add an extra "virtual" detector for common mode noise for
        # every optics tube
        fmin, fknee, alpha, net = np.array(
            args.common_mode_noise.split(",")
        ).astype(np.float64)
        hw = sotodlib.hardware.get_example()
        for itube, tube in enumerate(sorted(hw.data["tubes"].keys())):
            d = "common_mode_{}".format(tube)
            detectors.append(d)
            rates[d] = args.samplerate
            fmins[d] = fmin
            fknees[d] = fknee
            alphas[d] = alpha
            NETs[d] = net
            indices[d] = 100000 + itube

    noise = AnalyticNoise(
        rate=rates,
        fmin=fmins,
        detectors=detectors,
        fknee=fknees,
        alpha=alphas,
        NET=NETs,
        indices=indices,
    )

    if args.common_mode_noise:
        # Update the mixing matrix in the noise operator
        mixmatrix = {}
        keys = set()
        for det in focalplane.keys():
            tube = focalplane[det]["tube"]
            common = "common_mode_{}".format(tube)
            mixmatrix[det] = {det : 1, common : 1}
            keys.add(det)
            keys.add(common)
        # There should probably be an accessor method to update the
        # mixmatrix in the TOAST Noise object.
        if noise._mixmatrix is not None:
            raise RuntimeError("Did not expect non-empty mixing matrix")
        noise._mixmatrix = mixmatrix
        noise._keys = list(sorted(keys))

    return noise


@function_timer
def get_elevation_noise(args, comm, data, key="noise"):
    """ Insert elevation-dependent noise

    """
    timer = Timer()
    timer.start()
    fsample = args.samplerate
    for obs in data.obs:
        tod = obs["tod"]
        fp = obs["focalplane"]
        noise = obs[key]
        for det in tod.local_dets:
            if det not in noise.keys:
                raise RuntimeError(
                    'Detector "{}" does not have a PSD in the noise object'.format(det)
                )
            A = fp[det]["A"]
            C = fp[det]["C"]
            psd = noise.psd(det)
            try:
                # Some TOD classes provide a shortcut to Az/El
                _, el = tod.read_azel(detector=det)
            except Exception as e:
                azelquat = tod.read_pntg(detector=det, azel=True)
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, _ = qa.to_position(azelquat)
                el = np.pi / 2 - theta
            el = np.median(el)
            # Scale the analytical noise PSD. Pivot is at el = 50 deg.
            psd[:] *= (A / np.sin(el) + C) ** 2
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Elevation noise")
    return


@function_timer
def get_breaks(comm, all_ces, nces, args):
    """ List operational day limits in the list of CES:s.

    """
    breaks = []
    if args.skip_daymaps:
        return breaks
    do_break = False
    for i in range(nces - 1):
        # If current and next CES are on different days, insert a break
        tz = args.timezone / 24
        start1 = all_ces[i][3]  # MJD start
        start2 = all_ces[i + 1][3]  # MJD start
        scan1 = all_ces[i][4]
        scan2 = all_ces[i + 1][4]
        if scan1 != scan2 and do_break:
            breaks.append(nces + i + 1)
            do_break = False
            continue
        day1 = int(start1 + tz)
        day2 = int(start2 + tz)
        if day1 != day2:
            if scan1 == scan2:
                # We want an entire CES, even if it crosses the day bound.
                # Wait until the scan number changes.
                do_break = True
            else:
                breaks.append(nces + i + 1)

    nbreak = len(breaks)
    if nbreak < comm.ngroups - 1:
        if comm.world_rank == 0:
            log.info(
                "WARNING: there are more process groups than observing days. "
                "Will try distributing by observation.")
        breaks = []
        for i in range(nces - 1):
            scan1 = all_ces[i][4]
            scan2 = all_ces[i + 1][4]
            if scan1 != scan2:
                breaks.append(nces + i + 1)
        nbreak = len(breaks)

    if nbreak != comm.ngroups - 1:
        raise RuntimeError(
            "Number of observing days ({}) does not match number of process "
            "groups ({}).".format(nbreak + 1, comm.ngroups)
        )
    return breaks


@function_timer
def create_observation(args, comm, all_ces_tot, ices, noise):
    """ Create a TOAST observation.

    Create an observation for the CES scan defined by all_ces_tot[ices].

    """
    ces, site, telescope, fp, fpradius, detquats, weather = all_ces_tot[ices]
    totsamples = int((ces.stop_time - ces.start_time) * args.samplerate)

    # create the TOD for this observation

    try:
        tod = TODGround(
            comm.comm_group,
            detquats,
            totsamples,
            detranks=comm.comm_group.size,
            firsttime=ces.start_time,
            rate=args.samplerate,
            site_lon=site.lon,
            site_lat=site.lat,
            site_alt=site.alt,
            azmin=ces.azmin,
            azmax=ces.azmax,
            el=ces.el,
            scanrate=args.scanrate,
            scan_accel=args.scan_accel,
            CES_start=None,
            CES_stop=None,
            sun_angle_min=args.sun_angle_min,
            coord=args.coord,
            sampsizes=None,
        )
    except RuntimeError as e:
        raise RuntimeError(
            'Failed to create TOD for {}-{}-{}: "{}"'
            "".format(ces.name, ces.scan, ces.subscan, e)
        )

    # Create the observation

    obs = {}
    obs["name"] = "CES-{}-{}-{}-{}-{}".format(
        site.name, telescope.name, ces.name, ces.scan, ces.subscan
    )
    obs["tod"] = tod
    obs["baselines"] = None
    obs["noise"] = noise
    obs["id"] = int(ces.mjdstart * 10000)
    obs["intervals"] = tod.subscans
    obs["site"] = site.name
    obs["site_id"] = site.id
    obs["telescope"] = telescope.name
    obs["telescope_id"] = telescope.id
    obs["fpradius"] = fpradius
    obs["weather"] = weather
    obs["start_time"] = ces.start_time
    obs["altitude"] = site.alt
    obs["season"] = ces.season
    obs["date"] = ces.start_date
    obs["MJD"] = ces.mjdstart
    obs["focalplane"] = fp
    obs["rising"] = ces.rising
    obs["mindist_sun"] = ces.mindist_sun
    obs["mindist_moon"] = ces.mindist_moon
    obs["el_sun"] = ces.el_sun
    return obs


@function_timer
def create_observations(args, comm, schedules):
    """ Create and distribute TOAST observations for every CES in schedules.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    data = toast.Data(comm)

    # Loop over the schedules, distributing each schedule evenly across
    # the process groups.  For now, we'll assume that each schedule has
    # the same number of operational days and the number of process groups
    # matches the number of operational days.  Relaxing these constraints
    # will cause the season break to occur on different process groups
    # for different schedules and prevent splitting the communicator.

    for schedule in schedules:

        if args.weather is None:
            site, all_ces, focalplane, telescope = schedule
            weather = None
        else:
            site, all_ces, weather, focalplane, telescope = schedule

        fpradius = get_focalplane_radius(args, focalplane)

        # Focalplane information for this schedule
        detectors = sorted(focalplane.keys())
        detquats = {}
        for d in detectors:
            detquats[d] = focalplane[d]["quat"]

        all_ces_tot = []
        nces = len(all_ces)
        for ces in all_ces:
            all_ces_tot.append((ces, site, telescope, focalplane, fpradius, detquats, weather))

        breaks = get_breaks(comm, all_ces, nces, args)

        groupdist = toast.distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            # Noise model for this CES
            noise = get_analytic_noise(args, focalplane)
            obs = create_observation(args, comm, all_ces_tot, ices, noise)
            data.obs.append(obs)

    if args.skip_atmosphere and args.skip_noise:
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_azel_quats()

    if comm.comm_group.rank == 0:
        log.info("Group # {:4} has {} observations.".format(comm.group, len(data.obs)))

    if len(data.obs) == 0:
        raise RuntimeError(
            "Too many tasks. Every MPI task must "
            "be assigned to at least one observation."
        )

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Simulated scans")

    # Split the data object for each telescope for separate mapmaking.
    # We could also split by site.

    if len(schedules) > 1:
        telescope_data = data.split("telescope")
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ("all", data))
    return data, telescope_data


@function_timer
def expand_pointing(args, comm, data):
    """ Expand boresight pointing to every detector.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    hwprpm = args.hwprpm
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = args.hwpsteptime

    if comm.world_rank == 0:
        log.info("Expanding pointing")

    pointing = OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        hwprpm=hwprpm,
        hwpstep=hwpstep,
        hwpsteptime=hwpsteptime,
        single_precision=True,
    )

    pointing.exec(data)

    # Only purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.export is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Pointing generation")

    return


@function_timer
def get_submaps(args, comm, data):
    """ Get a list of locally hit pixels and submaps on every process.

    """
    log = Logger.get()

    if not args.skip_bin or args.input_map or args.input_pysm_model:
        if comm.world_rank == 0:
            log.info("Scanning local pixels")
        timer = Timer()
        timer.start()

        # Prepare for using distpixels objects
        nside = args.nside
        subnside = 16
        if subnside > nside:
            subnside = nside
        subnpix = 12 * subnside * subnside

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(data)
        if localpix is None:
            raise RuntimeError(
                "Process {} has no hit pixels. Perhaps there are fewer "
                "detectors than processes in the group?".format(comm.world_rank)
            )

        # find the locally hit submaps.
        localsm = np.unique(np.floor_divide(localpix, subnpix))

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        timer.stop()
        if comm.world_rank == 0:
            timer.report("Identify local submaps")
    else:
        localpix, localsm, subnpix = None, None, None
    return localpix, localsm, subnpix


@function_timer
def add_sky_signal(data, totalname, signalname):
    """ Add previously simulated sky signal to the atmospheric noise.

    """
    if signalname is not None:
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                cachename_in = "{}_{}".format(signalname, det)
                cachename_out = "{}_{}".format(totalname, det)
                ref_in = tod.cache.reference(cachename_in)
                if tod.cache.exists(cachename_out):
                    ref_out = tod.cache.reference(cachename_out)
                    ref_out += ref_in
                else:
                    ref_out = tod.cache.put(cachename_out, ref_in)
                del ref_in, ref_out
    return


@function_timer
def simulate_sky_signal(args, comm, data, schedules, subnpix, localsm):
    """ Use PySM to simulate smoothed sky signal.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()
    # Convolve a signal TOD from PySM
    if comm.world_rank == 0:
        log.info("Simulating sky signal with PySM")

    map_dist = (
        None if comm is None
        else pysm.MapDistribution(nside=args.nside, mpi_comm=comm.comm_rank)
    )
    pysm_component_objects = []
    pysm_model = []
    for model_tag in args.input_pysm_model.split(","):

        if not model_tag.startswith("SO"):
            pysm_model.append(model_tag)
        else:

            if model_tag == "SO_x1_cib":
                pysm_component_objects.append(
                    so_pysm_models.WebSkyCIB(
                        websky_version="0.3",
                        interpolation_kind="linear",
                        nside=args.nside,
                        map_dist=map_dist,
                    )
                )
            elif model_tag == "SO_x1_ksz":
                pysm_component_objects.append(
                    so_pysm_models.WebSkySZ(
                        version="0.3",
                        nside=args.nside,
                        map_dist=map_dist,
                        sz_type="kinetic",
                    )
                )
            elif model_tag == "SO_x1_tsz":
                pysm_component_objects.append(
                    so_pysm_models.WebSkySZ(
                        version="0.3",
                        nside=args.nside,
                        map_dist=map_dist,
                        sz_type="thermal",
                    )
                )
            elif model_tag.startswith("SO_x1_cmb"):
                lensed = "unlensed" not in model_tag
                include_solar_dipole = "solar" in model_tag
                pysm_component_objects.append(
                    so_pysm_models.WebSkyCMBMap(
                        websky_version="0.3",
                        lensed=lensed,
                        include_solar_dipole=include_solar_dipole,
                        seed=1,
                        nside=args.nside,
                        map_dist=map_dist,
                    )
                )
            else:
                pysm_component_objects.append(
                    so_pysm_models.get_so_models(
                        model_tag, args.nside, map_dist=map_dist
                    )
                )

    if comm.world_rank == 0:
        log.info("Simulating sky signal with PySM")

    signalname = "signal"
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=signalname,
        pysm_model=pysm_model,
        pysm_component_objects=pysm_component_objects,
        focalplanes=[s[3] for s in schedules],
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.apply_beam,
        coord="G", # setting G doesn't perform any rotation
        map_dist=map_dist,
    )
    assert args.coord in "CQ", "Input SO models are always in Equatorial coordinates"
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("PySM")
    return signalname


@function_timer
def scan_sky_signal(args, comm, data, localsm, subnpix):
    """ Scan sky signal from a map.

    """
    signalname = None

    if args.input_map:
        log = Logger.get()
        timer = Timer()
        if comm.world_rank == 0:
            log.info("Scanning input map")
        timer.start()
        npix = 12 * args.nside ** 2
        # Scan the sky signal
        if comm.world_rank == 0 and not os.path.isfile(args.input_map):
            raise RuntimeError("Input map does not exist: {}".format(args.input_map))
        distmap = tm.DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=3,
            dtype=np.float32,
            submap=subnpix,
            local=localsm,
        )
        distmap.read_healpix_fits(args.input_map)
        scansim = OpSimScan(distmap=distmap, out="signal")
        scansim.exec(data)
        signalname = "signal"

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        timer.stop()
        if comm.world_rank == 0:
            timer.report("Read and sample map")

    return signalname


@function_timer
def setup_madam(args):
    """ Create a Madam parameter dictionary.

    Initialize the Madam parameters from the command line arguments.

    """
    pars = {}

    if args.madam_nside_cross:
        cross = args.madam_nside_cross
    else:
        cross = args.nside // 2
    submap = 16
    if submap > args.nside:
        submap = args.nside

    pars["temperature_only"] = False
    pars["force_pol"] = True
    pars["kfirst"] = not args.skip_destripe
    pars["write_map"] = not args.skip_destripe
    pars["write_binmap"] = not args.skip_bin
    pars["write_matrix"] = not args.skip_hits
    pars["write_wcov"] = not args.skip_hits
    pars["write_hits"] = not args.skip_hits
    pars["nside_cross"] = cross
    pars["nside_submap"] = submap
    if args.no_madam_allreduce:
        pars["allreduce"] = False
    else:
        pars["allreduce"] = True
    pars["reassign_submaps"] = True
    pars["pixlim_cross"] = 1e-3
    pars["pixmode_cross"] = 2
    pars["pixlim_map"] = 1e-2
    pars["pixmode_map"] = 2
    # Instead of fixed detector weights, we'll want to use scaled noise
    # PSD:s that include the atmospheric noise
    pars["radiometers"] = True
    pars["noise_weights_from_psd"] = True

    if args.madampar is not None:
        pat = re.compile(r"\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*")
        comment = re.compile(r"^#.*")
        with open(args.madampar, "r") as f:
            for line in f:
                if comment.match(line) is None:
                    result = pat.match(line)
                    if result is not None:
                        key, value = result.group(1), result.group(2)
                        pars[key] = value

    pars["base_first"] = args.madam_baseline_length
    pars["basis_order"] = args.madam_baseline_order
    pars["nside_map"] = args.nside
    if args.madam_noisefilter:
        if args.madam_baseline_order != 0:
            raise RuntimeError(
                "Madam cannot build a noise filter when baseline"
                "order is higher than zero."
            )
        pars["kfilter"] = True
    else:
        pars["kfilter"] = False
    pars["precond_width_min"] = 0
    pars["precond_width_max"] = args.madam_precond_width
    pars["fsample"] = args.samplerate
    pars["iter_max"] = args.madam_iter_max
    pars["file_root"] = args.madam_prefix
    return pars


@function_timer
def scale_atmosphere_by_bandpass(args, comm, data, totalname, mc):
    """ Scale atmospheric fluctuations by bandpass.

    Assume that cached signal under totalname is pure atmosphere
    and scale the absorption coefficient according to the bandpass.

    If the focalplane is included in the observation and defines
    bandpasses for the detectors, the scaling is computed for each
    detector separately.

    """
    if args.skip_atmosphere:
        return

    timer = Timer()
    log = Logger.get()

    if comm.world_rank == 0:
        log.info("Scaling atmosphere by bandpass")

    timer.start()
    for obs in data.obs:
        tod = obs["tod"]
        todcomm = tod.mpicomm
        site_id = obs["site_id"]
        weather = obs["weather"]
        if "focalplane" in obs:
            focalplane = obs["focalplane"]
        else:
            focalplane = None
        start_time = obs["start_time"]
        weather.set(site_id, mc, start_time)
        altitude = obs["altitude"]
        air_temperature = weather.air_temperature
        surface_pressure = weather.surface_pressure
        pwv = weather.pwv
        # Use the entire processing group to sample the absorption
        # coefficient as a function of frequency
        freqmin = 0
        freqmax = 1000
        nfreq = 10001
        freqstep = (freqmax - freqmin) / (nfreq - 1)
        nfreq_task = int(nfreq // todcomm.size) + 1
        my_ifreq_min = nfreq_task * todcomm.rank
        my_ifreq_max = min(nfreq, nfreq_task * (todcomm.rank + 1))
        my_nfreq = my_ifreq_max - my_ifreq_min
        if my_nfreq > 0:
            if atm_available_utils:
                my_freqs = freqmin + np.arange(my_ifreq_min, my_ifreq_max) * freqstep
                my_absorption = atm_absorption_coefficient_vec(
                    altitude,
                    air_temperature,
                    surface_pressure,
                    pwv,
                    my_freqs[0],
                    my_freqs[-1],
                    my_nfreq,
                )
            else:
                raise RuntimeError(
                    "Atmosphere utilities from libaatm are not available"
                )
        else:
            my_freqs = np.array([])
            my_absorption = np.array([])
        freqs = np.hstack(todcomm.allgather(my_freqs))
        absorption = np.hstack(todcomm.allgather(my_absorption))
        # loading = atm_atmospheric_loading(altitude, pwv, freq)
        for det in tod.local_dets:
            # Use detector bandpass from the focalplane
            center = focalplane[det]["bandcenter_ghz"]
            width = focalplane[det]["bandwidth_ghz"]
            nstep = 101
            # Interpolate the absorption coefficient to do a top hat
            # integral across the bandpass
            det_freqs = np.linspace(center - width / 2, center + width / 2, nstep)
            absorption_det = np.mean(np.interp(det_freqs, freqs, absorption))
            cachename = "{}_{}".format(totalname, det)
            ref = tod.cache.reference(cachename)
            ref *= absorption_det
            del ref

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Atmosphere scaling")
    return


@function_timer
def update_atmospheric_noise_weights(args, comm, data, freq, mc):
    """ Update atmospheric noise weights.

    Estimate the atmospheric noise level from weather parameters and
    encode it as a noise_scale in the observation.  Madam will apply
    the noise_scale to the detector weights.  This approach assumes
    that the atmospheric noise dominates over detector noise.  To be
    more precise, we would have to add the squared noise weights but
    we do not have their relative calibration.

    """
    if args.weather and not args.skip_atmosphere:
        if atm_available_utils:
            timer = Timer()
            timer.start()
            for obs in data.obs:
                site_id = obs["site_id"]
                weather = obs["weather"]
                start_time = obs["start_time"]
                weather.set(site_id, mc, start_time)
                altitude = obs["altitude"]
                absorption = atm_absorption_coefficient(
                    altitude,
                    weather.air_temperature,
                    weather.surface_pressure,
                    weather.pwv,
                    freq,
                )
                obs["noise_scale"] = absorption * weather.air_temperature
            if comm.comm_world is not None:
                comm.comm_world.barrier()
            timer.stop()
            if comm.world_rank == 0:
                timer.report("Atmosphere weighting")
        else:
            raise RuntimeError("Atmosphere utilities from libaatm are not available")
    else:
        for obs in data.obs:
            obs["noise_scale"] = 1.0

    return


@function_timer
def simulate_atmosphere(args, comm, data, mc, totalname):
    if args.skip_atmosphere:
        return
    timer = Timer()
    log = Logger.get()
    if comm.world_rank == 0:
        log.info("Simulating atmosphere")
        if args.atm_cache and not os.path.isdir(args.atm_cache):
            try:
                os.makedirs(args.atm_cache)
            except FileExistsError:
                pass
    # Simulate the atmosphere signal
    timer.start()
    atm = OpSimAtmosphere(
        out=totalname,
        realization=mc,
        lmin_center=args.atm_lmin_center,
        lmin_sigma=args.atm_lmin_sigma,
        lmax_center=args.atm_lmax_center,
        gain=args.atm_gain,
        lmax_sigma=args.atm_lmax_sigma,
        zatm=args.atm_zatm,
        zmax=args.atm_zmax,
        xstep=args.atm_xstep,
        ystep=args.atm_ystep,
        zstep=args.atm_zstep,
        nelem_sim_max=args.atm_nelem_sim_max,
        verbosity=int(args.debug),
        z0_center=args.atm_z0_center,
        z0_sigma=args.atm_z0_sigma,
        apply_flags=False,
        common_flag_mask=args.common_flag_mask,
        cachedir=args.atm_cache,
        flush=args.flush,
        wind_dist=args.atm_wind_dist,
    )
    atm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Atmosphere simulation")
    return


@function_timer
def simulate_noise(args, comm, data, mc, totalname):
    if args.skip_noise:
        return
    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0:
        log.info("Simulating noise")
    timer.start()
    nse = OpSimNoise(out=totalname, realization=mc)
    nse.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Noise simulation")
    return

@function_timer
def simulate_sss(args, comm, data, mc, totalname):
    if not args.simulate_ground:
        return
    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0:
        log.info("Simulating sss")
    timer.start()
    nse = OpSimScanSynchronousSignal(out=totalname, realization=mc, path=args.groundmap)
    nse.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("sss simulation")
    return


@function_timer
def scramble_gains(args, comm, data, mc, totalname):
    if args.gain_sigma:
        log = Logger.get()
        timer = Timer()
        if comm.world_rank == 0:
            log.info("Scrambling gains")
        timer.start()
        scrambler = OpGainScrambler(
            sigma=args.gain_sigma, name=totalname, realization=mc
        )
        scrambler.exec(data)
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        timer.stop()
        if comm.world_rank == 0:
            timer.report("Gain scrambling")
    return


def setup_output(args, comm, mc):
    outpath = "{}/{:08}".format(args.outdir, mc)
    if comm.world_rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


@function_timer
def apply_polyfilter(args, comm, data, totalname):
    if args.polyorder:
        log = Logger.get()
        timer = Timer()
        if comm.world_rank == 0:
            log.info("Polyfiltering signal")
        timer.start()
        polyfilter = OpPolyFilter(
            order=args.polyorder, name=totalname, common_flag_mask=args.common_flag_mask
        )
        polyfilter.exec(data)
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        timer.stop()
        if comm.world_rank == 0:
            timer.report("Polynomial filtering")
    return


@function_timer
def apply_groundfilter(args, comm, data, totalname):
    if args.groundorder is None:
        return
    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0:
        log.info("Ground filtering signal")
    timer.start()
    groundfilter = OpGroundFilter(
        filter_order=args.groundorder,
        name=totalname,
        common_flag_mask=args.common_flag_mask,
    )
    groundfilter.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Ground filtering")
    return


@function_timer
def output_tidas(args, comm, data, totalname):
    if args.tidas is None:
        return
    tidas_path = os.path.abspath(args.tidas)

    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0:
        log.info(
            "Exporting data to a TIDAS volume at {}".format(tidas_path),
        )
    timer.start()
    export = OpTidasExport(
        tidas_path,
        TODTidas,
        backend="hdf5",
        use_intervals=True,
        create_opts={"group_dets": "sim"},
        ctor_opts={"group_dets": "sim"},
        cache_name=totalname,
    )
    export.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.Barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report(
            "Wrote simulated data to {}:{}"
            "".format(tidas_path, "total"),
        )
    return


@function_timer
def export_TOD(args, comm, data, totalname, other=None):
    if args.export is None:
        return

    log = Logger.get()
    timer = Timer()

    # Only import spt3g if we are writing out so3g files
    from spt3g import core as core3g
    from sotodlib.data.toast import ToastExport

    path = os.path.abspath(args.export)

    if comm.world_rank == 0:
        log.info("Exporting data to directory tree at {}".format(path))
    timer.start()
    export = ToastExport(
        path,
        prefix=args.bands,
        use_intervals=True,
        cache_name=totalname,
        cache_copy=other,
        mask_flag_common=data.obs[0]['tod'].TURNAROUND,
        filesize=2**30,
        units=core3g.G3TimestreamUnits.Tcmb,
    )
    export.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.Barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report(
            "Wrote simulated data to {}:{}"
            "".format(path, "total"))

    return


@function_timer
def get_time_communicators(comm, data):
    """ Split the world communicator by time.

    """
    time_comms = [("all", comm.comm_world)]

    # A process will only have data for one season and one day.  If more
    # than one season is observed, we split the communicator to make
    # season maps.

    my_season = data.obs[0]["season"]
    seasons = np.array(comm.comm_world.allgather(my_season))
    do_seasons = np.any(seasons != my_season)
    if do_seasons:
        season_comm = comm.comm_world.Split(my_season, comm.world_rank)
        time_comms.append((str(my_season), season_comm))

    # Split the communicator to make daily maps.  We could easily split
    # by month as well

    my_day = int(data.obs[0]["MJD"])
    my_date = data.obs[0]["date"]
    days = np.array(comm.comm_world.allgather(my_day))
    do_days = np.any(days != my_day)
    if do_days:
        day_comm = comm.comm_world.Split(my_day, comm.world_rank)
        time_comms.append((my_date, day_comm))

    return time_comms


@function_timer
def apply_madam(
    args,
    comm,
    time_comms,
    data,
    telescope_data,
    madampars,
    mc,
    firstmc,
    outpath,
    detweights,
    totalname_madam,
    first_call=True,
    extra_prefix=None,
):
    """ Use libmadam to bin and optionally destripe data.

    Bin and optionally destripe all conceivable subsets of the data.

    """
    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0:
        log.info("Making maps")
    timer.start()

    pars = copy.deepcopy(madampars)
    pars["path_output"] = outpath
    file_root = pars["file_root"]
    if extra_prefix is not None:
        file_root += "_{}".format(extra_prefix)

    if first_call:
        if mc != firstmc:
            pars["write_matrix"] = False
            pars["write_wcov"] = False
            pars["write_hits"] = False
    else:
        pars["kfirst"] = False
        pars["write_map"] = False
        pars["write_binmap"] = True
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False

    outputs = [
        pars["write_map"] in [True, "t", "T"],
        pars["write_binmap"] in [True, "t", "T"],
        pars["write_hits"] in [True, "t", "T"],
        pars["write_wcov"] in [True, "t", "T"],
        pars["write_matrix"] in [True, "t", "T"],
    ]
    if not np.any(outputs):
        if comm.world_rank == 0:
            log.info("No Madam outputs requested.  Skipping.")
        return

    if args.madam_noisefilter or not pars["kfirst"]:
        madam_intervals = None
    else:
        madam_intervals = "intervals"
    madam = OpMadam(
        params=pars,
        detweights=detweights,
        name=totalname_madam,
        common_flag_mask=args.common_flag_mask,
        purge_tod=False,
        intervals=madam_intervals,
        conserve_memory=args.conserve_memory,
    )

    if "info" in madam.params:
        info = madam.params["info"]
    else:
        info = 3

    for time_name, time_comm in time_comms:
        timer1 = Timer()
        for tele_name, tele_data in telescope_data:
            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if args.skip_daymaps:
                    continue
                if (len(telescope_data) > 1) and (tele_name == "all"):
                    # Skip daily maps over multiple telescopes
                    continue
                if first_call:
                    # Do not destripe daily maps
                    kfirst_save = pars["kfirst"]
                    write_map_save = pars["write_map"]
                    write_binmap_save = pars["write_binmap"]
                    pars["kfirst"] = False
                    pars["write_map"] = False
                    pars["write_binmap"] = True

            timer1.start()
            madam.params["file_root"] = "{}_{}_{}_time_{}".format(
                file_root, tele_name, args.bands, time_name
            )
            if time_comm == comm.comm_world:
                madam.params["info"] = info
            else:
                # Cannot have verbose output from concurrent mapmaking
                madam.params["info"] = 0
            if time_comm.rank == 0:
                log.info("Mapping {}".format(madam.params["file_root"]))
            madam.exec(tele_data, time_comm)
            time_comm.barrier()
            timer1.stop()
            if time_comm.rank == 0:
                timer1.report(
                    "Mapping {}".format(madam.params["file_root"]))
            if len(time_name.split("-")) == 3 and first_call:
                # Restore destriping parameters
                pars["kfirst"] = kfirst_save
                pars["write_map"] = write_map_save
                pars["write_binmap"] = write_binmap_save
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Madam")

    return


def main():
    env = Environment.get()
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_so_sim (total)")

    mpiworld, procs, rank = get_world()
    if rank == 0:
        env.print()
    if mpiworld is None:
        log.info("Running serially with one process at {}".format(str(datetime.now())))
    else:
        if rank == 0:
            log.info(
                "Running with {} processes at {}".format(procs, str(datetime.now()))
            )

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = Comm(world=mpiworld)

    memreport(comm.comm_world, "at the beginning of the pipeline")

    if comm.world_rank == 0:
        print(
            "Running with {} processes at {}".format(
                comm.comm_world.size, str(datetime.now())
            ),
            flush=True,
        )

    args, comm = parse_arguments(comm)

    # Initialize madam parameters

    madampars = setup_madam(args)

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

    # Split the communicator for day and season mapmaking

    time_comms = get_time_communicators(comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data)

    memreport(comm.comm_world, "after pointing")

    # Optionally rewrite the noise PSD:s in each observation to include
    # elevation-dependence
    get_elevation_noise(args, comm, data)

    # Prepare auxiliary information for distributed map objects

    _, localsm, subnpix = get_submaps(args, comm, data)

    memreport(comm.comm_world, "after submaps")

    if args.input_pysm_model:
        signalname = simulate_sky_signal(args, comm, data, schedules, subnpix, localsm)
    else:
        signalname = scan_sky_signal(args, comm, data, localsm, subnpix)

    memreport(comm.comm_world, "after PySM")

    # Set up objects to take copies of the TOD at appropriate times

    totalname = "total"

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    for mc in range(firstmc, firstmc + nmc):

        if comm.world_rank == 0:
            log.info("Processing MC = {}".format(mc))

        simulate_atmosphere(args, comm, data, mc, totalname)

        memreport(comm.comm_world, "after atmosphere")

        scale_atmosphere_by_bandpass(args, comm, data, totalname, mc)

        # update_atmospheric_noise_weights(args, comm, data, freq, mc)

        add_sky_signal(data, totalname, signalname)

        simulate_noise(args, comm, data, mc, totalname)
        simulate_sss(args, comm, data, mc, totalname)

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

        memreport(comm.comm_world, "after noise")

        scramble_gains(args, comm, data, mc, totalname)

        if mc == firstmc:
            # For the first realization and frequency, optionally
            # export the timestream data.
            output_tidas(args, comm, data, totalname)
            #export_TOD(args, comm, data, totalname, other=[signalname])
            export_TOD(args, comm, data, totalname)

            memreport(comm.comm_world, "after export")

        outpath = setup_output(args, comm, mc)

        # Bin and destripe maps

        apply_madam(
            args,
            comm,
            time_comms,
            data,
            telescope_data,
            madampars,
            mc,
            firstmc,
            outpath,
            detweights,
            totalname,
            first_call=True,
        )

        memreport(comm.comm_world, "after madam")

        if args.polyorder is not None or args.groundorder is not None:

            # Filter signal

            apply_polyfilter(args, comm, data, totalname)

            apply_groundfilter(args, comm, data, totalname)

            memreport(comm.comm_world, "after filter")

            # Bin maps

            apply_madam(
                args,
                comm,
                time_comms,
                data,
                telescope_data,
                madampars,
                mc,
                firstmc,
                outpath,
                detweights,
                totalname,
                first_call=False,
                extra_prefix="filtered",
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
