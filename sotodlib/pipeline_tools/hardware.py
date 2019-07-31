# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.pipeline_tools import Telescope
from toast.timing import function_timer, Timer
from toast.utils import Logger

from .. import hardware


class SOTelescope(Telescope):
    def __init__(self, name):
        self.name = name
        self.id = {
            # Use the same telescope index for all SATs to re-use the
            # atmospheric simulation
            #'LAT' : 0, 'SAT0' : 1, 'SAT1' : 2, 'SAT2' : 3, 'SAT3' : 4
            "LAT": 0,
            "SAT0": 4,
            "SAT1": 4,
            "SAT2": 4,
            "SAT3": 4,
        }[name]


def add_hw_args(parser):
    parser.add_argument(
        "--hardware", required=False, default=None, help="Input hardware file"
    )
    parser.add_argument(
        "--thinfp",
        required=False,
        type=np.int,
        help="Thin the focalplane by this factor",
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
    return


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


def get_det_params(
    detdata,
    band_net,
    band_fknee,
    band_fmin,
    band_alpha,
    band_A,
    band_C,
    band_lower,
    band_center,
    band_upper,
):
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
def load_focalplanes(args, comm, schedules, verbose=False):
    """ Attach a focalplane to each of the schedules.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    # Load focalplane information

    bands = args.bands.split(",")
    tubes = args.tubes.split(",")
    telescopes = []
    hwexample = hardware.get_example()
    for tube in tubes:
        for telescope, teledata in hwexample.data["telescopes"].items():
            if tube in teledata["tubes"]:
                telescopes.append(telescope)
                break

    focalplanes = []
    if comm.world_rank == 0:
        timer1 = Timer()
        for telescope, band, tube in zip(telescopes, bands, tubes):
            timer1.start()
            if args.hardware:
                if verbose:
                    log.info(
                        "Loading hardware configuration from {}..."
                        "".format(args.hardware)
                    )
                hw = hardware.Hardware(args.hardware)
            else:
                if verbose:
                    log.info("Simulating default hardware configuration")
                hw = hardware.get_example()
                hw.data["detectors"] = hardware.sim_telescope_detectors(hw, telescope)
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
                    "tubes={}, match={}".format(telescopes, tubes, match)
                )
            # Transfer the detector information into a TOAST dictionary
            focalplane = {}
            banddata = hw.data["bands"][band]
            (
                band_net,
                band_fknee,
                band_fmin,
                band_alpha,
                band_A,
                band_C,
                band_lower,
                band_center,
                band_upper,
            ) = get_band_params(banddata)
            for idet, (detname, detdata) in enumerate(hw.data["detectors"].items()):
                (net, fknee, fmin, alpha, A, C, center, width) = get_det_params(
                    detdata,
                    band_net,
                    band_fknee,
                    band_fmin,
                    band_alpha,
                    band_A,
                    band_C,
                    band_lower,
                    band_center,
                    band_upper,
                )
                wafer = detdata["wafer"]
                # Determine which tube has this wafer
                for tube, tubedata in hw.data["tubes"].items():
                    if wafer in tubedata["wafers"]:
                        break
                # RNG index for this detector
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
                "".format(telescope, tube, band, len(focalplane))
            )
    focalplanes = comm.comm_world.bcast(focalplanes)
    telescopes = comm.comm_world.bcast(telescopes)

    if len(schedules) == 1:
        schedules *= len(focalplanes)

    if len(focalplanes) != len(schedules):
        raise RuntimeError("Number of focalplanes must equal number of schedules")

    detweights = {}
    for schedule, focalplane, telescope in zip(schedules, focalplanes, telescopes):
        schedule.append(focalplane)
        schedule.append(SOTelescope(telescope))
        for detname, detdata in focalplane.items():
            # Transfer the detector properties from the band dictionary to the detectors
            net = detdata["NET"]
            # And build a dictionary of detector weights
            detweight = 1.0 / (args.sample_rate * net * net)
            if detname in detweights and detweights[detname] != detweight:
                raise RuntimeError("Detector weight for {} changes".format(detname))
            detweights[detname] = detweight

    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Loading focalplane(s)")
    return detweights, focalplanes
