# Copyright (c) 2019-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.pipeline_tools import Telescope, Focalplane, Site, Schedule, CES
from toast.timing import function_timer, Timer
from toast.utils import Logger

from ...core import Hardware
from ...sim_hardware import get_example, sim_telescope_detectors
from .noise import get_analytic_noise


FOCALPLANE_RADII_DEG = {
    "LAT" : 3.6,
    "SAT1" : 17.8,
    "SAT2" : 17.8,
    "SAT3" : 17.8,
    "SAT4" : 17.2,
}


class SOTelescope(Telescope):
    def __init__(self, name):
        site = Site("Atacama", lat="-22.958064", lon="-67.786222", alt=5200)
        super().__init__(name, site=site)
        self.id = {
            # Use the same telescope index for all SATs to re-use the
            # atmospheric simulation
            #'LAT' : 0, 'SAT0' : 1, 'SAT1' : 2, 'SAT2' : 3, 'SAT3' : 4
            "LAT": 0,
            "SAT1": 4,
            "SAT2": 4,
            "SAT3": 4,
            "SAT4": 4,
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
        help="Comma-separated list of bands: f030 (27GHz), f040 (39GHz), "
        "f090 (93GHz), f150 (145GHz), "
        "f230 (225GHz), f290 (285GHz). "
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tube_slots",
        help="Comma-separated list of optics tube slots: c1 (UHF), i5 (UHF), "
        " i6 (MF), i1 (MF), i3 (MF), i4 (MF), o6 (LF),"
        " ST1 (MF), ST2 (MF), ST3 (UHF), ST4 (LF)."

    )
    group.add_argument(
        "--wafer_slots",
        help="Comma-separated list of optics tube slots. "
    )
    return


class BandParams:
    def __init__(self, band_name, band_data):
        self.name = band_name
        self.net = band_data["NET"] * 1e-6  # uK -> K
        self.fknee = band_data["fknee"] * 1e-3  # mHz -> Hz
        self.fmin = band_data["fmin"] * 1e-3  # mHz -> Hz
        # self.alpha = banddata[band]["alpha"]
        self.alpha = 1  # hardwire a sensible number. 3.5 is not realistic.
        self.A = band_data["A"]
        self.C = band_data["C"]
        self.lower = band_data["low"]  # GHz
        self.center = band_data["center"]  # GHz
        self.upper = band_data["high"]  # GHz
        return


class DetectorParams:
    def __init__(self, det_data, band, wafer_slot, tube_slot, telescope, index):
        """
        Args:
            det_data (dict) :  Dictionary of detector parameters
                from sotodlib.hardware
            band (BandParams) :  band parameters act as defaults
            wafer_slot (int) :  wafer slot number
            tube_slot (str) :  tube slot name
            telescope (str) :  telescope name
            index (int) :  RNG index
        """

        def get_par(key, default, scale=1):
            if key in det_data:
                return det_data[key] * scale
            else:
                return default

        self.band = band.name
        self.det_data = det_data
        self.net = get_par("NET", band.net, 1e-6)  # uK -> K
        self.fknee = get_par("fknee", band.fknee, 1e-3)  # mHz -> Hz
        self.fmin = get_par("fmin", band.fmin, 1e-3)  # mHz -> Hz
        self.alpha = get_par("alpha", band.alpha)
        self.alpha = 1  # hardwire a sensible number. 3.5 is not realistic.
        self.A = get_par("A", band.A)
        self.C = get_par("C", band.C)
        self.lower = get_par("low", band.lower)  # GHz
        self.center = get_par("center", band.center)  # GHz
        self.upper = get_par("high", band.upper)  # GHz
        self.center = 0.5 * (self.lower + self.upper)
        self.width = self.upper - self.lower
        self.wafer_slot = wafer_slot
        self.tube_slot = tube_slot
        self.telescope = telescope
        self.index = index
        return

    def get_dict(self):
        det_dict = {
            "NET": self.net,
            "fknee": self.fknee,
            "fmin": self.fmin,
            "alpha": self.alpha,
            "A": self.A,
            "C": self.C,
            "quat": self.det_data["quat"],
            "fwhm": self.det_data["fwhm"],
            "freq": self.center,
            "bandcenter_ghz": self.center,
            "bandwidth_ghz": self.width,
            "index": self.index,
            "telescope": self.telescope,
            "tube_slot": self.tube_slot,
            "wafer_slot": self.wafer_slot,
            "band": self.band,
        }
        return det_dict


@function_timer
def get_hardware(args, comm, verbose=False):
    """ Get the hardware configuration, either from file or by simulating.
    Then trim it down to the bands that were selected.
    """
    log = Logger.get()
    telescope = get_telescope(args, comm, verbose=verbose)
    timer = Timer()
    if comm.world_rank == 0:
        timer.start()
        if args.hardware:
            log.info(
                "Loading hardware configuration from {}..." "".format(args.hardware)
            )
            hw = Hardware(args.hardware)
            timer.report_clear("Load hardware map")
        else:
            log.info("Simulating default hardware configuration")
            hw = get_example()
            hw.data["detectors"] = sim_telescope_detectors(hw, telescope.name)
            timer.report_clear("Simulate hardware map")
        # Construct a running index for all detectors across all
        # telescopes for independent noise realizations
        det_index = {}
        for idet, det in enumerate(sorted(hw.data["detectors"])):
            det_index[det] = idet
        match = {"band": args.bands.replace(",", "|")}
        tube_slots = None
        if args.wafer_slots is not None:
            match["wafer_slot"]  = args.wafer_slots.split(",")
        elif args.tube_slots is not None:
            tube_slots = args.tube_slots.split(",")
        # If one provides both telescopes and tube_slots, the tube_slots matching *either*
        # will be concatenated
        #hw = hw.select(telescopes=[telescope.name], tube_slots=tube_slots, match=match)
        hw = hw.select(tube_slots=tube_slots, match=match)
        if args.thinfp:
            # Only accept a fraction of the detectors for
            # testing and development
            delete_detectors = []
            for det_name in hw.data["detectors"].keys():
                if (det_index[det_name] // 4) % args.thinfp != 0:
                    delete_detectors.append(det_name)
            for det_name in delete_detectors:
                del hw.data["detectors"][det_name]
        ndetector = len(hw.data["detectors"])
        if ndetector == 0:
            raise RuntimeError(
                "No detectors match query: telescope={}, "
                "tube_slots={}, match={}".format(telescope.name, tube_slots, match)
            )
        log.info(
            f"Telescope = {telescope.name} tube_slots = {args.tube_slots}, "
            f"wafer_slots = {args.wafer_slots}, bands = {args.bands}, "
            f"thinfp = {args.thinfp} matches {ndetector} detectors"
        )
        timer.report_clear("Select detectors")
    else:
        hw = None
        det_index = None
    if comm.comm_world is not None:
        hw = comm.comm_world.bcast(hw)
        det_index = comm.comm_world.bcast(det_index)
        if comm.world_rank == 0:
            timer.report_clear("Broadcast hardware map")
    return hw, telescope, det_index


@function_timer
def get_telescope(args, comm, verbose=False):
    """ Determine which telescope matches the detector selections
    """
    telescope = None
    if comm.world_rank == 0:
        hwexample = get_example()
        if args.wafer_slots is not None:
            wafer_slots = args.wafer_slots.split(",")
            wafer_map = hwexample.wafer_map()
            tube_slots = [wafer_map["tube_slots"][ws] for ws in wafer_slots]
        else:
            tube_slots = args.tube_slots.split(",")
        for tube_slot in tube_slots:
            for telescope_name, telescope_data in hwexample.data[
                "telescopes"
            ].items():
                if tube_slot in telescope_data["tube_slots"]:
                    if telescope is None:
                        telescope = SOTelescope(telescope_name)
                    elif telescope.name != telescope_name:
                        raise RuntimeError(
                            "Tubes '{}' span more than one telescope".format(tube_slots)
                        )
                    break
            if telescope is None:
                raise RuntimeError(
                    "Failed to match tube_slot = '{}' with a telescope".format(tube_slot)
                )
    if comm.comm_world is not None:
        telescope = comm.comm_world.bcast(telescope)
    return telescope


def get_focalplane(args, comm, hw, det_index, verbose=False):
    """ Translate hardware configuration into a TOAST focalplane dictionary
    """
    if comm.world_rank == 0:
        detector_data = {}
        band_params = {}
        for band_name, band_data in hw.data["bands"].items():
            band_params[band_name] = BandParams(band_name, band_data)
        # User may force the effective focal plane radius to be larger
        # than the default.  This will widen the size of the simulated
        # atmosphere but has not other effect for the time being.
        fpradius = None
        try:
            fpradius = args.focalplane_radius_deg
        except:
            pass
        if fpradius is None:
            fpradius = 0
        for det_name, det_data in hw.data["detectors"].items():
            # RNG index for this detector
            index = det_index[det_name]
            wafer_slot = det_data["wafer_slot"]
            # Determine which tube_slot has this wafer
            for tube_name, tube_data in hw.data["tube_slots"].items():
                if wafer_slot in tube_data["wafer_slots"]:
                    break
            # Determine which telescope has this tube slot
            for telescope_name, telescope_data in hw.data["telescopes"].items():
                if tube_name in telescope_data["tube_slots"]:
                    break
            fpradius = max(fpradius, FOCALPLANE_RADII_DEG[telescope_name])
            det_params = DetectorParams(
                det_data,
                band_params[det_data["band"]],
                wafer_slot,
                tube_name,
                telescope_name,
                index,
            )
            detector_data[det_name] = det_params.get_dict()
        # Create a focal plane in the telescope
        focalplane = Focalplane(
            detector_data=detector_data,
            sample_rate=args.sample_rate,
            radius_deg=fpradius,
        )
        # Update the noise model to include potential common modes
        get_analytic_noise(args, comm, focalplane)
    else:
        focalplane = None
    if comm.comm_world is not None:
        focalplane = comm.comm_world.bcast(focalplane)
    return focalplane

@function_timer
def load_focalplanes(args, comm, schedules, verbose=False):
    """ Attach a focalplane to each of the schedules.

    Args:
        schedules (list) :  List of Schedule instances.
            Each schedule has two members, telescope
            and ceslist, a list of CES objects.
    Returns:
        detweights (dict) : Inverse variance noise weights for every
            detector across all focal planes. In [K_CMB^-2].
            They can be used to bin the TOD.
    """
    # log = Logger.get()
    timer = Timer()
    timer.start()

    # Load focalplane information

    timer1 = Timer()
    timer1.start()
    hw, telescope, det_index = get_hardware(args, comm, verbose=verbose)
    focalplane = get_focalplane(args, comm, hw, det_index, verbose=verbose)
    telescope.focalplane = focalplane

    if comm.world_rank == 0 and verbose:
        timer1.report_clear("Collect focaplane information")

    for schedule in schedules:
        # Replace the telescope created from reading the observing schedule but
        # keep the weather object
        weather = schedule.telescope.site.weather
        schedule.telescope = telescope
        schedule.telescope.site.weather = weather

    detweights = telescope.focalplane.detweights

    timer.stop()
    if (comm.comm_world is None or comm.world_rank == 0) and verbose:
        timer.report("Loading focalplane")
    return detweights
