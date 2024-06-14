# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Focalplane simulation tools.
"""

import re

from collections import OrderedDict

from copy import deepcopy

import numpy as np

from .core import Hardware


def sim_detectors_toast(hw, tele, tube_slots=None):
    """Update hardware model with simulated detector positions.

    Given a Hardware model, generate all detector properties for the specified
    telescope and optionally a subset of optics tube slots (for the LAT).  The
    detector dictionary of the hardware model is updated in place.

    This function requires the toast subpackage (and hence toast) to be
    importable.

    Args:
        hw (Hardware): The hardware object to update.
        tele (str): The telescope name.
        tube_slots (list, optional): The optional list of tube slots to include.

    Returns:
        None

    """
    try:
        from . import toast as sotoast
    except ImportError:
        msg = "Toast package is not importable, cannot simulate detector positions"
        raise RuntimeError(msg)

    sotoast.sim_focalplane.sim_telescope_detectors(
        hw, tele, tube_slots=tube_slots,
    )


def sim_detectors_physical_optics(hw, tele, tube_slots=None):
    """Update hardware model with simulated detector positions.

    Given a Hardware model, generate all detector properties for the specified
    telescope and optionally a subset of optics tube slots (for the LAT).  The
    detector dictionary of the hardware model is updated in place.

    This function uses information from physical optics simulations to estimate
    the location of detectors.

    Args:
        hw (Hardware): The hardware object to update.
        tele (str): The telescope name.
        tube_slots (list, optional): The optional list of tube slots to include.

    Returns:
        None

    """
    raise NotImplementedError("Not yet implemented")


def sim_nominal():
    """Return a simulated nominal hardware configuration.

    This returns a simulated Hardware object with the nominal instrument
    properties / metadata, but with an empty set of detector locations.

    This can then be passed to one of the detector simulation functions
    to build up the list of detectors.

    Returns:
        (Hardware): Hardware object with nominal metadata.

    """
    cnf = OrderedDict()

    bands = OrderedDict()

    bnd = OrderedDict()
    bnd["center"] = 25.7
    bnd["low"] = 21.7
    bnd["high"] = 29.7
    bnd["bandpass"] = ""
    bnd["NET"] = 435.1
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    # Noise elevation scaling fits from Carlos Sierra
    # These numbers are for V3 LAT baseline
    bnd["A"] = 0.06
    bnd["C"] = 0.92
    bnd["NET_corr"] = 1.10
    bands["LAT_f030"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 38.9
    bnd["low"] = 30.9
    bnd["high"] = 46.9
    bnd["bandpass"] = ""
    bnd["NET"] = 281.5
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.16
    bnd["C"] = 0.79
    bnd["NET_corr"] = 1.02
    bands["LAT_f040"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 92.0
    bnd["low"] = 79.0
    bnd["high"] = 105.0
    bnd["bandpass"] = ""
    bnd["NET"] = 361.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.16
    bnd["C"] = 0.80
    bnd["NET_corr"] = 1.09
    bands["LAT_f090"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 147.5
    bnd["low"] = 130.0
    bnd["high"] = 165.0
    bnd["bandpass"] = ""
    bnd["NET"] = 352.4
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.17
    bnd["C"] = 0.78
    bnd["NET_corr"] = 1.01
    bands["LAT_f150"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 225.7
    bnd["low"] = 196.7
    bnd["high"] = 254.7
    bnd["bandpass"] = ""
    bnd["NET"] = 724.4
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.29
    bnd["C"] = 0.62
    bnd["NET_corr"] = 1.02
    bands["LAT_f230"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 285.4
    bnd["low"] = 258.4
    bnd["high"] = 312.4
    bnd["bandpass"] = ""
    bnd["NET"] = 1803.9
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.36
    bnd["C"] = 0.53
    bnd["NET_corr"] = 1.00
    bands["LAT_f290"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 25.7
    bnd["low"] = 21.7
    bnd["high"] = 29.7
    bnd["bandpass"] = ""
    bnd["NET"] = 314.1
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    # Noise elevation scaling fits from Carlos Sierra
    # These numbers are for V3 SAT baseline
    bnd["A"] = 0.06
    bnd["C"] = 0.92
    bnd["NET_corr"] = 1.06
    bands["SAT_f030"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 38.9
    bnd["low"] = 30.9
    bnd["high"] = 46.9
    bnd["bandpass"] = ""
    bnd["NET"] = 225.8
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.19
    bnd["C"] = 0.76
    bnd["NET_corr"] = 1.01
    bands["SAT_f040"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 92.0
    bnd["low"] = 79.0
    bnd["high"] = 105.0
    bnd["bandpass"] = ""
    bnd["NET"] = 245.1
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.19
    bnd["C"] = 0.76
    bnd["NET_corr"] = 1.04
    bands["SAT_f090"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 147.5
    bnd["low"] = 130.0
    bnd["high"] = 165.0
    bnd["bandpass"] = ""
    bnd["NET"] = 250.2
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.23
    bnd["C"] = 0.70
    bnd["NET_corr"] = 1.02
    bands["SAT_f150"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 225.7
    bnd["low"] = 196.7
    bnd["high"] = 254.7
    bnd["bandpass"] = ""
    bnd["NET"] = 540.3
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.35
    bnd["C"] = 0.54
    bnd["NET_corr"] = 1.00
    bands["SAT_f230"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 285.4
    bnd["low"] = 258.4
    bnd["high"] = 312.4
    bnd["bandpass"] = ""
    bnd["NET"] = 1397.5
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = 0.42
    bnd["C"] = 0.45
    bnd["NET_corr"] = 1.00
    bands["SAT_f290"] = bnd

    # Special "band" for dark bolometers

    bnd = OrderedDict()
    bnd["center"] = np.nan
    bnd["low"] = np.nan
    bnd["high"] = np.nan
    bnd["bandpass"] = ""
    bnd["NET"] = 1000.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 1.0
    bnd["A"] = np.nan
    bnd["C"] = np.nan
    bnd["NET_corr"] = 1.00
    bands["NC"] = bnd

    cnf["bands"] = bands

    wafer_slots = OrderedDict()

    wtypes = ["LAT_UHF", "SAT_UHF", "LAT_MF", "SAT_MF", "LAT_LF", "SAT_LF"]
    wcnt = {
        "LAT_LF": 1*3,
        "LAT_MF": 4*3,
        "LAT_UHF": 2*3,
        "SAT_LF": 1*7,
        "SAT_MF": 2*7,
        "SAT_UHF": 1*7
    }
    wnp = {
        "LAT_LF": 37,
        "LAT_MF": 432,
        "LAT_UHF": 432,
        "SAT_LF": 37,
        "SAT_MF": 432,
        "SAT_UHF": 432
    }
    wpixmm = {
        "LAT_LF": 18.0,
        "LAT_MF": 5.3,
        "LAT_UHF": 5.3,
        "SAT_LF": 18.0,
        "SAT_MF": 5.3,
        "SAT_UHF": 5.3
    }
    wrhombgap = {
        "LAT_MF": 0.71,
        "LAT_UHF": 0.71,
        "SAT_MF": 0.71,
        "SAT_UHF": 0.71
    }
    wbd = {
        "LAT_LF": ["LAT_f030", "LAT_f040"],
        "LAT_MF": ["LAT_f090", "LAT_f150"],
        "LAT_UHF": ["LAT_f230", "LAT_f290"],
        "SAT_LF": ["SAT_f030", "SAT_f040"],
        "SAT_MF": ["SAT_f090", "SAT_f150"],
        "SAT_UHF": ["SAT_f230", "SAT_f290"]
    }
    windx = 0
    cardindx = 0
    for wt in wtypes:
        for ct in range(wcnt[wt]):
            wn = "w{:02d}".format(windx)
            wf = OrderedDict()
            wf["type"] = wt
            if ((wt == "LAT_LF") or (wt == "SAT_LF")):
                wf["packing"] = "S"
            else:
                wf["packing"] = "F"
                wf["rhombusgap"] = wrhombgap[wt]
            wf["npixel"] = wnp[wt]
            wf["pixsize"] = wpixmm[wt]
            wf["bands"] = wbd[wt]
            wf["card_slot"] = "card_slot{:02d}".format(cardindx)
            wf["wafer_name"] = ""
            cardindx += 1
            wafer_slots[wn] = wf
            windx += 1

    cnf["wafer_slots"] = wafer_slots

    tube_slots = OrderedDict()

    woff = {
        "LAT_LF": 0,
        "LAT_MF": 0,
        "LAT_UHF": 0,
        "SAT_LF": 0,
        "SAT_MF": 0,
        "SAT_UHF": 0
    }

    ltubes = ["LAT_UHF", "LAT_UHF", "LAT_MF", "LAT_MF", "LAT_MF", "LAT_MF", "LAT_LF"]

    # The optics tubes are arranged using several conventions and here we map between
    # them.  Note that for this hardware model, the projection on the sky is defined
    # at 60 degree elevation with no boresight rotation.

    # TOAST hexagon layout positions in Xi/Eta coordinates
    ltube_toasthex_pos = [0, 1, 2, 3, 5, 6, 10]

    # "Optics" locations as given in several SO slide decks.  Note this is flipped in
    # some figures.  See:
    # https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/101974017/Focal+Plane+Coordinates
    ltube_optics_pos = [1, 3, 5, 4, 8, 9, 12]

    # Cryo team names for these positions
    ltube_cryonames=["c1", "i5", "i6", "i1", "i3", "i4", "o6"]

    lat_ufm_slot = [
        0,
        1,
        2,
    ]

    lat_ufm_thetarot = [
        240.0,
        0.0,
        120.0,
    ]

    for tindx in range(7):
        nm = ltube_cryonames[tindx]
        ttyp = ltubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 128.4
        tb["wafer_slots"] = list()
        tb["wafer_slot_angle"] = [
            lat_ufm_thetarot[tw] for tw in range(3)
        ] # Degrees
        # The "slot" here is the relative slot (ws0 - ws2) within the tube.
        tb["wafer_ufm_slot"] = list()
        for tw in range(3):
            off = 0
            for w, props in cnf["wafer_slots"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        props["tube_index"] = tw
                        tb["wafer_slots"].append(w)
                        tb["wafer_ufm_slot"].append(lat_ufm_slot[tw])
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["toast_hex_pos"] = ltube_toasthex_pos[tindx]
        tb["optics_pos"] = ltube_optics_pos[tindx]
        tb["tube_name"] = ""
        tb["receiver_name"] = ""
        tube_slots[nm] = tb

    # These are taken from:
    # https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/101974017/Focal+Plane+Coordinates#UFM-Layout.1
    hex_to_ufm_slot = [
        0,
        2,
        1,
        6,
        5,
        4,
        3,
    ]
    hex_to_ufm_loc = [
        "AX",
        "NE",
        "NO",
        "NW",
        "SW",
        "SO",
        "SE",
    ]
    hex_to_ufm_thetarot = [
        240.0,
        0.0,
        300.0,
        180.0,
        120.0,
        60.0,
        60.0,
    ]

    stubes = ["SAT_MF", "SAT_MF", "SAT_UHF", "SAT_LF"]
    for tindx in range(4):
        nm = "ST{:d}".format(tindx+1)
        ttyp = stubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 128.4
        tb["wafer_slots"] = list()
        tb["wafer_slot_angle"] = [
            hex_to_ufm_thetarot[tw] for tw in range(7)
        ] # Degrees
        # The "slot" here is the relative slot (ws0 - ws6) within the tube.
        tb["wafer_ufm_slot"] = list()
        # The "loc" here is the compass direction name (e.g. NO, NE, SW, etc.)
        tb["wafer_ufm_loc"] = list()
        for tw in range(7):
            off = 0
            for w, props in cnf["wafer_slots"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        props["tube_index"] = tw
                        tb["wafer_slots"].append(w)
                        tb["wafer_ufm_slot"].append(hex_to_ufm_slot[tw])
                        tb["wafer_ufm_loc"].append(hex_to_ufm_loc[tw])
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["toast_hex_pos"] = 0
        tb["optics_pos"] = 0
        tb["tube_name"] = ""
        tb["receiver_name"] = ""
        tube_slots[nm] = tb

    cnf["tube_slots"] = tube_slots

    telescopes = OrderedDict()

    tele = OrderedDict()
    tele["tube_slots"] = ["c1", "i5", "i6", "i1", "i3", "i4", "o6"]
    tele["platescale"] = 0.00495
    # This tube spacing in mm corresponds to 1.78 degrees projected on
    # the sky at a plate scale of 0.00495 deg/mm.
    tele["tubespace"] = 359.6
    fwhm = OrderedDict()
    fwhm["LAT_f030"] = 7.4
    fwhm["LAT_f040"] = 5.1
    fwhm["LAT_f090"] = 2.2
    fwhm["LAT_f150"] = 1.4
    fwhm["LAT_f230"] = 1.0
    fwhm["LAT_f290"] = 0.9
    tele["fwhm"] = fwhm
    tele["platform_name"] = ""
    telescopes["LAT"] = tele

    fwhm_sat = OrderedDict()
    fwhm_sat["SAT_f030"] = 91.0
    fwhm_sat["SAT_f040"] = 63.0
    fwhm_sat["SAT_f090"] = 30.0
    fwhm_sat["SAT_f150"] = 17.0
    fwhm_sat["SAT_f230"] = 11.0
    fwhm_sat["SAT_f290"] = 9.0

    tele = OrderedDict()
    tele["tube_slots"] = ["ST1"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = fwhm_sat
    tele["platform_name"] = ""
    telescopes["SAT1"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST2"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = fwhm_sat
    tele["platform_name"] = ""
    telescopes["SAT2"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST3"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = fwhm_sat
    tele["platform_name"] = ""
    telescopes["SAT3"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST4"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = fwhm_sat
    tele["platform_name"] = ""
    telescopes["SAT4"] = tele

    cnf["telescopes"] = telescopes

    card_slots = OrderedDict()
    crate_slots = OrderedDict()

    crt_indx = 0

    for tel in cnf["telescopes"]:
        crn = "crate_slot{:02d}".format(crt_indx)
        crt = OrderedDict()
        crt["card_slots"] = list()
        crt["telescope"] = tel
        crt["crate_name"] = ""

        ## get all the wafer card numbers for a telescope
        tb_wfrs = [cnf["tube_slots"][t]["wafer_slots"] for t in cnf["telescopes"][tel]["tube_slots"]]
        tl_wfrs = [i for sl in tb_wfrs for i in sl]
        wafer_cards = [cnf["wafer_slots"][w]["card_slot"] for w in tl_wfrs]

        # add all cards to the card table and assign to crates
        for crd in wafer_cards:
            cdprops = OrderedDict()
            cdprops["nbias"] = 12
            cdprops["nAMC"] = 2
            cdprops["nchannel"] = 1764
            cdprops["card_name"] = ""
            card_slots[crd] = cdprops

            crt["card_slots"].append(crd)

            # name new crates when current one is full
            if ('S' in tel and len(crt["card_slots"]) >=4) or len(crt["card_slots"]) >=6:
                crate_slots[crn] = crt
                crt_indx += 1
                crn = "crate_slot{:02d}".format(crt_indx)
                crt = OrderedDict()
                crt["card_slots"] = list()
                crt["telescope"] = tel
                crt["crate_name"] = ""

        # each telescope starts with a new crate
        crate_slots[crn] = crt
        crt_indx += 1

    cnf["card_slots"] = card_slots
    cnf["crate_slots"] = crate_slots

    # Add an empty set of detectors here, in case the user just wants access to
    # the hardware metadata.
    cnf["detectors"] = OrderedDict()

    hw = Hardware()
    hw.data = cnf

    return hw


def telescope_tube_wafer():
    """Global mapping of telescopes, tubes, and wafers used in simulations.

    This mapping is here rather than core.hardware, so that we could put
    alternate definitions there for actual fielded configurations.

    Returns:
        (dict):  The mapping

    """
    hw = sim_nominal()
    result = dict()
    for tele_name, tele_props in hw.data["telescopes"].items():
        tb = dict()
        for tube_name in tele_props["tube_slots"]:
            tube_props = hw.data["tube_slots"][tube_name]
            tb[tube_name] = list(tube_props["wafer_slots"])
        result[tele_name] = tb
    return result
