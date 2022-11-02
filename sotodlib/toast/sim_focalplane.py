# Copyright (c) 2018-2022 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Focalplane simulation tools.
"""

import re
from collections import OrderedDict

import astropy.units as u
import numpy as np
from toast.instrument_coords import xieta_to_quat, quat_to_xieta
from toast.instrument_sim import (
    hex_nring,
    hex_xieta_row_col,
    hex_layout,
    rhomb_dim,
    rhomb_xieta_row_col,
    rhombus_layout,
    rhomb_gamma_angles_qu,
)
import toast.qarray as qa


def rhombus_hex_layout(rhombus_npos, rhombus_width, gap, pos_rotate=None, killpos=None):
    """
    Construct a hexagon from 3 rhombi.

    Args:
        rhombus_npos (int):  The number of positions in one rhombus.
        rhombus_width (Quantity):  The angle subtended by the width of one rhombus
            along the short dimension.
        gap (Quantity):  The angular gap between the edges of the rhombi.
        pos_rotate (array, Quantity): An additional angle rotation of each position
            on each rhombus before the rhombus is rotated into place.
        killpos (list, optional): Pixel indices to remove for mechanical
            reasons.

    Returns:
        (array):  Quaternion array of pixel positions.

    """
    # width in radians
    width_rad = rhombus_width.to_value(u.radian)

    # Gap between rhombi in radians
    gap_rad = gap.to_value(u.radian)

    # Quaternion offsets of the 3 rhombi
    centers = [
        xieta_to_quat(
            0.25 * np.sqrt(3.0) * width_rad + 0.5 * gap_rad,
            -0.25 * width_rad - gap_rad / (2 * np.sqrt(3.0)),
            np.pi / 6,
        ),
        xieta_to_quat(
            0.0,
            0.5 * width_rad + gap_rad / np.sqrt(3.0),
            -0.5 * np.pi,
        ),
        xieta_to_quat(
            -0.25 * np.sqrt(3.0) * width_rad - 0.5 * gap_rad,
            -0.25 * width_rad - gap_rad / (2 * np.sqrt(3.0)),
            5 * np.pi / 6,
        ),
    ]

    # Quaternion array of outputs, without missing pixel locations.
    nkill = len(killpos)
    killset = set(killpos)
    result = np.zeros((3 * rhombus_npos - nkill, 4), dtype=np.float64)

    # Pre-rotation of all pixels
    pos_ang = u.Quantity(np.zeros(rhombus_npos, dtype=np.float64), u.radian)
    if pos_rotate is not None:
        pos_ang = pos_rotate

    all_quat = dict()
    for irhomb, cent in enumerate(centers):
        quat = rhombus_layout(
            rhombus_npos,
            rhombus_width,
            "",
            "",
            pos_ang,
            center=cent,
            pos_offset=irhomb * rhombus_npos,
        )
        all_quat.update(quat)
    pquat = {int(x): y["quat"] for x, y in all_quat.items()}

    poff = 0
    for p, q in pquat.items():
        if p not in killset:
            result[poff, :] = q
            poff += 1

    return result


def sim_wafer_detectors(
    hw,
    wafer_slot,
    platescale,
    fwhm,
    band=None,
    center=None,
):
    """Generate detector properties for a wafer.

    Given a Hardware configuration, generate all detector properties for
    the specified wafer and optionally only the specified band.

    Args:
        hw (Hardware): The hardware properties.
        wafer_slot (str): The wafer slot name.
        platescale (float): The plate scale in degrees / mm.
        fwhm (dict): Dictionary of nominal FWHM values in arcminutes for
            each band.
        band (str, optional): Optionally only use this band.
        center (array, optional): The quaternion offset of the center.

    Returns:
        (OrderedDict): The properties of all selected detectors.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)

    # The properties of this wafer
    wprops = hw.data["wafer_slots"][wafer_slot]

    # The readout card and its properties
    card_slot = wprops["card_slot"]
    cardprops = hw.data["card_slots"][card_slot]

    # The bands
    bands = wprops["bands"]
    if band is not None:
        if band in bands:
            bands = [band]
        else:
            raise RuntimeError(
                "band '{}' not valid for wafer '{}'".format(band, wafer_slot)
            )

    # Lay out the pixel locations depending on the wafer type.  Also
    # compute the polarization orientation rotation, as well as the A/B
    # handedness for the Sinuous detectors.

    npix = wprops["npixel"]
    pixsep = platescale * wprops["pixsize"]
    layout_A = None
    layout_B = None
    handed = None
    kill = []
    if wprops["packing"] == "F":
        # Feedhorn (NIST style)
        gap = platescale * wprops["rhombusgap"]
        nrhombus = npix // 3

        # This dim is also the number of pixels along the short axis.
        dim = rhomb_dim(nrhombus)

        # This is the center-center distance along the short axis
        width = (dim - 1) * pixsep

        # The orientation within each rhombus alternates between zero and 45
        # degrees.  However there is an offset.  We choose this arbitrarily
        # for the nominal rhombus position, and then the rotation of the
        # other 2 rhombi will naturally modulate this.
        pol_A = np.zeros(nrhombus, dtype=np.float64)
        pol_B = np.zeros(nrhombus, dtype=np.float64)
        poloff = 22.5
        for p in range(nrhombus):
            # get the row / col of the pixel
            row, col = rhomb_xieta_row_col(nrhombus, p)
            if np.mod(row, 2) == 0:
                pol_A[p] = 0.0 + poloff
            else:
                pol_A[p] = 45.0 + poloff
            pol_B[p] = 90.0 + pol_A[p]

        # We are going to remove 2 pixels for mechanical reasons
        kf = dim * (dim - 1) // 2
        kill = [(dim * dim + kf), (dim * dim + kf) + dim - 2]
        layout_A = rhombus_hex_layout(
            nrhombus,
            width * u.degree,
            gap * u.degree,
            pos_rotate=pol_A * u.degree,
            killpos=kill,
        )
        layout_B = rhombus_hex_layout(
            nrhombus,
            width * u.degree,
            gap * u.degree,
            pos_rotate=pol_B * u.degree,
            killpos=kill,
        )
    elif wprops["packing"] == "S":
        # Sinuous (Berkeley style)
        # This is the center-center distance along the vertex-vertex axis
        width = (2 * (hex_nring(npix) - 1)) * pixsep

        # We rotate the hex layout 30 degrees about the center
        hex_cent = qa.from_axisangle(zaxis, np.pi / 6)

        # The sinuous handedness is chosen so that A/B pairs of pixels have the
        # same nominal orientation but trail each other along the
        # vertex-vertex axis of the hexagon.  The polarization orientation
        # changes every other column

        if npix == 37:
            pol_A = np.array(
                [
                    45.0,
                    0.0,
                    45.0,
                    45.0,
                    45.0,
                    45.0,
                    0.0,
                    0.0,
                    0.0,
                    45.0,
                    45.0,
                    0.0,
                    0.0,
                    0.0,
                    45.0,
                    45.0,
                    0.0,
                    0.0,
                    0.0,
                    45.0,
                    0.0,
                    0.0,
                    45.0,
                    45.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    45.0,
                    45.0,
                    0.0,
                    0.0,
                    45.0,
                    45.0,
                    45.0,
                ]
            )
            handed = [
                "R",
                "L",
                "R",
                "L",
                "L",
                "R",
                "L",
                "R",
                "L",
                "R",
                "L",
                "R",
                "R",
                "R",
                "L",
                "R",
                "L",
                "R",
                "R",
                "L",
                "R",
                "L",
                "R",
                "L",
                "R",
                "L",
                "L",
                "L",
                "L",
                "R",
                "L",
                "R",
                "L",
                "R",
                "L",
                "L",
                "L",
            ]
            pol_B = 90.0 + pol_A
        else:
            handed = list()
            pol_A = np.zeros(npix, dtype=np.float64)
            pol_B = np.zeros(npix, dtype=np.float64)
            for p in range(npix):
                row, col = hex_xieta_row_col(npix, p)
                if np.mod(col, 2) == 0:
                    handed.append("L")
                else:
                    handed.append("R")
                if np.mod(col, 4) < 2:
                    pol_A[p] = 0.0
                else:
                    pol_A[p] = 45.0
                pol_B[p] = 90.0 + pol_A[p]
        quat_A = hex_layout(
            npix, width * u.degree, "", "", pol_A * u.degree, center=hex_cent
        )
        quat_B = hex_layout(
            npix, width * u.degree, "", "", pol_B * u.degree, center=hex_cent
        )

        layout_A = np.zeros((len(quat_A), 4), dtype=np.float64)
        pquat_A = {int(x): y["quat"] for x, y in quat_A.items()}
        for p, q in pquat_A.items():
            layout_A[p, :] = q

        layout_B = np.zeros((len(quat_B), 4), dtype=np.float64)
        pquat_B = {int(x): y["quat"] for x, y in quat_B.items()}
        for p, q in pquat_B.items():
            layout_B[p, :] = q
    else:
        raise RuntimeError("Unknown wafer packing '{}'".format(wprops["packing"]))

    # Now we go through each pixel and create the orthogonal detectors for
    # each band.
    dets = OrderedDict()

    # chan_per_AMC = cardprops["nchannel"] // cardprops["nAMC"]
    chan_per_AMC = 910
    chan_per_mux = 64
    chan_per_bias = cardprops["nchannel"] // cardprops["nbias"]
    readout_freq_range = np.linspace(4.0, 6.0, chan_per_AMC)
    readout_freq = np.append(readout_freq_range, readout_freq_range)

    doff = 0
    p = 0
    idoff = int(wafer_slot[-2:]) * 10000
    for px in range(npix):
        if px in kill:
            continue
        pstr = "{:03d}".format(p)
        for b in bands:
            for pl, layout, pol in zip(
                ["A", "B"],
                [layout_A, layout_B],
                [pol_A, pol_B],
            ):
                dprops = OrderedDict()
                dprops["wafer_slot"] = wafer_slot
                dprops["ID"] = idoff + doff
                dprops["pixel"] = pstr
                dprops["band"] = b
                dprops["fwhm"] = fwhm[b]
                dprops["pol"] = pl
                if handed is not None:
                    dprops["handed"] = handed[p]
                # Made-up assignment to readout channels
                dprops["card_slot"] = card_slot
                dprops["channel"] = doff
                dprops["AMC"] = doff // chan_per_AMC
                dprops["bias"] = doff // chan_per_bias
                dprops["readout_freq_GHz"] = readout_freq[doff]
                dprops["bondpad"] = doff - (doff // chan_per_mux) * chan_per_mux
                dprops["mux_position"] = doff // chan_per_mux
                # Layout quaternion offset is from the origin.  Now we apply
                # the rotation of the wafer center.
                if center is not None:
                    dprops["quat"] = qa.mult(center, layout[p]).flatten()
                else:
                    dprops["quat"] = layout[p].flatten()
                dprops["detector_name"] = ""
                dname = "{}_p{}_{}_{}".format(wafer_slot, pstr, b, pl)
                dets[dname] = dprops
                doff += 1
        p += 1

    return dets


def sim_telescope_detectors(hw, tele, tube_slots=None):
    """Update hardware model with simulated detector positions.

    Given a Hardware model, generate all detector properties for the specified
    telescope and optionally a subset of optics tube slots (for the LAT).  The
    detector dictionary of the hardware model is updated in place.

    This uses helper functions for focalplane layout from the upstream toast
    package

    Args:
        hw (Hardware): The hardware object to update.
        tele (str): The telescope name.
        tube_slots (list, optional): The optional list of tube slots to include.

    Returns:
        None

    """

    zaxis = np.array([0, 0, 1], dtype=np.float64)
    thirty = np.pi / 6.0
    # The properties of this telescope
    teleprops = hw.data["telescopes"][tele]
    platescale = teleprops["platescale"]
    fwhm = teleprops["fwhm"]

    # The tubes
    alltubes = teleprops["tube_slots"]
    ntube = len(alltubes)
    if tube_slots is None:
        tube_slots = alltubes
    else:
        for t in tube_slots:
            if t not in alltubes:
                raise RuntimeError(
                    "Invalid tube_slot '{}' for telescope '{}'".format(t, tele)
                )

    alldets = OrderedDict()
    if ntube == 1:
        # This is a SAT.  We have one tube at the center.
        tubeprops = hw.data["tube_slots"][tube_slots[0]]
        waferspace = tubeprops["waferspace"] * platescale

        # Wafers are arranged in a rotated hexagon shape, however these locations
        # are rotated from a normal layout.
        wcenters = hex_layout(
            7,
            (3.0 * np.sqrt(3.0) / 2) * waferspace * u.degree,
            "",
            "",
            np.zeros(7) * u.degree,
        )
        centers = np.zeros((7, 4), dtype=np.float64)
        qrot = qa.from_axisangle(zaxis, -np.pi / 6)
        for p, q in wcenters.items():
            centers[int(p), :] = qa.mult(qrot, q["quat"])

        windx = 0
        for wafer_slot in tubeprops["wafer_slots"]:
            dets = sim_wafer_detectors(
                hw, wafer_slot, platescale, fwhm, center=centers[windx]
            )
            alldets.update(dets)
            windx += 1
    else:
        # This is the LAT.  We layout detectors for the case of 30 degree elevation
        # and no boresight rotation.  Compute the tube centers.

        # Inter-tube spacing
        tubespace = teleprops["tubespace"]

        # Pre-rotation of the tube arrangement
        tuberot = 0.0 * np.ones(19, dtype=np.float64)

        # Hexagon layout
        tube_quats = hex_layout(
            19,
            4 * (tubespace * platescale) * u.degree,
            "",
            "",
            tuberot * u.degree,
        )
        tcenters = np.zeros((19, 4), dtype=np.float64)
        for p, q in tube_quats.items():
            tcenters[int(p), :] = q["quat"]

        tindx = 0
        for tube_slot in tube_slots:
            tubeprops = hw.data["tube_slots"][tube_slot]
            waferspace = tubeprops["waferspace"]
            location = tubeprops["toast_hex_pos"]

            wradius = 0.5 * (waferspace * platescale * np.pi / 180.0)
            qwcenters = [
                xieta_to_quat(-wradius, wradius / np.sqrt(3.0), thirty * 4),
                xieta_to_quat(0.0, -2.0 * wradius / np.sqrt(3.0), 0.0),
                xieta_to_quat(wradius, wradius / np.sqrt(3.0), -thirty * 4),
            ]

            centers = list()
            for qwc in qwcenters:
                centers.append(qa.mult(tcenters[location], qwc))

            windx = 0
            for wafer_slot in tubeprops["wafer_slots"]:
                dets = sim_wafer_detectors(
                    hw, wafer_slot, platescale, fwhm, center=centers[windx]
                )
                # dname = list(dets.keys())[0]
                # dquat = dets[dname]["quat"]
                # ddir = qa.rotate(dquat, zaxis)
                # dxi, deta, dgamma = quat_to_xieta(dquat)
                # print(
                #     f"tube {tube_slot} ({location}), wafer {wafer_slot}, det 0 dir = {ddir}, xi/eta/gamma = {dxi}, {deta}, {dgamma}"
                # )
                alldets.update(dets)
                windx += 1
            tindx += 1

        # Rotate the focalplane to the nominal horizontal orientation
        fp_rot = qa.from_axisangle(zaxis, 0.5 * np.pi)
        for d, props in alldets.items():
            props["quat"] = qa.mult(fp_rot, props["quat"])

    if "detectors" in hw.data:
        hw.data["detectors"].update(alldets)
    else:
        hw.data["detectors"] = alldets
