# Copyright (c) 2018-2023 Simons Observatory.
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
    rhombus_hex_layout,
)
import toast.qarray as qa
import toast.utils

from sotodlib.io.metadata import read_dataset
import sotodlib.core.metadata.loader as loader


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

    By convention, polarization angle quantities are stored in degrees and
    fwhm is stored in arcminutes.  These units are stripped to ease serialization
    to TOML and JSON.  The units are restored when constructing focalplane tables
    for TOAST.

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
    pixsep = platescale * wprops["pixsize"] * u.degree
    handed = None
    kill = []
    if wprops["packing"] == "F":
        # Feedhorn (NIST style)
        # The "gap" is the **additional** space beyond the normal pixel
        # separation.  For S.O., we always have the gap equal to nominal
        # pixel spacing.
        gap = 0.0 * u.degree
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
        pol_A = u.Quantity(pol_A, u.degree)
        pol_B = u.Quantity(pol_B, u.degree)

        layout_A = rhombus_hex_layout(
            nrhombus, width, "", "", gap=gap, pol=pol_A
        )
        layout_B = rhombus_hex_layout(
            nrhombus, width, "", "", gap=gap, pol=pol_B
        )

        # We are going to remove 2 pixels for mechanical reasons
        kf = dim * (dim - 1) // 2
        kill = [(dim * dim + kf), (dim * dim + kf) + dim - 2]

        # The orientation of the wafer with respect to the focalplane
        wafer_rot_rad = 0.0
    elif wprops["packing"] == "S":
        # Sinuous (Berkeley style)
        # This is the center-center distance along the vertex-vertex axis
        width = (2 * (hex_nring(npix) - 1)) * pixsep

        # The orientation of the wafer with respect to the focalplane.  This
        # is 30 degrees.
        wafer_rot_rad = np.pi / 6
        hex_cent = qa.from_axisangle(zaxis, wafer_rot_rad)

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

        pol_A = u.Quantity(pol_A, u.degree)
        pol_B = u.Quantity(pol_B, u.degree)
        layout_A = hex_layout(npix, width, "", "", pol_A, center=hex_cent)
        layout_B = hex_layout(npix, width, "", "", pol_B, center=hex_cent)
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
        if npix < 100:
            pxstr = f"{px:02d}"
        else:
            pxstr = f"{px:03d}"
        pstr = f"{p:03d}"
        for b in bands:
            for pl, layout, in zip(
                ["A", "B"],
                [layout_A, layout_B],
            ):
                dprops = OrderedDict()
                dprops["wafer_slot"] = wafer_slot
                dprops["ID"] = idoff + doff
                dprops["pixel"] = pstr
                dprops["band"] = b
                dprops["fwhm"] = fwhm[b]
                dprops["pol"] = pl

                # Polarization angle in wafer basis.  This is the gamma angle
                # returned by the layout functions above, less the rotation
                # of the wafer.
                wafer_gamma = layout[pxstr]["gamma"]
                dprops["pol_ang_wafer"] = np.degrees(wafer_gamma - wafer_rot_rad)

                # Physical polarization orientation on wafer is always expressed
                # as 0 or 45.
                dprops["pol_orientation_wafer"] = dprops["pol_ang_wafer"] % 90

                # Polarization angle in focalplane basis.  This is, by
                # definition, equal to the gamma angle computed by the layout
                # functions.  However, we must also account for the extra rotation
                # of the center offset passed to this function.
                if center is not None:
                    dprops["quat"] = qa.mult(center, layout[pxstr]["quat"]).flatten()
                    _, _, temp_gamma = quat_to_xieta(dprops["quat"])
                    dprops["gamma"] = np.degrees(temp_gamma)
                else:
                    dprops["quat"] = layout[pxstr]["quat"].flatten()
                    dprops["gamma"] = np.degrees(wafer_gamma)
                dprops["pol_ang"] = dprops["gamma"]

                if handed is not None:
                    dprops["handed"] = handed[px]
                # Made-up assignment to readout channels
                dprops["card_slot"] = card_slot
                dprops["channel"] = doff
                dprops["AMC"] = doff // chan_per_AMC
                dprops["bias"] = doff // chan_per_bias
                dprops["readout_freq_GHz"] = readout_freq[doff]
                dprops["bondpad"] = doff - (doff // chan_per_mux) * chan_per_mux
                dprops["mux_position"] = doff // chan_per_mux

                dprops["detector_name"] = ""
                dname = "{}_p{}_{}_{}".format(wafer_slot, pstr, b, pl)
                dets[dname] = dprops
                doff += 1
        p += 1

    return dets


def load_wafer_data(filename, array_name, sub_array_name=None):
    """Load a Det_Info file (like those made by make-det-info-wafer) and return
    a specific array.

    Allows you to replace an array name with a different one (since we
    currently do not have a complete set of array hardware mappings) and
    uses the replacement name pattern to replace bandpass information. If using
    the sub_array_name it assumes array_name is an MF array

    filename(str): path to det_info hdf5 file
    array_name(str): array name to load from the file
    sub_array_name(str): work-around to change the array name while we
        only have some arrays
    """

    rs = read_dataset(filename, array_name)
    rs.keys = loader._filter_items("dets:", rs.keys)

    bp = rs.keys.index("wafer.bandpass")

    if sub_array_name is not None:
        switches = {
            "L" : {"f090" : "f030", "f150" : "f040"},
            "U" : {"f090" : "f230", "f150" : "f290"},
        }
        for i in range(len(rs)):
            row = rs.rows[i]
            rs.rows[i] = tuple([
                r.replace(array_name, sub_array_name) for r in row[:2]
            ]) + row[2:]

            if rs.rows[i][bp] == "NC" or "M" in sub_array_name:
                continue
            if "L" in sub_array_name:
                rs.rows[i] = tuple([
                    rs.rows[i][0].replace(
                        rs.rows[i][bp], switches["L"][rs.rows[i][bp]]
                    ),
                    rs.rows[i][1]
                ]) + rs.rows[i][2:bp] \
                + tuple([switches["L"][rs.rows[i][bp]]]) + rs.rows[i][bp+1:]
            elif "U" in sub_array_name:
                rs.rows[i] = tuple([
                    rs.rows[i][0].replace(
                        rs.rows[i][bp], switches["U"][rs.rows[i][bp]]
                    ),
                    rs.rows[i][1]
                ]) + rs.rows[i][2:bp] + \
                tuple([switches["U"][rs.rows[i][bp]]]) + rs.rows[i][bp + 1:]

    wafer = loader.convert_det_info(
        rs,
        rs["det_id"]

    ).wafer
    return wafer


def load_wafer_detectors(
    hw,
    wafer_slot,
    platescale,
    fwhm,
    det_info_file,
    array_name,
    sub_array_name=None,
    band=None,
    center=None,
    no_darks=False
):

    """Load detector properties for a wafer. Meant to act similarly to
    sim_wafer_detectors

    Given a Hardware configuration, load all detector properties for
    the specified wafer and optionally only the specified band.

    Args:
        hw (Hardware): The hardware properties.
        wafer_slot (str): The wafer slot name.
        platescale (float): The plate scale in degrees / mm.
        fwhm (dict): Dictionary of nominal FWHM values in arcminutes for
            each band.
        det_info_file (str): filename of detector info file
        array_name (str): array to load from file
        sub_array_name (str, optional): array to change to, defaults to
            hw.data["wafer_slots"][slot]["wafer_name"] if not set
        band (str, optional): Optionally only use this band.
        center (array, optional): The quaternion offset of the center.
        no_darks (bool, optional): if true, will not return dark detectors

    Returns:
        (OrderedDict): The properties of all selected detectors.

    """

    # The properties of this wafer
    wprops = hw.data["wafer_slots"][wafer_slot]
    tele = wprops["type"].split("_")[0]

    # The readout card and its properties
    card_slot = wprops["card_slot"]
    cardprops = hw.data["card_slots"][card_slot]

    # The bands
    bands = wprops["bands"]
    if band is not None:
        if band in bands:
            bands = [band]
        else:
            msg = f"band '{band}' not valid for wafer '{wafer_slot}'"
            raise RuntimeError(msg)

    if sub_array_name is None:
        sub_array_name = wprops["wafer_name"]
    wafer = load_wafer_data(det_info_file, array_name, sub_array_name)

    dets = OrderedDict()
    # FIXME: double check this
    # FIXME: this seems REALLY fragile.  Can we do something better?
    poloff = np.degrees(np.nanmin(wafer.angle)) - 22.5

    for i, detname in enumerate(wafer.dets.vals):
        if wafer.dets.vals[i] == 'NO_MATCH':
            continue

        dprops = OrderedDict()
        dprops["wafer_slot"] = wafer_slot
        dprops["ID"] = toast.utils.name_UID(detname)
        dprops["pixel"] = detname.split("_")[-1][:-1]
        if wafer.bandpass[i] != "NC":
            dprops["band"] = f"{tele[:3]}_{wafer.bandpass[i]}"
        else:
            dprops["band"] = "NC"

        if dprops["band"] not in bands and dprops["band"] != "NC":
            continue
        if no_darks and dprops["band"] == "NC":
            continue

        dprops["fwhm"] = fwhm[dprops["band"]] if dprops["band"] in fwhm else np.nan
        dprops["pol"] = wafer.pol[i]

        # This should be the same as the detector gamma angle relative to focalplane
        # coordinates.
        dprops["gamma"] = np.degrees(wafer.angle[i])
        dprops["pol_ang"] = dprops["gamma"]

        # Polarization angle in the wafer coordinate system
        dprops["pol_ang_wafer"] = np.round(np.degrees(wafer.angle[i]) - poloff, 2)

        # This angle seems meaningless for wafers assembled out of rhomboids
        dprops["pol_orientation_wafer"] = dprops["pol_ang_wafer"] % 90

        ## channels aren't assigned until Tunes are made, so just ints
        ## with tunes this will be 512*smurf_band + smurf_channel. not available
        ## with just hardware mapping files.
        dprops["channel"] = i
        ## card slot will be the stream_id name for the wafer slot
        dprops["card_slot"] = f"stream_id_{wafer_slot}"

        ## readout related info
        dprops["bias"] = wafer.bias_line[i]
        dprops["AMC"] = {"N" : 0, "S" : 1}[wafer.coax[i]]
        dprops["readout_freq_GHz"] = wafer.design_freq_mhz[i] / 1000
        dprops["bondpad"] = wafer.bond_pad[i]
        dprops["mux_position"] = wafer.mux_position[i]

        quat = xieta_to_quat(
            np.radians(wafer.det_x[i] * platescale),
            np.radians(wafer.det_y[i] * platescale),
            wafer.angle[i],
        )
        if center is not None:
            dprops["quat"] = qa.mult(center, quat).flatten()
            # Update gamma angle to include the center rotation
            _, _, temp_gamma = quat_to_xieta(dprops["quat"])
            dprops["gamma"] = np.degrees(temp_gamma)
            dprops["pol_ang"] = dprops["gamma"]
        else:
            dprops["quat"] = quat.flatten()
        dprops["detector_name"] = f"{wafer_slot}{detname}"
        dets[dprops["detector_name"]] = dprops

    return dets


def sim_telescope_detectors(hw, tele, tube_slots=None, det_info=None, no_darks=False):
    """Update hardware model with simulated or loaded detector positions.

    Given a Hardware model, generate all detector properties for the specified
    telescope and optionally a subset of optics tube slots (for the LAT).  The
    detector dictionary of the hardware model is updated in place.

    This uses helper functions for focalplane layout from the upstream toast
    package

    Args:
        hw (Hardware): The hardware object to update.
        tele (str): The telescope name.
        tube_slots (list, optional): The optional list of tube slots to include.
        det_info (tuple, optional): Detector info database used to load real
            array hardware.
    Returns:
        None

    """

    zaxis = np.array([0, 0, 1], dtype=np.float64)
    thirty = np.pi / 6.0
    sixty = np.pi / 3.0

    # det_info parameters
    det_info_file = None
    det_info_version = None
    if det_info is not None:
        if len(det_info) < 2:
            raise RuntimeError("Expected at least 2 elements in det_info tuple")
        det_info_file = det_info[0]
        det_info_version = det_info[1]

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
                msg = f"Invalid tube_slot '{t}' for telescope '{tele}'"
                raise RuntimeError(msg)

    alldets = OrderedDict()
    if ntube == 1:
        # This is a SAT.  We have one tube at the center.
        tubeprops = hw.data["tube_slots"][tube_slots[0]]
        waferspace = tubeprops["waferspace"] * platescale
        wafer_slot_ang_deg = tubeprops["wafer_slot_angle"]

        # Wafers are arranged in a rotated hexagon shape, however these
        # locations are rotated from a normal layout.
        wcenters = hex_layout(
            7,
            2 * waferspace * u.degree,
            "",
            "",
            u.Quantity(wafer_slot_ang_deg, u.degree),
        )
        centers = np.zeros((7, 4), dtype=np.float64)
        if det_info_file is None:
            qrot = qa.from_axisangle(zaxis, -thirty)
        else:
            qrot = qa.from_axisangle(zaxis, thirty)
            rots = [thirty, thirty, -thirty, -3*thirty, 5*thirty, 3*thirty, thirty]
        
        for p, q in wcenters.items():
            quat = q["quat"]
            if det_info_file is not None:
                # Add an additional rotation of the wafer before
                # repositioning the wafer center
                quat2 = qa.from_axisangle(zaxis, rots[int(p)])
                quat = qa.mult(quat, quat2)
            centers[int(p), :] = qa.mult(qrot, quat)

        for windx, wafer_slot in enumerate(tubeprops["wafer_slots"]):
            if det_info_file is not None:
                dets = load_wafer_detectors(
                    hw, 
                    wafer_slot, 
                    platescale, 
                    fwhm,
                    det_info_file, 
                    det_info_version, 
                    center=centers[windx],
                    no_darks=no_darks,
                )
            else:
                dets = sim_wafer_detectors(
                    hw, wafer_slot, platescale, fwhm, center=centers[windx]
                )
            alldets.update(dets)
    else:
        # This is the LAT.  We layout detectors for the case of 30
        # degree elevation and no boresight rotation.  Compute the
        # tube centers.

        # Inter-tube spacing
        tubespace = teleprops["tubespace"]

        # Hexagon layout
        tube_quats = hex_layout(
            19,
            4 * (tubespace * platescale) * u.degree,
            "",
            "",
            np.zeros(19, dtype=np.float64) * u.degree,
        )
        tcenters = np.array(
            [q["quat"] for p, q in tube_quats.items()]
        )

        for tindx, tube_slot in enumerate(tube_slots):
            tubeprops = hw.data["tube_slots"][tube_slot]
            waferspace = tubeprops["waferspace"]
            location = tubeprops["toast_hex_pos"]
            wafer_slot_ang_deg = tubeprops["wafer_slot_angle"]
            wafer_slot_ang_rad = np.radians(wafer_slot_ang_deg)

            wradius = 0.5 * (waferspace * platescale * np.pi / 180.0)
            if det_info_file is None:
                qwcenters = [
                    xieta_to_quat(
                        -wradius, wradius / np.sqrt(3.0), thirty * 4 + wafer_slot_ang_rad[0]
                    ),
                    xieta_to_quat(
                        0.0, -2.0 * wradius / np.sqrt(3.0), wafer_slot_ang_rad[1]
                    ),
                    xieta_to_quat(
                        wradius, wradius / np.sqrt(3.0), -thirty * 4 + wafer_slot_ang_rad[2]
                    ),
                ]
            else:
                qwcenters = [
                    xieta_to_quat(
                        -wradius, wradius / np.sqrt(3.0), 7*thirty + wafer_slot_ang_rad[0]
                    ),
                    xieta_to_quat(
                        0.0, -2.0 * wradius / np.sqrt(3.0), 3*thirty + wafer_slot_ang_rad[1]
                    ),
                    xieta_to_quat(
                        wradius, wradius / np.sqrt(3.0), -thirty + wafer_slot_ang_rad[2]
                    ),
                ]

            centers = list()
            for qwc in qwcenters:
                centers.append(qa.mult(tcenters[location], qwc))

            for windx, wafer_slot in enumerate(tubeprops["wafer_slots"]):
                if det_info_file is not None:
                    dets = load_wafer_detectors(
                        hw,
                        wafer_slot,
                        platescale,
                        fwhm,
                        det_info_file,
                        det_info_version,
                        center=centers[windx],
                        no_darks=no_darks,
                    )
                else:
                    dets = sim_wafer_detectors(
                        hw, wafer_slot, platescale, fwhm, center=centers[windx]
                    )
                # Accumulate to the total
                alldets.update(dets)

        # Rotate the focalplane to the nominal horizontal orientation.
        fp_rot = qa.from_axisangle(zaxis, 0.5 * np.pi)
        for d, props in alldets.items():
            props["quat"] = qa.mult(fp_rot, props["quat"])
            # Update gamma angle to include this rotation
            _, _, temp_gamma = quat_to_xieta(props["quat"])
            props["gamma"] = np.degrees(temp_gamma)
            props["pol_ang"] = props["gamma"]

    if "detectors" in hw.data:
        hw.data["detectors"].update(alldets)
    else:
        hw.data["detectors"] = alldets
