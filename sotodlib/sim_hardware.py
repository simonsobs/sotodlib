# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Focalplane simulation tools.
"""

import re

from collections import OrderedDict

from copy import deepcopy

import numpy as np

import quaternionarray as qa

from .core import Hardware

# FIXME:  much of this code is copy/pasted from the toast source, simply to
# avoid a dependency.  Once we can "pip install toast", we should consider
# just calling that package.


def ang_to_quat(offsets):
    """Convert cartesian angle offsets and rotation into quaternions.

    Each offset contains two angles specifying the distance from the Z axis
    in orthogonal directions (called "X" and "Y").  The third angle is the
    rotation about the Z axis.  A quaternion is computed that first rotates
    about the Z axis and then rotates this axis to the specified X/Y angle
    location.

    Args:
        offsets (list of arrays):  Each item of the list has 3 elements for
            the X / Y angle offsets in radians and the rotation in radians
            about the Z axis.

    Returns:
        (list): List of quaternions, one for each item in the input list.

    """
    out = list()

    zaxis = np.array([0, 0, 1], dtype=np.float64)

    for off in offsets:
        angrot = qa.rotation(zaxis, off[2])
        wx = np.sin(off[0])
        wy = np.sin(off[1])
        wz = np.sqrt(1.0 - (wx*wx + wy*wy))
        wdir = np.array([wx, wy, wz])
        posrot = qa.from_vectors(zaxis, wdir)
        out.append(qa.mult(posrot, angrot))

    return out


def hex_nring(npos):
    """Return the number of rings in a hexagonal layout.

    For a hexagonal layout with a given number of positions, return the
    number of rings.

    Args:
        npos (int): The number of positions.

    Returns:
        (int): The number of rings.

    """
    test = npos - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    if test != 0:
        raise RuntimeError("{} is not a valid number of positions for a "
                           "hexagonal layout".format(npos))
    return nrings


def hex_row_col(npos, pos):
    """Return the location of a given position.

    For a hexagonal layout, indexed in a "spiral" scheme (see hex_layout),
    this function returnes the "row" and "column" of a single position.
    The row is zero along the main vertex-vertex axis, and is positive
    or negative above / below this line of positions.

    Args:
        npos (int): The number of positions.
        pos (int): The position.

    Returns:
        (tuple): The (row, column) location of the position.

    """
    if pos >= npos:
        raise ValueError("position value out of range")
    test = npos - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    if pos == 0:
        row = 0
        col = nrings - 1
    else:
        test = pos - 1
        ring = 1
        while (test - 6 * ring) >= 0:
            test -= 6 * ring
            ring += 1
        sector = int(test / ring)
        steps = np.mod(test, ring)
        coloff = nrings - ring - 1
        if sector == 0:
            row = steps
            col = coloff + 2*ring - steps
        elif sector == 1:
            row = ring
            col = coloff + ring - steps
        elif sector == 2:
            row = ring - steps
            col = coloff
        elif sector == 3:
            row = -steps
            col = coloff
        elif sector == 4:
            row = -ring
            col = coloff + steps
        elif sector == 5:
            row = -ring + steps
            col = coloff + ring + steps
    return (row, col)


def hex_layout(npos, width, rotate=None):
    """Compute positions in a hexagon layout.

    Place the given number of positions in a hexagonal layout projected on
    the sphere and centered at z axis.  The width specifies the angular
    extent from vertex to vertex along the "X" axis.  For example::

        Y ^             O O O
        |              O O O O
        |             O O + O O
        +--> X         O O O O
                        O O O

    Each position is numbered 0..npos-1.  The first position is at the center,
    and then the positions are numbered moving outward in rings.

    Args:
        npos (int): The number of positions packed onto wafer.
        width (float): The angle (in degrees) subtended by the width along
            the X axis.
        rotate (array, optional): Optional array of rotation angles in degrees
            to apply to each position.

    Returns:
        (array): Array of quaternions for the positions.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
    sixty = np.pi/3.0
    thirty = np.pi/6.0
    rtthree = np.sqrt(3.0)
    rtthreebytwo = 0.5 * rtthree

    angdiameter = width * np.pi / 180.0

    # find the angular packing size of one detector
    nrings = hex_nring(npos)
    posdiam = angdiameter / (2 * nrings - 2)

    result = np.zeros((npos, 4), dtype=np.float64)

    for pos in range(npos):
        if pos == 0:
            # center position has no offset
            posrot = nullquat
        else:
            # Not at the center, find ring for this position
            test = pos - 1
            ring = 1
            while (test - 6 * ring) >= 0:
                test -= 6 * ring
                ring += 1
            sectors = int(test / ring)
            sectorsteps = np.mod(test, ring)

            # Convert angular steps around the ring into the angle and distance
            # in polar coordinates.  Each "sector" of 60 degrees is essentially
            # an equilateral triangle, and each step is equally spaced along
            # the edge opposite the vertex:
            #
            #          O
            #         O O (step 2)
            #        O   O (step 1)
            #       X O O O (step 0)
            #
            # For a given ring, "R" (center is R=0), there are R steps along
            # the sector edge.  The line from the origin to the opposite edge
            # that bisects this triangle has length R*sqrt(3)/2.  For each
            # equally-spaced step, we use the right triangle formed with this
            # bisection line to compute the angle and radius within this
            # sector.

            # The distance from the origin to the midpoint of the opposite
            # side.
            midline = rtthreebytwo * float(ring)

            # the distance along the opposite edge from the midpoint (positive
            # or negative)
            edgedist = float(sectorsteps) - 0.5 * float(ring)

            # the angle relative to the midpoint line (positive or negative)
            relang = np.arctan2(edgedist, midline)

            # total angle is based on number of sectors we have and the angle
            # within the final sector.
            posang = sectors * sixty + thirty + relang

            posdist = rtthreebytwo * posdiam * float(ring) / np.cos(relang)

            posx = np.sin(posdist) * np.cos(posang)
            posy = np.sin(posdist) * np.sin(posang)
            posz = np.cos(posdist)
            posdir = np.array([posx, posy, posz], dtype=np.float64)
            norm = np.sqrt(np.dot(posdir, posdir))
            posdir /= norm

            posrot = qa.from_vectors(zaxis, posdir)

        if rotate is None:
            result[pos] = posrot
        else:
            prerot = qa.rotation(zaxis, rotate[pos] * np.pi / 180.0)
            result[pos] = qa.mult(posrot, prerot)

    return result


def rhomb_dim(npos):
    """Compute the dimensions of a rhombus.

    For a rhombus with the specified number of positions, return the dimension
    of one side.  This function is just a check around a sqrt.

    Args:
        npos (int): The number of positions.

    Returns:
        (int): The dimension of one side.

    """
    dim = int(np.sqrt(float(npos)))
    if dim**2 != npos:
        raise ValueError("The number of positions for a rhombus must "
                         "be square")
    return dim


def rhomb_row_col(npos, pos):
    """Return the location of a given position.

    For a rhombus layout, indexed from top to bottom (see rhombus_layout),
    this function returnes the "row" and "column" of a position.  The column
    starts at zero on the left hand side of a row.

    Args:
        npos (int): The number of positions.
        pos (int): The position.

    Returns:
        (tuple): The (row, column) location of the position.

    """
    if pos >= npos:
        raise ValueError("position value out of range")
    dim = rhomb_dim(npos)
    col = pos
    rowcnt = 1
    row = 0
    while (col - rowcnt) >= 0:
        col -= rowcnt
        row += 1
        if row >= dim:
            rowcnt -= 1
        else:
            rowcnt += 1
    return (row, col)


def rhombus_layout(npos, width, rotate=None):
    """Compute positions in a hexagon layout.

    This particular rhombus geometry is essentially a third of a
    hexagon.  In other words the aspect ratio of the rhombus is
    constrained to have the long dimension be sqrt(3) times the short
    dimension.

    The rhombus is projected on the sphere and centered on the Z axis.
    The X axis is along the short direction.  The Y axis is along the longer
    direction.  For example::

                          O
        Y ^              O O
        |               O O O
        |              O O O O
        +--> X          O O O
                         O O
                          O

    Each position is numbered 0..npos-1.  The first position is at the
    "top", and then the positions are numbered moving downward and left to
    right.

    The extent of the rhombus is directly specified by the width parameter
    which is the angular extent along the X direction.

    Args:
        npos (int): The number of positions in the rhombus.
        width (float): The angle (in degrees) subtended by the width along
            the X axis.
        rotate (array, optional): Optional array of rotation angles in degrees
            to apply to each position.

    Returns:
        (array): Array of quaternions for the positions.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    rtthree = np.sqrt(3.0)

    angwidth = width * np.pi / 180.0
    dim = rhomb_dim(npos)

    # find the angular packing size of one detector
    posdiam = angwidth / (dim - 1)

    result = np.zeros((npos, 4), dtype=np.float64)

    for pos in range(npos):
        posrow, poscol = rhomb_row_col(npos, pos)

        rowang = 0.5 * rtthree * ((dim - 1) - posrow) * posdiam
        relrow = posrow
        if posrow >= dim:
            relrow = (2 * dim - 2) - posrow
        colang = (float(poscol) - float(relrow) / 2.0) * posdiam
        distang = np.sqrt(rowang**2 + colang**2)
        zang = np.cos(distang)
        posdir = np.array([colang, rowang, zang], dtype=np.float64)
        norm = np.sqrt(np.dot(posdir, posdir))
        posdir /= norm

        posrot = qa.from_vectors(zaxis, posdir)

        if rotate is None:
            result[pos] = posrot
        else:
            prerot = qa.rotation(zaxis, rotate[pos] * np.pi / 180.0)
            result[pos] = qa.mult(posrot, prerot)

    return result


def rhombus_hex_layout(rhombus_npos, rhombus_width, gap, rhombus_rotate=None,
                       killpix=None):
    """
    Construct a hexagon from 3 rhombi.

    Args:
        rhombus_npos (int): The number of positions in one rhombus.
        rhombus_width (float): The angle (in degrees) subtended by the
            width of one rhombus along the X axis.
        gap (float): The gap between the edges of the rhombi, in degrees.
        rhombus_rotate (array, optional): An additional angle rotation of
            each position on each rhombus before the rhombus is rotated
            into place.
        killpix (list, optional): Pixel indices to remove for mechanical
            reasons.

    Returns:
        (dict): Keys are the hexagon position and values are quaternions.

    """
    sixty = np.pi / 3.0
    thirty = np.pi / 6.0

    # rhombus dim
    dim = rhomb_dim(rhombus_npos)

    # width in radians
    radwidth = rhombus_width * np.pi / 180.0

    # First layout one rhombus
    rquat = rhombus_layout(rhombus_npos, rhombus_width,
                           rotate=rhombus_rotate)

    # angular separation of rhombi
    gap *= np.pi / 180.0

    # half-width of rhombus in radians
    halfwidth = 0.5 * radwidth

    # width of one pixel
    pixwidth = radwidth / (dim - 1)

    # Compute the individual rhombus centers.  This is the shift of origin
    # in the X direction for the "vertical" rhombus.
    shift = halfwidth + (0.5 * pixwidth) + ((0.5 * gap) / np.cos(thirty))

    centers = [
        np.array([shift, 0.0, 0.0]),
        np.array([-shift * np.cos(sixty), shift * np.sin(sixty),
                  2 * sixty]),
        np.array([-shift * np.cos(sixty), -shift * np.sin(sixty),
                  4 * sixty])
    ]
    qcenters = ang_to_quat(centers)

    nkill = len(killpix)
    result = np.zeros((3 * rhombus_npos - nkill, 4), dtype=np.float64)

    off = 0
    px = 0
    for qc in qcenters:
        for p in range(rhombus_npos):
            if px not in killpix:
                result[off] = qa.mult(qc, rquat[p])
                off += 1
            px += 1

    return result


def sim_wafer_detectors(hw, wafer_slot, platescale, fwhm, band=None,
                        center=np.array([0, 0, 0, 1], dtype=np.float64)):
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
            raise RuntimeError("band '{}' not valid for wafer '{}'"
                               .format(band, wafer_slot))

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
            row, col = rhomb_row_col(nrhombus, p)
            if np.mod(row, 2) == 0:
                pol_A[p] = 0.0 + poloff
            else:
                pol_A[p] = 45.0 + poloff
            pol_B[p] = 90.0 + pol_A[p]
        # We are going to remove 2 pixels for mechanical reasons
        kf = dim * (dim - 1) // 2
        kill = [(dim*dim+kf), (dim*dim+kf) + dim - 2]
        layout_A = rhombus_hex_layout(nrhombus, width, gap,
                                      rhombus_rotate=pol_A, killpix=kill)
        layout_B = rhombus_hex_layout(nrhombus, width, gap,
                                      rhombus_rotate=pol_B, killpix=kill)
    elif wprops["packing"] == "S":
        # Sinuous (Berkeley style)
        # This is the center-center distance along the vertex-vertex axis
        width = (2 * (hex_nring(npix) - 1)) * pixsep
        # The sinuous handedness is chosen so that A/B pairs of pixels have the
        # same nominal orientation but trail each other along the
        # vertex-vertex axis of the hexagon.  The polarization orientation
        # changes every other column

        if npix==37:
            pol_A=np.array([45.,0.,45.,45.,45.,45.,0.,0.,0.,45.,45.,0.,0.,0.,45.,45.,0.,0.,0.,
                            45.,0.,0.,45.,45.,0.,0.,0.,0.,0.,0.,45.,45.,0.,0.,45.,45.,45.])
            handed=["R","L","R","L","L","R","L","R","L","R","L","R","R","R","L","R","L","R","R",
                    "L","R","L","R","L","R","L","L","L","L","R","L","R","L","R","L","L","L"]
            pol_B=90.0+pol_A
        else:
            handed = list()
            pol_A = np.zeros(npix, dtype=np.float64)
            pol_B = np.zeros(npix, dtype=np.float64)
            for p in range(npix):
                row, col = hex_row_col(npix, p)
                if np.mod(col, 2) == 0:
                    handed.append("L")
                else:
                    handed.append("R")
                if np.mod(col, 4) < 2:
                    pol_A[p] = 0.0
                else:
                    pol_A[p] = 45.0
                pol_B[p] = 90.0 + pol_A[p]
        layout_A = hex_layout(npix, width, rotate=pol_A)
        layout_B = hex_layout(npix, width, rotate=pol_B)
    else:
        raise RuntimeError(
            "Unknown wafer packing '{}'".format(wprops["packing"]))

    # Now we go through each pixel and create the orthogonal detectors for
    # each band.
    dets = OrderedDict()

    #chan_per_AMC = cardprops["nchannel"] // cardprops["nAMC"]
    chan_per_AMC = 910
    chan_per_mux = 64
    chan_per_bias = cardprops["nchannel"] // cardprops["nbias"]
    readout_freq_range=np.linspace(4.,6.,chan_per_AMC)
    readout_freq=np.append(readout_freq_range,readout_freq_range)
    
    doff = 0
    p = 0
    idoff = int(wafer_slot[-2:]) * 10000
    for px in range(npix):
        if px in kill:
            continue
        pstr = "{:03d}".format(p)
        for b in bands:
            for pl, layout, pol in zip(
                    ["A", "B"], [layout_A, layout_B], [pol_A, pol_B],
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
                dprops["bondpad"] = doff-(doff//chan_per_mux)*chan_per_mux
                dprops["mux_position"] = doff//chan_per_mux
                # Layout quaternion offset is from the origin.  Now we apply
                # the rotation of the wafer center.
                dprops["quat"] = qa.mult(center, layout[p]).flatten()
                dprops["detector_name"]= ""
                dname = "{}_p{}_{}_{}".format(wafer_slot, pstr, b, pl)
                dets[dname] = dprops
                doff += 1
        p += 1

    return dets


def sim_telescope_detectors(hw, tele, tube_slots=None):
    """Generate detector properties for a telescope.

    Given a Hardware model, generate all detector properties for the specified
    telescope and optionally a subset of optics tube slots (for the LAT).

    Args:
        hw (Hardware): The hardware object to use.
        tele (str): The telescope name.
        tube_slots (list, optional): The optional list of tube slots to include.

    Returns:
        (OrderedDict): The properties of all selected detectors.

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
                raise RuntimeError("Invalid tube_slot '{}' for telescope '{}'"
                                   .format(t, tele))

    alldets = OrderedDict()
    if ntube == 1:
        # This is a SAT.  We have one tube at the center.
        tubeprops = hw.data["tube_slots"][tube_slots[0]]
        waferspace = tubeprops["waferspace"]

        shift = waferspace * platescale * np.pi / 180.0
        wcenters = [
            np.array([0.0, 0.0, 0.0]),
            np.array([shift * np.cos(thirty), shift * np.sin(thirty), 0.0]),
            np.array([0.0, shift, 0.0]),
            np.array([-shift * np.cos(thirty), shift * np.sin(thirty), 0.0]),
            np.array([-shift * np.cos(thirty), -shift * np.sin(thirty), 0.0]),
            np.array([0.0, -shift, 0.0]),
            np.array([shift * np.cos(thirty), -shift * np.sin(thirty), 0.0])
        ]
        centers = ang_to_quat(wcenters)

        windx = 0
        for wafer_slot in tubeprops["wafer_slots"]:
            dets = sim_wafer_detectors(hw, wafer_slot, platescale, fwhm,
                                       center=centers[windx])
            alldets.update(dets)
            windx += 1
    else:
        # This is the LAT.  Compute the tube centers.
        # Rotate each tube by 90 degrees, so that it is pointed "down".
        tubespace = teleprops["tubespace"]
        tuberot = 90.0 * np.ones(19, dtype=np.float64)
        tcenters = hex_layout(19, 4 * (tubespace * platescale), rotate=tuberot)

        tindx = 0
        for tube_slot in tube_slots:
            tubeprops = hw.data["tube_slots"][tube_slot]
            waferspace = tubeprops["waferspace"]
            location = tubeprops["location"]

            wradius = 0.5 * (waferspace * platescale * np.pi / 180.0)
            wcenters = [
                np.array([np.tan(thirty) * wradius, wradius, thirty*4]),
                np.array([-wradius / np.cos(thirty), 0.0, thirty*8]),
                np.array([np.tan(thirty) * wradius, -wradius, 0.0])
            ]
            qwcenters = ang_to_quat(wcenters)
            centers = list()
            for qwc in qwcenters:
                centers.append(qa.mult(tcenters[location], qwc))

            windx = 0
            for wafer_slot in tubeprops["wafer_slots"]:
                dets = sim_wafer_detectors(hw, wafer_slot, platescale, fwhm,
                                           center=centers[windx])
                alldets.update(dets)
                windx += 1
            tindx += 1
    return alldets


def get_example():
    """Return an example Hardware config with the required sections.

    The returned Hardware object has 4 fake detectors as an example.  These
    detectors can be replaced by the results of other simulation functions.

    Returns:
        (Hardware): Hardware object with example parameters.

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
    bnd["alpha"] = 3.5
    # Noise elevation scaling fits from Carlos Sierra
    # These numbers are for V3 LAT baseline
    bnd["A"] = 0.09
    bnd["C"] = 0.87
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.25
    bnd["C"] = 0.64
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.14
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.17
    bnd["C"] = 0.76
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.30
    bnd["C"] = 0.58
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.36
    bnd["C"] = 0.49
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
    bnd["alpha"] = 3.5
    # Noise elevation scaling fits from Carlos Sierra
    # These numbers are for V3 SAT baseline
    bnd["A"] = 0.10
    bnd["C"] = 0.87
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.26
    bnd["C"] = 0.66
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.17
    bnd["C"] = 0.78
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.23
    bnd["C"] = 0.71
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.43
    bnd["C"] = 0.44
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
    bnd["alpha"] = 3.5
    bnd["A"] = 0.48
    bnd["C"] = 0.38
    bnd["NET_corr"] = 1.00
    bands["SAT_f290"] = bnd

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
    ltubepos = [0, 1, 2, 3, 5, 6, 10]
    ltube_cryonames=["c1", "i5", "i6", "i1", "i3", "i4", "o6"]
    for tindx in range(7):
        nm = ltube_cryonames[tindx]
        ttyp = ltubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 128.4
        tb["wafer_slots"] = list()
        for tw in range(3):
            off = 0
            for w, props in cnf["wafer_slots"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        tb["wafer_slots"].append(w)
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["location"] = ltubepos[tindx]
        tb["tube_name"] = ""
        tb["receiver_name"] = ""
        tube_slots[nm] = tb

    stubes = ["SAT_MF", "SAT_MF", "SAT_UHF","SAT_LF"]
    for tindx in range(4):
        nm = "ST{:d}".format(tindx+1)
        ttyp = stubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 128.4
        tb["wafer_slots"] = list()
        for tw in range(7):
            off = 0
            for w, props in cnf["wafer_slots"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        tb["wafer_slots"].append(w)
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["location"] = 0
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
    fwhm_sat["SAT_f030"] = 7.4
    fwhm_sat["SAT_f040"] = 5.1
    fwhm_sat["SAT_f090"] = 2.2
    fwhm_sat["SAT_f150"] = 1.4
    fwhm_sat["SAT_f230"] = 1.0
    fwhm_sat["SAT_f290"] = 0.9
    sfwhm = OrderedDict()
    scale = 0.09668 / 0.00495
    for k, v in fwhm_sat.items():
        sfwhm[k] = float(int(scale * v * 10.0) // 10)

    tele = OrderedDict()
    tele["tube_slots"] = ["ST1"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    tele["platform_name"] = ""
    telescopes["SAT1"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST2"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    tele["platform_name"] = ""
    telescopes["SAT2"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST3"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    tele["platform_name"] = ""
    telescopes["SAT3"] = tele

    tele = OrderedDict()
    tele["tube_slots"] = ["ST4"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
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

    pl = ["A", "B"]
    hand = ["L", "R"]
    bandarr=["SAT_f030","SAT_f040"]

    dets = OrderedDict()
    for d in range(4):
        dprops = OrderedDict()
        dprops["wafer_slot"] = "w42"
        dprops["ID"] = d
        dprops["pixel"] = "000"
        bindx = d % 2
        dprops["band"] = bandarr[bindx]
        dprops["fwhm"] = 1.0
        dprops["pol"] = pl[bindx]
        dprops["handed"] = hand[bindx]
        dprops["card_slot"] = "card_slot42"
        dprops["channel"] = d
        dprops["AMC"] = 0
        dprops["bias"] = 0
        dprops["readout_freq_GHz"] = 4.
        dprops["bondpad"] = 0
        dprops["mux_position"] = 0
        dprops["quat"] = np.array([0.0, 0.0, 0.0, 1.0])
        dprops["detector_name"] = ""
        dname = "w{}_p{}_{}_{}".format("42", "000", dprops["band"],
                                     dprops["pol"])
        dets[dname] = dprops

    cnf["detectors"] = dets

    hw = Hardware()
    hw.data = cnf

    return hw
