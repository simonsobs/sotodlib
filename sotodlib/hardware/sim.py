# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Focalplane simulation tools.
"""

import re

from collections import OrderedDict

from copy import deepcopy

import numpy as np

import quaternionarray as qa

from sotoddb import DetDB


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
    test = npos - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    posdiam = angdiameter / (2 * nrings - 1)

    result = np.zeros((npos, 4), dtype=np.float64)

    for pos in range(npos):
        if pos == 0:
            # center position has no offset
            result[pos] = nullquat
            continue
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
        # an equilateral triangle, and each step is equally spaced along the
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
        # bisection line to compute the angle and radius within this sector.

        # the distance from the origin to the midpoint of the opposite side.
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
    posdiam = angwidth / dim

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


def rhombus_hex_layout(rhombus_npos, rhombus_width, gap, rhombus_rotate=None):
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

    Returns:
        (dict): Keys are the hexagon position and values are quaternions.

    """
    sixty = np.pi / 3.0
    thirty = np.pi / 6.0

    # First layout one rhombus
    rquat = rhombus_layout(rhombus_npos, rhombus_width,
                           rotate=rhombus_rotate)

    # angular separation of rhombi
    gap *= np.pi / 180.0

    # compute the individual rhombus centers
    shift = (0.5 * gap) / np.cos(thirty)
    centers = [
        np.array([shift, 0.0, 0.0]),
        np.array([-shift * np.cos(sixty), shift * np.sin(sixty),
                  2 * sixty]),
        np.array([-shift * np.cos(sixty), -shift * np.sin(sixty),
                  4 * sixty])
    ]
    qcenters = ang_to_quat(centers)

    result = np.zeros((3 * rhombus_npos, 4), dtype=np.float64)

    off = 0
    for qc in qcenters:
        for p in range(rhombus_npos):
            result[off] = qa.mult(qc, rquat[p])
            off += 1

    return result


def sim_wafer_detectors(conf, wafer, platescale, band=None,
                        center=np.array([0, 0, 0, 1], dtype=np.float64)):
    """Generate detector properties for a wafer.

    Given a configuration dictionary, generate all detector properties for
    the specified wafer and optionally only the specified band.

    Args:
        conf (dict): The hardware properties, as read from a TOML config file.
        wafer (str): The wafer name.
        platescale (float): The plate scale in degrees / mm.
        band (str, optional): Optionally only use this band.
        center (array): The quaternion offset of the center.

    Returns:
        (OrderedDict): The properties of all selected detectors.

    """
    # The properties of this wafer
    wprops = conf["wafers"][wafer]
    # The readout card and its properties
    card = wprops["card"]
    cardprops = conf["cards"][card]
    # The bands
    bands = wprops["bands"]
    if band is not None:
        if band in bands:
            bands = [band]
        else:
            raise RuntimeError("band '{}' not valid for wafer '{}'"
                               .format(band, wafer))

    # Lay out the pixel locations depending on the wafer type.  Also
    # compute the polarization orientation rotation, as well as the A/B
    # handedness for the Sinuous detectors.

    npix = wprops["npixel"]
    pixsep = platescale * wprops["pixsize"]
    layout_A = None
    layout_B = None
    handed = None
    if wprops["packing"] == "F":
        # Feedhorn (NIST style)
        gap = platescale * wprops["rhombusgap"]
        nrhombus = npix // 3
        # This dim is also the number of pixels along the short axis.
        dim = rhomb_dim(nrhombus)
        # This is the center-center distance along the short axis
        width = (dim - 1) * pixsep
        # In this formalism, the "gap" includes the radius of the pixel on
        # either side of the gap.
        realgap = pixsep + gap
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
                pol_B[p] = 45.0 + poloff
            else:
                pol_A[p] = 45.0 + poloff
                pol_B[p] = 0.0 + poloff
        layout_A = rhombus_hex_layout(nrhombus, width, realgap,
                                      rhombus_rotate=pol_A)
        layout_B = rhombus_hex_layout(nrhombus, width, realgap,
                                      rhombus_rotate=pol_B)
    elif wprops["packing"] == "S":
        # Sinuous (Berkeley style)
        # This is the center-center distance along the vertex-vertex axis
        width = (2 * (hex_nring(npix) - 1)) * pixsep
        # The sinuous handedness is chosen so that A/B pairs of pixels have the
        # same nominal orientation but trail each other along the
        # vertex-vertex axis of the hexagon.  The polarization orientation
        # changes every other column
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
                pol_B[p] = 45.0
            else:
                pol_A[p] = 45.0
                pol_B[p] = 0.0
        layout_A = hex_layout(npix, width, rotate=pol_A)
        layout_B = hex_layout(npix, width, rotate=pol_B)
    else:
        raise RuntimeError(
            "Unknown wafer packing '{}'".format(wprops["packing"]))

    # Now we go through each pixel and create the orthogonal detectors for
    # each band.
    dets = OrderedDict()

    chan_per_coax = cardprops["nchannel"] // cardprops["ncoax"]
    chan_per_bias = cardprops["nchannel"] // cardprops["nbias"]

    doff = 0
    for p in range(npix):
        pstr = "{:03d}".format(p)
        for b in bands:
            for pl, layout in zip(["A", "B"], [layout_A, layout_B]):
                dprops = OrderedDict()
                dprops["wafer"] = wafer
                dprops["pixel"] = p
                dprops["band"] = b
                dprops["pol"] = pl
                if handed is not None:
                    dprops["handed"] = handed[p]
                # Made-up assignment to readout channels
                dprops["card"] = card
                dprops["channel"] = doff
                dprops["coax"] = doff // chan_per_coax
                dprops["bias"] = doff // chan_per_bias
                # Layout quaternion offset is from the origin.  Now we apply
                # the rotation of the wafer center.
                dprops["quat"] = qa.mult(center, layout[p]).flatten()
                dname = "{}_{}_{}_{}".format(wafer, pstr, b, pl)
                dets[dname] = dprops
                doff += 1

    return dets


#
#
#
# def sim_telescope_detectors(conf, tele, tubes=None):
#     """Generate detector properties for a telescope.
#
#     Given a configuration dictionary, generate all detector properties for
#     the specified telescope and optionally a subset of optics tubes (for the
#     LAT).
#
#     Args:
#         conf (dict): The hardware properties, as read from a TOML config file.
#         tele (str): The telescope name.
#         tubes (list, optional): The optional list of tubes to include.
#
#     Returns:
#         (OrderedDict): The properties of all selected detectors.
#
#     """
#
#
#
#
#
# def create_tube(conf, tube, platescale,
#                 center=np.array([0,0,0,1], dtype=np.float64)):
#     """
#     Construct a tube from packed hexagons.
#
#     Given the configuration dictionary with hardware properties, layout a
#     tube of detectors
#
#     FIXME: this should actually do different things based on whether the
#         wafers are actual hexagons or 3 rhombi (i.e. different detector
#         technology).
#
#     Args:
#         nhex (int): the number of hexagons in the tube (1, 3 or 7)
#         rdim (int): the number of pixels along the edge of
#             one rhombus.
#         angwidth (float): the angle in degrees subtended by the projected
#             tube.
#         prefix (str): tube prefix string for the detector names.
#         suffix (str): tube suffix string for the detector names.
#         detoff (int): starting detector number offset.
#         center (array): 4-element quaternion rotation specifying where
#             to place this tube.
#     """
#     sixty = np.pi/3.0
#     thirty = np.pi/6.0
#     rtthree = np.sqrt(3.0)
#     nvalid = [1, 3, 7]
#
#     if nhex not in nvalid:
#         raise RuntimeError("unsupported number of wafers")
#
#     wafer_centers = None
#
#     wangwidth = None
#     if nhex == 3:
#         wangwidth = 0.95 * 0.5 * angwidth
#         # Angle from the tube center to the ring of wafer centers
#         zang = 1.25 * 0.5 * wangwidth * np.pi / 180.0
#
#         centers = [
#             np.array([zang, 0.0, 0.0]),
#             np.array([-np.cos(sixty)*zang, np.sin(sixty)*zang, 2*sixty]),
#             np.array([-np.cos(sixty)*zang, -np.sin(sixty)*zang, 4*sixty])
#         ]
#
#         wcenters = compute_centers(centers)
#
#         wafer_centers = { x : y for x, y in enumerate(wcenters) }
#     else:
#         # Use the toast hex layout function to compute the quaternion centers
#         # of the wafers.
#         hn = sfp.hex_nring(nhex)
#         wangwidth = 0.95 * angwidth / (hn + 1.0)
#         wcang = 1.02 * np.cos(thirty) * angwidth
#         wcenters = sfp.hex_layout(nhex, wcang, "", "",
#                                   30.0*np.ones(nhex))
#         wafer_centers = { int(x) : wcenters[x]["quat"] for x in \
#                          sorted(wcenters.keys()) }
#
#     # Rotate the wafer centers to the tube center
#     for h in range(nhex):
#         #wafer_centers[h] = qa.mult(wafer_centers[h], center)
#         wafer_centers[h] = qa.mult(center, wafer_centers[h])
#
#     # Now create a wafer at each location with the desired properties
#
#     tdets = dict()
#     woff = 0
#     for w in range(nhex):
#         wprefix = "{}{:01d}-".format(prefix, w)
#         wsuffix = ""
#         wdets = build_hexagon(rdim, wangwidth, wprefix, wsuffix, detoff+woff,
#                               center=wafer_centers[w])
#         tdets.update(wdets)
#         woff += len(wdets)
#
#     return tdets
#
#
# DB_TABLES = [
#     ("detprops", [
#         "`det_id` integer",
#         "`det_name` varchar(16)",
#         "`time0` integer",
#         "`time1` integer",
#         "`telescope` integer",
#         "`tube` integer",
#         "`wafer_type` character(3)",
#         "`pixel_type` character(3)",
#         "`band` character(3)",
#         "`det_type` character(2)",
#     ]),
#     ("geometry", [
#         "`det_id` integer",
#         "`time0` integer",
#         "`time1` integer",
#         "`qx` double",
#         "`qy` double",
#         "`qz` double",
#         "`qw` double",
#         "`pol` double",
#     ]),
# ]
#
#
# def sim_small_aperature(pixels, bands, beams, readout, telescope, dbpath=None):
#     """
#     Create a nominal hardware layout for a small aperature telescope.
#
#     This function returns a focalplane for a single tube only (centered
#     on the telescope boresight).
#
#     This assumes:
#         - Each tube has a 35 degree field of view.
#         - 7 hexagons per tube (each with 3 rhombi).
#
#     Args:
#         pixels (dict): the dictionary of pixel properties.
#         bands (dict): the dictionary of band properties.
#         beams (dict): the dictionary of beam properties.
#         readout (dict): the dictionary of readout properties.
#         telescope (int): the telescope number.
#         dbpath (str): the database path to create.  If None, return the
#             properties as a dictionary.
#
#     Returns (dict):
#         If db=None the dictionary of detector properties, else None.
#
#     """
#     tmap = {
#         0 : "SLF",
#         1 : "SMF",
#         2 : "SMF",
#         3 : "SHF"
#     }
#     pixtype = tmap[telescope]
#
#     tubewidth = 35.0
#     nhex = 7
#
#     suffix = ""
#
#     tname = pixels[pixtype]["telescope"]
#
#     dets = None
#     if dbpath is None:
#         dets = dict()
#     else:
#         dets = DetDB(map_file=dbpath)
#         for n, d in TABLES:
#             db.create_table(n, d)
#
#     waferoff = telescope * 7
#     wafernpix = 3 * (pixels[pixtype]["rhombus_dim"])**2
#     wafernband = len(pixels[pixtype]["bands"])
#     waferdetperpix = wafernband * 2
#     waferndet = wafernpix * waferdetperpix
#
#     detoff = 0
#     boff = 0
#     for band in pixels[pixtype]["bands"]:
#         prefix = "S{:01d}-{}-".format(telescope, band)
#
#         # Create the detector list for this band
#         bdets = build_tube(nhex, pixels[pixtype]["rhombus_dim"], tubewidth,
#                           prefix, suffix, detoff)
#
#         # Lookup the beam
#         beamkey = None
#         for bk, bv in beams.items():
#             if (bv["band"] == band) and (bv["telescope"] == tname):
#                 beamkey = bk
#                 break
#         if beamkey is None:
#             raise RuntimeError("Cannot find band {} and telescope {} in beam "
#                                " properties".format(band, tname))
#
#         # Readout is made up.  We associate each wafer with a squid, and then
#         # assign a "channel" to each detector based on its index on the wafer.
#         # For real data, we would have to read in some other information about
#         # how the wafers were wired up.
#
#         # set properties
#         bd = list(bdets.keys())
#         doff = 0
#         for d in bd:
#             dets[d] = OrderedDict()
#             # this line is needed so we can serialize
#             dets[d]["quat"] = bdets[d]["quat"].tolist()
#             dets[d]["telescope"] = tname
#             dets[d]["pixel"] = pixtype
#             dets[d]["band"] = band
#             dets[d]["beam"] = beamkey
#             dets[d]["squid"] = waferoff + (doff // (2 * wafernpix))
#             dets[d]["channel"] = 2 * wafernpix * boff + (doff % (2 * wafernpix))
#             doff += 1
#
#         detoff += len(bd)
#         boff += 1
#
#     return dets
#
#
# def sim_large_aperature(pixels, bands, beams, readout, dbpath=None):
#     """
#     Create a nominal hardware layout for the large aperature telescope.
#
#     This assumes
#         - 13 tubes on a single telescope, packed in truncated hexagon
#         - Only the central 7 tubes will be populated initially
#         - 1.2 degree FOV for a single tube
#         - 1 tube of LF, 2 of HF, and 4 tubes of MF
#         - 3 hexagons in each tube
#         - 3 rhombi in each hexagon
#
#     Args:
#         pixels (dict): the dictionary of pixel properties.
#         bands (dict): the dictionary of band properties.
#         beams (dict): the dictionary of beam properties.
#         readout (dict): the dictionary of readout properties.
#         dbpath (str): the database path to create.  If None, return the
#             properties as a dictionary.
#
#     Returns (dict):
#         If db=None the dictionary of detector properties, else None.
#
#     """
#     sixty = np.pi/3.0
#     thirty = np.pi/6.0
#     zaxis = np.array([0, 0, 1], dtype=np.float64)
#
#     tubewidth = 1.2
#     ntube = 7
#
#     suffix = ""
#
#     # Compute the tube center locations
#
#     tcang = 3 * tubewidth
#     tcenters = sfp.hex_layout(ntube, tcang, "", "", 30.0*np.ones(ntube))
#
#     tube_centers = { int(x) : tcenters[x]["quat"] for x in \
#                      sorted(tcenters.keys()) }
#
#     # Rotate all tubes by 30 degrees
#     rot = qa.rotation(zaxis, thirty)
#     for t in tube_centers.keys():
#         tube_centers[t] = qa.mult(tube_centers[t], rot)
#
#     # Place tubes at each location.  We have HF in center, HF and LF on
#     # opposing sides, and 2 MF on the alternate sides of the hexagon.
#     # For each tube we iterate over the bands.
#
#     tube_map = {
#         0 : "LHF",
#         1 : "LHF",
#         2 : "LMF",
#         3 : "LMF",
#         4 : "LLF",
#         5 : "LMF",
#         6 : "LMF"
#     }
#
#     # LAT tubes have 3 wafers
#     nhex = 3
#
#     dets = dict()
#     detoff = 0
#
#     # offset by SAT wafers (7 per telescope)
#     waferoff = 4 * 7
#
#     for tb in range(len(tube_map)):
#         pixtype = tube_map[tb]
#
#         wafernpix = 3 * (pixels[pixtype]["rhombus_dim"])**2
#         boff = 0
#
#         for band in pixels[pixtype]["bands"]:
#             prefix = "L{:01d}-{}-".format(tb, band)
#
#             # Create the detector list for this band
#             bdets = build_tube(nhex, pixels[pixtype]["rhombus_dim"], tubewidth,
#                               prefix, suffix, detoff, center=tube_centers[tb])
#
#             # Lookup the beam
#             beamkey = None
#             for bk, bv in beams.items():
#                 if (bv["band"] == band) and (bv["telescope"] == "LAT"):
#                     beamkey = bk
#                     break
#             if beamkey is None:
#                 raise RuntimeError("Cannot find band {} and telescope LAT in "
#                                    " beam properties".format(band))
#
#             # Readout is made up.  We associate each wafer with a squid, and
#             # then assign a "channel" to each detector based on its index on
#             # the wafer.  For real data, we would have to read in some other
#             # information about how the wafers were wired up.
#
#             # set properties
#             bd = list(bdets.keys())
#             doff = 0
#             for d in bd:
#                 dets[d] = OrderedDict()
#                 # this line is needed so we can serialize
#                 dets[d]["quat"] = bdets[d]["quat"].tolist()
#                 dets[d]["telescope"] = "LAT"
#                 dets[d]["pixel"] = pixtype
#                 dets[d]["band"] = band
#                 dets[d]["beam"] = beamkey
#                 dets[d]["squid"] = waferoff + (doff // (2 * wafernpix))
#                 dets[d]["channel"] = 2 * wafernpix * boff + (doff % (2 * wafernpix))
#                 doff += 1
#
#             detoff += len(bd)
#             boff += 1
#         waferoff += nhex
#
#     return dets
