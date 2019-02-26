# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Detector / Hardware database creation and access.
"""

import re

from collections import OrderedDict

from copy import deepcopy

import numpy as np

from sotoddb import DetDB

import toast.qarray as qa
import toast.tod.sim_focalplane as sfp


def read_info(path):
    """Read a property file.

    This is a simple text file with space delimiters.  The first column are the
    keys of the resulting dictionary.  The remaining columns are placed in a
    dictionary with names given by the first row.  The second row contains the
    units of the properties.

    Args:
        path (str): the path to the space delimited text file.

    Returns:
        dict: dictionary of things and their properties.

    """
    def convertval(val):
        try:
            ret = int(val)
            return ret
        except:
            try:
                ret = float(val)
                return ret
            except:
                return val

    things = OrderedDict()
    prop_names = None
    prop_units = None
    with open(path, "r") as f:
        for line in f:
            if re.match(r"^#.*", line) is not None:
                continue
            if prop_names is None:
                prop_names = line.split()
            elif prop_units is None:
                prop_units = line.split()
            else:
                fields = line.split()
                key = fields[0]
                props = OrderedDict()
                for fld in range(1, len(fields)):
                    sub = fields[fld].split(',')
                    if len(sub) == 1:
                        props[prop_names[fld]] = convertval(sub[0])
                    else:
                        props[prop_names[fld]] = [ convertval(s) for s in sub ]
                    props["{}_units".format(prop_names[fld])] = prop_units[fld]
                things[key] = props
    return things


def compute_centers(offsets):
    """
    Convert cartesian angle offsets and rotation into quaternions.
    """
    centers = list()

    zaxis = np.array([0, 0, 1], dtype=np.float64)

    for off in offsets:
        angrot = qa.rotation(zaxis, off[2])
        wx = np.sin(off[0])
        wy = np.sin(off[1])
        wz = np.sqrt(1.0 - (wx*wx + wy*wy))
        wdir = np.array([wx, wy, wz])
        posrot = qa.from_vectors(zaxis, wdir)
        centers.append(qa.mult(posrot, angrot))

    return centers


def build_hexagon(rdim, angwidth, prefix, suffix, detoff,
                  center=np.array([0,0,0,1], dtype=np.float64)):
    """
    Construct a hexagon from 3 rhombi.

    Args:
        rdim (int): the number of pixels along the edge of
            one rhombus.
        angwidth (float): the angle in degrees subtended by the projected
            hexagon.
        prefix (str): prefix string for the detector names.
        suffix (str): suffix string for the detector names.
        detoff (int): detector number offset.
        center (array): 4-element quaternion rotation specifying where
            to place this hexagon.

    Returns:
        dict: dictionary of detectors and their properties.

    """
    sixty = np.pi/3.0
    thirty = np.pi/6.0
    rtthree = np.sqrt(3.0)

    # number of pixels per rhombus
    rpix = rdim**2

    # angular separation of rhombi
    margin = 0.60 * angwidth * np.pi / 180.0

    # rhombus width

    rwidth = angwidth / rtthree

    # compute the individual rhombus centers

    centers = [
        np.array([0.5*margin, 0.0, 0.0]),
        np.array([-0.5*np.cos(sixty)*margin, 0.5*np.sin(sixty)*margin,
                  2*sixty]),
        np.array([-0.5*np.cos(sixty)*margin, -0.5*np.sin(sixty)*margin,
                  4*sixty])
    ]

    qcenters = compute_centers(centers)

    # polarization orientation pattern

    apol = sfp.rhomb_pol_angles_qu(rpix, offset=0.0)
    bpol = sfp.rhomb_pol_angles_qu(rpix, offset=90.0)

    woff = 0
    dets = dict()
    detfwhm = dict()
    detcolor = dict()
    detpolcolor = dict()
    for qc in qcenters:
        # for each pixel, create 2 detectors with orthogonal orientations
        rdets = sfp.rhombus_layout(rpix, rwidth, "", "", apol, qc)

        for d, props in rdets.items():
            dprops = deepcopy(props)
            uid = (2 * int(d)) + detoff + woff
            detname = "{}{:05d}A{}".format(prefix, uid, suffix)
            dprops["quat"] = qa.mult(center, dprops["quat"])
            dets[detname] = dprops

        rdets = sfp.rhombus_layout(rpix, rwidth, "", "", bpol, qc)

        for d, props in rdets.items():
            dprops = deepcopy(props)
            uid = (2 * int(d) + 1) + detoff + woff
            detname = "{}{:05d}B{}".format(prefix, uid, suffix)
            dprops["quat"] = qa.mult(center, dprops["quat"])
            dets[detname] = dprops

        woff += 2 * len(rdets)

    return dets


def build_tube(nhex, rdim, angwidth, prefix, suffix, detoff,
               center=np.array([0,0,0,1], dtype=np.float64)):
    """
    Construct a tube from packed hexagons.

    FIXME: this should actually do different things based on whether the
        wafers are actual hexagons or 3 rhombi (i.e. different detector
        technology).

    Args:
        nhex (int): the number of hexagons in the tube (1, 3 or 7)
        rdim (int): the number of pixels along the edge of
            one rhombus.
        angwidth (float): the angle in degrees subtended by the projected
            tube.
        prefix (str): tube prefix string for the detector names.
        suffix (str): tube suffix string for the detector names.
        detoff (int): starting detector number offset.
        center (array): 4-element quaternion rotation specifying where
            to place this tube.
    """
    sixty = np.pi/3.0
    thirty = np.pi/6.0
    rtthree = np.sqrt(3.0)
    nvalid = [1, 3, 7]

    if nhex not in nvalid:
        raise RuntimeError("unsupported number of wafers")

    wafer_centers = None

    wangwidth = None
    if nhex == 3:
        wangwidth = 0.95 * 0.5 * angwidth
        # Angle from the tube center to the ring of wafer centers
        zang = 1.25 * 0.5 * wangwidth * np.pi / 180.0

        centers = [
            np.array([zang, 0.0, 0.0]),
            np.array([-np.cos(sixty)*zang, np.sin(sixty)*zang, 2*sixty]),
            np.array([-np.cos(sixty)*zang, -np.sin(sixty)*zang, 4*sixty])
        ]

        wcenters = compute_centers(centers)

        wafer_centers = { x : y for x, y in enumerate(wcenters) }
    else:
        # Use the toast hex layout function to compute the quaternion centers
        # of the wafers.
        hn = sfp.hex_nring(nhex)
        wangwidth = 0.95 * angwidth / (hn + 1.0)
        wcang = 1.02 * np.cos(thirty) * angwidth
        wcenters = sfp.hex_layout(nhex, wcang, "", "",
                                  30.0*np.ones(nhex))
        wafer_centers = { int(x) : wcenters[x]["quat"] for x in \
                         sorted(wcenters.keys()) }

    # Rotate the wafer centers to the tube center
    for h in range(nhex):
        #wafer_centers[h] = qa.mult(wafer_centers[h], center)
        wafer_centers[h] = qa.mult(center, wafer_centers[h])

    # Now create a wafer at each location with the desired properties

    tdets = dict()
    woff = 0
    for w in range(nhex):
        wprefix = "{}{:01d}-".format(prefix, w)
        wsuffix = ""
        wdets = build_hexagon(rdim, wangwidth, wprefix, wsuffix, detoff+woff,
                              center=wafer_centers[w])
        tdets.update(wdets)
        woff += len(wdets)

    return tdets


DB_TABLES = [
    ("detprops", [
        "`det_id` integer",
        "`det_name` varchar(16)",
        "`time0` integer",
        "`time1` integer",
        "`telescope` integer",
        "`tube` integer",
        "`wafer_type` character(3)",
        "`pixel_type` character(3)",
        "`band` character(3)",
        "`det_type` character(2)",
    ]),
    ("geometry", [
        "`det_id` integer",
        "`time0` integer",
        "`time1` integer",
        "`qx` double",
        "`qy` double",
        "`qz` double",
        "`qw` double",
        "`pol` double",
    ]),
]


def sim_small_aperature(pixels, bands, beams, readout, telescope, dbpath=None):
    """
    Create a nominal hardware layout for a small aperature telescope.

    This function returns a focalplane for a single tube only (centered
    on the telescope boresight).

    This assumes:
        - Each tube has a 35 degree field of view.
        - 7 hexagons per tube (each with 3 rhombi).

    Args:
        pixels (dict): the dictionary of pixel properties.
        bands (dict): the dictionary of band properties.
        beams (dict): the dictionary of beam properties.
        readout (dict): the dictionary of readout properties.
        telescope (int): the telescope number.
        dbpath (str): the database path to create.  If None, return the
            properties as a dictionary.

    Returns (dict):
        If db=None the dictionary of detector properties, else None.

    """
    tmap = {
        0 : "SLF",
        1 : "SMF",
        2 : "SMF",
        3 : "SHF"
    }
    pixtype = tmap[telescope]

    tubewidth = 35.0
    nhex = 7

    suffix = ""

    tname = pixels[pixtype]["telescope"]

    dets = None
    if dbpath is None:
        dets = dict()
    else:
        dets = DetDB(map_file=dbpath)
        for n, d in TABLES:
            db.create_table(n, d)

    waferoff = telescope * 7
    wafernpix = 3 * (pixels[pixtype]["rhombus_dim"])**2
    wafernband = len(pixels[pixtype]["bands"])
    waferdetperpix = wafernband * 2
    waferndet = wafernpix * waferdetperpix

    detoff = 0
    boff = 0
    for band in pixels[pixtype]["bands"]:
        prefix = "S{:01d}-{}-".format(telescope, band)

        # Create the detector list for this band
        bdets = build_tube(nhex, pixels[pixtype]["rhombus_dim"], tubewidth,
                          prefix, suffix, detoff)

        # Lookup the beam
        beamkey = None
        for bk, bv in beams.items():
            if (bv["band"] == band) and (bv["telescope"] == tname):
                beamkey = bk
                break
        if beamkey is None:
            raise RuntimeError("Cannot find band {} and telescope {} in beam "
                               " properties".format(band, tname))

        # Readout is made up.  We associate each wafer with a squid, and then
        # assign a "channel" to each detector based on its index on the wafer.
        # For real data, we would have to read in some other information about
        # how the wafers were wired up.

        # set properties
        bd = list(bdets.keys())
        doff = 0
        for d in bd:
            dets[d] = OrderedDict()
            # this line is needed so we can serialize
            dets[d]["quat"] = bdets[d]["quat"].tolist()
            dets[d]["telescope"] = tname
            dets[d]["pixel"] = pixtype
            dets[d]["band"] = band
            dets[d]["beam"] = beamkey
            dets[d]["squid"] = waferoff + (doff // (2 * wafernpix))
            dets[d]["channel"] = 2 * wafernpix * boff + (doff % (2 * wafernpix))
            doff += 1

        detoff += len(bd)
        boff += 1

    return dets


def sim_large_aperature(pixels, bands, beams, readout, dbpath=None):
    """
    Create a nominal hardware layout for the large aperature telescope.

    This assumes
        - 13 tubes on a single telescope, packed in truncated hexagon
        - Only the central 7 tubes will be populated initially
        - 1.2 degree FOV for a single tube
        - 1 tube of LF, 2 of HF, and 4 tubes of MF
        - 3 hexagons in each tube
        - 3 rhombi in each hexagon

    Args:
        pixels (dict): the dictionary of pixel properties.
        bands (dict): the dictionary of band properties.
        beams (dict): the dictionary of beam properties.
        readout (dict): the dictionary of readout properties.
        dbpath (str): the database path to create.  If None, return the
            properties as a dictionary.

    Returns (dict):
        If db=None the dictionary of detector properties, else None.

    """
    sixty = np.pi/3.0
    thirty = np.pi/6.0
    zaxis = np.array([0, 0, 1], dtype=np.float64)

    tubewidth = 1.2
    ntube = 7

    suffix = ""

    # Compute the tube center locations

    tcang = 3 * tubewidth
    tcenters = sfp.hex_layout(ntube, tcang, "", "", 30.0*np.ones(ntube))

    tube_centers = { int(x) : tcenters[x]["quat"] for x in \
                     sorted(tcenters.keys()) }

    # Rotate all tubes by 30 degrees
    rot = qa.rotation(zaxis, thirty)
    for t in tube_centers.keys():
        tube_centers[t] = qa.mult(tube_centers[t], rot)

    # Place tubes at each location.  We have HF in center, HF and LF on
    # opposing sides, and 2 MF on the alternate sides of the hexagon.
    # For each tube we iterate over the bands.

    tube_map = {
        0 : "LHF",
        1 : "LHF",
        2 : "LMF",
        3 : "LMF",
        4 : "LLF",
        5 : "LMF",
        6 : "LMF"
    }

    # LAT tubes have 3 wafers
    nhex = 3

    dets = dict()
    detoff = 0

    # offset by SAT wafers (7 per telescope)
    waferoff = 4 * 7

    for tb in range(len(tube_map)):
        pixtype = tube_map[tb]

        wafernpix = 3 * (pixels[pixtype]["rhombus_dim"])**2
        boff = 0

        for band in pixels[pixtype]["bands"]:
            prefix = "L{:01d}-{}-".format(tb, band)

            # Create the detector list for this band
            bdets = build_tube(nhex, pixels[pixtype]["rhombus_dim"], tubewidth,
                              prefix, suffix, detoff, center=tube_centers[tb])

            # Lookup the beam
            beamkey = None
            for bk, bv in beams.items():
                if (bv["band"] == band) and (bv["telescope"] == "LAT"):
                    beamkey = bk
                    break
            if beamkey is None:
                raise RuntimeError("Cannot find band {} and telescope LAT in "
                                   " beam properties".format(band))

            # Readout is made up.  We associate each wafer with a squid, and
            # then assign a "channel" to each detector based on its index on
            # the wafer.  For real data, we would have to read in some other
            # information about how the wafers were wired up.

            # set properties
            bd = list(bdets.keys())
            doff = 0
            for d in bd:
                dets[d] = OrderedDict()
                # this line is needed so we can serialize
                dets[d]["quat"] = bdets[d]["quat"].tolist()
                dets[d]["telescope"] = "LAT"
                dets[d]["pixel"] = pixtype
                dets[d]["band"] = band
                dets[d]["beam"] = beamkey
                dets[d]["squid"] = waferoff + (doff // (2 * wafernpix))
                dets[d]["channel"] = 2 * wafernpix * boff + (doff % (2 * wafernpix))
                doff += 1

            detoff += len(bd)
            boff += 1
        waferoff += nhex

    return dets
