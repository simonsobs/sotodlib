# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware configuration utilities.
"""

import os
import re
import copy

from collections import OrderedDict

import gzip

import numpy as np

import toml


def select(conf, match=dict()):
    """Select a subset of detectors.

    Given the input config dictionary, select detectors whose properties
    match some criteria.  If a matching expression is not specified for a
    given property name, then this is equivalent to selecting all values of
    that property.

    Each key of the "match" dictionary should be the name of a detector
    property to be considered for selection (e.g. band, wafer, pol, pixel).
    The value is a matching expression which can be:

        - A list of explicit values to match.
        - A string containing a regex expression to apply.

    Example:
        Imagine you wanted to select all 90GHz detectors on wafers 25 and 26
        which have "A" polarization and are located in pixels 20-29 (recall
        the "." matches a single character)::

        newconf = select(conf,
                         {"wafer": ["25", "26"],
                          "band": "MF.1",
                          "pol": "A",
                          "pixel": "02."})

    Args:
        conf (dict): The config dictionary.
        match (dict): The dictionary of property names and their matching
            expressions.

    """
    dets = conf["detectors"]
    newconf = OrderedDict()

    # Copy over auxilliary info
    aux = [
        "cards",
        "crates",
        "bands",
        "wafers",
        "tubes",
        "telescopes"
    ]
    for ax in aux:
        newconf[ax] = copy.deepcopy(conf[ax])

    # Build regex matches for each property
    reg = dict()
    for k, v in match.items():
        if isinstance(v, list):
            reg[k] = re.compile(r"("+"|".join(v)+r")")
        else:
            reg[k] = re.compile(v)

    # Go through all detectors selecting things that match all fields
    newdets = OrderedDict()
    for d, props in dets.items():
        keep = True
        for k, v in reg.items():
            if k in props:
                test = v.match(props[k])
                if test is None:
                    keep = False
                    break
        if keep:
            newdets[d] = copy.deepcopy(props)
    newconf["detectors"] = newdets
    return newconf


def dump(path, conf, overwrite=False, compress=False):
    """Write a config to a TOML file.

    Dump a config dictionary to a TOML format file, optionally compressing
    the contents with gzip and optionally overwriting the file.

    Args:
        path (str): The file to write.
        conf (dict): The config dictionary.
        overwrite (bool): If True, overwrite the file if it exists.  If False,
            then existing files will cause an exception.
        compress (bool): If True, compress the data with gzip on write.

    Returns:
        None

    """
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise RuntimeError("Dump path {} already exists.  Use "
                               "overwrite option".format(path))
    if compress:
        with gzip.open(path, "wb") as f:
            dstr = toml.dumps(conf)
            f.write(dstr.encode())
    else:
        with open(path, "w") as f:
            dstr = toml.dumps(conf)
            f.write(dstr.encode())

    return


def load(path):
    """Read a config from a TOML file.

    The file can either be regular text or a gzipped version of a TOML file.

    Args:
        path (str): The file to read.

    Returns:
        (OrderedDict): The config dictionary.

    """
    dstr = None
    conf = None
    try:
        with gzip.open(path, "rb") as f:
            dstr = f.read()
            conf = toml.loads(dstr.decode())
    except OSError:
        with open(path, "r") as f:
            dstr = f.read()
            conf = toml.loads(dstr)
    return conf


def get_example():
    """Return an example config with the required sections.

    The returned config dictionary has 4 fake detectors as an example.  These
    detectors can be replaced by the results of other simulation functions.

    Returns:
        (OrderedDict): Dictionary of hardware parameters.

    """
    cnf = OrderedDict()

    cards = OrderedDict()
    crates = OrderedDict()

    cdindx = 0
    for cr in range(8):
        crn = "{:d}".format(cr)
        crt = OrderedDict()
        crt["cards"] = list()
        for cd in range(6):
            crd = "{:02d}".format(cdindx)
            cdprops = OrderedDict()
            cdprops["nbias"] = 12
            cdprops["ncoax"] = 2
            cdprops["nchannel"] = 2000
            cards[crd] = cdprops
            crt["cards"].append(crd)
            cdindx += 1
        crates[crn] = crt

    cnf["cards"] = cards
    cnf["crates"] = crates

    bands = OrderedDict()

    bnd = OrderedDict()
    bnd["center"] = 27.0
    bnd["low"] = 25.0
    bnd["high"] = 29.0
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["LF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 39.0
    bnd["low"] = 36.5
    bnd["high"] = 41.5
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["LF2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 93.0
    bnd["low"] = 86.0
    bnd["high"] = 100.0
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["MFF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 145.0
    bnd["low"] = 134.0
    bnd["high"] = 156.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["MFF2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 93.0
    bnd["low"] = 86.0
    bnd["high"] = 100.0
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["MFS1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 145.0
    bnd["low"] = 134.0
    bnd["high"] = 156.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["MFS2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 225.0
    bnd["low"] = 208.0
    bnd["high"] = 242.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["UHF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 280.0
    bnd["low"] = 259.0
    bnd["high"] = 301.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bands["UHF2"] = bnd

    cnf["bands"] = bands

    wafers = OrderedDict()

    wtypes = ["UHF", "MFF", "MFS", "LF"]
    wcnt = {
        "LF": 1*7 + 1*3,
        "MFF": 1*7 + 2*3,
        "MFS": 1*7 + 2*3,
        "UHF": 1*7 + 2*3
    }
    wnp = {
        "LF": 37,
        "MFF": 432,
        "MFS": 397,
        "UHF": 432
    }
    wpixmm = {
        "LF": 18.0,
        "MFF": 5.3,
        "MFS": 5.6,
        "UHF": 5.3
    }
    wrhombgap = {
        "MFF": 0.71,
        "UHF": 0.71,
    }
    wbd = {
        "LF": ["LF1", "LF2"],
        "MFF": ["MFF1", "MFF2"],
        "MFS": ["MFS1", "MFS2"],
        "UHF": ["UHF1", "UHF2"]
    }
    windx = 0
    cardindx = 0
    for wt in wtypes:
        for ct in range(wcnt[wt]):
            wn = "{:02d}".format(windx)
            wf = OrderedDict()
            wf["type"] = wt
            if (wt == "LF") or (wt == "MFS"):
                wf["packing"] = "S"
            else:
                wf["packing"] = "F"
                wf["rhombusgap"] = wrhombgap[wt]
            wf["npixel"] = wnp[wt]
            wf["pixsize"] = wpixmm[wt]
            wf["bands"] = wbd[wt]
            wf["card"] = "{:02d}".format(cardindx)
            cardindx += 1
            if cardindx > 47:
                cardindx = 0
            wafers[wn] = wf
            windx += 1

    cnf["wafers"] = wafers

    tubes = OrderedDict()

    woff = {
        "LF": 0,
        "MFF": 0,
        "MFS": 0,
        "UHF": 0
    }

    ltubes = ["UHF", "UHF", "MFF", "MFF", "MFS", "MFS", "LF"]
    for tindx in range(7):
        nm = "LT{:d}".format(tindx)
        ttyp = ltubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 127.89
        tb["wafers"] = list()
        for tw in range(3):
            off = 0
            for w, props in cnf["wafers"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        tb["wafers"].append(w)
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["location"] = tindx
        tubes[nm] = tb

    stubes = ["UHF", "MFF", "MFS", "LF"]
    for tindx in range(4):
        nm = "ST{:d}".format(tindx)
        ttyp = stubes[tindx]
        tb = OrderedDict()
        tb["type"] = ttyp
        tb["waferspace"] = 127.89
        tb["wafers"] = list()
        for tw in range(7):
            off = 0
            for w, props in cnf["wafers"].items():
                if props["type"] == ttyp:
                    if off == woff[ttyp]:
                        tb["wafers"].append(w)
                        woff[ttyp] += 1
                        break
                    off += 1
        tb["location"] = 0
        tubes[nm] = tb

    cnf["tubes"] = tubes

    telescopes = OrderedDict()

    tele = OrderedDict()
    tele["tubes"] = ["LT0", "LT1", "LT2", "LT3", "LT4", "LT5", "LT6"]
    tele["platescale"] = 0.00495
    tele["tubespace"] = 450.0
    fwhm = OrderedDict()
    fwhm["LF1"] = 7.4
    fwhm["LF2"] = 5.1
    fwhm["MFF1"] = 2.2
    fwhm["MFF2"] = 1.4
    fwhm["MFS1"] = 2.2
    fwhm["MFS2"] = 1.4
    fwhm["UHF1"] = 1.0
    fwhm["UHF2"] = 0.9
    tele["fwhm"] = fwhm
    telescopes["LAT"] = tele

    sfwhm = OrderedDict()
    scale = 0.09668 / 0.00495
    for k, v in fwhm.items():
        sfwhm[k] = float(int(scale * v * 10.0) // 10)

    tele = OrderedDict()
    tele["tubes"] = ["ST0"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    telescopes["SAT0"] = tele

    tele = OrderedDict()
    tele["tubes"] = ["ST1"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    telescopes["SAT1"] = tele

    tele = OrderedDict()
    tele["tubes"] = ["ST2"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    telescopes["SAT2"] = tele

    tele = OrderedDict()
    tele["tubes"] = ["ST3"]
    tele["platescale"] = 0.09668
    tele["fwhm"] = sfwhm
    telescopes["SAT3"] = tele

    cnf["telescopes"] = telescopes

    pl = ["A", "B"]
    hand = ["L", "R"]

    dets = OrderedDict()
    for d in range(4):
        dprops = OrderedDict()
        dprops["wafer"] = "42"
        dprops["ID"] = d
        dprops["pixel"] = "000"
        bindx = d % 2
        dprops["band"] = "LF{}".format(bindx)
        dprops["fwhm"] = 1.0
        dprops["pol"] = pl[bindx]
        dprops["handed"] = hand[bindx]
        dprops["card"] = "42"
        dprops["channel"] = d
        dprops["coax"] = 0
        dprops["bias"] = 0
        dprops["quat"] = np.array([0.0, 0.0, 0.0, 1.0])
        dname = "fake_{}_{}_{}_{}".format("42", "000", dprops["band"],
                                     dprops["pol"])
        dets[dname] = dprops

    cnf["detectors"] = dets

    return cnf
