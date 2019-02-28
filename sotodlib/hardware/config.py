# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Configuration file utilities.
"""

from collections import OrderedDict

import numpy as np

import toml


def get_example():
    """Return an example config with the required sections.

    The purpose of these configuration files is to generate a mock hardware
    model for simulations.  This model can then be used to populate databases
    that look and feel like the real databases that will be in use.

    Once real hardware exists and its properties are measured and known, then
    the databases can be updated directly by other software.

    Returns:
        (OrderedDict): Dictionary of all parameters.

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
    tele["tubespace"] = 600.0
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

    return cnf
