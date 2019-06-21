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


class Hardware(object):
    """Class representing a specific hardware configuration.

    The data is stored in a dictionary, and can be loaded / dumped to disk
    as well as trimmed to include only a subset of detectors.

    Args:
        path (str, optional): If specified, configuration is loaded from this
            file during construction.

    """
    def __init__(self, path=None):
        self.data = OrderedDict()
        if path is not None:
            self.load(path)

    def dump(self, path, overwrite=False, compress=False):
        """Write hardware config to a TOML file.

        Dump data to a TOML format file, optionally compressing the contents
        with gzip and optionally overwriting the file.

        Args:
            path (str): The file to write.
            overwrite (bool): If True, overwrite the file if it exists.
                If False, then existing files will cause an exception.
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
                dstr = toml.dumps(self.data)
                f.write(dstr.encode())
        else:
            with open(path, "w") as f:
                dstr = toml.dumps(self.data)
                f.write(dstr)
        return

    def load(self, path):
        """Read data from a TOML file.

        The file can either be regular text or a gzipped version of a TOML
        file.

        Args:
            path (str): The file to read.

        Returns:
            None

        """
        dstr = None
        try:
            with gzip.open(path, "rb") as f:
                dstr = f.read()
                self.data = toml.loads(dstr.decode())
        except OSError:
            with open(path, "r") as f:
                dstr = f.read()
                self.data = toml.loads(dstr)
        return

    def wafer_map(self):
        """Construct wafer mapping to other auxilliary data.

        Given the current data state, build dictionaries to go from wafers
        to all other non-detector info:  telescopes, tubes, cards, crates,
        and bands.  This is a convenient mapping when pruning the hardware
        information or doing other kinds of lookups.

        Returns:
            (dict): Nested dictionaries from wafers to other properties.

        """
        result = OrderedDict()

        tube_to_tele = dict()
        for tele, props in self.data["telescopes"].items():
            for tb in props["tubes"]:
                tube_to_tele[tb] = tele

        wafer_to_tube = dict()
        for tb, props in self.data["tubes"].items():
            for wf in props["wafers"]:
                wafer_to_tube[wf] = tb

        crate_to_card = dict()
        for crate, props in self.data["crates"].items():
            for card in props["cards"]:
                crate_to_card[card] = crate

        result["cards"] = {x: y["card"]
                           for x, y in self.data["wafers"].items()}
        result["crates"] = {x: crate_to_card[y["card"]]
                            for x, y in self.data["wafers"].items()}
        result["bands"] = {x: y["bands"]
                           for x, y in self.data["wafers"].items()}
        result["tubes"] = wafer_to_tube
        result["telescopes"] = {x: tube_to_tele[wafer_to_tube[x]] for x in
                                list(self.data["wafers"].keys())}
        return result

    def select(self, telescopes=None, tubes=None, match=dict()):
        """Select a subset of detectors.

        Select detectors whose properties match some criteria.  A new Hardware
        object is created and returned.  If a matching expression is not
        specified for a given property name, then this is equivalent to
        selecting all values of that property.

        Before selecting on detector properties, any telescope / tube filtering
        criteria are first applied.

        Each key of the "match" dictionary should be the name of a detector
        property to be considered for selection (e.g. band, wafer, pol, pixel).
        The value is a matching expression which can be:

            - A list of explicit values to match.
            - A string containing a regex expression to apply.

        Example:
            Imagine you wanted to select all 90GHz detectors on wafers 25 and
            26 which have "A" polarization and are located in pixels 20-29
            (recall the "." matches a single character)::

                new = hw.select(match={"wafer": ["25", "26"],
                                "band": "MF.1",
                                "pol": "A",
                                "pixel": "02."})

        Args:
            telescopes (str): A regex string to apply to telescope names or a
                list of explicit names.
            tubes (str): A regex string to apply to tube names or a list of
                explicit names.
            match (dict): The dictionary of property names and their matching
                expressions.

        Returns:
            (Hardware): A new Hardware instance with the selected detectors.

        """
        # First parse any telescope and tube options into a list of wafers
        wselect = None
        tbselect = None
        if telescopes is not None:
            tbselect = list()
            for tele in telescopes:
                tbselect.extend(self.data["telescopes"][tele]["tubes"])
        if tubes is not None:
            if tbselect is None:
                tbselect = list()
            tbselect.extend(tubes)
        if tbselect is not None:
            wselect = list()
            for tb in tbselect:
                wselect.extend(self.data["tubes"][tb]["wafers"])

        dets = self.data["detectors"]

        # Build regex matches for each property
        reg = dict()
        if "wafer" in match:
            # Handle wafer case separately, since we need to merge any
            # match with our telescope / tube selection of wafers above.
            k = "wafer"
            v = match[k]
            if wselect is None:
                # Just the regular behavior
                if isinstance(v, list):
                    reg[k] = re.compile(r"("+"|".join(v)+r")")
                else:
                    reg[k] = re.compile(v)
            else:
                # Merge our selection
                wall = list(wselect)
                if isinstance(v, list):
                    wall.extend(v)
                else:
                    wall.append(v)
                reg[k] = re.compile(r"("+"|".join(wall)+r")")
        elif wselect is not None:
            # No pattern in the match dictionary, just our list from the
            # telescope / tube selection.
            reg["wafer"] = re.compile(r"("+"|".join(wselect)+r")")

        for k, v in match.items():
            if (k == "wafer"):
                # Already handled above
                continue
            else:
                if isinstance(v, list):
                    reg[k] = re.compile(r"("+"|".join(v)+r")")
                else:
                    reg[k] = re.compile(v)

        # Go through all detectors selecting things that match all fields
        newwafers = set()
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
                newwafers.add(props["wafer"])
                newdets[d] = copy.deepcopy(props)

        # Now compute the reduced set of auxilliary data needed for these
        # detectors.
        wafermap = self.wafer_map()

        # Copy this data
        hw = Hardware()
        hw.data = OrderedDict()
        for k, v in wafermap.items():
            hw.data[k] = OrderedDict()
            tocopy = set()
            for wf in newwafers:
                if isinstance(v[wf], list):
                    for iv in v[wf]:
                        tocopy.add(iv)
                else:
                    tocopy.add(v[wf])
            for elem in tocopy:
                hw.data[k][elem] = copy.deepcopy(self.data[k][elem])

        # Copy over the wafer data
        hw.data["wafers"] = OrderedDict()
        for wf in newwafers:
            hw.data["wafers"][wf] = copy.deepcopy(self.data["wafers"][wf])

        # And the detectors...
        hw.data["detectors"] = newdets

        return hw


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
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    # Noise elevation scaling fits from Carlos Sierra
    # These numbers are for V3 LAT baseline
    bnd["A"] = 0.09
    bnd["C"] = 0.87
    bands["LF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 38.9
    bnd["low"] = 30.9
    bnd["high"] = 46.9
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.25
    bnd["C"] = 0.64
    bands["LF2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 92.0
    bnd["low"] = 79.0
    bnd["high"] = 105.0
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.14
    bnd["C"] = 0.80
    bands["MFF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 147.5
    bnd["low"] = 130.0
    bnd["high"] = 165.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.17
    bnd["C"] = 0.76
    bands["MFF2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 88.6
    bnd["low"] = 75.6
    bnd["high"] = 101.6
    bnd["bandpass"] = ""
    bnd["NET"] = 300.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.19
    bnd["C"] = 0.74
    bands["MFS1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 146.5
    bnd["low"] = 128.0
    bnd["high"] = 165.0
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.19
    bnd["C"] = 0.73
    bands["MFS2"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 225.7
    bnd["low"] = 196.7
    bnd["high"] = 254.7
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.30
    bnd["C"] = 0.58
    bands["UHF1"] = bnd

    bnd = OrderedDict()
    bnd["center"] = 285.4
    bnd["low"] = 258.4
    bnd["high"] = 312.4
    bnd["bandpass"] = ""
    bnd["NET"] = 400.0
    bnd["fknee"] = 50.0
    bnd["fmin"] = 0.01
    bnd["alpha"] = 3.5
    bnd["A"] = 0.36
    bnd["C"] = 0.49
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
    ltubepos = [0, 1, 2, 3, 5, 6, 10]
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
        tb["location"] = ltubepos[tindx]
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
    # This tube spacing in mm corresponds to 1.78 degrees projected on
    # the sky at a plate scale of 0.00495 deg/mm.
    tele["tubespace"] = 359.6
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
    
    cards = OrderedDict()
    crates = OrderedDict()
    
    crt_indx = 0
    
    for tel in cnf["telescopes"]:
        crn = "{:d}".format(crt_indx)
        crt = OrderedDict()
        crt["cards"] = list()
        crt["telescope"] = tel
    
        ## get all the wafer card numbers for a telescope
        tb_wfrs = [cnf["tubes"][t]["wafers"] for t in cnf["telescopes"][tel]["tubes"]]
        tl_wfrs = [i for sl in tb_wfrs for i in sl]
        wafer_cards = [cnf["wafers"][w]["card"] for w in tl_wfrs]
        
        # add all cards to the card table and assign to crates
        for crd in wafer_cards:
            cdprops = OrderedDict()
            cdprops["nbias"] = 12
            cdprops["ncoax"] = 2
            cdprops["nchannel"] = 2000
            cards[crd] = cdprops
            
            crt["cards"].append(crd)
            
            # name new crates when current one is full
            if ('S' in tel and len(crt["cards"]) >=4) or len(crt["cards"]) >=6:
                crates[crn] = crt
                crt_indx += 1
                crn = "{:d}".format(crt_indx)
                crt = OrderedDict()
                crt["cards"] = list()
                crt["telescope"] = tel
        
        # each telescope starts with a new crate
        crates[crn] = crt
        crt_indx += 1

    cnf["cards"] = cards
    cnf["crates"] = crates

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
        dname = "{}_{}_{}_{}".format("42", "000", dprops["band"],
                                     dprops["pol"])
        dets[dname] = dprops

    cnf["detectors"] = dets

    hw = Hardware()
    hw.data = cnf

    return hw
