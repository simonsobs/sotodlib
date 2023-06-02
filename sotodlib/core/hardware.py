# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware configuration utilities.
"""

import os
import re
import copy

from collections import OrderedDict

import gzip

import astropy.units as u
import numpy as np

import toml


# The extra rotation to apply to the projected LAT focalplane on the
# sky is given by:
#
#      R = Elevation - Offset - Corotator
#
# and the offset is defined as the "design elevation" which places
# the focalplane at the correct orientation with no co-rotation.

LAT_COROTATOR_OFFSET = u.Quantity(60.0, u.degree)


def build_readout_id(creation_time, wafer_slot, channel):
    """Construct a simulated readout_id.

    We build this string from the stream_id (wafer slot), the overall
    creation time (treating this as the last tuning time), and the
    readout channel.  We convert the readout channel into a smurf band
    and smurf channel assuming 8 bands and 512 channels per band.

    Args:
        creation_time (float):  The POSIX ctime of the simulation start.
        wafer_slot (str):  The stream_id / wafer_slot
        channel (int):  The readout channel (0-4095).

    Returns:
        (str):  The fake readout_id.

    """
    creation_time = int(creation_time)
    smurf_band = channel // 8
    smurf_channel = channel % 8
    return f"{wafer_slot}_{creation_time:10d}_{smurf_band}_{smurf_channel}"

def parse_readout_id(readout_id):
    """Split a readout_id into its parts.

    Args:
        creation_time (float):  The POSIX ctime of the simulation start.
        wafer_slot (str):  The stream_id / wafer_slot
        channel (int):  The readout channel (0-4095).

    Returns:
        (tuple):  The wafer_slot, creation time, readout channel

    """
    pat = re.compile(r"(.*)_(.*)_(.*)_(.*)")
    mat = pat.match(readout_id)
    if mat is None:
        raise ValueError(f"Readout ID {readout_id} is invalid")
    wf = mat.group(1)
    ct = float(mat.group(2))
    smband = int(mat.group(3))
    smchan = int(mat.group(4))
    channel = smband * 8 + smchan
    return (wf, ct, channel)

def sim_wafer_names( hw ):
    """Adds SO generic UFM names to the hardware model based on the type of the wafer
       Ex: Uv1, Mv4, Lv3, etc
    """
    c = [1,1,1]
    for wafer in hw.data["wafer_slots"]:
        wprops = hw.data["wafer_slots"][wafer]
        if "UHF" in wprops["type"]:
            pre = "Uv"
            i = 0
        elif "MF" in wprops["type"]:
            pre = "Mv"
            i = 1
        elif "LF" in wprops["type"]:
            pre = "Lv"
            i = 2
        else:
            raise ValueError(f"Unknown band type {wprops['type']} for wafer {wafer}")

        wprops["wafer_name"] = f"{pre}{c[i]}"
        c[i] += 1


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

        Given the current data state, build dictionaries to go from wafer_slots
        to all other non-detector info:  telescopes, tube_slots, card_slots, crate_slots,
        and bands.  This is a convenient mapping when pruning the hardware
        information or doing other kinds of lookups.

        Returns:
            (dict): Nested dictionaries from wafers to other properties.

        """
        result = OrderedDict()

        tube_to_tele = dict()
        for tele, props in self.data["telescopes"].items():
            for tb in props["tube_slots"]:
                tube_to_tele[tb] = tele

        wafer_to_tube = dict()
        for tb, props in self.data["tube_slots"].items():
            for wf in props["wafer_slots"]:
                wafer_to_tube[wf] = tb

        crate_to_card = dict()
        for crate, props in self.data["crate_slots"].items():
            for card in props["card_slots"]:
                crate_to_card[card] = crate

        result["card_slots"] = {x: y["card_slot"]
                           for x, y in self.data["wafer_slots"].items()}
        result["crate_slots"] = {x: crate_to_card[y["card_slot"]]
                            for x, y in self.data["wafer_slots"].items()}
        result["bands"] = {x: y["bands"]
                           for x, y in self.data["wafer_slots"].items()}
        result["tube_slots"] = wafer_to_tube
        result["telescopes"] = {x: tube_to_tele[wafer_to_tube[x]] for x in
                                list(self.data["wafer_slots"].keys())}
        return result

    def select(self, telescopes=None, tube_slots=None, match=dict()):
        """Select a subset of detectors.

        Select detectors whose properties match some criteria.  A new Hardware
        object is created and returned.  If a matching expression is not
        specified for a given property name, then this is equivalent to
        selecting all values of that property.

        Before selecting on detector properties, any telescope / tube_slot filtering
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

                new = hw.select(match={"wafer_slot": ["w25", "w26"],
                                "band": "SAT_f090",
                                "pol": "A",
                                "pixel": "02."})

        Args:
            telescopes (str): A regex string to apply to telescope names or a
                list of explicit names.
            tube_slots (str): A regex string to apply to tube_slot names or a list of
                explicit names.
            match (dict): The dictionary of property names and their matching
                expressions.

        Returns:
            (Hardware): A new Hardware instance with the selected detectors.

        """
        # First parse any telescope and tube_slot options into a list of wafers
        wselect = None
        tbselect = None
        if telescopes is not None:
            tbselect = list()
            for tele in telescopes:
                tbselect.extend(self.data["telescopes"][tele]["tube_slots"])
        if tube_slots is not None:
            if tbselect is None:
                tbselect = list()
            tbselect.extend(tube_slots)
        if tbselect is not None:
            wselect = list()
            for tb in tbselect:
                wselect.extend(self.data["tube_slots"][tb]["wafer_slots"])

        dets = self.data["detectors"]

        # Build regex matches for each property
        reg = dict()
        if "wafer_slot" in match:
            # Handle wafer case separately, since we need to merge any
            # match with our telescope / tube_slot selection of wafers above.
            k = "wafer_slot"
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
            reg["wafer_slot"] = re.compile(r"("+"|".join(wselect)+r")")

        for k, v in match.items():
            if (k == "wafer_slot"):
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
                newwafers.add(props["wafer_slot"])
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
        hw.data["wafer_slots"] = OrderedDict()
        for wf in newwafers:
            hw.data["wafer_slots"][wf] = copy.deepcopy(self.data["wafer_slots"][wf])

        # And the detectors...
        hw.data["detectors"] = newdets

        return hw
