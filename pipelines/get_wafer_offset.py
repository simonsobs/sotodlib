#!/usr/bin/env python

# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse
import os
import sys
from collections import OrderedDict

import healpy as hp
import numpy as np
import toast.qarray as qa

import sotodlib.sim_hardware as hardware


def main():
    parser = argparse.ArgumentParser(
        description="This program measures the median offset of subset of "
        "detectors from boresight.",
        usage="get_wafer_offset [options] (use --help for details)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tube_slots",
        help="Comma-separated list of optics tube slots: c1 (UHF), i5 (UHF), "
        " i6 (MF), i1 (MF), i3 (MF), i4 (MF), o6 (LF).  "
    )
    group.add_argument(
        "--wafer_slots",
        help="Comma-separated list of optics tube slots. "
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse offsets")

    args = parser.parse_args()

    hw = hardware.get_example()

    # Which telescope?

    if args.wafer_slots is not None:
        wafer_slots = args.wafer_slots.split(",")
        wafer_map = hw.wafer_map()
        tube_slots = [wafer_map["tube_slots"][ws] for ws in wafer_slots]
    else:
        tube_slots = args.tube_slots.split(",")

    telescope = None
    for tube_slot in tube_slots:
        for telescope_name, telescope_data in hw.data["telescopes"].items():
            if tube_slot in telescope_data["tube_slots"]:
                if telescope is None:
                    telescope = telescope_name
                elif telescope != telescope.name:
                    raise RuntimeError(
                        f"Tubes '{tube_slots}' span more than one telescope"
                    )
        if telescope is None:
            raise RuntimeError(
                f"Failed to match tube_slot = '{tube_slot}' with a telescope"
            )

    # Which detectors?

    hw.data["detectors"] = hardware.sim_telescope_detectors(hw, telescope)

    match = {}
    tube_slots = None
    if args.wafer_slots is not None:
        match["wafer_slot"]  = args.wafer_slots.split(",")
    elif args.tube_slots is not None:
        tube_slots = args.tube_slots.split(",")

    hw = hw.select(tube_slots=tube_slots, match=match)
    ndet = len(hw.data["detectors"])

    # print(f"tube_slots = {tube_slots}, match = {match} leaves {ndet} detectors")

    # Average detector offset

    vec_mean = np.zeros(3)
    zaxis = np.array([0, 0, 1])
    for det_name, det_data in hw.data["detectors"].items():
        quat = det_data["quat"]
        vec = qa.rotate(quat, zaxis)
        vec_mean += vec
    vec_mean /= ndet

    # Radius

    all_dist = []
    for det_name, det_data in hw.data["detectors"].items():
        quat = det_data["quat"]
        vec = qa.rotate(quat, zaxis)
        all_dist.append(np.degrees(np.arccos(np.dot(vec_mean, vec))))
    dist_max = np.amax(all_dist)

    # Translate into Az/El offsets at el=0

    rot = hp.Rotator(rot=[0, 90, 0])
    vec_mean = rot(vec_mean)
    az_offset, el_offset = hp.vec2dir(vec_mean, lonlat=True)

    el_offset *= -1
    if args.reverse:
        az_offset *= -1
        el_offset *= -1

    print(f"{az_offset:.3f} {el_offset:.3f} {dist_max:.3f}")

    return


if __name__ == "__main__":
    main()
