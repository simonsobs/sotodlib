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
from astropy import units as u

from ... import sim_hardware as hardware
from ...core.hardware import LAT_COROTATOR_OFFSET


XAXIS, YAXIS, ZAXIS = np.eye(3)


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
        help="Comma-separated list of wafer slots. "
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse offsets")
    # LAT specific params
    parser.add_argument(
        "--lat_corotate",
        required=False,
        default=True,
        action="store_true",
        help="Rotate LAT receiver to maintain focalplane orientation",
        dest="corotate_lat",
    )
    parser.add_argument(
        "--no_lat_corotate",
        required=False,
        default=True,
        action="store_false",
        help="Do not rotate LAT receiver to maintain focalplane orientation",
        dest="corotate_lat",
    )
    parser.add_argument(
        "--lat_elevation_deg",
        required=False,
        default=None,
        type=float,
        help="Observing elevation of the LAT if not co-rotating",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Verbose output",
        dest="verbose",
    )
    parser.set_defaults(corotate_lat=True)

    args = parser.parse_args()

    hw = hardware.sim_nominal()

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

    hardware.sim_detectors_toast(hw, telescope)

    match = {}
    tube_slots = None
    if args.wafer_slots is not None:
        match["wafer_slot"]  = args.wafer_slots.split(",")
    elif args.tube_slots is not None:
        tube_slots = args.tube_slots.split(",")

    hw = hw.select(tube_slots=tube_slots, match=match)
    ndet = len(hw.data["detectors"])

    # print(f"tube_slots = {tube_slots}, match = {match} leaves {ndet} detectors")

    # LAT rotation

    lat_rot = None
    if telescope == "LAT":
        if not args.corotate_lat:
            # We are not co-rotating, compute the rotation for the specified
            # elevation.
            if args.lat_elevation_deg is None:
                raise RuntimeError(
                    "You must set the LAT observing elevation when not co-rotating."
                )
            lat_ang = args.lat_elevation_deg - LAT_COROTATOR_OFFSET.to_value(u.deg)
            lat_rot = qa.from_axisangle(
                ZAXIS,
                np.radians(lat_ang),
            )
    else:
        if args.lat_elevation_deg is not None:
            raise RuntimeError("Observing elevation does not matter for SAT")

    # Average detector offset

    vec_mean = np.zeros(3)
    for det_name, det_data in hw.data["detectors"].items():
        quat = det_data["quat"]
        if lat_rot is not None:
            quat = qa.mult(lat_rot, det_data["quat"])
        vec = qa.rotate(quat, ZAXIS)
        vec_mean += vec
    vec_mean /= ndet

    # Radius

    all_dist = []
    for det_name, det_data in hw.data["detectors"].items():
        quat = det_data["quat"]
        if lat_rot is not None:
            quat = qa.mult(lat_rot, det_data["quat"])
        vec = qa.rotate(quat, ZAXIS)
        all_dist.append(np.degrees(np.arccos(np.dot(vec_mean, vec))))
    dist_max = np.amax(all_dist)

    # Wafers

    if args.tube_slots is None:
        wafer_slots = set(wafer_slots)
    else:
        wafer_slots = set()
        for tube_slot in tube_slots:
            wafer_slots.update(hw.data["tube_slots"][tube_slot]["wafer_slots"])
    waferstring = ""
    for wafer_slot in sorted(wafer_slots):
        waferstring += f" {wafer_slot}"

    # Translate into Az/El offsets at el=0

    rot = hp.Rotator(rot=[0, 90, 0])
    vec_mean = rot(vec_mean)
    az_offset, el_offset = hp.vec2dir(vec_mean, lonlat=True)

    el_offset *= -1
    if args.reverse:
        az_offset *= -1
        el_offset *= -1

    if args.verbose:
        print("{:8} {:8} {:8} {:8}".format("Az [deg]", "El [deg]", "Dist [deg]", "Wafer"))
        print(f"{np.degrees(az_offset):8.3f} {np.degrees(el_offset):8.3f} {dist_max:8.3f} {waferstring:8}")
    else:
        print(f"{az_offset:.3f} {el_offset:.3f} {dist_max:.3f}" + waferstring)

    return


if __name__ == "__main__":
    main()
