#!/usr/bin/env python3

# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This workflow constructs a simulated SO telescope and writes it to
a file that is usable by toast.ops.SimGround.

"""

import argparse

from astropy import units as u

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
from toast.io.observation_hdf_save import save_instrument_file

from ..instrument import simulated_telescope


def main(opts=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic TOAST instrument file"
    )

    parser.add_argument(
        "--output", required=False, default="telescope.h5", help="Output HDF5 file"
    )
    parser.add_argument(
        "--hardware", required=False, default=None, help="Input hardware file"
    )
    parser.add_argument(
        "--det_info_file",
        required=False,
        default=None,
        help="Input detector info file for real hardware maps",
    )
    parser.add_argument(
        "--det_info_version",
        required=False,
        default=None,
        help="Detector info file version such as 'Cv4'",
    )
    parser.add_argument(
        "--thinfp",
        required=False,
        type=int,
        help="Thin the focalplane by this factor",
    )
    parser.add_argument(
        "--bands",
        required=False,
        default=None,
        help="Comma-separated list of bands: LAT_f030 (27GHz), LAT_f040 (39GHz), "
        "LAT_f090 (93GHz), LAT_f150 (145GHz), "
        "LAT_f230 (225GHz), LAT_f290 (285GHz), "
        "SAT_f030 (27GHz), SAT_f040 (39GHz), "
        "SAT_f090 (93GHz), SAT_f150 (145GHz), "
        "SAT_f230 (225GHz), SAT_f290 (285GHz). ",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--telescope",
        required=False,
        default=None,
        help="Telescope to simulate: LAT, SAT1, SAT2, SAT3, SAT4.",
    )
    group.add_argument(
        "--tube_slots",
        required=False,
        default=None,
        help="Comma-separated list of optics tube slots: c1 (LAT_UHF), i5 (LAT_UHF), "
        " i6 (LAT_MF), i1 (LAT_MF), i3 (LAT_MF), i4 (LAT_MF), o6 (LAT_LF),"
        " ST1 (SAT_MF), ST2 (SAT_MF), ST3 (SAT_UHF), ST4 (SAT_LF).",
    )
    group.add_argument(
        "--wafer_slots",
        required=False,
        default=None,
        help="Comma-separated list of wafer slots. ",
    )

    parser.add_argument(
        "--sample_rate",
        required=False,
        default=10,
        help="Sampling rate",
        type=float,
    )

    parser.add_argument(
        "--realization",
        required=False,
        default=None,
        help="Realization index",
        type=int,
    )

    args = parser.parse_args(args=opts)

    if args.hardware is None:
        if (
            args.telescope is None
            and args.wafer_slots is None
            and args.tube_slots is None
        ):
            msg = "You must specify a hardware file or one of telescope, "
            msg += "wafer_slots, or tube_slots"
            print(msg)
            return

    # Simulated telescope
    telescope = simulated_telescope(
        hwfile=args.hardware,
        det_info_file=args.det_info_file,
        det_info_version=args.det_info_version,
        telescope_name=args.telescope,
        sample_rate=args.sample_rate * u.Hz,
        bands=args.bands,
        wafer_slots=args.wafer_slots,
        tube_slots=args.tube_slots,
        thinfp=args.thinfp,
        comm=None,
    )

    save_instrument_file(f"{args.output}:/", telescope, None)


if __name__ == "__main__":
    main()
