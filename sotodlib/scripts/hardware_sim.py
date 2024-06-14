# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulate the nominal hardware model.
"""

import sys
import argparse

from collections import OrderedDict

from ..sim_hardware import sim_nominal, sim_detectors_toast


def main():
    parser = argparse.ArgumentParser(
        description="This program simulates the current nominal hardware\
            model and dumps it to disk",
        usage="so_hardware_sim [options] (use --help for details)")

    parser.add_argument(
        "--out", required=False, default="hardware",
        help="Name (without extensions) of the output hardware file"
    )

    parser.add_argument(
        "--plain", required=False, default=False, action="store_true",
        help="Write plain text (without gzip compression)"
    )

    parser.add_argument(
        "--overwrite", required=False, default=False, action="store_true",
        help="Overwrite any existing output file."
    )

    args = parser.parse_args()

    print("Getting nominal config...", flush=True)
    hw = sim_nominal()
    for tele, teleprops in hw.data["telescopes"].items():
        print("Simulating detectors for telescope {}...".format(tele),
              flush=True)
        sim_detectors_toast(hw, tele)

    if args.plain:
        outpath = "{}.toml".format(args.out)
        print("Dumping config to {}...".format(outpath))
        hw.dump(outpath, overwrite=args.overwrite, compress=False)
    else:
        outpath = "{}.toml.gz".format(args.out)
        print("Dumping config to {}...".format(outpath))
        hw.dump(outpath, overwrite=args.overwrite, compress=True)

    return
