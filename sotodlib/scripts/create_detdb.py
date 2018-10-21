# Copyright (c) 2018 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Detector DB creation.
"""

import sys
import os
import argparse

from ..db.hardware import sim_small_aperature, sim_large_aperature


def main():
    parser = argparse.ArgumentParser(\
        description="Create a new detector DB",
        usage="sotod_create_detdb [options] (use --help for details)")

    parser.add_argument("--out", required=False, default="so_det.db",
        help="output detector DB")

    args = parser.parse_args(sys.argv)

    return
