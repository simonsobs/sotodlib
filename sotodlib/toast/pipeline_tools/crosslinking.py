# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse

import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..crosslinking import OpCrossLinking


def add_crosslinking_args(parser):
    parser.add_argument(
        "--crosslinking",
        required=False,
        action="store_true",
        help="Write crosslinking map",
        dest="write_crosslinking",
    )
    parser.add_argument(
        "--no-crosslinking",
        required=False,
        action="store_false",
        help="Do not write crosslinking map",
        dest="write_crosslinking",
    )
    parser.set_defaults(write_crosslinking=False)
    parser.add_argument(
        "--crosslinking-prefix",
        required=False,
        help="Prefix to apply to output map",
    )
    try:
        parser.add_argument(
            "--zip",
            required=False,
            action="store_true",
            help="Compress the map outputs",
            dest="zip_maps",
        )
        parser.add_argument(
            "--no-zip",
            required=False,
            action="store_false",
            help="Do not compress the map outputs",
            dest="zip_maps",
        )
        parser.set_defaults(zip_maps=True)
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--out", required=False, default=".", help="Output directory",
        )
    except argparse.ArgumentError:
        pass

    return


def compute_crosslinking(args, comm, data, detweights=None, verbose=True):
    if not args.write_crosslinking:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    crosslinking = OpCrossLinking(
        outdir=args.out,
        outprefix=args.crosslinking_prefix,
        common_flag_mask=args.common_flag_mask,
        flag_mask=255,
        zip_maps=args.hn_zip,
        rcond_limit=1e-3,
        detweights=detweights,
    )
    crosslinking.exec(data)
    timer.report_clear("Compute crosslinking map")

    return
