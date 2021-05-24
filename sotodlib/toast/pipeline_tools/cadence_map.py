# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse

import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..cadence_map import OpCadenceMap


def add_cadence_map_args(parser):
    parser.add_argument(
        "--cadence-map",
        required=False,
        action="store_true",
        help="Write cadence map",
        dest="write_cadence_map",
    )
    parser.add_argument(
        "--no-cadence_map",
        required=False,
        action="store_false",
        help="Do not write cadence map",
        dest="write_cadence_map",
    )
    parser.set_defaults(write_cadence_map=False)
    parser.add_argument(
        "--cadence-map-prefix",
        required=False,
        help="Prefix to apply to output map",
    )
    try:
        parser.add_argument(
            "--out", required=False, default=".", help="Output directory",
        )
    except argparse.ArgumentError:
        pass

    return


def compute_cadence_map(args, comm, data, verbose=True):
    if not args.write_cadence_map:
        return
    log = Logger.get()
    if comm.world_rank == 0:
        log.info("Computing cadence map")
    timer = Timer()
    timer.start()
    cadence = OpCadenceMap(
        outdir=args.out,
        outprefix=args.cadence_map_prefix,
        common_flag_mask=args.common_flag_mask,
        flag_mask=255,
    )
    cadence.exec(data)
    if comm.world_rank == 0:
        timer.report_clear("Compute cadence map")

    return
