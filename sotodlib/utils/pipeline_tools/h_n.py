# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ...modules import OpHn


def add_h_n_args(parser):
    parser.add_argument(
        "--hn-outdir",
        required=False,
        default="hn_maps",
        help="output directory for h_n maps",
    )
    parser.add_argument(
        "--hn-prefix",
        required=False,
        help="Prefix to apply to output maps",
    )
    parser.add_argument(
        "--hn-min",
        required=False,
        type=np.int,
        default=1,
        help="Minimum `n` for h_n maps.",
    )
    parser.add_argument(
        "--hn-max",
        required=False,
        default=0,
        type=np.int,
        help="Maximum `n` for h_n maps.  Use --hn-max < 1 to disable",
    )
    parser.add_argument(
        "--hn-zip",
        required=False,
        action="store_true",
        help="Compress the h_n maps using zip.",
        dest="hn_zip",
    )
    parser.add_argument(
        "--no-hn-zip",
        required=False,
        action="store_false",
        help="Compress the h_n maps using zip.",
        dest="hn_zip",
    )
    parser.set_defaults(hn_zip=False)
    return


def compute_h_n(args, comm, data, verbose=True):
    if args.hn_max < args.hn_min:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    hnop = OpHn(
        outdir=args.hn_outdir,
        outprefix=args.hn_prefix,
        nmin=args.hn_min,
        nmax=args.hn_max,
        common_flag_mask=args.common_flag_mask,
        flag_mask=255,
        zip_maps=args.hn_zip,
    )
    hnop.exec(data)
    timer.report_clear("Compute h_n")

    return
