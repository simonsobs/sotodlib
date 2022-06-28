# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..sim_hwpss import OpSimHWPSS


def add_sim_hwpss_args(parser):
    parser.add_argument(
        "--hwpss-file",
        required=False,
        help="Database of HWPSS by frequency and incident angle",
    )
    parser.add_argument(
        "--simulate-hwpss",
        required=False,
        action="store_true",
        help="Simulate HWPSS",
        dest="simulate_hwpss",
    )
    parser.add_argument(
        "--no-simulate-hwpss",
        required=False,
        action="store_false",
        help="Do not simulate HWPSS",
        dest="simulate_hwpss",
    )
    parser.set_defaults(simulate_hwpss=False)
    return


def simulate_hwpss(args, comm, data, mc, name):
    if not args.simulate_hwpss:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    hwpssop = OpSimHWPSS(name=name, fname_hwpss=args.hwpss_file, mc=mc)
    hwpssop.exec(data)
    timer.report_clear("Simulate HWPSS")

    return
