# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..time_constant import OpTimeConst


def add_time_constant_args(parser):
    parser.add_argument(
        "--tau-convolve",
        required=False,
        action="store_true",
        help="Convolve the data to simulate a time constant",
        dest="tau_convolve"
    )
    parser.add_argument(
        "--tau-deconvolve",
        required=False,
        action="store_true",
        help="De-convolve the data to compensate for a time constant",
        dest="tau_deconvolve"
    )
    parser.add_argument(
        "--tau-value",
        required=False,
        type=np.float,
        help="Value of the time constant in seconds.",
    )
    parser.add_argument(
        "--tau-sigma",
        required=False,
        type=np.float,
        help="Relative width of time constant errors applied in "
        "deconvolution.  Randomized by detector and observation.",
    )
    return


def convolve_time_constant(args, comm, data, name, verbose=True):
    if not args.tau_convolve:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    tauop = OpTimeConst(name=name, tau=args.tau_value, inverse=False)
    tauop.exec(data)
    timer.report_clear("Convolve time constant")

    return


def deconvolve_time_constant(args, comm, data, name, realization=0, verbose=True):
    if not args.tau_deconvolve:
        return

    log = Logger.get()
    timer = Timer()
    timer.start()
    tauop = OpTimeConst(
        name=name,
        tau=args.tau_value,
        inverse=True,
        tau_sigma=args.tau_sigma,
        realization=realization,
    )
    tauop.exec(data)
    timer.report_clear("De-convolve time constant")

    return
