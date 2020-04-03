# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer

from ..modules import OpTimeConst


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
        help="Value of the time constant time constant",
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


def deconvolve_time_constant(args, comm, data, name, verbose=True):
    if not args.tf_convolve:
        return

    log = Logger.get()
    timer = Timer()
    timer.start()
    tauop = OpTimeConst(name=name, tau=args.tf_tau, inverse=True)
    tauop.exec(data)
    timer.report_clear("De-convolve time constant")

    return
