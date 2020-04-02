# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer

from ..modules import OpTimeConst


def add_transfer_function_args(parser):
    parser.add_argument(
        "--tf-convolve",
        required=False,
        action="store_true",
        help="Convolve the data to simulate a transfer function",
        dest="tf_convolve"
    )
    parser.add_argument(
        "--tf-deconvolve",
        required=False,
        action="store_true",
        help="De-convolve the data to compensate for a transfer function",
        dest="tf_deconvolve"
    )
    parser.add_argument(
        "--tf-tau",
        required=False,
        type=np.float,
        help="Value of the transfer function time constant",
        dest="tf_deconvolve"
    )
    return


def convolve_transfer_function(args, comm, data, name, verbose=True):
    if not args.tf_convolve:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    tauop = OpTimeConst(name=name, tau=args.tf_tau, inverse=False)
    tauop.exec(data)
    timer.report_clear("Convolve transfer function")

    return


def deconvolve_transfer_function(args, comm, data, name, verbose=True):
    if not args.tf_convolve:
        return

    log = Logger.get()
    timer = Timer()
    timer.start()
    tauop = OpTimeConst(name=name, tau=args.tf_tau, inverse=True)
    tauop.exec(data)
    timer.report_clear("De-convolve transfer function")

    return
