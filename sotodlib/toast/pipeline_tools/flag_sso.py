# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..flag_sso import OpFlagSSO


def add_flag_sso_args(parser):
    parser.add_argument(
        "--flag-sso",
        required=False,
        action="append",
        help="SSO name and flagging radius as <name>,<radius-arcmin>. "
        "There can be multiple --flag-sso flags",
    )

    return


def apply_flag_sso(args, comm, data, verbose=True):
    if args.flag_sso is None:
        return

    timer = Timer()
    timer.start()

    for arg in args.flag_sso:
        sso_name, sso_radius = arg.split(",")
        sso_radius = np.radians(float(sso_radius) / 60)
        if comm.world_rank == 0 and verbose:
            print("Flagging {}".format(sso_name), flush=True)

        flag_sso = OpFlagSSO(sso_name, sso_radius)
        flag_sso.exec(data)

        if comm.world_rank == 0 and verbose:
            timer.report_clear("Flag {}".format(sso_name))

    return
