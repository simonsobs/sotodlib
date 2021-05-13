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
        "There can be multiple --flag-sso arguments",
    )
    parser.add_argument(
        "--flag-sso-mask",
        default=1,
        type=np.uint8,
        help="Bitmask to apply when masking SSOs",
    )

    return


def apply_flag_sso(args, comm, data, verbose=True):
    if args.flag_sso is None:
        return

    if comm.world_rank == 0 and verbose:
        print(f"Flagging SSO:s", flush=True)

    timer = Timer()
    timer.start()

    sso_names = []
    sso_radii = []
    for arg in args.flag_sso:
        sso_name, sso_radius = arg.split(",")
        sso_radius = np.radians(float(sso_radius) / 60)
        sso_names.append(sso_name)
        sso_radii.append(sso_radius)

    flag_sso = OpFlagSSO(sso_names, sso_radii, flag_mask=args.flag_sso_mask)
    flag_sso.exec(data)

    if comm.world_rank == 0 and verbose:
        timer.report_clear(f"Flag {sso_names}")

    return
