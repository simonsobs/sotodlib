# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ..sim_sso import OpSimSSO


def add_sim_sso_args(parser):
    parser.add_argument(
        "--simulate-sso",
        required=False,
        help="Comma-separated list of solar system objects (SSOs) to simulate",
    )
    parser.add_argument(
        "--beam-file",
        required=False,
        help="Pickle file containing instrumental beam",
    )

    return


def apply_sim_sso(args, comm, data, mc, totalname, verbose=True):
    if args.simulate_sso is None:
        return

    if args.beam_file is None:
        raise RuntimeError("Cannot simulate SSOs without a beam file")

    timer = Timer()
    timer.start()

    for sso_name in args.simulate_sso.split(","):
        if comm.world_rank == 0 and verbose:
            print("Simulating {}".format(sso_name), flush=True)

        sim_sso = OpSimSSO(sso_name, args.beam_file, out=totalname)
        sim_sso.exec(data)

        if comm.world_rank == 0 and verbose:
            timer.report_clear("Simulate {}".format(sso_name))

    return


