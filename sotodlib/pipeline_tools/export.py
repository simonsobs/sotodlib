# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import sys

import numpy as np

from toast.timing import function_timer, Timer
from toast.tod import TODGround
from toast.utils import Logger


def add_export_args(parser):
    parser.add_argument(
        "--export", required=False, default=None, help="Output TOD export path"
    )
    parser.add_argument(
        "--export-key",
        required=False,
        default=None,
        help="Group exported TOD by a detector trait: wafer, card, crate or tube",
    )
    parser.add_argument(
        "--export-compress",
        required=False,
        action="store_true",
        help="Re-digitize and compress the exported signal.",
        dest="compress",
    )
    parser.add_argument(
        "--no-export-compress",
        required=False,
        action="store_false",
        help="Do not re-digitize and compress the exported signal.",
        dest="compress",
    )
    parser.set_defaults(compress=False)
    return


@function_timer
def export_TOD(args, comm, data, totalname, schedules, other=None, verbose=True):
    if args.export is None:
        return

    log = Logger.get()
    timer = Timer()

    # Only import spt3g if we are writing out so3g files
    from spt3g import core as core3g
    from ..data.toast_export import ToastExport

    path = os.path.abspath(args.export)

    key = args.export_key
    if key is not None:
        prefix = "{}_{}".format(args.bands, key)
        det_groups = {}
        for schedule in schedules:
            for (
                det_name,
                det_data,
            ) in schedule.telescope.focalplane.detector_data.items():
                value = det_data[key]
                if value not in det_groups:
                    det_groups[value] = []
                det_groups[value].append(det_name)
    else:
        prefix = args.bands
        det_groups = None

    if comm.world_rank == 0 and verbose:
        log.info("Exporting data to directory tree at {}".format(path))

    timer.start()
    export = ToastExport(
        path,
        prefix=prefix,
        use_intervals=True,
        cache_name=totalname,
        cache_copy=other,
        mask_flag_common=TODGround.TURNAROUND,
        filesize=2 ** 30,
        units=core3g.G3TimestreamUnits.Tcmb,
        detgroups=det_groups,
        compress=args.compress,
    )
    export.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.Barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Wrote simulated data to {}:{}" "".format(path, "total"))

    return
