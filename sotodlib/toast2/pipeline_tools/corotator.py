# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger
import toast.qarray as qa

from ...core.hardware import LAT_COROTATOR_OFFSET_DEG


XAXIS, YAXIS, ZAXIS = np.eye(3)

def add_corotator_args(parser):
    parser.add_argument(
        "--corotate-lat",
        required=False,
        action="store_true",
        help="Rotate LAT receiver to maintain focalplane orientation",
        dest="corotate_lat",
    )
    parser.add_argument(
        "--no-corotate-lat",
        required=False,
        action="store_false",
        help="Do not Rotate LAT receiver to maintain focalplane orientation",
        dest="corotate_lat",
    )
    parser.set_defaults(corotate_lat=True)
    return


def rotate_focalplane(args, data, comm):
    """ The LAT focalplane projected on the sky rotates as the cryostat
    (co-rotator) tilts.  Usually the tilt is the same as the observing
    elevation to maintain constant angle between the mirror and the cryostat.

    This method must be called *before* expanding the detector pointing
    from boresight.
    """

    log = Logger.get()
    timer = Timer()
    timer.start()

    for obs in data.obs:
        if obs["telescope"] != "LAT":
            continue
        tod = obs["tod"]
        cache_name = "corotator_angle_deg"
        if tod.cache.exists(cache_name):
            corotator_angle = tod.cache.reference(cache_name)
        else:
            # If a vector of co-rotator angles isn't already cached,
            # make one now from the observation metadata.  This will
            # ensure they get recorded in the so3g files.
            corotator_angle = obs["corotator_angle_deg"]
            offset, nsample = tod.local_samples
            tod.cache.put(cache_name, np.zeros(nsample) + corotator_angle)
        el = np.degrees(tod.read_boresight_el())
        rot = qa.rotation(
            ZAXIS, np.radians(corotator_angle + el + LAT_COROTATOR_OFFSET_DEG)
        )
        quats = tod.read_boresight()
        quats[:] = qa.mult(quats, rot)
        try:
            # If there are horizontal boresight quaternions, they need
            # to be rotated as well.
            quats = tod.read_boresight(azel=True)
            quats[:] = qa.mult(quats, rot)
        except Exception as e:
            pass

    if comm.comm_world is None or comm.comm_world.rank == 0:
        timer.report_clear("Rotate focalplane")

    return
