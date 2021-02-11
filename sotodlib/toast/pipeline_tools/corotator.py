# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger
import toast.qarray as qa

from ...core.hardware import LAT_COROTATOR_OFFSET_DEG


XAXIS, YAXIS, ZAXIS = np.eye(3)


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
            corotator_angle = obs["corotator_angle_deg"]
            offset, nsample = tod.local_samples
            tod.cache.put(cache_name, np.zeros(nsample) + corotator_angle)
        rot = qa.rotation(
            ZAXIS, np.radians(corotator_angle + LAT_COROTATOR_OFFSET_DEG)
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

    if comm is None or comm.comm_world.rank == 0:
        timer.report_clear("Rotate focalplane")

    return
