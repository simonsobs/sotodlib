# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import re

import numpy as np

import traitlets

import numpy as np

from astropy import units as u

import healpy as hp

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from toast.ops.operator import Operator

from toast.utils import Environment, Logger, Timer

from toast.observation import default_values as defaults

from ...core.hardware import LAT_COROTATOR_OFFSET


XAXIS, YAXIS, ZAXIS = np.eye(3)

@trait_docs
class CoRotator(Operator):
    """Compute the LAT focalplane rotation.

    The optical design configuration of the LAT projected on the sky has the "O6" cryo
    tube in the positive azimuth direction when the telescope is at 60 degrees
    elevation.

    If corotate_lat is False, then the projected focalplane will rotate on the sky.
    The saved corotation angle will be zero.  If corotate_lat is True, then the
    corotation angle is set to compensate for the rotation caused by the observing
    elevation.

    The rotation of the projected focalplane is given by:

        R = elevation - offset - corotator angle

    This operator should be applied *before* expanding the detector pointing
    from boresight.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    boresight_azel = Unicode(
        defaults.boresight_azel,
        allow_none=True,
        help="Observation shared key for boresight Az/El",
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        allow_none=True,
        help="Observation shared key for boresight RA/Dec",
    )

    corotator_angle = Unicode(
        "corotator_angle",
        allow_none=True,
        help="Observation shared key for corotation angle",
    )

    elevation = Unicode(
        defaults.elevation,
        allow_none=True,
        help="Observation shared key for boresight elevation",
    )

    corotate_lat = Bool(
        True,
        help="If True, rotate LAT receiver to maintain projected focalplane orientation"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        for obs in data.obs:
            if re.match(r"LAT.*", obs.telescope.name) is None:
                msg = f"Skipping corotation for telescope '{obs.telescope.name}'"
                log.debug(msg)
                continue

            el_rad = obs.shared[self.elevation].data

            if self.corotator_angle not in obs.shared:
                obs.shared.create_column(
                    self.corotator_angle,
                    shape=(obs.n_local_samples,),
                    dtype=np.float64,
                )

            corot = None
            if self.corotate_lat:
                # We are corotating the receiver.  We set this to a fixed angle
                # based on the nominal elevation for the scan.
                if obs.comm_col_rank == 0:
                    if "scan_el" in obs:
                        scan_el = obs["scan_el"].to_value(u.radian)
                    else:
                        scan_el = np.mean(el_rad)
                    scan_el_deg = scan_el * 180.0 / np.pi
                    corot = np.zeros(obs.n_local_samples) + (
                        scan_el - LAT_COROTATOR_OFFSET.to_value(u.radian)
                    )
                    corot_deg = corot * 180.0 / np.pi
                    msg = f"LAT Co-rotation:  obs {obs.name} at scan El = "
                    msg += f"{scan_el_deg:0.2f} degrees, rotating by "
                    msg += f"{np.mean(corot_deg):0.2f} average degrees"
                    log.info(msg)
                obs.shared[self.corotator_angle].set(
                    corot,
                    offset=(0,),
                    fromrank=0,
                )
            else:
                # We are not co-rotating.  Set the angle to zero.
                if obs.comm_col_rank == 0:
                    msg = f"LAT Co-rotation:  obs {obs.name} disabled"
                    log.info(msg)
                    corot = np.zeros(obs.n_local_samples)
                obs.shared[self.corotator_angle].set(
                    corot,
                    offset=(0,),
                    fromrank=0,
                )

            # Now correct the boresight quaternions

            rot = None
            if obs.comm_col_rank == 0:
                rot = qa.from_axisangle(
                    ZAXIS,
                    el_rad - LAT_COROTATOR_OFFSET.to_value(u.rad) - obs.shared[self.corotator_angle].data,
                )

            for name in [self.boresight_radec, self.boresight_azel]:
                if name is None:
                    continue
                quats = obs.shared[name]
                new_quats = None
                if obs.comm_col_rank == 0:
                    new_quats = qa.mult(rot, quats)
                quats.set(new_quats, offset=(0, 0), fromrank=0)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [
                self.elevation
            ]
        }
        if self.boresight_azel is not None:
            req["shared"].append(self.boresight_azel)
        if self.boresight_radec is not None:
            req["shared"].append(self.boresight_radec)
        return req

    def _provides(self):
        return {
            "shared": [
                self.corotator_angle,
            ]
        }

    def _accelerators(self):
        return list()
