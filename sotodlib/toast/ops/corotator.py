# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

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
    """The LAT focalplane projected on the sky rotates as the cryostat
    (co-rotator) tilts.  Usually the tilt is the same as the observing
    elevation to maintain constant angle between the mirror and the cryostat.

    This operator be applied *before* expanding the detector pointing
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
        help="Observation shared key for corotator_angle",
    )

    elevation = Unicode(
        defaults.elevation,
        allow_none=True,
        help="Observation shared key for boresight elevation",
    )

    corotate_lat = Bool(
        True, help="Rotate LAT receiver to maintain focalplane orientation"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        for obs in data.obs:
            if obs.telescope.name != "LAT":
                continue

            if self.corotate_lat:
                obs["corotator_angle"] = -obs["scan_el"]
            else:
                obs["corotator_angle"] = -60 * u.deg

            obs.shared.create_column(
                self.corotator_angle,
                shape=(obs.n_local_samples,),
                dtype=np.float64,
            )
            corotator_angle = obs.shared[self.corotator_angle]
            corotator_angle.set(
                np.zeros(obs.n_local_samples)
                + obs["corotator_angle"].to_value(u.rad),
                offset=(0,),
                fromrank=0,
            )

            el = obs.shared[self.elevation]  # In radians
            rot = qa.rotation(
                ZAXIS,
                corotator_angle.data + el.data +
                LAT_COROTATOR_OFFSET.to_value(u.rad),
            )
            for name in [self.boresight_radec, self.boresight_azel]:
                if name is None:
                    continue
                quats = obs.shared[name]
                new_quats = qa.mult(quats, rot)
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
