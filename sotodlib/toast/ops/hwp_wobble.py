# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

import toast
from toast.timing import function_timer
from toast.traits import (
    trait_docs,
    Int,
    Unicode,
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults
import toast.qarray as qa


@trait_docs
class HWPWobbleCorrect(Operator):
    """Correct the boresight pointing due to HWP wobble."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    wobble_meta = Unicode(
        "wobble_params:",
        help="The metadata prefix for the wobble parameters",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, help="Observation shared key for HWP angle"
    )

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if self.hwp_angle not in ob.shared:
                msg = f"HWP angle '{self.hwp_angle}' not defined for "
                msg += f"observation {ob.name}.  Skipping wobble correction."
                log.warning(msg)
                continue

            # The raw wobble correction files have just one number for each
            # wafer.  This gets expanded by the context into a duplicate
            # number for each detector.  Amplitudes are in arcmin, and phases
            # are in radians.
            amp_name = f"{self.wobble_meta}amp"
            phase_name = f"{self.wobble_meta}phase"
            fp_table = ob.telescope.focalplane.detector_data
            if amp_name not in fp_table.colnames:
                msg = f"HWP wobble meta data '{amp_name}' not found "
                msg += f"in observation {ob.name}.  Skipping."
                log.warning(msg)
                continue

            wobble_amp = fp_table[0][amp_name]
            wobble_phase = fp_table[0][phase_name]
            amp = np.radians(wobble_amp / 60.0)
            phase = wobble_phase
            hwp = ob.shared[self.hwp_angle].data

            dxi = amp * np.cos(hwp - phase)
            deta = -amp * np.sin(hwp - phase)

            deflq = toast.instrument_coords.xieta_to_quat(dxi, deta, 0.0)

            # Apply deflection to boresight pointing
            if ob.comm_col_rank == 0:
                bore = qa.mult(ob.shared[self.boresight_azel].data, qa.inv(deflq))
            else:
                bore = None
            ob.shared[self.boresight_azel].set(bore)
            if ob.comm_col_rank == 0:
                bore = qa.mult(ob.shared[self.boresight_radec].data, qa.inv(deflq))
            else:
                bore = None
            ob.shared[self.boresight_radec].set(bore)
            del bore

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.wobble_meta],
            "shared": [self.hwp_angle, self.boresight_radec, self.boresight_azel],
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        return dict()
