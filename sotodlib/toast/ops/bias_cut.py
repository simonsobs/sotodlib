# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

import traitlets

import toast
from toast.timing import function_timer, Timer
from toast.traits import (
    trait_docs,
    Int,
    Unicode,
    Tuple,
    Bool,
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults


@trait_docs
class DetBiasCuts(Operator):
    """Use IV and bias step data to cut poorly biased detectors."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detcal_prefix = Unicode(
        "det_cal:",
        help="The metadata prefix for detector calibration",
    )

    rfrac_range = Tuple(
        (0.1, 0.7),
        help="The (lower, upper) bounds for rfrac selection",
    )

    psat_range = Tuple(
        (),
        help="The (lower, upper) bounds for P_SAT from IV analysis",
    )

    rn_range = Tuple(
        (),
        help="The (lower, upper) range of r_n for det selection",
    )

    si_nan = Bool(False, help="If True, flag dets where s_i is NaN")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for signal",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for detector flags",
    )

    bias_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to apply to detectors that are poorly biased",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("bias_mask")
    def _check_bias_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Bias mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        bg_key = f"{self.detcal_prefix}bg"
        r_tes_key = f"{self.detcal_prefix}r_tes"
        r_frac_key = f"{self.detcal_prefix}r_frac"
        p_sat_key = f"{self.detcal_prefix}p_sat"
        r_n_key = f"{self.detcal_prefix}r_n"
        s_i_key = f"{self.detcal_prefix}s_i"

        # Each process updates the detector flags for their local dets.
        for ob in data.obs:
            bias_flags = dict()
            fp = ob.telescope.focalplane
            cnames = set(fp.detector_data.colnames)
            for det in ob.select_local_detectors(detectors, flagmask=self.det_mask):
                fp_det = fp[det]
                if bg_key not in cnames:
                    msg = f"{ob.name}:{det} no focalplane key '{bg_key}', cutting"
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if fp_det[bg_key] < 0:
                    msg = f"{ob.name}:{det} {bg_key}={fp_det[bg_key]} < 0."
                    msg += " Cutting."
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if r_tes_key not in cnames:
                    msg = f"{ob.name}:{det} no focalplane key '{r_tes_key}', cutting"
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if fp_det[r_tes_key] <= 0:
                    msg = f"{ob.name}:{det} {r_tes_key}={fp_det[r_tes_key]} <= 0."
                    msg += " Cutting."
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if r_frac_key not in cnames:
                    msg = f"{ob.name}:{det} no focalplane key '{r_frac_key}', cutting"
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if (
                    fp_det[r_frac_key] < self.rfrac_range[0]
                    or fp_det[r_frac_key] > self.rfrac_range[1]
                ):
                    msg = f"{ob.name}:{det} {r_frac_key}={fp_det[r_frac_key]} outside"
                    msg += f" range {self.rfrac_range}. Cutting."
                    log.debug(msg)
                    bias_flags[det] = self.bias_mask
                    continue
                if len(self.psat_range) != 0:
                    if len(self.psat_range) != 2:
                        msg = "psat_range should be a 2-tuple with the low, high values"
                        raise RuntimeError(msg)
                    if p_sat_key not in cnames:
                        msg = f"{ob.name}:{det} no focalplane key '{p_sat_key}',"
                        msg += " but PSAT range specified. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
                    if (
                        fp_det[p_sat_key] * 1e12 < self.psat_range[0]
                        or fp_det[p_sat_key] * 1e12 > self.psat_range[1]
                    ):
                        msg = f"{ob.name}:{det} {p_sat_key}={fp_det[p_sat_key]} outside"
                        msg += f" range {self.psat_range}. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
                if len(self.rn_range) != 0:
                    if len(self.rn_range) != 2:
                        msg = "rn_range should be a 2-tuple with the low, high values"
                        raise RuntimeError(msg)
                    if r_n_key not in cnames:
                        msg = f"{ob.name}:{det} no focalplane key '{r_n_key}',"
                        msg += " but r_n range specified. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
                    if (
                        fp_det[r_n_key] < self.rn_range[0]
                        or fp_det[r_n_key] > self.rn_range[1]
                    ):
                        msg = f"{ob.name}:{det} {r_n_key}={fp_det[r_n_key]} outside"
                        msg += f" range {self.rn_range}. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
                if self.si_nan:
                    if s_i_key not in cnames:
                        msg = f"{ob.name}:{det} no focalplane key '{s_i_key}',"
                        msg += " but s_i NaN check is True. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
                    if np.isnan(fp_det[s_i_key]):
                        msg = f"{ob.name}:{det} {s_i_key} is NaN. Cutting."
                        log.debug(msg)
                        bias_flags[det] = self.bias_mask
                        continue
            ob.update_local_detector_flags(bias_flags)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.times],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()
