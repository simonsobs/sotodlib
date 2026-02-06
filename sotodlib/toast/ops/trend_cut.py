# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from astropy import units as u

import traitlets

import toast
from toast.timing import function_timer, Timer
from toast.traits import (
    trait_docs,
    Int,
    Unicode,
    Float,
    Quantity,
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults


@trait_docs
class DetTrendCuts(Operator):
    """Use piecewise changes in slope to detect unlocked detectors.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    max_trend = Float(
        2.5,
        help="Slope (in phase units) at which detectors are unlocked"
    )

    t_piece = Quantity(
        10 * u.second,
        help="Time for the piecewise chunks",
    )

    max_samples = Int(
        500,
        help="Max samples per chunk for trend calc.  Longer chunks are decimated",
    )

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

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional shared flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for detector flags",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    trend_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to apply to detectors that are outlier trends",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("trend_mask")
    def _check_trend_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Trend mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            # Relative timestamps within the observation
            reltime = ob.shared[self.times].data.copy()
            t_0 = reltime[0]
            reltime -= t_0

            # Compute chunking
            slices = self._compute_chunk_slices(ob, reltime)

            # Flag detectors with any chunk-wise slope beyond the limit
            trend_flags = dict()
            for det in ob.select_local_detectors(detectors, flagmask=self.det_mask):
                slopes = self._compute_trend(ob, reltime, slices, det)
                max_abs_slope = np.amax(np.absolute(slopes))
                if max_abs_slope > self.max_trend:
                    msg = f"{ob.name}:{det} max trend slope {max_abs_slope} is "
                    msg += f"greater than limit ({self.max_trend}), cutting."
                    log.debug(msg)
                    trend_flags[det] = self.trend_mask
                else:
                    msg = f"{ob.name}:{det} max trend slope {max_abs_slope} is "
                    msg += f"less than limit ({self.max_trend}), keeping."
                    log.verbose(msg)

            # Update per-detector flags
            ob.update_local_detector_flags(trend_flags)

    def _compute_chunk_slices(self, obs, reltime):
        # Sample rate
        (rate, dt, dt_min, dt_max, dt_std) = toast.utils.rate_from_times(reltime)

        # Approximate target samples per chunk
        n_samples_per_piece = int(self.t_piece.to_value(u.second) * rate)

        # Distribute total samples into uniform pieces
        dist = toast.dist.distribute_uniform(len(reltime), n_samples_per_piece)

        # For each chunk we either use all samples or we downsample to meet the
        # max samples per chunk.
        slices = list()
        for offset, n_elem in dist:
            if n_elem > self.max_samples:
                step = 1 + n_elem // self.max_samples
            else:
                step = 1
            slices.append(slice(offset, offset + n_elem, step))
        return slices

    def _compute_trend(self, obs, reltime, slices, det):
        # This is a copy of the algorithm that is used in 
        # sotodlib.tod_ops.flags.get_trending_flags()
        signal = obs.detdata[self.det_data][det]
        flags = obs.shared[self.shared_flags].data & self.shared_flag_mask
        if self.det_flags is not None:
            flags |= obs.detdata[self.det_flags][det] & self.det_flag_mask
        slopes = list()
        for slc in slices:
            good = flags[slc] == 0
            t = reltime[slc][good]
            s = signal[slc][good]

            t_mean = np.mean(t)
            t_var = np.var(t, mean=t_mean)
            s_mean = np.mean(s)
            ts_mean = np.mean(t * s)

            slopes.append((ts_mean - t_mean * s_mean) / t_var)
        return np.array(slopes)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.times],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()
