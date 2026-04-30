# Copyright (c) 2024-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.fft import convolve
from toast.utils import Logger, rate_from_times
from toast.traits import trait_docs, Int, Unicode
from toast.timing import function_timer, Timer
from toast.observation import default_values as defaults
from toast.ops import Operator

from sotodlib.tod_ops import filters


@trait_docs
class ReadoutFilter(Operator):
    """Apply a readout filter to the signal"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional shared flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    iir_params = Unicode(
        "iir_params", help="Observation key for readout filter parameters"
    )

    wafer_key = Unicode("det_info:stream_id", help="Focalplane key for the wafer name")

    debug_root = Unicode(
        None,
        allow_none=True,
        help="Optional root filename for debug plots",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if self.iir_params not in ob:
                msg = "Cannot apply readout filter. "
                msg += f"'{self.iir_params}' does not exist in '{ob.name}'"
                raise RuntimeError(msg)

            # Get the sample rate from the data.  We also have nominal sample rates
            # from the noise model and also from the focalplane.
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )
            order = int(np.ceil(np.log(ob.n_local_samples) / np.log(2)))
            n_fft = 2 ** (order + 1)
            freq = np.fft.rfftfreq(n_fft, d=1.0 / rate)

            # Get valid local detectors
            local_dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            local_set_dets = set(local_dets)

            # Get the rows of the focalplane table containing these dets
            det_table = ob.telescope.focalplane.detector_data
            det_to_row = {
                y: x for x, y in enumerate(det_table["name"]) if y in local_set_dets
            }
            fp_rows = np.array([det_to_row[x] for x in local_dets])

            # Get the set of all stream IDs
            all_wafers = set(det_table[self.wafer_key][fp_rows])

            # The IIR filter parameters will either be in a single,
            # top-level dictionary or they are organized per-UFM.
            if (
                "per_stream" in ob[self.iir_params]
                and ob[self.iir_params]["per_stream"]
            ):
                # We need to filter one wafer at a time.
                for wf in all_wafers:
                    wafer_dets = [
                        x
                        for x, y in zip(local_dets, det_table[self.wafer_key][fp_rows])
                        if y == wf
                    ]
                    signal = ob.detdata[self.det_data][wafer_dets, :]
                    flags, mask = self._get_flags(ob, wafer_dets)
                    self._filter_detectors(rate, freq, signal, ob[wf], flags, mask)
            else:
                # We are filtering all detectors at once
                signal = ob.detdata[self.det_data][local_dets, :]
                flags, mask = self._get_flags(ob, local_dets)
                self._filter_detectors(
                    rate, freq, signal, ob[self.iir_params], flags, mask
                )

    @function_timer
    def _filter_detectors(self, rate, freq, det_array, iir_props, flags, mask):
        # Get the common filter kernel for all detectors
        iir_filter = filters.iir_filter(iir_params=iir_props)(freq, None)

        # Deconvolve
        convolve(
            det_array,
            rate,
            flags=flags,
            flag_mask=mask,
            kernel_freq=freq,
            kernels=iir_filter,
            kernel_func=None,
            deconvolve=True,
            algorithm="numpy",
            debug=self.debug_root,
        )

    def _get_flags(self, obs, fdets):
        """Helper function to get flags and mask to pass to convolution."""
        if self.det_flags is None:
            flags = None
            flag_mask = None
        else:
            flags = obs.detdata[self.det_flags][fdets, :]
            if self.shared_flags is not None:
                # These shared flags will effectively be propagated to
                # detector flags by this operator
                shflg = self.det_flag_mask * np.array(
                    obs.shared[self.shared_flags].data & self.shared_flag_mask,
                    dtype=np.uint8,
                )
                for detflag in flags:
                    detflag |= shflg
            flag_mask = self.det_flag_mask
        return flags, flag_mask

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": self.times,
            "detdata": self.det_data,
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()
