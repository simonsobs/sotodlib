# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.utils import Logger, rate_from_times
from toast.traits import trait_docs, Int, Unicode
from toast.timing import function_timer
from toast.observation import default_values as defaults
from toast.ops import Operator

from sotodlib.tod_ops import filters
from pixell import fft


@trait_docs
class ReadoutFilter(Operator):
    """Apply a readout filter to the signal

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    iir_params = Unicode(
        "iir_params", help="Observation key for readout filter parameters"
    )

    wafer_key = Unicode(
        "det_info:stream_id", help="Focalplane key for the wafer name"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        gcomm = data.comm.comm_group

        for ob in data.obs:
            # Get the sample rate from the data.  We also have nominal sample rates
            # from the noise model and also from the focalplane.
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )
            freq = fft.rfftfreq(ob.n_local_samples, dt)

            if self.iir_params not in ob:
                msg = f"Cannot apply readout filter. "
                msg += f"'{self.iir_params}' does not exist in '{ob.name}'"
                raise RuntimeError(msg)

            # Get valid local detectors
            local_dets = ob.select_local_detectors()
            local_set_dets = set(local_dets)

            # Get the rows of the focalplane table containing these dets
            det_table = ob.telescope.focalplane.detector_data
            fp_rows = np.array(
                [x for x, y in enumerate(det_table["name"]) if y in local_set_dets]
            )

            # Get the set of all stream IDs
            all_wafers = set(det_table[self.wafer_key][fp_rows])

            # The IIR filter parameters will either be in a single, top-level dictionary
            # or they are organized per-UFM.
            if (
                "per_stream" in ob[self.iir_params] and
                ob[self.iir_params]["per_stream"]
            ):
                # We need to filter one wafer at a time.
                for wf in all_wafers:
                    wafer_dets = [
                        x for x, y in zip(
                            local_dets, det_table[self.wafer_key][fp_rows]
                        ) if y == wf
                    ]
                    signal = ob.detdata[self.det_data][wafer_dets, :]
                    self._filter_detectors(freq, signal, ob[wf])
            else:
                # We are filtering all detectors at once
                signal = ob.detdata[self.det_data][local_dets, :]
                self._filter_detectors(freq, signal, ob[self.iir_params])

    def _filter_detectors(self, freq, det_array, iir_props):
        ndet, nsample = det_array.shape

        # Array of detector ffts
        fsig = fft.rfft(det_array)

        # Apply iir filter kernel
        iir_filter = filters.iir_filter(iir_params=iir_props)(freq, None)
        fsig /= iir_filter

        # Inverse fft
        fft.irfft(fsig, det_array, normalize=True)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared" : self.times,
            "detdata": self.det_data,
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()
