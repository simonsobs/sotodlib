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

    timeconst = Unicode(
        "timeconst", help="Observation key for time constant"
    )

    readout_filter_cal = Unicode(
        "readout_filter_cal", help="Observation key for readout filter gain"
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

            if self.iir_params not in ob:
                msg = f"Cannot apply readout filter. "
                msg += f"'{self.iir_params}' does not exist in '{ob.name}'"
                raise RuntimeError(msg)
            signal = ob.detdata[self.det_data].data
            ndet, nsample = np.atleast_2d(signal).shape

            # From https://github.com/simonsobs/pwg-scripts/blob/99557fb4410a4751b0416687431b2713e38085b7/pwg-pmn/simple_ml_mapmaker/mapmaker.py#L230
            # Fourier-space calibration
            fsig = fft.rfft(signal)
            freq = fft.rfftfreq(nsample, dt)
            # iir filter
            iir_filter  = filters.iir_filter()(freq, ob)
            fsig /= iir_filter

            if self.timeconst is not None:
                if self.timeconst in ob:
                    fsig /= filters.timeconst_filter(None)(freq, ob)
                else:
                    msg = f"Cannot deconvolve time constant. "
                    msg += f"'{self.timeconst}' not in '{ob.name}'"
                    log.warning_rank(msg, comm=gcomm)
            fft.irfft(fsig, signal, normalize=True)

            # Correct for the readout filter gain
            # (unclear why there is one)
            if self.readout_filter_cal is not None:
                if self.readout_filter_cal in ob:
                    gain = ob[self.readout_filter_cal]
                    signal /= gain[:, np.newaxis]
                else:
                    msg = f"Cannot correct for readout filter gain. "
                    msg += f"'{self.readout_filter_cal}' not in '{ob.name}'"
                    log.warning_rank(msg, comm=gcomm)
        return

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
