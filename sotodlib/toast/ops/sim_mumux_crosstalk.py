# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np
import toast.rng
from astropy import units as u
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer, Timer
from toast.traits import Int, Unicode, trait_docs
from toast.utils import Environment, Logger, unit_conversion

from .mumux_crosstalk_util import detmap_available, pos_to_chi

# https://github.com/simonsobs/bolocalc-so-model/blob/master/V3r7/V3r7_Baseline
# Total optical power in pW
P_OPT = {
    "SAT_f030" : 0.158,
    "SAT_f040" : 0.267,
    "SAT_f090" : 1.51,
    "SAT_f150" : 2.69,
    "SAT_f230" : 7.27,
    "SAT_f290" : 11.18,
    "LAT_f030" : 0.21,
    "LAT_f040" : 1.02,
    "LAT_f090" : 1.04,
    "LAT_f150" : 2.23,
    "LAT_f230" : 7.50,
    "LAT_f290" : 12.48,
}
# Optical power due to standard atmosphere
P_ATM = {
    "SAT_f030" : 0.059685257,
    "SAT_f040" : 0.618673466,
    "SAT_f090" : 0.811076040,
    "SAT_f150" : 1.343902747,
    "SAT_f230" : 4.118517930,
    "SAT_f290" : 7.056159517,
    "LAT_f030" : 0.042984911,
    "LAT_f040" : 0.495809196,
    "LAT_f090" : 0.509838815,
    "LAT_f150" : 1.000122526,
    "LAT_f230" : 3.359030038,
    "LAT_f290" : 6.284548710,
}
# Optical effiency between the detector and the atmosphere
ETA_ATM = {
    "SAT_f030" : 0.160,
    "SAT_f040" : 0.278,
    "SAT_f090" : 0.206,
    "SAT_f150" : 0.267,
    "SAT_f230" : 0.310,
    "SAT_f290" : 0.344,
    "LAT_f030" : 0.115,
    "LAT_f040" : 0.216,
    "LAT_f090" : 0.130,
    "LAT_f150" : 0.199,
    "LAT_f230" : 0.252,
    "LAT_f290" : 0.305,
}
# Saturation power [pW]
P_SAT = {
    "SAT_f030" : 1.08,
    "SAT_f040" : 4.62,
    "SAT_f090" : 3.42,
    "SAT_f150" : 9.37,
    "SAT_f230" : 29.4,
    "SAT_f290" : 31.8,
    "LAT_f030" : 1.08,
    "LAT_f040" : 4.62,
    "LAT_f090" : 3.42,
    "LAT_f150" : 9.37,
    "LAT_f230" : 29.4,
    "LAT_f290" : 31.8,
}
# Bolometer Resistance [Ohm]
R_BOLO = 0.008
# Readout noise fraction
R_FRAC = 0.5


@trait_docs
class SimMuMUXCrosstalk(Operator):
    """Simulate nonlinear muMUX crosstalk

    Simulate inductive and capacitive crosstalk in the readout:

    phase_target = phase_target(true) + chi*sin(phase_source - phase_target)

    """

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for optional detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional telescope flagging",
    )

    realization = Int(0, help="Realization ID")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_chi(self, det_target, det_source):
        """ Return the crosstalk strength parameter for a pair of detectors
        """
        # FIXME: implement _get_chi()
        # We probably want to draw a table of chi somewhere
        if det_target == det_source:
            chi = 0
        else:
            chi = 1e-3
        return chi

    def _temperature_to_squid_phase(
            self, input_signal, Phi0, dPhi0dT
    ):
        """ Translate temperature-valued signal into SQUID phase
        """
        output_signal = input_signal.copy()
        output_signal = Phi0 + input_signal * dPhi0dT
        return output_signal

    def _squid_phase_to_temperature(self, input_signal, dPhi0dT):
        """ Translate SQUID phase-valued signal into temperature
        """
        output_signal = input_signal / dPhi0dT
        return output_signal

    def _draw_Phi0(self, obs, focalplane, detectors, vmin=0.3, vmax=1.3):
        """ Draw initial SQUID phases from a flat distribution
        """
        Phi0 = {}
        for det in detectors:
            # randomize Phi0 in a reproducible manner
            counter1 = obs.session.uid
            counter2 = self.realization
            key1 = focalplane[det]["uid"]
            key2 = 234561

            x = toast.rng.random(
                1,
                sampler="uniform_01",
                key=(key1, key2),
                counter=(counter1, counter2),
            )[0]
            v = vmin + x * (vmax - vmin)
            Phi0[det] = v

        return Phi0

    def _evaluate_dPhi0dT(self, obs, signal, detectors, rows, Phi0):
        """ Estimate how the SQUID phase in each detector changes
        with the sky temperature
        """
        focalplane = obs.telescope.focalplane
        bandpass = focalplane.bandpass

        if self.shared_flags is not None:
            common_good = (
                obs.shared[self.shared_flags].data & self.shared_flag_mask
            ) == 0
        else:
            common_good = np.ones(obs.n_local_samples, dtype=bool)

        dPhi0dT = {}
        for row, det in zip(rows, detectors):
            band = focalplane[det]["band"]
            cfreq = bandpass.center_frequency(det) * 1e-9  # GHz
            det_flags = obs.detdata[self.det_flags][det]
            good = np.logical_and(
                common_good, (det_flags & self.det_flag_mask) == 0
            )
            #import pdb
            #pdb.set_trace()
            median_signal = np.median(signal[row][good])
            P_opt = P_OPT[band] * 1e-12  # W
            P_atm_ref = P_ATM[band] * 1e-12  # W
            efficiency = ETA_ATM[band]
            P_sat = P_SAT[band] * 1e-12  # W
            P_atm = efficiency * bandpass.optical_loading(det, median_signal)  # W
            P_opt += P_atm - P_atm_ref
            dPdT = bandpass.kcmb2w(det)  # K_CMB -> W
            dIdP = 1 / np.sqrt((P_sat - P_opt) * R_FRAC * R_BOLO)  # W -> A
            dPhi0dI = 1 / 9e-6  # A -> [rad]
            dPhi0dT[det] = dPdT * dIdP * dPhi0dI  # K_CMB -> [rad]

        return dPhi0dT

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            raise RuntimeError(
                "SimMuMUXCrosstalk cannot be run on subsets of detectors"
            )

        for obs in data.obs:
            # Get the original number of process rows in the observation
            proc_rows = obs.dist.process_rows

            if self.det_data not in obs.detdata:
                msg = f"Cannot apply crosstalk: {self.det_data} "
                msg += "does not exist in {obs.name}"
                raise RuntimeError(msg)

            # Redistribute the data. For crosstalk, each process requires
            # all detectors
            # Duplicate just the fields of the observation we will use
            temp_obs = obs.duplicate(
                times=self.times,
                meta=[],
                shared=[self.shared_flags],
                detdata=[self.det_data, self.det_flags],
                intervals=[],
            )
            temp_obs.redistribute(1, times=self.times, override_sample_sets=None)

            # Crosstalk the detector data
            det_data = temp_obs.detdata[self.det_data]
            detectors = det_data.keys()
            rows = det_data.indices(detectors)
            # Determine the units and potential scaling factor
            det_units = det_data.units
            det_scale = unit_conversion(det_units, u.K)
            focalplane = temp_obs.telescope.focalplane

            chis = pos_to_chi(focalplane, detectors)

            # Make a copy of the detector data in K_CMB
            input_data = det_data.data.copy() * det_scale
            output_data = det_data.data  # just a reference

            Phi0 = self._draw_Phi0(temp_obs, focalplane, detectors)
            dPhi0dT = self._evaluate_dPhi0dT(
                temp_obs, input_data, detectors, rows, Phi0
            )

            # For each detector-detector pair:
            #     Get crosstalk strength, chi
            #     Generate output data by mixing input data
            for row_target, det_target in zip(rows, detectors):
                crosstalk = np.zeros_like(input_data[row_target])
                target_squid_phase = self._temperature_to_squid_phase(
                    input_data[row_target],
                    Phi0[det_target],
                    dPhi0dT[det_target],
                )
                for row_source, det_source in zip(rows, detectors):
                    if (det_target, det_source) in chis:
                        chi = chis[(det_target, det_source)]
                    else:
                        continue
                    source_squid_phase = self._temperature_to_squid_phase(
                        input_data[row_source],
                        Phi0[det_source],
                        dPhi0dT[det_source],
                    )
                    crosstalk += chi * np.sin(
                        source_squid_phase - target_squid_phase
                    )

                # Translate crosstalk into temperature units and scale to
                # match input data
                output_data[row_target] += self._squid_phase_to_temperature(
                    crosstalk, dPhi0dT[det_target]
                ) / det_scale

            # Redistribute back
            temp_obs.redistribute(
                proc_rows,
                times=self.times,
                override_sample_sets=obs.dist.sample_sets,
            )

            # Copy data to original observation
            for det in obs.select_local_detectors():
                # Unit conversion does not preserve offset so we do it
                # explicitly here
                offset_old = np.median(obs.detdata[self.det_data][det])
                offset_new = np.median(temp_obs.detdata[self.det_data][det])
                obs.detdata[self.det_data][det] = (
                    temp_obs.detdata[self.det_data][det] - offset_new + offset_old
                )

            # Free data copy
            temp_obs.clear()
            del temp_obs

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.stokes_weights.requires()
        req["shared"].append(self.hwp_angle)
        req["detdata"].append(self.weights)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [
                self.det_data,
            ],
        }
        return prov

    def _accelerators(self):
        return list()
