# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import numpy as np
import toast.rng
from astropy import units as u
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer, Timer
from toast.traits import Bool, Int, Unicode, trait_docs
from toast.utils import Environment, Logger, unit_conversion

try:
    import jbolo.jbolo_funcs as jf
    from jbolo.utils import load_sim
    jbolo_available = True
except:
    jbolo_available = False

from .mumux_crosstalk_util import detmap_available, pos_to_chi

# JBolo sims to use
JBOLO_MODELS = {
    'SAT_LF' : os.path.expandvars("$JBOLO_MODELS_PATH/V3r7_JBolo/V3r7_Baseline/SAT/V3r7_Baseline_SAT_LF.yaml"),
    'SAT_MF' : os.path.expandvars("$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/SAT/V4r0_Baseline_SAT_MF.yaml"),
    'SAT_UHF' : os.path.expandvars("$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/SAT/V4r0_Baseline_SAT_UHF.yaml"),
    'LAT_LF' : os.path.expandvars("$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_LF.yaml"),
    'LAT_MF' : os.path.expandvars("$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_MF.yaml"),
    'LAT_UHF' : os.path.expandvars("$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_UHF.yaml")
}

JBOLO_CHANNELS = {
    "SAT_f030" : "LF_1",
    "SAT_f040" : "LF_2",
    "SAT_f090" : "MF_1",
    "SAT_f150" : "MF_2",
    "SAT_f230" : "UHF_1",
    "SAT_f290" : "UHF_2",
    "LAT_f030" : "LF_1",
    "LAT_f040" : "LF_2",
    "LAT_f090" : "MF_1",
    "LAT_f150" : "MF_2",
    "LAT_f230" : "UHF_1",
    "LAT_f290" : "UHF_2"
}

# Bolometer Resistance [Ohm]
R_BOLO = 0.008
# Readout noise fraction
R_FRAC = 0.5
# Shunt Resistance [Ohm]
R_SHUNT = 400e-6


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

    random_Phi0 = Bool(
        True,
        help="Draw new phase offsets for every observation and realization",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _draw_Phi0(self, obs, focalplane, detectors, vmin=0.0, vmax=1.0):
        """ Draw initial SQUID phases from a flat distribution
        """
        Phi0 = {}
        for det in detectors:
            # randomize Phi0 in a reproducible manner
            if self.random_Phi0:
                counter1 = obs.session.uid
                counter2 = self.realization
            else:
                counter1, counter2 = 0, 0
            key1 = focalplane[det]["uid"]
            key2 = 234561

            x = toast.rng.random(
                1,
                sampler="uniform_01",
                key=(key1, key2),
                counter=(counter1, counter2),
            )[0]
            v = vmin + x * (vmax - vmin)
            # Convert from Phi0 to Phi
            v *= 2 * np.pi
            Phi0[det] = v

        return Phi0

    def _evaluate_dPhi0dT(self, obs, signal, detectors, rows, Phi0, pwv, elevation):
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

        # Create dicts for JBolo values
        P_opts = {}
        P_atm_refs = {}
        efficiencies = {}
        P_sats = {}

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

            if band not in P_opts.keys():
                # Compute detector properties using JBolo
                jbolo_channel = JBOLO_CHANNELS[band]
                jbolo_model   = JBOLO_MODELS[band[:4] + jbolo_channel.split('_')[0]]
                jsim = load_sim(jbolo_model)
                jsim['sources']['atmosphere']['elevation'] = elevation # Degrees
                jsim['sources']['atmosphere']['pwv'] = int(pwv) # Microns
                jf.run_optics(jsim)
                jf.run_bolos(jsim)
                P_opts[band] = float(jsim['outputs'][jbolo_channel]['P_opt']) # W
                P_atm_refs[band] = float(jsim['outputs'][jbolo_channel]['sources']['atmosphere']['P_opt']) # W
                efficiencies[band] = float(jsim['outputs'][jbolo_channel]['sources']['atmosphere']['effic_cumul_avg'])
                P_sats[band] = float(jsim['outputs'][jbolo_channel]['P_sat']) # W

            # Pull from cached values
            P_opt = P_opts[band]
            P_atm_ref = P_atm_refs[band]
            efficiency = efficiencies[band]
            P_sat = P_sats[band]

            # Compute conversion
            P_atm = efficiency * bandpass.optical_loading(det, median_signal)  # W
            P_opt += P_atm - P_atm_ref
            dPdT = bandpass.kcmb2w(det)  # K_CMB -> W
            R_TES = R_FRAC * R_BOLO
            assert P_sat > P_opt # Check for saturated detectors
            I_TES = np.sqrt((P_sat - P_opt) / R_TES)
            dIdP = -1 / (I_TES * (R_TES - R_SHUNT))  # W -> A
            dPhi0dI = 1 / 9e-6  # A -> [rad]
            dPhidPhi0 = 2 * np.pi
            dPhi0dT[det] = dPdT * dIdP * dPhi0dI * dPhidPhi0  # K_CMB -> [rad]

        return dPhi0dT

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            raise RuntimeError(
                "SimMuMUXCrosstalk cannot be run on subsets of detectors"
            )

        # Check for JBOLO installation
        if not jbolo_available:
            raise RuntimeError(
                "Cannot calculate detector parameters -- no JBolo installation"
            )
        # Check for JBOLO data path
        if 'JBOLO_PATH' not in os.environ.keys() or 'JBOLO_MODELS_PATH' not in os.environ.keys():
            raise RuntimeError(
                "Cannot calculate detector parameters -- no JBolo models available"
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

            # Get observation parameters for dPhi0dT calculation
            pwv = obs.telescope.site.weather.pwv.to_value(u.um)
            el  = obs["scan_el"].to_value(u.degree)

            Phi0 = self._draw_Phi0(temp_obs, focalplane, detectors)
            dPhi0dT = self._evaluate_dPhi0dT(
                temp_obs, input_data, detectors, rows, Phi0, pwv, el
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
                    # Add crosstalk if not collided resonator
                    if not np.isnan(chi):
                        crosstalk += chi * np.sin(
                            source_squid_phase - target_squid_phase
                        )
                    else:
                        # Otherwise flag
                        temp_obs.detdata[self.det_flags][det_source] |= self.det_flag_mask
                        temp_obs.detdata[self.det_flags][det_target] |= self.det_flag_mask

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
                # Propagate data
                obs.detdata[self.det_data][det] = (
                    temp_obs.detdata[self.det_data][det] - offset_new + offset_old
                )
                # Propagate flags
                obs.detdata[self.det_flags][det] |= temp_obs.detdata[self.det_flags][det]

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
