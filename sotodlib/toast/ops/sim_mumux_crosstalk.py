# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import re
import numpy as np
import toast.rng
import toast.qarray as qa
import toast.ops
from astropy import units as u
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer, Timer
from toast.traits import Bool, Instance, Int, Unicode, trait_docs
from toast.utils import Environment, Logger, unit_conversion

try:
    # NB: Requires specific version of JBolo
    # https://github.com/kmharrington/jbolo
    import jbolo.jbolo_funcs as jf
    from jbolo.utils import load_sim

    jbolo_available = True
except:
    jbolo_available = False

from .mumux_crosstalk_util import detmap_available, pos_to_chi

# JBolo sims to use
JBOLO_MODELS = {
    "SAT_LF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V3r7_JBolo/V3r7_Baseline/SAT/V3r7_Baseline_SAT_LF.yaml"
    ),
    "SAT_MF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/SAT/V4r0_Baseline_SAT_MF.yaml"
    ),
    "SAT_UHF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/SAT/V4r0_Baseline_SAT_UHF.yaml"
    ),
    "LAT_LF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_LF.yaml"
    ),
    "LAT_MF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_MF.yaml"
    ),
    "LAT_UHF": os.path.expandvars(
        "$JBOLO_MODELS_PATH/V4r0/V4r0_Baseline/LAT/V4r0_Baseline_LAT_UHF.yaml"
    ),
}

JBOLO_CHANNELS = {
    "SAT_f030": "LF_1",
    "SAT_f040": "LF_2",
    "SAT_f090": "MF_1",
    "SAT_f150": "MF_2",
    "SAT_f230": "UHF_1",
    "SAT_f290": "UHF_2",
    "LAT_f030": "LF_1",
    "LAT_f040": "LF_2",
    "LAT_f090": "MF_1",
    "LAT_f150": "MF_2",
    "LAT_f230": "UHF_1",
    "LAT_f290": "UHF_2",
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

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator for det Az/El pointing.  If None, boresight elevation is used.",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
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

    # The name of the temporary, scaled, detector data field.
    _temp_detdata_name = "temp_umux_crosstalk_input"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _temperature_to_squid_phase(self, input_signal, Phi0, dPhi0dT):
        """Translate temperature-valued signal into SQUID phase"""
        return Phi0 + input_signal * dPhi0dT

    def _squid_phase_to_temperature(self, input_signal, dPhi0dT):
        """Translate SQUID phase-valued signal into temperature"""
        return input_signal / dPhi0dT

    def _draw_Phi0(self, obs, focalplane, detectors, vmin=0.0, vmax=1.0):
        """Draw initial SQUID phases from a flat distribution"""
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

    def _evaluate_dPhi0dT(self, data, detectors, Phi0, pwv, boresight_el):
        """Estimate how the SQUID phase in each detector changes
        with the sky temperature
        """
        log = Logger.get()
        # The input data passed to this function only has one observation.
        obs = data.obs[0]

        focalplane = obs.telescope.focalplane
        bandpass = focalplane.bandpass

        if self.shared_flags is not None:
            common_good = (
                obs.shared[self.shared_flags].data & self.shared_flag_mask
            ) == 0
        else:
            common_good = np.ones(obs.n_local_samples, dtype=bool)

        # Are we using the boresight or per-detector elevation?
        if boresight_el is None:
            # Per detector elevation
            use_det_el = True
        else:
            use_det_el = False
            # We only need to compute one set of properties per band.  Cache
            # these and re-use them as needed.
            P_opts = {}
            P_sats = {}

        dPhi0dT = {}
        for det in detectors:
            raw_band = focalplane[det]["band"]
            if raw_band == "DARK" or raw_band[0] == "f":
                # We are using real data with bands like "f090", "f150", etc
                # Determine the telescope type from the name.
                wafer_band = focalplane[det]["det_info:wafer:bandpass"]
                if re.match(r"^sat.*", obs.telescope.name) is not None:
                    band = f"SAT_{wafer_band}"
                else:
                    band = f"LAT_{wafer_band}"
            else:
                # Synthetic observation
                band = raw_band

            det_flags = obs.detdata[self.det_flags][det]
            good = np.logical_and(common_good, (det_flags & self.det_flag_mask) == 0)

            jbolo_channel = JBOLO_CHANNELS[band]
            jbolo_model = JBOLO_MODELS[band[:4] + jbolo_channel.split("_")[0]]

            if use_det_el:
                # We are computing properties for every detector
                self.detector_pointing.apply(data, detectors=[det])
                _, detel, _ = qa.to_lonlat_angles(
                    obs.detdata[self.detector_pointing.quats][det]
                )
                elevation = np.degrees(np.median(detel[good]))
                jsim = load_sim(jbolo_model)
                jsim["sources"]["atmosphere"]["elevation"] = elevation  # Degrees
                jsim["sources"]["atmosphere"]["pwv"] = int(pwv)  # Microns
                jf.run_optics(jsim)
                jf.run_bolos(jsim)
                P_opt = float(jsim["outputs"][jbolo_channel]["P_opt"])
                P_sat = float(jsim["outputs"][jbolo_channel]["P_sat"])
            else:
                # We are using the boresight elevation
                elevation = boresight_el
                if band not in P_opts.keys():
                    # Compute detector properties using JBolo
                    jsim = load_sim(jbolo_model)
                    jsim["sources"]["atmosphere"]["elevation"] = elevation  # Degrees
                    jsim["sources"]["atmosphere"]["pwv"] = int(pwv)  # Microns
                    jf.run_optics(jsim)
                    jf.run_bolos(jsim)
                    P_opts[band] = float(jsim["outputs"][jbolo_channel]["P_opt"])  # W
                    P_sats[band] = float(jsim["outputs"][jbolo_channel]["P_sat"])  # W

                # Pull from cached values
                P_opt = P_opts[band]
                P_sat = P_sats[band]

            # Compute conversion
            dPdT = bandpass.kcmb2w(det)  # K_CMB -> W
            R_TES = R_FRAC * R_BOLO

            # Check for saturated detectors, flag if so.  I_TES will also
            # be NaN in that case.
            if P_sat < P_opt:
                msg = f"{obs.name}, det {det} is saturated, cutting"
                log.debug(msg)
                obs.detdata[self.det_flags][det] |= self.det_flag_mask
                obs.update_local_detector_flags({det: self.det_mask})

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
        if (
            "JBOLO_PATH" not in os.environ.keys()
            or "JBOLO_MODELS_PATH" not in os.environ.keys()
        ):
            raise RuntimeError(
                "Cannot calculate detector parameters -- no JBolo models available"
            )

        for obs in data.obs:
            # Get the original number of process rows in the observation.  We
            # use this when redistributing back to the original distribution.
            proc_rows = obs.dist.process_rows

            if self.det_data not in obs.detdata:
                msg = f"Cannot apply crosstalk: {self.det_data} "
                msg += "does not exist in {obs.name}"
                raise RuntimeError(msg)

            log.debug_rank(f"Crosstalk sim begin {obs.name}", comm=obs.comm.comm_group)

            # Before redistributing the data, compute any global values that are
            # constant over the whole observation.
            if self.detector_pointing is None:
                # We are using the boresight elevation
                if self.shared_flags is not None:
                    good = (
                        obs.shared[self.shared_flags].data & self.shared_flag_mask
                    ) == 0
                else:
                    good = np.ones(obs.n_local_samples, dtype=bool)
                boresight_el = np.degrees(
                    np.median(obs.shared[defaults.elevation].data[good])
                )
            else:
                boresight_el = None

            # Make a copy of this observation and redistribute it so that each
            # process has all detectors for a slice of time.  Duplicate just the fields
            # of the observation we will use.  Put this temporary obs in a Data object
            # so we can use other operators on it.
            shared_fields = [self.shared_flags]
            if self.detector_pointing is not None:
                shared_fields.append(self.detector_pointing.boresight)
            temp_obs = obs.duplicate(
                times=self.times,
                meta=[],
                shared=shared_fields,
                detdata=[self.det_data, self.det_flags],
                intervals=[],
            )
            temp_obs.redistribute(1, times=self.times, override_sample_sets=None)
            temp_data = Data(comm=data.comm)
            temp_data.obs.append(temp_obs)

            # The valid detectors we will consider.
            good_dets = temp_obs.select_local_detectors(
                selection=detectors, flagmask=self.det_mask
            )

            # Focalplane for this observation
            focalplane = temp_obs.telescope.focalplane

            # Compute the cross talk amplitudes
            chis = pos_to_chi(focalplane, good_dets)

            # Find the scaling from the original detector units to K_CMB
            det_data = temp_obs.detdata[self.det_data]
            det_units = det_data.units
            det_scale = unit_conversion(det_units, u.K)

            # Make a scaled copy of the original detector data, so that we can modify
            # the output.
            toast.ops.Copy(detdata=[(self.det_data, self._temp_detdata_name)]).apply(
                temp_data
            )
            toast.ops.CalibrateDetectors(
                det_data=self._temp_detdata_name, cal_value=det_scale, cal_units=u.K
            ).apply(temp_data)
            input_data = temp_obs.detdata[self._temp_detdata_name]

            # Get observation parameters for dPhi0dT calculation
            pwv = obs.telescope.site.weather.pwv.to_value(u.um)

            Phi0 = self._draw_Phi0(temp_obs, focalplane, good_dets)
            dPhi0dT = self._evaluate_dPhi0dT(
                temp_data, good_dets, Phi0, pwv, boresight_el
            )

            # For each detector-detector pair:
            #     Get crosstalk strength, chi
            #     Generate output data by mixing input data
            for det_target in good_dets:
                crosstalk = np.zeros_like(input_data[det_target])
                target_squid_phase = self._temperature_to_squid_phase(
                    input_data[det_target],
                    Phi0[det_target],
                    dPhi0dT[det_target],
                )
                first_samp = temp_obs.local_index_offset
                last_samp = first_samp + temp_obs.n_local_samples
                det_msg = f"{det_target}[{first_samp}:{last_samp}] "
                det_msg += f"(phi0={Phi0[det_target]:0.2e}, "
                det_msg += f"dphi0dT={dPhi0dT[det_target]:0.2e})"
                for det_source in good_dets:
                    if (det_target, det_source) in chis:
                        chi = chis[(det_target, det_source)]
                    else:
                        # No contribution from this source detector
                        continue
                    source_squid_phase = self._temperature_to_squid_phase(
                        input_data[det_source],
                        Phi0[det_source],
                        dPhi0dT[det_source],
                    )

                    # Add crosstalk if not collided resonator
                    if not np.isnan(chi):
                        crosstalk += chi * np.sin(
                            source_squid_phase - target_squid_phase
                        )
                        det_msg += f"\n  {det_source} (phi0={Phi0[det_target]:0.2e}, "
                        det_msg += f"dphi0dT={dPhi0dT[det_target]:0.2e}) "
                        det_msg += f"chi = {chi:0.2e}"
                    else:
                        # If collided, flag both detectors
                        msg = f"{obs.name}: collision, cutting {det_target} "
                        msg += f"and {det_source}"
                        log.debug(msg)
                        temp_obs.update_local_detector_flags(
                            {
                                det_source: self.det_mask,
                                det_target: self.det_mask,
                            },
                        )
                        temp_obs.detdata[self.det_flags][det_target] |= (
                            self.det_flag_mask
                        )
                        temp_obs.detdata[self.det_flags][det_source] |= (
                            self.det_flag_mask
                        )
                log.debug(det_msg)

                # Translate crosstalk into temperature units and scale to
                # match input data
                det_data[det_target] += (
                    self._squid_phase_to_temperature(crosstalk, dPhi0dT[det_target])
                    / det_scale
                )

            # Redistribute back
            temp_obs.redistribute(
                proc_rows,
                times=self.times,
                override_sample_sets=obs.dist.sample_sets,
            )

            # Copy data to original observation
            for det in good_dets:
                # Unit conversion does not preserve offset so we do it
                # explicitly here
                offset_old = np.median(obs.detdata[self.det_data][det])
                offset_new = np.median(temp_obs.detdata[self.det_data][det])
                # Propagate data
                obs.detdata[self.det_data][det] = (
                    temp_obs.detdata[self.det_data][det] - offset_new + offset_old
                )
                # Propagate flags.
                obs.detdata[self.det_flags][det] |= temp_obs.detdata[self.det_flags][
                    det
                ]
                obs.update_local_detector_flags(temp_obs.local_detector_flags)

            # Free data copy
            temp_obs.clear()
            del temp_obs
            del temp_data

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.times],
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.detector_pointing is not None:
            req.update(self.detector_pointing.requires())
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
