# Copyright (c) 2018-2022 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import pickle

import ephem
import h5py
import healpy as hp
import numpy as np
import traitlets
from astropy import constants
from astropy import units as u
from scipy.constants import au as AU
from scipy.interpolate import RectBivariateSpline
from toast import qarray as qa
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer
from toast.traits import (Bool, Float, Instance, Int, Quantity, Unicode,
                          trait_docs)
from toast.utils import Environment, Logger, Timer, unit_conversion

XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimStimulator(Operator):
    """Simulate stimulator calibration signal"""

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask_stimulator = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    shared_flag_mask_unstable = Int(
        255, help="Bit mask value applied during stimulator calibration"
    )

    chopper_rates = Unicode(
        "7,16,35,65,100,125,150",
        help="Chopper modulation rates as comma-separated values [Hz]",
    )

    chopper_blade_count = Int(4, help="Number of chopper blades")

    chopper_blade_width = Quantity(45 * u.degree, help="Chopper blade_width")

    chopper_acceleration = Quantity(
        300 * u.radian / u.s ** 2, help="Angular acceleration of the chopper"
    )

    chopper_step_time = Quantity(60 * u.s, help="Single chopper frequency step length")

    heater_temperature = Quantity(77 * u.mK, help="Stimulator target temperature")

    heater_aperture_diameter = Quantity(44 * u.mm, help="Heater aperture at chopper")

    heater_aperture_distance = Quantity(70.5 * u.mm, help="Heater aperture from chopper axis")

    blackbody_temperature = Quantity(17 * u.mK, help="Chopper temperature")

    chopper_state = Unicode("chopper_state", help="Observation key for chopper state")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    stimulator_temperature = Unicode(
        "stimulator", help="Observation shared key for stimulator temperature"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _get_chopper_state(self, obs):
        """Chopper modulates the stimulator signal"""

        my_times = obs.shared[self.times].data

        comm = obs.comm_row
        if comm is None or comm.size == 1:
            times_tot = my_times
        else:
            times_tot = np.hstack(comm.allgather(my_times))

        # Evaluate the fraction of heater aperture that is visible
        # as a function of chopper angle

        d = self.heater_aperture_distance.to_value(u.m)
        R = self.heater_aperture_diameter.to_value(u.m) / 2
        D = np.sqrt(d ** 2 - R ** 2)
        period = 2 * np.pi / self.chopper_blade_count

        # Half of the aperture angle measured from the chopper spin axis
        alpha = np.arcsin(R / d)

        # Chopper blade width
        gamma = self.chopper_blade_width.to_value(u.radian)

        def get_ratio(beta):
            """Return the fraction of covered aperture.

            Beta measures the overlap between the blade and the aperture.
            Defined only for beta=[0,alpha]. Ratio(alpha)=0.5
            """
            tanbeta = np.tan(beta)
            c = 2 * np.sqrt(
                (D + R * tanbeta) ** 2 / (1 + tanbeta ** 2) - D ** 2
            )
            theta = 2 * np.arcsin(0.5 * c / R)
            ratio = 0.5 * (theta - np.sin(theta)) / np.pi
            return ratio

        # One full cycle
        all_beta = []
        all_ratio = []
        n = 1000
        eps = 1e-6  # avoid singularities

        # Blade covers first half of the aperture
        beta = np.linspace(0, alpha, n, endpoint=False)
        ratio = get_ratio(beta)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # Blade covers second half of the aperture
        beta = np.linspace(alpha + eps, 2 * alpha, n, endpoint=False)
        ratio = 1 - get_ratio(2 * alpha - beta)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # Blade fully covers aperture
        beta = np.linspace(2 * alpha, gamma, n, endpoint=False)
        ratio = np.ones_like(beta)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # Blade uncovers first half of the aperture
        beta = np.linspace(gamma, gamma + alpha, n, endpoint=False)
        ratio = 1 - get_ratio(beta - gamma)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # Blade uncovers second half of the aperture
        beta = np.linspace(gamma + alpha + eps, gamma + 2 * alpha, n, endpoint=False)
        ratio = get_ratio(gamma + 2 * alpha - beta)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # No coverage
        beta = np.linspace(gamma + 2 * alpha, 2 * np.pi / self.chopper_blade_count, n)
        ratio = np.zeros_like(beta)
        all_beta.append(beta)
        all_ratio.append(ratio)

        # Concatenate
        beta = np.hstack(all_beta)
        ratio = np.hstack(all_ratio)

        # Interpolate
        # phase = np.linspace(0, 2 * np.pi, 1000)
        # illumination = np.interp(phase % period, beta, ratio)

        # Now simulate the phase

        current_phase = 0
        current_rate = 0
        current_time = times_tot[0]
        all_times = []
        all_phases = []
        unstable_periods = []
        n = 100
        rates = []
        for rate in self.chopper_rates.split(","):
            rates.append(float(rate) / self.chopper_blade_count * 2 * np.pi)
        accel = self.chopper_acceleration.to_value(u.radian / u.second ** 2)
        for rate in rates:
            # Acceleration/deceleration
            sign = np.sign(rate - current_rate)
            time_accel = sign * (rate - current_rate) / accel
            times = np.linspace(0, time_accel, n, endpoint=False)
            all_times.append(times + current_time)
            all_phases.append(
                0.5 * sign * accel * times ** 2 + current_rate * times + current_phase
            )
            unstable_periods.append([current_time, current_time + time_accel])
            current_phase += 0.5 * sign * accel * time_accel ** 2 + current_rate * time_accel
            current_time += time_accel
            current_rate = rate
            # Coasting
            time_coast = self.chopper_step_time.to_value(u.s) - time_accel
            times = np.linspace(0, time_coast, n, endpoint=False)
            all_times.append(times + current_time)
            all_phases.append(current_rate * times + current_phase)
            current_time += time_coast
            current_phase += current_rate * time_coast

        # Add last data points to support extrapolation and concatenate

        all_times.append([current_time, current_time + 1e6])
        all_phases.append([current_phase, current_phase])
        phase_times = np.hstack(all_times)
        phases = np.hstack(all_phases)

        my_phase = np.interp(my_times, phase_times, phases)
        my_state = np.interp(my_phase % period, beta, ratio)

        # Deposit in the Observation

        obs.shared.create_column(
            self.chopper_state, shape=(obs.n_local_samples,), dtype=np.float64
        )
        obs.shared[self.chopper_state].set(my_state, offset=(0,), fromrank=0)

        flags = np.zeros(obs.n_local_samples, dtype=np.uint8) \
            + self.shared_flag_mask_stimulator

        for start, stop in unstable_periods:
            ind = np.logical_and(my_times > start, my_times < stop)
            flags[ind] |= self.shared_flag_mask_unstable

        obs.shared[self.shared_flags].data |= flags

        return

    @function_timer
    def _get_stimulator_temperature(self, obs):
        """Time-dependent stimulator temperature"""
        times = obs.shared[self.times].data

        # Very simple model, constant temperature throughout the observation,

        temperature = np.ones(obs.n_local_samples) * self.heater_temperature.to_value(u.K)

        # Deposit in the Observation
        obs.shared.create_column(
            self.stimulator_temperature, shape=(obs.n_local_samples,), dtype=np.float64
        )
        obs.shared[self.stimulator_temperature].set(temperature, offset=(0,), fromrank=0)

        return

    def _get_stimulator_signal(self, obs, band, quat):

        # FIXME: figure out the actual amplitude of the signal in K_CMB

        chopper = obs.shared[self.chopper_state].data
        temperature = obs.shared[self.stimulator_temperature].data

        # This is placeholder code.  Proper calculation should account for
        # detector bandpass, location on the focalplane and possibly for
        # the chopper temperature

        signal = temperature * chopper + self.blackbody_temperature.to_value(u.K)

        return signal

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "shared_flags", :
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimStimulator")

        for obs in data.obs:
            if "STIMULATOR" not in obs.name.upper():
                log.debug_rank(
                    f"SimStimulator: {obs.name} is not a stimulator calibration observation.",
                    comm=obs.comm.comm_group,
                )
                continue
            self._get_chopper_state(obs)
            self._get_stimulator_temperature(obs)

            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane
            for det in dets:
                signal = obs.detdata[self.det_data][det]
                band = focalplane[det]["band"]
                quat = focalplane[det]["quat"]

                stim = self._get_stimulator_signal(obs, band, quat)

                signal += scale * stim

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["shared"].append(self.times)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [self.chopper_state, self.stimulator_temperature],
            "detdata": [self.det_data],
        }
        return prov

    def _accelerators(self):
        return list()
