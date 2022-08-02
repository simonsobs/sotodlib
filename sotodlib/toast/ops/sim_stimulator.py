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
from toast.utils import Environment, Logger, Timer

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

    chopper_rates = Unicode("1,2,4", help="Chopper modulation rates as comma-separated values [Hz]")

    chopper_step_time = Quantity(60 * u.s, help="Single chopper frequency step length")

    heater_temperature = Quantity(77 * u.mK, help="Stimulator target temperature")

    blackbody_temperature = Quantity(17 * u.mK, help="Chopper temperature")

    chopper_state = Unicode("chopper_state", help="Observation key for chopper state")

    stimulator_temperature = Unicode(
        "stimulator_temperature", help="Observation key for chopper state"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def get_chopper_state(self, obs):
        """Chopper modulates the stimulator signal"""
        times = obs.shared[self.times].data

        comm = obs.comm_row
        if comm is None or comm.size == 1:
            times_tot = times
        else:
            times_tot = np.hstack(comm.allgather(times))

        fast_times = np.linspace(
            times_tot[0], times_tot[-1], times_tot.size * 10
        )
        fsample = 1 / (fast_times[1] - fast_times[0])
        step_len = int(self.chopper_step_time.to_value(u.s) * fsample)
        step_times = fast_times[:step_len] - fast_times[0]
        fast_phases = np.zeros(fast_times.size)
        last_phase = 0
        rates = [float(rate) for rate in self.chopper_rates.split(",")]
        for step, rate in enumerate(rates):
            istart = step * step_len
            istop = istart + step_len
            ind = slice(istart, istop)
            fast_phases[ind] = 2 * np.pi * rate * step_times + last_phase
            last_phase = fast_phases[istop - 1]

        state = (
            np.sin(np.interp(times, fast_times, fast_phases)) > 0
        ).astype(np.float64)

        # Deposit in the Observation
        obs.shared.create_column(
            self.chopper_state, shape=(obs.n_local_samples,), dtype=np.float64
        )
        obs.shared[self.chopper_state].set(state, offset=(0,), fromrank=0)

        return

    @function_timer
    def get_stimulator_temperature(self, obs):
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

    def get_stimulator_signal(self, obs, band, quat):

        # FIXME: figure out the actual amplitude of the signal in K_CMB

        chopper = obs.shared[self.chopper_state].data
        temperature = obs.shared[self.stimulator_temperature]

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
                    comm=obs.comm,
                )
                continue
            self.get_chopper_state(obs)
            self.get_stimulator_temperature(obs)

            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets)
            focalplane = obs.telescope.focalplane
            for det in dets:
                signal = obs.detdata[self.det_data][det]
                band = focalplane[det]["band"]
                quat = focalplane[det]["quat"]

                stim = self.get_stimulator_signal(obs, band, quat)

                signal += stim

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
