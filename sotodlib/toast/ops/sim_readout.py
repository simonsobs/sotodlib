# Copyright (c) 2018-2023 Simons Observatory.
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
class SimReadout(Operator):
    """Simulate various readout-related systematics"""

    API = Int(0, help="Internal interface version for this operator")

    realization = Int(0, help="The noise realization index")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask_readout = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    glitch_rate = Quantity(1.0 * u.Hz, help="Glitch occurrence rate")

    glitch_amplitude_center = Quantity(
        1 * u.K, help="Center of Gaussian distribution for glitch amplitude"
    )

    glitch_amplitude_sigma = Quantity(
        1 * u.K, help="Width of Gaussian distribution for glitch amplitude"
    )

    simulate_glitches = Bool(True, help="Enable glitch simulation")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _add_glitches(self, obs_id, times, focalplane, signal, dets):
        """Simulate glitches"""
        if not self.simulate_glitches:
            return
        fsample = focalplane.sample_rate
        events_per_sample = self.glitch_rate.to_value(u.Hz) / fsample.to_value(u.Hz)
        nsample = times.size
        for det in dets:
            det_id = focalplane[det]["uid"]
            sig = signal[det]
            np.random.seed(
                int(obs_id + det_id  + self.realization) % 2**32
            )
            event_counts = np.random.poisson(events_per_sample, nsample)
            event_indices = np.argwhere(event_counts).ravel()
            nindex = event_indices.size
            # Drawing amplitudes for every affected sample is not the
            # same as drawing amplitudes for every event but it is fast.
            # If we routinely have more than one event occurring during
            # the same time stamp we are in trouble
            amplitudes = self.glitch_amplitude_center \
                         + np.random.randn(nindex) * self.glitch_amplitude_sigma
            events = amplitudes.to_value(signal.units) \
                     * event_counts[event_indices]
            sig[event_indices] += events
        return

    @function_timer
    def _add_jumps(self, obs_id, times, focalplane, signal, dets):
        """Simulate baseline jumps"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _fail_bolometers(self, obs_id, times, focalplane, signal, dets):
        """Simulate bolometers that fail to bias or otherwise are unusable"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _misidentify_channels(self, obs_id, times, focalplane, signal, dets):
        """Swap detector data between two confused readout channels"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "shared_flags", "det_data":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimReadout")

        for obs in data.obs:
            obs_id = obs.uid
            times = obs.shared[self.times].data
            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane
            signal = obs.detdata[self.det_data]
    
            self._add_glitches(obs_id, times, focalplane, signal, dets)
            self._add_jumps(obs_id, times, focalplane, signal, dets)
            self._fail_bolometers(obs_id, times, focalplane, signal, dets)
            self._misidentify_channels(obs_id, times, focalplane, signal, dets)
            
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.times, self.shared_flags],
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [],
            "detdata": [self.det_data],
        }
        return prov

    def _accelerators(self):
        return list()
