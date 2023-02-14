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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _add_glitches(self, times, focalplane, signal, dets):
        """Simulate glitches"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _add_jumps(self, times, focalplane, signal, dets):
        """Simulate baseline jumps"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _fail_bolometers(self, times, focalplane, signal, dets):
        """Simulate bolometers that fail to bias or otherwise are unusable"""
        for det in dets:
            sig = signal[det]
        return

    @function_timer
    def _misidentify_channels(self, times, focalplane, signal, dets):
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
            times = obs.shared[self.times].data
            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane
            signal = obs.detdata[self.det_data]
    
            self._add_glitches(times, focalplane, signal, dets)
            self._add_jumps(times, focalplane, signal, dets)
            self._fail_bolometers(times, focalplane, signal, dets)
            self._misidentify_channels(times, focalplane, signal, dets)
            
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
