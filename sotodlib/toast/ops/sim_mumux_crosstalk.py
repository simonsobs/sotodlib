# Copyright (c) 2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import h5py
import os
import pickle

import traitlets

import numpy as np

from astropy import constants
from astropy import units as u

import ephem

import healpy as hp

from scipy.constants import au as AU
from scipy.interpolate import RectBivariateSpline, splrep, splev

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from toast.ops.operator import Operator

from toast.utils import Environment, Logger, Timer, unit_conversion

from toast.observation import default_values as defaults


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimMuMUXCrosstalk(Operator):
    """Simulate nonlinear muMUX crosstalk

    """

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "stokes_weights",:
            value = getattr(self, trait)
            if value is None:
                msg = f"You must set `{trait}` before running SimMuMUXCrosstalk"
                raise RuntimeError(msg)

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            det_scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane

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
