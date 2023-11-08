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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_chi(self, det_target, det_source):
        """ Return the crosstalk strength parameter for a pair of detectors
        """
        # FIXME: implement _get_chi()
        if det_target == det_source:
            chi = 0
        else:
            chi = 1e-3
        return chi

    def _temperature_to_rf_phase(self, input_signal):
        """ Translate temperature-valued signal into RF phase
        """
        # FIXME: implement _temperature_to_rf_phase()
        output_signal = input_signal.copy()
        return output_signal

    def _temperature_to_squid_phase(self, input_signal):
        """ Translate temperature-valued signal into SQUID phase
        """
        # FIXME: implement _temperature_to_squid_phase()
        output_signal = input_signal.copy()
        return output_signal

    def _rf_phase_to_temperature(self, input_signal):
        """ Translate RF phase-valued signal into temperature
        """
        # FIXME: implement _rf_phase_to_temperature()
        output_signal = input_signal.copy()
        return output_signal

    def _mix_detector_data(self, target_squid_phase, source_squid_phase, chi):
        """ Evaluate and return the additive crosstalk term
        """
        # FIXME: double-check this
        return chi * np.sin(source_squid_phase - target_squid_phase)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            raise RuntimeError(
                "SimMuMUXCrosstalk cannot be run on subsets of detectors"
            )

        for obs in data.obs:
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
                shared=[],
                detdata=[self.det_data],
                intervals=[],
            )
            temps_obs.redistribute(1, times=self.times, override_sample_sets=None)

            # Crosstalk the detector data
            det_data = temp_obs.det_data[self.det_data]
            detectors = det_data.keys()
            rows = det_data.indices(detectors)
            # Determine the units and potential scaling factor
            det_units = det_data.units
            det_scale = unit_conversion(det_units, u.K)
            focalplane = temp_obs.telescope.focalplane

            # Make a copy of the detector data in K_CMB
            input_data = det_data.data.copy() * det_scale
            output_data = det_data.data  # just a reference

            # For each detector-detector pair:
            #     Get crosstalk strength, chi
            #     Generate output data by mixing input data
            for row_target, det_target in zip(rows, detectors):
                target_rf_phase = self._temperature_to_rf_phase(
                    input_data[row_target]
                )
                target_squid_phase = self._temperature_to_squid_phase(
                    input_data[row_target]
                )
                for row_source, det_source in zip(rows, detectors):
                    chi = self._get_chi(det_target, det_source)
                    if chi == 0:
                        continue
                    source_squid_phase = self._temperature_to_squid_phase(
                        input_data[row_source]
                    )
                    target_rf_phase += chi * np.sin(
                        source_squid_phase - target_squid_phase
                    )

                # Translate output data into temperature units and scale to
                # match input data
                output_data[row_target] = self._rf_phase_to_temperature(
                    target_rf_phase
                ) / det_scale

            # Redistribute back
            temp_obs.redistribute(
                proc_rows, times=self.times, override_sample_sets=obs.dist.sample_set
            )

            # Copy data to original observation
            obs.detdata[self.det_data][:] = temp_obs.detdata[self.det_data][:]

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
