# Copyright (c) 2018-2024 Simons Observatory.
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
from toast.timing import function_timer, Timer
from toast.traits import (Bool, Float, Instance, Int, Quantity, Unicode,
                          trait_docs)
from toast.utils import Environment, Logger, unit_conversion

XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimWireGrid(Operator):
    """Simulate wiregrid calibration signal"""

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask_wiregrid = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    shared_flag_mask_unstable = Int(
        255, help="Bit mask value applied during wire grid rotation"
    )

    wiregrid_angle = Unicode(
        "wiregrid_angle", help="Observation shared key for wiregrid angle"
    )

    wiregrid_angle_start = Quantity(
        0 * u.degree, help="Starting orientation of the wiregrid"
    )

    wiregrid_step_size = Quantity(22.5 * u.degree, help="Step between orientations")

    wiregrid_step_length = Quantity(10 * u.s, help="Time between orientations")

    wiregrid_angular_speed = Quantity(180 * u.degree / u.second, help="Rotation speed")

    wiregrid_angular_acceleration = Quantity(
        180 * u.degree / u.second**2, help="Rotation acceleration"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    detector_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector weights",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _get_wiregrid_angle(self, obs):
        """Simulate the motion of the wiregrid and raise appropriate telescope flags"""
        times = obs.shared[self.times].data

        comm = obs.comm_row
        if comm is None or comm.size == 1:
            times_tot = times
        else:
            times_tot = np.hstack(comm.allgather(times))

        t_accel = self.wiregrid_angular_speed / self.wiregrid_angular_acceleration
        dist_accel = 0.5 * self.wiregrid_angular_acceleration * t_accel**2
        if 2 * dist_accel < self.wiregrid_step_size:
            # wiregrid reaches coasting rate between acceleration and deceleration
            dist_coast = self.wiregrid_step_size - 2 * dist_accel
            t_coast = dist_coast / self.wiregrid_angular_speed
        else:
            dist_accel = self.wiregrid_step_size / 2
            t_accel = np.sqrt(2 * dist_accel / self.wiregrid_angular_acceleration)
            t_coast = 0 * u.second
        dist_coast = t_coast * self.wiregrid_angular_speed

        t_static = self.wiregrid_step_length.to_value(u.second)
        t_accel = t_accel.to_value(u.second)
        t_coast = t_coast.to_value(u.second)

        # Simulate the wiregrid angle for the entire observation,
        # one step at a time
        angles = np.zeros(times_tot.size)
        flags = (
            np.zeros(times_tot.size, dtype=np.uint8) + self.shared_flag_mask_wiregrid
        )
        t = times_tot[0]
        istep = 0
        sample = 0
        angle = self.wiregrid_angle_start
        while t < times_tot[-1]:
            # Stationary
            ind = np.logical_and(times_tot >= t, times_tot < t + t_static)
            angles[ind] = angle.to_value(u.radian)
            t += t_static

            # Accelerating
            ind = np.logical_and(times_tot >= t, times_tot < t + t_accel)
            t_ind = (times_tot[ind] - t) * u.second
            angles[ind] = (
                angle + 0.5 * self.wiregrid_angular_acceleration * t_ind**2
            ).to_value(u.radian)
            flags[ind] |= self.shared_flag_mask_unstable
            t += t_accel
            angle += dist_accel
            if t_coast > 0:
                rate = self.wiregrid_angular_speed

                # Coasting
                ind = np.logical_and(times_tot >= t, times_tot < t + t_coast)
                t_ind = (times_tot[ind] - t) * u.second
                angles[ind] = (angle + self.wiregrid_angular_speed * t_ind).to_value(
                    u.radian
                )
                flags[ind] |= self.shared_flag_mask_unstable
                t += t_coast
                angle += dist_coast
            else:
                rate = t_accel * u.second * self.wiregrid_angular_acceleration

            # Decelerating
            ind = np.logical_and(times_tot >= t, times_tot < t + t_accel)
            t_ind = (times_tot[ind] - t) * u.second
            angles[ind] = (
                angle
                + rate * t_ind
                - 0.5 * self.wiregrid_angular_acceleration * t_ind**2
            ).to_value(u.radian)
            flags[ind] |= self.shared_flag_mask_unstable
            t += t_accel
            angle += dist_accel

            # Start the next step
            istep += 1

        # Extract the local piece of the simulated angle
        n_sample = obs.n_local_samples
        istart = obs.local_index_offset
        ind = slice(istart, istart + n_sample)
        angles = angles[ind]
        flags = flags[ind]

        # Deposit in the Observation
        obs.shared.create_column(
            self.wiregrid_angle, shape=(n_sample,), dtype=np.float64
        )
        obs.shared[self.wiregrid_angle].set(angles, offset=(0,), fromrank=0)
        obs.shared[self.shared_flags].data |= flags

        return

    def get_wiregrid_signal(self, obs, band):
        wiregrid_angle = obs.shared[self.wiregrid_angle].data

        # FIXME: figure out the actual amplitude of the signal in K_CMB

        amplitude = 1

        I = amplitude * np.ones_like(wiregrid_angle)
        Q = amplitude * np.cos(2 * wiregrid_angle)
        U = amplitude * np.sin(2 * wiregrid_angle)

        return I, Q, U

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "detector_pointing", "detector_weights", "shared_flags":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimWireGrid")

        for obs in data.obs:
            if "WIREGRID" not in obs.name.upper():
                log.debug_rank(
                    f"SimWireGrid: {obs.name} is not a wiregrid calibration observation.",
                    comm=obs.comm.comm_group,
                )
                continue
            self._get_wiregrid_angle(obs)

            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane
            for det in dets:
                signal = obs.detdata[self.det_data][det]

                # Compute detector quaternions and Stokes weights
                obs_data = Data(comm=data.comm)
                obs_data._internal = data._internal
                obs_data.obs = [obs]
                self.detector_pointing.apply(obs_data, detectors=[det])
                self.detector_weights.apply(obs_data, detectors=[det])
                obs_data.obs.clear()
                del obs_data

                band = focalplane[det]["band"]

                I, Q, U = self.get_wiregrid_signal(obs, band)

                weights = obs.detdata[self.detector_weights.weights][det]
                weight_mode = self.detector_weights.mode
                if "I" in weight_mode:
                    ind = weight_mode.index("I")
                    weights_I = weights[:, ind].copy()
                else:
                    weights_I = 0
                if "Q" in weight_mode:
                    ind = weight_mode.index("Q")
                    weights_Q = weights[:, ind].copy()
                else:
                    weights_Q = 0
                if "U" in weight_mode:
                    ind = weight_mode.index("U")
                    weights_U = weights[:, ind].copy()
                else:
                    weights_U = 0

                signal += scale * (I * weights_I + Q * weights_Q + U * weights_U)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["shared"].append(self.times)
        req["detdata"].append(self.weights)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [self.wiregrid_angle],
            "detdata": [self.det_data],
        }
        return prov

    def _accelerators(self):
        return list()
