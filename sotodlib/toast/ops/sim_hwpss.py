# Copyright (c) 2018-2021 Simons Observatory.
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
from scipy.interpolate import RectBivariateSpline

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from toast.ops.operator import Operator

from toast.utils import Environment, Logger, Timer

from toast.observation import default_values as defaults


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimHWPSS(Operator):
    """ Simulate HWP synchronous signal """

    API = Int(0, help="Internal interface version for this operator")

    hwp_angle = Unicode(
        defaults.hwp_angle, help="Observation shared key for HWP angle"
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

    fname_hwpss = Unicode(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data/hwpss_per_chi.pck",
        ),
        help="File containing measured or estimated HWPSS profiles",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "detector_pointing", :
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(
                    f"You must set `{trait}` before running SimHWPSS"
                )

        if not os.path.isfile(self.fname_hwpss):
            raise RuntimeError(f"{self.fname_hwpss} does not exist!")

        # theta is the incident angle, also known as the radial distance
        #     to the boresight.
        # chi is the HWP rotation angle

        with open(self.fname_hwpss, "rb") as fin:
            self.thetas, self.chis, self.all_stokes = pickle.load(fin)

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets)
            focalplane = obs.telescope.focalplane
            # Get HWP angle
            chi = obs.shared[self.hwp_angle].data
            for det in dets:
                signal = obs.detdata[self.det_data][det]
                band = focalplane[det]["band"]
                freq = {
                    "SAT_f030" : "027",
                    "SAT_f040" : "039",
                    "SAT_f090" : "093",
                    "SAT_f150" : "145",
                    "SAT_f230" : "225",
                    "SAT_f290" : "278",
                }[band]

                # Get incident angle

                det_quat = focalplane[det]["quat"]
                det_theta, det_phi, det_psi = qa.to_angles(det_quat)

                # Compute Stokes weights

                iweights = 1
                qweights = np.cos(2 * det_psi)
                uweights = np.sin(2 * det_psi)

                # Convert Az/El quaternion of the detector into elevation

                obs_data = Data(comm=data.comm)
                obs_data._internal = data._internal
                obs_data.obs = [obs]
                self.detector_pointing.apply(obs_data, detectors=[det])
                obs_data.obs.clear()
                del obs_data

                azel_quat = obs.detdata[
                    self.stokes_weights.detector_pointing.quats
                ][det]
                theta, phi = qa.to_position(azel_quat)
                el = np.pi / 2 - theta

                # Get polarization weights

                weights = obs.detdata[self.stokes_weights.weights][det]
                iweights, qweights, uweights = weights.T

                # Interpolate HWPSS to incident angle equal to the
                # radial distance from the focalplane (HWP) center

                theta_deg = np.degrees(det_theta)
                itheta_high = np.searchsorted(self.thetas, theta_deg)
                itheta_low = itheta_high - 1

                theta_low = self.thetas[itheta_low]
                theta_high = self.thetas[itheta_high]
                r = (theta_deg - theta_low) / (theta_high - theta_low)

                transmission = (
                    (1 - r) * self.all_stokes[freq]["transmission"][itheta_low]
                    + r * self.all_stokes[freq]["transmission"][itheta_high]
                )
                reflection = (
                    (1 - r) * self.all_stokes[freq]["reflection"][itheta_low]
                    + r * self.all_stokes[freq]["reflection"][itheta_high]
                )
                emission = (
                    (1 - r) * self.all_stokes[freq]["emission"][itheta_low]
                    + r * self.all_stokes[freq]["emission"][itheta_high]
                )

                # Scale HWPSS for observing elevation

                el_ref = np.radians(50)
                scale = np.sin(el_ref) / np.sin(el)

                # Observe HWPSS with the detector

                iquv = (transmission + reflection).T
                iquss = (
                    iweights * np.interp(chi, self.chis, iquv[0]) +
                    qweights * np.interp(chi, self.chis, iquv[1]) +
                    uweights * np.interp(chi, self.chis, iquv[2])
                ) * scale

                iquv = emission.T
                iquss += (
                    iweights * np.interp(chi, self.chis, iquv[0]) +
                    qweights * np.interp(chi, self.chis, iquv[1]) +
                    uweights * np.interp(chi, self.chis, iquv[2])
                )

                iquss -= np.median(iquss)

                # Co-add with the cached signal

                signal += iquss

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
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
