# Copyright (c) 2018-2024 Simons Observatory.
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

import toast.rng
from toast.timing import function_timer
from toast import qarray as qa
from toast.data import Data
from toast.traits import trait_docs, Float, Int, Unicode, Instance, Bool
from toast.ops.operator import Operator
from toast.utils import Logger, unit_conversion
from toast.observation import default_values as defaults


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimHWPSS(Operator):
    """Simulate HWP synchronous signal.

    NOTE:  The HWPSS template is interpolated (with a 5th order spline interpolation).
    Interpolation errors produce spurious peaks in the frequency domain which are 7 to
    8 orders of magnitude below the amplitude of the original signal.

    """

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps, only used for drift.",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, help="Observation shared key for HWP angle"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    atmo_data = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for simulated atmosphere "
        "(modulates part of the HWPSS)",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a Stokes weights operator",
    )

    fname_hwpss = Unicode(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data/hwpss_per_chi.pck",
        ),
        help="File containing measured or estimated HWPSS profiles",
    )

    drift_rate = Float(
        None,
        allow_none=True,
        help="If non-zero, the width of the Gaussian distribution to draw "
        "drift rate [1/hour] from.  All detectors will observe the same "
        "drift rate."
    )

    hwpss_random_drift = Bool(
        False,
        help="If True, the hwpss drift will be a random signal"
        "following a 1/f^alpha spectrum, and fully correlated between "
        "detectors."
    )

    hwpss_drift_alpha = Float(
        1.0,
        help="The power law exponent of the HWPSS random drift."
    )

    hwpss_drift_rms = Float(
        0.01,
        help="RMS of the relative HWPSS fluctuations."
    )

    hwpss_drift_coupling_center = Float(
        1.0,
        help="Mean coupling strength between the detectors and the HWPSS "
        "random drift mode."
    )

    hwpss_drift_coupling_width = Float(
        0.0,
        help="Width of the coupling strength distribution between the "
        "detectors and the HWPSS random drift mode."
    )

    realization = Int(0, help="Realization ID")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "stokes_weights",:
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
            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)
            det_units = obs.detdata[self.det_data].units
            det_scale = unit_conversion(u.K, det_units)

            focalplane = obs.telescope.focalplane
            fs = focalplane.sample_rate.to_value(u.Hz)

            if self.drift_rate is not None and self.drift_rate != 0:
                # Randomize the drift in a reproducible manner
                counter1 = obs.session.uid
                counter2 = self.realization
                key1 = 683584
                key2 = 476365
                x = toast.rng.random(
                    1,
                    sampler="gaussian",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                )[0]
                drift_rate = self.drift_rate * x
                # Translate to actual drift
                t = obs.shared[self.times].data
                tmean = np.mean(t)  # Assumes data distribution by detector
                drift = drift_rate * (t - tmean) / 3600
            elif self.hwpss_random_drift and self.hwpss_drift_alpha is not None:
                # Generate the HWPSS random drift common mode
                counter1 = obs.session.uid
                counter2 = self.realization
                key1 = 683584
                key2 = 476365
                nsamp = obs.shared[self.times].data.size
                w = toast.rng.random(
                    nsamp,
                    sampler="gaussian",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                ).array()
                freqs = np.fft.rfftfreq(nsamp, 1/fs)
                drift_psd = np.zeros_like(freqs)
                drift_psd[1:] = np.abs(1 / freqs[1:])**self.hwpss_drift_alpha
                drift = np.fft.irfft(np.fft.rfft(w) * np.sqrt(drift_psd))
                drift *= self.hwpss_drift_rms / np.std(drift)
            else:
                drift = 0

            # Get HWP angle
            chi = obs.shared[self.hwp_angle].data
            for det in dets:
                signal = obs.detdata[self.det_data][det]
                if self.atmo_data is None:
                    atmo = None
                else:
                    atmo = obs.detdata[self.atmo_data][det]
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
                det_theta, det_phi, det_psi = qa.to_iso_angles(det_quat)

                # Compute Stokes weights

                iweights = 1
                qweights = np.cos(2 * det_psi)
                uweights = np.sin(2 * det_psi)

                # Convert Az/El quaternion of the detector into elevation

                obs_data = Data(comm=data.comm)
                obs_data._internal = data._internal
                obs_data.obs = [obs]
                self.stokes_weights.apply(obs_data, detectors=[det])
                obs_data.obs.clear()
                del obs_data

                azel_quat = obs.detdata[
                    self.stokes_weights.detector_pointing.quats
                ][det]
                theta, phi, _ = qa.to_iso_angles(azel_quat)
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

                # HWPSS not from atmosphere

                transmission_wo_atmo = (
                    (1 - r) * self.all_stokes[freq]["transmission_wo_atmo"][itheta_low]
                    + r * self.all_stokes[freq]["transmission_wo_atmo"][itheta_high]
                )
                reflection_wo_atmo = (
                    (1 - r) * self.all_stokes[freq]["reflection_wo_atmo"][itheta_low]
                    + r * self.all_stokes[freq]["reflection_wo_atmo"][itheta_high]
                )
                # Thermal emission from the HWP is not driven by the atmosphere
                emission = (
                    (1 - r) * self.all_stokes[freq]["emission_wo_atmo"][itheta_low]
                    + r * self.all_stokes[freq]["emission_wo_atmo"][itheta_high]
                )

                # HWPSS from atmosphere

                transmission_atmo = (
                    (1 - r) * self.all_stokes[freq]["transmission_atmo"][itheta_low]
                    + r * self.all_stokes[freq]["transmission_atmo"][itheta_high]
                )
                reflection_atmo = (
                    (1 - r) * self.all_stokes[freq]["reflection_atmo"][itheta_low]
                    + r * self.all_stokes[freq]["reflection_atmo"][itheta_high]
                )

                if atmo is None:
                    transmission = transmission_wo_atmo + transmission_atmo
                    reflection = reflection_wo_atmo + reflection_atmo
                else:
                    transmission = transmission_wo_atmo
                    reflection = reflection_wo_atmo

                # Scale HWPSS for observing elevation

                el_ref = np.radians(50)
                scale = np.sin(el_ref) / np.sin(el)

                # Observe HWPSS with the detector

                iquv = transmission + reflection
                iquv = (iquv - np.mean(iquv.T, 1)).T
                iquss = (
                    iweights * splev(chi, splrep(self.chis, iquv[0], k=5)) +
                    qweights * splev(chi, splrep(self.chis, iquv[1], k=5)) +
                    uweights * splev(chi, splrep(self.chis, iquv[2], k=5))
                ) * scale

                if atmo is not None:
                    # Atmospheric HWPSS is modulated by the relative
                    # atmospheric fluctuation
                    modulation = atmo / np.median(atmo)
                    iquv = transmission_atmo + reflection_atmo
                    iquv = (iquv - np.mean(iquv.T, 1)).T
                    iquss += (
                        iweights * splev(chi, splrep(self.chis, iquv[0], k=5)) +
                        qweights * splev(chi, splrep(self.chis, iquv[1], k=5)) +
                        uweights * splev(chi, splrep(self.chis, iquv[2], k=5))
                    ) * scale * modulation

                iquv = (emission - np.mean(emission.T, 1)).T
                iquss += (
                    iweights * splev(chi, splrep(self.chis, iquv[0], k=5)) +
                    qweights * splev(chi, splrep(self.chis, iquv[1], k=5)) +
                    uweights * splev(chi, splrep(self.chis, iquv[2], k=5))
                )

                if self.hwpss_random_drift:
                    # Apply detector couplings to HWPSS random drift common mode
                    key1 = obs.telescope.uid
                    key2 = obs.session.uid
                    counter1 = self.realization
                    counter2 = focalplane[det]["uid"]
                    gaussian = toast.rng.random(
                        1,
                        sampler="gaussian",
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )[0]
                    coupling = (
                        self.hwpss_drift_coupling_center
                        + gaussian * self.hwpss_drift_coupling_width
                    )
                else:
                    coupling = 1.0

                # Co-add with the cached signal

                signal += det_scale * iquss * (1 + drift * coupling)

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
