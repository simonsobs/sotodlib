# Copyright (c) 2018-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import h5py
import os
import pickle

import traitlets

import numpy as np

from astropy import units as u

import ephem

from scipy.interpolate import RectBivariateSpline

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from toast.ops.operator import Operator

from toast.utils import Environment, Logger, Timer

from toast.observation import default_values as defaults

from . import utils


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400.0 + 2440587.5


def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5


def to_DJD(t):
    # Convert Unix time stamp to Dublin Julian date
    # (days since 1899-12-31 12:00:00)
    # This is the time format used by PyEphem
    return to_JD(t) - 2415020


@trait_docs
class SimSSO(Operator):
    """Operator that generates Solar System Object timestreams.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    sso_name = Unicode(
        None,
        allow_none=True,
        help="Name of the SSO(s), must be recognized by pyEphem",
    )

    beam_file = Unicode(
        None,
        allow_none=True,
        help="Pickle file that stores the simulated beam",
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

    @traitlets.validate("sso_name")
    def _check_sso_name(self, proposal):
        sso_names = proposal["value"]
        try:
            for sso_name in sso_names.split(","):
                sso = getattr(ephem, sso_name)()
        except AttributeError:
            raise traitlets.TraitError(f"{sso_name} is not a valid SSO name")
        return sso_names

    @traitlets.validate("beam_file")
    def _check_beam_file(self, proposal):
        beam_file = proposal["value"]
        if not os.path.isfile(beam_file):
            raise traitlets.TraitError(f"{beam_file} is not a valid beam file")
        return beam_file

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

        for trait in "sso_name", "beam_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(
                    f"You must set `{trait}` before running SimSSO"
                )

        for sso_name in self.sso_name.split(","):
            sso = getattr(ephem, sso_name)()

            timer = Timer()
            timer.start()

            if data.comm.group_rank == 0:
                log.debug(f"{data.comm.group} : Simulating {sso_name}")

            for obs in data.obs:
                observer = ephem.Observer()
                site = obs.telescope.site
                observer.lon = site.earthloc.lon.to_value(u.radian)
                observer.lat = site.earthloc.lat.to_value(u.radian)
                observer.elevation = site.earthloc.height.to_value(u.meter)
                observer.epoch = ephem.J2000
                observer.temp = 0  # in Celcius
                observer.compute_pressure()

                prefix = f"{comm.group} : {obs.name}"

                # Get the observation time span and compute the horizontal
                # position of the SSO
                times = obs.shared[self.times].data
                sso_az, sso_el, sso_dist, sso_diameter = self._get_sso_position(
                    data, sso, times, observer
                )

                # Make sure detector data output exists
                dets = obs.select_local_detectors(detectors)
                obs.detdata.ensure(self.det_data, detectors=dets)

                self._observe_sso(
                    data, obs, sso_name, sso_az, sso_el, sso_dist, sso_diameter,
                    prefix, dets,
                )

            if data.comm.group_rank == 0:
                timer.stop()
                log.debug(
                    f"{data.comm.group} : Simulated and observed {sso_name} in "
                    f"{timer.seconds():.1f} seconds"
                )

        return

    def _get_planet_temp(self, sso_name):
        """
        Get the thermodynamic planet temperature given
        the frequency
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hf = h5py.File(os.path.join(dir_path, "data/planet_data.h5"), "r")
        if sso_name in hf.keys():
            tb = np.array(hf.get(sso_name)) * u.K
            freq = np.array(hf.get("freqs_ghz")) * u.GHz
            temp = utils.tb2tcmb(tb, freq)
        else:
            raise ValueError(
                f"Unknown planet name: '{sso_name}' not in {hf.keys()}"
            )
        return freq, temp

    def _get_beam_map(self, det, sso_diameter, ttemp_det):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        # Read in the simulated beam
        with open(self.beam_file, "rb") as f_t:
            beam_dic = pickle.load(f_t)
        description = beam_dic["size"]  # 2d array [[size, res], [n, 1]]
        model = beam_dic["data"]
        res = description[0][1] * u.degree
        beam_solid_angle = np.sum(model) * res ** 2

        n = int(description[1][0])
        size = description[0][0]
        sso_radius_avg = np.average(sso_diameter) / 2
        sso_solid_angle = np.pi * sso_radius_avg ** 2
        amp = ttemp_det * (
            sso_solid_angle.to_value(u.rad ** 2)
            / beam_solid_angle.to_value(u.rad ** 2)
        )
        w = np.radians(size / 2)
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        model *= amp
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w ** 2 + w ** 2)
        return beam, r

    @function_timer
    def _get_sso_position(self, data, sso, times, observer):
        """
        Calculate the SSO horizontal position
        """
        log = Logger.get()
        timer = Timer()
        timer.start()
        sso_az = np.zeros(times.size)
        sso_el = np.zeros(times.size)
        sso_dist = np.zeros(times.size)
        sso_diameter = np.zeros(times.size)
        for i, t in enumerate(times):
            observer.date = to_DJD(t)
            sso.compute(observer)
            sso_az[i] = sso.az
            sso_el[i] = sso.alt
            sso_dist[i] = sso.earth_distance
            sso_diameter[i] = sso.size
        sso_az *= u.radian
        sso_el *= u.radian
        sso_dist *= u.AU
        sso_diameter *= u.arcsec
        if data.comm.group_rank == 0:
            timer.stop()
            log.verbose(
                f"{data.comm.group} : Computed {sso.name} position in "
                f"{timer.seconds():.1f} seconds"
            )
        return sso_az, sso_el, sso_dist, sso_diameter

    @function_timer
    def _observe_sso(
            self,
            data,
            obs,
            sso_name,
            sso_az,
            sso_el,
            sso_dist,
            sso_diameter,
            prefix,
            dets,
    ):
        """
        Observe the SSO with each detector in tod
        """
        log = Logger.get()
        timer = Timer()

        for det in dets:
            timer.clear()
            timer.start()
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            # Compute detector quaternions

            obs_data = Data(comm=data.comm)
            obs_data._internal = data._internal
            obs_data.obs = [obs]
            self.detector_pointing.apply(obs_data, detectors=[det])
            obs_data.obs.clear()
            del obs_data

            azel_quat = obs.detdata[self.detector_pointing.quats][det]

            # Convert Az/El quaternion of the detector into angles
            theta, phi = qa.to_position(azel_quat)

            # Azimuth is measured in the opposite direction
            # than longitude
            az = 2 * np.pi - phi
            el = np.pi / 2 - theta

            # Convolve the planet SED with the detector bandpass
            planet_freq, planet_temp = self._get_planet_temp(sso_name)
            det_temp = bandpass.convolve(det, planet_freq, planet_temp)

            beam, radius = self._get_beam_map(det, sso_diameter, det_temp)

            # Interpolate the beam map at appropriate locations
            x = (az - sso_az.to_value(u.rad)) * np.cos(el)
            y = el - sso_el.to_value(u.rad)
            r = np.sqrt(x ** 2 + y ** 2)
            good = r < radius
            sig = beam(x[good], y[good], grid=False)
            signal[good] += sig

            timer.stop()
            if data.comm.world_rank == 0:
                log.verbose(
                    f"{prefix} : Simulated and observed {sso_name} in {det} in "
                    f"{timer.seconds():.1f} seconds"
                )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [
                self.times,
            ],
        }
        return req

    def _provides(self):
        return {
            "detdata": [
                self.det_data,
            ]
        }

    def _accelerators(self):
        return list()
