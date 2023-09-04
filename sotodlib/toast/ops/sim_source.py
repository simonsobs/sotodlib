# Copyright (c) 2018-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import h5py
import copy

import traitlets

import numpy as np

from astropy import units as u

import ephem

from scipy.interpolate import RectBivariateSpline

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Float, Instance, List, Bool, Quantity

from toast.ops.operator import Operator
from toast.instrument import Focalplane

from toast.utils import Logger, Timer, unit_conversion

from toast.observation import default_values as defaults

from sotodlib.coords import local

from . import utils


def spectrum(power, fc, sigma, base, err_fc, noise, az_size, el_size, dist):
    """Generate a power spectrum in W/m^2/sr/Hz for a source.

    The source has a delta-like emission and a top-hat beam in 2D.

    Args:
        power: total power of the signal in dBm
        fc: frequency of the signal in GHz
        sigma: width of the signal in kHz
        base: base level of the signal in dBm
        err_fc: error in the signal frequency in kHz
        noise: noise of the signal in W/Hz
        ang_size: beam size of the signal in degrees as [az_size, el_size]
        dist: distance of the source in meter

    Returns:
        (tuple): The (frequency, spectrum) arrays.

    """

    fc *= 1e9  ### Conversion to Hz
    err_fc *= 1e3

    fc = fc + np.random.normal(0, err_fc)

    sigma *= 1e3  ### Conversion to Hz

    factor = 10  # Increase by a factor the width of the signal
    freq_step = 10

    freq = np.arange(0.5, 400, freq_step) * 1e9
    central = np.arange(-factor * sigma, factor * sigma, sigma) + fc

    # Add edges to fill the gap for future interpolations
    nstep = 20
    edge_l = np.linspace(fc - freq_step * 1e9, fc - factor * sigma, nstep)
    edge_u = np.linspace(fc + factor * sigma, fc + freq_step * 1e9, nstep)

    edges = np.append(edge_l, edge_u)
    freq = np.append(freq, central)

    freq = np.sort(np.unique(np.append(freq, edges)))

    mask = np.in1d(freq, central)

    signal = np.zeros_like(freq)

    signal[mask] = power - 10 * np.log10(sigma)
    signal[~mask] = base

    signal = 10 ** (signal / 10) / 1000  ### Conversion from dBm to W

    signal += np.random.normal(0, noise, size=(len(signal)))

    az_size = np.radians(az_size) / 2 / np.pi  # Half size in unit of pi
    el_size = np.radians(el_size) / 2 / np.pi  # Half size in unit of pi

    signal /= (
        4 * np.pi * az_size * np.sin(el_size * np.pi)
    )  # Normalize in the solid angle
    signal /= dist**2  # Normalize based on the distance

    # Set units
    freq *= u.Hz
    signal *= u.W / (u.m**2 * u.Hz)

    return freq, signal


@trait_docs
class SimSource(Operator):
    """Operator that generates an Artificial Source timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    source_init_dist = Quantity(
        u.Quantity(500.0, u.meter),
        help="Initial distance of the artificial source in meters",
    )

    variable_elevation = Bool(False, help="Set True to change the drone elevation")

    keep_distance = Bool(
        False,
        help="Set True to maintain the distance always the same throughout a scan",
    )

    focalplane = Instance(
        klass=Focalplane,
        allow_none=True,
        help="Focalplane instance used for FoV calculation",
    )

    source_err = List(
        [0, 0, 0],
        help="Source Position Error in ECEF Coordinates as [[X, Y, Z]] in meters",
    )

    source_size = Float(
        0.1,
        help="Source Size in meters",
    )

    source_amp = Float(help="Max amplitude of the source in dBm")

    source_freq = Float(help="Central frequency of the source in GHz")

    source_width = Float(help="Width of the source signal in kHz")

    source_baseline = Float(help="Baseline signal level of the source in dBm")

    source_noise = Float(help="Noise level in W/Hz")

    source_beam_az = Float(
        115,
        help="Beam size along the azimuthal axis of the source in degrees",
    )

    source_beam_el = Float(
        65,
        help="Beam size along the elevation axis of the source in degrees",
    )

    source_pol_angle = Float(
        90,
        help="Angle of the polarization vector emitted by the source in degrees (0 means parallel to the gorund and 90 vertical)",
    )

    source_pol_angle_error = Float(
        0,
        help="Error in the angle of the polarization vector",
    )

    polarization_fraction = Float(
        1,
        help="Polarization fraction of the emitted signal",
    )

    beam_file = Unicode(
        None,
        allow_none=True,
        help="HDF5 file that stores the simulated beam",
    )

    wind_gusts_amp = Quantity(
        u.Quantity(0.0, u.Unit("m / s")), help="Amplitude of gusts of wind"
    )

    wind_gusts_duration = Quantity(
        u.Quantity(0.0, u.second), help="Duration of each gust of wind"
    )

    wind_gusts_number = Float(0, help="Number of wind gusts")

    wind_damp = Float(
        0, help="Dampening effect to reduce the movement of the drone due to gusts"
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

    elevation = Unicode(
        defaults.elevation,
        allow_none=True,
        help="Observation shared key for boresight elevation",
    )

    azimuth = Unicode(
        defaults.azimuth,
        allow_none=True,
        help="Observation shared key for azimuth",
    )

    @traitlets.validate("beam_file")
    def _check_beam_file(self, proposal):
        beam_file = proposal["value"]
        if beam_file is not None and not os.path.isfile(beam_file):
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

    @traitlets.validate("detector_weights")
    def _check_detector_weights(self, proposal):
        detweights = proposal["value"]
        if detweights is not None:
            if not isinstance(detweights, Operator):
                raise traitlets.TraitError(
                    "detector_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "quats",
                "weights",
                "mode",
            ]:
                if not detweights.has_trait(trt):
                    msg = f"detector_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detweights

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store of per-detector beam properties.  Eventually we could modify the
        # operator traits to list files per detector, per wafer, per tube, etc.
        # For now, we use the same beam for all detectors, so this will have only
        # one entry.
        self.beam_props = dict()

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

        for trait in "beam_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimSource")

        timer = Timer()
        timer.start()

        if data.comm.group_rank == 0:
            log.debug(f"{data.comm.group} : Simulating Source")

        for obs in data.obs:
            observer = ephem.Observer()
            site = obs.telescope.site
            observer.lon = site.earthloc.lon.to_value(u.radian)
            observer.lat = site.earthloc.lat.to_value(u.radian)
            observer.elevation = site.earthloc.height.to_value(u.meter)

            prefix = f"{comm.group} : {obs.name}"

            # Get the observation time span and compute the horizontal
            # position of the SSO
            times = obs.shared[self.times].data

            (
                source_az,
                source_el,
                source_dist,
                source_diameter,
            ) = self._get_source_position(obs, observer, times)

            # Make sure detector data output exists
            dets = obs.select_local_detectors(detectors)

            obs.detdata.ensure(self.det_data, detectors=dets, create_units=u.K)

            det_units = obs.detdata[self.det_data].units

            scale = unit_conversion(u.K, det_units)

            self._observe_source(
                data,
                obs,
                source_az,
                source_el,
                source_dist,
                source_diameter,
                prefix,
                dets,
                scale,
            )

        if data.comm.group_rank == 0:
            timer.stop()
            log.debug(
                f"{data.comm.group} : Simulated and observed Source in "
                f"{timer.seconds():.1f} seconds"
            )

        return

    @function_timer
    def _get_source_position(self, obs, observer, times):

        log = Logger.get()
        timer = Timer()
        timer.start()

        az_start = np.median(np.array(obs.shared[self.azimuth])) * u.rad

        az_init = np.ones_like(times) * az_start

        if not self.variable_elevation:
            el_init = (
                np.ones(len(times))
                * np.amax(np.array(obs.shared[self.elevation]))
                * u.rad
            )
            el_start = el_init[0]

        else:
            # Always consider a discending drone, it is easier to fly this way

            FoV = self.focalplane.field_of_view

            el_start = np.array(obs.shared[self.elevation])[0] * u.rad + FoV / 2
            el_end = el_start - FoV
            el_init = np.linspace(el_start, el_end, len(times), endpoint=True)

        if self.keep_distance:
            distance = np.ones_like(times) * self.source_init_dist
        else:
            distance = self.source_init_dist * np.cos(el_start) / np.cos(el_init)

        if np.any(np.array(self.source_err) >= 1e-4) or self.wind_gusts_amp.value != 0:

            E, N, U = local.hor2enu(az_init, el_init, distance)
            X, Y, Z = local.enu2ecef(
                E, N, U, observer.lon, observer.lat, observer.elevation, ell="WGS84"
            )

            if np.any(np.array(self.source_err) >= 1e-4):
                X = (
                    X
                    + np.random.normal(0, self.source_err[0], size=(len(times)))
                    * X.unit
                )
                Y = (
                    Y
                    + np.random.normal(0, self.source_err[1], size=(len(times)))
                    * Y.unit
                )
                Z = (
                    Z
                    + np.random.normal(0, self.source_err[2], size=(len(times)))
                    * Z.unit
                )

            if self.wind_gusts_amp.value != 0:

                delta_t = np.amin(np.diff(times))

                samples = int(self.wind_gusts_duration.value / delta_t)

                dx = np.zeros((self.wind_gusts_number, samples))
                dy = np.zeros_like(dx)
                dz = np.zeros_like(dx)

                # Compute random wind direction for any wind gust
                v = np.random.rand(self.wind_gusts_number, 3)
                wind_direction = v / np.linalg.norm(v)

                # Compute the angles using the versor direction
                theta = np.arccos(wind_direction[:, 2])
                phi = np.arctan2(wind_direction[:, 1], wind_direction[:, 0])

                base = np.reshape(
                    np.tile(np.arange(0, samples + 1), self.wind_gusts_number),
                    (self.wind_gusts_number, samples + 1),
                )

                dt = base * delta_t

                wind_amp = self.wind_gusts_amp * self.drone_damp

                dz = wind_amp * wind_direction[:, 2][:, np.newaxis] * dt
                dx = (
                    wind_amp
                    * np.sin(theta[:, np.newaxis])
                    * np.cos(phi[:, np.newaxis])
                    * dt
                )
                dy = (
                    wind_amp
                    * np.sin(theta[:, np.newaxis])
                    * np.sin(phi[:, np.newaxis])
                    * dt
                )

                # Create an array of position returning to the origin
                dx = np.hstack((dx, np.flip(dx, axis=1)))
                dy = np.hstack((dy, np.flip(dy, axis=1)))
                dz = np.hstack((dz, np.flip(dz, axis=1)))

                idx = np.arange(len(X))
                idx_wind = np.ones(self.wind_gusts_number)

                while np.any(np.diff(idx_wind) < 2.5 * samples):
                    idx_wind = np.random.choice(idx, self.wind_gusts_number)

                idxs = (
                    np.hstack(
                        (
                            base,
                            base[:, -1][:, np.newaxis]
                            + np.ones(self.wind_gusts_number, dtype=int)[:, np.newaxis]
                            + base,
                        )
                    )
                    + idx_wind
                ).flatten()

                (good,) = np.where(idxs < len(X))
                valid = np.arange(0, len(good), dtype=int)

                X[idxs[good]] += dx.flatten()[valid]
                Y[idxs[good]] += dy.flatten()[valid]
                Z[idxs[good]] += dz.flatten()[valid]

            X_tel, Y_tel, Z_tel, _, _, _ = local.lonlat2ecef(
                observer.lon, observer.lat, observer.elevation
            )

            E, N, U, _, _, _ = local.ecef2enu(
                X_tel,
                Y_tel,
                Z_tel,
                X,
                Y,
                Z,
                0,
                0,
                0,
                0,
                0,
                0,
                observer.lon,
                observer.lat,
            )

            source_az, source_el, source_distance, _, _, _ = local.enu2hor(
                E, N, U, 0, 0, 0
            )

        else:
            source_az = az_init.copy()
            source_el = el_init.copy()
            source_distance = distance.copy()

        size = local._check_quantity(self.source_size, u.m)
        size = (size / source_distance) * u.rad

        obs["source_az"] = source_az
        obs["source_el"] = source_el
        obs["source_distance"] = source_distance

        # Create a shared data object with the source location
        source_coord = np.column_stack(
            [-source_az.to_value(u.degree), source_el.to_value(u.degree)]
        )
        obs.shared.create_column("source", (len(source_az), 2), dtype=np.float64)
        if obs.comm.group_rank == 0:
            obs.shared["source"].set(source_coord)
        else:
            obs.shared["source"].set(None)

        if obs.comm.group_rank == 0:
            timer.stop()
            log.verbose(
                f"{obs.comm.group} : Computed source position in "
                f"{timer.seconds():.1f} seconds"
            )

        return source_az, source_el, source_distance, size

    def _get_source_temp(self, distance):

        dist = np.median(distance.value)  ### Get the median distance

        amp = self.source_amp
        fc = self.source_freq
        sigma = self.source_width
        base = self.source_baseline
        noise = self.source_noise

        freq, spec = spectrum(
            amp,
            fc,
            sigma,
            base,
            sigma,
            noise,
            self.source_beam_az,
            self.source_beam_el,
            dist,
        )

        temp = utils.s2tcmb(spec, freq)

        return freq, temp

    def _get_beam_map(self, det, source_diameter, ttemp_det):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        # Read in the simulated beam.  We could add operator traits to
        # specify whether to load different beams based on detector,
        # wafer, tube, etc and check that key here.
        if "ALL" in self.beam_props:
            # We have already read the single beam file.
            beam_dic = self.beam_props["ALL"]
        else:
            with h5py.File(self.beam_file, 'r') as f_t:
                beam_dic = {}
                beam_dic["data"] = f_t["beam"][:]
                beam_dic["size"] = [[f_t["beam"].attrs["size"], f_t["beam"].attrs["res"]], [f_t["beam"].attrs["npix"], 1]]
                self.beam_props["ALL"] = beam_dic
        description = beam_dic["size"]  # 2d array [[size, res], [n, 1]]
        model = beam_dic["data"]
        res = description[0][1] * u.degree
        beam_solid_angle = np.sum(model) * res**2

        n = int(description[1][0])
        size = description[0][0]
        source_radius_avg = np.average(source_diameter) / 2
        source_solid_angle = np.pi * source_radius_avg**2

        amp = ttemp_det * (
            source_solid_angle.to_value(u.rad**2)
            / beam_solid_angle.to_value(u.rad**2)
        )
        w = np.radians(size / 2)
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        model *= amp
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w**2 + w**2)
        return beam, r

    @function_timer
    def _observe_source(
        self,
        data,
        obs,
        source_az,
        source_el,
        source_dist,
        source_diameter,
        prefix,
        dets,
        scale,
    ):
        """
        Observe the Source with each detector in tod
        """
        log = Logger.get()
        timer = Timer()

        source_freq, source_temp = self._get_source_temp(source_dist)

        # Get a view of the data which contains just this single
        # observation
        obs_data = data.select(obs_uid=obs.uid)

        for det in dets:
            timer.clear()
            timer.start()
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            # Compute detector quaternions and Stokes weights

            self.detector_pointing.apply(obs_data, detectors=[det])
            if self.detector_weights is not None:
                self.detector_weights.apply(obs_data, detectors=[det])

            azel_quat = obs.detdata[self.detector_pointing.quats][det]

            # Convert Az/El quaternion of the detector into angles
            theta, phi, _ = qa.to_iso_angles(azel_quat)

            # Azimuth is measured in the opposite direction
            # than longitude
            az = 2 * np.pi - phi
            el = np.pi / 2 - theta

            # Convolve the planet SED with the detector bandpass
            det_temp = bandpass.convolve(det, source_freq, source_temp)

            beam, radius = self._get_beam_map(det, source_diameter, det_temp)

            # Interpolate the beam map at appropriate locations
            az_diff = (az - source_az.to_value(u.rad) + np.pi) % (2 * np.pi) - np.pi
            x = az_diff * np.cos(el)
            y = el - source_el.to_value(u.rad)
            r = np.sqrt(x**2 + y**2)
            good = r < radius
            sig = beam(x[good], y[good], grid=False)

            # Stokes weights for observing polarized source
            if self.detector_weights is None:
                weights_I = 1
                weights_Q = 0
                weights_U = 0
            else:
                weights = obs.detdata[self.detector_weights.weights][det]
                weight_mode = self.detector_weights.mode
                if "I" in weight_mode:
                    ind = weight_mode.index("I")
                    weights_I = weights[good, ind].copy()
                else:
                    weights_I = 0
                if "Q" in weight_mode:
                    ind = weight_mode.index("Q")
                    weights_Q = weights[good, ind].copy()
                else:
                    weights_Q = 0
                if "U" in weight_mode:
                    ind = weight_mode.index("U")
                    weights_U = weights[good, ind].copy()
                else:
                    weights_U = 0

            pfrac = self.polarization_fraction
            angle = np.radians(
                self.source_pol_angle
                + np.random.normal(0, self.source_pol_angle_error, size=(len(sig)))
            )

            sig *= weights_I + pfrac * (
                np.cos(2 * angle) * weights_Q + np.sin(2 * angle) * weights_U
            )

            signal[good] += sig

            timer.stop()
            if data.comm.world_rank == 0:
                log.verbose(
                    f"{prefix} : Simulated and observed source in {det} in "
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
            "detdata": [
                self.det_data,
                self.quats_azel,
            ],
        }

        if self.weights is not None:
            req["weights"].append(self.weights)

        return req

    def _provides(self):
        return {
            "detdata": [
                self.det_data,
            ],
            "shared": [
                "source",
            ],
        }

    def _accelerators(self):
        return list()
