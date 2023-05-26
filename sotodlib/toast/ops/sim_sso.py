# Copyright (c) 2018-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import h5py
import os

import traitlets
import numpy as np
from astropy import units as u
import ephem
from scipy.constants import h, c, k
from scipy.interpolate import RectBivariateSpline
from scipy.signal import fftconvolve

import toast
from toast.timing import function_timer
from toast import qarray as qa
from toast.data import Data
from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance
from toast.ops.operator import Operator
from toast.utils import Environment, Logger, Timer
from toast.observation import default_values as defaults
from toast.coordinates import azel_to_radec

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
    """Operator that generates Solar System Object timestreams."""

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
        help="HDF5 file that stores the simulated beam",
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

    finite_sso_radius = Bool(
        False, help="Treat sources as finite and convolve beam with a disc."
    )

    polarization_fraction = Float(
        0, help="Polarization fraction for all simulated SSOs",
    )

    polarization_angle = Quantity(
        0 * u.deg,
        help="Polarization angle for all simulated SSOs. Measured in the same "
        "as `detector_weights`",
    )

    @traitlets.validate("sso_name")
    def _check_sso_name(self, proposal):
        sso_names = proposal["value"]
        if sso_names is not None:
            try:
                for sso_name in sso_names.split(","):
                    sso = getattr(ephem, sso_name)()
            except AttributeError:
                raise traitlets.TraitError(f"{sso_name} is not a valid SSO name")
        return sso_names

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

        for trait in "sso_name", "beam_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimSSO")

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
                if data.comm.group_rank == 0:
                    log.debug(
                        f"{data.comm.group} : {sso_name} at mean "
                        f"Az {np.mean(sso_az.to_value(u.degree))} deg, "
                        f"El {np.mean(sso_el.to_value(u.degree))} deg"
                    )

                # Store the SSO location

                source_coord_azel = np.column_stack(
                    [-sso_az.to_value(u.degree), sso_el.to_value(u.degree)]
                )

                obs.shared.create_column("source", (len(sso_az), 2), dtype=np.float64)
                if obs.comm.group_rank == 0:
                    obs.shared["source"].set(source_coord_azel)
                else:
                    obs.shared["source"].set(None)

                # Make sure detector data output exists.  If not, create it
                # with units of Kelvin.

                dets = obs.select_local_detectors(detectors)

                exists = obs.detdata.ensure(
                    self.det_data, detectors=dets, create_units=u.K
                )

                det_units = obs.detdata[self.det_data].units

                scale = toast.utils.unit_conversion(u.K, det_units)

                self._observe_sso(
                    data,
                    obs,
                    sso_name,
                    sso_az,
                    sso_el,
                    sso_dist,
                    sso_diameter,
                    prefix,
                    dets,
                    scale,
                )

            if data.comm.group_rank == 0:
                timer.stop()
                log.debug(
                    f"{data.comm.group} : Simulated and observed {sso_name} in "
                    f"{timer.seconds():.1f} seconds"
                )

        return

    @function_timer
    def _get_sso_temperature(self, sso_name):
        """
        Get the thermodynamic SSO temperature given
        the frequency
        """
        log = Logger.get()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hf = h5py.File(os.path.join(dir_path, "data/planet_data.h5"), "r")
        if sso_name == "Moon":
            freq = np.linspace(0, 1000, 1001)[1:] * 1e9
            T = 300 # Kelvin
            emissivity = 1.0
            tb = 1 / (k / (h * freq) * np.log(1 + (np.exp(h * freq / (k * T)) - 1) / emissivity))
            freq = freq * 1e-9 * u.GHz
            temp = utils.tb2tcmb(tb * u.K, freq)
        elif sso_name in hf.keys():
            tb = np.array(hf.get(sso_name)) * u.K
            freq = np.array(hf.get("freqs_ghz")) * u.GHz
            temp = utils.tb2tcmb(tb, freq)
        else:
            raise ValueError(f"Unknown planet name: '{sso_name}' not in {hf.keys()}")
        return freq, temp

    @function_timer
    def _get_beam_map(self, det, sso_diameter, ttemp_det):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        # Read in the simulated beam.  We could add operator traits to
        # specify whether to load different beams based on detector,
        # wafer, tube, etc and check that key here.
        log = Logger.get()
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
        size = description[0][0] * u.degree
        sso_radius_avg = np.average(sso_diameter) / 2
        sso_solid_angle = np.pi * sso_radius_avg**2
        amp = ttemp_det.to_value(u.K) * (
                    sso_solid_angle.to_value(u.rad**2) / beam_solid_angle.to_value(u.rad**2)
        )
        w = size.to_value(u.rad) / 2
        if self.finite_sso_radius:
            # Convolve the beam model with a disc rather than point-like source
            w_sso = sso_radius_avg.to_value(u.rad)
            n_sso = int(w_sso // res.to_value(u.rad)) * 2 + 3
            w_sso = (n_sso - 1) // 2 * res.to_value(u.rad)
            x_sso = np.linspace(-w_sso, w_sso, n_sso)
            y_sso = np.linspace(-w_sso, w_sso, n_sso)
            X, Y = np.meshgrid(x_sso, y_sso)
            source = np.zeros([n_sso, n_sso])
            source[X**2 + Y**2 < sso_radius_avg.to_value(u.rad)**2] = 1
            source *= amp / np.sum(source)
            model = fftconvolve(source, model, mode="full")
            # the convolved model is now larger than the pure beam model
            w += w_sso
            n += n_sso - 1
        else:
            # Treat the source as point-like. Reasonable approximation
            # if SSO radius << FWHM
            if sso_solid_angle > 0.1 * beam_solid_angle:
                log.warning("Ignoring non-negligible source diameter.  SSO image will be too narrow.")
            model *= amp
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w**2 + w**2)
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
        scale,
    ):
        """
        Observe the SSO with each detector in tod
        """
        log = Logger.get()
        timer = Timer()

        if self.polarization_fraction != 0 and self.detector_weights is None:
            raise RuntimeError(
                "Cannot simulate polarized SSOs without detector weights"
            )

        # Get a view of the data which contains just this single
        # observation
        obs_data = data.select(obs_name=obs.name)

        zaxis = np.array([0.0, 0.0, 1.0])
        bore_quat = obs_data.obs[0].shared[defaults.boresight_azel][:]
        bore_lon, bore_lat, _ = qa.to_lonlat_angles(bore_quat)

        beam = None
        for idet, det in enumerate(dets):
            timer.clear()
            timer.start()
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            self.detector_pointing.apply(obs_data, detectors=[det])
            if self.polarization_fraction != 0:
                self.detector_weights.apply(obs_data, detectors=[det])
            det_quat = obs_data.obs[0].detdata[self.detector_pointing.quats][det]

            # Convert Az/El quaternion of the detector into angles
            theta, phi, _ = qa.to_iso_angles(det_quat)

            # Azimuth is measured in the opposite direction
            # than longitude
            az = -phi
            el = np.pi / 2 - theta

            mean_az = np.mean(az)
            mean_el = np.mean(el)

            # Convolve the planet SED with the detector bandpass
            sso_freq, sso_temp = self._get_sso_temperature(sso_name)
            det_temp = bandpass.convolve(det, sso_freq, sso_temp) * u.K
            if beam is None or not "ALL" in self.beam_props:
                beam, radius = self._get_beam_map(det, sso_diameter, det_temp)

            # Interpolate the beam map at appropriate locations

            x = (az - sso_az.to_value(u.rad)) * np.cos(el)
            y = el - sso_el.to_value(u.rad)
            r = np.sqrt(x**2 + y**2)

            good = r < radius
            sig = beam(x[good], y[good], grid=False)
            if self.polarization_fraction != 0:
                self._observe_polarization(sig, obs, det, good)
            signal[good] += scale * sig

            timer.stop()
            if data.comm.world_rank == 0:
                log.verbose(
                    f"{prefix} : Simulated and observed {sso_name} in {det} in "
                    f"{timer.seconds():.1f} seconds"
                )

        return

    def _observe_polarization(self, sig, obs, det, good):
        # Stokes weights for observing polarized source
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
        angle = self.polarization_angle.to_value(u.radian)

        sig *= weights_I + pfrac * (
            np.cos(2 * angle) * weights_Q + np.sin(2 * angle) * weights_U
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


# def plot_projected_quats(
#     outfile, qbore=None, qdet=None, valid=slice(None), scale=1.0, planet=None
# ):
#     """Plot a list of quaternion arrays in longitude / latitude."""

#     toast.vis.set_matplotlib_backend()
#     import matplotlib.pyplot as plt

#     # Convert boresight and detector quaternions to angles

#     qbang = None
#     if qbore is not None:
#         qbang = np.zeros((3, qbore.shape[0]), dtype=np.float64)
#         qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(qbore)
#         qbang[0] *= 180.0 / np.pi
#         qbang[1] *= 180.0 / np.pi
#         lon_min = np.amin(qbang[0])
#         lon_max = np.amax(qbang[0])
#         lat_min = np.amin(qbang[1])
#         lat_max = np.amax(qbang[1])

#     qdang = None
#     if qdet is not None:
#         qdang = np.zeros((qdet.shape[0], 3, qdet.shape[1]), dtype=np.float64)
#         for det in range(qdet.shape[0]):
#             qdang[det, 0], qdang[det, 1], qdang[det, 2] = qa.to_lonlat_angles(qdet[det])
#             qdang[det, 0] *= 180.0 / np.pi
#             qdang[det, 1] *= 180.0 / np.pi
#         lon_min = np.amin(qdang[:, 0])
#         lon_max = np.amax(qdang[:, 0])
#         lat_min = np.amin(qdang[:, 1])
#         lat_max = np.amax(qdang[:, 1])

#     # Set the sizes of shapes based on the plot range

#     span_lon = lon_max - lon_min
#     span_lat = lat_max - lat_min
#     span = max(span_lon, span_lat)
#     # bmag = 0.5 * span * scale
#     # dmag = 0.2 * span * scale
#     bmag = 0.2
#     dmag = 0.1

#     if span_lat > span_lon:
#         fig_y = 10
#         fig_x = fig_y * (span_lon / span_lat)
#         if fig_x < 4:
#             fig_x = 4
#     else:
#         fig_x = 10
#         fig_y = fig_x * (span_lat / span_lon)
#         if fig_y < 4:
#             fig_y = 4

#     figdpi = 100

#     fig = plt.figure(figsize=(fig_x, fig_y), dpi=figdpi)
#     ax = fig.add_subplot(1, 1, 1, aspect="equal")

#     # Compute the font size to use for detector labels
#     fontpix = 0.1 * figdpi
#     fontpt = int(0.75 * fontpix)

#     # Plot source if we have it

#     if planet is not None:
#         ax.plot(planet[:, 0], planet[:, 1], color="purple", marker="+")

#     # Plot boresight if we have it

#     if qbang is not None:
#         ax.scatter(qbang[0][valid], qbang[1][valid], color="black", marker="x")
#         for ln, lt, ps in np.transpose(qbang)[valid]:
#             wd = 0.05 * bmag
#             dx = bmag * np.sin(ps)
#             dy = -bmag * np.cos(ps)
#             ax.arrow(
#                 ln,
#                 lt,
#                 dx,
#                 dy,
#                 width=wd,
#                 head_width=4.0 * wd,
#                 head_length=0.2 * bmag,
#                 length_includes_head=True,
#                 ec="red",
#                 fc="red",
#             )

#     # Plot detectors if we have them

#     if qdang is not None:
#         for idet, dang in enumerate(qdang):
#             ax.scatter(dang[0][valid], dang[1][valid], color="blue", marker=".")
#             for ln, lt, ps in np.transpose(dang)[valid]:
#                 wd = 0.05 * dmag
#                 dx = dmag * np.sin(ps)
#                 dy = -dmag * np.cos(ps)
#                 ax.arrow(
#                     ln,
#                     lt,
#                     dx,
#                     dy,
#                     width=wd,
#                     head_width=4.0 * wd,
#                     head_length=0.2 * dmag,
#                     length_includes_head=True,
#                     ec="blue",
#                     fc="blue",
#                 )
#             ax.text(
#                 dang[0][valid][0] + (idet % 2) * 1.5 * dmag,
#                 dang[1][valid][0] + 1.0 * dmag,
#                 f"{idet:02d}",
#                 color="k",
#                 fontsize=fontpt,
#                 horizontalalignment="center",
#                 verticalalignment="center",
#                 bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
#             )

#     # Invert x axis so that longitude reflects what we would see from
#     # inside the celestial sphere
#     plt.gca().invert_xaxis()

#     ax.set_xlabel("Longitude Degrees", fontsize="medium")
#     ax.set_ylabel("Latitude Degrees", fontsize="medium")

#     fig.suptitle("Projected Pointing and Polarization on Sky")

#     plt.savefig(outfile)
#     plt.close()
