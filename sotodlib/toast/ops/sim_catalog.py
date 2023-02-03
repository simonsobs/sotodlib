# Copyright (c) 2018-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import h5py
import os
import pickle

from astropy import units as u
import ephem
import healpy as hp
import numpy as np
from scipy.constants import h, c, k
from scipy.interpolate import RectBivariateSpline
from scipy.signal import fftconvolve
import toml
import traitlets

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
class SimCatalog(Operator):
    """Operator that generates Solar System Object timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    catalog_file = Unicode(
        None,
        allow_none=True,
        help="Name of the TOML catalog file",
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

    @traitlets.validate("catalog_file")
    def _check_catalog_file(self, proposal):
        filename = proposal["value"]
        if filename is not None and not os.path.isfile(filename):
            raise traitlets.TraitError(f"Catalog file does not exist: {filename}")
        return filename

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

    @function_timer
    def _load_catalog(self):
        # Load the TOML into a dictionary
        with open(self.catalog_file, "r") as f:
            self.catalog = toml.loads(f.read())
        # Translate each source position into a vector for rapid
        # distance calculations
        for source_name, source_dict in self.catalog.items():
            lon = source_dict["ra_deg"]
            lat = source_dict["dec_deg"]
            source_dict["vec"] = hp.dir2vec(lon, lat, lonlat=True)
        return

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

        for trait in "catalog_file", "beam_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                msg = f"You must set `{trait}` before running SimCatalog"
                raise RuntimeError(msg)

        self._load_catalog()

        for obs in data.obs:
            prefix = f"{comm.group} : {obs.name}"

            # Make sure detector data output exists.  If not, create it
            # with units of Kelvin.

            dets = obs.select_local_detectors(detectors)
            exists = obs.detdata.ensure(
                self.det_data, detectors=dets, create_units=u.K
            )
            det_units = obs.detdata[self.det_data].units
            scale = toast.utils.unit_conversion(u.K, det_units)

            self._observe_catalog(
                data,
                obs,
                prefix,
                dets,
                scale,
            )

        return

    @function_timer
    def _get_beam_map(self, det):
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
            with open(self.beam_file, "rb") as f_t:
                beam_dic = pickle.load(f_t)
                self.beam_props["ALL"] = beam_dic
        description = beam_dic["size"]  # 2d array [[size, res], [n, 1]]
        model = beam_dic["data"].copy()
        model /= np.amax(model)
        res = description[0][1] * u.degree
        beam_solid_angle = np.sum(model) * res**2

        n = int(description[1][0])
        size = description[0][0] * u.degree
        amp = 1 / beam_solid_angle.to_value(u.rad**2)
        w = size.to_value(u.rad) / 2
        model *= amp
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w**2 + w**2)
        return beam, r, beam_solid_angle

    @function_timer
    def _observe_catalog(
        self,
        data,
        obs,
        prefix,
        dets,
        scale,
    ):
        """
        Observe the catalog with each detector in tod
        """
        log = Logger.get()

        # Get a view of the data which contains just this single
        # observation
        obs_data = data.select(obs_name=obs.name)

        times_mjd = to_MJD(obs.shared[self.times].data)
        beam = None

        for idet, det in enumerate(dets):
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            self.detector_pointing.apply(obs_data, detectors=[det])
            det_quat = obs_data.obs[0].detdata[self.detector_pointing.quats][det]

            # Convert Az/El quaternion of the detector into angles
            # `psi` includes the rotation to the detector polarization
            #sensitive direction

            det_theta, det_phi, det_psi = qa.to_iso_angles(det_quat)
            det_vec = hp.dir2vec(det_theta, det_phi).T.copy()

            if beam is None or not "ALL" in self.beam_props:
                beam, beam_radius, beam_solid_angle = self._get_beam_map(det)
            dp_radius = np.cos(beam_radius)

            for source_name, source_dict in self.catalog.items():
                # Is this source close enough to register?
                dp = np.dot(det_vec, source_dict["vec"])
                hit = dp > dp_radius
                nhit = np.sum(hit)
                if nhit == 0:
                    continue

                # Get the appropriate source SED and convolve with the
                # detector bandpass
                if "times_mjd" in source_dict:
                    source_times = np.array(source_dict["times_mjd"])
                    ind = np.array(np.searchsorted(source_times, times_mjd))
                    # When time stamps fall outside the period covered by
                    # source time, we assume the source went quiet
                    good = np.logical_and(ind > 0, ind < len(source_times))
                    hit *= good
                    nhit = np.sum(hit)
                    if nhit == 0:
                        # This source is not active during our observation
                        continue
                    ind = ind[hit]
                    lengths = source_times[ind] - source_times[ind - 1]
                    right_weights = (source_times[ind] - times_mjd[hit]) \
                                    / lengths
                    left_weights = 1 - right_weights
                    # useful shorthands
                    freq = np.array(source_dict["freqs_ghz"]) * u.GHz
                    seds = np.array(source_dict["flux_density_mjysr"]) * u.MJy / u.sr
                    # Mean SED used for bandpass convolution
                    wright = np.mean(right_weights)
                    wleft = 1 - wright
                    cindex = int(np.median(ind))
                    sed_mean = wleft * seds[cindex - 1] + wright * seds[cindex]
                    # Time-dependent amplitude to scale the mean SED
                    cfreq = bandpass.center_frequency(det, alpha=-1)
                    amplitudes = []
                    for sed in seds:
                        # Interpolate the SED to the detector central frequency
                        # in log-log domain where power-law spectra are
                        # linear
                        amp = np.exp(np.interp(
                                np.log(cfreq.to_value(u.GHz)),
                                np.log(freq.to_value(u.GHz)),
                                np.log(sed.to_value(u.MJy / u.sr))
                        ))
                        amplitudes.append(amp)
                    amplitudes = np.array(amplitudes)
                    # This is the time-dependent amplitude relative to
                    # sed_mean
                    amplitude = (
                        left_weights * amplitudes[ind - 1] +
                        right_weights * amplitudes[ind]
                    )
                    amplitude /=  (
                        wleft * amplitudes[cindex - 1] +
                        wright * amplitudes[cindex]
                    )
                    if "pol_frac" in source_dict:
                        pol_fracs = np.array(source_dict["pol_frac"])
                        pol_frac = (
                            left_weights * pol_fracs[ind - 1] +
                            right_weights * pol_fracs[ind]
                        )
                        pol_angles = np.unwrap(
                            np.radians(source_dict["pol_angle_deg"])
                        )
                        pol_angle = np.array(
                            left_weights * pol_angles[ind - 1] +
                            right_weights * pol_angles[ind]
                        )
                    else:
                        pol_frac = None
                        pol_angle = None
                else:
                    freq = np.array(source_dict["freqs_ghz"]) * u.GHz
                    sed_mean = np.array(source_dict["flux_density_mjysr"]) * u.MJy / u.sr
                    if "pol_frac" in source_dict:
                        pol_frac = np.array(source_dict["pol_frac"])
                        pol_angle = np.radians(source_dict["pol_angle_deg"])
                    else:
                        pol_frac = None
                        pol_angle = None
                    amplitude = 1
                    
                # Convolve the SED with the detector bandpass
                flux_density = bandpass.convolve(
                    det,
                    freq,
                    sed_mean.to_value(u.MJy / u.sr),
                )
                temperature = (
                    flux_density
                    / beam_solid_angle.to_value(u.rad**2)
                    / bandpass.kcmb2mjysr(det)
                )

                # Modulate the temperature in time
                temperature = temperature * amplitude

                # FIXME: modulate temperature by polarization

                # Interpolate the beam map at appropriate locations
                source_theta = np.radians(90 - source_dict["dec_deg"])
                source_phi = np.radians(source_dict["ra_deg"])
                x = (det_phi[hit] - source_phi) * np.cos(np.pi / 2 - det_theta[hit])
                y = det_theta[hit] - source_theta
                #import pdb
                #pdb.set_trace()
                sig = beam(x, y, grid=False) * temperature
                signal[hit] += scale * sig

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
