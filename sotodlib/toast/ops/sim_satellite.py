# Copyright (c) 2026 Simons Observatory.
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
from toast.timing import function_timer, Timer
from toast import qarray as qa
from toast.data import Data
from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance
from toast.ops.operator import Operator
from toast.utils import Environment, Logger
from toast.observation import default_values as defaults
from toast.coordinates import azel_to_radec

from . import utils


def Site:
    """ The Site class shall have all traits needed to translate
    satellite position into horizontal frame
    """
    def __init__(self):
        pass


def Satellite:
    """ Container for all traits needed to simulate TOD for one satellite
    """
    def __init__(self, name=None, polarized=False, temperature=None):
        self.name = name
        self.polarized = polarized
        self.temperature = temperature
        self._get_sed()

    def _get_sed(self):
        """Derive the spectral energy distribution for future
        convolution
        """
        self.freq = None
        self.sed = None
        return

    def compute_position(self, site, times):
        """ Compute and store the satellite position in the horizontal
        frame
        """
        # This could be stored in the node shared memory
        self.t = times
        self.az = None
        self.el = None

    def clear_position(self):
        """ Release memory used to store position """
        del self.t
        del self.az
        del self.el


@trait_docs
class SimSatellite(Operator):
    """Operator that generates satellite timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    beam_file = Unicode(
        None,
        allow_none=True,
        help="HDF5 file that stores the simulated beam",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
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

    finite_radii = Bool(
        False, help="Treat sources as finite and convolve beam with a disc."
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
                raise RuntimeError(f"You must set `{trait}` before running SimSatellite")

        site = Site()

        for obs in data.obs:
            satellites = self._draw_satellites(obs)

            # Make sure detector data output exists.  If not, create it
            # with units of Kelvin.
            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            exists = obs.detdata.ensure(
                self.det_data, detectors=dets, create_units=u.K
            )
            det_units = obs.detdata[self.det_data].units
            scale = toast.utils.unit_conversion(u.K, det_units)

            for satellite in satellites:
                satellite.compute_position(site, times)
                self._observe_satellite(data, obs, satellite, dets, scale)
                satellite.clear_position()
                
                if data.comm.group_rank == 0:
                    timer.stop()
                    log.info(
                        f"{data.comm.group} : Simulated and observed {satellite.name} in "
                        f"{timer.seconds():.1f} seconds"
                    )

            if data.comm.group_rank == 0:
                timer.stop()
                log.info(
                    f"{data.comm.group} : Simulated and observed satellites in "
                    f"{timer.seconds():.1f} seconds"
                )

        return

    @function_timer
    def _draw_satellites(self, obs):
        """Generate a population of satellites
        """
        times = obs.shared[self.times].data
        satellites = []
        return satellites

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
                beam_dic["size"] = [
                    [f_t["beam"].attrs["size"], f_t["beam"].attrs["res"]],
                    [f_t["beam"].attrs["npix"], 1]
                ]
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
            # if satellite radius << FWHM
            if sso_solid_angle > 0.1 * beam_solid_angle:
                log.warning(
                    "Ignoring non-negligible source diameter. "
                    "Satellite image will be too narrow."
                )
            model *= amp
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w**2 + w**2)
        return beam, r

    @function_timer
    def _observe_satellite(
        self,
        data,
        obs,
        satellite,
        dets,
        scale,
    ):
        """
        Observe the satellite with each detector in `dets`
        """
        log = Logger.get()

        if satellite.polarized and self.detector_weights is None:
            raise RuntimeError(
                "Cannot simulate polarized satellites without detector weights"
            )

        # Get a view of the data which contains just this single
        # observation
        obs_data = data.select(obs_name=obs.name)

        beam = None
        for idet, det in enumerate(dets):
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            self.detector_pointing.apply(obs_data, detectors=[det])
            if satellite.polarized:
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

            # Convolve the satellite SED with the detector bandpass
            det_temp = bandpass.convolve(det, satellite.freq, satellite.sed) * u.K
            if beam is None or not "ALL" in self.beam_props:
                beam, radius = self._get_beam_map(det, sso_diameter, det_temp)

            # Interpolate the beam map at appropriate locations

            az_diff = (az - satellite.az.to_value(u.rad) + np.pi) % (2 * np.pi) - np.pi
            x = az_diff * np.cos(el)
            y = el - satellite.el.to_value(u.rad)
            r = np.sqrt(x**2 + y**2)

            good = r < radius
            sig = beam(x[good], y[good], grid=False)
            if self.polarization_fraction != 0:
                self._observe_polarization(sig, obs, det, good)
            signal[good] += scale * sig

            if data.comm.world_rank == 0:
                log.verbose(
                    f"{prefix} : Simulated and observed {satellite.name} in {det}"
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
