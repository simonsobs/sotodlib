import os
import pickle

import traitlets

import numpy as np

from astropy import constants
from astropy import units as u

import ephem

from scipy.interpolate import RectBivariateSpline

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Float, Instance, List, Bool

from toast.ops.operator import Operator

from toast.utils import Environment, Logger, Timer

from toast.observation import default_values as defaults

from ...coords.local import *



def tb2s(tb, nu):
    """ Convert blackbody temperature to spectral
    radiance s_nu at frequency nu
    Args:
        tb: float or array
            blackbody temperature, unit: Kelvin
        nu: float or array (with same dimension as tb)
            frequency where the spectral radiance is evaluated, unit: Hz
    Return:
        s_nu: same dimension as tb
            spectral radiance s_nu, unit: W*sr−1*m−2*Hz−1
    """
    h = constants.h.value
    c = constants.c.value
    k_b = constants.k_B.value

    x = h * nu / (k_b * tb)

    return 2 * h * nu ** 3 / c ** 2 / (np.exp(x) - 1)


def s2tcmb(s_nu, nu):
    """ Convert spectral radiance s_nu at frequency nu to t_cmb,
    t_cmb is defined in the CMB community as the offset from the
    mean CMB temperature assuming a linear relation between t_cmb
    and s_nu, the t_cmb/s_nu slope is evalutated at the mean CMB
    temperature.
    Args:
        s_nu: float or array
            spectral radiance s_nu, unit: W*sr−1*m−2*Hz−1
        nu: float or array (with same dimension as s_nu)
            frequency where the evaluation is perfomed, unit: Hz
    Return:
        t_cmb: same dimension as s_nu
            t_cmb, unit: Kelvin_cmb
    """
    T_cmb = 2.72548  # K from Fixsen, 2009, ApJ 707 (2): 916–920
    h = constants.h.value
    c = constants.c.value
    k_b = constants.k_B.value

    x = h * nu / (k_b * T_cmb)

    slope = 2 * k_b * nu ** 2 / c ** 2 * ((x / 2) / np.sinh(x / 2)) ** 2

    return s_nu / slope


# def tb2tcmb(spectrum, nu):
#     """Convert blackbody spectrum to t_cmb
#     as defined above
#     Args:
#         tb: float or array
#             blackbody temperature, unit: Kelvin
#         nu: float or array (with same dimension as tb)
#             frequency where the spectral radiance is evaluated, unit: Hz
#     Return
#         t_cmb: same dimension as tb
#             t_cmb, unit: Kelvin_cmb
#     """
#     s_nu = tb2s(tb, nu)
#     return s2tcmb(s_nu, nu)

a = 6378137.0 #semiaxis major
b = 6356752.31424518 #semiaxis minor

def spectrum(power, fc, sigma, base, err_fc, noise, az_size, el_size, dist):

    '''
    Generate a power spectrum in W/m^2/sr/Hz for a source with a delta-like emission and
    a top-hat beam in 2D
    Parameters:
    - power: total power of the signal in dBm
    - fc: frequency of the signal in GHz
    - sigma: width of the signal in kHz
    - base: base level of the signal in dBm
    - err_fc: error in the signal frequency in kHz
    - noise: noise of the signal in W/Hz
    - ang_size: beam size of the signal in degrees as [az_size, el_size]
    - dist: distance of the source in meter
    '''

    fc *= 1e9 ### Conversion to Hz
    err_fc *= 1e3

    fc = fc+np.random.normal(0, err_fc)

    sigma *= 1e3 ### Conversion to Hz

    factor = 10 #Increase by a factor the width of the signal
    freq_step = 10

    freq = np.arange(0.5,400,freq_step)*1e9
    central = np.arange(-factor*sigma, factor*sigma, sigma)+fc

    #Add edges to fill the gap for future interpolations
    nstep = 20
    edge_l = np.linspace(fc-freq_step*1e9, fc-factor*sigma, nstep)
    edge_u = np.linspace(fc+factor*sigma, fc+freq_step*1e9, nstep)

    edges = np.append(edge_l, edge_u)
    freq = np.append(freq, central)

    freq = np.sort(np.unique(np.append(freq, edges)))

    mask = np.in1d(freq, central)

    signal = np.zeros_like(freq)

    signal[mask] = power-10*np.log10(sigma)
    signal[~mask] = base

    signal = 10**(signal/10)/1000 ### Conversion from dBm to W

    signal += np.random.normal(0, noise, size=(len(signal)))

    az_size = np.radians(az_size)/2/np.pi  #Half size in unit of pi
    el_size = np.radians(el_size)/2/np.pi  #Half size in unit of pi

    signal /= (4*np.pi*az_size*np.sin(el_size*np.pi)) #Normalize in the solid angle
    signal /= dist**2 #Normalize based on the distance

    return freq, signal

@trait_docs
class SimSource(Operator):
    """
    Operator that generates an Artificial Source timestreams.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    source_init_dist = Float(
        help = 'Initial distance of the artificial source in meters',
    )

    source_pos = List(
        required=False,
        help = 'Source Position in ECEF Coordinates as [time, X, Y, Z] in meters',
    )

    source_err_flag = Bool(
        False,
        help = 'Set True to compute the position error of the drone',
    )

    source_err = List(
        [0,0,0],
        help = 'Source Position Error in ECEF Coordinates as [[X, Y, Z]] in meters',
    )

    source_size = Float(
        0.1,
        help = 'Source Size in meters',
    )

    source_amp = Float(
        help = 'Max amplitude of the source in dBm'
    )

    source_freq = Float(
        help = 'Central frequency of the source in GHz'
    )

    source_width = Float(
        help = 'Width of the source signal in kHz'
    )

    source_baseline = Float(
        help = 'Baseline signal level of the source in dBm'
    )

    source_noise = Float(
        help = 'Noise level in W/Hz'
    )

    source_beam_az = Float(
        115,
        help = 'Beam size along the azimuthal axis of the source in degrees',
    )

    source_beam_el = Float(
        65,
        help = 'Beam size along the elevation axis of the source in degrees',
    )


    source_pol_angle = Float(
        90,
        help = 'Angle of the polarization vector emitted by the source in degrees (0 means parallel to the gorund and 90 vertical)',
    )

    source_pol_angle_error = Float(
        0,
        help = 'Error in the angle of the polarization vector',
    )

    polarization_fraction = Float(
        1,
        help = 'Polarization fraction of the emitted signal',
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

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

        for trait in "beam_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(
                    f"You must set `{trait}` before running SimSource"
                )

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

            source_az, source_el, source_dist, source_diameter = self._get_source_position(data, obs, observer, times)

            # Make sure detector data output exists
            dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=dets)

            self._observe_source(data, obs, source_az, source_el, source_dist, source_diameter, prefix, dets)

        if data.comm.group_rank == 0:
            timer.stop()
            log.debug(
                f"{data.comm.group} : Simulated and observed Source in "
                f"{timer.seconds():.1f} seconds"
            )

        return

    @function_timer
    def _get_source_position(self, data, obs, observer, times):

        log = Logger.get()
        timer = Timer()
        timer.start()

        if self.source_init_dist != 0 :
            # FIXME : Drone position should be chosen so that all
            #         detectors sweep across it, even when simulating
            #         only a subset (e.g. wafer)
            # az_init = float(np.random.choice(obs.shared[self.azimuth], 1))*u.rad
            # el_init = float(np.random.choice(obs.shared[self.elevation], 1))*u.rad
            az_init = np.median(np.array(obs.shared[self.azimuth])) * u.rad
            el_init = np.amax(np.array(obs.shared[self.elevation])) * u.rad

            E, N, U = hor2enu(az_init, el_init, self.source_init_dist)
            X, Y, Z = enu2ecef(E, N, U, observer.lat, observer.lon, observer.elevation, ell='WGS84')

            if self.source_err_flag:
                X = X+np.random.normal(0, self.source_err[0], size=(len(times)))*X.unit
                Y = Y+np.random.normal(0, self.source_err[1], size=(len(times)))*Y.unit
                Z = Z+np.random.normal(0, self.source_err[2], size=(len(times)))*z.unit
            else:
                X = np.ones_like(times)*X
                Y = np.ones_like(times)*Y
                Z = np.ones_like(times)*Z

        else:
            if self.source_pos.size > 0:
                X = self.source_pos[:,1]*u.meter
                Y = self.source_pos[:,2]*u.meter
                Z = self.source_pos[:,3]*u.meter

            else:
                raise RuntimeError(
                    "Select one between the source distance and the source ECEF position"
                )

        X_tel, Y_tel, Z_tel, _, _, _  = lonlat2ecef(observer.lat, observer.lon, observer.elevation)

        E, N, U, delta_E, delta_N, delta_U = ecef2enu(X_tel, Y_tel, Z_tel, \
                                                      X, Y, Z, \
                                                      0, 0, 0, \
                                                      self.source_err[0], self.source_err[1], self.source_err[2], \
                                                      observer.lat, observer.lon)

        source_az, source_el, source_distance, delta_az, delta_el, delta_srange = enu2hor(E, N, U, delta_E, delta_N, delta_U)



        if np.any(delta_az != 0) or np.any(delta_el != 0) or np.any(delta_srange != 0):
            if delta_az.size == 1:
                source_az = source_az+np.random.normal(0, delta_az.value, size=(len(source_az)))*source_az.unit
            else:
                source_az = source_az+np.random.normal(0, np.amax(delta_az).value, size=(len(source_az)))*source_az.unit

            if delta_el.size == 1:
                source_el = source_el+np.random.normal(0, delta_el.value, size=(len(source_el)))*source_el.unit
            else:
                source_el = source_el+np.random.normal(0, np.amax(delta_el).value, size=(len(source_el)))*source_el.unit

            if delta_srange.size == 1:
                source_distance = source_distance+np.random.normal(0, delta_srange.value, size=(len(source_distance)))*source_distance.unit
            else:
                source_distance = source_distance+np.random.normal(0, np.amax(delta_srange).value, size=(len(source_distance)))*source_distance.unit

        size = check_quantity(self.source_size, u.m)

        size = (size/source_distance)*u.rad

        if len(source_az) != len(times):
            if len(source_az) > 1:
                source_az = np.interp(times, self.source_pos[:,0], source_az.value)*source_az.unit
            else:
                source_az = np.ones_like(times)*source_az

        if len(source_el) != len(times):
            if len(source_el) > 1:
                source_el = np.interp(times, self.source_pos[:,0], source_el.value)*source_el.unit
            else:
                source_el = np.ones_like(times)*source_el

        if len(source_distance) != len(times):
            if len(source_distance) > 1:
                source_distance = np.interp(times, self.source_pos[:,0], source_distance.value)*source_distance.unit
            else:
                source_distance = np.ones_like(times)*source_distance

        if len(size) != len(times):
            if len(size) > 1:
                size = np.interp(times, self.source_pos[:,0], size.value)*size.unit
            else:
                size = np.ones_like(times)*size

        obs['source_az'] = source_az
        obs['source_el'] = source_el

        if data.comm.group_rank == 0:
            timer.stop()
            log.verbose(
                f"{data.comm.group} : Computed source position in "
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

        freq, spec = spectrum(amp, fc, sigma, base, sigma, noise, \
                              self.source_beam_az, self.source_beam_el, dist)

        freq = freq *u.Hz

        temp = s2tcmb(spec, freq.to_value(u.Hz)) * u.K

        return freq, temp

    def _get_beam_map(self, det, source_diameter, ttemp_det):
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
        source_radius_avg = np.average(source_diameter) / 2
        source_solid_angle = np.pi * source_radius_avg ** 2

        amp = ttemp_det * (
            source_solid_angle.to_value(u.rad ** 2)/beam_solid_angle.to_value(u.rad ** 2)
        )
        w = np.radians(size / 2)
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        model *= amp
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w ** 2 + w ** 2)
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
    ):
        """
        Observe the Source with each detector in tod
        """
        log = Logger.get()
        timer = Timer()

        source_freq, source_temp = self._get_source_temp(source_dist)

        for det in dets:
            timer.clear()
            timer.start()
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            # Compute detector quaternions and Stokes weights

            obs_data = Data(comm=data.comm)
            obs_data._internal = data._internal
            obs_data.obs = [obs]
            self.detector_pointing.apply(obs_data, detectors=[det])
            self.detector_weights.apply(obs_data, detectors=[det])
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
            det_temp = bandpass.convolve(det, source_freq, source_temp)

            beam, radius = self._get_beam_map(det, source_diameter, det_temp)

            # Interpolate the beam map at appropriate locations
            x = (az - source_az.to_value(u.rad)) * np.cos(el)
            y = el - source_el.to_value(u.rad)
            r = np.sqrt(x ** 2 + y ** 2)
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
            angle = np.radians(self.source_pol_angle + np.random.normal(0, self.source_pol_angle_error, size=(len(sig))))

            sig *= weights_I + pfrac * (np.cos(angle)*weights_Q + np.sin(angle)*weights_U)

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
            ]
        }

    def _accelerators(self):
        return list()
