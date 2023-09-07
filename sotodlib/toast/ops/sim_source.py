# Copyright (c) 2018-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import h5py

import traitlets

import numpy as np
import copy

from astropy import units as u

import ephem

from scipy.interpolate import RectBivariateSpline
from scipy.signal import square

from toast.timing import function_timer

from toast import qarray as qa

from toast.data import Data

from toast.traits import trait_docs, Int, Unicode, Float, Instance, List, Quantity

from toast.ops.operator import Operator
from toast.instrument import Focalplane

from toast.utils import Logger, Timer, unit_conversion

from toast.observation import default_values as defaults

from sotodlib.coords import local

from . import utils

def source_spectrum(fc, width, power, gain, amplitude, diameter, noise_bw, noise_out):

    """Generate a power spectrum for a source.

    The source has a delta-like emission.

    Args:
        fc (Quantity): Central frequency of the source signal
        width (Quantity): total width of the source signal
        power (Float): total power of the source signal in dBm
        gain (Float): Antenna gain of the source in dBi
        amplitude (Float): amplitude of the signal with respect to the the base in dB
        diameter (Quantity): source diameter
        noise_bw (Quantity): in-band noise
        noise_out (Quantity): out-of-band noise
    Returns:
        I_nu (Quantity): spectrum of the source

    """

    amplitude = 10**(-amplitude/10)
    power_watts = 10**(power/10)/1000

    freqs = np.linspace(20., 350., 1001, endpoint=True)*1e9 * u.Hz
    fc = fc.to(u.Hz)
    width = width.to(u.Hz)
    radius = diameter.to(u.meter)/2
    
    if np.amin(np.diff(freqs)) >= 2*width:
        idx = np.argmin(np.abs(freqs-fc))
        
        idx = np.array([idx-1, idx, idx+1])
    
    else:
        fmin = fc-width
        fmax = fc+width
        
        idx_min = np.argmin(np.abs(freqs-fmin))
        idx_max = np.argmin(np.abs(freqs-fmax))
        
        idx = np.arange(idx_min, idx_max+1, dtype=int)
        
    bandwidth = np.ptp(freqs[idx])
        
    rho_bw = power_watts/bandwidth/(1+amplitude) * u.W

    power_bw = rho_bw*bandwidth

    out1 = np.ptp(freqs[:idx[0]])
    out2 = np.ptp(freqs[idx[-1]+1:])

    rho_out = power_bw*amplitude/(out1+out2)

    signal = np.zeros(len(freqs)) * u.W / u.Hz

    if noise_bw.value != 0:
        rho_bw += np.random.normal(0, np.abs(noise_bw), len(rho_bw))

    if noise_out.value != 0:
        rho_out += np.random.normal(0, np.abs(noise_out), len(rho_out))

    signal[:idx[0]] = rho_out
    signal[idx] = rho_bw
    signal[idx[-1]+1:] = rho_out
    
    g = 10**(gain/10)
       
    I_nu = signal*g/np.pi/radius**2
    
    return freqs, I_nu

class SimulateDroneMovement:
    
    def __init__(self, params):

        """Class containing the necessary functions to simulate a drone scanning above the telescope.

        Args:
            params (Dictionary): Dictionary with the parameters required to generate the movement of the drone.
                                 Keys available on the dictionary:
                                    azimuth_starting (Quantity): Starting value for the azimuth
                                    azimuth_range (Quantity): Range of the azimuth scan
                                    azimuth_velocity (Quantity): Maximum velocity of the drone along the azimuthal axis
                                    azimuth_acceleration (Quantity): Maximum velocity of the drone along the azimuthal axis
                                    azimuth_direction (String): Determine if the azimuth value is increasing or decreasing.
                                                                Accepted values: 'increasing' or 'decresing'
                                    elevation_starting (Quantity): Starting value for the elevation
                                    elevation_range (Quantity): Range of the elevation scan
                                    elevation_velocity (Quantity): Maximum velocity of the drone along the elevational axis
                                    elevation_acceleration (Quantity): Maximum velocity of the drone along the elevational axis
                                    elevation_direction (String): Determine if the elevation value is increasing or decreasing.
                                                                  Accepted values: 'increasing' or 'decresing'
                                    elevation_step (Quantity): Step along the elevation axis for a grid scan
                                    scan_time (Quantity): Time of the scan
                                    scan_type (String): Type of the scan
                                                        Accepted values:
                                                            elevation_only
                                                            azimuth_only
                                                            elevation_only_single
                                                            azimuth_only_single
                                                            grid
                                                            fixed
        
        """
        
        self.params = params
        
    def _check_maximum_velocity(self, delta_axis, vel, acc, direction):

        """Internal Function to control if the drone achieves the maximum velocity along a specific axis.
        Args:
            delta_axis (Quantity): span of the scan
            vel (Quantity): maximum velocity achievable
            acc (Quantity): value of the acceleration
            direction (Float): value that indicates the direction of movement

        Returns:
            t_acc (Quantity): total time of the acceleration (deceleration) phase
            t_vel (Quantity): total time at the maximum velocity achivable (0 if v_max < vel)
            v_max (Quantity): maximum velocity of the drone
        
        """
        
        fact = delta_axis-np.abs(vel**2/acc)
        
        if fact > 0 :
            t_acc = np.abs(vel/acc)
            t_vel = (delta_axis-np.abs(vel**2/acc))/np.abs(vel)
            v_max = vel
        else:
            v_max = np.sqrt(np.abs(delta_axis*acc))*direction
            t_acc = np.abs(v_max/acc)
            t_vel = 0
            
        return t_acc, t_vel, v_max

    def create_scan(self, time_interp=None):

        """ Calculate the movement of the drone using the parameters dictionary.
        Args:
            times_interp (Quantity): times where to compute the position using a linear interpolation

        Return:
            time (Quantity): total duration of the scan
            azimuth (Quantity): positions during the scan along the azimuth axis
            elevation (Quantity): positions during the scan along the elevation axis
        """

        if self.params['scan_type'] == 'elevation_only':
            time, azimuth, elevation = self.single_axis_scan(axis='elevation')
        elif self.params['scan_type'] == 'azimuth_only':
            time, azimuth, elevation = self.single_axis_scan(axis='azimuth')
        elif self.params['scan_type'] == 'elevation_only_single':

            if self.params['elevation_direction'].lower() == 'decreasing':
                direction = -1
            else:
                direction = 1

            time, elevation = self.single_movement(axis = 'elevation',
                                                   direction = direction,
                                                   axis0 = self.params['elevation_starting'])

            azimuth = np.ones(len(elevation))*self.params['azimuth_starting']

        elif self.params['scan_type'] == 'azimuth_only_single':

            if self.params['azimuth_direction'].lower() == 'decreasing':
                direction = -1
            else:
                direction = 1

            time, azimuth = self.single_movement(axis = 'azimuth',
                                                 direction = direction,
                                                 axis0 = self.params['azimuth_starting'])

            elevation = np.ones(len(azimuth))*self.params['elevation_starting']

        elif self.params['scan_type'] == 'grid_scan':

            time, azimuth, elevation = self.grid_scan()
        
        elif self.params['scan_type'] == 'fixed':

            time = np.linspace(0, self.params['scan_time'], 1001, endpoint=True)
            azimuth = np.ones(len(time))*self.params['azimuth_starting']
            elevation = np.ones(len(time))*self.params['elevation_starting']

        if time_interp is not None:

            azimuth = np.interp(time_interp-time_interp[0], time, azimuth)
            elevation = np.interp(time_interp-time_interp[0], time, elevation)

            time = copy.copy(time_interp)
        
        return time, azimuth, elevation

    def single_movement(self, axis=None, direction=1, axis0=0, step=False, step_value=0):

        """ Calculate the movement along a specific axis for a single movement (i.e. up to down or down to up,
        left to right or right to left).
        Args:
            axis (String): axis along the movement needs to be generated
            direction (Float): value that indicates the direction of movement
            axis0 (Quantity): Starting point of the movement
            step (Bool): True is the movement is a stepping movement during a grid scan
            step_value (Quantity): range value of the step

        Return:
            time (Quantity): total duration of the single movement
            axis_val (Quantity): positions during the single movement
        """
        
        range_string = axis+'_range'
        vel_string = axis+'_velocity'
        acc_string = axis+'_acceleration'
        
        vel = self.params[vel_string]*direction
        acc = self.params[acc_string]*direction
        if step:
            delta_axis = step_value
        else:
            delta_axis = self.params[range_string]
            
        t_acc, t_vel, v_max = self._check_maximum_velocity(delta_axis, vel, acc, direction)
        
        axis_val = np.array([]) * u.rad
        time = np.array([]) * u.s
        
        time_acc = np.linspace(0, t_acc, 101, endpoint=True)
        time = np.append(time, time_acc)
        
        dx_acc = 0.5*acc*time_acc**2
        axis_val = np.append(axis_val, axis0+dx_acc)

        if t_vel > 0:
            time_vel = np.linspace(0, t_vel, 101 , endpoint=True)
            time = np.append(time, time[-1]+time_vel[1:])

            dx_vel = v_max*time_vel[1:]
            axis_val = np.append(axis_val, axis_val[-1]+dx_vel)
            
        dx_dec = v_max*time_acc[1:]-0.5*acc*time_acc[1:]**2
        axis_val = np.append(axis_val, axis_val[-1]+dx_dec)
        
        time = np.append(time, time[-1]+time_acc[1:])
        
        return time, axis_val
    
    def single_axis_scan(self, axis='elevation'):

        """ Calculate the movement along a specific axis for a scan (i.e. multiple up to down/down to up,
        multiple left to right/right to left).
        Args:
            axis (String): axis along the movement needs to be generated

        Return:
            time (Quantity): total duration of the scan
            azimuth (Quantity): positions during the scan along the azimuth axis
            elevation (Quantity): positions during the scan along the elevation axis
        """
        
        axis_value = np.array([]) * u.rad
        time = np.array([]) * u.s
        
        axis0_string = axis+'_starting'
        
        axis0 = copy.copy(self.params[axis0_string])
        t0 = 0
        
        if self.params[axis+'_direction'].lower() == 'decreasing':
            direction = -1
        else:
            direction = 1
            
        self.scan_cycles = 0
        self._scan_idxs = np.array([], dtype=int)
        
        start = axis+'_starting'
        rng = axis+'_range'
        
        while True:
            time_temp, axis_temp = self.single_movement(axis=axis,
                                                        direction=direction,
                                                        axis0 = axis0)
            axis0 = axis_temp[-1]
            direction *= -1
            
            time = np.append(time, time_temp+t0)
            axis_value = np.append(axis_value, axis_temp)
            t0 = copy.copy(time[-1])
            
            self.scan_cycles += 1
            
            if time[-1] > self.params['scan_time']:
                
                idx = np.argmin(np.abs(time-self.params['scan_time']))
                
                time = time[:idx+1]
                axis_value = axis_value[:idx+1]
                self._scan_idxs = np.append(self._scan_idxs, len(time)-1)
                
                if direction == 1:
                    self.scan_cycles += (axis_value[-1]-self.params[start])/self.params[rng]
                elif direction == -1:
                    self.scan_cycles += ((self.params[start]+self.params[rng]-axis_value[-1]))/self.params[rng]
                
                break
            
            self._scan_idxs = np.append(self._scan_idxs, len(time)-1)
            
        self.scan_cycles /= 2
        self.scan_time = time
        if axis == 'elevation':
            elevation = axis_value
            azimuth = np.ones(len(axis_value))*self.params['azimuth_starting']
        if axis == 'azimuth':
            azimuth = axis_value
            elevation = np.ones(len(axis_value))*self.params['elevation_starting']
            
        return time, azimuth, elevation
    
    def grid_scan(self):

        """ Calculate the movement during a grid scan with an elevation step.
        Args:
            None
        Return:
            time (Quantity): total duration of the scan
            azimuth (Quantity): positions during the scan along the azimuth axis
            elevation (Quantity): positions during the scan along the elevation axis
        """
        
        el_step = self.params['el_step']
        
        az0 = self.params['azimuth_starting']
        el0 = self.params['elevation_starting']
        
        if self.params['azimuth_direction'].lower() == 'decreasing':
            azimuth_direction = -1
        else:
            azimuth_direction = 1
        
        if self.params['elevation_direction'].lower() == 'decreasing':
            elevation_direction = -1
        else:
            elevation_direction = 1
        
        el_cumulative = 0 * u.degree
        t0 = 0 * u.s
        
        time = np.array([]) * u.s
        azimuth_value = np.array([]) * u.rad
        elevation_value = np.array([]) * u.rad
        
        self.scan_cycles = 0
        self._scan_idxs = np.array([], dtype=int)
        
        while True:
            
            time_temp, axis_temp = self.single_movement(axis = 'azimuth',
                                                        direction = azimuth_direction,
                                                        axis0 = az0,
                                                        points = 101)
            
            azimuth_direction *= -1
            az0 = copy.copy(axis_temp[-1])
            
            time = np.append(time, time_temp[1:]+t0)
            azimuth_value = np.append(azimuth_value, axis_temp[1:])
            elevation_value = np.append(elevation_value, np.ones_like(axis_temp[1:])*el0)
            t0 = copy.copy(time[-1])
            
            if time[-1] > self.params['scan_time']:
                
                idx = np.argmin(np.abs(time-self.params['scan_time']))
                
                time = time[:idx+1]
                azimuth_value = azimuth_value[:idx+1]
                elevation_value = elevation_value[:idx+1]
                
                break
                
            el_cumulative += el_step
            
            if el_cumulative >= self.params['elevation_range']:
                time_temp, axis_temp = self.single_movement(axis='elevation',
                                                            direction=elevation_direction,
                                                            axis0=el0,
                                                            step=True,
                                                            step_value=np.abs(self.params['elevation_range']-el_cumulative),
                                                            points=101)
                el_cumulative = 0
                elevation_direction *= -1
                self.scan_cycles += 1
                self._scan_idxs = np.append(self._scan_idxs, len(time)+len(time_temp)-1)
            else:
                time_temp, axis_temp = self.single_movement(axis='elevation',
                                                            direction=elevation_direction,
                                                            axis0=el0,
                                                            step=True,
                                                            step_value=el_step,
                                                            points=101)

            el0 = copy.copy(axis_temp[-1])
            time = np.append(time, time_temp[1:]+t0)
            elevation_value = np.append(elevation_value, axis_temp[1:])
            azimuth_value = np.append(azimuth_value, np.ones_like(axis_temp[1:])*az0)
            
            t0 = copy.copy(time[-1])
            
            if time[-1] > self.params['scan_time']:
                
                idx = np.argmin(np.abs(time-self.params['scan_time']))
                
                time = time[:idx+1]
                azimuth_value = azimuth_value[:idx+1]
                elevation_value = elevation_value[:idx+1]
                
                self._scan_idxs = np.append(self._scan_idxs, len(time)-1)
                
                if elevation_direction == 1:
                    self.scan_cycles += (elevation_value[-1]-self.params['elevation_starting'])/self.params['elevation_range']
                elif elevation_direction == -1:
                    self.scan_cycles += ((self.params['elevation_starting']+self.params['elevation_range']-elevation_value[-1]))/self.params['elevation_range']
                
                break
                
        idx, = np.where(np.diff(time)!=-1000)
            
        azimuth = azimuth_value
        elevation = elevation_value

        return time, azimuth, elevation

@trait_docs
class SimSource(Operator):
    """Operator that generates an Artificial Source timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    # Drone Position parameters

    source_distance = Quantity(
        500*u.meter,
        help="Initial distance of the artificial source in meters",
    )

    source_azimuth_range = Quantity(
        u.Quantity(0, u.degree),
        help = 'Range of the scan along the azimuthal axis'
    )

    source_azimuth_velocity = Quantity(
        u.Quantity(2, u.Unit("deg / s")),
        help = 'Maximum velocity of the drone along the azimuthal axis'
    )

    source_azimuth_acceleration = Quantity(
        u.Quantity(2, u.Unit("deg / s / s")),
        help = 'Maximum acceleration of the drone along the azimuthal axis'
    )

    source_azimuth_direction = Unicode(
        'increasing',
        help = 'Determine if the azimuth value is increasing or decreasing'
    )

    source_elevation_range = Quantity(
        u.Quantity(0, u.degree),
        help = 'Range of the scan along the elevation axis'
    )

    source_elevation_velocity = Quantity(
        u.Quantity(2, u.Unit("deg / s")),
        help = 'Maximum velocity of the drone along the elevation axis'
    )

    source_elevation_acceleration = Quantity(
        u.Quantity(2, u.Unit("deg / s / s")),
        help = 'Maximum acceleration of the drone along the elevation axis'
    )

    source_elevation_direction = Unicode(
        'increasing',
        help = 'Determine if the elevation value is increasing or decreasing'
    )

    source_elevation_step = Quantity(
        u.Quantity(0, u.degree),
        help = 'Step along the elevation axis for a grid scan'
    )

    source_scan_type = Unicode(
        'elevation_only',
        help = 'Type of the scan'
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

    source_size = Quantity(
        u.Quantity(0.1, u.meter),
        help = 'Source Size in meters',
    )

    source_power = Float(
        help = 'Max amplitude of the source in dBm'
    )

    source_fc = Quantity(
        u.Quantity(90e9, u.Hz),
        help = 'Central frequency of the source'
    )

    source_width = Quantity(
        u.Quantity(100e3, u.Hz),
        help = 'Width of the source signal'
    )

    source_amplitude = Float(
        help = 'Amplitude of the source signal with the respect to the background in dB'
    )

    source_gain = Float(
        help = 'Gain of the source Antenna in dBi'
    )

    source_noise_bw = Quantity(
        u.Quantity(0, u.W / u.Hz),
        help = 'White noise level in the emission band'
    )

    source_noise_out = Quantity(
        u.Quantity(0, u.W / u.Hz),
        help = 'White noise level outside the emission band'
    )

    source_freq_chopping = Quantity(
        u.Quantity(0, u.Hz),
        help = 'Frequency of the source chopping system'
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

    #Drone parameters

    drone_temp = Quantity(
        u.Quantity(0, u.K),
        help = 'BlackBody temperature of the Drone'
    )

    drone_emiss = Float(
        0,
        help = 'Emissivity of the drone'
    )

    drone_size = Quantity(
        u.Quantity(0, u.meter),
        help = 'Drone size in meters',
    )

    #Wind Parameters

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
                source_diameter
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
                times
            )

        if data.comm.group_rank == 0:
            timer.stop()
            log.debug(
                f"{data.comm.group} : Simulated and observed Source in "
                f"{timer.seconds():.1f} seconds"
            )

        return

    def _check_scan_type(self, scan_string):

        log = Logger.get()

        available_scans = ['elevation_only', 'azimuth_only', 'elevation_only_single', \
                           'azimuth_only_single', 'grid', 'fixed']

        if scan_string in available_scans:
            return scan_string
        else:
            log.debug(
                "Invalid Scan Type input"
                "Set scan to elevation_only"
            )

            return 'elevation_only'

    @function_timer
    def _get_source_position(self, obs, observer, times):

        log = Logger.get()
        timer = Timer()
        timer.start()

        scan_type = self._check_scan_type(self.source_scan_type)

        FoV = self.focalplane.field_of_view

        if self.source_azimuth_range == 0:
            az_range = FoV
        else:
            az_range = self.source_azimuth_range
        if self.source_elevation_range == 0:
            el_range = FoV
        else:
            el_range = self.source_elevation_range

        if scan_type == 'azimuth_only' or scan_type == 'azimuth_only_single':

            if self.source_azimuth_direction == 'increasing':
                az_start = np.amin(np.array(obs.shared[self.azimuth])) * u.rad
                az_start -= az_range/2
            else:
                az_start = np.amax(np.array(obs.shared[self.azimuth])) * u.rad
                az_start += az_range/2
            
            el_start = np.median(np.array(obs.shared[self.elevation])) * u.rad

        elif scan_type == 'elevation_only' or scan_type == 'elevation_only_single':

            if self.source_elevation_direction == 'increasing':
                el_start = np.amin(np.array(obs.shared[self.elevation])) * u.rad
                el_start -= el_range/2
            else:
                el_start = np.amax(np.array(obs.shared[self.elevation])) * u.rad
                el_start += el_range/2
            
            az_start = np.median(np.array(obs.shared[self.azimuth])) * u.rad

        elif scan_type == 'grid_scan':

            if self.source_elevation_direction == 'increasing':
                el_start = np.amin(np.array(obs.shared[self.elevation])) * u.rad
                el_start -= el_range/2
            else:
                el_start = np.amax(np.array(obs.shared[self.elevation])) * u.rad
                el_start += el_range/2

            if self.source_azimuth_direction == 'increasing':
                az_start = np.amin(np.array(obs.shared[self.azimuth])) * u.rad
                az_start -= az_range/2
            else:
                az_start = np.amax(np.array(obs.shared[self.azimuth])) * u.rad
                az_start += az_range/2

        source_scan_params = {
            'azimuth_starting': az_start,
            'azimuth_range': az_range,
            'azimuth_velocity': self.source_azimuth_velocity,
            'azimuth_acceleration': self.source_azimuth_acceleration,
            'azimuth_direction': self.source_azimuth_direction,
            'elevation_starting': el_start,
            'elevation_range': el_range,
            'elevation_velocity': self.source_elevation_velocity,
            'elevation_acceleration': self.source_elevation_acceleration,
            'elevation_direction': self.source_elevation_direction,
            'elevation_step': self.source_elevation_step,
            'scan_time': np.ptp(times) * u.s,
            'scan_type': scan_type
        }

        scan = SimulateDroneMovement(source_scan_params)

        _, source_azimuth, source_elevation = scan.create_scan(time_interp=times*u.s)

        if np.any(np.array(self.source_err) >= 1e-4) or self.wind_gusts_amp.value != 0:

            E, N, U = local.hor2enu(source_azimuth, source_elevation, self.source_distance)
            X, Y, Z = local.enu2ecef(E, N, U, observer.lon, observer.lat, observer.elevation, ell='WGS84')

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

            source_az, source_el, source_dist, _,_,_ = local.enu2hor(E, N, U, 0,0,0)

        else:
            source_az = source_azimuth.copy()
            source_el = source_elevation.copy()
            source_dist = self.source_distance.copy()

        size = local._check_quantity(self.source_size, u.m)
        size = (size/source_dist)*u.rad

        obs['source_az'] = source_az
        obs['source_el'] = source_el
        obs['source_distance'] = source_dist

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

        return source_az, source_el, source_dist, size

    def _get_source_temp(self):

        power = self.source_power
        fc = self.source_fc
        width = self.source_width
        gain = self.source_gain
        diameter = self.source_size
        amplitude = self.source_amplitude
        noise_bw = self.source_noise_bw
        noise_out = self.source_noise_out
        
        freq, spec = source_spectrum(fc, width, power, gain, amplitude, diameter, noise_bw, noise_out)

        temp = utils.s2tcmb(spec, freq)

        return freq, temp

    def _get_drone_temp(self):
        """
        Compute the drone CMB temperature as a grey body
        """

        freqs = np.linspace(20., 350., 1001, endpoint=True)*1e9 * u.Hz
        temp = utils.tb2tcmb(self.drone_temp, freqs)*self.drone_emiss

        return freqs, temp

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
        times
    ):
        """
        Observe the Source with each detector in tod
        """
        log = Logger.get()
        timer = Timer()

        source_freq, source_temp = self._get_source_temp()
        
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

            sig = np.zeros(len(times))
            sig[good] = beam(x[good], y[good], grid=False)

            if self.drone_temp > 250*u.K and self.drone_emiss > 0:
                
                drone_diameter = local._check_quantity(self.drone_size, u.m)
                drone_diameter = (drone_diameter/source_dist)*u.rad

                drone_freq, drone_temp = self._get_drone_temp()
                det_drone_temp = bandpass.convolve(det, drone_freq, drone_temp)

                beam_drone = self._get_beam_map(det, drone_diameter, det_drone_temp)

                sig_drone = np.zeros(len(times))
                sig_drone[good] = beam_drone(x[good], y[good], grid=False)

                sig += sig_drone

            if self.source_freq_chopping.value > 0:
                sampling = 1/np.mean(np.diff(times))
                chop = square(2*np.pi*sampling*self.source_freq_chopping)
                chop[chop<1] = 0

                sig *= chop

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

            pfrac = self.polarization_fraction
            angle = np.radians(
                self.source_pol_angle
                + np.random.normal(0, self.source_pol_angle_error, size=(len(sig)))
            )
            
            I = sig.copy()
            Q = pfrac * sig * np.cos(2 * angle)
            U = pfrac * sig * np.sin(2 * angle)

            drone_sig = (I * weights_I
                         + Q * weights_Q
                         + U * weights_U
                         )

            signal += drone_sig

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
