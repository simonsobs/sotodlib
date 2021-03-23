# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import numpy as np
from toast.utils import Logger
from toast.timing import function_timer, Timer
from toast.op import Operator
import ephem
from toast.mpi import MPI
import toast.qarray as qa
import healpy as hp
from scipy.constants import au as AU
from scipy.interpolate import RectBivariateSpline
import pickle
import h5py

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


class OpSimSSO(Operator):
    """Operator which generates Solar System Object timestreams.

    Args:
        name (str): Name of the SSO, must be recognized by pyEphem
        beam_file: the pickle file that stores the simulated beam
        out (str): accumulate data to the cache with name
            <out>_<detector>.  If the named cache objects do not exist,
            then they are created.
        report_timing (bool):  Print out time taken to initialize,
             simulate and observe
    """

    def __init__(self, sso_name,  beam_file, out="sso", report_timing=False):
        # Call the parent class constructor
        super().__init__()

        self.sso_name = sso_name
        self.sso = getattr(ephem, sso_name)()
        self._beam_file = beam_file
        self._out = out
        self._report_timing = report_timing
        return

    @function_timer
    def exec(self, data):
        """Generate timestreams.
        Args:
            data (toast.Data): The distributed data.
        Returns:
            None
        """

        log = Logger.get()
        group = data.comm.group
        for obs in data.obs:
            try:
                obsname = obs["name"]
                focalplane = obs["focalplane"]
            except Exception:
                obsname = "observation"
                focalplane = None

            observer = ephem.Observer()
            observer.lon = obs['site'].lon
            observer.lat = obs['site'].lat
            observer.elevation = obs['site'].alt  # In meters
            observer.epoch = "2000"
            observer.temp = 0  # in Celcius
            observer.compute_pressure()

            prefix = "{} : {} : ".format(group, obsname)
            tod = obs['tod']
            comm = tod.mpicomm
            rank = 0
            if comm is not None:
                rank = comm.rank
            site = obs['site'].id

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                log.info("{}Setting up SSO simulation".format(prefix))

            # Get the observation time span and compute the horizontal
            # position of the SSO
            times = tod.local_times()
            sso_az, sso_el, sso_dist, sso_dia = self._get_sso_position(times, observer)

            tmr = Timer()
            if self._report_timing:
                if comm is not None:
                    comm.Barrier()
                tmr.start()

            self._observe_sso(sso_az, sso_el, sso_dist, sso_dia, tod, comm, prefix, focalplane)

            del sso_az, sso_el, sso_dist

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}Simulated and observed SSO signal".format(prefix))
        return
    
    def _get_planet_temp(self, sso_name):
        """
        Get the thermodynamic planet temperature given
        the frequency
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        hf = h5py.File(os.path.join(dir_path, 'data/planet_data.h5'), 'r')
        if sso_name in hf.keys():
            self.ttemp = np.array(hf.get(sso_name))
            self.t_freqs = np.array(hf.get('freqs_ghz'))
        else:
            raise ValueError('Unknown planet name')


    def _get_beam_map(self, det, sso_dia, ttemp_det):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        #Read in the simulated beam
        with open(self._beam_file, 'rb') as f_t:
            beam_dic = pickle.load(f_t)
        description = beam_dic['size'] # 2d array [[size, res], [n, 1]]
        model = beam_dic['data']
        res = description[0][1]
        beam_solid_angle = np.sum(model)*np.radians(res)**2

        n = int(description[1][0])
        size = description[0][0]
        sso_radius_avg = np.average(sso_dia)/2. # in arcsed
        sso_solid_angle = np.pi*np.radians(sso_radius_avg/3600)**2
        amp = ttemp_det * sso_solid_angle/beam_solid_angle
        w = np.radians(size/2)
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        model *= amp
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w ** 2 + w ** 2)
        return beam, r

    @function_timer
    def _get_sso_position(self, times, observer):
        """
        Calculate the SSO horizontal position
        """
        # FIXME: we could parallelize here and also interpolate the
        # SSO position from a low sample rate vector
        """
        tmin = times[0]
        tmax = times[-1]
        tmin_tot = tmin
        tmax_tot = tmax
        if comm is not None:
            tmin_tot = comm.allreduce(tmin, op=MPI.MIN)
            tmax_tot = comm.allreduce(tmax, op=MPI.MAX)
        """
        sso_az = np.zeros(times.size)
        sso_el = np.zeros(times.size)
        sso_dist = np.zeros(times.size)
        sso_dia = np.zeros(times.size)
        for i, t in enumerate(times):
            observer.date = to_DJD(t)
            self.sso.compute(observer)
            sso_az[i] = self.sso.az
            sso_el[i] = self.sso.alt
            sso_dist[i] = self.sso.earth_distance * AU
            sso_dia[i] = self.sso.size
        return sso_az, sso_el, sso_dist, sso_dia

    @function_timer
    def _observe_sso(self, sso_az, sso_el, sso_dist, sso_dia, tod, comm, prefix, focalplane):
        """
        Observe the SSO with each detector in tod
        """
        log = Logger.get()
        rank = 0
        if comm is not None:
            rank = comm.rank
        tmr = Timer()
        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            tmr.start()

        nsamp = tod.local_samples[1]

        if rank == 0:
            log.info("{}Observing the SSO signal".format(prefix))
        
        band_dict = {'f030' : 27, 'f040': 39, 'f090': 93,
             '150': 145 , 'f230': 225, 'f290': 285}

        for band in band_dict.keys():
            if band in prefix:
               freq = band_dict[band]
               break

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self._out, det)
            if tod.cache.exists(cachename):
                ref = tod.cache.reference(cachename)
            else:
                ref = tod.cache.create(cachename, np.float64, (nsamp,))

            try:
                # Some TOD classes provide a shortcut to Az/El
                az, el = tod.read_azel(detector=det)
            except Exception as e:
                azelquat = tod.read_pntg(detector=det, azel=True)
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, phi = qa.to_position(azelquat)
                # Azimuth is measured in the opposite direction
                # than longitude
                az = 2 * np.pi - phi
                el = np.pi / 2 - theta

            if "bandpass_transmission" in focalplane[det]:
                # We have full bandpasses for the detector
                bandpass_freqs = focalplane[det]["bandpass_freq_ghz"]
                bandpass = focalplane[det]["bandpass_transmission"]
            else:
                if "bandcenter_ghz" in focalplane[det]:
                    # Use detector bandpass from the focalplane
                    center = focalplane[det]["bandcenter_ghz"]
                    width = focalplane[det]["bandwidth_ghz"]
                else:
                    # Use default values for the entire focalplane
                    if freq is None:
                        raise RuntimeError(
                            "You must supply the nominal frequency if bandpasses "
                            "are not available"
                        )
                    center = freq
                    width = 0.2 * freq
                bandpass_freqs = np.array([center - width / 2, center + width / 2])
                bandpass = np.ones(2)   
        
            nstep = 1001
            fmin, fmax = bandpass_freqs[0], bandpass_freqs[-1]
            det_freqs = np.linspace(fmin, fmax, nstep)
            det_bandpass = np.interp(det_freqs, bandpass_freqs, bandpass)
            det_bandpass /= np.sum(det_bandpass)

            self._get_planet_temp(self.sso_name)
            ttemp_det = np.interp(det_freqs, self.t_freqs, self.ttemp)
            ttemp_det = np.sum(ttemp_det * det_bandpass)
            beam, radius = self._get_beam_map(det, sso_dia, ttemp_det)

            # Interpolate the beam map at appropriate locations
            x = (az - sso_az) * np.cos(el)
            y = el - sso_el
            r = np.sqrt(x ** 2 + y ** 2)
            good = r < radius
            sig = beam(x[good], y[good], grid=False)
            ref[:][good] += sig

            del ref, sig, beam

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}OpSimSSO: Observe signal".format(prefix))
        return
