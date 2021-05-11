# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import numpy as np
from toast.utils import Logger
from toast.timing import function_timer, Timer
from toast.op import Operator
import ephem
from astropy import constants
from toast.mpi import MPI
import toast.qarray as qa
import healpy as hp

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

    

class OpFlagSSO(Operator):
    """Operator which flags detector data in the vicinity of solar system objects

    Args:
        sso_name (str):  Name of the SSO, must be recognized by pyEphem
        sso_radius (float):  Radius around the source to flag [radians]
        flag_name (str):  Detector flags to modify
        flag_mask (str):  Flag bits to raise
        out (str): accumulate data to the cache with name
            <out>_<detector>.  If the named cache objects do not exist,
            then they are created.
        report_timing (bool):  Print out time taken to initialize,
             simulate and observe
    """

    def __init__(self, sso_name, sso_radius, flag_name="flags", flag_mask=1):
        # Call the parent class constructor
        super().__init__()

        self.sso_name = sso_name
        self.sso = getattr(ephem, sso_name)()
        self.sso_radius = sso_radius
        self.flag_name = flag_name
        self.flag_mask = flag_mask
        return

    @function_timer
    def exec(self, data):
        """Generate and apply flags
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

            tod = obs['tod']

            # Get the observation time span and compute the horizontal
            # position of the SSO
            times = tod.local_times()
            sso_az, sso_el = self._get_sso_position(times, observer)

            self._flag_sso(sso_az, sso_el, tod, focalplane)

            del sso_az, sso_el

        return
    
    @function_timer
    def _get_sso_position(self, times, observer):
        """
        Calculate the SSO horizontal position
        """
        sso_az = np.zeros(times.size)
        sso_el = np.zeros(times.size)
        for i, t in enumerate(times):
            observer.date = to_DJD(t)
            self.sso.compute(observer)
            sso_az[i] = self.sso.az
            sso_el[i] = self.sso.alt
        return sso_az, sso_el

    @function_timer
    def _flag_sso(self, sso_az, sso_el, tod, focalplane):
        """
        Flag the SSO for each detector in tod
        """

        nsamp = tod.local_samples[1]

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self.flag_name, det)
            if tod.cache.exists(cachename):
                ref = tod.cache.reference(cachename)
            else:
                ref = tod.cache.create(cachename, np.uint8, (nsamp,))

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

            # Flag samples within search radius
            x = (az - sso_az) * np.cos(el)
            y = el - sso_el
            r = np.sqrt(x ** 2 + y ** 2)
            good = r < self.radius
            ref[good] |= self.flag_mask

            del ref

        return
