# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import pickle
import sys

import numpy as np

import toast
import toast.qarray as qa


XAXIS, YAXIS, ZAXIS = np.eye(3)


class OpSimHWPSS(toast.Operator):
    """ Simulate HWP synchronous signal """

    def __init__(self, name, fname_hwpss, mc=0):
        """
        Arguments:
        name(str) : Cache prefix to operate on
        """
        self._name = name
        if not os.path.isfile(fname_hwpss):
            raise RuntimeError(f"{fname_hwpss} does not exist!")
        self.fname_hwpss = fname_hwpss

        with open(fname_hwpss, "rb") as fin:
            self.thetas, self.chis, self.all_stokes = pickle.load(fin)

        self._mc = mc

        return

    def exec(self, data):
        
        for obs in data.obs:
            tod = obs["tod"]
            focalplane = obs["focalplane"]
            # Get HWP angle
            chi = tod.local_hwp_angle()
            for det in tod.local_dets:
                signal = tod.local_signal(det, self._name)
                band = focalplane[det]["band"]
                freq = {
                    "f030" : "027",
                    "f040" : "039",
                    "f090" : "093",
                    "f150" : "145",
                    "f230" : "225",
                    "f290" : "278",
                }[band]

                # Get incident angle
                
                det_quat = focalplane[det]["quat"]
                theta, phi = qa.to_position(det_quat)
                
                # Get observing elevation
                
                try:
                    # Some TOD classes provide a shortcut to Az/El
                    az, el = tod.read_azel(detector=det)
                except Exception as e:
                    azelquat = tod.read_pntg(detector=det, azel=True)
                    el = np.pi / 2 - qa.to_position(azelquat)[0]

                # Get polarization weights

                weights = tod.cache.reference("weights_{}".format(det))
                iweights, qweights, uweights = weights.T
        
                # Interpolate HWPSS to incident angle

                theta_deg = np.degrees(theta)
                itheta_high = np.searchsorted(self.thetas, theta_deg)
                itheta_low = itheta_high - 1

                theta_low = self.thetas[itheta_low]
                theta_high = self.thetas[itheta_high]
                r = (theta_deg - theta_low) / (theta_high - theta_low)

                transmission = (
                    (1 - r) * self.all_stokes[freq]["transmission"][itheta_low]
                    + r * self.all_stokes[freq]["transmission"][itheta_high]
                )
                reflection = (
                    (1 - r) * self.all_stokes[freq]["reflection"][itheta_low]
                    + r * self.all_stokes[freq]["reflection"][itheta_high]
                )
                emission = (
                    (1 - r) * self.all_stokes[freq]["emission"][itheta_low]
                    + r * self.all_stokes[freq]["emission"][itheta_high]
                )
                
                # Scale HWPSS for observing elevation

                el_ref = np.radians(50)
                scale = np.sin(el_ref) / np.sin(el)
                
                # Observe HWPSS with the detector

                iquv = (transmission + reflection).T
                iquss = (
                    iweights * np.interp(chi, self.chis, iquv[0]) +
                    qweights * np.interp(chi, self.chis, iquv[1]) +
                    uweights * np.interp(chi, self.chis, iquv[2])
                ) * scale

                iquv = emission.T
                iquss += (
                    iweights * np.interp(chi, self.chis, iquv[0]) +
                    qweights * np.interp(chi, self.chis, iquv[1]) +
                    uweights * np.interp(chi, self.chis, iquv[2])
                )

                iquss -= np.median(iquss)

                # Co-add with the cached signal

                signal += iquss

        return
