# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import numpy as np

import toast

from toast.map import covariance_apply, covariance_invert, DistPixels
from toast.todmap.todmap_math import OpAccumDiag


class OpCrossLinking(toast.Operator):
    """ Evaluate an ACT-style crosslinking map

    """

    weight_name = "hweight"
    dummy_name = "dummy_signal"
    signal_name = "constant_signal"

    def __init__(
            self,
            weights="weights",
            outdir=".",
            outprefix="",
            common_flag_mask=1,
            flag_mask=1,
            zip_maps=True,
            rcond_limit=1e-3,
    ):
        """
        An operator that computes and writes out ACT-style cross-linking
        factor maps.

        Arguments:
        weights(str) : Cache prefix to retrieve pointing weights from.
        outdir(str) : Directory to write maps to.o
        outprefix(str) : Prefix to append to the file names.
        common_flag_mask(uint8) : Which common flag bits to check
        flag_mask(uint8) : Which detector flag bits to check
        rcond_limit(float) : Reciprocal condition number threshold to
            accept pixels.
        """
        self._weights = weights
        self._outdir = outdir
        if outprefix is None:
            self._outprefix = ""
        else:
            self._outprefix = outprefix
        self._common_flag_mask = common_flag_mask
        self._flag_mask = flag_mask
        self._zip_maps = zip_maps
        self._rcond_limit = rcond_limit

    def _get_weights(self, obs):
        """ Evaluate the special pointing matrix
        """

        tod = obs["tod"]
        nsample = tod.local_samples[1]
        focalplane = obs["focalplane"]
        # Create one constant signal for the observation and make it an
        # alias for all detectors
        tod.cache.put(
            self.dummy_name, np.ones(nsample, dtype=np.float64), replace=True,
        )
        
        for det in tod.local_dets:
            if "pol_angle_deg" in focalplane[det]:
                psi = np.radians(focalplane[det]["pol_angle_deg"])
            elif "pol_angle_rad" in focalplane[det]:
                psi = focalplane[det]["pol_angle_rad"]
            else:
                raise RuntimeError(
                    "Could not find polarization angle for {} in focalplane."
                    "".format(det)
                )
            weights_name_in = "{}_{}".format(self._weights, det)
            weights_in = tod.cache.reference(weights_name_in)

            iw, qw, uw = weights_in.T
            # The polarization weights may include a polarization
            # efficiency term we don't want
            eta = np.sqrt(qw ** 2 + uw ** 2)
            cos2psi_in = qw / eta
            sin2psi_in = uw / eta
            cos2psi_det = np.cos(2 * psi)
            sin2psi_det = np.sin(2 * psi)
            # Subtract the detector polarization angle from the polarization weights
            # Rather than evaluate tons of sines and cosines
            cos2psi_out = cos2psi_in * cos2psi_det + sin2psi_in * sin2psi_det
            sin2psi_out = sin2psi_in * cos2psi_det - cos2psi_in * sin2psi_det

            weights_out = np.vstack([np.ones(nsample), cos2psi_out, sin2psi_out]).T
            weights_name_out = "{}_{}".format(self.weight_name, det)
            tod.cache.put(weights_name_out, weights_out, replace=True)

            # Need a constant signal to map
            signal_name = "{}_{}".format(self.signal_name, det)
            tod.cache.add_alias(signal_name, self.dummy_name)
        return

    def _purge_weights(self, obs):
        """ Discard special pointing matrix and dummy signal
        """
        tod = obs["tod"]
        tod.cache.clear(self.weight_name + "_.*")
        tod.cache.clear(self.signal_name + "_.*")
        tod.cache.clear(self.dummy_name)
        return

    def exec(self, data):
        comm = data.comm.comm_world

        if comm.rank == 0:
            os.makedirs(self._outdir, exist_ok=True)

        for obs in data.obs:
            self._get_weights(obs)
        
        dist_map = DistPixels(data, comm=comm, nnz=3, dtype=np.float64)
        dist_map.data.fill(0)
        # FIXME:  no detector weights applied yet.
        OpAccumDiag(
            zmap=dist_map,
            common_flag_mask=self._common_flag_mask,
            flag_mask=self._flag_mask,
            weights=self.weight_name,
            name=self.signal_name,
        ).exec(data)
        dist_map.allreduce()

        fname = os.path.join(self._outdir, self._outprefix + "crosslinking.fits")
        if self._zip_maps:
            fname += ".gz"
        dist_map.write_healpix_fits(fname)

        del dist_map
        for obs in data.obs:
            self._purge_weights(obs)

        return
