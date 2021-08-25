# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import numpy as np

import toast

from toast.map import covariance_apply, covariance_invert, DistPixels
from toast.todmap.todmap_math import OpAccumDiag
from toast import qarray as qa


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
            detweights=None,
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
        self._detweights = detweights

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
            # measure the scan direction wrt the local meridian
            # for each sample
            quat = tod.read_pntg(detector=det)
            theta, phi = qa.to_position(quat)
            theta = np.pi / 2 - theta
            # scan direction across the reference sample
            dphi = (np.roll(phi, -1) - np.roll(phi, 1))
            dtheta = np.roll(theta, -1) - np.roll(theta, 1)
            # except first and last sample
            for dx, x in (dphi, phi), (dtheta, theta):
                dx[0] = x[1] - x[0]
                dx[-1] = x[-1] - x[-2]
            # scale dphi to on-sky
            dphi *= np.cos(theta)
            # Avoid overflows
            tiny = np.abs(dphi) < 1e-30
            if np.any(tiny):
                ang = np.zeros(nsample)
                ang[tiny] = np.sign(dtheta) * np.sign(dphi) * np.pi / 2
                not_tiny = np.logical_not(tiny)
                ang[not_tiny] = np.arctan(dtheta[not_tiny] / dphi[not_tiny])
            else:
                ang = np.arctan(dtheta / dphi)

            weights_out = np.vstack(
                [np.ones(nsample), np.cos(2 * ang), np.sin(2 * ang)]
            ).T
            weights_name_out = f"{self.weight_name}_{det}"
            tod.cache.put(weights_name_out, weights_out, replace=True)

            # Need a constant signal to map
            signal_name = f"{self.signal_name}_{det}"
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
        # FIXME:  no detector weights are fixed, not per observation.
        OpAccumDiag(
            zmap=dist_map,
            common_flag_mask=self._common_flag_mask,
            flag_mask=self._flag_mask,
            weights=self.weight_name,
            name=self.signal_name,
            detweights=self._detweights,
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
