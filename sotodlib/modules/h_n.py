# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import numpy as np

import toast

from toast.map import covariance_apply, covariance_invert, DistPixels
from toast.todmap.todmap_math import OpAccumDiag
import toast.qarray as qa


class OpHn(toast.Operator):
    """ Evaluate geometrical h_n factors to support map-based simulations.

    """

    cos_1_prefix = "cos_1"
    sin_1_prefix = "sin_1"
    cos_n_prefix = "cos_n"
    sin_n_prefix = "sin_n"
    hweight_name = "hweight"

    def __init__(
            self,
            weights="weights",
            outdir=".",
            outprefix="",
            nmin=1,
            nmax=4,
            common_flag_mask=1,
            flag_mask=1,
            zip_maps=True,
    ):
        """
        An operator that computes and writes out averaged maps of
        cos(n * psi) and sin(n * psi)

        Arguments:
        weights(str) : Cache prefix to retrieve pointing weights from.
        outdir(str) : Directory to write maps to.
        outprefix(str) : Prefix to append to the file names.
        nmin(int) : Minimum `n` to evaluate.
        nmax(int) : Maximum `n` to evaluate.
        common_flag_mask(uint8) : Which common flag bits to check
        flag_mask(uint8) : Which detector flag bits to check
        """
        self._weights = weights
        self._outdir = outdir
        if outprefix is None:
            self._outprefix = ""
        else:
            self._outprefix = outprefix
        self._nmin = nmin
        self._nmax = nmax
        self._common_flag_mask = common_flag_mask
        self._flag_mask = flag_mask
        self._zip_maps = zip_maps

    def _get_h_n(self, data, n):
        """ Compute and store the next order of h_n
        """
        for obs in data.obs:
            tod = obs["tod"]
            focalplane = obs["focalplane"]
            # HWP angle is not yet used but will be needed soon
            try:
                hwpang = tod.local_hwp_angle()
            except:
                hwpang = None
            if n == 1:
                nsample = tod.local_samples[1]
                hweight = np.ones(nsample, dtype=np.float64)
                tod.cache.put(self.hweight_name, hweight, replace=True)
            for det in tod.local_dets:
                cos_1_name = "{}_{}".format(self.cos_1_prefix, det)
                sin_1_name = "{}_{}".format(self.sin_1_prefix, det)
                cos_n_name = "{}_{}".format(self.cos_n_prefix, det)
                sin_n_name = "{}_{}".format(self.sin_n_prefix, det)
                if n == 1:
                    quats = tod.local_pointing(det)
                    theta, phi, psi = qa.to_angles(quats)
                    cos_n_new = np.cos(psi)
                    sin_n_new = np.sin(psi)
                    tod.cache.put(cos_1_name, cos_n_new, replace=True)
                    tod.cache.put(sin_1_name, sin_n_new, replace=True)
                    weight_name = "{}_{}".format(self.hweight_name, det)
                    tod.cache.add_alias(weight_name, self.hweight_name)
                else:
                    # Use the angle sum identities to evaluate the
                    # next cos(n * psi) and sin(n * psi)
                    cos_1 = tod.cache.reference(cos_1_name)
                    sin_1 = tod.cache.reference(sin_1_name)
                    cos_n_old = tod.cache.reference(cos_n_name).copy()
                    sin_n_old = tod.cache.reference(sin_n_name).copy()
                    cos_n_new = cos_n_old * cos_1 - sin_n_old * sin_1
                    sin_n_new = sin_n_old * cos_1 + cos_n_old * sin_1
                tod.cache.put(cos_n_name, cos_n_new, replace=True)
                tod.cache.put(sin_n_name, sin_n_new, replace=True)
        return

    def _save_h_n(self, data, dist_map, inv_hits, n):
        """ Accumulate and save the next order of h_n
        """
        for name, prefix in ("cos", self.cos_n_prefix), ("sin", self.sin_n_prefix):
            dist_map.data.fill(0.0)
            OpAccumDiag(
                zmap=dist_map,
                name=prefix,
                weights=self.hweight_name,
                common_flag_mask=self._common_flag_mask,
                flag_mask=self._flag_mask,
            ).exec(data)
            dist_map.allreduce()
            covariance_apply(inv_hits, dist_map)
            fname = os.path.join(
                self._outdir, self._outprefix + "{}_{}.fits".format(name, n),
            )
            if self._zip_maps:
                fname += ".gz"
            dist_map.write_healpix_fits(fname)

        return


    def exec(self, data):
        comm = data.comm.comm_world

        if comm.rank == 0:
            os.makedirs(self._outdir, exist_ok=True)
        
        dist_map = DistPixels(data, comm=comm, nnz=1, dtype=np.float64)
        hits = DistPixels(data, comm=comm, nnz=1, dtype=np.int64)
        inv_hits = DistPixels(data, comm=comm, nnz=1, dtype=np.float64)
        hits.data.fill(0)
        OpAccumDiag(
            hits=hits,
            common_flag_mask=self._common_flag_mask,
            flag_mask=self._flag_mask,
        ).exec(data)
        hits.allreduce()
        inv_hits.data[:] = hits.data[:]
        del hits
        #covariance_invert(inv_hits, 0)
        good = inv_hits.data != 0
        inv_hits.data[good] = 1 / inv_hits.data[good]

        for n in range(1, self._nmax + 1):
            self._get_h_n(data, n)
            if n >= self._nmin:
                self._save_h_n(data, dist_map, inv_hits, n)

        for obs in data.obs:
            tod = obs["tod"]
            for prefix in (
                    self.hweight_name, self.cos_1_prefix, self.sin_1_prefix,
                    self.cos_n_prefix, self.sin_n_prefix):
                tod.cache.clear(prefix + "_.*")

        del inv_hits
        del dist_map

        return
