from collections import OrderedDict
import glob
import os
import pickle
import random
import re
import subprocess
import sys
from time import time

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import numpy as np
from scipy.optimize import leastsq
import scipy.signal

import toast
from toast.pixels_io_healpix import read_healpix
from toast.scripts import toast_healpix_coadd


# Map ordering:
# - maps in MapCache are kept in NESTED ordering for efficient co-add
# - maps outside of MapCache are always in RING ordering for anafast


def fit_noise_model(ell, cl):
    """Fit a 1 / ell noise model to the power spectrum"""
    nbin = 50
    # binmin * r ** (nbin - 1) = binmax
    # log(r) = log(binmax / binmin) / (nbin - 1)
    # binmin = 50
    binmin = 20
    binmax = 600
    r = np.exp(np.log(binmax / binmin) / (nbin - 1))
    xbin = []
    ybin = []
    lower = binmin
    for ibin in range(nbin):
        upper = lower * r
        ii = np.logical_and(lower <= ell, ell < upper)
        xbin.append(np.mean(ell[ii]))
        ybin.append(np.mean(cl[ii]))
        lower = upper
    xbin = np.array(xbin)
    ybin = np.array(ybin) * 1e12

    def model(x):
        amp, ellknee, alpha = x
        return amp * (1 + (xbin / ellknee)**alpha)

    x0 = [np.median(ybin), 100, -1]
    if True:
        def resid(x):
            return np.log(ybin) - np.log(model(x))
        result, cov, info, msg, ierr = leastsq(resid, x0, full_output=True)
        if ierr not in range(1, 6):
            print(prefix + f"Noise model failed: {msg}")
            return None
        amp, ellknee, alpha = result
    else:
        def resid(x):
            return np.sum((ybin - model(x))**2)
        result = minimize(resid, x0, options={"disp" : True})
        if result.success == False:
            print(prefix + f"Noise model failed: {result.message}")
            return None
        amp, ellknee, alpha = result.x
    
    return (amp, ellknee, alpha)


def get_hardcopies(individual, mapcache=None, nflip=None, plot=True, verbosity=2):
    """ Make sure co-added real data, signflips and simulated map
    matching the individual filter configuration.
    """
    individual.get_map(None, mapcache, signflip=False)

    if nflip is not None:
        nflip_orig = individual.nflip
        individual.nflip = nflip
        individual._draw_weights()
    individual.get_map(None, mapcache, signflip=True)
    fname_signflips = individual.outmap_signflips

    individual.get_sim_map(None, mapcache)

    cltot = get_cl(individual.outmap, save=True, verbosity=verbosity)
    clnoises = []
    for fname_signflip in individual.outmap_signflips:
        clnoise = get_cl(fname_signflip, fname_invcov=individual.invcov, save=True, verbosity=verbosity)
        clnoises.append(clnoise)
    if nflip is not None:
        clnoise1 = np.mean(clnoises[:nflip_orig], 0)
        clnoise2 = np.mean(clnoises[nflip_orig:], 0)
    else:
        clnoise1 = np.mean(clnoises, 0)
        clnoise2 = None
    clsim, clinput, tf = get_tf(individual.outmap_sim, individual.fname_input, save=True)

    if plot:
        nrow, ncol = 1, 3
        fig = plt.figure(figsize=[6 * ncol, 4 * nrow])
        ell = np.arange(cltot[0].size)
        for col in range(3):
            ax = fig.add_subplot(nrow, ncol, col + 1)
            ax.loglog(ell[2:], cltot[col][2:] / tf[col][2:], label="Total")
            if clnoise2 is None:
                ax.loglog(ell[2:], clnoise1[col][2:] / tf[col][2:], label=f"Signflip")
            else:
                ax.loglog(ell[2:], clnoise1[col][2:] / tf[col][2:], label=f"Signflip # 1")
                ax.loglog(ell[2:], clnoise2[col][2:] / tf[col][2:], label=f"Signflip # 2")
        ax.legend(loc="best")

    return cltot, clnoise1, clnoise2, clsim, clinput, tf


def get_cl(fname_map, fname_invcov=None, prefix="", cache=None, save=False, verbosity=1):
    """Return the noise-weighted pseudo spectrum of the provided map"""

    if fname_map.endswith("_map.h5"):
        fname_cl = fname_map.replace("_map.h5", "_cl.fits")
    elif fname_map.endswith("_map.fits"):
        fname_cl = fname_map.replace("_map.fits", "_cl.fits")
    else:
        msg = prefix + f"Don't know how to synthesize a C_ell filename for {fname_map}"
        raise RuntimeError(msg)

    if cache is not None and fname_cl in cache:
        cl = cache[fname_cl]
    elif os.path.isfile(fname_cl):
        cl = hp.read_cl(fname_cl)
        invcov = None
    else:
        if cache is not None and fname_map in cache:
            # Expand and reorder the map in cache
            compressed, good, npix = cache[fname_map]
            nmap, ngood = compressed.shape
            m = np.zeros([nmap, npix])
            m[:, good] = compressed
            m = hp.reorder(m, n2r=True)
        else:
            m = read_healpix(fname_map, [0, 1, 2], nest=False)
        if fname_invcov is None:
            fname_invcov = fname_map.replace("map", "invcov")
            pattern = re.compile(".*(_signflip[0-9]{4}).*")
            match = pattern.match(fname_invcov)
            if match is not None:
                fname_invcov = fname_invcov.replace(match.groups()[0], "")
        if fname_map == fname_invcov:
            msg = prefix + f"Don't know how to synthesize invcov filename for {fname_map}"
            raise RuntimeError(msg)
        if cache is not None and fname_invcov in cache:
            compressed, good, npix = cache[fname_invcov]
            nmap, ngood = compressed.shape
            invcov = np.zeros([nmap, npix])
            invcov[:, good] = compressed
            invcov = hp.reorder(invcov[0], n2r=True)  # Use II for weighting
        else:
            invcov = read_healpix(fname_invcov, nest=False)  # Use II for weighting
        fsky = np.mean(invcov**2)
        if verbosity > 2:
            print(prefix + f"Running anafast for {fname_map}")
        lmax = 2 * hp.get_nside(m)
        cl = hp.anafast(m * invcov, lmax=lmax, iter=0) / fsky
        if cache is not None:
            cache[fname_cl] = cl
        if save:
            hp.write_cl(fname_cl, cl, overwrite=True)
            if verbosity > 1:
                print(prefix + f"Wrote {fname_cl}")

    return cl


def get_tf(fname_map, fname_input, smooth=True, prefix="", cache=None, save=False, verbosity=1):
    """Return the noise-weighted transfer function of the provided map"""

    if cache is not None and fname_input in cache:
        input_map = cache[fname_input]
    else:
        input_map = read_healpix(fname_input, [0, 1, 2], nest=False)

    if fname_map.endswith("_map.h5"):
        fname_cl = fname_map.replace("_map.h5", "_cl.fits")
    elif fname_map.endswith("_map.fits"):
        fname_cl = fname_map.replace("_map.fits", "_cl.fits")
    else:
        msg = prefix + f"Don't know how to synthesize a C_ell filename for {fname_map}"
        raise RuntimeError(msg)

    fname_tf = fname_cl.replace("_cl.fits", "_tf.fits")

    # Simulated C_ell

    if cache is not None and fname_cl in cache:
        cl = cache[fname_cl]
    elif os.path.isfile(fname_cl):
        cl = hp.read_cl(fname_cl)
        invcov = None
    else:
        if cache is not None and fname_map in cache:
            compressed, good, npix = cache[fname_map]
            nmap, ngood = compressed.shape
            m = np.zeros([nmap, npix])
            m[:, good] = compressed
            m = hp.reorder(m, n2r=True)
        else:
            m = read_healpix(fname_map, [0, 1, 2], nest=False)
        fname_invcov = fname_map.replace("map", "invcov")
        if fname_map == fname_invcov:
            msg = prefix + f"Don't know how to synthesize invcov filename for {fname_map}"
            raise RuntimeError(msg)
        if cache is not None and fname_invcov in cache:
            compressed, good, npix = cache[fname_invcov]
            nmap, ngood = compressed.shape
            invcov = np.zeros([nmap, npix])
            invcov[:, good] = compressed
            invcov = hp.reorder(invcov[0], n2r=True)  # Use II for weighting
        else:
            invcov = read_healpix(fname_invcov, nest=False)  # Use II for weighting
        fsky = np.mean(invcov**2)
        if verbosity > 2:
            print(prefix + f"Running anafast for {fname_cl}")
        lmax = 2 * hp.get_nside(m)
        cl = hp.anafast(m * invcov, lmax=lmax, iter=0) / fsky
        if cache is not None:
            cache[fname_cl] = cl
        if save:
            hp.write_cl(fname_cl, cl, overwrite=True)
            if verbosity > 1:
                print(prefix + f"Wrote {fname_cl}")

    # Input C_ell

    fname_cl_input = fname_cl.replace("_cl.fits", "_cl.input.fits")
    if cache is not None and fname_cl_input in cache:
        cl_input = cache[fname_cl_input]
    elif os.path.isfile(fname_cl_input):
        cl_input = hp.read_cl(fname_cl_input)
    else:
        if invcov is None:
            fname_invcov = fname_map.replace("map", "invcov")
            if cache is not None and fname_invcov in cache:
                compressed, good, npix = cache[fname_invcov]
                nmap, ngood = compressed.shape
                invcov = np.zeros([nmap, npix])
                invcov[:, good] = compressed
                invcov = hp.reorder(invcov[0], n2r=True)  # Use II for weighting
            else:
                invcov = read_healpix(fname_invcov, nest=False)  # Use II for weighting
            fsky = np.mean(invcov**2)
        if verbosity > 2:
            print(prefix + f"Running anafast for {fname_cl_input}")
        lmax = 2 * hp.get_nside(input_map)
        cl_input = hp.anafast(input_map * invcov, lmax=lmax, iter=0) / fsky
        if cache is not None:
            cache[fname_cl_input] = cl_input
        if save:
            hp.write_cl(fname_cl_input, cl_input, overwrite=True)

    # Transfer function

    if cache is not None and fname_tf in cache:
        tf = cache[fname_tf]
    elif os.path.isfile(fname_tf):
        tf = hp.read_cl(fname_tf)
    else:

        tf = np.zeros_like(cl)
        good = cl_input != 0
        tf[good] = cl[good] / cl_input[good]

        if smooth:
            lmax = tf.shape[1] - 1
            ell = np.arange(lmax + 1)
            wkernel = 1
            tf_copy = tf.copy()
            while True:
                wkernel *= 2
                kernel = np.ones(wkernel) / wkernel
                istart = wkernel
                r = (ell - wkernel // 2) / wkernel
                r[r < 0] = 0
                r[r > 1] = 1
                for i in range(len(tf)):
                    rough = tf[i].copy()
                    smooth = scipy.signal.convolve(tf_copy[i], kernel, mode="same")
                    tf[i] = (1 - r) * rough + r * smooth
                if 4 * wkernel > lmax:
                    for i in range(len(tf)):
                        ind = np.argmax(tf[i])
                        tf[i][ind:] = tf[i][ind]
                    tf[tf > 1] = 1
                    break

        if cache is not None:
            cache[fname_tf] = tf
        if save:
            hp.write_cl(fname_tf, tf, overwrite=True)

    return cl, cl_input, tf


class WaferObservation:
    def __init__(self, obs, ws, band, paths):
        self.obs = obs
        self.ws = ws
        self.band = band
        self.paths = paths
        self.nmap = len(paths)


class Individual:
    def __init__(
            self,
            observations,
            theory,
            outdir,
            fname_input,
            outroot,
            outroot_tf,
            band,
            band_tf,
            genotype=None,
            rmutate=0.01,
            prefix="",
            start_order=None,
            nflip=10,
            mode="IQU",
            wbin=30,
            score_function="optimistic",
            verbosity=1,
    ):
        """ Initialize an individual

        Args:
            observations : all wafer observations
            theory : Theory spectrum (for fitness)
            outdir : working directory for co-adds
            fname_input : simulated input map path
            outroot : root directory for the maps to coadd
            outroot_tf : root directory for the simulated maps to coadd
            band : Frequency of all maps
            band_tf : Frequency of all simulated maps
            genotype : indices of maps for all wafer observations
            rmutate : mutation rate per individual
            start_order : if not None, use this as a starting order for
                all genes
            nflip (int) : Number of sign flips to prepare for
            mode : 'IQU' or 'I'
            wbin (int) : Width of ell-bins in the score function
            score_function (str) : Name of the score function
        """
        self.observations = observations
        self.theory = theory
        self.outdir = outdir
        self.fname_input = fname_input
        self.outroot = outroot
        self.outroot_tf = outroot_tf
        self.band = band
        self.band_tf = band_tf
        self.nobs = len(self.observations)
        self.rmutate = rmutate
        if genotype is None:
            self.draw(start_order)
        else:
            self.genotype = genotype
        self._fitness = None
        self.prefix = prefix
        self.nflip = nflip
        self.mode = mode
        self.wbin = wbin
        self.score_function = score_function
        self.verbosity = verbosity
        self._draw_weights()

    def clone(self):
        clone = Individual(
            self.observations,
            self.theory,
            self.outdir,
            self.fname_input,
            self.outroot,
            self.outroot_tf,
            self.band,
            self.band_tf,
            genotype=self.genotype,
            rmutate=self.rmutate,
            prefix=self.prefix,
            nflip=self.nflip,
            mode=self.mode,
            wbin=self.wbin,
            score_function=self.score_function
        )
        clone.signflip_weights = self.signflip_weights.copy()
        return clone

    @property
    def all_paths(self):
        all_paths = []
        for obs in self.observations:
            for fname_map in obs.paths:
                all_paths.append(fname_map)
                fname_invcov = fname_map.replace("_map.h5", "_invcov.h5")
                all_paths.append(fname_map)
                all_paths.append(fname_invcov)
                all_paths.append(
                    fname_map.replace(
                        self.outroot, self.outroot_tf
                    ).replace(self.band, self.band_tf)
                )
                all_paths.append(
                    fname_invcov.replace(
                        self.outroot, self.outroot_tf
                    ).replace(self.band, self.band_tf)
                )
        return all_paths

    def _draw_weights(self):
        """ Draw signflip weights for each wafer observation """
        if not hasattr(self, "signflip_weights"):
            self.signflip_weights = []
        n_exist = len(self.signflip_weights)
        for iflip in range(n_exist, self.nflip):
            np.random.seed(48356246 + iflip)
            self.signflip_weights.append(
                (np.random.rand(self.nobs) * 2).astype(int) * 2 - 1
            )
        return

    def weights_signflip(self, realization):
        """ Construct a dictionary of signflip weights for the current
        filter configuration """
        weights = OrderedDict()
        for i, obs in enumerate(self.observations):
            imap = self.genotype[i]
            if imap >= 0:
                path = obs.paths[imap]
                weights[path] = self.signflip_weights[realization][i]
        return weights

    @property
    def weights_sim(self):
        """ Construct a trivial dictionary of simulated map weights """
        weights = OrderedDict()
        for path in self.paths:
            path_sim = path.replace(
                self.outroot, self.outroot_tf
            ).replace(self.band, self.band_tf)
            weights[path_sim] = (1.0, 1.0)
        return weights

    @property
    def name(self):
        genotype_string = "".join((self.genotype + 1).astype(str))
        return toast.utils.name_UID(genotype_string)

    def draw(self, order):
        """ Set imap to one of the alternative paths or -1 """
        self.genotype = np.zeros(self.nobs, dtype=int)
        if order is None:
            for i, obs in enumerate(self.observations):
                self.genotype[i] = int(np.random.rand() * (obs.nmap + 1)) - 1
        else:
            for i, obs in enumerate(self.observations):
                # Some filter configurations may be missing for some
                # observations
                if order >= obs.nmap:
                    print(f"WARNING: Observation # {i} : obs={obs.obs} : ws={obs.ws} : band={obs.band} only has {obs.nmap} configurations but order = {order}")  # DEBUG
                self.genotype[i] = min(order, obs.nmap - 1)
        return

    @property
    def paths(self):
        paths = []
        for i, obs in enumerate(self.observations):
            imap = self.genotype[i]
            if imap >= 0:
                paths.append(obs.paths[imap])
        return paths

    def reproduce(self, other):
        genotype = np.zeros_like(self.genotype)
        limit = self.rmutate / len(genotype)
        for i, obs in enumerate(self.observations):
            if np.random.randn() < limit:
                # No inheritance, draw a new value
                genotype[i] = int(np.random.rand() * (obs.nmap + 1)) - 1
            else:
                if np.random.rand() > 0.5:
                    genotype[i] = self.genotype[i]
                else:
                    genotype[i] = other.genotype[i]
        offspring = Individual(
            self.observations,
            self.theory,
            self.outdir,
            self.fname_input,
            self.outroot,
            self.outroot_tf,
            self.band,
            self.band_tf,
            genotype,
            mode=self.mode,
        )
        return offspring

    def optimize(self, gene, return_result=None, mapcache=None, input_map=None):
        """ Using the current genotype, brute force the optimal
        configuration of a specific gene
        """

        # Make sure the coadded maps are available
        if return_result is None:
            result = OrderedDict()
        else:
            result = return_result
        self.get_map(result, mapcache, persistent=True, signflip=True)
        self.get_sim_map(result, mapcache, persistent=True)

        # Create all possible candidates for this gene
        obs = self.observations[gene]
        candidates = []
        for i in range(obs.nmap + 1):
            genotype = self.genotype.copy()
            genotype[gene] = i - 1
            if self.genotype[gene] == genotype[gene]:
                candidates.append(self)
                continue
            candidate = self.clone()
            candidate.genotype = genotype
            # print(f" **** Candidate # {i}, name = {candidate.name}")  # DEBUG
            candidate.prefix = self.prefix
            candidates.append(candidate)
            # Coadd the real map for evaluating fitness
            if self.genotype[gene] == -1:
                path_subtract = None
            else:
                path_subtract = obs.paths[self.genotype[gene]]
            if genotype[gene] == -1:
                path_add = None
            else:
                path_add = obs.paths[genotype[gene]]
            for iflip in range(self.nflip):
                outmap = os.path.join(
                    f"{candidate.outdir}",
                    f"{candidate.name}_signflip{iflip:04}_filtered_map.fits",
                )
                invcov = f"{candidate.outdir}/{candidate.name}_filtered_invcov.fits"
                weights = {self.outmap_signflips[iflip] : (1.0, 1.0)}
                if path_subtract is not None:
                    weight = self.weights_signflip(iflip)[path_subtract]
                    weights[path_subtract] = (-weight, -1.0)
                if path_add is not None:
                    weight = candidate.weights_signflip(iflip)[path_add]
                    weights[path_add] = (weight, 1.0)
                t1 = time()
                # DEBUG begin
                # msg = f"DEBUG : Calling coadd with "
                # fname = self.outmap_signflips[iflip]
                # msg += f"input = {fname} : {np.std(mapcache[fname][0])} "
                # if path_subtract is not None:
                #     msg += f"subtract = {path_subtract} : {np.std(mapcache[path_subtract][0])} "
                # if path_add is not None:
                #     msg += f"subtract = {path_add} : {np.std(mapcache[path_add][0])} "
                # print(msg)
                # DEBUG end
                toast_healpix_coadd.main(
                    opts=[
                        "--outmap",
                        outmap,
                        "--invcov",
                        invcov,
                        "DUMMY_INPUT",
                    ],
                    comm=None,
                    cache=mapcache,
                    result=result,
                    prefix=self.prefix,
                    weights=weights,
                )
                # DEBUG begin
                # msg = f"DEBUG : Coadd done with "
                # msg += f"output = {outmap} : {np.std(result[outmap][0])}"
                # print(msg)
                # DEBUG end
                if outmap not in result:
                    msg = self.prefix + f"ERROR : coadded map, not found {outmap}"
                    raise RuntimeError(msg)
                if self.verbosity > 1:
                    print(self.prefix + f"Coadded {outmap} in {time()-t1:.1f}s.")
                m = result[outmap]
                ic = result[invcov]
                good = np.argwhere(m[0] != 0).ravel()
                nmap, npix = m.shape
                result[outmap] = (m[:, good], good, npix)
                result[invcov] = (ic[:, good], good, npix)
            # Coadd the simulated CMB map for evaluating fitness
            if path_subtract is None:
                path_subtract_sim = None
            else:
                path_subtract_sim = path_subtract.replace(
                    self.outroot, self.outroot_tf
                ).replace(self.band, self.band_tf)
            if path_add is None:
                path_add_sim = None
            else:
                path_add_sim = path_add.replace(
                    self.outroot, self.outroot_tf
                ).replace(self.band, self.band_tf)
            outmap_sim = f"{candidate.outdir}/{candidate.name}_sim_filtered_map.fits"
            invcov_sim = f"{candidate.outdir}/{candidate.name}_sim_filtered_invcov.fits"
            weights_sim = {self.outmap_sim : (1.0, 1.0)}
            if path_subtract_sim is not None:
                weights_sim[path_subtract_sim] = (-1.0, -1.0)
            if path_add_sim is not None:
                candidate.weights_sim[path_add_sim] = (1.0, 1.0)
                weights_sim[path_add_sim] = (1.0, 1.0)
            t1 = time()
            toast_healpix_coadd.main(
                opts=[
                    "--outmap",
                    outmap_sim,
                    "--invcov",
                    invcov_sim,
                    "DUMMY_INPUT",
                ],
                comm=None,
                cache=mapcache,
                result=result,
                prefix=self.prefix,
                weights=weights_sim,
            )
            if outmap_sim not in result:
                msg = self.prefix + f"ERROR : coadded map, not found {outmap_sim}"
                raise RuntimeError(msg)
            if self.verbosity > 1:
                print(self.prefix + f"Coadded {outmap_sim} in {time()-t1:.1f}s.")
            m = result[outmap_sim]
            ic = result[invcov_sim]
            good = np.argwhere(m[0] != 0).ravel()
            nmap, npix = m.shape
            result[outmap_sim] = (m[:, good], good, npix)
            result[invcov_sim] = (ic[:, good], good, npix)

        fitnesses = [
            candidate.fitness(
                return_result=result, mapcache=mapcache, input_map=input_map
            )
            for candidate in candidates
        ]
        ibest = np.argmax(fitnesses)
        best = candidates[ibest]
        """
        # DEBUG begin
        # if np.amax(fitnesses) > self.fitness() * 2.0:
        if best.fitness() < self.fitness() * .99:
            if mapcache is None:
                rank = 9999
            else:
                rank = mapcache.comm.rank
            print(self.prefix + f"ERROR : Fitness is getting worse. {best.fitness()} < {self.fitness()}")
            fname = f"debug_{rank}.pck"
            with open(fname, "wb") as f:
                pickle.dump([self, best, candidates], f)
            print(self.prefix + f"Wrote {fname}")
            import matplotlib.pyplot as plt
            nrow, ncol = 1, 3
            fig = plt.figure(figsize=[ncol * 6, nrow * 4])
            ax1 = fig.add_subplot(nrow, ncol, 1)
            ax2 = fig.add_subplot(nrow, ncol, 2)
            ax3 = fig.add_subplot(nrow, ncol, 3)
            for c in candidates:
                lmax = c.cl[0].size - 1
                ell = np.arange(lmax + 1)
                ax1.loglog(ell[2:], c.cl[2][2:], label=f"{c.name} {c.fitness()}")
                ax2.loglog(ell[2:], c.tf[2][2:], label=f"{c.name} {c.fitness()}")
                ax3.loglog(
                    ell[2:], c.cl[2][2:] * c.tf[2][2:], label=f"{c.name} {c.fitness()}"
                )
            ax2.legend(loc="best")
            fname = f"candidate_cls.{rank}.png"
            fig.savefig(fname)
            print(self.prefix + f"Wrote {fname}")
            plt.close()
            fig = plt.figure(figsize=[12, 6])
            for i, c in enumerate(candidates):
                m1 = result[c.outmap].copy()
                m2 = result[c.outmap_sim].copy()
                m1[m1 == 0] = hp.UNSEEN
                m2[m2 == 0] = hp.UNSEEN
                hp.mollview(m1[0], title=c.name, sub=[2, 6, 1 + i])
                hp.mollview(m2[0], title=c.name, sub=[2, 6, 7 + i])
            fname = f"candidate_maps.{rank}.png"
            fig.savefig(fname)
            print(self.prefix + f"Wrote {fname}")
            plt.close()
        """
        if self.verbosity > 2:
            print(f" ***** Candidate fitnesses: {fitnesses}")
            print(f" ***** Current / best fitness = {self.fitness()} / {best.fitness()}")
            print(f" **** Current gene # {gene} = {self.genotype[gene]}")
        return best

    def differentiate(self, gene, return_result=None, mapcache=None, input_map=None):
        """ Using the current genotype, derive two candidates that are
        one step away in a specific gene
        """

        # Make sure the coadded maps are available.  This way the step
        # maps are cheap to evaluate
        if return_result is None:
            result = OrderedDict()
        else:
            result = return_result
        self.get_map(result, mapcache, persistent=True, signflip=True)
        self.get_sim_map(result, mapcache, persistent=True)
        fitness = self.fitness(
            return_result=result, mapcache=mapcache, input_map=input_map
        )

        # Create all possible candidates for this gene
        obs = self.observations[gene]
        current = self.genotype[gene]
        candidate_genes = []
        # Direction of more relaxed filtering
        if current == 0:
            # Already the most relaxed configuration
            candidate_genes.append(None)
        elif current == -1:
            # Turn reject into most aggressive filter
            candidate_genes.append(obs.nmap - 1)
        else:
            # Drop filter order by one
            candidate_genes.append(current - 1)
        # Direction of more aggressive filtering
        if current == -1:
            # Already being rejected
            candidate_genes.append(None)
        elif current == obs.nmap - 1:
            # Turn most aggressive filter into reject
            candidate_genes.append(-1)
        else:
            # Increase filter order by one
            candidate_genes.append(current + 1)

        candidates = []
        fitnesses = []
        for candidate_gene in candidate_genes:
            if candidate_gene is None:
                candidates.append(None)
                fitnesses.append(0)
                continue
            genotype = self.genotype.copy()
            genotype[gene] = candidate_gene
            candidate = self.clone()
            candidate.genotype = genotype
            candidate.prefix = self.prefix
            # Coadd the real map for evaluating fitness
            if self.genotype[gene] == -1:
                path_subtract = None
            else:
                path_subtract = obs.paths[self.genotype[gene]]
            if genotype[gene] == -1:
                path_add = None
            else:
                path_add = obs.paths[genotype[gene]]
            for iflip in range(self.nflip):
                outmap = f"{candidate.outdir}/{candidate.name}_signflip{iflip:04}_filtered_map.fits"
                invcov = f"{candidate.outdir}/{candidate.name}_filtered_invcov.fits"
                weights = {self.outmap_signflips[iflip] : (1.0, 1.0)}
                if path_subtract is not None:
                    weight = self.weights_signflip(iflip)[path_subtract]
                    weights[path_subtract] = (-weight, -1.0)
                if path_add is not None:
                    weight = candidate.weights_signflip(iflip)[path_add]
                    weights[path_add] = (weight, 1.0)
                t1 = time()
                toast_healpix_coadd.main(
                    opts=[
                        "--outmap",
                        outmap,
                        "--invcov",
                        invcov,
                        "DUMMY_INPUT",
                    ],
                    comm=None,
                    cache=mapcache,
                    result=result,
                    prefix=self.prefix,
                    weights=weights,
                )
                if outmap not in result:
                    msg = self.prefix + f"ERROR : coadded map, not found {outmap}"
                    raise RuntimeError(msg)
                if self.verbosity > 1:
                    print(self.prefix + f"Coadded {outmap} in {time()-t1:.1f}s.")
                m = result[outmap]
                ic = result[invcov]
                good = np.argwhere(m[0] != 0).ravel()
                nmap, npix = m.shape
                result[outmap] = (m[:, good], good, npix)
                result[invcov] = (ic[:, good], good, npix)
            # Coadd the simulated CMB map for evaluating fitness
            if path_subtract is None:
                path_subtract_sim = None
            else:
                path_subtract_sim = path_subtract.replace(
                    self.outroot, self.outroot_tf
                ).replace(self.band, self.band_tf)
            if path_add is None:
                path_add_sim = None
            else:
                path_add_sim = path_add.replace(
                    self.outroot, self.outroot_tf
                ).replace(self.band, self.band_tf)
            outmap_sim = f"{candidate.outdir}/{candidate.name}_sim_filtered_map.fits"
            invcov_sim = f"{candidate.outdir}/{candidate.name}_sim_filtered_invcov.fits"
            weights_sim = {self.outmap_sim : (1.0, 1.0)}
            if path_subtract_sim is not None:
                weights_sim[path_subtract_sim] = (-1.0, -1.0)
            if path_add_sim is not None:
                candidate.weights_sim[path_add_sim] = (1.0, 1.0)
                weights_sim[path_add_sim] = (1.0, 1.0)
            t1 = time()
            toast_healpix_coadd.main(
                opts=[
                    "--outmap",
                    outmap_sim,
                    "--invcov",
                    invcov_sim,
                    "DUMMY_INPUT",
                ],
                comm=None,
                cache=mapcache,
                result=result,
                prefix=self.prefix,
                weights=weights_sim,
            )
            if outmap_sim not in result:
                msg = self.prefix + f"ERROR : coadded map, not found {outmap_sim}"
                raise RuntimeError(msg)
            if self.verbosity > 1:
                print(self.prefix + f"Coadded {outmap_sim} in {time()-t1:.1f}s.")
            m = result[outmap_sim]
            invcov = result[invcov_sim]
            good = np.argwhere(m[0] != 0).ravel()
            nmap, npix = m.shape
            result[outmap_sim] = (m[:, good], good, npix)
            result[invcov_sim] = (invcov[:, good], good, npix)

            candidates.append(candidate)
            fitnesses.append(candidate.fitness(
                return_result=result, mapcache=mapcache, input_map=input_map
            ))

        return candidates

    def get_map(self, result, mapcache=None, persistent=False, signflip=True):
        """ Load or co-add the real data OR sign flips and inverse covariance

        Args:
            result (dict) : dictionary for the co-added maps. Key is the
                intended path
            mapcache : Preloaded maps for faster co-add
            persistent (bool) : If True, the co-added map is added as an
                entry to mapcache
        """

        if signflip:
            self.outmap_signflips = []
            for iflip in range(self.nflip):
                self.outmap_signflips.append(
                    os.path.join(
                        self.outdir,
                        f"{self.name}_signflip{iflip:04}_filtered_map.fits",
                    )
                )
            outmaps = self.outmap_signflips
        else:
            self.outmap = f"{self.outdir}/{self.name}_filtered_map.fits"
            outmaps = [self.outmap]
        self.invcov = f"{self.outdir}/{self.name}_filtered_invcov.fits"

        for iflip, outmap in enumerate(outmaps):
            if result is not None and outmap in result and self.invcov in result:
                if self.verbosity > 2:
                    print(self.prefix + f"{outmap} already in 'result'")
                continue
            else:
                if self.verbosity > 2:
                    print(self.prefix + f"{outmap} NOT found in 'result'")

            if mapcache is not None and outmap in mapcache:
                if self.verbosity > 2:
                    print(self.prefix + f"Retrieving {outmap} from cache")
                result[outmap] = mapcache[outmap]
                if self.verbosity > 2:
                    print(self.prefix + f"Retrieving {self.invcov} from cache")
                result[self.invcov] = mapcache[self.invcov]
            elif os.path.isfile(outmap):
                if result is not None:
                    if self.verbosity > 2:
                        print(self.prefix + f"Loading {outmap} from disk")
                    m = hp.read_map(outmap, None, nest=True)
                    if self.verbosity > 2:
                        print(self.prefix + f"Loading {self.invcov} from disk")
                    invcov = read_healpix(self.invcov, None, nest=True)
                    good = np.argwhere(m[0] != 0).ravel()
                    nmap, npix = m.shape
                    result[outmap] = (m[:, good], good, npix)
                    result[self.invcov] = (invcov[:, good], good, npix)
                    if mapcache is not None and persistent:
                        # Write the maps to mapcache in compressed NESTED ordering
                        mapcache[outmap] = result[outmap]
                        mapcache[self.invcov] = result[self.invcov]
            else:
                if self.verbosity > 1:
                    print(self.prefix + f"Coadding {outmap}")
                if signflip:
                    weights = OrderedDict()
                    for key, value in self.weights_signflip(iflip).items():
                        weights[key] = (value, 1.0)
                else:
                    weights = OrderedDict()
                    for path in self.paths:
                        weights[path] = (1.0, 1.0)
                t1 = time()
                toast_healpix_coadd.main(
                    opts=[
                        "--outmap",
                        outmap,
                        "--invcov",
                        self.invcov,
                        "DUMMY_INPUT",
                    ],
                    comm=None,
                    # cache=None,
                    cache=mapcache,
                    result=result,
                    prefix=self.prefix,
                    weights=weights,
                )
                # DEBUG begin
                # msg = f"DEBUG2 : Coadd done with "
                # msg += f"output = {outmap} : {np.std(result[outmap][0])}"
                # print(msg)
                # if signflip:
                #     from mpi4py import MPI
                #     MPI.COMM_WORLD.Barrier()
                #     MPI.COMM_WORLD.Abort()
                # DEBUG end
                if (
                        result is None and not os.path.isfile(outmap)
                ) or (
                    result is not None and outmap not in result
                ):
                    msg = self.prefix + f"ERROR : coadded map, not found {outmap}"
                    raise RuntimeError(msg)
                if self.verbosity > 1:
                    print(self.prefix + f"Coadded {outmap} in {time()-t1:.1f}s.")
                if result is not None:
                    m = result[outmap]
                    ic = result[self.invcov]
                    good = np.argwhere(m[0] != 0).ravel()
                    nmap, npix = m.shape
                    result[outmap] = (m[:, good], good, npix)
                    result[self.invcov] = (ic[:, good], good, npix)
                    if mapcache is not None and persistent:
                        # Write the maps to mapcache in compressed NESTED ordering
                        mapcache[outmap] = result[outmap]
                        mapcache[self.invcov] = result[self.invcov]
        return

    def get_sim_map(self, result, mapcache=None, persistent=False):
        """ Load or co-add the real data map and inverse covariance
        Args:
            result (dict) : dictionary for the co-added maps. Key is the
                intended path
            mapcache : Preloaded maps for faster co-add
            persistent (bool) : If True, the co-added map is added as an
                entry to mapcache
        """
        self.outmap_sim = f"{self.outdir}/{self.name}_sim_filtered_map.fits"
        self.invcov_sim = f"{self.outdir}/{self.name}_sim_filtered_invcov.fits"
        if result is not None and self.outmap_sim in result and self.invcov_sim in result:
            return

        if mapcache is not None and self.outmap_sim in mapcache:
            if self.verbosity > 2:
                print(self.prefix + f"Retrieving {self.outmap_sim} from cache")
            result[self.outmap_sim] = mapcache[self.outmap_sim]
            if self.verbosity > 2:
                print(self.prefix + f"Retrieving {self.invcov_sim} from cache")
            result[self.invcov_sim] = mapcache[self.invcov_sim]
        elif os.path.isfile(self.outmap_sim):
            if result is not None:
                if self.verbosity > 2:
                    print(self.prefix + f"Loading {self.outmap_sim} from disk")
                m = read_healpix(self.outmap_sim, None, nest=True)
                if self.verbosity > 2:
                    print(self.prefix + f"Loading {self.invcov_sim} from disk")
                invcov = read_healpix(self.invcov_sim, None, nest=True)
                good = np.argwhere(m[0] != 0).ravel()
                nmap, npix = m.shape
                result[self.outmap_sim] = (m[:, good], good, npix)
                result[self.invcov_sim] = (invcov[:, good], good, npix)
                if mapcache is not None and persistent:
                    mapcache[self.outmap_sim] = result[self.outmap_sim]
                    mapcache[self.invcov_sim] = result[self.invcov_sim]
        else:
            if self.verbosity > 1:
                print(self.prefix + f"Coadding {self.outmap_sim}")
            weights = OrderedDict()
            for path in self.paths:
                path_sim = path.replace(
                    self.outroot, self.outroot_tf
                ).replace(self.band, self.band_tf)
                weights[path_sim] = (1.0, 1.0)
            t1 = time()
            toast_healpix_coadd.main(
                opts=[
                    "--outmap",
                    self.outmap_sim,
                    "--invcov",
                    self.invcov_sim,
                    "DUMMY_INPUT",
                ],
                comm=None,
                cache=mapcache,
                result=result,
                prefix=self.prefix,
                weights=weights,
            )
            if (
                    result is None and not os.path.isfile(self.outmap_sim)
            ) or (
                result is not None and self.outmap_sim not in result
            ):
                msg = self.prefix + f"ERROR : coadded map, not found {self.outmap_sim}"
                raise RuntimeError(msg)
            if self.verbosity > 1:
                print(self.prefix + f"Coadded {self.outmap_sim} in {time()-t1:.1f}s.")
            if result is not None:
                m = result[self.outmap_sim]
                ic = result[self.invcov_sim]
                good = np.argwhere(m[0] != 0).ravel()
                nmap, npix = m.shape
                result[self.outmap_sim] = (m[:, good], good, npix)
                result[self.invcov_sim] = (ic[:, good], good, npix)
                if mapcache is not None and persistent:
                    mapcache[self.outmap_sim] = result[self.outmap_sim]
                    mapcache[self.invcov_sim] = result[self.invcov_sim]
        return

    def fitness(
            self,
            return_result=None,
            mapcache=None,
            input_map=None,
            hardcopy=False,
            recompute=False,
    ):
        """ Return the S/N of the transfer function-corrected noise spectrum """

        if self._fitness is None or recompute:
            if hardcopy:
                result = None
            elif return_result is None:
                result = OrderedDict()
            else:
                result = return_result
            if input_map is not None:
                # Use the provided input map to reduce I/O time
                result[self.fname_input] = input_map
            t0 = time()

            # Retrieve or coadd the maps
            self.get_map(result, mapcache, signflip=True)
            self.get_sim_map(result, mapcache)

            # Measure the co-added C_ell and the transfer function
            t1 = time()
            self.cl_signflips = []
            for iflip in range(self.nflip):
                self.cl_signflips.append(
                    get_cl(
                        self.outmap_signflips[iflip],
                        prefix=self.prefix,
                        cache=result,
                        verbosity=self.verbosity,
                    )
                )
            self.cl_sim, self.cl_sim_input, self.tf = get_tf(
                self.outmap_sim,
                self.fname_input,
                prefix=self.prefix,
                cache=result,
            )
            if self.verbosity > 2:
                print(self.prefix + f"Ran anafast in {time()-t1:.1f}s.")
            cls = self.cl_signflips
            lmax = cls[0][0].size - 1
            ell = np.arange(lmax + 1)
            good = self.tf != 0
            for cl in cls:
                cl[good] /= self.tf[good]

            # Optimize the S/N for a signal of our choice.
            # FIXME: Should we include lensing as a noise source in measuring S/N?
            if self.mode == "I":
                signal = self.theory["TT"][:lmax - 1] * 1e-12
            else:
                signal = self.theory["BB_primordial"][:lmax - 1] * 1e-12
            # Define "noise" as the mean signflip spectrum
            noise = np.zeros_like(signal)
            for cl in cls:
                if self.mode == "I":
                    # TT noise
                    noise += cl[0][2:]
                else:
                    # EE + BB noise
                    noise += cl[1][2:] + cl[2][2:]
            noise /= self.nflip
            # Joint EE + BB transfer function (just to count DoF) and
            # penalize extensive mode loss
            tf = np.sqrt(self.tf[1, 2:] * self.tf[2, 2:])
            mask = tf > 1e-3  # Don't trust TF below this level
            mask[np.isnan(tf)] = False
            mask[noise == 0] = False

            # Multipoles are not independent, bin the signal
            signal = signal[mask].copy()
            noise = noise[mask].copy()
            tf = tf[mask].copy()

            ell = ell[2:][mask].copy()
            binmin = 20
            binmax = min(600, lmax)
            ellbin = []
            signalbin = []
            noisebin = []
            nmodebin = []
            # Constant bin width
            lower = binmin
            while lower < binmax:
                upper = lower + self.wbin
                ii = np.logical_and(lower <= ell, ell < upper)
                if np.sum(ii) != 0:
                    ellbin.append(np.mean(ell[ii]))
                    signalbin.append(np.mean(signal[ii]))
                    noisebin.append(np.mean(noise[ii]))
                    # MASTER paper, Eq. (36)
                    # nmodebin.append(np.sum(np.sqrt((2 * ell[ii] + 1) * tf[ii])))
                    # Including the transfer function in mode count
                    # assumes that every filtered mode is lost.  That is
                    # overly pessimistic for cross-linked experiments
                    if self.score_function == "pessimistic":
                        nmodebin.append(np.sum((2 * ell[ii] + 1) * tf[ii]))
                    elif (
                            self.score_function == "optimistic"
                            or self.score_function == "classic"
                    ):
                        nmodebin.append(np.sum(2 * ell[ii] + 1))
                    else:
                        msg = f"Unknown score function: {self.score_function}"
                        raise RuntimeError(msg)
                lower = upper
            self.ellbin = np.array(ellbin)
            self.signalbin = np.array(signalbin)
            self.noisebin = np.array(noisebin)
            self.nmodebin = np.array(nmodebin)

            # self._fitness = np.sum(self.signalbin * self.nmodebin / self.noisebin)
            # self._fitness = np.sum(self.signalbin * np.sqrt(self.nmodebin) / self.noisebin)
            if self.score_function == "classic":
                self._fitness = np.sum(self.signalbin * self.nmodebin / self.noisebin)
            else:
                # This score function minimizes sigma(r) assuming that
                #   delta(C_ell) = N_ell / SQRT(nmode)
                self._fitness = np.sum(self.signalbin**2 * self.nmodebin / self.noisebin**2)
            if np.abs(self._fitness) < 1e-6:
                print(
                    self.prefix + f"ERROR : fitness is too low: {self._fitness}. "
                    f"signal = {np.sum(self.signalbin)**2} "
                    f"nmode = {np.sum(self.nmodebin)} "
                    f"noise = {np.sum(self.noisebin)} "
                )
            if self.verbosity > 1:
                print(
                    self.prefix + f"Analyzed fitness={self._fitness} in {time()-t0:.1f}s."
                )

            # DEBUG begin
            # for i in range(100):
            #     fname = f"debug_{i:04}.pck"
            #     if not os.path.isfile(fname):
            #         break
            # with open(fname, "wb") as f:
            #     pickle.dump([self, result], f)
            # print(self.prefix + f"Wrote {fname}")
            # DEBUG end

            if return_result is None:
                del result

        return self._fitness
