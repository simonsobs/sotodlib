#!/usr/bin/env python3

# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script finds the atomic map filter combination that optimizes
the signal to noise in the coadd.  Ported from the original at:

pwg-scripts/pwg-tds/sat-filter-study/common_mode_study/optimize.py

With updates to support different directory structures and some other
options.

"""

import argparse
import os
import re
import pickle
from time import time

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import numpy as np
from ruamel.yaml import YAML

import toast
from toast.mpi import MPI
from toast.pixels_io_healpix import read_healpix
from toast.utils import memreport
from toast.traits import string_to_trait, trait_to_string

from .so_fb_optimize_ga import WaferObservation, Individual, fit_noise_model
from .so_fb_optimize_mapcache import MapCache


def load_first_guess(comm, outdir, input_map, mode):
    first_guess = None
    all_paths = None

    if comm is None or comm.rank == 0:
        fname = f"{outdir}/optimized_step0000.pck"
        if os.path.isfile(fname):
            print(f"Loading first guess from {fname}")
            with open(fname, "rb") as f:
                first_guess = pickle.load(f)
            try:
                all_paths = first_guess.all_paths
            except:
                # Old style pickle file
                first_guess, all_paths = first_guess
    if comm is not None:
        all_paths = comm.bcast(all_paths)
        if all_paths is not None:
            # Successful load. Broadcast the first_guess 
            first_guess = comm.bcast(first_guess)

    return first_guess, all_paths


def save_first_guess(comm, outdir, first_guess, all_paths):
    if comm is None or comm.rank == 0:
        fname = f"{outdir}/optimized_step0000.pck"
        print(f"Saving first guess to {fname}")
        with open(fname, "wb") as f:
            pickle.dump(first_guess, f)


def list_paths(indir, indir_tf, band, config_pattern):
    if comm is None or comm.rank == 0:
        obs_pattern = f"{indir}/obs_*"
        obsdirs = glob.glob(obs_pattern)
        if len(obsdirs) == 0:
            msg = f"No directories match pattern = {obs_pattern}"
            raise RuntimeError(msg)
        if config_pattern is None:
            config_test = None
        else:
            config_test = re.compile(config_pattern)
        all_obs = []
        nobs = 0
        nmap = 0
        print(f"Finding all map paths")
        all_paths = []
        for obsdir in obsdirs:
            obs = os.path.basename(obsdir)
            for ws in range(7):
                paths = []
                map_pattern = f"{obsdir}/{band}/filterbin*_{ws}_noiseweighted_filtered_map.h5"
                fnames = glob.glob(map_pattern)
                if len(fnames) == 0:
                    print(f"No observation maps match pattern = '{map_pattern}'")
                    continue
                for fname_map in sorted(fnames):
                    # Assume that the configuration is identified by a
                    # string of numbers following "filterbin" in the map
                    # name

                    if config_test is not None:
                        config = re.search(
                            ".*filterbin([0-9]+)_.*.h5", fname_map
                        ).groups()[0]
                        if config_test.match(config) is None:
                            continue
                    # v39, ver=001
                    """
                    if config[0] != "1":
                        # print(f"Warning: rejecting filter config = {config}")
                        continue
                    if config[2] != "0":
                        # print(f"Warning: rejecting filter config = {config}")
                        continue
                    """
                    # v39, ver=002
                    """
                    if config[0] != "0":
                        # print(f"Warning: rejecting filter config = {config}")
                        continue
                    if config[2] != "0":
                        # print(f"Warning: rejecting filter config = {config}")
                        continue
                    """
                    # Temporarily reject certain configurations
                    # ver=011 & 012
                    """
                    if config[0] != "1":
                        print(f"Warning: rejecting filter config = {config}")
                        continue
                    if config[2] != "0":
                        print(f"Warning: rejecting filter config = {config}")
                        continue
                    """

                    paths.append(fname_map)
                    # Separately record all paths for caching later
                    fname_invcov = fname_map.replace(
                        "_noiseweighted_filtered_map.h5", "_filtered_invcov.h5"
                    )
                    all_paths.append(fname_map)
                    all_paths.append(fname_invcov)
                    all_paths.append(fname_map.replace(indir, indir_tf))
                    all_paths.append(fname_invcov.replace(indir, indir_tf))
                if len(paths) != 0:
                    all_obs.append(WaferObservation(obs, ws, band, paths))
                    nobs += 1
                    nmap += len(paths)
        print(
            f"Found a total of {nmap} maps in {nobs} wafer observations "
            f"and {len(obsdirs)} sessions."
        )
    else:
        all_paths = None
        all_obs = None

    if comm is not None:
        all_paths = comm.bcast(all_paths)
        all_obs = comm.bcast(all_obs)

    return all_paths, all_obs




def main(opts=None, comm=None):
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    rank = 0
    if comm is not None:
        rank = comm.rank

        parser = argparse.ArgumentParser(description="Find optimal filter configuration")
    
    parser.add_argument(
        "--hybridize",
        required=False,
        default=False,
        action="store_true",
        help="Hybridize better candidates by combining best steps",
    )

    parser.add_argument(
        "--band",
        default="f150",
        help="One of f090, f150, f230, f280",
    )

    parser.add_argument(
        "--mode",
        default="IQU",
        help="One of I, IQU",
    )

    parser.add_argument(
        "--sampling-mode",
        default="step",
        help="'step' or 'full'",
    )

    parser.add_argument(
        "--fname_input",
        default="cmb_map_SAT_{band}_w25.healpix.nside0512.fits",
        help="Input map to compare to simulations",
    )

    parser.add_argument(
        "--cachedir",
        default="map_cache",
        help="Map cache to use",
    )

    parser.add_argument(
        "--indir",
        required=True,
        help="Input directory for filtered observation maps",
    )

    parser.add_argument(
        "--indir-tf",
        required=True,
        help="Input directory for simulated observation maps",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for optimization outputs",
    )

    parser.add_argument(
        "--nstep",
        default=1000,
        type=int,
        help="Maximum number of optimization steps",
    )

    parser.add_argument(
        "--wbin",
        default=30,
        type=int,
        help="Width of ell bins in score function",
    )

    parser.add_argument(
        "--score-function",
        default="optimistic",
        help="Score function to maximize",
    )

    parser.add_argument(
        "--nflip",
        default=10,
        type=int,
        help="Number of signflips",
    )

    parser.add_argument(
        "--theory",
        default="planck2018.pck",
        help="Pickle file with CAMB results for evaluating SNR",
    )

    parser.add_argument(
        "--config-pattern",
        required=False,
        help="Regex to match against the filter configuration string",
    )

    args = parser.parse_args(args=opts)

    fname_input = args.fname_input.format(band=args.band)

    if comm is None or comm.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        # Load fiducial C_ell
        with open(args.theory, "rb") as f:
            theory = pickle.load(f)
        theory_ell = theory["ells"]
        # theory = [
        #     theory["TT"],
        #     theory["EE"],
        #     theory["BB_primordial"] + theory["BB_lensing"]
        # ]
        # Apply Gaussian beam to fiducial C_ell
        fwhm_arcmin = {
            "f030" : 91.0,
            "f040" : 63.0,
            "f090" : 30.0,
            "f150" : 17.0,
            "f230" : 11.0,
            "f290" : 9.0,
        }[args.band]
        bl = hp.gauss_beam(
            np.radians(fwhm_arcmin / 60), lmax=theory_ell[-1], pol=True
        ).T
        theory["TT"] *= bl[0][2:]**2
        theory["EE"] *= bl[1][2:]**2
        theory["BB_primordial"] *= bl[2][2:]**2
        theory["BB_lensing"] *= bl[2][2:]**2
    else:
        theory = None
    if comm is not None:
        theory = comm.bcast(theory)


    if rank == 0:
        print(f"Running Optimize with {ntask} NPI tasks")
    prefix = f"{rank:04} : "

    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    if rank == 0:
        print(f"Start of the workflow:  {mem}")

    # (
    #     indir,
    #     indir_tf,
    #     outdir,
    #     band,
    #     theory,
    #     fname_input,
    #     nstep,
    #     mode,
    #     cachedir,
    #     wbin,
    #     score_function,
    #     sampling_mode,
    #     nflip,
    #     hybridize,
    #     pattern,
    # ) = init(comm)

    mapcache = MapCache(comm=comm, prefix=prefix, cachedir=f"{args.cachedir}/{args.band}")
    input_map = read_healpix(fname_input, [0, 1, 2], nest=False)

    first, all_paths = load_first_guess(comm, args.outdir, input_map, args.mode)

    # Temporary fix for old, saved guesses
    if first is not None and not hasattr(first, "band"):
        first.band = args.band
        first.band_tf = args.band
    # Temporary fix ends

    # Temporary fix for old, saved guesses
    if first is not None and not hasattr(first, "wbin"):
        first.wbin = args.wbin
        first.score_function = args.score_function
    # Temporary fix ends

    if all_paths is None:
        all_paths, all_obs = list_paths(args.indir, args.indir_tf, args.band, args.config_pattern)

    mapcache.load()
    if not mapcache.loaded:
        if comm is None or comm.rank == 0:
            # Check that all listed maps exist
            missing = ""
            for path in all_paths:
                if not os.path.isfile(path):
                    missing += f"{path}\n"
            if len(missing) != 0:
                msg = f"ERROR: following maps are not available on disk:\n{missing}"
                raise RuntimeError(missing)
        if comm is not None:
            comm.Barrier()
        mapcache.load_maps(all_paths)
        mapcache.save()
    if comm is not None:
        comm.Barrier()
    cache = mapcache.to_dict()

    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    if rank == 0:
        print(f"After loading map cache:  {mem}")

    if first is None:
        first = generate_first_guess(
            comm,
            all_obs,
            theory,
            args.outdir,
            fname_input,
            args.indir,
            args.indir_tf,
            args.band,
            cache,
            all_paths,
            input_map,
            args.mode,
            args.wbin,
            args.score_function,
            args.nflip,
        )
        save_first_guess(comm, args.outdir, first, all_paths)

    first.prefix = prefix
    first.outdir = args.outdir

    if rank == 0:
        nobs = len(first.observations)
        nobs_task = int(nobs / ntask)
        if nobs_task * ntask < nobs:
            nobs_task += 1
        print(
            f"There are {nobs} wafer observations, {ntask} tasks and {nobs/ntask} "
            f"observations per task."
        )
        ntask_use = int(nobs / nobs_task)
        if ntask_use * nobs_task < nobs:
            ntask_use += 1
        ntask_idle = ntask - ntask_use
        print(f"There will be {ntask_idle} tasks with no observations.")

    current = first
    fitness = [current.fitness()]

    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    if rank == 0:
        print(f"Before iterations:  {mem}")

    for step in range(1, args.nstep + 1):
        # for step in range(700, nstep + 1):
        next_ = get_next(
            comm,
            prefix,
            args.outdir,
            current,
            first,
            step,
            cache,
            all_paths,
            input_map,
            args.sampling_mode,
            args.band,  # temporary
            args.wbin,
            args.score_function,
            args.hybridize,
        )
        if next_ is None or next_.name == current.name:
            if args.sampling_mode == "step":
                if comm is None or comm.rank == 0:
                    print(
                        f"Failed to improve fitness with steps, switching to "
                        f"full optimization"
                    )
                args.sampling_mode = "full"
                next_ = current
            else:
                break
        if comm is None or comm.rank == 0:
            plot(
                args.outdir,
                first,
                next_,
                step,
                args.mode,
                args.sampling_mode,
                args.wbin,
                args.score_function,
                args.hybridize,
            )
            save_config(args.outdir, next_, step)
        next_.prefix = prefix
        current = next_
        fitness.append(current.fitness())

    plot_fitness(comm, args.outdir, fitness)
    if comm is None or comm.rank == 0:
        if next_ is None:
            save_config(args.outdir, current)
        else:
            save_config(args.outdir, next_)






def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(opts=None, comm=world)


if __name__ == "__main__":
    cli()
