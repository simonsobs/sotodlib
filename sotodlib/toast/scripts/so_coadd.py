#!/usr/bin/env python3

# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script builds a coadded map, with optional noise weighting
based on spectral properties.
"""

import argparse
import os
import sys
import traceback

import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.table import QTable

import toast
from toast.timing import Timer
from toast.scripts.toast_healpix_coadd import main as toast_hpx_coadd_main


def get_files(args):
    obs = dict()
    iobs = 0
    for root, dirs, files in os.walk(args.in_root):
        for dir in dirs:
            if args.use_fits:
                map_file = f"{args.mapmaker_name}_map.fits"
            else:
                map_file = f"{args.mapmaker_name}_map.h5"
            map_path = os.path.join(root, dir, map_file)
            cl_path = os.path.join(args.stats_root, dir, "pseudo_cl.ecsv")
            if os.path.isfile(map_path):
                obs[dir] = {"map": map_path}
                if os.path.isfile(cl_path):
                    obs[dir]["cl"] = cl_path
                else:
                    obs[dir]["cl"] = None
                    msg = f"{dir} does not have a pseudo_cl file at {cl_path},"
                    msg += " will use the median weight or 1.0"
                    print(msg, flush=True)
                obs[dir]["index"] = iobs
                iobs += 1
    return obs


def get_weights(args, obs):
    weight_file = f"{args.coadd_root}_files.txt"
    plot_file = f"{args.coadd_root}_weights.pdf"
    ell_min = args.ell_weight_min
    ell_max = args.ell_weight_max

    files = [None for x in range(len(obs))]
    weights = -1 * np.ones(len(obs), dtype=np.float64)

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(6, 16), dpi=100)

    for oname, oprops in obs.items():
        files[oprops["index"]] = oprops["map"]
        if oprops["cl"] is None:
            continue
        # Load spectra
        spec = QTable.read(oprops["cl"], format="ascii.ecsv")

        # Compute the inverse noise weights
        ell_slc = slice(int(ell_min), int(ell_max), 1)
        spc_mean = np.mean(spec["cl_BB"].to_value(u.uK**2)[ell_slc])
        weights[oprops["index"]] = spc_mean
        tt_mean = np.mean(spec["cl_TT"].to_value(u.uK**2)[ell_slc])
        ee_mean = np.mean(spec["cl_EE"].to_value(u.uK**2)[ell_slc])
        print(f"{oname}:  {tt_mean} {ee_mean} {spc_mean}", flush=True)

        # Plot spectra
        ell = np.arange(len(spec))
        for icomp, comp in enumerate([spec["cl_TT"], spec["cl_EE"], spec["cl_BB"]]):
            axs[icomp].loglog(
                ell[2:],
                comp.to_value(u.uK**2)[2:],
                color="black",
                label=oname,
                linewidth=0.5,
            )

    # Add plot axis labels
    for icomp, comp in enumerate(["TT", "EE", "BB"]):
        axs[icomp].axvline(ell_min, color="red")
        axs[icomp].axvline(ell_max, color="red")
        axs[icomp].set_xlabel(r"Multipole, $\ell$")
        axs[icomp].set_ylabel(r"C$_\ell^{" + comp + r"}$ [$\mu$K$^2$]")

    # Compute histogram and add plot
    bad = weights < 0
    good = np.logical_not(bad)
    if len(good) > 0:
        # We have some spectra
        wt_med = np.median(weights[good])
        cutfactor = 3
        wt_cut = cutfactor * wt_med

        # Plot spectral values
        counts, bins = np.histogram(weights[good], bins=50, range=(0.0, wt_cut))
        axs[-1].stairs(counts, bins, color="black")
        axs[-1].axvline(wt_med, color="green", label="Median Weight")
        axs[-1].axvline(wt_cut, color="red", label="Cutoff for Coadd")
        axs[-1].legend(loc="best")

        # Apply cuts and relative weights
        remove = weights > wt_cut
        keep = np.logical_not(remove)
        weights[remove] = -1

        # FIXME: Investigate whether this noise weighting actually does anything
        # weights[keep] /= wt_med
        weights[keep] = 1.0

        # Handle maps with no spectra available.  We assign these the median
        # value.
        # weights[bad] = wt_med
        weights[bad] = 1.0
    else:
        # Set all weights to 1.0
        weights[:] = 1.0

    fig.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    # Write weights file
    with open(weight_file, "w") as f:
        for fname, wt in zip(files, weights):
            if wt > 0:
                f.write(f"{fname} {wt:0.12e}\n")


def main(opts=None, comm=None):
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    rank = 0
    if comm is not None:
        rank = comm.rank

    parser = argparse.ArgumentParser(description="Coadd maps with optional weighting")

    parser.add_argument(
        "--in_root",
        required=True,
        type=str,
        default=None,
        help="The top-level input directory of per-observation mapmaking products",
    )

    parser.add_argument(
        "--mapmaker_name",
        required=False,
        type=str,
        default="mapmaker",
        help="The base name of the mapmaker output files.",
    )

    parser.add_argument(
        "--use_fits",
        required=False,
        action="store_true",
        default=False,
        help="Output maps are in FITS, not HDF5 format.",
    )

    parser.add_argument(
        "--stats_root",
        required=False,
        type=str,
        default=None,
        help="The top-level directory with collected stats",
    )

    parser.add_argument(
        "--coadd_root",
        required=True,
        type=str,
        default=None,
        help="The root name for coadded maps and covariance.",
    )

    parser.add_argument(
        "--ell_weight_min",
        required=False,
        type=float,
        default=80.0,
        help="The minimum ell value to consider for spectral noise weights",
    )

    parser.add_argument(
        "--ell_weight_max",
        required=False,
        type=float,
        default=120.0,
        help="The maximum ell value to consider for spectral noise weights",
    )

    args = parser.parse_args(args=opts)
    if args.ell_weight_min >= args.ell_weight_max:
        raise RuntimeError("Ell range does not make sense")

    # One process computes the spectral noise weights / cuts
    if rank == 0:
        obs = get_files(args)
        get_weights(args, obs)
    if comm is not None:
        comm.barrier()

    # Coadd all the maps
    opts = [
        "--outmap",
        f"{args.coadd_root}_map.h5",
        "--cov",
        f"{args.coadd_root}_cov.h5",
        "--invcov",
        f"{args.coadd_root}_invcov.h5",
        "--hits",
        f"{args.coadd_root}_hits.h5",
        "--rcond_limit",
        "1.0e-3",
        "--double_precision",
        f"{args.coadd_root}_files.txt",
    ]
    toast_hpx_coadd_main(opts=opts, comm=comm)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(opts=None, comm=world)


if __name__ == "__main__":
    cli()
