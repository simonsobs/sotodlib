#!/usr/bin/env python3

# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script builds a coadded map, with optional noise weighting
based on spectral properties.
"""

import argparse
import os

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.table import QTable, Column

import toast
from toast.pixels_io_healpix import read_healpix
from toast.pixels_io_wcs import read_wcs
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


def compute_coadd_spectra(args):
    hits_file = f"{args.coadd_root}_hits.h5"
    map_file = f"{args.coadd_root}_map.h5"
    invcov_file = f"{args.coadd_root}_invcov.h5"
    try:
        hits = read_healpix(hits_file, nest=False)
        use_healpix = True
    except Exception as ehpx:
        try:
            hits = read_wcs(hits_file)
            use_healpix = False
        except Exception as ewcs:
            msg = "Cannot read hits file as either healpix or WCS format"
            raise RuntimeError(msg)

    if not use_healpix:
        raise NotImplementedError("WCS processing not yet implemented")

    good = hits > 3
    bad = np.logical_not(good)
    nside = hp.get_nside(hits)
    data_map, data_header = read_healpix(map_file, field=None, nest=False, h=True)

    if "UNITS" in data_header:
        map_units = u.Unit(data_header["UNITS"])
    else:
        map_units = u.K
    spec_units = map_units**2

    # We just use the intensity covariance for weighting
    invcov = read_healpix(invcov_file, field=(0,), nest=False)
    invcov /= np.amax(invcov)
    invcov[bad] = 0.0

    fsky = np.mean(invcov**2)
    weighted_map = data_map * invcov
    weighted_map[:, bad] = 0.0
    mono = np.mean(weighted_map[0, good])
    weighted_map[0, good] -= mono

    lmax = 3 * nside
    cl = hp.anafast(weighted_map, lmax=lmax, iter=3) / fsky
    cl_file_fits = f"{args.coadd_root}_cl.fits"
    hp.write_cl(cl_file_fits, cl, dtype=np.float64, overwrite=True)

    cl_file = f"{args.coadd_root}_cl.ecsv"
    cl_table = QTable(
        [
            Column(name="cl_TT", data=cl[0], unit=spec_units),
            Column(name="cl_EE", data=cl[1], unit=spec_units),
            Column(name="cl_BB", data=cl[2], unit=spec_units),
            Column(name="cl_TE", data=cl[3], unit=spec_units),
            Column(name="cl_EB", data=cl[4], unit=spec_units),
            Column(name="cl_TB", data=cl[5], unit=spec_units),
        ]
    )
    cl_table.meta["toast_version"] = toast.__version__
    cl_table.write(cl_file, format="ascii.ecsv", overwrite=True)

    # Plot in uK
    scale = 1.0 * spec_units
    cl_uK = cl * scale.to_value(u.uK**2)

    img_file = f"{args.coadd_root}_cl.pdf"
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ell = np.arange(cl[0].size)
    for icomp, comp in enumerate(["TT", "EE", "BB"]):
        ax = fig.add_subplot(3, 1, icomp + 1)
        ax.loglog(ell[2:], cl_uK[icomp][2:], color="black")
        ax.set_xlabel(r"Multipole, $\ell$")
        ax.set_ylabel(r"C$_\ell^{" + comp + r"}$ [$\mu$K$^2$]")
    fig.tight_layout()
    plt.savefig(img_file)
    plt.close()


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

    # Compute the pseudo spectra of the coadd
    compute_coadd_spectra(args)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(opts=None, comm=world)


if __name__ == "__main__":
    cli()
