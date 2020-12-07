#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse
import glob
import os
import sys

import numpy as np

from toast.mpi import MPI
import toast
import toast.todmap
import toast.tod
import toast.pipeline_tools as toast_tools

from sotodlib.toast.load import load_data


def main():
    # Our toast communicator- use the default for now, which is one
    # process group spanning all processes.

    comm = toast.Comm()

    parser = argparse.ArgumentParser(
        description="Make a Healpix map of a given observation ",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "obsdir", help="Input directory for all observations"
    )
    parser.add_argument(
        "--obs", required=False, help="Comma-separated list of observations"
    )
    parser.add_argument(
        "--nside", required=False, type=np.int, default=1024, help="Map resolution",
    )
    toast_tools.add_polyfilter_args(parser)

    args = parser.parse_args()

    print("Loading data from {}".format(args.obsdir), flush=True)

    if args.obs is None:
        paths = glob.glob(args.obsdir + "/*")
        obs = [os.path.basename(path) for path in paths]
        print("Found {} observations: {}".format(len(obs), obs))
    else:
        obs = []
        for path in args.obs.split(","):
            obs.append(path)
        print("Mapping {} observations: {}".format(len(obs), obs))

    if comm.world_rank == 0:
        print("Loading data from disk...", flush=True)

    # Load our selected data
    data = load_data(args.obsdir, obs=obs, comm=comm)

    # Everybody look at their data
    my_world_rank = data.comm.world_rank
    my_group_rank = data.comm.group_rank
    for ob in data.obs:
        tod = ob["tod"]
        my_dets = tod.local_dets
        my_first_samp, my_nsamp = tod.local_samples
        msg = "proc {} with group rank {} has {} dets for samples {} - {}".format(
            my_world_rank, my_group_rank, len(my_dets), my_first_samp,
            my_first_samp + my_nsamp - 1
        )
        print(msg, flush=True)


    if comm.world_rank == 0:
        # Plot some local data from the first observation
        import matplotlib.pyplot as plt
        ob = data.obs[0]
        tod = ob["tod"]
        boloname = tod.local_dets[0]
        bolodata = tod.cache.reference("signal_{}".format(boloname))
        fig = plt.figure()
        plt.plot(np.arange(len(bolodata)), bolodata)
        plt.savefig("bolo_{}.png".format(boloname))
        del bolodata

    # Construct a pointing matrix
    pointing = toast.todmap.OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        pixels="pixels",
        weights="weights"
    )
    pointing.exec(data)

    # Filter signal

    if args.apply_polyfilter:
        toast_tools.apply_polyfilter(args, comm, data, "signal")

    # Bin the TOD to a map

    mapmaker = toast.todmap.OpMapMaker(
        nside=args.nside,
        nnz=3,
        name="signal",
        pixels="pixels",
        intervals=None,
        baseline_length=None,
        use_noise_prior=False,
        outdir=".",
    )
    mapmaker.exec(data)

    # Plot

    if comm.world_rank == 0:
        # Plot the hit map
        import healpy as hp
        import matplotlib.pyplot as plt

        plt.figure(figsize=[18, 12])

        hits = hp.read_map("hits.fits")
        imax = np.argmax(hits)
        lon, lat = hp.pix2ang(hp.get_nside(hits), imax, lonlat=True)
        hits[hits == 0] = hp.UNSEEN
        hp.mollview(hits, xsize=1200, sub=[2, 2, 1], title="hits")
        hp.gnomview(
            hits,
            rot=(lon, lat),
            xsize=800,
            reso=1.0,
            sub=[2, 2, 2],
            title="hits, lon={:.2f}deg, lat={:.2f}deg".format(lon, lat),
        )

        binned = hp.read_map("binned.fits")
        binned[binned == 0] = hp.UNSEEN
        plt.savefig("binned.png")
        hp.mollview(binned, xsize=1200, sub=[2, 2, 3], title="binned")
        hp.gnomview(
            binned,
            rot=(lon, lat),
            xsize=800,
            reso=1.0,
            sub=[2, 2, 4],
            title="binned, lon={:.2f}deg, lat={:.2f}deg".format(lon, lat),
        )
        fname = "{}.png".format(os.path.basename(args.obsdir))
        plt.savefig(fname)
        print("Plot saved in", fname)


if __name__ == "__main__":
    main()
