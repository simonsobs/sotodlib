#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import sys

import numpy as np

from toast.mpi import MPI
import toast
import toast.map as tm
import toast.tod as tt

from sotodlib.hardware import get_example, sim_telescope_detectors
from sotodlib.data.toast_load import load_data


def binned_map(data, npix, subnpix, out="."):
    """Make a binned map

    This function should exist in toast, but all the pieces do.  If we are
    doing MCs we break these operations into two pieces and only generate
    the noise weighted map each realization.

    """
    # The global MPI communicator
    cworld = data.comm.comm_world

    # get locally hit pixels
    lc = tm.OpLocalPixels()
    localpix = lc.exec(data)

    # find the locally hit submaps.
    localsm = np.unique(np.floor_divide(localpix, subnpix))

    # construct distributed maps to store the covariance,
    # noise weighted map, and hits
    invnpp = tm.DistPixels(comm=cworld, size=npix, nnz=6, dtype=np.float64,
                           submap=subnpix, local=localsm)
    hits = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=1,
                         dtype=np.int64, submap=subnpix, local=localsm)
    zmap = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=3,
                         dtype=np.float64, submap=subnpix, local=localsm)

    invnpp.data.fill(0.0)
    hits.data.fill(0)
    zmap.data.fill(0.0)

    start = MPI.Wtime()
    if cworld.rank == 0:
        print("Accumulating hits and N_pp'^-1 ...", flush=True)

    # Setting detweights to None gives uniform weighting.
    build_invnpp = tm.OpAccumDiag(
        detweights=None,
        invnpp=invnpp,
        hits=hits,
        zmap=zmap,
        name="signal",
        pixels="pixels",
        weights="weights",
        common_flag_name="flags_common",
        common_flag_mask=1
    )
    build_invnpp.exec(data)

    invnpp.allreduce()
    hits.allreduce()
    zmap.allreduce()

    cworld.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("Building hits and N_pp^-1 took {:.3f} s".format(elapsed),
              flush=True)

    if cworld.rank == 0:
        print("Writing hits and N_pp'^-1 ...", flush=True)

    hits.write_healpix_fits(os.path.join(out, "hits.fits"))
    invnpp.write_healpix_fits(os.path.join(out, "invnpp.fits"))

    start = stop
    if cworld.rank == 0:
        print("Inverting N_pp'^-1 ...", flush=True)

    # invert it
    tm.covariance_invert(invnpp, 1.0e-3)

    cworld.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("Inverting N_pp^-1 took {:.3f} s".format(elapsed),
              flush=True)

    if cworld.rank == 0:
        print("Writing N_pp' ...", flush=True)
    invnpp.write_healpix_fits(os.path.join(out, "npp.fits"))

    start = stop

    if cworld.rank == 0:
        print("Computing binned map ...", flush=True)

    tm.covariance_apply(invnpp, zmap)

    cworld.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("Computing binned map took {:.3f} s"
              .format(elapsed), flush=True)
    start = stop

    if cworld.rank == 0:
        print("Writing binned map ...", flush=True)
    zmap.write_healpix_fits(os.path.join(out, "binned.fits"))
    if cworld.rank == 0:
        print("Binned map done", flush=True)

    return


# Our toast communicator- use the default for now, which is one
# process group spanning all processes.
comm = toast.Comm()

if comm.world_rank == 0:
    print("Simulating all detector properties...", flush=True)
# First, get the list of detectors we want to use
# (Eventually we would load this from disk.  Here we simulate it.)
hw = get_example()
dets = sim_telescope_detectors(hw, "LAT")
hw.data["detectors"] = dets

if comm.world_rank == 0:
    print("Selecting detectors...", flush=True)
# Downselect to just 10 pixels on one wafer
#small_hw = hw.select(match={"wafer": "41", "pixel": "00."})
#small_hw = hw.select(match={"wafer": "41"})
small_hw = hw.select(match={"wafer": "40"})
#small_hw = hw.select(match={"band": "LF1"})
if comm.world_rank == 0:
    small_hw.dump("selected.toml", overwrite=True)

# The data directory (this is a single band)
# dir = "/project/projectdirs/sobs/sims/pipe-s0001/datadump_LAT_LF1"
if len(sys.argv) > 1:
    dir = sys.argv[1]
    print("Loading data from {}".format(dir), flush=True)
else:
    dir = "datadump_LAT_LF1"
# dir = "/home/kisner/scratch/sobs/pipe/datadump_LAT_LF1"

# Here we divide the data for each observation into a process grid.
# "detranks = 1" is one extreme, where every process has all detectors for
# some number of frame-sized chunks.  The other extreme is where each process
# has the full time length for a subset of detectors.
#
# (all detectors for some number of frame-sized chunks):
detranks = 1
# (some detectors for the whole observation):
# detranks = comm.group_size

if comm.world_rank == 0:
    print("Loading data from disk...", flush=True)

# Load our selected data
data = load_data(
    dir,
    obs=["CES-ATACAMA-LAT-Tier1DEC-035..-045_RA+040..+050-0-0"],
    comm=comm,
    dets=small_hw,
    detranks=detranks
)

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
nside = 2048
pointing = tt.OpPointingHpix(
    nside=nside,
    nest=True,
    mode="IQU",
    pixels="pixels",
    weights="weights"
)
pointing.exec(data)

# Apply a polynomial filter per subscan
polyfilter = tt.OpPolyFilter(
    order=3,
    name="signal",
    common_flag_name="flags_common",
    common_flag_mask=1
)
polyfilter.exec(data)

# Make a binned map
npix = 12 * nside**2
subnpix = 12 * 16**2
binned_map(data, npix, subnpix)

if comm.world_rank == 0:
    # Plot the hit map
    import healpy as hp
    import matplotlib.pyplot as plt
    hits = hp.read_map("hits.fits")
    hp.gnomview(hits, rot=(41.4, -43.4), xsize=800, reso=2.0)
    plt.savefig("hits.png")
    binned = hp.read_map("binned.fits")
    hp.gnomview(binned, rot=(41.4, -43.4), xsize=800, reso=2.0)
    plt.savefig("binned.png")
