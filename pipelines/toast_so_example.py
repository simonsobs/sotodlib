#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import numpy as np

from toast.mpi import MPI
import toast
import toast.map as tm
import toast.tod as tt

from sotodlib.hardware import get_example, sim_telescope_detectors
from sotodlib.data.toast_load import load_data


def binned_map(data, npix, subnpix, out="."):
    start = MPI.Wtime()

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
    start = stop

    hits.write_healpix_fits(os.path.join(out, "hits.fits"))
    invnpp.write_healpix_fits(os.path.join(out, "invnpp.fits"))

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("Writing hits and N_pp^-1 took {:.3f} s".format(elapsed),
              flush=True)
    start = stop

    # invert it
    tm.covariance_invert(invnpp, 1.0e-3)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Inverting N_pp^-1 took {:.3f} s".format(elapsed),
              flush=True)
    start = stop

    invnpp.write_healpix_fits(os.path.join(out, "npp.fits"))

    cworld.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("Writing N_pp took {:.3f} s".format(elapsed),
              flush=True)
    start = stop

    tm.covariance_apply(invnpp, zmap)

    cworld.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if cworld.rank == 0:
        print("  Computing binned map took {:.3f} s"
              .format(elapsed), flush=True)
    start = stop

    zmap.write_healpix_fits(os.path.join(out, "binned.fits"))
    return


# First, get the list of detectors we want to use

# (Eventually we would load this from disk.  Here we simulate it.)
hw = get_example()
dets = sim_telescope_detectors(hw, "LAT")
hw.data["detectors"] = dets

# Dowselect to just 10 pixels on one wafer
small_hw = hw.select(match={"wafer": ["00"], "pixel": "00."})

# The data directory (this is a single band)
dir = "/project/projectdirs/sobs/sims/pipe-s0001/datadump_LAT_UHF1"

# Our toast communicator- use the default for now, which is one
# process group spanning all processes.
comm = toast.Comm()

# Load our selected data
data = load_data(dir, comm=comm, dets=small_hw)

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
