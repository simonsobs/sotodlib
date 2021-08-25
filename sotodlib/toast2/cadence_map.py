# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import h5py
import healpy as hp
import numpy as np

import toast

from toast.mpi import MPI


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400.0 + 2440587.5


def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5


class OpCadenceMap(toast.Operator):
    """ Tabulate which days each pixel on the map is visited.
    """

    def __init__(
            self,
            pixels="pixels",
            outdir=".",
            outprefix="",
            common_flag_mask=1,
            flag_mask=1,
            nest=True,
    ):
        self.pixels = pixels
        self.outdir = outdir
        self.outprefix = outprefix
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask
        self.nest = nest

    def exec(self, data):
        comm = data.comm.comm_world
        if comm is None:
            rank = 0
        else:
            rank = comm.rank

        npix = data[f"{self.pixels}_npix"]

        if rank == 0:
            os.makedirs(self.outdir, exist_ok=True)

        # determine the number of modified Julian days

        tmin = 1e30
        tmax = -1e30
        for obs in data.obs:
            tod = obs["tod"]
            times = tod.local_times()
            tmin = min(tmin, times[0])
            tmax = max(tmax, times[-1])

        if comm is not None:
            tmin = comm.allreduce(tmin, MPI.MIN)
            tmax = comm.allreduce(tmax, MPI.MAX)

        MJD_start = int(to_MJD(tmin))
        MJD_stop = int(to_MJD(tmax)) + 1
        nday = MJD_stop - MJD_start

        # Flag all pixels that are observed on each MJD

        if rank == 0:
            all_hit = np.zeros([nday, npix], dtype=bool)

        buflen = 10  # Number of days to process at once
        buf = np.zeros([buflen, npix], dtype=bool)
        day_start = MJD_start
        while day_start < MJD_stop:
            day_stop = min(MJD_stop, day_start + buflen)
            if rank == 0:
                print(
                    f"Processing {MJD_start} <= {day_start} - {day_stop} <= {MJD_stop}"
                )
            buf[:, :] = False
            for obs in data.obs:
                tod = obs["tod"]
                times = tod.local_times()
                days = to_MJD(times).astype(int)
                if days[0] >= day_stop or days[-1] < day_start:
                    continue
                cflag = (tod.local_common_flags() & self.common_flag_mask) != 0
                for day in range(day_start, day_stop):
                    good = days == day
                    if not np.any(good):
                        continue
                    good[cflag] = False
                    for det in tod.local_dets:
                        flag = tod.local_flags(det) & self.flag_mask
                        mask = np.logical_and(good, flag == 0)
                        pixels = tod.cache.reference(f"{self.pixels}_{det}")
                        mask[pixels < 0] = False
                        buf[day - day_start][pixels[mask]] = True
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.LOR)
            if rank == 0:
                for i in range(day_start, day_stop):
                    all_hit[i - MJD_start] = buf[i - day_start]
            day_start = day_stop

        if rank == 0:
            if self.outprefix is None:
                outprefix = ""
            else:
                outprefix = self.outprefix
                if not outprefix.endswith("_"):
                    outprefix += "_"
            if False:
                # FITS output (exhausts memory)
                fname = os.path.join(self.outdir, outprefix + "cadence.fits")
                header = [
                    ("MJDSTART", MJD_start, "First MJD"),
                    ("MJDSTOP", MJD_stop, "Last MJD"),
                ]
                hp.write_map(
                    fname,
                    all_hit,
                    nest=self.nest,
                    dtype=bool,
                    extra_header=header,
                    overwrite=True,
                )
            else:
                fname = os.path.join(self.outdir, outprefix + "cadence.h5")
                with h5py.File(fname, "w") as f:
                    dset = f.create_dataset("cadence", data=all_hit)
                    dset.attrs["MJDSTART"] = MJD_start
                    dset.attrs["MJDSTOP"] = MJD_stop
                    dset.attrs["NESTED"] = self.nest
            print(f"Wrote cadence map to {fname}.", flush=True)
