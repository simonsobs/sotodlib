import os
import pickle
import sys
from time import time

import numpy as np
from toast.pixels_io_healpix import read_healpix
from mpi4py import MPI
from pshmem import MPIShared


class MapCache:
    def __init__(self, comm=None, dtype=np.float32, prefix=None, cachedir="map_cache", shared=True):
        self.comm = comm
        if self.comm is None:
            self.rank = 0
            self.ntask = 1
            self.prefix = ""
        else:
            self.rank = self.comm.rank
            self.ntask = self.comm.size
        if prefix is None:
            self.prefix = f"{self.rank:04} : "
        else:
            self.prefix = prefix
        self.dtype = dtype
        self.fnames = None
        self.npix = None
        self.map_cache = None
        self.mask_cache = None
        self.map_offsets = None
        self.mask_offsets = None
        self.cachedir = cachedir
        self.loaded = False
        self.shared = shared

    def _load_maps(self, fnames):
        """ Load the maps on one process """

        map_offset = 0
        mask_offset = 0
        my_map_cache = []
        my_mask_cache = []
        my_map_offsets = {}
        my_mask_offsets = {}
        npix = None
        for fname in fnames:
            # Load
            t1 = time()
            m = read_healpix(fname, None, nest=True, dtype=self.dtype, verbose=False)
            print(self.prefix + f"Loaded {fname} in {time()-t1:.3f}s")
            m = np.atleast_2d(m)
            if npix is None:
                npix = m[0].size
            else:
                if npix != m[0].size:
                    raise RuntimeError("Size mismatch")
            # Compress
            good = np.argwhere(m[0] != 0).ravel()
            m = m[:, good]
            # Cache
            my_map_cache.append(np.hstack(m))
            my_mask_cache.append(good)
            my_map_offsets[fname] = slice(map_offset, map_offset + m.size)
            my_mask_offsets[fname] = slice(mask_offset, mask_offset + good.size)
            # Update offsets
            map_offset += m.size
            mask_offset += good.size

        if len(my_map_cache) > 0:
            my_map_cache = np.hstack(my_map_cache)
            my_mask_cache = np.hstack(my_mask_cache)
        else:
            my_map_cache = np.empty(0, self.dtype)
            my_mask_cache = np.empty(0, int)
            

        return npix, my_map_cache, my_mask_cache, my_map_offsets, my_mask_offsets

    def load_maps(self, fnames):
        """ Load the maps and place them in shared memory """

        t1 = time()
        self.fnames = fnames

        nfile = len(self.fnames)
        nfile_per_rank = nfile // self.ntask
        if nfile_per_rank * self.ntask < nfile:
            nfile_per_rank += 1
        my_start = nfile_per_rank * self.rank
        my_stop = min(nfile, my_start + nfile_per_rank)
        (
            npix, my_map_cache, my_mask_cache, my_map_offsets, my_mask_offsets
        ) = self._load_maps(self.fnames[my_start : my_stop])

        if self.comm is not None:
            self.comm.Barrier()
            if self.comm.rank == 0:
                print(
                    self.prefix + f"{nfile} maps loaded in {time() - t1:.1f}s. "
                    f"Gathering loaded maps to the root process"
                )
            t2 = time()
            npix = self.comm.bcast(npix)
            # Gather maps
            nrecv_map = self.comm.gather(my_map_cache.size)
            if self.comm.rank == 0:
                nrecv_tot = np.sum(nrecv_map)
                map_cache = np.empty(nrecv_tot, dtype=self.dtype)
            else:
                map_cache = None
            offset = 0
            for isend in range(self.comm.size):
                if self.comm.rank == 0:
                    start = offset
                    stop = start + nrecv_map[isend]
                    if isend == 0:
                        map_cache[start:stop] = my_map_cache
                    else:
                        self.comm.Recv(map_cache[start:stop], source=isend, tag=isend)
                    offset = stop
                elif self.comm.rank == isend:
                    self.comm.Send(my_map_cache, dest=0, tag=isend)
                    del my_map_cache
                self.comm.Barrier()
            map_offset_list = self.comm.gather(my_map_offsets)
            # Gather masks
            nrecv_mask = self.comm.gather(my_mask_cache.size)
            if self.comm.rank == 0:
                nrecv_tot = np.sum(nrecv_mask)
                mask_cache = np.empty(nrecv_tot, dtype=int)
            else:
                mask_cache = None
            offset = 0
            for isend in range(self.comm.size):
                if self.comm.rank == 0:
                    start = offset
                    stop = start + nrecv_mask[isend]
                    if isend == 0:
                        mask_cache[start:stop] = my_mask_cache
                    else:
                        self.comm.Recv(mask_cache[start:stop], source=isend, tag=isend)
                    offset = stop
                elif self.comm.rank == isend:
                    self.comm.Send(my_mask_cache, dest=0, tag=isend)
                    del my_mask_cache
                self.comm.Barrier()
            mask_offset_list = self.comm.gather(my_mask_offsets)
            self.comm.Barrier()
            if self.comm.rank == 0:
                print(self.prefix + f"Gathered maps in {time() - t2:.1f}s.")
        else:
            map_cache = my_map_cache
            mask_cache = my_mask_cache
            map_offset_list = [my_map_offsets]
            mask_offset_list = [my_mask_offsets]

        self.npix = npix

        if self.rank == 0:
            # Translate the offsets to the global array
            offset = 0
            for itask in range(1, self.ntask):
                offset += nrecv_map[itask - 1]
                map_offsets = map_offset_list[itask]
                for fname in map_offsets:
                    old = map_offsets[fname]
                    new = slice(old.start + offset, old.stop + offset)
                    my_map_offsets[fname] = new
            my_map_cache = map_cache
            offset = 0
            for itask in range(1, self.ntask):
                offset += nrecv_mask[itask - 1]
                mask_offsets = mask_offset_list[itask]
                for fname in mask_offsets:
                    old = mask_offsets[fname]
                    new = slice(old.start + offset, old.stop + offset)
                    my_mask_offsets[fname] = new
            my_mask_cache = mask_cache

            map_cache_size = my_map_cache.size
            mask_cache_size = my_mask_cache.size
        else:
            my_map_cache = None
            my_mask_cache = None
            map_cache_size = None
            mask_cache_size = None
            my_map_offsets = None
            my_mask_offsets = None

        if self.comm is not None:
            map_cache_size = self.comm.bcast(map_cache_size)
            mask_cache_size = self.comm.bcast(mask_cache_size)

        if self.comm is None:
            self.map_offsets = my_map_offsets
            self.mask_offsets = my_mask_offsets
        else:
            self.map_offsets = self.comm.bcast(my_map_offsets)
            self.mask_offsets = self.comm.bcast(my_mask_offsets)

        # Allocate and populate shared memory

        if self.shared:
            if self.rank == 0:
                gb = map_cache_size * 4 / 2**30
                print(self.prefix + f"Allocating {gb:.3f}GB to hold maps")
            self.map_cache = MPIShared((map_cache_size,), self.dtype, self.comm)
            if self.rank == 0:
                gb = self.map_cache.data.nbytes / 2**30
                print(self.prefix + f"Allocated {gb:.3f}GB to hold maps")
            self.map_cache.set(my_map_cache, fromrank=0)
        else:
            if self.rank == 0:
                gb = map_cache_size * 4 / 2**30 * self.ntask
                print(self.prefix + f"Allocating {gb:.3f}GB to hold maps")
            self.map_cache = self.comm.bcast(my_map_cache)
        del my_map_cache

        if self.shared:
            if self.rank == 0:
                gb = mask_cache_size * 8 / 2**30
                print(self.prefix + f"Allocating {gb:.3f}GB to hold indices")
            self.mask_cache = MPIShared((mask_cache_size,), np.int64, self.comm)
            if self.rank == 0:
                gb = self.mask_cache.data.nbytes / 2**30
                print(self.prefix + f"Allocated {gb:.3f}GB to hold indices")
            self.mask_cache.set(my_mask_cache, fromrank=0)
        else:
            if self.rank == 0:
                gb = mask_cache_size * 8 / 2**30 * self.ntask
                print(self.prefix + f"Allocating {gb:.3f}GB to hold indices")
            self.mask_cache = self.comm.bcast(my_mask_cache)
        del my_mask_cache

        if self.rank == 0:
            print(self.prefix + f"Loaded and cached all maps in {time() - t1:.1f}s")

        self.loaded = True

        return

    def to_dict(self):
        cache = {}
        for fname in self.fnames:
            map_ind = self.map_offsets[fname]
            mask_ind = self.mask_offsets[fname]
            m = self.map_cache[map_ind]
            good = self.mask_cache[mask_ind]
            ngood = good.size
            if ngood == 0:
                print(self.prefix + f"WARNING: {fname} is empty")
                cache[fname] = (m, good, self.npix)
            else:
                cache[fname] = (m.reshape(-1, ngood, copy=False), good, self.npix)
        return cache

    def save(self):
        t1 = time()
        if self.cachedir is None:
            # Do not save
            return

        if self.comm.rank == 0:
            os.makedirs(self.cachedir, exist_ok=True)

            fname_meta = os.path.join(self.cachedir, "meta.pck")
            fname_map = os.path.join(self.cachedir, "map_cache.npy")
            fname_mask = os.path.join(self.cachedir, "mask_cache.npy")
            fname_map_offsets = os.path.join(self.cachedir, "map_offsets.pck")
            fname_mask_offsets = os.path.join(self.cachedir, "mask_offsets.pck")
            
            print(f"Writing {fname_meta}")
            with open(fname_meta, "wb") as f:
                pickle.dump([self.npix, self.dtype], f)

            print(f"Writing {fname_map}")
            np.save(fname_map, self.map_cache)

            print(f"Writing {fname_map_offsets}")
            with open(fname_map_offsets, "wb") as f:
                pickle.dump(self.map_offsets, f)

            print(f"Writing {fname_mask}")
            np.save(fname_mask, self.mask_cache)

            print(f"Writing {fname_mask_offsets}")
            with open(fname_mask_offsets, "wb") as f:
                pickle.dump(self.mask_offsets, f)

            print(self.prefix + f"Saved cache in {time() - t1:.1f}s")

        return

    def load(self):
        t1 = time()
        if self.cachedir is None:
            # Do not save
            return

        if self.comm.rank == 0:
            fname_meta = os.path.join(self.cachedir, "meta.pck")
            fname_map = os.path.join(self.cachedir, "map_cache.npy")
            fname_mask = os.path.join(self.cachedir, "mask_cache.npy")
            fname_map_offsets = os.path.join(self.cachedir, "map_offsets.pck")
            fname_mask_offsets = os.path.join(self.cachedir, "mask_offsets.pck")
            not_found = False
            for fname in [
                    fname_meta,
                    fname_map,
                    fname_mask,
                    fname_map_offsets,
                    fname_mask_offsets,
            ]:
                if not os.path.isfile(fname):
                    print(f"File not found: {fname}")
                    not_found = True
            if not_found:
                print(f"No cached maps in {self.cachedir}")
            else:
                print(f"Loading cached maps from {self.cachedir}")
                with open(fname_meta, "rb") as f:
                    npix, dtype = pickle.load(f)
                map_cache = np.load(fname_map)
                with open(fname_map_offsets, "rb") as f:
                    map_offsets = pickle.load(f)
                mask_cache = np.load(fname_mask)
                with open(fname_mask_offsets, "rb") as f:
                    mask_offsets = pickle.load(f)
                map_cache_size = map_cache.size
                mask_cache_size = mask_cache.size
        else:
            not_found = None
            npix = None
            dtype = None
            map_cache_size = None
            mask_cache_size = None
            map_cache = None
            map_offsets = None
            mask_cache = None
            mask_offsets = None

        not_found = self.comm.bcast(not_found)
        if not_found:
            return

        self.npix = self.comm.bcast(npix)
        self.dtype = self.comm.bcast(dtype)
        map_cache_size = self.comm.bcast(map_cache_size)
        mask_cache_size = self.comm.bcast(mask_cache_size)
        self.map_offsets = self.comm.bcast(map_offsets)
        self.mask_offsets = self.comm.bcast(mask_offsets)
        self.fnames = sorted(list(self.map_offsets.keys()))

        # Allocate shared memory

        if self.shared:
            self.map_cache = MPIShared((map_cache_size,), self.dtype, self.comm)
            self.mask_cache = MPIShared((mask_cache_size,), np.int64, self.comm)

            if self.rank == 0:
                gb = self.map_cache.data.nbytes / 2**30
                print(self.prefix + f"Allocated {gb:.3f}GB to hold maps")
                gb = self.mask_cache.data.nbytes / 2**30
                print(self.prefix + f"Allocated {gb:.3f}GB to hold indices")

            # Populate the memory from rank 0

            self.map_cache.set(map_cache, fromrank=0)
            self.mask_cache.set(mask_cache, fromrank=0)
        else:
            if self.rank == 0:
                gb = map_cache_size * 4 / 2**30 * self.ntask
                print(self.prefix + f"Allocating {gb:.3f}GB to hold maps")
            self.map_cache = self.comm.bcast(map_cache)
            if self.rank == 0:
                gb = mask_cache_size * 8 / 2**30 * self.ntask
                print(self.prefix + f"Allocating {gb:.3f}GB to hold indices")
            self.mask_cache = self.comm.bcast(mask_cache)

        if self.rank == 0:
            print(self.prefix + f"Loaded cache in {time() - t1:.1f}s")

        self.loaded = True

        return
