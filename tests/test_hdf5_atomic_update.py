#!/usr/bin/env python3
# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test atomic modification of HDF5 files."""

import argparse
import os
import multiprocessing as mp
import shutil

import numpy as np
import h5py

from sotodlib.core.util import H5ContextManager


def create_fake_dataset(grp, name):
    shape = (10, 100)
    rng = np.random.default_rng()
    arr = rng.uniform(low=-1.0, high=1.0, size=shape)
    ds = grp.create_dataset(name, shape=shape, dtype=np.float64, data=arr)
    return ds


def create_fake_file(path):
    nds = 4
    with h5py.File(path, "w") as hf:
        for ids in range(nds):
            dname = f"{ids:02d}"
            _ = create_fake_dataset(hf, dname)
    return nds


def append_dataset_file(hf):
    if len(hf.keys()) == 0:
        # First dataset
        dname = "00"
        nnew = 1
    else:
        dlast = list(hf.keys())[-1]
        idlast = int(dlast)
        idnext = idlast + 1
        nnew = idnext + 1
        dname = f"{idnext:02d}"
    _ = create_fake_dataset(hf, dname)
    return nnew


def check_file(hf, n_expected):
    nds = 0
    for ids, (dname, ds) in enumerate(hf.items()):
        dcheck = f"{ids:02d}"
        if dname != dcheck:
            return False
        nds += 1
    if nds != n_expected:
        return False
    return True


def modifier_func(path, n_append):
    # Open the file in append mode and add some datasets
    with H5ContextManager(path, mode="a") as hf:
        for iapp in range(n_append):
            _ = append_dataset_file(hf)


def main(opts=None):
    parser = argparse.ArgumentParser(description="Test HDF5 atomic updates")

    parser.add_argument(
        "--test_dir",
        required=False,
        type=str,
        default="test_hdf5_atomic",
        help="The testing directory",
    )

    parser.add_argument(
        "--n_append",
        required=False,
        type=int,
        default=4,
        help="The number of datasets to append on each worker",
    )

    args = parser.parse_args(args=opts)

    outpath = os.path.join(args.test_dir, "test_inode_replace.h5")

    if os.path.isdir(args.test_dir):
        shutil.rmtree(args.test_dir)
    os.makedirs(args.test_dir)

    # Spawn a new process, not fork
    ctx = mp.get_context("spawn")

    # Create initial file
    nds = create_fake_file(outpath)

    # Reader opens file.  Test the direct method here instead of a context.
    hf = H5ContextManager(outpath, mode="r").open()

    # File should have original number of datasets
    result = check_file(hf, nds)
    if not result:
        msg = f"Failed to create initial file with {nds} datasets"
        print(msg, flush=True)
        return 1

    # Imagine the reader process is an rsync, and the open file is a large
    # HDF5 metadata file.  Meanwhile the file might be opened multiple times
    # by writing processes that append little datasets.  Try to capture
    # that here.

    # While file is open for read, append to file from separate process
    writer = ctx.Process(
        target=modifier_func,
        args=(outpath, args.n_append),
    )
    writer.start()
    writer.join()

    # And again...
    writer = ctx.Process(
        target=modifier_func,
        args=(outpath, args.n_append),
    )
    writer.start()
    writer.join()

    # And again...
    writer = ctx.Process(
        target=modifier_func,
        args=(outpath, args.n_append),
    )
    writer.start()
    writer.join()

    # Check that our open file handle is still the same data
    result = check_file(hf, nds)
    if not result:
        msg = f"Failed to preserve original file contents with {nds} datasets"
        print(msg, flush=True)
        return 1

    # The classmethod to close the handle will release the inode containing
    # the original file version, whose name has since been redirected (3 times)
    # to different inodes.
    H5ContextManager.close(hf)

    # Starting process re-opens the file and now the contents should be updated
    new_nds = nds + 3 * args.n_append
    with H5ContextManager(outpath, mode="r") as hf:
        result = check_file(hf, new_nds)
    if not result:
        msg = f"Failed to load updated file with {new_nds} datasets"
        print(msg, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    main()
