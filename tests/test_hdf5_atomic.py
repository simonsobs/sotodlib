# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test atomic modification of HDF5 files
"""

import os

import unittest
from unittest import TestCase

import numpy as np
import h5py

from sotodlib.core.util import H5ContextManager

from ._helpers import create_outdir, mpi_multi

from .test_hdf5_atomic_update import main as inode_main


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class HDF5AtomicTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(fixture_name)

    def create_fake_dataset(self, grp, name):
        shape = (10, 100)
        rng = np.random.default_rng()
        arr = rng.uniform(low=-1.0, high=1.0, size=shape)
        ds = grp.create_dataset(name, shape=shape, dtype=np.float64, data=arr)
        return ds

    def create_fake_file(self, path):
        nds = 4
        with h5py.File(path, "w") as hf:
            for ids in range(nds):
                dname = f"{ids:02d}"
                _ = self.create_fake_dataset(hf, dname)
        return nds

    def append_dataset_file(self, hf):
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
        _ = self.create_fake_dataset(hf, dname)
        return nnew

    def check_file(self, hf, n_expected):
        nds = 0
        for ids, (dname, ds) in enumerate(hf.items()):
            dcheck = f"{ids:02d}"
            if dname != dcheck:
                return False
            nds += 1
        if nds != n_expected:
            return False
        return True

    def test_read_only(self):
        outpath = os.path.join(self.outdir, "test_read_only.h5")
        nds = self.create_fake_file(outpath)
        with H5ContextManager(outpath, "r") as hf:
            self.assertTrue(self.check_file(hf, nds))

    def test_append(self):
        outpath = os.path.join(self.outdir, "test_append.h5")
        nds = self.create_fake_file(outpath)
        with H5ContextManager(outpath, mode="a") as hf:
            nds = self.append_dataset_file(hf)
            self.assertTrue(self.check_file(hf, nds))

        # Append mode should also work if the file does not
        # yet exist
        os.remove(outpath)
        with H5ContextManager(outpath, mode="a") as hf:
            nds = self.append_dataset_file(hf)
            self.assertTrue(self.check_file(hf, nds))

    def test_write(self):
        outpath = os.path.join(self.outdir, "test_write.h5")
        nds = 4
        with H5ContextManager(outpath, mode="w") as hf:
            for ids in range(nds):
                _ = self.append_dataset_file(hf)
        with H5ContextManager(outpath, mode="r") as hf:
            self.assertTrue(self.check_file(hf, nds))

    def test_read_write(self):
        outpath = os.path.join(self.outdir, "test_read_write.h5")
        nds = self.create_fake_file(outpath)
        nappend = 2
        ntotal = nds + nappend
        with H5ContextManager(outpath, mode="r+") as hf:
            for ids in range(nappend):
                _ = self.append_dataset_file(hf)
        with H5ContextManager(outpath, mode="r") as hf:
            self.assertTrue(self.check_file(hf, ntotal))

    def test_inode_replace(self):
        # Test the case where a writing process has a file open for reading and then
        # a separate process opens the file for appending.  Since the append mode of
        # H5ContextManager makes a temp copy, the appended file is at a new inode, even
        # after it is renamed with the original name.  Meanwhile, the reading process
        # has an open handle to the original inode, which is unmodified and not
        # deleted until it is closed.
        opts = [
            "--test_dir",
            self.outdir,
        ]
        ret = inode_main(opts=opts)
        self.assertEqual(ret, 0)
