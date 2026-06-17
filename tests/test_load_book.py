import unittest
import tempfile
import os

import numpy as np
from sotodlib import core
from sotodlib.io import load_book

# Import spt3g after so3g for backwards compat.
import so3g
from spt3g import core as spt3g_core


class TestLoadBook(unittest.TestCase):

    def test_100_load_file(self):
        ndet, nsamp = 100, 2000
        with tempfile.TemporaryDirectory() as tempdir:
            filenames = []
            filename = os.path.join(tempdir, 'test_%03i.g3')
            wi, w, wfc = 0, None, -1
            for f in load_book._sim_g3_generator(ndet, nsamp):
                if (wfc := (wfc + 1) % 5) == 0:
                    w = spt3g_core.G3Writer(filename % wi)
                    filenames.append(filename % wi)
                    wi += 1
                w.Process(f)
            del w

            assert len(filenames) > 1

            # Load it all
            aman0 = load_book.load_book_file(filenames)
            assert aman0.dets.count == ndet
            assert aman0.samps.count == nsamp

            # Load sample subset
            sslice = slice(123, 1435)
            aman1 = load_book.load_book_file(
                filenames, samples=(sslice.start, sslice.stop))
            assert (aman1.signal  == aman0.signal[:, sslice]).all()

            # Load det subset
            sub_idx = [12, 43, 35, 98]
            sub_dets = [aman0.dets.vals[i] for i in sub_idx]
            aman1 = load_book.load_book_file(
                filenames, dets=sub_dets)
            for i, j in enumerate(sub_idx):
                assert (aman1.signal[i] == aman0.signal[j]).all()

            # Load both subset
            aman1 = load_book.load_book_file(
                filenames, samples=(sslice.start, sslice.stop), dets=sub_dets)
            for i, j in enumerate(sub_idx):
                assert (aman1.signal[i] == aman0.signal[j, sslice]).all()
