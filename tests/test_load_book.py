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
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, 'test.g3')
            w = spt3g_core.G3Writer(filename)
            for f in load_book._sim_g3_generator(100, 1000):
                w.Process(f)
            del w

            aman0 = load_book.load_book_file(filename)
            
