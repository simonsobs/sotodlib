# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test Data Processing Modules
"""

import unittest
from unittest import TestCase

import numpy as np
from spt3g import core

from sotodlib.core.g3_core import DataG3Module
from sotodlib.g3_filter import Filter, LowPassButterworth
from sotodlib.g3_condition import (Detrend, Retrend, MeanSubtract,
                                         MedianSubtract, Decimate, Resample)

import sotodlib.g3_sim as data_sim

from ._helpers import mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class DataTest(TestCase):

    def setUp(self):
        self.frames = data_sim.noise_scan_frames(input='signal')

    def test_DataG3Module(self):
        ### Test that it works in a pipeline
        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(DataG3Module, input='signal', output=None)
        p.Run()

        ### test that it works on individual frames
        x = DataG3Module(input='signal', output=None)
        x.apply(self.frames[0])

        ### Test that it works as an inline function
        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(DataG3Module.from_function(lambda d:d+1, input='signal', output=None))
        p.Run()

    def test_filters(self):
        ### Test general filter
        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(Filter, input='signal', output=None,
                 filter_function=lambda freqs:np.ones_like(freqs))
        p.Run()

        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(LowPassButterworth)
        p.Run()

    def test_condition(self):
        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(Detrend,input='signal', output=None)
        p.Add(Retrend,input='signal', output=None)
        p.Run()

        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(MeanSubtract,input='signal', output=None)
        p.Add(MedianSubtract,input='signal', output=None)
        p.Run()

        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(Decimate,input='signal', output=None)
        p.Run()

        p = core.G3Pipeline()
        p.Add(data_sim.PipelineSeeder(self.frames))
        p.Add(Resample,input='signal', output=None)
        p.Run()



if __name__ == '__main__':
    unittest.main()
