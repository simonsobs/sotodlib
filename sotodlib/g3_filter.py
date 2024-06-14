# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Data Filtering

This module contains code for filtering data in G3Frames

"""
import numpy as np
import scipy.signal as signal

from spt3g import core
from .core.g3_core import DataG3Module

class Filter(DataG3Module):
    """
    G3Module that takes the G3Timestream map and applies generic filter

    Attributes:
        input (str): the key to a G3Timestream map of the source
        output (str): key of G3Timestream map of output data.
            if None, input will be overwritten with output
        filter_function (function): function that takes frequency in Hz and
            returns a frequency filter

    TODO:
        Get rid of numpy fft functions and get faster / better parallizable
        options.
    """

    def __init__(self, input='signal', output='signal_filtered', filter_function=None):
        if filter_function is None:
            raise ValueError('Missing Filter Definition')
        self.filter_function = filter_function
        super().__init__(input, output)

    def process(self, data, det_name):
        """
        Args:
            data (G3Timestream): data for a single detector
            det_name (str): the detector name in the focal plane
                in case it's needed for accessing calibration info
        Returns:
            np.array filtered by filter_function
        """
        freqs = np.fft.rfftfreq(data.n_samples, core.G3Units.Hz/data.sample_rate)
        return np.fft.irfft( np.fft.rfft(data)*self.filter_function(freqs) )

class LowPassButterworth(Filter):
    """
    G3Module for a LowPassButterworth filter

    Attributes:
            order (int): order of butterworth
            fc (float): cutoff frequency in Hertz
            gain (float): filter gain
    """
    def __init__(self, input='signal', output='signal_filtered',
                     order=2, fc=1, gain=1):
        self.order=2
        self.fc=fc
        self.gain=gain
        super().__init__(input, output, self.filter_function)

    def filter_function(self, freqs):
        b, a = signal.butter(self.order, 2*np.pi*self.fc, 'lowpass', analog=True)
        return self.gain*np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])
