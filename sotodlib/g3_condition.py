# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Data Conditioning

This module contains code for conditioning G3Timestream data in G3Frames

"""
import numpy as np
import scipy.signal as signal

from spt3g import core
from .core.g3_core import DataG3Module

class MeanSubtract(DataG3Module):
    def process(self, data, det_name):
        return data-np.nanmean(data)

class MedianSubtract(DataG3Module):
    def process(self, data, det_name):
        return data-np.nanmedian(data)

class Detrend(DataG3Module):
    """
    Module for Detrending data. Information is added to the frame so that the
    resulting data can be easily re-trended (if, for example the detrend is done
    just for filtering).
    """

    def __init__(self, input='signal', output=None,
                        info='detrend_values', type='linear'):
        """
        Args:
            info (str): key for where the values will be saved
            type (str): can be 'constant' or 'linear'
        """
        self.type=type
        if self.type=='linear':
            self.deg=1
        elif self.type=='constant':
            self.deg=0
        else:
            raise ValueError("type must be 'linear' or 'constant'")
        self.info = info
        super().__init__(input, output)

    def __call__(self, f):
        if f.type == core.G3FrameType.Scan:
            self.detrend_vals = core.G3MapVectorDouble()

        super().__call__(f)

        if f.type == core.G3FrameType.Scan:
            f[self.info] = self.detrend_vals

    def process(self, data, det_name):
        x=np.arange(data.n_samples)
        self.detrend_vals[det_name] = np.polyfit(x, data, deg=self.deg)

        return data - np.polyval(self.detrend_vals[det_name], x)

class Retrend(DataG3Module):
    """
    Module for Retrending data that was Detrended with Detrend
    """
    def __init__(self, input='signal', output=None,
                        detrend_info='detrend_values'):
        """
        Args:
            info (str): key for where the values from detrending are saved
        """

        self.info = detrend_info
        super().__init__(input, output)

    def __call__(self, f):
        if f.type == core.G3FrameType.Scan:
            if self.info not in f.keys():
                raise ValueError('No Detrending information in {}'.format(self.info))
            else:
                self.retrend = f[self.info]

        super().__call__(f)

        if f.type == core.G3FrameType.Scan:
            f.pop(self.info)

    def process(self, data, det_name):
        x=np.arange(data.n_samples)
        return data + np.polyval(self.retrend[det_name], x)

class Decimate(DataG3Module):
    """
    Module for decimating data. Uses scipy.signal.decimate()
    """
    def __init__(self, input='signal', output=None, q=5, **kwargs):
        """
        Arguments:
            q (int): The downsampling factor
            kwargs: can include any of the optional parameters for
                scipy.signal.decimate
        """
        self.decimate_params = {'q': q, 'zero_phase': True}
        self.decimate_params.update(kwargs)

        super().__init__(input, output)

    def process(self, data, det_name):
        return signal.decimate(data, **self.decimate_params)

class Resample(DataG3Module):
    """
    Module for resampling data. Uses scipy.signal.resample()
    """
    def __init__(self, input='signal', output=None, num=3000, **kwargs):
        """
        Arguments:
            num (int): The number of samples in the resampled signal.
            kwargs: can include any of the optional parameters for
                scipy.signal.resample
        """
        self.resample_params = {'num':num}
        self.resample_params.update(kwargs)

        super().__init__(input, output)

    def process(self, data, det_name):
        return signal.resample(data, **self.resample_params)
