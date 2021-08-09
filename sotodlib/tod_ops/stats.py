"""
Useful tools for statistical analyses
"""
from scipy.stats import skewnorm, binned_statistic
import numpy as np
from .. import core

def gauss(x, mean, sigma, A):
    """
    Gaussian curve defined mean, sigma, and scaled by an amplitude A.
    Note that maximum value!= A; rather the area under the curve == A.
    
    Arguments:
    
        x: value or array of points
        
        mean: mean of distribution
            
        sigma: sd of distribution

        A: amplitude by which to scale distribution
            
    Returns:
    
		y: The value(s) of the curve at x
    """
    return A * np.exp(-((x-mean)/sigma)**2/2) / np.sqrt(2*np.pi*sigma**2)      

def skewgauss(x, mean, sigma, A, sk):
    """
    Skewgaussian/skewnormal curve defined mean, sigma, and skew parameter, and scaled by an amplitude A.
    Note that maximum value!= A; rather the area under the curve == A.
    
    Arguments:
    
        x: value or array of points
        
        mean: mean of distribution
            
        sigma: sd of distribution

        A: amplitude by which to scale distribution

        sk: skew parameter.  When sk is 0, this becomes a normal distribution.
            
    Returns:
    
		y: The value(s) of the curve at x
    """
    return A * skewnorm.pdf((x-mean)/sigma, sk)


def average_to(aman, DT, signal=None, timestamps=None, tmin=None, tmax=None, append_aman=False):
    """
	Bins and averages input signal and timestamps to a new (regular) sample rate.
	This is most useful for downsampling (paricularly non-regularly sapmled) data.
    
    Arguments:
    
        aman: AxisManager with timestamps and data to resample.
        
        DT: Time separation of new sample rate (in units of input timestamps); this is 1/sample-frequency.
            
        signal: 1- or 2-d array of data to resample. If None, uses aman.signal.

        timestamps: 1-d array of timestamps. If None, uses aman.timestamps.

        tmin: desired start time for resampled data. If None, uses beginning of timestamps.

        tmax: desired stop time for resampled data. If None, uses end of timestamps.

        append_aman: return AxisManager with resampled time and data appended to the provided aman. If aman=None this returns None.
            
    Returns:
    
		t: resampled timestamps

		d: resampled data -- will have nans in bins where there was no input data
		
                aman: aman containing resampled timestamps and data. The new sample axis will be called avgDT_samps, the resampled timestamps will be called avgDT_timestamps, and the resampled signal will be called avgDT_signal (where DT is the provided time seperation of the new sample rate). This is only returned if append_aman is True.
    """
    if signal is None:
        signal = aman.signal
    if timestamps is None:
        timestamps = aman.timestamps

    if tmin==None:
        tmin = timestamps[0]
    if tmax==None:
        tmax = timestamps[-1]

    bins = np.arange(tmin-DT/2,tmax+DT/2,DT)
    d, bins, _ = binned_statistic(timestamps, signal, bins=bins)
    t = (bins[1:]+bins[:-1])/2

    if append_aman:
        if aman is None:
            return t, d, None
        aman = core.AxisManager(aman, core.OffsetAxis(f'avg{DT}_samps', len(t)))

        aman.wrap(f'avg{DT}_timestamps', t, [(0, f'avg{DT}_samps')])
        aman.wrap(f'avg{DT}_signal', d, [(0, 'dets'), (1, f'avg{DT}_samps')])
        return t, d, aman

    return t, d
