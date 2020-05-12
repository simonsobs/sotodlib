import numpy as np
import scipy.stats as stats

from so3g.proj import Ranges, RangesMatrix

from .tod_ops import filters
from .tod_ops import fourier_filter

def get_turnaround_flags(tod, qlim=1, merge=True, name='turnarounds'):
    """Flag the scan turnaround times.

    Parameters:
    -----------
    tod: AxisManager object
    qlim: percentile used to find turnaround
    merge: If true, merge into tod.flags
    name: name of flag when merged into tod.flags

    Returns:
    --------
    Ranges object

    """
    az = tod.boresight.az
    lo, hi = np.percentile(az, [qlim,100-qlim])
    m = np.logical_or(az < lo, az > hi)
    
    flag = Ranges.from_bitmask(m)
    if merge:
        tod.flags.wrap(name, flag)
    return flag 


def get_glitch_flags(tod, n_sig, t_glitch, hp_fc, buffer,
                    merge=True, name='glitchs'):
    """Direct translation from moby2 for quick testing"""
    # f-space filtering
    filt = filters.high_pass_sine2(hp_fc) * filters.gaussian_filter(t_glitch)
    fvec = fourier_filter(tod, filt, detrend='linear', resize='zero_pad')
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    thres = 0.741 * stats.iqr(fvec, axis=1) * n_sig
    # get flags
    msk = fvec > thres[:,None]
    flag = RangesMatrix( [Ranges.from_bitmask(m) for m in msk])
    flag.buffer(buffer)
    
    if merge:
        tod.flags.wrap(name, flag)
    return flag