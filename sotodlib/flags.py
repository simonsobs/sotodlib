import numpy as np
import scipy.stats as stats

from so3g.proj import Ranges, RangesMatrix

from .tod_ops import filters
from .tod_ops import fourier_filter

def get_turnaround_flags(tod, qlim=1, az=None, merge=True, 
                         overwrite=False,name='turnarounds'):
    """Flag the scan turnaround times.

    Args:
        tod: AxisManager object
        qlim: percentile used to find turnaround
        az: The azimuth signal to look for turnarounds. If None it defaults to
            tod.boresight.az
        merge: If true, merge into tod.flags
        overwrite: If true and merge is True, write over existing flag
        name: name of flag when merged into tod.flags

    Returns:
        flag: Ranges object of turn-arounds 

    """
    if az is None:
        az = tod.boresight.az
    lo, hi = np.percentile(az, [qlim,100-qlim])
    m = np.logical_or(az < lo, az > hi)
    
    flag = Ranges.from_bitmask(m)
    
    if merge:
        if name in tod.flags and not overwrite:
            raise ValueError('Flag name {} already exists in tod.flags'.format(name))
        elif name in tod.flags:
            tod.flags[name] = flag
        else:
            tod.flags.wrap(name, flag)
    return flag


def get_glitch_flags(tod, t_glitch=0.002, hp_fc=0.5, n_sig=10, buffer=200, 
                     signal=None, merge=True, 
                     overwrite=False, name='glitches'):
    """ Find glitches with fourier filtering
    Translation from moby2 as starting point
    
    Args:
        tod (AxisManager): the tod 
        t_glitch (float): Gaussian filter width
        hp_fc: high pass filter cutoff
        n_sig (int or float): significance of detection
        buffer (int): amount to buffer flags around found location
        signal (str): if None, defaults to 'signal'
        merge (bool): if true, add to tod.flags
        name (string): name of flag to add to tod.flags
        overwrite (bool): if true, write over flag. if false, don't
    
    Returns:
        flag: RangesMatrix object of glitches
    """
    
    if signal is None:
        signal = 'signal'
    # f-space filtering
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(t_sigma=0.002)
    fvec = fourier_filter(tod, filt, detrend='linear', 
                          signal_name=signal, resize='zero_pad')
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    thres = 0.741 * stats.iqr(fvec, axis=1) * n_sig
    # get flags
    msk = fvec > thres[:,None]
    flag = RangesMatrix( [Ranges.from_bitmask(m) for m in msk])
    flag.buffer(buffer)
    
    if merge:
        if name in tod.flags and not overwrite:
            raise ValueError('Flag name {} already exists in tod.flags'.format(name))
        elif name in tod.flags:
            tod.flags[name] = flag
        else:
            tod.flags.wrap(name, flag)
        
    return flag

def get_trending_flags(aman, max_trend=5*np.pi, n_pieces=1, signal=None,
                 merge=True, overwrite=True, name='trends'):
    """ Flag Detectors with trends larger than max_trend.
    
    Args:
        aman (AxisManager): the tod 
        max_trend: maxmimum amount to always the detectors to change by. 
            The default is pi for use in phase units.
        n_pieces: number of pieces to cut the timestream in to to look for 
                    trends
        signal: Signal to use to generate flags, default is aman.signal.
        merge (bool): if true, merges the generated flag into aman
        overwrite (bool): if true, write over flag. if false, don't
        name (string): name of flag to add to aman.flags if merge is True
    
    Returns:
        flag: RangesMatrix object of glitches
    """
    if overwrite and name in aman.flags:
        aman.flags.move(name, None)
    
    if signal is None:
        signal = aman.signal
        
    signal = np.atleast_2d( signal )
    signal = signal[:,:aman.samps.count//n_pieces*n_pieces].reshape((signal.shape[0], n_pieces,-1))

    piece_size = signal.shape[2]
    slopes = signal[:,:,-1]-signal[:,:,0]

    bad_slopes = np.abs(slopes)>max_trend

    
    if n_pieces == 1:
        cut = bad_slopes[:,0]
    else:
        cut = aman.flags.get_zeros()
        dets = np.unique(np.where(bad_slopes)[0])
        for d in dets:
            clear = np.where(bad_slopes[d])[0]
            for c in clear:
                if c == n_pieces-1:
                    cut[d].add_interval(int(c*piece_size), cut.ranges[d].count-1)
                else:
                    cut[d].add_interval(int(c*piece_size), int((c+1)*piece_size))
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError('Flag name {} already exists in aman.flags'.format(name))
        elif name in aman.flags:
            aman.flags[name] = cut
        else:
            aman.flags.wrap(name, cut)
