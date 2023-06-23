import numpy as np
import scipy.stats as stats

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from so3g.proj import Ranges, RangesMatrix

from . import core
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

def get_glitch_flags(
    aman, 
    t_glitch=0.002, 
    hp_fc=0.5, 
    n_sig=10, 
    buffer=200,
    detrend=None,
    signal=None, 
    merge=True,
    overwrite=False, 
    name='glitches',
    full_output=False
):
    """ Find glitches with fourier filtering
    Translation from moby2 as starting point

    Args:
        aman (AxisManager): the tod
        t_glitch (float): Gaussian filter width
        hp_fc: high pass filter cutoff
        n_sig (int or float): significance of detection
        buffer (int): amount to buffer flags around found location
        detrend (str): detrend method to pass to fourier_filter
        signal (str): if None, defaults to 'signal'
        merge (bool): if true, add to aman.flags
        name (string): name of flag to add to aman.flags
        overwrite (bool): if true, write over flag. if false, raise ValueError
            if name already exists in AxisManager
        full_output (bool): if true, return sparse matrix with the significance of
            the detected glitches

    Returns:
        flag: RangesMatrix object of glitches
    """

    if signal is None:
        signal = 'signal'
    # f-space filtering
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(t_sigma=t_glitch)
    fvec = fourier_filter(aman, filt, detrend=detrend,
                          signal_name=signal, resize='zero_pad')
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    iqr_range = 0.741 * stats.iqr(fvec, axis=1)
    # get flags
    msk = fvec > iqr_range[:,None]*n_sig
    flag = RangesMatrix( [Ranges.from_bitmask(m) for m in msk])
    flag.buffer(buffer)

    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError('Flag name {} already exists in tod.flags'.format(name))
        elif name in aman.flags:
            aman.flags[name] = flag
        else:
            aman.flags.wrap(name, flag)

    if full_output:
        indptr = np.append( 0,
                           np.cumsum( [np.sum(msk[i])
                                       for i in range(aman.dets.count)]))
        indices = np.concatenate( [np.where(msk[i])[0]
                                   for i in range(aman.dets.count) ])
        data = np.concatenate( [ fvec[i][msk[i]]/iqr_range[i]
                                for i in range(aman.dets.count)  ])
        smat = csr_array( (data, indices, indptr),
                         shape=( aman.dets.count, aman.samps.count))
        glitches = core.AxisManager( 
            aman.dets, 
            aman.samps,
        )
        glitches.wrap("glitch_flags", flag, [(0,"dets"),(1,"samps")])
        glitches.wrap("glitch_detection", smat, [(0, 'dets'), (1,'samps')])
        return flag, glitches

    return flag

def get_trending_flags(aman, max_trend=5*np.pi, n_pieces=1, signal=None,
                     merge=True, overwrite=True, name='trends',
                     full_output=False):
    """ Flag Detectors with trends larger than max_trend. Note, this is really
    a max-min calculator.

    Args:
        aman (AxisManager): the tod
        max_trend: maxmimum amount to always the detectors to change by.
            The default is for use with phase units.
        n_pieces: number of pieces to cut the timestream in to to look for
                    trends
        signal: Signal to use to generate flags, default is aman.signal.
        merge (bool): if true, merges the generated flag into aman
        overwrite (bool): if true, write over flag. if false, don't
        name (string): name of flag to add to aman.flags if merge is True
        full_output(bool): if true, returns calculated slope sizes
        
    Returns:
        flag: RangesMatrix object of glitches
        trends: if full_output is true, calculated slopes and the
        sample edges where they were calculated. 
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
                    cut[d].add_interval(int(c*piece_size), cut.ranges[d].count)
                else:
                    cut[d].add_interval(int(c*piece_size), int((c+1)*piece_size))
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError('Flag name {} already exists in aman.flags'.format(name))
        elif name in aman.flags:
            aman.flags[name] = cut
        else:
            aman.flags.wrap(name, cut)
    
    if full_output:
        samp_edges = np.linspace(0, piece_size*n_pieces, n_pieces+1)
        trends = core.AxisManager( 
            aman.dets, 
            aman.samps,
            core.OffsetAxis("trend_bins", len(samp_edges)-1),
        )
        trends.wrap("samp_start", samp_edges[:-1], [(0,"trend_bins")])
        trends.wrap("trends", slopes, [(0,"dets"),(1,"trend_bins")])
        trends.wrap("trend_flags", cut, [(0,"dets"),(1,"samps")])
        return cut, trends
    
    return cut
