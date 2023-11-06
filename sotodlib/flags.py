import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from so3g.proj import Ranges, RangesMatrix

from . import core
from .tod_ops import filters
from .tod_ops import fourier_filter


def get_turnaround_flags(
    aman, t_buffer=1.8 ,az=None, merge=True, merge_lr=True, overwrite=False, 
    name="turnarounds", method="scanspeed", kernel_size=400, num_min=10, qlim=1
):

    """
    Args:
        aman: AxisManager object
        t_buffer: buffer length to avoid the flactuation of daz [sec]
        az: The azimuth signal to look for turnarounds. If None, it defaults to
            tod.boresight.az
        merge: If true, merge turn-around flag into aman.flags
        merge_lr: If true, merge left/right scan flag into aman.flags
        overwrite: If true and merge is True, write over existing flag
        name: name of flag when merged into tod.flags
        method: If "az", flag is calculated looking at az itself.
                If "scanspeed", flag is calculated looking at daz/dt.
        kernel_size: size of kernel in square wave search
        num_min: if flag has shorter range than num_min (in index unit) because of 
                 daz/dt flactuation, that part will be removed.

    Returns:
        flag: Ranges object of turn-arounds
    """
    
    # Get daz
    if az == None : 
        az = np.copy(aman.boresight.az)
             
    daz = np.diff(az)
    daz = np.append(daz,daz[-1])

    
    # flag of turn-around mask      
    assert method=="az" or method=="scanspeed", "Invalid argument 'method'."

    # original method 
    if method=="az" : 
        lo, hi = np.percentile(az, [qlim, 100 - qlim])
        m = np.logical_or(az < lo, az > hi)

        ta_flag = Ranges.from_bitmask(m)
    # daz/dt
    else : 
        # derive approx daz
        approx_daz = np.mean(daz[daz>np.percentile(daz, 95)])

        # derive approx aprox daz/dt and rough number of samples in one_scan period 
        dt = np.mean(np.diff(aman.timestamps))
        approx_dazdt = approx_daz / dt
        approx_samps_onescan = int(np.ptp(az) / approx_dazdt / dt)

        # Make a step-function like matched filter
        kernel = np.zeros(kernel_size)
        # normalize with approx_daz so that peaks height be ~1
        kernel[:kernel_size//2] = 1/approx_daz
        kernel[kernel_size//2:] = - 1/approx_daz
        kernel /= kernel_size

        # convolve signal with the kernel
        matched = np.convolve(np.diff(aman.boresight.az), kernel, mode='same')

        # find peaks in matched daz, height=0.1, distance=approx_samps_onescan//3 would be robust enough
        peaks_pos, properties_pos = find_peaks(matched, height=0.1, distance=approx_samps_onescan//3)
        peaks_neg, properties_neg = find_peaks(-matched, height=0.1, distance=approx_samps_onescan//3)
        peaks = np.sort(np.append(peaks_pos,peaks_neg))
        mean_length = np.mean(np.diff(peaks[1:-1]))
        

            
        # Is first positive or negative?
        is_pos = np.zeros(len(peaks),dtype=bool)
        if peaks_pos[0] < peaks_neg[0] : is_pos[0::2] = True
        else : is_pos[1::2] = True
        
        # fill flags 
        turnaround_mask = np.zeros_like(aman.timestamps,dtype=bool)
        _left_flag = np.zeros_like(aman.timestamps,dtype=bool)
        _right_flag = np.zeros_like(aman.timestamps,dtype=bool) 
        nbuffer = int(t_buffer/dt) # A finite buffer is needed to avoid the fluctuation of daz/dt

        for ip,p in enumerate(peaks) : 
            turnaround_mask[p-nbuffer:p+nbuffer] = True
            turnaround_mask[p-nbuffer:p+nbuffer] = True
            
            if ip < len(peaks)-1 : 
                if is_pos[ip] : _right_flag[peaks[ip]:peaks[ip+1]] = True
                else : _left_flag[peaks[ip]:peaks[ip+1]] = True
            elif ip == len(peaks)-1:
                if is_pos[ip] : _right_flag[peaks[ip]:int(peaks[ip]+mean_length)] = True
                else : _left_flag[peaks[ip]:int(peaks[ip]+mean_length)] = True                


        # treatment of begining and ending parts 
        turnaround_mask[:peaks[0]] = True
        turnaround_mask[peaks[-1]+int(mean_length):] = True   
        
        # flag if the end of the observation is stationary
        is_daz_zero = (np.abs(daz) < 0.00005)
        last_moving_index = np.max(np.where(is_daz_zero==False)[0]) 
        if len(daz) - last_moving_index > nbuffer : 
            turnaround_mask[last_moving_index:] = True
            _left_flag[last_moving_index:] = False
            _right_flag[last_moving_index:] = False
            
        ta_flag = Ranges.from_bitmask(turnaround_mask)
        left_flag = Ranges.from_bitmask(_left_flag)
        right_flag = Ranges.from_bitmask(_right_flag)
    
    
    # merge turn-around
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError("Flag name {} already exists in aman.flags".format(name))
        elif name in aman.flags:
            aman.flags[name] = ta_flag
        else:
            aman.flags.wrap(name, ta_flag)
            
    # merge left/right mask 
    if merge_lr:
        if ("left_scan" in aman.flags or "right_scan" in aman.flags ) and not overwrite:
            raise ValueError("Flag name left/right_flag already exists in aman.flags")
        else : 
            if "left_scan" in aman.flags:
                aman.flags["left_scan"] = left_flag
            else :
                aman.flags.wrap("left_scan", left_flag)
                
            if "right_scan" in aman.flags:
                aman.flags["right_scan"] = right_flag
            else :
                aman.flags.wrap("right_scan", right_flag)

    return ta_flag
    

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
    name="glitches",
    full_output=False,
):
    """Find glitches with fourier filtering
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
        signal = "signal"
    # f-space filtering
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(
        t_sigma=t_glitch
    )
    fvec = fourier_filter(
        aman, filt, detrend=detrend, signal_name=signal, resize="zero_pad"
    )
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    if fvec.shape[1] > 50000:
        ds = int(fvec.shape[1]/20000)
    else: 
        ds = 1
    iqr_range = 0.741 * stats.iqr(fvec[:,::ds], axis=1)
    # get flags
    msk = fvec > iqr_range[:, None] * n_sig
    flag = RangesMatrix([Ranges.from_bitmask(m) for m in msk])
    flag.buffer(buffer)

    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError("Flag name {} already exists in tod.flags".format(name))
        elif name in aman.flags:
            aman.flags[name] = flag
        else:
            aman.flags.wrap(name, flag)

    if full_output:
        indptr = np.append(
            0, np.cumsum([np.sum(msk[i]) for i in range(aman.dets.count)])
        )
        indices = np.concatenate([np.where(msk[i])[0] for i in range(aman.dets.count)])
        data = np.concatenate(
            [fvec[i][msk[i]] / iqr_range[i] for i in range(aman.dets.count)]
        )
        smat = csr_array(
            (data, indices, indptr), shape=(aman.dets.count, aman.samps.count)
        )
        glitches = core.AxisManager(
            aman.dets,
            aman.samps,
        )
        glitches.wrap("glitch_flags", flag, [(0, "dets"), (1, "samps")])
        glitches.wrap("glitch_detection", smat, [(0, "dets"), (1, "samps")])
        return flag, glitches

    return flag


def get_trending_flags(
    aman,
    max_trend=1.2,
    n_pieces=1,
    max_samples=500,
    signal=None,
    timestamps=None,
    merge=True,
    overwrite=True,
    name="trends",
    full_output=False,
):
    """
    Flag Detectors with trends larger than max_trend.
    This function can be used to find unlocked detectors.
    Note that this is a rough cut and unflagged detectors can still have poor tracking.

    Args:
        aman (AxisManager): the tod
        max_trend: Slope at which detectors are unlocked.
                   The default is for use with phase units.
        n_pieces: number of pieces to cut the timestream in to to look for trends.
        max_samples: Maximum samples to compute the slope with.
        signal: Signal to use to generate flags, default is aman.signal.
        timestamps: Timestamps to use to generate flags, default is aman.timestamps.
        merge (bool): if true, merges the generated flag into aman
        overwrite (bool): if true, write over flag. if false, don't
        name (string): name of flag to add to aman.flags if merge is True
        full_output(bool): if true, returns calculated slope sizes

    Returns:
        cut: RangesMatrix of trending regions
        trends: if full_output is true, calculated slopes and the
        sample edges where they were calculated.
    """
    if 'flags' not in aman:
        overwrite = False
        merge = False
    if overwrite and name in aman.flags:
        aman.flags.move(name, None)

    if signal is None:
        signal = aman.signal
    signal = np.atleast_2d(signal)
    if timestamps is None:
        timestamps = aman.timestamps
    assert len(timestamps) == signal.shape[1]

    slopes = np.zeros((len(signal), 0))
    cut = np.zeros((len(signal), 0), dtype=bool)
    samp_edges = [0]
    for t, s in zip(
        np.array_split(timestamps, n_pieces), np.array_split(signal, n_pieces, 1)
    ):
        samps = len(t)
        # Cheap downsampling
        if len(t) > max_samples:
            n = len(t) // max_samples
            t = t[::n]
            s = s[:, ::n]
        _slopes = ((t * s).mean(axis=1) - t.mean() * s.mean(axis=1)) / (
            (t**2).mean() - (t.mean()) ** 2
        )
        cut = np.hstack(
            (cut, np.tile((np.abs(_slopes) > max_trend)[..., np.newaxis], samps))
        )
        if full_output:
            slopes = np.hstack((slopes, _slopes[..., np.newaxis]))
            samp_edges.append(samp_edges[-1] + samps)
    cut = RangesMatrix.from_mask(cut)

    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError("Flag name {} already exists in aman.flags".format(name))
        if name in aman.flags:
            aman.flags[name] = cut
        else:
            aman.flags.wrap(name, cut)

    if full_output:
        samp_edges = np.array(samp_edges)
        trends = core.AxisManager(
            aman.dets,
            core.OffsetAxis("samps", len(timestamps)),
            core.OffsetAxis("trend_bins", len(samp_edges) - 1),
        )
        trends.wrap("samp_start", samp_edges[:-1], [(0, "trend_bins")])
        trends.wrap("trends", slopes, [(0, "dets"), (1, "trend_bins")])
        trends.wrap("trend_flags", cut, [(0, "dets"), (1, "samps")])
        return cut, trends

    return cut
