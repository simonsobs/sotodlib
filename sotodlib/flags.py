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


def get_turnaround_flags_by_dazdt(
    aman, t_buffer=1.8 ,az=None, merge=True, overwrite=False, name="turnarounds_dazdt"
):

    """
    Args:
        aman: AxisManager object
        t_buffer: buffer length to avoid the flactuation of daz [sec]
        merge: If true, merge into tod.flags
        overwrite: If true and merge is True, write over existing flag
        name: name of flag when merged into tod.flags

    Returns:
        flag: Ranges object of turn-arounds
    """
    
    # Get daz and time 
    if az == None : 
        az = np.copy(aman.boresight.az)
    daz = np.diff(az)
    daz = np.append(daz,daz[-1])
    time = aman.timestamps - aman.timestamps[0]

    # derive approx daz
    approx_daz = np.mean(daz[daz>np.percentile(daz, 95)])

    # derive approx aprox daz/dt and rough number of samples in one_scan period 
    dt = np.mean(np.diff(aman.timestamps))
    approx_dazdt = approx_daz / dt
    approx_samps_onescan = int(np.ptp(az) / approx_dazdt / dt)

    # Make a step-function like matched filter
    kernel_size = 400
    kernel = np.zeros(kernel_size)
    # normalize with approx_daz so that peaks height be ~1
    kernel[:kernel_size//2] = 1/approx_daz
    kernel[kernel_size//2:] = - 1/approx_daz
    kernel /= kernel_size

    # convolve signal with the kernel
    matched = np.convolve(np.diff(aman.boresight.az), kernel, mode='same')
    matched[:kernel_size] = 0
    matched[-kernel_size:] = 0 

    # find peaks in matched daz, height=0.1, distance=approx_samps_onescan//3 would be robust enough
    peaks, properties = find_peaks(matched, height=0.1, distance=approx_samps_onescan//3)

    # turn around flag 
    turnaround_mask = np.zeros_like(aman.timestamps,dtype=bool)

    ### flagging
    
    mean_length = np.mean(np.diff(peaks))
    prev = 0    
    nbuffer = int(t_buffer/dt) # A finite buffer is needed to avoid the fluctuation of daz/dt

    # Check the beggining part 
    first_is_positive = False
    if peaks[0] < mean_length//2 : 
        first_is_positive = True

    # search for edges of square waves looking at the peaks
    for p in peaks : 
        start = prev
        end = p
        length = p-prev
        middle = start + length//2
        prev = p

        turnaround_mask[middle-nbuffer:middle+nbuffer] = True
        turnaround_mask[prev-nbuffer:prev+nbuffer] = True
        
    # we don't know the end of the last square wave, so use the mean of wave width
    start = prev
    end = prev + mean_length
    middle = start + mean_length//2
    
    turnaround_mask[int(middle-nbuffer):int(middle+nbuffer)] = True
    turnaround_mask[int(prev-nbuffer):int(prev+nbuffer)] = True    
    
    # treatment of the begining
    if first_is_positive : 
        turnaround_mask[:peaks[0]+nbuffer] = True

    else : 
        turnaround_mask[:int(peaks[0]-mean_length//2)] = True
        
    ta_flag = Ranges.from_bitmask(turnaround_mask)

    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError("Flag name {} already exists in aman.flags".format(name))
        elif name in aman.flags:
            aman.flags[name] = ta_flag
        else:
            aman.flags.wrap(name, ta_flag)
    return ta_flag
    

def get_leftright_flags_by_dazdt(
    aman, t_buffer=1.8, w_turnaround=True, az=None, merge=True, overwrite=False, name=["left", "right"]
):

    """
    Args:
        aman: AxisManager object
        t_buffer: buffer length to avoid the flactuation of daz [sec]
        merge: If true, merge into tod.flags
        overwrite: If true and merge is True, write over existing flag
        name: list of names of flags when merged into tod.flags

    Returns:
        flag: Ranges object of left/right scan
    """
    
    # Get daz and time 
    if az == None : 
        az = np.copy(aman.boresight.az)
    daz = np.diff(az)
    daz = np.append(daz,daz[-1])
    time = aman.timestamps - aman.timestamps[0]

    # derive approx daz
    approx_daz = np.mean(daz[daz>np.percentile(daz, 95)])

    # derive approx aprox daz/dt and rough number of samples in one_scan period 
    dt = np.mean(np.diff(aman.timestamps))
    approx_dazdt = approx_daz / dt
    approx_samps_onescan = int(np.ptp(az) / approx_dazdt / dt)

    # Make a step-function like matched filter
    kernel_size = 400
    kernel = np.zeros(kernel_size)
    # normalize with approx_daz so that peaks height be ~1
    kernel[:kernel_size//2] = 1/approx_daz
    kernel[kernel_size//2:] = - 1/approx_daz
    kernel /= kernel_size

    # convolve signal with the kernel
    matched = np.convolve(np.diff(aman.boresight.az), kernel, mode='same')
    matched[:kernel_size] = 0
    matched[-kernel_size:] = 0 

    # find peaks in matched daz, height=0.1, distance=approx_samps_onescan//3 would be robust enough
    peaks, properties = find_peaks(matched, height=0.1, distance=approx_samps_onescan//3)

    # flag that involves turnaround part 
    left_ent = (daz > 0.)
    right_ent = (daz <= 0.)
    
    # flag that does not involve turnaround part
    left = np.zeros_like(aman.timestamps,dtype=bool)
    right = np.zeros_like(aman.timestamps,dtype=bool)

    ### flagging
    
    mean_length = np.mean(np.diff(peaks))
    prev = 0    
    nbuffer = int(t_buffer/dt) # A finite buffer is needed to avoid the fluctuation of daz/dt

    # Check the beggining part 
    first_is_positive = False
    if peaks[0] < mean_length//2 : 
        first_is_positive = True

    # search for edges of square waves looking at the peaks
    for p in peaks : 
        start = prev
        end = p
        length = p-prev
        middle = start + length//2
        prev = p

        left[start+nbuffer:middle-nbuffer] = True
        right[middle+nbuffer:end-nbuffer] = True
        
    # we don't know the end of the last square wave, so use the mean of wave width
    start = prev
    end = prev + mean_length
    middle = start + mean_length//2
    
    left[int(start+nbuffer):int(middle-nbuffer)] = True
    right[int(middle+nbuffer):int(end-nbuffer)] = True
    
    # treatment of the begining
    if first_is_positive : 
        left[:peaks[0]+nbuffer] = False
        right[:peaks[0]+nbuffer] = False
    else : 
        right[int(peaks[0]-mean_length//2):peaks[0]-nbuffer] = True
        left[:peaks[0]+nbuffer] = False
        
    if w_turnaround : 
        left_flag = Ranges.from_bitmask(left_ent)
        right_flag = Ranges.from_bitmask(right_ent)        
    else : 
        left_flag = Ranges.from_bitmask(left)
        right_flag = Ranges.from_bitmask(right)
        
    if merge:
        if (name[0] in aman.flags or name[1] in aman.flags) and not overwrite:
            raise ValueError("Flag name {} already exists in aman.flags".format(name))
        else : 
            if name[0] in aman.flags:
                aman.flags[name[0]] = left_flag
            else:            
                aman.flags.wrap(name[0], left_flag)
            if name[1] in aman.flags:
                aman.flags[name[1]] = right_flag
            else:
                aman.flags.wrap(name[1], right_flag)

    return left_flag, right_flag
        


def get_turnaround_flags(
    tod, qlim=1, az=None, merge=True, overwrite=False, name="turnarounds"
):
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
    lo, hi = np.percentile(az, [qlim, 100 - qlim])
    m = np.logical_or(az < lo, az > hi)

    flag = Ranges.from_bitmask(m)

    if merge:
        if name in tod.flags and not overwrite:
            raise ValueError("Flag name {} already exists in tod.flags".format(name))
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
