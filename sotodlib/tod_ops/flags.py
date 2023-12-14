import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from so3g.proj import Ranges, RangesMatrix

from .. import core
from . import filters
from . import fourier_filter


def get_det_bias_flags(aman, detcal=None, rfrac_range=(0.1, 0.7),
                      psat_range=(0, 15), merge=True, overwrite=True,
                      name='det_bias_flags'):
    """
    Function for selecting detectors in appropriate bias range.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    detcal : AxisManager
        AxisManager containing detector calibration information
        from bias steps and IVs. If None defaults to aman.det_cal.
    rfrac_range : Tuple
        Tuple (lower_bound, upper_bound) for rfrac det selection.
    psat_range : Tuple
        Tuple (lower_bound, upper_bound) for P_SAT from IV analysis.
        P_SAT in the IV analysis is the bias power at 90% Rn in pW.
    merge : bool
        If true, merges the generated flag into aman.
    overwrite : bool
        If true, write over flag. If false, don't.
    name : str
        Name of flag to add to aman.flags if merge is True.

    Returns
    -------
    mask : RangesMatrix
        RangesMatrix shaped N_dets x N_samps that is True is the detector
        is flagged to be cut and false if it should be kept based on 
        the rfrac, and psat ranges. To create a boolean mask from
        the RangesMatrix that can be used for aman.restrict() use
        ``keep = ~has_all_cut(mask)`` and then restrict with 
        ``aman.restrict('dets', aman.dets.vals[keep])``.
    """
    if detcal is None:
        if 'det_cal' not in aman:
            raise ValueError("AxisManager missing required 'det_cal' field " 
                             "with detector calibration information")
        detcal = aman.det_cal

    if 'flags' not in aman:
        overwrite = False
        merge = False
    if overwrite and name in aman.flags:
        aman.flags.move(name, None)
    
    msk = np.all([detcal.bg >= 0,
                  detcal.r_tes > 0,
                  detcal.r_frac >= rfrac_range[0],
                  detcal.r_frac <= rfrac_range[1],
                  detcal.p_sat*1e12 >= psat_range[0],
                  detcal.p_sat*1e12 <= psat_range[1]], axis=0)
    # Expand mask to ndets x nsamps RangesMatrix
    x = Ranges(aman.samps.count)
    mskexp = RangesMatrix([Ranges.ones_like(x) if Y
                           else Ranges.zeros_like(x) for Y in msk])
    
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {name} already exists in aman.flags")
        if name in aman.flags:
            aman.flags[name] = mskexp
        else:
            aman.flags.wrap(name, mskexp, [(0, 'dets'), (1, 'samps')])
    
    return mskexp

def get_turnaround_flags(aman, az=None, method='scanspeed', name='turnarounds',
                         merge=True, merge_lr=True, overwrite=True, 
                         t_buffer=2., kernel_size=400, peak_threshold=0.1, rel_distance_peaks=0.3,
                         truncate=False, qlim=1):
    """
    Compute turnaround flags for a dataset.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    az : Array
        (Optional). Azimuth data for turnaround flag computation. If not provided, it uses ``aman.boresight.az.``
    method : str
        (Optional). The method for computing turnaround flags. Options are ``az`` or ``scanspeed``.
    name : str
        (Optional). The name of the turnaround flag in ``aman.flags``. Default is ``turnarounds``
    merge : bool
        (Optional). Merge the computed turnaround flags into ``aman.flags`` if ``True``.
    merge_lr : bool
        (Optional). Merge left and right scan flags as ``aman.flags.left_scan`` and ``aman.flags.right_scan`` if ``True``.
    overwrite : bool
        (Optional). Overwrite an existing flag in ``aman.flags`` with the same name.
    t_buffer : float
        (Optional). Buffer time (in seconds) for flagging turnarounds in ``scanspeed`` method.
    kernel_size : int
        (Optional). Size of the step-wise matched filter kernel used in ``scanspeed`` method.
    peak_threshold : float
        (Optional). Peak threshold for identifying peaks in the matched filter response.
        It is a value used to determine the minimum peak height in the signal.
    rel_distance_peaks : float
        (Optional). Relative distance between peaks.
        It specifies the minimum distance between peaks as a fraction of the approximate number of samples in one scan period.
    truncate : bool
        (Optional). Truncate unstable scan segments if True in ``scanspeed`` method.
    qlim : float
        (Optional). Azimuth threshold percentile for ``az`` method turnaround detection.

    Returns
    -------
    Ranges : RangesMatrix
        The turnaround flags as a Ranges object.
    """
    if az is None : 
        az = aman.boresight.az
        
    if method not in ['az', 'scanspeed']:
        raise ValueError('Unsupported method. Supported methods are `az` or `scanspeed`')
    
    # `az` method: flag turnarounds based on azimuth threshold specifled by qlim
    elif method=='az':
        lo, hi = np.percentile(az, [qlim, 100 - qlim])
        m = np.logical_or(az < lo, az > hi)
        ta_flag = Ranges.from_bitmask(m)
    
    # `scanspeed` method: flag turnarounds based on scanspeed.
    elif method=='scanspeed':
        daz = np.diff(az)
        daz = np.append(daz,daz[-1])
        approx_daz = np.median(daz[daz>np.percentile(daz, 95)])
        
        # derive approximate number of samples in one_scan period
        approx_samps_onescan = int(np.ptp(az) / approx_daz)
        # update approx_samps_onescan with detrending
        x = np.linspace(0, 1, az.shape[0])
        slope = az[-approx_samps_onescan:].mean() - az[:approx_samps_onescan].mean()
        approx_samps_onescan = int(np.ptp(az - slope*x) / approx_daz)

        # Make a step-function like matched filter. Kernel is normarized to make peak height ~1
        kernel = np.ones(kernel_size) / approx_daz / kernel_size
        kernel[kernel_size//2:] *= -1

        # convolve signal with the kernel
        pad_init = np.ones(kernel_size//2) * daz[0]
        pad_last = np.ones(kernel_size//2) * daz[-1]
        matched = np.convolve(np.hstack([pad_init, daz, pad_last]), kernel, mode='same')
        matched = matched[kernel_size//2:-kernel_size//2]

        # find peaks in matched daz
        peaks, _ = find_peaks(np.abs(matched), height=peak_threshold, distance=rel_distance_peaks*approx_samps_onescan)
        is_pos = matched[peaks] > 0

        # update approx_samps_onescan
        approx_samps_onescan = int(np.mean(np.diff(peaks)))
        
        # flags turnarounds, left/right scans
        _ta_flag = np.zeros(aman.samps.count, dtype=bool)
        _left_flag = np.zeros(aman.samps.count, dtype=bool)
        _right_flag = np.zeros(aman.samps.count, dtype=bool)
        
        dt = np.mean(np.diff(aman.timestamps))
        nbuffer_half = int(t_buffer/dt//2)
        
        for ip,p in enumerate(peaks[:-1]):
            _ta_flag[p-nbuffer_half:p+nbuffer_half] = True
            if is_pos[ip]: 
                _right_flag[peaks[ip]:peaks[ip+1]] = True
            else:
                _left_flag[peaks[ip]:peaks[ip+1]] = True
        _ta_flag[peaks[-1]-nbuffer_half:peaks[-1]+nbuffer_half] = True
        
        # Check the initial/last part. If the daz is the same as the other scaning part,
        # the part is regarded as left or right scan. If not, flagged as `_truncate_flag`, which
        # will be truncated if `truncate` is True, or flagged as `turnarounds` if `truncate` is False.
        _truncate_flag = np.zeros(aman.samps.count, dtype=bool)
        daz_right = daz[_right_flag & ~_ta_flag]
        daz_right_mean, daz_right_std, daz_right_samps = daz_right.mean(), daz_right.std(), daz_right.shape[0]
        daz_left = daz[_left_flag & ~_ta_flag]
        daz_left_mean, daz_left_std, daz_left_samps = daz_left.mean(), daz_left.std(), daz_left.shape[0]
        
        part_slices_ta_masked = [slice(None, np.where(_ta_flag)[0][0]), slice(np.where(_ta_flag)[0][-1], None)]
        part_slices_ta_unmasked = [slice(None, peaks[0]), slice(peaks[-1], None)]
        
        for part_slice_ta_masked, part_slice_ta_unmasked in zip(part_slices_ta_masked, part_slices_ta_unmasked):
            daz_part = daz[part_slice_ta_masked]
            daz_part_mean, daz_part_std, daz_part_samps = daz_part.mean(), daz_part.std(), daz_part.shape[0]
            if np.isclose(daz_part_mean, daz_right_mean, rtol=0, atol=3*daz_right_std/np.sqrt(daz_right_samps)) and \
                np.isclose(daz_part_std, daz_right_std, rtol=1, atol=0):
                _right_flag[part_slice_ta_unmasked] = True
            elif np.isclose(daz_part_mean, daz_left_mean, rtol=0, atol=3*daz_right_std/np.sqrt(daz_left_samps)) and \
                np.isclose(daz_part_std, daz_left_std, rtol=1, atol=0):
                _left_flag[part_slice_ta_unmasked] = True
            else:
                _truncate_flag[part_slice_ta_unmasked] = True
        
        # Check if flagging works
        check_sum = _left_flag.astype(int) + _right_flag.astype(int) + _truncate_flag.astype(int)
        check_sum = np.all(np.ones(aman.samps.count, dtype=int) == check_sum)
        if not check_sum:
            raise ValueError('Check sum failed. There are samples not allocated any of left, right, or truncate.')
        
        # merge left/right mask
        left_flag = Ranges.from_bitmask(_left_flag)
        right_flag = Ranges.from_bitmask(_right_flag)
        if merge_lr:
            if ("left_scan" in aman.flags or "right_scan" in aman.flags ) and not overwrite:
                raise ValueError("Flag name left/right_flag already exists in aman.flags.")
            else : 
                if "left_scan" in aman.flags:
                    aman.flags["left_scan"] = left_flag
                else :
                    aman.flags.wrap("left_scan", left_flag)
                if "right_scan" in aman.flags:
                    aman.flags["right_scan"] = right_flag
                else :
                    aman.flags.wrap("right_scan", right_flag)

        # truncate unstable scan before the first turnaround or after the last turnaround
        if truncate:
            valid_slice = slice(*np.where(~_truncate_flag)[0][[0, -1]])
            aman.restrict('samps', valid_slice)
            ta_flag = Ranges.from_bitmask(_ta_flag[valid_slice])
        else:
            ta_flag = Ranges.from_bitmask(np.logical_or(_ta_flag, _truncate_flag))
    
    # merge turnaround flags
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError("Flag name {} already exists in aman.flags".format(name))
        elif name in aman.flags:
            aman.flags[name] = ta_flag
        else:
            print(ta_flag)
            aman.flags.wrap(name, ta_flag)   
    if method == 'az':
        ta_exp = RangesMatrix([ta_flag for i in range(aman.dets.count)])
        return ta_exp
    if method == 'scanspeed':
        ta_exp = RangesMatrix([ta_flag for i in range(aman.dets.count)])
        left_exp = RangesMatrix([left_flag for i in range(aman.dets.count)])
        right_exp = RangesMatrix([right_flag for i in range(aman.dets.count)])
        return ta_exp, left_exp, right_exp
    
def get_glitch_flags(aman,
                     t_glitch=0.002,
                     hp_fc=0.5,
                     n_sig=10,
                     buffer=200,
                     detrend=None,
                     signal=None,
                     merge=True,
                     overwrite=False,
                     name="glitches",
                     full_output=False):
    """
    Find glitches with fourier filtering. Translation from moby2 as starting point

    Parameters
    ----------
    aman : AxisManager
        The tod.
    t_glitch : float
        Gaussian filter width.
    hp_fc : float
        High pass filter cutoff.
    n_sig : int or float
        Significance of detection.
    buffer : int
        Amount to buffer flags around found location
    detrend : str
        Detrend method to pass to fourier_filter
    signal : str
        Field name in aman to detect glitches on if None, defaults to ``signal``
    merge : bool)
        If true, add to ``aman.flags``
    name : string
        Name of flag to add to ``aman.flags``
    overwrite : bool
        If true, write over flag. If false, raise ValueError if name already exists in AxisManager
    full_output : bool
        If true, return sparse matrix with the significance of the detected glitches

    Returns
    -------
    flag : RangesMatrix
        RangesMatrix object containing glitch mask.
    """

    if signal is None:
        signal = "signal"
    # f-space filtering
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(t_sigma=t_glitch)
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


def get_trending_flags(aman,
                       max_trend=1.2,
                       n_pieces=1,
                       max_samples=500,
                       signal=None,
                       timestamps=None,
                       merge=True,
                       overwrite=True,
                       name="trends",
                       full_output=False):
    """
    Flag Detectors with trends larger than max_trend. 
    This function can be used to find unlocked detectors.
    Note that this is a rough cut and unflagged detectors can still have poor tracking.

    Parameters
    ----------
    aman : AxisManager
        The tod
    max_trend : float
        Slope at which detectors are unlocked. The default is for use with phase units.
    n_pieces : int
        Number of pieces to cut the timestream in to to look for trends.
    max_samples : int
        Maximum samples to compute the slope with.
    signal : array
        (Optional). Signal to use to generate flags, if None default is aman.signal.
    timestamps : array
        (Optional). Timestamps to use to generate flags, default is aman.timestamps.
    merge : bool
        If true, merges the generated flag into aman.
    overwrite : bool
        If true, write over flag. If false, don't.
    name : str
        Name of flag to add to aman.flags if merge is True.
    full_output : bool
        If true, returns calculated slope sizes

    Returns
    -------
    cut : RangesMatrix
        RangesMatrix of trending regions
    trends : AxisManager
        If full_output is true, calculated slopes and the sample edges where they were calculated.
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
