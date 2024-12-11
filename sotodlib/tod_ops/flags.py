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
from .. import coords
from . import filters
from . import fourier_filter 
from . import sub_polyf

def get_det_bias_flags(aman, detcal=None, rfrac_range=(0.1, 0.7),
                       psat_range=None, rn_range=None, si_nan=False,
                       merge=True, overwrite=True,
                       name='det_bias_flags', full_output=False):
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
        If None, no flags are not applied from P_SAT. 
    rn_range : Tuple
        Tuple (lower_bound, upper_bound) for r_n det selection.
    si_nan : bool
        If true, flag dets where s_i is NaN. Default is false.
    merge : bool
        If true, merges the generated flag into aman.
    overwrite : bool
        If true, write over flag. If false, don't.
    name : str
        Name of flag to add to aman.flags if merge is True.
    full_output : bool
        If true, returns the full output with separated RangesMatrices

    Returns
    -------
    msk_aman : AxisManager
        AxisManager containing RangesMatrix shaped N_dets x N_samps
        that is True if the detector is flagged to be cut and false
        if it should be kept based on the rfrac, and psat ranges. 
        To create a boolean mask from the RangesMatrix that can be
        used for aman.restrict() use ``keep = ~has_all_cut(mask)``
        and then restrict with ``aman.restrict('dets', aman.dets.vals[keep])``.
        If full_output is True, this will contain multiple RangesMatrices.
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

    ranges = [detcal.bg >= 0,
              detcal.r_tes > 0,
              detcal.r_frac >= rfrac_range[0],
              detcal.r_frac <= rfrac_range[1]
             ]    
    if psat_range is not None:
        ranges.append(detcal.p_sat*1e12 >= psat_range[0])
        ranges.append(detcal.p_sat*1e12 <= psat_range[1])
    if rn_range is not None:
        ranges.append(detcal.r_n >= rn_range[0])
        ranges.append(detcal.r_n <= rn_range[1])
    if si_nan:
        ranges.append(np.isnan(detcal.s_i) == False)

    msk = ~(np.all(ranges, axis=0))
    # Expand mask to ndets x nsamps RangesMatrix
    if 'samps' in aman:
        x = Ranges(aman.samps.count)
        mskexp = RangesMatrix([Ranges.ones_like(x) if Y
                            else Ranges.zeros_like(x) for Y in msk])
        msk_aman = core.AxisManager(aman.dets, aman.samps)
        msk_aman.wrap(name, mskexp, [(0, 'dets'), (1, 'samps')])
    else:
        mskexp = msk
        msk_aman = core.AxisManager(aman.dets)
        msk_aman.wrap(name, mskexp, [(0, 'dets')])
    
    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {name} already exists in aman.flags")
        if name in aman.flags:
            aman.flags[name] = mskexp
        else:
            aman.flags.wrap(name, mskexp, [(0, 'dets'), (1, 'samps')])

    if full_output:
        msks = []
        ranges = [detcal.bg >= 0,
                  detcal.r_tes > 0,
                  detcal.r_frac >= rfrac_range[0],
                  detcal.r_frac <= rfrac_range[1]
                 ]
        if psat_range is not None:
            ranges.append(detcal.p_sat*1e12 >= psat_range[0])
            ranges.append(detcal.p_sat*1e12 <= psat_range[1])

        for range in ranges:
            msk = ~(np.all([range], axis=0))
            msks.append(RangesMatrix([Ranges.ones_like(x) if Y
                                      else Ranges.zeros_like(x) for Y in msk]))

        msk_names = ['bg', 'r_tes', 'r_frac_gt', 'r_frac_lt']
        
        if psat_range is not None:
            msk_names.extend(['p_sat_gt', 'p_sat_lt'])
            
        for i, msk in enumerate(msks):
            if 'samps' in aman:
                msk_aman.wrap(f'{msk_names[i]}_flags', msk, [(0, 'dets'), (1, 'samps')])
            else:
                msk_aman.wrap(f'{msk_names[i]}_flags', msk, [(0, 'dets')])
    
    return msk_aman

def get_turnaround_flags(aman, az=None, method='scanspeed', name='turnarounds',
                         merge=True, merge_lr=True, overwrite=True, 
                         t_buffer=2., kernel_size=400, peak_threshold=0.1, rel_distance_peaks=0.3,
                         truncate=False, qlim=1, merge_subscans=True, turnarounds_in_subscan=False):
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
    merge_subscans : bool
        (Optional). Also merge an AxisManager with subscan information.
    turnarounds_in_subscan : bool
        (Optional). Turnarounds are included as part of a subscan.

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
            if np.isclose(daz_part_mean, daz_right_mean, rtol=0.01, atol=3.*daz_right_std/np.sqrt(daz_right_samps)) and \
                np.isclose(daz_part_std, daz_right_std, rtol=3., atol=0):
                _right_flag[part_slice_ta_unmasked] = True
            elif np.isclose(daz_part_mean, daz_left_mean, rtol=0.01, atol=3.*daz_right_std/np.sqrt(daz_left_samps)) and \
                np.isclose(daz_part_std, daz_left_std, rtol=3., atol=0):
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
            valid_i_start, valid_i_end = np.where(~_truncate_flag)[0][0], np.where(~_truncate_flag)[0][-1]
            aman.restrict('samps', (aman.samps.offset + valid_i_start, aman.samps.offset+valid_i_end))
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
            aman.flags.wrap(name, ta_flag)   

    if merge_subscans:
        get_subscans(aman, merge=True, include_turnarounds=turnarounds_in_subscan)

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
                     signal_name=None,
                     merge=True,
                     overwrite=False,
                     name="glitches",
                     full_output=False,
                     edge_guard=2000,
                     subscan=False):
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
    signal_name : str
        Field name in aman to detect glitches on if None, defaults to ``signal``
    merge : bool)
        If true, add to ``aman.flags``
    name : string
        Name of flag to add to ``aman.flags``
    overwrite : bool
        If true, write over flag. If false, raise ValueError if name already exists in AxisManager
    full_output : bool
        If true, return sparse matrix with the significance of the detected glitches
    edge_guard : int
        Number of samples at the beginning and end of the tod to exclude from
        the returned glitch RangesMatrix. Defaults to 2000 samples (10 sec).
    subscan : bool
        If True, compute the glitch threshold on a per-subscan basis. Includes turnarounds.

    Returns
    -------
    flag : RangesMatrix
        RangesMatrix object containing glitch mask.
    """

    if signal_name is None:
        signal_name = "signal"
    # f-space filtering
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(t_sigma=t_glitch)
    fvec = fourier_filter(
        aman, filt, detrend=detrend, signal_name=signal_name, resize="zero_pad"
    )
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    if fvec.shape[1] > 50000:
        ds = int(fvec.shape[1]/20000)
    else: 
        ds = 1

    if subscan:
        # We include turnarounds
        subscan_indices = np.concatenate([aman.flags.left_scan.ranges(), (~aman.flags.left_scan).ranges()])
    else:
        subscan_indices = np.array([[0, fvec.shape[1]]])

    msk = np.zeros_like(fvec, dtype='bool')
    for ss in subscan_indices:
        iqr_range = 0.741 * stats.iqr(fvec[:,ss[0]:ss[1]:ds], axis=1)
        # get flags
        msk[:,ss[0]:ss[1]] = fvec[:,ss[0]:ss[1]] > iqr_range[:, None] * n_sig
    msk[:,:edge_guard] = False
    msk[:,-edge_guard:] = False
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
                       t_piece=500,
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
    t_piece : float
        Duration in seconds of each pieces to cut the timestream in to to look for trends
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
    
    # This helps with floating point precision
    # Not modifying inplace since we don't want to touch aman.timestamps
    timestamps = timestamps - timestamps[0]

    slopes = np.zeros((len(signal), 0))
    cut = np.zeros((len(signal), 0), dtype=bool)
    samp_edges = [0]
    
    # Get sampling rate
    fs = 1. / np.nanmedian(np.diff(timestamps))
    n_samples_per_piece = int(t_piece * fs)
    # How many pieces can timestamps be divided into
    n_pieces = len(timestamps) // n_samples_per_piece
    
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

def get_dark_dets(aman, merge=True, overwrite=True, dark_flags_name='darks'):
    """
    Identify and flag dark detectors in the given aman object.

    Parameters:
    ----------
    aman : AxisManager
        The tod.
    merge : bool, optional
        If True, merge the dark detector flags into the aman.flags. Default is True.
    overwrite : bool, optional
        If True, overwrite existing flags with the same name. Default is True.
    dark_flags_name : str, optional
        The name to use for the dark detector flags in aman.flags. Default is 'darks'.
    
    Returns:
    -------
    mskdarks: RangesMatrix
        A matrix of ranges indicating the dark detectors.

    Raises:
    -------
    ValueError
        If merge is True and dark_flags_name already exists in aman.flags and overwrite is False.
    """
    darks = np.array(aman.det_info.wafer.type != 'OPTC')
    x = Ranges(aman.samps.count)
    mskdarks = RangesMatrix([Ranges.ones_like(x) if Y
                                else Ranges.zeros_like(x) for Y in darks])
    
    if merge:
        if dark_flags_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {dark_flags_name} already exists in aman.flags")
        if dark_flags_name in aman.flags:
            aman.flags[dark_flags_name] = mskdarks
        else:
            aman.flags.wrap(dark_flags_name, mskdarks, [(0, 'dets'), (1, 'samps')])

    return mskdarks

def get_source_flags(aman, merge=True, overwrite=True, source_flags_name='source_flags',
                     mask=None, center_on=None, res=None, max_pix=None):
    if merge:
        wrap = source_flags_name
    else:
        wrap = None
    if res:
        res = np.radians(res/60)
    source_flags = coords.planets.compute_source_flags(tod=aman, wrap=wrap, mask=mask, center_on=center_on, res=res, max_pix=max_pix)
    
    if merge:
        if source_flags_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {source_flags_name} already exists in aman.flags")
        if source_flags_name in aman.flags:
            aman.flags[source_flags_name] = source_flags
        else:
            aman.flags.wrap(source_flags_name, source_flags, [(0, 'dets'), (1, 'samps')])

    return source_flags

def get_ptp_flags(aman, signal_name='signal', kurtosis_threshold=5,
                  merge=False, overwrite=False, ptp_flag_name='ptp_flag',
                  outlier_range=(0.5, 2.)):
    """
    Returns a ranges matrix that indicates if the peak-to-peak (ptp) of
    the tod is valid based on the kurtosis of the distribution of ptps. The
    threshold is set by ``kurtosis_threshold``.

    Parameters
    ----------
    aman : AxisManager
        The tod
    signal_name : str
        Signal to estimate flags off of. Default is ``signal``.
    kurtosis_threshold : float
        Maximum allowable kurtosis of the distribution of peak-to-peaks.
        Default is 5.
    merge : bool
        Merge RangesMatrix into ``aman.flags``. Default is False.
    overwrite : bool
        Whether to write over any existing data in ``aman.flags[ptp_flag_name]`` 
        if merge is True. Default is False.
    ptp_flag_name : str
        Field name used when merge is True. Default is ``ptp_flag``.
    outlier_range : tuple
        (lower, upper) bound of the initial cut before estimating the kurtosis.

    Returns
    -------
    mskptps : RangesMatrix
        RangesMatrix of detectors with acceptable peak-to-peaks. 
        All ones if the detector should be cut.

    """
    det_mask = np.full(aman.dets.count, True, dtype=bool)
    ptps_full = np.ptp(aman[signal_name], axis=1)
    ratio = ptps_full/np.median(ptps_full)
    outlier_mask = (ratio<outlier_range[0]) | (outlier_range[1]<ratio)
    det_mask[outlier_mask] = False
    if np.any(np.logical_not(np.isfinite(aman[signal_name][det_mask]))):
        raise ValueError(f"There is a nan in {signal_name} in aman {aman.obs_info['obs_id']} !!!")
    while True:
        if len(aman.dets.vals[det_mask]) > 0:
            ptps = np.ptp(aman[signal_name][det_mask], axis=1)
        else:
            break
        kurtosis_ptp = stats.kurtosis(ptps)
        if np.abs(kurtosis_ptp) < kurtosis_threshold:
            break
        else:
            max_is_bad_factor = np.max(ptps)/np.median(ptps)
            min_is_bad_factor = np.median(ptps)/np.min(ptps)
            if max_is_bad_factor > min_is_bad_factor:
                det_mask[ptps_full >= np.max(ptps)] = False
            else:
                det_mask[ptps_full <= np.min(ptps)] = False
    x = Ranges(aman.samps.count)
    mskptps = RangesMatrix([Ranges.zeros_like(x) if Y
                             else Ranges.ones_like(x) for Y in det_mask])
    if merge:
        if ptp_flag_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {ptp_flag_name} already exists in aman.flags")
        if ptp_flag_name in aman.flags:
            aman.flags[ptp_flag_name] = mskptps
        else:
            aman.flags.wrap(ptp_flag_name, mskptps, [(0, 'dets'), (1, 'samps')])

    return mskptps

def get_inv_var_flags(aman, signal_name='signal', nsigma=5,
                      merge=False, overwrite=False, inv_var_flag_name='inv_var_flag'):
    """
    Returns a ranges matrix that indicates if the inverse variance (inv_var) of
    the tod is greater than ``nsigma`` away from the median.
    
    Parameters
    ----------
    aman : AxisManager
        The tod
    signal_name : str
        Signal to estimate flags off of. Default is ``signal``.
    nsigma : float
        Maximum allowable deviation from the median inverse variance.
        Default is 5.
    merge : bool
        Merge RangesMatrix into ``aman.flags``. Default is False.
    overwrite : bool
        Whether to write over any existing data in ``aman.flags[inv_var_flag_name]`` 
        if merge is True. Default is False.
    inv_var_flag_name : str
        Field name used when merge is True. Default is ``inv_var_flag``.

    Returns
    -------
    mskptps : RangesMatrix
        RangesMatrix of detectors with acceptable inverse variance. 
        All ones if the detector should be cut.

    """
    ivar = 1.0/np.var(aman[signal_name], axis=-1)
    sigma = (np.percentile(ivar,84) - np.percentile(ivar, 16))/2
    det_mask = ivar > np.median(ivar) + nsigma*sigma
    x = Ranges(aman.samps.count)
    mskinvar = RangesMatrix([Ranges.ones_like(x) if Y
                             else Ranges.zeros_like(x) for Y in det_mask])
    if merge:
        if inv_var_flag_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {inv_var_flag_name} already exists in aman.flags")
        if inv_var_flag_name in aman.flags:
            aman.flags[inv_var_flag_name] = mskinvar
        else:
            aman.flags.wrap(inv_var_flag_name, mskinvar, [(0, 'dets'), (1, 'samps')])

    return mskinvar

def get_subscans(aman, merge=True, include_turnarounds=False, overwrite=True):
    """
    Returns an axis manager with information about subscans.
    This includes direction and a ranges matrix (subscans samps)
    True inside each subscan.

    Parameters
    ----------
    aman : AxisManager
        Input AxisManager.
    merge : bool
        Merge into aman as 'subscan_info'
    include_turnarounds : bool
        Include turnarounds in the subscan ranges
    overwrite : bool
        If true, write over subscan_info.

    Returns
    -------
    subscan_aman : AxisManager
        AxisManager containing information about the subscans.
        "direction" is a (subscans,) array of strings 'left' or 'right'
        "subscan_flags" is a (subscans, samps) RangesMatrix; True inside the subscan.
    """
    if not include_turnarounds:
        ss_ind = (~aman.flags.turnarounds).ranges() # sliceable indices (first inclusive, last exclusive) for subscans
    else:
        left = aman.flags.left_scan.ranges()
        right = aman.flags.right_scan.ranges()
        start_left = 0 if (left[0,0] < right[0,0]) else 1
        ss_ind = np.empty((left.shape[0] + right.shape[0], 2), dtype=left.dtype)
        ss_ind[start_left::2] = left
        ss_ind[(start_left-1)%2::2] = right

    start_inds, end_inds = ss_ind.T
    n_subscan = ss_ind.shape[0]
    tt = aman.timestamps
    subscan_aman = core.AxisManager(aman.samps, core.IndexAxis("subscans", n_subscan))

    is_left = aman.flags.left_scan.mask()[start_inds]
    subscan_aman.wrap('direction', np.array(['left' if is_left[ii] else 'right' for ii in range(n_subscan)]), [(0, 'subscans')])

    rm = RangesMatrix([Ranges.from_array(np.atleast_2d(ss), tt.size) for ss in ss_ind])
    subscan_aman.wrap('subscan_flags', rm, [(0, 'subscans'), (1, 'samps')]) # True in the subscan
    if merge:
        name = 'subscan_info'
        if overwrite and name in aman:
            aman.move(name, None)
        aman.wrap(name, subscan_aman)
    return subscan_aman

def get_subscan_signal(aman, arr, isub=None, trim=False):
    """
    Split an array into subscans.

    Parameters
    ----------
    aman : AxisManager
        Input AxisManager.
    arr : Array
        Input array.
    isub : int
        Index of the desired subscan. May also be a list of indices.
        If None, all are used.
    trim : bool
        Do not include size-zero arrays from empty subscans in the output.

    Returns
    -------
    out : list
        If isub is a scalar, return an Array of arr cut on the samps axis to the given subscan.
        If isub is a list or None, return a list of such Arrays.
    """
    if isinstance(arr, str):
        arr = aman[arr]
    if np.isscalar(isub):
        out = apply_rng(arr, aman.subscan_info.subscan_flags[isub])
        if trim and out.size == 0:
            out = None
    else:
        if isub is None:
            isub = range(len(aman.subscan_info.subscan_flags))
        out = [apply_rng(arr, aman.subscan_info.subscan_flags[ii]) for ii in isub]
        if trim:
            out = [x for x in out if x.size > 0]

    return out


def apply_rng(arr, rng):
    """
    Apply a Ranges object to an array. rng should be True on the samples you want to keep.

    Parameters
    ----------
    arr : Array
        Array containing the signal. Should have one axis of len (samps).
    rng : Ranges
        Ranges object of len (samps) selecting the desired range.
    """
    if rng.ranges().size == 0:
        slices = [slice(0,0)] #  Return an empty array if rng is empty
    else:
        slices = [slice(*irng) for irng in rng.ranges()]

    # Identify the samps axis
    isamps = np.where(np.array(arr.shape) == rng.count)[0]
    if isamps.size != 1:
        # Check for axis mismatch between arr and rng, or multiple axes with the same size
        raise RuntimeError("Could not identify axis matching Ranges")
    # Apply ranges
    out = []
    for slc in slices:
        ndslice = tuple((slice(None) if ii != isamps[0] else slc for ii in range(arr.ndim)))
        out.append(arr[ndslice])
    return np.concatenate(out, axis=isamps[0])

def wrap_stats(aman, info_aman_name, info, info_names, merge=True):
    """
    Wrap multiple stats into a new aman, checking for subscan information. Stats can be (dets,) or (dets, subscans).

    Parameters
    ----------
    aman : AxisManager
        Input AxisManager.
    info_aman_name : str
        Name for info_aman when wrapped into input.
    info : Array
        (stats, dets,) or (stats, dets, subscans) containing the information you want to wrap.
    info_names : list
        List of str names for each entry in the new aman.
    merge : bool
        If True merge info_aman into aman.

    Returns
    -------
    info_aman : AxisManager
        (dets,) or (dets, subscans) aman with a field for each item in info_names.
    """
    info_names = np.atleast_1d(info_names)
    info = np.atleast_2d(info)
    if info.shape == (len(info_names), aman.dets.count): # (stats, dets)
        if len(info_names) == aman.dets.count and aman.dets.count == aman.subscan_info.subscans.count:
            raise RuntimeError("Cannot infer axis mapping") # Catch corner case
        info_aman = core.AxisManager(aman.dets)
        axmap = [(0, 'dets')]

    else:
        info = np.atleast_3d(info) # (stats, dets, subscans)
        info_aman = core.AxisManager(aman.dets, aman.subscan_info.subscans)
        axmap = [(0, 'dets'), (1, 'subscans')]

    for ii in range(len(info_names)):
        info_aman.wrap(info_names[ii], info[ii], axmap)
    if merge:
        if info_aman_name in aman.keys():
            aman[info_aman_name].merge(info_aman)
        else:
            aman.wrap(info_aman_name, info_aman)
    return info_aman

def get_stats(aman, signal, stat_names, split_subscans=False, mask=None, name="stats", merge=False):
    """
    Calculate basic statistics on a TOD or power spectrum.

    Parameters
    ----------
    aman : AxisManager
        Input AxisManager.
    signal : Array
        Input signal. Statistics will be computed over *axis 1*.
    stat_names : list
        List of strings identifying which statistics to run.
    split_subscans : bool
        If True statistics will be computed on subscans. Assumes aman.subscan_info exists already.
    mask : Array
        Mask to apply before computation. 1d array for advanced indexing (keep True), or a slice object.
    name : str
        Name of axis manager to add to aman if merge is True.
    """
    stat_names = np.atleast_1d(stat_names)
    fn_dict = {'mean': np.mean, 'median': np.median, 'ptp': np.ptp, 'std': np.std,
                     'kurtosis': stats.kurtosis, 'skew': stats.skew}

    if isinstance(signal, str):
        signal = aman[signal]
    if split_subscans:
        if mask is not None:
            raise ValueError("Cannot mask samples and split subscans")
        stats_arr = []
        for iss in range(aman.subscan_info.subscans.count):
            data = get_subscan_signal(aman, signal, iss)
            if data.size > 0:
                stats_arr.append([fn_dict[name](data, axis=1) for name in stat_names]) # Samps axis assumed to be 1
            else:
                stats_arr.append(np.full((len(stat_names), signal.shape[0]), np.nan)) # Add nans if subscan has been entirely cut
        stats_arr = np.array(stats_arr).transpose(1, 2, 0) # stat, dets, subscan
    else:
        if mask is None:
            mask = slice(None)
        stats_arr = np.array([fn_dict[name](signal[:, mask], axis=1) for name in stat_names]) # Samps axis assumed to be 1

    info_aman = wrap_stats(aman, name, stats_arr, stat_names, merge)
    return info_aman

def get_focalplane_flags(aman, merge=True, overwrite=True, invalid_flags_name='fp_flags'):
    """
    Generate flags for invalid detectors in the focal plane.
        The tod.
    merge : bool
        If true, merges the generated flag into aman.
    overwrite : bool
        If true, write over flag. If false, don't.
    invalid_flags_name : str
        Name of flag to add to aman.flags if merge is True.

    Returns
    -------
    msk_invalid_fp : RangesMatrix
        RangesMatrix of invalid detectors in the focal plane.
    """
    # Available detectors in focalplane
    xi_nan = np.isnan(aman.focal_plane.xi)
    eta_nan = np.isnan(aman.focal_plane.eta)
    x = Ranges(aman.samps.count)
    msk_invalid_fp = RangesMatrix([
        Ranges.ones_like(x) if Y else Ranges.zeros_like(x) 
        for Y in flag_invalid_fp
    ])
    flag_valid_fp = np.sum([xi_nan, eta_nan, gamma_nan], axis=0) == 0
    flag_invalid_fp = ~flag_valid_fp
    msk_invalid_fp = RangesMatrix([Ranges.ones_like(x) if Y else Ranges.zeros_like(x) for Y in flag_invalid_fp])
    
    if merge:
        if invalid_flags_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {invalid_flags_name} already exists in aman.flags")
        if invalid_flags_name in aman.flags:
            aman.flags[invalid_flags_name] = msk_invalid_fp
        else:
            aman.flags.wrap(invalid_flags_name, msk_invalid_fp, [(0, 'dets'), (1, 'samps')])

    return msk_invalid_fp


def get_badsubscan_flags(aman, nstd_threshold=3.0, Tptp_pW_threshold=0.5, kurt_threshold=0.5, 
                         skew_threshold=0.5, merge=False, overwrite=False, name="bad_subscan"):
    """
    Identify and flag bad subscans based on various statistical thresholds.
        The tod.
    nstd_threshold : float
        Threshold for standard deviation.
    Tptp_pW_threshold : float
        Threshold for peak-to-peak values in pW.
    kurt_threshold : float
        Threshold for kurtosis.
    skew_threshold : float
        Threshold for skewness.
    merge : bool
        If true, merges the generated flag into aman.
    overwrite : bool
        If true, write over flag. If false, don't.
    name : str
        Name of flag to add to aman.flags if merge is True.

    Returns
    -------
    badsubscan_flags : RangesMatrix
        RangesMatrix of bad subscans.
    """
    if 'flags' not in aman:
        overwrite = False
        merge = False
    if overwrite and name in aman.flags:
        aman.flags.move(name, None)

    subscan_indices_l = sub_polyf._get_subscan_range_index(aman.flags["left_scan"].mask())
    subscan_indices_r = sub_polyf._get_subscan_range_index(aman.flags["right_scan"].mask())
    subscan_indices = np.vstack([subscan_indices_l, subscan_indices_r])
    subscan_indices = subscan_indices[np.argsort(subscan_indices[:, 0])]

    num_dets = aman.dets.count
    num_subscans = len(subscan_indices)
    subscan_stats = {
        'Tptp': np.zeros((num_dets, num_subscans)),
        'Qstd': np.zeros((num_dets, num_subscans)),
        'Ustd': np.zeros((num_dets, num_subscans)),
        'Qkurt': np.zeros((num_dets, num_subscans)),
        'Ukurt': np.zeros((num_dets, num_subscans)),
        'Qskew': np.zeros((num_dets, num_subscans)),
        'Uskew': np.zeros((num_dets, num_subscans))
    }

    for subscan_i, subscan in enumerate(subscan_indices):
        _Tsig = aman.dsT[:, subscan[0]:subscan[1] + 1]
        _Qsig = aman.demodQ[:, subscan[0]:subscan[1] + 1]
        _Usig = aman.demodU[:, subscan[0]:subscan[1] + 1]

        subscan_stats['Tptp'][:, subscan_i] = np.ptp(_Tsig, axis=1)
        subscan_stats['Qstd'][:, subscan_i] = np.std(_Qsig, axis=1)
        subscan_stats['Ustd'][:, subscan_i] = np.std(_Usig, axis=1)
        subscan_stats['Qkurt'][:, subscan_i] = stats.kurtosis(_Qsig, axis=1)
        subscan_stats['Ukurt'][:, subscan_i] = stats.kurtosis(_Usig, axis=1)
        subscan_stats['Qskew'][:, subscan_i] = stats.skew(_Qsig, axis=1)
        subscan_stats['Uskew'][:, subscan_i] = stats.skew(_Usig, axis=1)

    median_Qstd = np.median(subscan_stats['Qstd'], axis=1)[:, np.newaxis]
    median_Ustd = np.median(subscan_stats['Ustd'], axis=1)[:, np.newaxis]

    badsubscan_indicator = (
        (subscan_stats['Tptp'] > Tptp_pW_threshold) |
        (subscan_stats['Qstd'] > median_Qstd * nstd_threshold) |
        (subscan_stats['Ustd'] > median_Ustd * nstd_threshold) |
        (np.abs(subscan_stats['Qkurt']) > kurt_threshold) |
        (np.abs(subscan_stats['Ukurt']) > kurt_threshold) |
        (np.abs(subscan_stats['Qskew']) > skew_threshold) |
        (np.abs(subscan_stats['Uskew']) > skew_threshold)
    )

    badsubscan_flags = np.zeros((num_dets, aman.samps.count), dtype=bool)
    for subscan_i, subscan in enumerate(subscan_indices):
        badsubscan_flags[:, subscan[0]:subscan[1] + 1] = badsubscan_indicator[:, subscan_i, np.newaxis]
    
    # Detectors which are "bad" for > 50% of the the subscan duration
    baddetector_flags = np.mean(badsubscan_flags, axis=1) > 0.5

    badsubscan_flags = RangesMatrix.from_mask(badsubscan_flags)
    baddetector_flags = Ranges.from_mask(baddetector_flags)

    if merge:
        if name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {name} already exists in aman.flags")
        aman.flags.wrap(name, badsubscan_flags)

    return badsubscan_flags, baddetector_flags


def whitenoi_fknee_cuts(aman, low_wn, high_wn, high_fk):
    """
    Evaluate white noise and fknee cuts based on provided boundaries.

    Parameters:
    aman : object
        An object containing noise fit statistics and noise model coefficients.
    low_wn : float or None
        The lower boundary for white noise. If None, white noise flagging is skipped.
    high_wn : float or None
        The upper boundary for white noise. If None, white noise flagging is skipped.
    high_fk : float or None
        The upper boundary for fknee. If None, fknee flagging is skipped.

    Returns:
    tuple or None
        A tuple containing flags for valid white noise and fknee if both boundaries are provided.
        If only one boundary is provided, returns the corresponding flag.
        If no boundaries are provided, returns None.
    """
    noise = aman.noise_fit_stats_signal.fit
    fk = noise[:, 0]
    wn = noise[:, 1]
    if low_wn is None:
        print(f"white noise boundaries are not defined, skipping.")
        flag_valid_wn = None
    else:
        flag_valid_wn = (low_wn < wn * 1e6) & (wn * 1e6 < high_wn)
    if high_fk is None:
        print(f"fknee boundaries are not defined, skipping.")
        flag_valid_fk = None
    else:
        flag_valid_fk = fk < high_fk
    if low_wn is not None and high_fk is not None:
        return flag_valid_wn, flag_valid_fk
    elif low_wn is not None:
        return flag_valid_wn
    elif high_fk is not None:
        return flag_valid_fk
    else:
        return None
