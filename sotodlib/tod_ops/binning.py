import numpy as np
import logging
logger = logging.getLogger(__name__)

def bin_signal(aman, bin_by, signal=None,
               range=None, bins=100, flags=None,
               weight_for_signal=None):
    """
    Bin time-ordered data by the ``bin_by`` and return the binned signal and its standard deviation.

    Parameters
    ----------
    aman : TOD
        The Axismanager object to be binned.
    bin_by : array-like
        The array by which signal is binned. It should has the same samps as aman.
    signal : str, optional
        The name of the signal to be binned. Defaults to aman.signal if not specified.
        Either 1D array of shape ``(samps,)`` or 2D array of shape ``(dets, samps)``.
    range : list or None
        A list specifying the bin range ([min, max]). Default is None, which means bin range
        is set to [min(bin_by), max(bin_by)].
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If `bins` is a sequence, `bins` overwrite `range`.
    flags : (str or RangesMatrix or Ranges), optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        If provided by a string, `aman.flags.get(flags)` is used for the flags.
        Default is no mask applied.
    weight_for_signal : array-like, optional
        Array of weights for the signal values. If None, all weights are assumed to be 1. You can get a apodizing window by
        'sotodlib.tod_ops.apodize.get_apodize_window_for_ends' or 'get_apodize_window_from_flags'.

    Returns
    -------
    Dictionary:
        - **bin_edges** (dict key): float array of bin edges length(bin_centers)+1.
        - **bin_centers** (dict key): center of each bin.
        - **bin_counts** (dict key): counts of binned samples. 
        - **binned_signal** (dict key): binned signal.
        - **binned_signal_sigma** (dict key): estimated sigma of binned signal.
    """
    if signal is None:
        signal = aman.signal
    elif isinstance(signal, str):
        signal = aman.get(signal)

    if signal.ndim == 1:
        is_1d = True
    elif signal.ndim == 2:
        is_1d = False
    else:
        raise ValueError(
            f"signal must be 1D (samps,) or 2D (dets, samps); got ndim={signal.ndim}")

    if signal.shape[-1] != aman.samps.count:
        raise ValueError(
            f"signal last axis ({signal.shape[-1]}) does not match "
            f"aman.samps.count ({aman.samps.count})")

    if range is None:
        range = (np.nanmin(bin_by), np.nanmax(bin_by))

    signal_dtype = signal.dtype
    
    if weight_for_signal is None:
        weight_for_signal = np.ones(aman.samps.count, signal_dtype)
        
    # get bin_edges
    bin_edges = np.histogram_bin_edges(bin_by, bins=bins, range=range,)
    bin_centers = (bin_edges[1] - bin_edges[0])/2. + bin_edges[:-1] # edge to center
    nbins = len(bin_centers)
    
    # get bin indices
    bin_indices = np.digitize(bin_by, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins-1)

    # Drop NaN bin_by and out-of-range samples so that np.digitize + np.clip
    # do not silently pile them into the first/last bin.
    base_valid = (
        np.isfinite(bin_by)
        & (bin_by >= range[0])
        & (bin_by <= range[1])
    )

    if flags is None:
        flag_is_2d = False
        m = base_valid
    else:
        if isinstance(flags, str):
            flags = aman.flags.get(flags)
        if (not is_1d) and flags.shape == (aman.dets.count, aman.samps.count):
            flag_is_2d = True
            m_2d = base_valid[None, :] & ~flags.mask()
        elif flags.shape == (aman.samps.count, ):
            flag_is_2d = False
            m = base_valid & ~flags.mask()
        else:
            raise ValueError('flags should have shape of (`dets`, `samps`) or (`samps`,)')

    if is_1d:
        if weight_for_signal.shape != (aman.samps.count,):
            raise ValueError(
                'weight_for_signal should have shape of (`samps`,) when signal is 1D')

        # prepare binned signal array
        binned_signal = np.full(nbins, np.nan, signal_dtype)
        binned_signal_squared_mean = np.full(nbins, np.nan, signal_dtype)
        binned_signal_sigma = np.full(nbins, np.nan, signal_dtype)

        bin_counts_dets = np.bincount(
            bin_indices[m], weights=weight_for_signal[m], minlength=nbins)
        mcnts = bin_counts_dets > 0
        binned_signal[mcnts] = np.bincount(
            bin_indices[m], weights=signal[m] * weight_for_signal[m], minlength=nbins
        )[mcnts] / bin_counts_dets[mcnts]
        binned_signal_squared_mean[mcnts] = np.bincount(
            bin_indices[m], weights=(signal[m] * weight_for_signal[m]) ** 2,
            minlength=nbins
        )[mcnts] / bin_counts_dets[mcnts]
        binned_signal_sigma[mcnts] = np.sqrt(
            np.abs(binned_signal_squared_mean[mcnts] - binned_signal[mcnts] ** 2)
        ) / np.sqrt(bin_counts_dets[mcnts])

    else:
        # prepare binned signal array
        binned_signal = np.full([aman.dets.count, nbins], np.nan, signal_dtype)
        binned_signal_squared_mean = np.full([aman.dets.count, nbins], np.nan, signal_dtype)
        binned_signal_sigma = np.full([aman.dets.count, nbins], np.nan, signal_dtype)
        bin_counts_dets = np.full([aman.dets.count, nbins], np.nan)

        for i, dets in enumerate(aman.dets.vals):
            if flag_is_2d:
                m = m_2d[i]

            if weight_for_signal.shape == (aman.dets.count, aman.samps.count):
                weight_for_signal_det = weight_for_signal[i]
            elif weight_for_signal.shape == (aman.samps.count, ):
                weight_for_signal_det = weight_for_signal
            else:
                raise ValueError('weight_for_signal should have shape of (`dets`, `samps`) or (`samps`,)')

            bin_counts_dets[i] = np.bincount(bin_indices[m], weights=weight_for_signal_det[m], minlength=nbins)
            mcnts = bin_counts_dets[i] > 0
            binned_signal[i][mcnts] = np.bincount(bin_indices[m], weights=signal[i][m]*weight_for_signal_det[m], minlength=nbins
                                                 )[mcnts]/bin_counts_dets[i][mcnts]
            binned_signal_squared_mean[i][mcnts] = np.bincount(bin_indices[m], weights=(signal[i][m]*weight_for_signal_det[m])**2, minlength=nbins
                                                 )[mcnts]/bin_counts_dets[i][mcnts]
            binned_signal_sigma[i][mcnts] = np.sqrt(np.abs(binned_signal_squared_mean[i,mcnts] - binned_signal[i,mcnts]**2)
                                                 ) / np.sqrt(bin_counts_dets[i][mcnts])

    return {'bin_edges': bin_edges, 'bin_centers': bin_centers, 'bin_counts': bin_counts_dets,
            'binned_signal': binned_signal, 'binned_signal_sigma': binned_signal_sigma}
