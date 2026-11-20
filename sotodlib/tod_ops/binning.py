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
        The array by which signal is binned. Any length is allowed, but it must be consistent
        with `signal` (and `flags` and `weight_for_signal`, if specified).
    signal : str or array-like of float, optional
        signal to be binned or its name. Defaults to aman.signal if not specified.
        Either 1D array or 2D array with shape ``(nsamps)`` or``(dets, nsamps)``,
        where nsamps is length of bin_by.
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
    signal = np.asarray(signal)

    if signal.ndim not in [1, 2]:
        raise ValueError(
            f"signal must be 1D or 2D; got ndim={signal.ndim}")
    is_1d = signal.ndim == 1
    signal = np.atleast_2d(signal)
    ndets = signal.shape[0]

    bin_by = np.asarray(bin_by)
    nsamps = len(bin_by)

    if signal.shape[-1] != nsamps:
        raise ValueError("signal shape does not match bin_by")

    if range is None:
        range = (np.nanmin(bin_by), np.nanmax(bin_by))

    signal_dtype = signal.dtype

    # get bin_edges
    bin_edges = np.histogram_bin_edges(bin_by, bins=bins, range=range,)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    nbins = len(bin_centers)

    # get bin indices
    bin_indices = np.digitize(bin_by, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins-1)

    # Drop NaN bin_by and out-of-range samples so that np.digitize + np.clip
    # do not silently pile them into the first/last bin.
    base_valid = (
        np.isfinite(bin_by)
        & (bin_by >= bin_edges[0])
        & (bin_by <= bin_edges[-1])
    )

    if flags is None:
        mask = np.broadcast_to(base_valid, (ndets, nsamps))
    else:
        if isinstance(flags, str):
            flags = aman.flags.get(flags)
        if (not is_1d) and flags.shape == (ndets, nsamps):
            mask = base_valid[None, :] & ~flags.mask()
        elif flags.shape == (nsamps, ):
            mask = np.broadcast_to(base_valid & ~flags.mask(),
                                   (ndets, nsamps))
        else:
            raise ValueError('shape of flags does not match')

    if weight_for_signal is None:
        weight_for_signal = np.ones(nsamps, signal_dtype)
    weight_for_signal = np.asarray(weight_for_signal)

    if (not is_1d) and weight_for_signal.shape == (ndets, nsamps):
        weights = weight_for_signal
    elif weight_for_signal.shape == (nsamps, ):
        weights = np.broadcast_to(weight_for_signal,
                                  (ndets, nsamps))
    else:
        raise ValueError('shape of weight_for_signal does not match')

    # prepare binned signal array
    binned_signal = np.full([ndets, nbins], np.nan, signal_dtype)
    binned_signal_squared_mean = np.full([ndets, nbins], np.nan, signal_dtype)
    binned_signal_sigma = np.full([ndets, nbins], np.nan, signal_dtype)
    bin_counts = np.full([ndets, nbins], np.nan)

    for i in np.arange(ndets):
        m = mask[i]
        w = weights[i]

        bin_counts[i] = np.bincount(bin_indices[m], weights=w[m], minlength=nbins)
        mcnts = bin_counts[i] > 0
        binned_signal[i][mcnts] = np.bincount(
            bin_indices[m], weights=signal[i][m] * w[m], minlength=nbins
        )[mcnts] / bin_counts[i][mcnts]
        binned_signal_squared_mean[i][mcnts] = np.bincount(
            bin_indices[m], weights=(signal[i][m] * w[m])**2, minlength=nbins
        )[mcnts] / bin_counts[i][mcnts]
        binned_signal_sigma[i][mcnts] = np.sqrt(
            np.abs(binned_signal_squared_mean[i, mcnts] - binned_signal[i, mcnts]**2)
        ) / np.sqrt(bin_counts[i][mcnts])

    if is_1d:
        binned_signal = binned_signal[0]
        binned_signal_sigma = binned_signal_sigma[0]
        bin_counts = bin_counts[0]

    return {'bin_edges': bin_edges, 'bin_centers': bin_centers, 'bin_counts': bin_counts,
            'binned_signal': binned_signal, 'binned_signal_sigma': binned_signal_sigma}
