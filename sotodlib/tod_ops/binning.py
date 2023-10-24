import numpy as np
import logging
logger = logging.getLogger(__name__)

def bin_signal(aman, bin_by, signal=None,
               range=None, bins=100, flags=None):
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
    range : list or None
        A list specifying the bin range ([min, max]). Default is None, which means bin range
        is set to [min(bin_by), max(bin_by)].
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If `bins` is a sequence, `bins` overwrite `range`.
    flags : RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.

    Returns
    -------
    Dictionary:
        - **bin_centers** (dict key): center of each bin
        - **binned_signal** (dict key): binned signal
        - **binned_signal_sigma** (dict key): estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if range is None:
        range = (np.nanmin(bin_by), np.nanmax(bin_by))
        
    # bin `bin_by` data
    bin_counts, bin_edges = np.histogram(bin_by, bins=bins, range=range)
    bin_centers = (bin_edges[1] - bin_edges[0])/2. + bin_edges[:-1] # edge to center
    nbins = len(bin_centers)
    
    # prepare binned signal array
    binned_signal = np.full([aman.dets.count, nbins], np.nan)
    binned_signal_squared_mean = np.full([aman.dets.count, nbins], np.nan)
    binned_signal_sigma = np.full([aman.dets.count, nbins], np.nan)

    # bin tod
    if flags is None:
        for i, dets in enumerate(aman.dets.vals):
            # find indexes of bins with non-zero counts
            mcnts = bin_counts > 0
            binned_signal[i][mcnts] = np.histogram(bin_by, bins=bins, range=range,
                                              weights=signal[i])[0][mcnts] / bin_counts[mcnts]

            binned_signal_squared_mean[i][mcnts] = np.histogram(bin_by, bins=bins, range=range,
                                                           weights=signal[i]**2)[0][mcnts] / bin_counts[mcnts]
            
        binned_signal_sigma[:, mcnts] = np.sqrt(np.abs(binned_signal_squared_mean[:,mcnts] - binned_signal[:,mcnts]**2)
                                     ) / np.sqrt(bin_counts[mcnts])
            
    else:
        for i, dets in enumerate(aman.dets.vals):
            if flags.shape == (aman.dets.count, aman.samps.count):
                m = ~flags.mask()[i]
            elif flags.shape == (aman.samps.count, ):
                m = ~flags.mask()
            else:
                raise ValueError('flags should have shape of (`dets`, `samps`) or (`samps`,)')
            
            bin_counts_masked, _ = np.histogram(bin_by[m], bins=bins, range=range)
            mcnts_masked = bin_counts_masked > 0
            
            binned_signal[i][mcnts_masked] = np.histogram(bin_by[m], bins=bins, range=range,
                                              weights=signal[i][m])[0][mcnts_masked] / bin_counts_masked[mcnts_masked]

            binned_signal_squared_mean[i][mcnts_masked] = np.histogram(bin_by[m], bins=bins, range=range,
                                                           weights=signal[i][m]**2)[0][mcnts_masked] / bin_counts_masked[mcnts_masked]

            binned_signal_sigma[i, mcnts_masked] = np.sqrt(np.abs(binned_signal_squared_mean[i,mcnts_masked] - binned_signal[i,mcnts_masked]**2)
                                         ) / np.sqrt(bin_counts[mcnts_masked])

    return {'bin_edges': bin_edges, 'bin_centers': bin_centers, 'binned_signal': binned_signal,
            'binned_signal_sigma': binned_signal_sigma}
