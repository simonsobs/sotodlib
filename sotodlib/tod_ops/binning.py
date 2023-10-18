import numpy as np
import logging
logger = logging.getLogger(__name__)

def bin_signal(aman, bin_by, signal=None,
               range=None, bins=100, flags=None):
    """
    Bin time-ordered data by the `bin_by` and return the binned signal and its standard deviation.

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
        is set to [min(bin_by), max(bin_by)] automaticaly by np.histogram.
    bins : int, optional
        The number of bins to use. Default is 100.
    flags : RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.

    Returns
    -------
    A dictionary which contains
        * 'bin_centers': center of each bin of hwp_angle
        * 'binned_signal': binned signal
        * 'binned_signal_sigma': estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal

    # binning hwp_angle tod
    bin_counts, bin_centers = np.histogram(bin_by, bins=bins, range=range)
    bin_centers = (bin_centers[1] - bin_centers[0])/2. + bin_centers[:-1] # edge to center
    
    # find bins with non-zero counts
    mcnts = bin_counts > 0
    
    # bin signal
    binned_signal = np.full([aman.dets.count, bins], np.nan)
    binned_signal_squared_mean = np.full([aman.dets.count, bins], np.nan)
    binned_signal_sigma = np.full([aman.dets.count, bins], np.nan)

    # get mask from flags
    if flags is None:
        m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
    else:
        m = ~flags.mask()

    # binning tod
    for i, dets in enumerate(aman.dets.vals):
        binned_signal[i][mcnts] = np.histogram(bin_by[m[i]], bins=bins, range=range,
                                          weights=signal[i][m[i]])[0][mcnts] / bin_counts[mcnts]

        binned_signal_squared_mean[i][mcnts] = np.histogram(bin_by[m[i]], bins=bins, range=range,
                                                       weights=signal[i][m[i]]**2)[0][mcnts] / bin_counts[mcnts]

    # get sigma of each bin
    binned_signal_sigma[:, mcnts] = np.sqrt(np.abs(binned_signal_squared_mean[:,mcnts] - binned_signal[:,mcnts]**2)
                                 ) / np.sqrt(bin_counts[mcnts])

    return {'bin_centers': bin_centers, 'binned_signal': binned_signal, 'binned_signal_sigma': binned_signal_sigma}
