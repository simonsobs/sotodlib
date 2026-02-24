import numpy as np
import scipy as sp
import scipy.stats as ss


def num_of_det(signal, x_pos, y_pos):
    """Return the number of detectors affected by the glitch.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector signals (unused, accepted for a uniform interface).
    x_pos : numpy.ndarray
        X focal-plane positions of affected detectors.
    y_pos : numpy.ndarray
        Y focal-plane positions (unused, accepted for a uniform interface).

    Returns
    -------
    int
        Number of detectors.
    """

    return len(x_pos)


def x_and_y_histogram_extent_ratio(signal, x_pos, y_pos):
    """Compute the ratio of the y to x histogram extents.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector signals (unused, accepted for a uniform interface).
    x_pos : numpy.ndarray
        X focal-plane positions of affected detectors.
    y_pos : numpy.ndarray
        Y focal-plane positions of affected detectors.

    Returns
    -------
    float
        Ratio ``(max(y) - min(y)) / (max(x) - min(x))``.
    """

    hist_ratio = (np.max(y_pos) - np.min(y_pos))/(np.max(x_pos) - np.min(x_pos))

    return hist_ratio

def mean_time_lags(signal, x_pos, y_pos):
    """Compute the mean absolute time lag between detector pairs.

    Uses cross-correlation via FFT to estimate the time delay between
    each pair of detectors, then returns the mean of the absolute
    values.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector TOD signals of shape ``(n_dets, n_samps)``.  Works
        better with detrended data.
    x_pos : numpy.ndarray
        X focal-plane positions (unused, accepted for a uniform interface).
    y_pos : numpy.ndarray
        Y focal-plane positions (unused, accepted for a uniform interface).

    Returns
    -------
    float
        Mean of the absolute time lags.
    """

    lags = np.full((len(signal), len(signal)), np.nan)


    for i in range(len(signal)):
            if len(signal[i]) >= 2:
                for j in range(len(signal)):
                    if j > i:

                        #compute the time delays between detector pair
                        time_delay_pos = np.fft.ifft(np.fft.fft(signal[i])*np.conjugate(np.fft.fft(signal[j])))

                        #find the maximum time delay; corresponds to the time shift required to achieve the maximum correlation
                        max_time_delay_pos = np.max(time_delay_pos)

                        time_delay_ind_t = np.where(time_delay_pos == max_time_delay_pos)[0][0]

                        #determine the difference between the required time shift and the length of the TOD
                        shift_t = time_delay_ind_t - len(signal[i])

                        #Take the smaller value between the required time shift and shift defined above. This allows for shifts backwards instead of looping around the TOD.
                        if np.abs(shift_t) < time_delay_ind_t:
                            lag_t = shift_t
                        else:
                            lag_t = time_delay_ind_t

                        lags[i, j] = lag_t

    time_lag = np.abs(np.nanmean(np.abs(lags)))

    return time_lag


def mean_correlation(signal, x_pos, y_pos):
    """Compute the mean absolute Pearson correlation between detector pairs.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector TOD signals of shape ``(n_dets, n_samps)``.  Works
        better with detrended data.
    x_pos : numpy.ndarray
        X focal-plane positions (unused, accepted for a uniform interface).
    y_pos : numpy.ndarray
        Y focal-plane positions (unused, accepted for a uniform interface).

    Returns
    -------
    float
        Mean of the absolute Pearson correlation coefficients.
    """

    corr_coeff = np.full((len(signal), len(signal)), np.nan)

    for i in range(len(signal)):
            if len(signal[i]) >= 2:
                for j in range(len(signal)):
                    if j >= i:

                        #compute the Pearson correlation coefficient between detector pair
                        corr_t = ss.pearsonr(signal[i], signal[j])[0]
                        corr_coeff[i, j] = corr_t
                        # corr_coeff[j, i] = corr_t

    mean_corr = np.nanmean(np.abs(corr_coeff))

    return mean_corr


def max_and_near_y_pos_ratio(signal, x_pos, y_pos):
    """Ratio of detectors near the y-histogram peak to total detectors.

    Counts the detectors whose y-positions fall within 0.1 of the
    peak histogram bin and divides by the total number of detectors.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector signals (unused, accepted for a uniform interface).
    x_pos : numpy.ndarray
        X focal-plane positions (unused, accepted for a uniform interface).
    y_pos : numpy.ndarray
        Y focal-plane positions of affected detectors.

    Returns
    -------
    float
        Fraction of detectors within 0.1 of the y-histogram peak.
    """

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #find all bins within 0.1 of either side of the maximum bin
    ind_close_to_max = np.where(np.abs(np.histogram(y_pos)[1] - np.histogram(y_pos)[1][ind_y_max]) <= 0.1)[0]

    sum_close = np.sum(np.histogram(y_pos)[0][ind_close_to_max[:-1]])

    det_num = len(y_pos)

    return sum_close/det_num



def max_and_adjacent_y_pos_ratio(signal, x_pos, y_pos):
    """Ratio of detectors in the peak and adjacent y-histogram bins to total.

    Sums the counts in the peak bin and its immediate neighbours,
    then divides by the total number of detectors.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector signals (unused, accepted for a uniform interface).
    x_pos : numpy.ndarray
        X focal-plane positions (unused, accepted for a uniform interface).
    y_pos : numpy.ndarray
        Y focal-plane positions of affected detectors.

    Returns
    -------
    float
        Fraction of detectors in the peak and adjacent y-histogram bins.
    """

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #check if there are adjacent bins on either side of the maximun bin
    if ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1] + np.histogram(y_pos)[0][ind_y_max + 1]

    elif ind_y_max + 1 > len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1]

    elif ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 < 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max + 1]


    det_num = len(y_pos)

    return sum_near/det_num


def compute_num_peaks(signal, x_pos, y_pos):
    """Count the number of peaks in the combined detector TOD.

    Smooths the per-sample maximum across detectors, selects prominent
    samples, and counts the peaks detected by
    :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    signal : numpy.ndarray
        Detector TOD signals of shape ``(n_dets, n_samps)``.  Works
        better with detrended data.
    x_pos : numpy.ndarray
        X focal-plane positions (unused, accepted for a uniform interface).
    y_pos : numpy.ndarray
        Y focal-plane positions (unused, accepted for a uniform interface).

    Returns
    -------
    int
        Number of peaks found.
    """

    #make a smoothing kernel
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size

    #smooth the data
    max_vals_t = np.convolve(np.max(signal, axis = 0), kernel, mode='same')

    mean_vals_t = np.convolve(np.mean(signal, axis = 0), kernel, mode='same')

    std_vals_t = np.std(signal)

    vals_for_peaks = np.zeros(len(max_vals_t))

    #check if the max value is >= the mean *3std or else use the mean value
    for i in range(len(max_vals_t)):

        if max_vals_t[i] >= mean_vals_t[i] + 3*std_vals_t:
            vals_for_peaks[i] = max_vals_t[i]

        else:
            vals_for_peaks[i] = mean_vals_t[i]

    #find the peaks in the combined TOD
    prom = np.max([1e-12,  np.abs(np.mean(vals_for_peaks)) + 2.*np.mean(std_vals_t)])
    peaks_t = sp.signal.find_peaks(vals_for_peaks, prominence = prom)[0]

    num_peaks_t = len(peaks_t)

    return num_peaks_t
