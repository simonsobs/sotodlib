"""
Useful tools for statistical analyses
"""
from scipy.stats import skewnorm, binned_statistic
from scipy.interpolate import interp1d
import scipy.optimize as opt
import scipy.signal as signal
import numpy as np
from .. import core
from . import fft_ops


def noise_model(
    aman, Pxx, freqs, filter_b, filter_a, flux_ramp_freq, tf_fix, wl, n, f_knee
):
    """
    Crude noise model

    Arguments:
        aman: AxisManager

        Pxx: PSD

        freqs: Frequency information associated with the PSD

        filter_b: Numerator of linear filter

        filter_a: Denominator of linear filter

        flux_ramp_freq: The flux ramp frequency

        tf_fix: Function to calculates adjustment to transfer function, if None then function that always returns 1 is used

        wl: The  white noise level

        n: Exponent of 1/f^n component

        f_knee: Frequency at which white noise = 1/f^n component

    returns:
        noise: The computed noise model
    """
    if tf_fix is None:

        def tf_fix(freq):
            return 1

    A = wl * (f_knee ** n)

    # The downsample filter is at the flux ramp frequency
    w, h = signal.freqz(filter_b, filter_a, worN=freqs, fs=flux_ramp_freq)

    tf = np.absolute(h)  # filter transfer function

    return (A / (freqs ** n) + wl) * tf * tf_fix(freqs)


def fit_psd(
    aman,
    signal=None,
    timestamps=None,
    Pxx=None,
    freqs=None,
    noise_model=noise_model,
    bounds=None,
    p0=None,
    nm_args=[],
    **psd_args,
):
    """
    Fits PSD to noise model

    Arguments:
        aman: AxisManager with signal and timestamps

        signal: The signal to make PSD from, if None then aman.signal is used

        timestamps: The timestamps to mak PSD from, if None then aman.timestamps is used

        Pxx: The PSD to fit, if None then PSD is generated with fft_ops.calc_psd

        freqs: The freqs associated with the PSD if None then aman.freqs is used

        noise_model: The function used to fit the PSD.
            The signature of this function is assumed to be:
                noise_model(aman, Pxx, freqs, *nm_args, *fit_params)

        bounds: Bounds to pass to the fitting function

        p0: Initial guess to pass to the fitting function

        nm_args: Array containing the arguments to be passed to the noise_model that arent aman, Pxx, freqs, or the fitting parameters

        **psd_args: kwargs to be passed to fft_ops.calc_psd
    Returns:
        fit_params: Array containing the fit params of the fit noise_model
        fit_pxx: The fit of the PSD

        Pxx: The PSD

        freqs: The freqs associated with the PSD
    """
    if Pxx is None:
        freqs, Pxx = fft_ops.calc_psd(aman, signal, timestamps, **psd_args)
    if freqs is None:
        freqs = aman.freqs

    def _noise_model(x, *fit_params):
        return noise_model(aman, Pxx[1:], freqs[1:], *nm_args, *fit_params)

    if p0 is None:
        from inspect import signature

        p0 = np.ones(len(signature(noise_model).parameters) - 3 - len(nm_args))

    if bounds is None:
        from inspect import signature

        b = np.ones(len(signature(noise_model).parameters) - 3 - len(nm_args))
        bounds = (-np.inf * b, np.inf * b)

    popt, pcov = opt.curve_fit(_noise_model, 0, Pxx[1:], bounds=bounds, p0=p0)

    return popt, _noise_model(0, *popt), Pxx, freqs


def fit_hist(
    aman, signal=None, hist=None, bins=None, skew=True, **hist_params
):
    """
    Fit gaussian function to histogram of some data

    Arguments:
        aman: AxisManager with data

        signal: The data to make histogram from, if None then aman.signal is used 

        hist: The histogram to fit, if None then histogram of data is used

        bins: The histogram bin edges expected length is (len(hist)+1), if None is provided but hist is provided then range(len(hist)+1) is used

        skew: If True fit with skewgauss, if False fit with gauss. Defaults to True
    Returns:
        fit_params: Array containing the following fit params:
           mean: mean of distribution

           sigma: sd of distribution

           A: amplitude by which to scale distribution

           sk: skew parameter.  When sk is 0, this becomes a normal distribution. Only returned if skew is True.

        hist: The histogram that was fit

        bins: The bins of the histogram
    """
    if hist is None:
        if signal is None:
            signal = getattr(aman, field)
        hist, _bins = np.histogram(signal, **hist_params)
    if bins is None:
        bins = range(len(hist) + 1)
    bins = (_bins[:-1] + _bins[1:]) / 2

    if skew:
        popt, pcov = opt.curve_fit(skewgauss, bins, hist)
    else:
        popt, pcov = opt.curve_fit(gauss, bins, hist)

    return popt, hist, _bins


def fit_sine(aman, signal=None, timestamps=None):
    """
    Fit sine function to some data, if data is 2d then each row is fit individually

    Arguments:
        aman: AxisManager with data and timestamps

        signal: The data to fit, if None then aman.signal is used/ 

        timestamps: The timestamps to fit against, if None than aman.timestamps is used

    Returns:
        fit_params: 2d array where each row contains the following fit params:
            A: Amplitude of the fit sine function

            freq: Frequency of the fit sine function

            phase: Phase of the fit sine function

            offset: Offset to add to the fit sine function
    """
    if signal is None:
        data = amn.signal 
    if timestamps is None:
        timestamps = aman.timestamps
    if len(signal.shape) == 1:
        signal = np.array([signal])

    fit_params = np.zeros((len(signal), 4))
    for i, dat in enumerate(signal):
        popt, pcov = opt.curve_fit(sine, timestamps, dat)
        fit_params[i] = popt

    return fit_params


def sine(timestamps, A, freq, phase, offset):
    """
    Sine function to fit against

    Arguments:

        timestamps: times to be input into the sine function

        A: Amplitude of sine function

        freq: Frequency of sine function

        phase: Phase of sine function

        offset: Offset to add to sine function

    Returns:

        sine: the values of the sine function at each timestamp
    """
    return A * np.sin(freq * timestamps + phase) + offset


def gauss(x, mean, sigma, A):
    """
    Gaussian curve defined mean, sigma, and scaled by an amplitude A.
    Note that maximum value!= A; rather the area under the curve == A.

    Arguments:

        x: value or array of points

        mean: mean of distribution

        sigma: sd of distribution

        A: amplitude by which to scale distribution

    Returns:

                y: The value(s) of the curve at x
    """
    return (
        A * np.exp(-(((x - mean) / sigma) ** 2) / 2) / np.sqrt(2 * np.pi * sigma ** 2)
    )


def skewgauss(x, mean, sigma, A, sk):
    """
    Skewgaussian/skewnormal curve defined mean, sigma, and skew parameter, and scaled by an amplitude A.
    Note that maximum value!= A; rather the area under the curve == A.

    Arguments:

        x: value or array of points

        mean: mean of distribution

        sigma: sd of distribution

        A: amplitude by which to scale distribution

        sk: skew parameter.  When sk is 0, this becomes a normal distribution.

    Returns:

                y: The value(s) of the curve at x
    """
    return A * skewnorm.pdf((x - mean) / sigma, sk)


def average_to(
    aman, DT, signal=None, timestamps=None, tmin=None, tmax=None, append_aman=False
):
    """
    Bins and averages input signal and timestamps to a new (regular) sample rate.
    This is most useful for downsampling (paricularly non-regularly sapmled) data.

    Arguments:

        aman: AxisManager with timestamps and data to resample.

        DT: Time separation of new sample rate (in units of input timestamps); this is 1/sample-frequency.

        signal: 1- or 2-d array of data to resample. If None, uses aman.signal.

        timestamps: 1-d array of timestamps. If None, uses aman.timestamps.

        tmin: desired start time for resampled data. If None, uses beginning of timestamps.

        tmax: desired stop time for resampled data. If None, uses end of timestamps.

        append_aman: return AxisManager with resampled time and data appended to the provided aman. If aman=None this returns None.

    Returns:

                t: resampled timestamps

                d: resampled data -- will have nans in bins where there was no input data

                aman: aman containing resampled timestamps and data. The new sample axis will be called avgDT_samps, the resampled timestamps will be called avgDT_timestamps, and the resampled signal will be called avgDT_signal (where DT is the provided time seperation of the new sample rate). This is only returned if append_aman is True.
    """
    if signal is None:
        signal = aman.signal
    if timestamps is None:
        timestamps = aman.timestamps

    if tmin == None:
        tmin = timestamps[0]
    if tmax == None:
        tmax = timestamps[-1]

    bins = np.arange(tmin - DT / 2, tmax + DT / 2, DT)
    d, bins, _ = binned_statistic(timestamps, signal, bins=bins)
    t = (bins[1:] + bins[:-1]) / 2

    if append_aman:
        if aman is None:
            return t, d, None
        aman = core.AxisManager(aman, core.OffsetAxis(f"avg{DT}_samps", len(t)))

        aman.wrap(f"avg{DT}_timestamps", t, [(0, f"avg{DT}_samps")])
        aman.wrap(f"avg{DT}_signal", d, [(0, "dets"), (1, f"avg{DT}_samps")])
        return t, d, aman

    return t, d


def interpolate_nans(
    aman,
    signal=None,
    timestamps=None,
):
    """
    Interpolate over nans in place

    Arguments:

        aman: AxisManager with timestamps and data to resample.

        signal: 1- or 2-d array of data to interpolate over. If None, uses aman.signal.

        timestamps: 1-d array of timestamps. If None, uses aman.timestamps. Cannot have nans present.
            If you want to interpolate over timestamps pass them as the signal argument and set this argument to something like np.arange(len(timestamps)

    """
    if signal is None:
        signal = aman.signal 
    if timestamps is None:
        timestamps = aman.timestamps 

    def _interpolate_nans(y, x):
        if len(x) == 0:
            return
        if len(x) != len(y):  # or
            return
        nans = np.isnan(y)
        if any(np.isnan(x)) or all(nans) or not any(nans):
            return
        nansi = np.where(nans)[0]
        goodi = np.where(~nans)[0]
        interp = interp1d(x[goodi], y[goodi], fill_value="extrapolate")
        y[nansi] = interp(nansi)

    np.apply_along_axis(_interpolate_nans, -1, signal, timestamps)
