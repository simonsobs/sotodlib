"""FFTs and related operations
"""
import sys
import numdifftools as ndt
import numpy as np
import pyfftw
import so3g
from scipy.optimize import minimize
from scipy.signal import welch
from sotodlib import core

from . import detrend_tod

def _get_num_threads():
    # Guess how many threads we should be using in FFT ops...
    return so3g.useful_info().get("omp_num_threads", 4)


def rfft(
    aman,
    detrend="linear",
    resize="zero_pad",
    window=np.hanning,
    axis_name="samps",
    signal_name="signal",
    delta_t=None,
):
    """Return the real fft of aman.signal_name along the axis axis_name.
        Does not change the data in the axis manager.

    Arguments:

        aman: axis manager

        detrend: Method of detrending to be done before ffting. Can
            be 'linear', 'mean', or None. Note that detrending here can be slow
            for large arrays.

        resize: How to resize the axis to increase fft speed. 'zero_pad'
            will increase to the next 2**N. 'trim' will cut out so the
            factorization of N contains only low primes. None will not
            change the axis length and might be quite slow.

        window: a function that takes N are returns an fft window
            Can be None if no windowing

        axis_name: name of axis you would like to fft along

        signal_name: name of the variable in aman to fft

        delta_t: if none, it will look for 'timestamps' in the axis manager
                and will otherwise assume 1. if not None, it should be the
                sampling rate along the axis you're ffting

    Returns:

        fft: the fft'd data

        freqs: the frequencies it is value at (since resizing is an option)
    """
    if len(aman._assignments[signal_name]) > 2:
        raise ValueError("rfft only works for 1D or 2D data streams")

    axis = getattr(aman, axis_name)

    if len(aman._assignments[signal_name]) == 1:
        n_det = 1
        main_idx = 0
        other_idx = None

    elif len(aman._assignments[signal_name]) == 2:
        checks = np.array(
            [x == axis_name for x in aman._assignments[signal_name]], dtype="bool"
        )
        main_idx = np.where(checks)[0][0]
        other_idx = np.where(~checks)[0][0]
        other_axis = getattr(aman, aman._assignments[signal_name][other_idx])
        n_det = other_axis.count

    if detrend is None:
        signal = np.atleast_2d(getattr(aman, signal_name))
    else:
        signal = detrend_tod(
            aman, detrend, axis_name=axis_name, signal_name=signal_name, in_place=True
        )

    if other_idx is not None and other_idx != 0:
        signal = signal.transpose()

    if window is not None:
        signal = signal * window(axis.count)[None, :]

    if resize == "zero_pad":
        k = int(np.ceil(np.log(axis.count) / np.log(2)))
        n = 2**k
    elif resize == "trim":
        n = find_inferior_integer(axis.count)
    elif resize is None:
        n = axis.count
    else:
        raise ValueError('resize must be "zero_pad", "trim", or None')

    a, b, t_fun = build_rfft_object(n_det, n, "FFTW_FORWARD")
    if resize == "zero_pad":
        a[:, : axis.count] = signal
        a[:, axis.count :] = 0
    elif resize == "trim":
        a[:] = signal[:, :n]
    else:
        a[:] = signal[:]

    t_fun()

    if delta_t is None:
        if "timestamps" in aman:
            delta_t = (aman.timestamps[-1] - aman.timestamps[0]) / axis.count
        else:
            delta_t = 1
    freqs = np.fft.rfftfreq(n, delta_t)

    if other_idx is not None and other_idx != 0:
        return b.transpose(), freqs

    return b, freqs


def build_rfft_object(n_det, n, direction="FFTW_FORWARD", **kwargs):
    """Build PyFFTW object for fft-ing

    Arguments:

        n_det: number of detectors (or just the arr.shape[0] for the
            array you are going to fft)

        n: number of samples in timestream

        direction: fft direction. Can be FFTW_FORWARD, FFTW_BACKWARD, or BOTH

        kwargs: additional arguments to pass to pyfftw.FFTW

    Returns:

        a: array for the real valued side of the fft

        b: array for the the complex side of the fft

        t_fun: function for performing FFT (two are returned if direction=='BOTH')
    """
    fftargs = {"threads": _get_num_threads(), "flags": ["FFTW_ESTIMATE"]}
    fftargs.update(kwargs)

    a = pyfftw.empty_aligned((n_det, n), dtype="float32")
    b = pyfftw.empty_aligned((n_det, (n + 2) // 2), dtype="complex64")
    
    if direction == "FFTW_FORWARD":
        t_fun = pyfftw.FFTW(a, b, direction=direction, **fftargs)
    elif direction == "FFTW_BACKWARD":
        t_fun = pyfftw.FFTW(b, a, direction=direction, **fftargs)
    elif direction == "BOTH":
        t_1 = pyfftw.FFTW(a, b, direction="FFTW_FORWARD", **fftargs)
        t_2 = pyfftw.FFTW(b, a, direction="FFTW_BACKWARD", **fftargs)
        return a, b, t_1, t_2
    else:
        raise ValueError("direction must be FFTW_FORWARD or FFTW_BACKWARD")

    return a, b, t_fun


def find_inferior_integer(target, primes=[2, 3, 5, 7, 11, 13]):
    """Find the largest integer less than or equal to target whose prime
    factorization contains only the integers listed in primes.

    """
    p = primes[0]
    n = np.floor(np.log(target) / np.log(p))
    best = p**n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p**n
        best_friend = find_inferior_integer(target / base, primes[1:])
        if (best_friend * base) >= best:
            best = best_friend * base
    return int(best)


def find_superior_integer(target, primes=[2, 3, 5, 7, 11, 13]):
    """Find the smallest integer less than or equal to target whose prime
    factorization contains only the integers listed in primes.

    """
    p = primes[0]
    n = np.ceil(np.log(target) / np.log(p))
    best = p**n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p**n
        best_friend = find_superior_integer(target / base, primes[1:])
        if (best_friend * base) <= best:
            best = best_friend * base
    return int(best)


def calc_psd(
    aman,
    signal=None,
    timestamps=None,
    max_samples=2**18,
    prefer='center',
    freq_spacing=None,
    merge=False, 
    overwrite=True,
    subscan=False,
    **kwargs
):
    """Calculates the power spectrum density of an input signal using signal.welch().
    Data defaults to aman.signal and times defaults to aman.timestamps.
    By default the nperseg will be set to power of 2 closest to the 1/50th of
    the samples used, this can be overridden by providing nperseg or freq_spacing.

    Arguments:
        aman (AxisManager): with (dets, samps) OR (channels, samps)axes.
        signal (float ndarray): data signal to pass to scipy.signal.welch().
        timestamps (float ndarray): timestamps associated with the data signal.
        max_samples (int): maximum samples along sample axis to send to welch.
        prefer (str): One of ['left', 'right', 'center'], indicating what
            part of the array we would like to send to welch if cuts are
            required.
        freq_spacing (float): The approximate desired frequency spacing of the PSD.
            If None the default nperseg of ~1/50th the signal length is used.
            If an nperseg is explicitly passed then that will be used.
        merge (bool): if True merge results into axismanager.
        overwrite (bool): if true will overwrite f, pxx axes.
        subscan (bool): if True, compute psd on subscans.
        **kwargs: keyword args to be passed to signal.welch().

    Returns:
        freqs: array of frequencies corresponding to PSD calculated from welch.
        Pxx: array of PSD values.
    """
    if signal is None:
        signal = aman.signal
    if subscan:
        freqs, Pxx = _calc_psd_subscan(aman, signal=signal, freq_spacing=freq_spacing, **kwargs)
        axis_map_pxx = [(0, "dets"), (1, "nusamps"), (2, "subscans")]
    else:
        if timestamps is None:
            timestamps = aman.timestamps

        n_samps = signal.shape[-1]
        if n_samps <= max_samples:
            start = 0
            stop = n_samps
        else:
            offset = n_samps - max_samples
            if prefer == "left":
                offset = 0
            elif prefer == "center":
                offset //= 2
            elif prefer == "right":
                pass
            else:
                raise ValueError(f"Invalid choice prefer='{prefer}'")
            start = offset
            stop = offset + max_samples
        fs = 1 / np.nanmedian(np.diff(timestamps[start:stop]))
        if "nperseg" not in kwargs:
            if freq_spacing is not None:
                nperseg = int(2 ** (np.around(np.log2(fs / freq_spacing))))
            else:
                nperseg = int(2 ** (np.around(np.log2((stop - start) / 50.0))))
            kwargs["nperseg"] = nperseg

        freqs, Pxx = welch(signal[:, start:stop], fs, **kwargs)
        axis_map_pxx = [(0, aman.dets), (1, "nusamps")]

    if merge:
        aman.merge( core.AxisManager(core.OffsetAxis("nusamps", len(freqs))))
        if overwrite:
            if "freqs" in aman._fields:
                aman.move("freqs", None)
            if "Pxx" in aman._fields:
                aman.move("Pxx", None)
        aman.wrap("freqs", freqs, [(0,"nusamps")])
        aman.wrap("Pxx", Pxx, axis_map_pxx)
    return freqs, Pxx

def _calc_psd_subscan(aman, signal=None, freq_spacing=None, **kwargs):
    """
    Calculate the power spectrum density of subscans using signal.welch().
    Data defaults to aman.signal. aman.timestamps is used for times.
    aman.subscan_info is used to identify subscans.
    See calc_psd for arguments.
    """
    from .flags import get_subscan_signal
    if signal is None:
        signal = aman.signal

    fs = 1 / np.nanmedian(np.diff(aman.timestamps))
    if "nperseg" not in kwargs:
        if freq_spacing is not None:
            nperseg = int(2 ** (np.around(np.log2(fs / freq_spacing))))
        else:
            duration_samps = np.asarray([np.ptp(x.ranges()) if x.ranges().size > 0 else 0 for x in aman.subscan_info.subscan_flags])
            duration_samps = duration_samps[duration_samps > 0]
            nperseg = int(2 ** (np.around(np.log2(np.median(duration_samps) / 4))))
        kwargs["nperseg"] = nperseg

    Pxx = []
    for iss in range(aman.subscan_info.subscans.count):
        signal_ss = get_subscan_signal(aman, signal, iss)
        axis = -1 if "axis" not in kwargs else kwargs["axis"]
        if signal_ss.shape[axis] >= kwargs["nperseg"]:
            freqs, pxx_sub = welch(signal_ss, fs, **kwargs)
            Pxx.append(pxx_sub)
        else:
            Pxx.append(np.full((signal.shape[0], kwargs["nperseg"]//2+1), np.nan)) # Add nans if subscan is too short
    Pxx = np.array(Pxx)
    Pxx = Pxx.transpose(1, 2, 0) # Dets, nusamps, subscans
    return freqs, Pxx

def calc_wn(aman, pxx=None, freqs=None, low_f=5, high_f=10, lowf_fk=1):
    """
    Function that calculates the white noise level as a median PSD value between
    two frequencies. Defaults to calculation of white noise between 5 and 10Hz.
    Defaults frequency information to a wrapped "freqs" field in aman.

    Arguments
    ---------
        aman (AxisManager):
            Uses aman.freq as frequency information associated with the PSD, pxx.

        pxx (Float array):
            Psd information to calculate white noise. Defaults to aman.pxx

        freqs (1d Float array):
            frequency information related to the psd. Defaults to aman.freqs

        low_f (Float):
            low frequency cutoff to calculate median psd value. Defaults to 5Hz

        high_f (float):
            high frequency cutoff to calculate median psd value. Defaults to 10Hz

    Returns
    -------
        wn: Float array of white noise levels for each psd passed into argument.
    """
    if freqs is None:
        freqs = aman.freqs

    if pxx is None:
        pxx = aman.Pxx

    fmsk = np.all([freqs >= low_f, freqs <= high_f], axis=0)
    if pxx.ndim == 1:
        wn2 = np.median(pxx[fmsk])
    else:
        wn2 = np.median(pxx[:, fmsk], axis=1)

    wn = np.sqrt(wn2)

    return wn


def noise_model(f, p):
    """
    Noise model for power spectrum with white noise, and 1/f noise.
    """
    fknee, w, alpha = p[0], p[1], p[2]
    return w * (1 + (fknee / f) ** alpha)


def neglnlike(params, x, y):
    model = noise_model(x, params)
    output = np.sum(np.log(model) + y / model)
    if not np.isfinite(output):
        return 1.0e30
    return output


def fit_noise_model(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs={},
    fwhite=(10, 100),
    lowf=1,
    merge_fit=False,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    freq_spacing=None,
    subscan=False
):
    """
    Fits noise model with white and 1/f noise to the PSD of signal.
    This uses a MLE method that minimizes a log likelihood. This is
    better for chi^2 distributed data like the PSD.

    Reference: http://keatonb.github.io/archivers/powerspectrumfits

    Args
    ----
    aman : AxisManager
        Axis manager which has samps axis aligned with signal.
    signal : nparray
        Signal sized ndets x nsamps to fit noise model to.
        Default is None which corresponds to aman.signal.
    f : nparray
        Frequency of PSD of signal.
        Default is None which calculates f, pxx from signal.
    pxx : nparray
        PSD sized ndets x len(f) which is fit to with model.
        Default is None which calculates f, pxx from signal.
    psdargs : dict
        Dictionary of optional argument for ``scipy.signal.welch``
    fwhite : tuple
        Low and high frequency used to estimate white noise for initial
        guess passed to ``scipy.signal.curve_fit``.
    lowf : tuple
        Frequency below which estimate of 1/f noise index and knee are estimated
        for initial guess passed to ``scipy.signal.curve_fit``.
    merge_fit : bool
        Merges fit and fit statistics into input axis manager.
    f_max : float
        Maximum frequency to include in the fitting. This is particularly
        important for lowpass filtered data such as that post demodulation
        if the data is not downsampled after lowpass filtering.
    merge_name : bool
        If ``merge_fit`` is True then addes into axis manager with merge_name.
    merge_psd : bool
        If ``merg_psd`` is True then adds fres and Pxx to the axis manager.
    freq_spacing : float
        The approximate desired frequency spacing of the PSD. Passed to calc_psd.
    Returns
    -------
    noise_fit_stats : AxisManager
        If merge_fit is False then axis manager with fit and fit statistics
        is returned otherwise nothing is returned and axis manager is wrapped
        into input aman.
    """
    if signal is None:
        signal = aman.signal

    if f is None or pxx is None:
        f, pxx = calc_psd(
            aman,
            signal=signal,
            timestamps=aman.timestamps,
            freq_spacing=freq_spacing,
            merge=merge_psd,
            subscan=subscan,
            **psdargs,
        )
    if subscan:
       fitout, covout = _fit_noise_model_subscan(aman, signal,  f, pxx, psdargs=psdargs,
                                                  fwhite=fwhite, lowf=lowf, f_max=f_max,
                                                  freq_spacing=freq_spacing)
       axis_map_fit = [(0, "dets"), (1, "noise_model_coeffs"), (2, aman.subscans)]
       axis_map_cov = [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs"), (3, aman.subscans)]
    else:
        eix = np.argmin(np.abs(f - f_max))
        f = f[1:eix]
        pxx = pxx[:, 1:eix]

        fitout = np.zeros((aman.dets.count, 3))
        # This is equal to np.sqrt(np.diag(cov)) when doing curve_fit
        covout = np.zeros((aman.dets.count, 3, 3))
        for i in range(aman.dets.count):
            p = pxx[i]
            wnest = np.median(p[((f > fwhite[0]) & (f < fwhite[1]))])
            pfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1)
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], wnest, -pfit[0]]
            bounds = [(0, None), (sys.float_info.min, None), (None, None)]
            res = minimize(neglnlike, p0, args=(f, p), bounds=bounds, method="Nelder-Mead")
            try:
                Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p), full_output=True)
                hessian_ndt, _ = Hfun(res["x"])
                # Inverse of the hessian is an estimator of the covariance matrix
                # sqrt of the diagonals gives you the standard errors.
                covout[i] = np.linalg.inv(hessian_ndt)
            except np.linalg.LinAlgError:
                covout[i] = np.full((3, 3), np.nan)
            fitout[i] = res.x
        axis_map_fit = [(0, "dets"), (1, "noise_model_coeffs")]
        axis_map_cov = [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")]

    noise_model_coeffs = ["fknee", "white_noise", "alpha"]
    noise_fit_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(
            name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U8")
        ),
    )
    noise_fit_stats.wrap("fit", fitout, axis_map_fit)
    noise_fit_stats.wrap("cov", covout, axis_map_cov)

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats


def _fit_noise_model_subscan(
    aman,
    signal,
    f,
    pxx,
    psdargs={},
    fwhite=(10, 100),
    lowf=1,
    f_max=100,
    freq_spacing=None,
):
    """
    Fits noise model with white and 1/f noise to the PSD of signal subscans.
    Args are as for fit_noise_model.
    """
    fitout = np.empty((aman.dets.count, 3, aman.subscan_info.subscans.count))
    covout = np.empty((aman.dets.count, 3, 3, aman.subscan_info.subscans.count))

    for isub in range(aman.subscan_info.subscans.count):
        if np.all(np.isnan(pxx[...,isub])): # Subscan has been fully cut
            fitout[..., isub] = np.full((aman.dets.count, 3), np.nan)
            covout[..., isub] = np.full((aman.dets.count, 3, 3), np.nan)
        else:
            noise_model = fit_noise_model(aman, f=f, pxx=pxx[...,isub], fwhite=fwhite, lowf=lowf, merge_fit=False,
                                          f_max=f_max, merge_psd=False, subscan=False)

            fitout[..., isub] = noise_model.fit
            covout[..., isub] = noise_model.cov

    return fitout, covout


def build_hpf_params_dict(
    filter_name,
    noise_fit=None,
    filter_params=None
):
    """
    Build the filter parameter dictionary from a provided
    dictionary or from noise fit results.

    Args
    ----
    filter_name : str
        Name of the filter to build the parameter dict for.
    noise_fit: AxisManager
        AxisManager containing the result of the noise model fit sized nparams x ndets.
    filter_params: dict
        Filter parameters dictionary to complement parameters
        derived from the noise fit (or to be used if noise fit is None).
    Returns
    -------
    filter_params : dict
        Returns a dictionary of the median values of the noise model fit parameters
        if noise_fit is not None, otherwise return the provided filter_params.
    """
    if noise_fit is not None:

        pars_mapping = {
            "high_pass_butter4": {
                "fc": "fknee",
            },
            "counter_1_over_f": {
                "fk": "fknee", 
                "n": "alpha"
            },
            "high_pass_sine2": {
                "cutoff": "fknee",
                "width": None
            }
        }

        if filter_name not in pars_mapping.keys():
            raise NotImplementedError(
                f"{filter_name} params from noise fit is not implemented"
            )
        
        noise_fit_array = noise_fit.fit
        noise_fit_params = noise_fit.noise_model_coeffs.vals
        
        median_params = np.median(noise_fit_array, axis=0)
        median_dict = {
            k: median_params[i]
            for i, k in enumerate(noise_fit_params)
        }

        params_dict = {}
        for k, v in pars_mapping[filter_name].items():
            if v is None:
                if (filter_params is None) or (k not in filter_params):
                    raise ValueError(
                        f"Required parameters {k} not found in config "
                         "and cannot be derived from noise fit."
                    )
                else:
                    params_dict.update({k: filter_params[k]})
            else:
                params_dict[k] = median_dict[v]

        filter_params = params_dict
    
    return filter_params
        
