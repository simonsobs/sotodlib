"""FFTs and related operations
"""
import numdifftools as ndt
import numpy as np
import pyfftw
import so3g
from so3g.proj import Ranges, RangesMatrix
from scipy.optimize import minimize
from scipy.signal import welch
from sotodlib import core, hwp
from sotodlib.tod_ops import detrend_tod


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
    merge_suffix=None,
    overwrite=True, 
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
        merge_suffix (str, optional): Suffix to append to the Pxx field name in aman. Defaults to None (merged as Pxx).
        overwrite (bool): if true will overwrite f, Pxx axes.
        **kwargs: keyword args to be passed to signal.welch().

    Returns:
        freqs: array of frequencies corresponding to PSD calculated from welch.
        Pxx: array of PSD values.
    """
    if signal is None:
        signal = aman.signal
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
            raise ValueError(f"Invalid choise prefer='{prefer}'")
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
    if merge:
        if 'nusamps' not in list(aman._axes.keys()):
            aman.merge(core.AxisManager(core.OffsetAxis("nusamps", len(freqs))))
            aman.wrap("freqs", freqs, [(0,"nusamps")])
        else:
            if len(freqs) != aman.nusamps.count:
                raise ValueError('New freqs does not match the shape of nusamps\
                                To avoid this, use the same value for nperseg')
        
        if merge_suffix is None:
            Pxx_name = 'Pxx'
        else:
            Pxx_name = f'Pxx_{merge_suffix}'
        
        if overwrite:
            if Pxx_name in aman._fields:
                aman.move(Pxx_name, None)
        aman.wrap(Pxx_name, Pxx, [(0,"dets"), (1,"nusamps")])
    return freqs, Pxx


def calc_wn(aman, pxx=None, freqs=None, low_f=5, high_f=10):
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


def noise_model(f, params, **fixed_param):
    """
    Noise model for power spectrum with white noise, and 1/f noise.
    If any fixed param is handed, that parameter is fixed in the fit.
    'alpha' or 'wn' can be fixed.
    """
    if 'alpha' in fixed_param.keys():
        if len(params)==2:
            alpha = fixed_param['alpha']
            fknee, wn = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
            return
    elif 'wn' in fixed_param.keys():
        if len(params)==2:
            wn = fixed_param['wn']
            fknee, alpha = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
            return
    elif len(fixed_param)==0:
        if len(params)==3:
            fknee, wn, alpha = params[0], params[1], params[2]
        else:
            raise ValueError('The number of fit parameters are invalid.')
            return
    else:
        raise ValueError('"alpha" or "wn" can be a fixed parameter.')
        return
    return wn * (1 + (fknee / f) ** alpha)


def neglnlike(params, x, y, bin_size=1, **fixed_param):
    model = noise_model(x, params, **fixed_param)
    output = np.sum((np.log(model) + y / model)*bin_size)
    if not np.isfinite(output):
        return 1.0e30
    return output

def calc_psd_mask(aman, psd_mask=None, f=None, pxx=None,
                mask_hwpss=True, hwp_freq=None, max_hwpss_mode=10, hwpss_width=((-0.4, 0.6), (-0.2, 0.2)),
                mask_peak=False, peak_freq=None, peak_width=(-0.002, +0.002),
                merge=True, overwrite=True
):
    """
    Function that calculates masks for hwpss or single peak in PSD.

    Arguments
    ---------
        aman : AxisManager
            Axis manager which has 'nusamps' axis.
        psd_mask : numpy.ndarray or Ranges
            Existing psd_mask to be updated. If None, a new mask is created.
        f : nparray
            Frequency of PSD of signal. If None, aman.freqs are used.
        pxx : nparray
            PSD of signal. If None, aman.Pxx are used.
        mask_hwpss : bool
            If True, hwpss are masked. Defaults to True.
        hwp_freq : float
            HWP frequency. If None, calculated based on aman.hwp_angle
        max_hwpss_mode : int
            Maximum hwpss mode to subtract. 
        hwpss_width : array-like
            If given in float, 
            nf-hwpss will be masked like (n * hwp_freq - width/2) < f < (n * hwp_freq + width/2).
            If given in array like [1.0, 0.4], 
            1f hwpss is masked like (hwp_freq - 0.5) < f < (hwp_freq + 0.5) and 
            nf hwpss are masked like (n * hwp_freq - 0.2) < f < (n * hwp_freq + 0.2).
            If given in array like [[-0.4, 0.6], [-0.2, 0.3]],
            1f hwpss is masked like (hwp_freq - 0.4) < f < (hwp_freq + 0.6) and
            nf are masked like (n * hwp_freq - 0.2) < f < (n * hwp_freq + 0.3).
        mask_peak : bool
            If True, single peak is masked.
        peak_freq : float
            Center frequency of the mask.
        peak_width : tuple
            Range to mask signal. The default masked range will be 
            (peak_freq - 0.002) < f < (peak_freq + 0.002).
        merge : bool
            if True merge results into axismanager.
        mode: str
            if "replace", existing PSD mask is replaced to new mask.
            If "add", new mask range is added to the existing one.
        overwrite: bool
            if true will overwrite aman.psd_mask.

    Returns
    -------
        psd_mask (nusamps): Ranges array. If merge == True, "psd_mask" is added to the aman.  
    """
    if psd_mask is None:
        psd_mask = np.zeros(aman.nusamps.count, dtype=bool)
    elif isinstance(psd_mask, so3g.RangesInt32):
        psd_mask = psd_mask.mask()
    if f is None:
        f = aman.freqs
    if pxx is None:
        pxx = aman.Pxx
    if mask_hwpss:
        hwp_freq = hwp.get_hwp_freq(aman.timestamps, aman.hwp_solution.hwp_angle)
        psd_mask = psd_mask | get_mask_for_hwpss(f, hwp_freq, max_mode=max_hwpss_mode, width=hwpss_width)
    if mask_peak:
        psd_mask = psd_mask | get_mask_for_single_peak(f, peak_freq, peak_width=peak_width)

    psd_mask = Ranges.from_bitmask(psd_mask)
    if merge:
        if overwrite:
            if "psd_mask" in aman:
                aman.move("psd_mask", None)
        aman.wrap("psd_mask", psd_mask, [(0,"nusamps")])
    return psd_mask

def calc_binned_psd(
    aman,
    f=None,
    pxx=None,
    unbinned_mode=3,
    base=1.2,
    merge=False, 
    overwrite=True,
):
    """
    Function that masks hwpss in PSD.

    Arguments
    ---------
        aman : AxisManager
            Axis manager which has samps axis aligned with signal.
        f : nparray
            Frequency of PSD of signal.
        pxx : nparray
            PSD of signal.
        mask : bool
            if True calculate binned psd with mask.
        unbinned_mode : int
            First Fourier modes up to this number are left un-binned.
        base : float (> 1)
            Base of the logspace bins. 
        merge : bool
            if True merge results into axismanager.
        overwrite: bool
            if true will overwrite f, pxx axes.

    Returns
    -------
        f_binned, pxx_binned, bin_size: binned frequency and PSD.
    """
    if f is None:
        f = aman.freqs
    if pxx is None:
        pxx = aman.Pxx
            
    f_bin, bin_size = binning_psd(f, unbinned_mode=unbinned_mode, base=base, return_bin_size=True)
    pxx_bin = binning_psd(pxx, unbinned_mode=unbinned_mode, base=base, return_bin_size=False)

    if merge:
        aman.merge(core.AxisManager(core.OffsetAxis("nusamps_bin", len(f_bin))))
        if overwrite:
            if "freqs_bin" in aman._fields:
                aman.move("freqs_bin", None)
            if "Pxx_bin" in aman._fields:
                aman.move("Pxx_bin", None)
            if "bin_size" in aman._fields:
                aman.move("bin_size", None)
        aman.wrap("freqs_bin", f_bin, [(0,"nusamps_bin")])
        aman.wrap("Pxx_bin", pxx_bin, [(0,"dets"),(1,"nusamps_bin")])
        aman.wrap("bin_size", bin_size, [(0,"dets"),(1,"nusamps_bin")])
    return f_bin, pxx_bin, bin_size

def fit_noise_model(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs=None,
    fwhite=(10, 100),
    lowf=1,
    merge_fit=False,
    f_min=None,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    mask=False,
    fixed_parameter=None,
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
    f_min : float
        Minimum frequency to include in the fitting.
        Default is None which selects f_min as the second index of f.
    f_max : float
        Maximum frequency to include in the fitting. This is particularly
        important for lowpass filtered data such as that post demodulation
        if the data is not downsampled after lowpass filtering.
    merge_name : bool
        If ``merge_fit`` is True then addes into axis manager with merge_name.
    merge_psd : bool
        If ``merge_psd`` is True then adds fres and Pxx to the axis manager.
    mask : bool
        If ``mask`` is True then PSD is masked with ``aman.psd_mask``.
    fixed_parameter : str
        This accepts None or 'wn' or 'alpha'. If 'wn' ('alpha') is given, 
        white noise level (alpha) is fixed to the initially estimated value.
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
        if psdargs is None:
            f, pxx = calc_psd(
                aman, signal=signal, timestamps=aman.timestamps, merge=merge_psd
            )
        else:
            f, pxx = calc_psd(
                aman,
                signal=signal,
                timestamps=aman.timestamps,
                merge=merge_psd,
                **psdargs,
            )
    if mask:
        if 'psd_mask' in aman:
            mask = ~aman.psd_mask.mask()
            f = f[mask]
            pxx = pxx[:, mask]
        else:
            print('"psd_mask" is not in aman. Masking is skipped.')

    eix = np.argmin(np.abs(f - f_max))
    if f_min is None:
        six = 1
    elif f_min < f[0]:
        six = 1
    else:
        six = np.argmin(np.abs(f - f_min))
    f = f[six:eix]
    pxx = pxx[:, six:eix]
    fitout = np.zeros((aman.dets.count, 3))
    # This is equal to np.sqrt(np.diag(cov)) when doing curve_fit
    covout = np.zeros((aman.dets.count, 3, 3))
    fixed_param = {}
    for i in range(aman.dets.count):
        p = pxx[i]
        wnest = np.median(p[((f > fwhite[0]) & (f < fwhite[1]))])*1.5 #conversion factor from median to mean
        if fixed_parameter == None:
            try:
                pfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], wnest, -pfit[0]]
        elif fixed_parameter == 'wn':
            try:
                pfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], -pfit[0]]
            fixed_param['wn'] = wnest
        elif fixed_parameter == 'alpha':
            try:
                pfit, vfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1, cov=True)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], wnest]
            fixed_param['alpha'] = -pfit[0]
        else:
            print('"fixed_parameter" is invalid.')
            return
        res = minimize(lambda params: neglnlike(params, f, p, **fixed_param), 
               p0, method="Nelder-Mead")
        try:
            #Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p), full_output=True)
            Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p, **fixed_param), full_output=True)
            hessian_ndt, _ = Hfun(res["x"])
            # Inverse of the hessian is an estimator of the covariance matrix
            # sqrt of the diagonals gives you the standard errors.
            covout_i = np.linalg.inv(hessian_ndt)            
        except np.linalg.LinAlgError:
            print(
                f"Cannot calculate Hessian for detector {aman.dets.vals[i]} skipping. (LinAlgError)"
            )
            covout_i = np.full((len(p0), len(p0)), np.nan)
        except IndexError:
            print(
                f"Cannot calculate Hessian for detector {aman.dets.vals[i]} skipping. (IndexError)"
            )
            covout_i = np.full((len(p0), len(p0)), np.nan)
        fitout_i = res.x
        if fixed_parameter == 'alpha':
            alpha_err = np.sqrt(vfit[0][0])
            covout_i = np.insert(covout_i, 2, 0, axis=0)
            covout_i = np.insert(covout_i, 2, 0, axis=1)
            covout_i[2][2] = alpha_err
            fitout_i = np.insert(fitout_i, 2, -pfit[0])
        elif fixed_parameter == 'wn':
            wnest_err = np.std(p[((f > fwhite[0]) & (f < fwhite[1]))])
            covout_i = np.insert(covout_i, 1, 0, axis=0)
            covout_i = np.insert(covout_i, 1, 0, axis=1)
            covout_i[1][1] = wnest_err
            fitout_i = np.insert(fitout_i, 1, wnest)
        covout[i] = covout_i
        fitout[i] = fitout_i

    noise_model_coeffs = ["fknee", "white_noise", "alpha"]
    noise_fit_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(
            name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U8")
        ),
    )
    noise_fit_stats.wrap("fit", fitout, [(0, "dets"), (1, "noise_model_coeffs")])
    noise_fit_stats.wrap(
        "cov",
        covout,
        [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")],
    )

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats


def fit_binned_noise_model(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs=None,
    fwhite=(10, 100),
    lowf=1,
    merge_fit=False,
    f_min=None,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    mask=False,
    fixed_parameter=None,
    unbinned_mode=3,
    base=1.2,
):
    """
    Fits noise model with white and 1/f noise to the PSD of binned signal.
    This uses a least square method since the distrubition of binned points 
    is close to the standard distribution.

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
    f_min : float
        Minimum frequency to include in the fitting.
        Default is None which selects f_min as the second index of f.
    f_max : float
        Maximum frequency to include in the fitting. This is particularly
        important for lowpass filtered data such as that post demodulation
        if the data is not downsampled after lowpass filtering.
    merge_name : bool
        If ``merge_fit`` is True then addes into axis manager with merge_name.
    merge_psd : bool
        If ``merge_psd`` is True then adds fres and Pxx to the axis manager.
    mask : bool
        If ``mask`` is True then PSD is masked with ``aman.psd_mask``.
    fixed_parameter : str
        This accepts None or 'wn' or 'alpha'. If 'wn' ('alpha') is given, 
        white noise level (alpha) is fixed to the initially estimated value.
    unbinned_mode : int
        First Fourier modes up to this number are left un-binned.
    base : float (> 1)
        Base of the logspace bins.
    limit_N : int
        If the number of data points in a bin is less than ``limit_N``, that bin is handled with
        chi2 distribution and its error is <psd>. Otherwise the central limit theorem
        is applied to the bin and the error of the bin is std(psd)/sqrt(len(psd)).

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
        if psdargs is None:
            f, pxx = calc_psd(
                aman, signal=signal, timestamps=aman.timestamps, merge=merge_psd
            )
        else:
            f, pxx = calc_psd(
                aman,
                signal=signal,
                timestamps=aman.timestamps,
                merge=merge_psd,
                **psdargs,
            )
    if mask:
        if 'psd_mask' in aman:
            mask = ~aman.psd_mask.mask()
            f = f[mask]
            pxx = pxx[:, mask]
        else:
            print('"psd_mask" is not in aman. Masking is skipped.')

    eix = np.argmin(np.abs(f - f_max))
    if f_min is None:
        six = 1
    else:
        six = np.argmin(np.abs(f - f_min))
    f = f[six:eix]
    pxx = pxx[:, six:eix]
    # binning
    f, pxx, bin_size = calc_binned_psd(aman, f=f, pxx=pxx, unbinned_mode=unbinned_mode,
                                       base=base, merge=False)
    fitout = np.zeros((aman.dets.count, 3))
    # This is equal to np.sqrt(np.diag(cov)) when doing curve_fit
    covout = np.zeros((aman.dets.count, 3, 3))
    fixed_param = {}
    for i in range(len(pxx)):
        p = pxx[i]
        wnest = np.median(p[((f > fwhite[0]) & (f < fwhite[1]))])*1.5 #conversion factor from median to mean
        if fixed_parameter == None:
            try:
                pfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], wnest, -pfit[0]]
        elif fixed_parameter == 'wn':
            try:
                pfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], -pfit[0]]
            fixed_param['wn'] = wnest
        elif fixed_parameter == 'alpha':
            try:
                pfit, vfit = np.polyfit(np.log10(f[f < lowf]), np.log10(p[f < lowf]), 1, cov=True)
            except np.linalg.LinAlgError:
                print(
                    f"Cannot fit 1/f for detector {aman.dets.vals[i]} skipping."
                )
                covout[i] = np.full((3, 3), np.nan)
                fitout[i] = np.full(3, np.nan)
                continue
            fidx = np.argmin(np.abs(10 ** np.polyval(pfit, np.log10(f)) - wnest))
            p0 = [f[fidx], wnest]
            fixed_param['alpha'] = -pfit[0]
        else:
            print('"fixed_parameter" is invalid.')
            return
        res = minimize(lambda params: neglnlike(params, f, p, bin_size=bin_size, **fixed_param), 
               p0, method="Nelder-Mead")
        try:
            #Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p), full_output=True)
            Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p, bin_size=bin_size, **fixed_param), full_output=True)
            hessian_ndt, _ = Hfun(res["x"])
            # Inverse of the hessian is an estimator of the covariance matrix
            # sqrt of the diagonals gives you the standard errors.
            covout_i = np.linalg.inv(hessian_ndt)            
        except np.linalg.LinAlgError:
            print(
                f"Cannot calculate Hessian for detector {aman.dets.vals[i]} skipping. (LinAlgError)"
            )
            covout_i = np.full((len(p0), len(p0)), np.nan)
        except IndexError:
            print(
                f"Cannot calculate Hessian for detector {aman.dets.vals[i]} skipping. (IndexError)"
            )
            covout_i = np.full((len(p0), len(p0)), np.nan)
        fitout_i = res.x
        if fixed_parameter == 'alpha':
            alpha_err = np.sqrt(vfit[0][0])
            covout_i = np.insert(covout_i, 2, 0, axis=0)
            covout_i = np.insert(covout_i, 2, 0, axis=1)
            covout_i[2][2] = alpha_err
            fitout_i = np.insert(fitout_i, 2, -pfit[0])
        elif fixed_parameter == 'wn':
            wnest_err = np.std(p[((f > fwhite[0]) & (f < fwhite[1]))])
            covout_i = np.insert(covout_i, 1, 0, axis=0)
            covout_i = np.insert(covout_i, 1, 0, axis=1)
            covout_i[1][1] = wnest_err
            fitout_i = np.insert(fitout_i, 1, wnest)
        covout[i] = covout_i
        fitout[i] = fitout_i

    noise_model_coeffs = ["fknee", "white_noise", "alpha"]
    noise_fit_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(
            name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U8")
        ),
    )
    noise_fit_stats.wrap("fit", fitout, [(0, "dets"), (1, "noise_model_coeffs")])
    noise_fit_stats.wrap(
        "cov",
        covout,
        [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")],
    )

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats

def get_mask_for_hwpss(freq, hwp_freq, max_mode=10, width=((-0.4, 0.6), (-0.2, 0.2))):
    """
    Function that returns boolean array to mask hwpss in PSD.

    Arguments
    ---------
        freq : nparray
            Frequency of PSD of signal.

        hwp_freq : float
            HWP frequency.

        max_mode : int
            Maximum hwpss mode to subtract. 

        width : float/tuple/list/array
            If given in float, hwpss will be masked like
            (hwp_freq - width/2) < f < (hwp_freq + width/2).
            If given in array like [1.0, 0.4], 1f hwpss is masked like
            (hwp_freq - 0.5) < f < (hwp_freq + 0.5) and Nf hwpss are masked like
            (hwp_freq - 0.2) < f < (hwp_freq + 0.2).
            If given in array like [[-0.4, 0.6], [-0.2, 0.3]],
            1f hwpss is masked like (hwp_freq - 0.4) < f < (hwp_freq + 0.6)
            and Nf are masked like (hwp_freq - 0.2) < f < (hwp_freq + 0.3).
            Usually 1f hwpss distrubites wider than other hwpss.

    Returns
    -------
        mask: Boolean array to mask frequency and power of the given PSD. 
            True in this array stands for the index of hwpss to mask.
    """
    if isinstance(width, (float, int)):
        width_minus = -width/2
        width_plus = width/2
        mask_arrays = []
        for n in range(max_mode):
            mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+1), peak_width=(width_minus, width_plus)))
    elif isinstance(width, (np.ndarray, list, tuple)):
        width = np.array(width)
        if len(width.shape) == 1:
            # mask for 1f
            width_minus = -width[0]/2
            width_plus = width[0]/2
            mask_arrays = [get_mask_for_single_peak(freq, hwp_freq, peak_width=(width_minus, width_plus))]
            # masks for Nf
            width_minus = -width[1]/2
            width_plus = width[1]/2
            for n in range(max_mode-1):
                mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+2), peak_width=(width_minus, width_plus)))
        elif len(width.shape) == 2:
            # mask for 1f
            width_minus = width[0][0]
            width_plus = width[0][1]
            mask_arrays = [get_mask_for_single_peak(freq, hwp_freq, peak_width=(width_minus, width_plus))]
            # masks for Nf
            width_minus = width[1][0]
            width_plus = width[1][1]
            for n in range(max_mode-1):
                mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+2), peak_width=(width_minus, width_plus)))
    mask = np.any(np.array(mask_arrays), axis=0)
    return mask
        
        
def get_mask_for_single_peak(f, peak_freq, peak_width=(-0.002, +0.002)):
    """
    Function that returns boolean array to masks single peak (e.g. scan synchronous signal) in PSD.

    Arguments
    ---------
        f : nparray
            Frequency of PSD of signal.

        peak_freq : float
            Center frequency of the mask.

        peak_width : tuple
            Range to mask signal. The default masked range will be 
            (peak_freq - 0.002) < f < (peak_freq + 0.002).

    Returns
    -------
        mask: Boolean array to mask the given PSD. 
            True in this array stands for the index of the single peak to mask.
    """
    mask = (f > peak_freq + peak_width[0])&(f < peak_freq + peak_width[1])
    return mask

def binning_psd(psd, unbinned_mode=3, base=1.2, return_bin_size=False, drop_nan=False):
    """
    Function to bin PSD.
    First several Fourier modes are left un-binned.
    Fourier modes higher than that are averaged into logspace bins.

    Parameters
    ----------
    psd : numpy.ndarray
        PSD of the signal. Can be a 1D or 2D array.
    unbinned_mode : int, optional
        First Fourier modes up to this number are left un-binned. Defaults to 3.
    base : float, optional
        Base of the logspace bins. Must be greater than 1. Defaults to 1.2.
    return_bin_size : bool, optional
        If True, the number of data points in the bins are returned. Defaults to False.
    drop_nan : bool, optional
        If True, drop the indices where psd is NaN. Defaults to False.

    Returns
    -------
    binned_psd : numpy.ndarray
        The binned PSD. If the input is 2D, the output will also be 2D with the same number of rows.
    bin_size : numpy.ndarray, optional
        The number of data points in each bin, only returned if return_bin_size is True.
    """
    if base <= 1:
        raise ValueError("base must be greater than 1")

    # Ensure psd is at least 2D for consistent processing
    psd = np.atleast_2d(psd)
    num_signals, num_samples = psd.shape
    
    # Initialize the binned PSD and optionally the bin sizes
    binned_psd = np.zeros((num_signals, unbinned_mode + 1))
    binned_psd[:, :unbinned_mode + 1] = psd[:, :unbinned_mode + 1]
    bin_size = np.ones((num_signals, unbinned_mode + 1)) if return_bin_size else None

    # Determine the number of bins and their indices
    N = int(np.ceil(np.emath.logn(base, num_samples - unbinned_mode)))
    binning_idx = np.unique(np.logspace(base, N, N, base=base, dtype=int) + unbinned_mode - 1)
    
    # Bin the PSD values for each signal
    new_binned_psd = []
    new_bin_size = []
    for start, end in zip(binning_idx[:-1], binning_idx[1:]):
        if start < end:  # Ensure the slice is not empty
            bin_mean = np.nanmean(psd[:, start:end], axis=1)
            new_binned_psd.append(bin_mean)
            if return_bin_size:
                new_bin_size.append(end - start)
    
    # Convert lists to numpy arrays and concatenate with initial values
    new_binned_psd = np.array(new_binned_psd).T  # Transpose to match dimensions
    binned_psd = np.hstack([binned_psd, new_binned_psd])
    if return_bin_size:
        new_bin_size = np.array(new_bin_size)
        bin_size = np.hstack([bin_size, np.tile(new_bin_size, (num_signals, 1))])

    if drop_nan:
        valid_indices = ~np.isnan(binned_psd).any(axis=0)
        binned_psd = binned_psd[:, valid_indices]
        if return_bin_size:
            bin_size = bin_size[:, valid_indices]

    if return_bin_size:
        return binned_psd, bin_size
    return binned_psd
