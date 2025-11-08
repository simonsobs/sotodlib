"""FFTs and related operations
"""
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing_extensions import Callable
from numpy.typing import NDArray
import warnings
import numdifftools as ndt
import numpy as np
import pyfftw
import so3g
import sys
from so3g.proj import Ranges
from scipy.optimize import minimize
from scipy.signal import welch
from scipy.stats import chi2
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

    axis = getattr(aman, axis_name)

    if len(aman._assignments[signal_name]) == 1:
        n_det = 1
        other_idx = None
    elif len(aman._assignments[signal_name]) == 2:
        checks = np.array(
            [x == axis_name for x in aman._assignments[signal_name]], dtype="bool"
        )
        other_idx = np.where(~checks)[0][0]
        other_axis = getattr(aman, aman._assignments[signal_name][other_idx])
        n_det = other_axis.count
    else:
        raise ValueError("rfft only works for 1D or 2D data streams")

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

    rfft = RFFTObj.for_shape(n_det, n, "FFTW_FORWARD")
    if resize == "zero_pad":
        rfft.a[:, : axis.count] = signal
        rfft.a[:, axis.count :] = 0
    elif resize == "trim":
        rfft.a[:] = signal[:, :n]
    else:
        rfft.a[:] = signal[:]

    rfft.t_forward()

    if delta_t is None:
        if "timestamps" in aman:
            delta_t = (aman.timestamps[-1] - aman.timestamps[0]) / axis.count
        else:
            delta_t = 1
    freqs = np.fft.rfftfreq(n, delta_t)

    if other_idx is not None and other_idx != 0:
        return rfft.b.transpose(), freqs

    return rfft.b, freqs


def _t_null(direction):
    raise ValueError(f"No {direction} FFT defined")

@dataclass
class RFFTObj:
    """
    Dataclass to store information needed for rfft.

    Attributes:

        n_det: Number of detectors this object was built for.

        n: Number of samples this object was built for.

        a: Buffer for the real part of the FFT.

        b: Buffer for the complex part of the FFT.

        t_forward: Function for performing the forward FFT.

        t_backward: Function for performing the backward FFT.
    """
    n_det: int
    n: int
    a: NDArray[np.float32]
    b: NDArray[np.complex64]
    t_forward: Callable = field(default=partial(_t_null, "forward"))
    t_backward: Callable = field(default=partial(_t_null, "backward"))

    def validate(self, shape):
        if shape != (self.n_det, self.n):
            raise ValueError("Data is wrong shape for the rfft object")

    @classmethod
    def for_shape(cls, n_det, n, direction="FFTW_FORWARD", **kwargs):
        """Build PyFFTW object for fft-ing
    
        Arguments:
    
            n_det: number of detectors (or just the arr.shape[0] for the
                array you are going to fft)
    
            n: number of samples in timestream
    
            direction: fft direction. Can be FFTW_FORWARD, FFTW_BACKWARD, or BOTH
    
            kwargs: additional arguments to pass to pyfftw.FFTW
    
        Returns:
    
            rfft_obj: An instance of RFFTObj
        """
        fftargs = {"threads": _get_num_threads(), "flags": ["FFTW_ESTIMATE"]}
        fftargs.update(kwargs)
    
        a = pyfftw.empty_aligned((n_det, n), dtype="float32")
        b = pyfftw.empty_aligned((n_det, (n + 2) // 2), dtype="complex64")
        
        t_forward = partial(_t_null, "forward") 
        t_backward = partial(_t_null, "backward") 
        if direction == "FFTW_FORWARD":
            t_forward = pyfftw.FFTW(a, b, direction=direction, **fftargs)
        elif direction == "FFTW_BACKWARD":
            t_backward = pyfftw.FFTW(b, a, direction=direction, **fftargs)
        elif direction == "BOTH":
            t_forward = pyfftw.FFTW(a, b, direction="FFTW_FORWARD", **fftargs)
            t_backward = pyfftw.FFTW(b, a, direction="FFTW_BACKWARD", **fftargs)
        else:
            raise ValueError("direction must be FFTW_FORWARD, FFTW_BACKWARD, or BOTH")
    
        rfft_obj = cls(n_det, n, a, b, t_forward, t_backward)
    
        return rfft_obj 

    @classmethod
    def for_tod(cls, tod, direction="FFTW_FORWARD", **kwargs):
        """
        Wrapper around ``for_shape`` that pulls the shape from a 2d array

        Arguments: 

            tod: 2D array to build rfft object for.
    
            direction: fft direction. Can be FFTW_FORWARD, FFTW_BACKWARD, or BOTH
    
            kwargs: additional arguments to pass to pyfftw.FFTW
    
        Returns:
    
            rfft_obj: An instance of RFFTObj
        """
        if len(tod.shape) != 2:
            raise ValueError("Can only build rfft object for 2D arrays")
        return cls.for_shape(tod.shape[0], tod.shape[1], direction, **kwargs)


@lru_cache
def find_inferior_integer(target, primes=(2, 3, 5, 7, 11, 13)):
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
        best_friend = getattr(find_inferior_integer, "__wrapped__", find_inferior_integer)(target / base, primes[1:])
        if (best_friend * base) >= best:
            best = best_friend * base
    return int(best)


@lru_cache
def find_superior_integer(target, primes=(2, 3, 5, 7, 11, 13)):
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
        best_friend = getattr(find_superior_integer, "__wrapped__", find_superior_integer)(target / base, primes[1:])
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
    subscan=False,
    full_output=False,
    aggregate=None,          # None | 'mean' | 'median' | callable(arr, axis=0)->1D
    label_axis='dets',
    **kwargs
):
    """
    Calculates the power spectrum density of an input signal using scipy.signal.welch().
    Data defaults to aman.signal and timestamps to aman.timestamps.

    By default the nperseg will be ~power of 2 closest to 1/50 of the samples used,
    unless explicitly overridden by nperseg or freq_spacing.

    Arguments
    ---------
        aman : AxisManager
            with (dets, samps) OR (channels, samps) axes.

        signal : ndarray, optional
            Data signal to pass to scipy.signal.welch(). Defaults to aman.signal.

        timestamps : ndarray, optional
            Timestamps corresponding to the data signal. Defaults to aman.timestamps.

        max_samples : int
            Max samples along sample axis to send to Welch.

        prefer : {'left','right','center'}
            How to select a slice if trimming to max_samples.

        freq_spacing : float, optional
            Approximate desired frequency spacing of the PSD. If None the default
            nperseg (~1/50th the length) is used unless nperseg is passed directly.

        merge : bool
            If True, merge results (freqs and Pxx) into aman.

        merge_suffix : str or None
            Suffix for the Pxx field name (Pxx_{suffix}). If None:
              - When aggregate is None: field name 'Pxx'
              - When aggregate is set: field name 'Pxx_agg'

        overwrite : bool
            If True, overwrite existing 'freqs' / 'Pxx*' / 'nseg' fields.

        subscan : bool
            If True, compute PSD on subscans and stack results.

        full_output : bool
            If True also return nseg (segments used by Welch) for debiasing.

        aggregate : None | 'mean' | 'median' | callable
            If None (default), keep per-detector PSDs.
              - Regular path  : Pxx shape (dets, nfreqs)
              - Subscan path  : Pxx shape (dets, nfreqs, nsubscans)
            If set, reduce over detectors (axis=0):
              - Regular path  : Pxx shape (nfreqs,)
              - Subscan path  : Pxx shape (nfreqs, nsubscans)

        label_axis : str
            Name of label axis for detectors/channels. Default 'dets'.

        **kwargs :
            Passed through to scipy.signal.welch() (e.g., nperseg, noverlap, window, axis).

    Returns
    -------
        freqs : ndarray
        Pxx   : ndarray
        nseg  : int or ndarray (only if full_output=True)

    Notes
    -----
    - For unbiased median white-noise estimates via calc_wn, use noverlap=0 and full_output=True.
    """
    if signal is None:
        signal = aman.signal

    if ("noverlap" not in kwargs) or ("noverlap" in kwargs and kwargs["noverlap"] != 0):
        warnings.warn(
            'calc_wn will be biased. noverlap argument of welch needs to be 0 '
            'to get an unbiased median white noise estimate.'
        )
    if not full_output:
        warnings.warn(
            'calc_wn will be biased. full_output argument of calc_psd needs to be True '
            'to get an unbiased median white noise estimate.'
        )

    if subscan:
        # --- Subscan path (delegates to helper and supports aggregate) ---
        if full_output:
            freqs, Pxx, nseg = _calc_psd_subscan(
                aman, signal=signal, freq_spacing=freq_spacing,
                full_output=True, aggregate=aggregate, **kwargs
            )
        else:
            freqs, Pxx = _calc_psd_subscan(
                aman, signal=signal, freq_spacing=freq_spacing,
                full_output=False, aggregate=aggregate, **kwargs
            )

        # Axis maps depend on aggregation
        if aggregate is None:
            axis_map_pxx = [(0, label_axis), (1, "nusamps"), (2, "subscans")]
        else:
            # aggregated over detectors -> (nusamps, subscans)
            axis_map_pxx = [(0, "nusamps"), (1, "subscans")]
        axis_map_nseg = [(0, "subscans")]

    else:
        # --- Regular (non-subscan) path ---
        if timestamps is None:
            timestamps = aman.timestamps

        n_samps = signal.shape[-1]
        if n_samps <= max_samples:
            start, stop = 0, n_samps
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
            start, stop = offset, offset + max_samples

        fs = 1 / np.nanmedian(np.diff(timestamps[start:stop]))
        if "nperseg" not in kwargs:
            if freq_spacing is not None:
                nperseg = int(2 ** (np.around(np.log2(fs / freq_spacing))))
            else:
                nperseg = int(2 ** (np.around(np.log2((stop - start) / 50.0))))
            kwargs["nperseg"] = nperseg

        if kwargs["nperseg"] > max_samples:
            nseg = 1
        else:
            nseg = int(max_samples / kwargs["nperseg"])

        freqs, Pxx = welch(signal[:, start:stop], fs, **kwargs)

        # Aggregate (reduce detectors) if requested
        if aggregate is not None:
            if aggregate == 'mean':
                Pxx = np.nanmean(Pxx, axis=0)      # -> (nfreqs,)
            elif aggregate == 'median':
                Pxx = np.nanmedian(Pxx, axis=0)    # -> (nfreqs,)
            elif callable(aggregate):
                Pxx = aggregate(Pxx, axis=0)       # -> (nfreqs,)
            else:
                raise ValueError("aggregate must be None, 'mean', 'median', or a callable(arr, axis=0)->1D")

        # Axis maps depend on aggregation
        if aggregate is None:
            axis_map_pxx = [(0, aman[label_axis]), (1, "nusamps")]
        else:
            axis_map_pxx = [(0, "nusamps")]
        axis_map_nseg = None if aggregate is None else None  # nseg is scalar here

    # --- Merge into aman if requested ---
    if merge:
        # ensure 'nusamps' axis and 'freqs'
        if 'nusamps' not in aman:
            aman.merge(core.AxisManager(core.OffsetAxis("nusamps", len(freqs))))
            aman.wrap("freqs", freqs, [(0, "nusamps")])
        else:
            if len(freqs) != aman.nusamps.count:
                raise ValueError(
                    "New freqs does not match the shape of existing nusamps. "
                    "Use the same nperseg (or freq_spacing) to keep consistency."
                )

        # name selection
        if merge_suffix is None:
            Pxx_name = 'Pxx' if aggregate is None else 'Pxx_agg'
        else:
            Pxx_name = f'Pxx_{merge_suffix}'

        if overwrite and (Pxx_name in aman._fields):
            aman.move(Pxx_name, None)
        aman.wrap(Pxx_name, Pxx, axis_map_pxx)

        if full_output:
            if overwrite and ("nseg" in aman._fields):
                aman.move("nseg", None)
            # For subscan path nseg is (subscans,), else it's scalar
            if subscan:
                aman.wrap("nseg", nseg, axis_map_nseg)
            else:
                aman.wrap("nseg", nseg, None)

    if full_output:
        return freqs, Pxx, nseg
    else:
        return freqs, Pxx


def _calc_psd_subscan(aman, signal=None, freq_spacing=None, full_output=False,
                      aggregate=None, **kwargs):
    """
    Calculate PSD on each subscan using scipy.signal.welch().
    Data defaults to aman.signal. Timestamps from aman.timestamps.
    Subscan windows from aman.subscan_info.

    Parameters
    ----------
    aman : AxisManager
    signal : ndarray, optional
        Defaults to aman.signal
    freq_spacing : float, optional
        If provided, sets nperseg ~ 2**round(log2(fs / freq_spacing)).
    full_output : bool
        If True, also return nseg per subscan.
    aggregate : None | 'mean' | 'median' | callable
        If None: return per-detector PSDs with shape (dets, nusamps, subscans).
        Otherwise: reduce over detectors (axis=0) and return shape (nusamps, subscans).
    **kwargs :
        Passed to scipy.signal.welch (e.g., nperseg, noverlap, window, axis).

    Returns
    -------
    freqs : ndarray
    Pxx   : ndarray
        (dets, nusamps, subscans) if aggregate is None;
        (nusamps, subscans) otherwise.
    nseg  : ndarray
        Number of segments per subscan (only if full_output=True).
    """
    from .flags import get_subscan_signal

    if signal is None:
        signal = aman.signal

    fs = 1 / np.nanmedian(np.diff(aman.timestamps))
    if "nperseg" not in kwargs:
        if freq_spacing is not None:
            nperseg = int(2 ** (np.around(np.log2(fs / freq_spacing))))
        else:
            # use typical subscan duration / 4
            duration_samps = np.asarray([
                np.ptp(x.ranges()) if x.ranges().size > 0 else 0
                for x in aman.subscan_info.subscan_flags
            ])
            duration_samps = duration_samps[duration_samps > 0]
            if duration_samps.size == 0:
                # fallback
                nperseg = int(2 ** (np.around(np.log2(signal.shape[-1] / 50.0))))
            else:
                nperseg = int(2 ** (np.around(np.log2(np.median(duration_samps) / 4))))
        kwargs["nperseg"] = nperseg

    Pxx_list, nseg_list = [], []
    for iss in range(aman.subscan_info.subscans.count):
        signal_ss = get_subscan_signal(aman, signal, iss)
        axis = -1 if "axis" not in kwargs else kwargs["axis"]
        nsamps = signal_ss.shape[axis]

        if nsamps >= kwargs["nperseg"]:
            freqs, pxx_sub = welch(signal_ss, fs, **kwargs)  # pxx_sub: (dets, nfreqs)
            Pxx_list.append(pxx_sub)
            nseg_list.append(int(nsamps / kwargs["nperseg"]))
        else:
            # too short: fill with NaNs to keep shapes consistent
            Pxx_list.append(np.full((signal.shape[0], kwargs["nperseg"]//2 + 1), np.nan))
            nseg_list.append(np.nan)

    nseg = np.array(nseg_list)                             # (subscans,)
    Pxx = np.array(Pxx_list)                                # (subscans, dets, nfreqs)
    Pxx = Pxx.transpose(1, 2, 0)                            # -> (dets, nfreqs, subscans)

    # Aggregate over detectors if requested
    if aggregate is not None:
        if aggregate == 'mean':
            Pxx = np.nanmean(Pxx, axis=0)                   # -> (nfreqs, subscans)
        elif aggregate == 'median':
            Pxx = np.nanmedian(Pxx, axis=0)                 # -> (nfreqs, subscans)
        elif callable(aggregate):
            Pxx = aggregate(Pxx, axis=0)                    # -> (nfreqs, subscans)
        else:
            raise ValueError("aggregate must be None, 'mean', 'median', or a callable(arr, axis=0)->1D")

    if full_output:
        return freqs, Pxx, nseg
    else:
        return freqs, Pxx


def calc_wn(aman, pxx=None, freqs=None, nseg=None, low_f=5, high_f=10):
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

        nseg (Int or 1d Int array):
            number of segmnents used for welch. Defaults to aman.nseg. This is
            necessary for debiasing median white noise estimation. welch PSD with
            non-overlapping n segments follows chi square distribution with
            2 * nseg degrees of freedom. The median of chi square distribution is
            biased from its average.

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

    if nseg is None:
        nseg = aman.get('nseg')

    if nseg is None:
        warnings.warn('white noise level estimated by median PSD is biased. '
                      'nseg is necessary to debias. Need to use following '
                      'arguments in calc_psd to get correct nseg. '
                      '`noverlap=0, full_output=True`')
        debias = None
    else:
        debias = 2 * nseg / chi2.ppf(0.5, 2 * nseg)

    fmsk = np.all([freqs >= low_f, freqs <= high_f], axis=0)
    if pxx.ndim == 1:
        wn2 = np.median(pxx[fmsk])
    else:
        wn2 = np.median(pxx[:, fmsk], axis=1)
    if debias is not None:
        if pxx.ndim == 3:
            wn2 *= debias[None, :]
        else:
            wn2 *= debias
    wn = np.sqrt(wn2)
    return wn

'''
def noise_model(f, params, **fixed_param):
    """
    Noise model for power spectrum with white noise, and 1/f noise.
    If any fixed param is handed, that parameter is fixed in the fit.
    'alpha' or 'wn' can be fixed.
    params = [wn, fknee, alpha]

    Minimal stabilization: enforce a tiny floor on fknee relative to the
    lowest frequency *actually used in the fit* to avoid unphysically
    small fknee solutions that destabilize the fit.
    """
    # --- unpack as before ---
    if 'wn' in fixed_param.keys():
        if len(params) == 2:
            wn = fixed_param['wn']
            fknee, alpha = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    elif 'alpha' in fixed_param.keys():
        if len(params) == 2:
            alpha = fixed_param['alpha']
            wn, fknee = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    elif len(fixed_param) == 0:
        if len(params) == 3:
            wn, fknee, alpha = params[0], params[1], params[2]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    else:
        raise ValueError('"alpha" or "wn" can be a fixed parameter.')

    # --- minimal fknee floor tied to the fit band ---
    # use the smallest positive frequency in the passed-in f array
    # (fit_noise_model already slices out f=0 via six=1)
    fpos = f[f > 0]
    if fpos.size == 0:
        # extreme corner case; keep behavior sane
        fmin_pos = 1e-6
    else:
        fmin_pos = float(np.nanmin(fpos))

    # floor fknee at a small fraction of the lowest fitted frequency
    # tweak 0.25 -> 0.1 or 0.5 if you want looser/tighter floor
    fknee_floor = max(1e-6, 0.25 * fmin_pos)
    fknee_eff = fknee if fknee >= fknee_floor else fknee_floor

    return wn**2 * (1 + (fknee_eff / f) ** alpha)

def noise_model_v0(f, params, **fixed_param):
    """
    Noise model for power spectrum with white noise, and 1/f noise.
    If any fixed param is handed, that parameter is fixed in the fit.
    'alpha' or 'wn' can be fixed.
    params = [wn, fknee, alpha]
    """
    #if 'wn' in fixed_param.keys():
    #    if len(params)==2:
    #        wn = fixed_param['wn']
    #        fknee, alpha = params[0], params[1]
    #    else:
    #        raise ValueError('The number of fit parameters are invalid.')
    #elif 'alpha' in fixed_param.keys():
    #    if len(params)==2:
    #        alpha = fixed_param['alpha']
    #        wn, fknee = params[0], params[1]
    #    else:
    #        raise ValueError('The number of fit parameters are invalid.')
    #elif len(fixed_param)==0:
    #    if len(params)==3:
    #        wn, fknee, alpha = params[0], params[1], params[2]
    #    else:
    #        raise ValueError('The number of fit parameters are invalid.')
    #else:
    #    raise ValueError('"alpha" or "wn" can be a fixed parameter.')
    #return wn**2 * (1 + (fknee / f) ** alpha)
    if 'wn' in fixed_param:
        log_wn = np.log(fixed_param['wn'])
        log_fknee, alpha = params
    elif 'alpha' in fixed_param:
        alpha = fixed_param['alpha']
        log_wn, log_fknee = params
    else:
        log_wn, log_fknee, alpha = params

    # Guard f away from 0
    f_safe = np.maximum(f, 1e-8)

    # log(model) = 2*log_wn + log(1 + exp(alpha*(log_fknee - log f)))
    t = alpha * (log_fknee - np.log(f_safe))
    t = np.clip(t, -50, 50)                # avoid overflow in exp
    log_model = 2.0*log_wn + np.log1p(np.exp(t))
    return np.exp(log_model)

def noise_model(f, params, **fixed_param):
    """
    Original model with minimal stabilization so 'Nelder-Mead' can't
    drive fknee -> 0 or alpha to absurd values (it ignores 'bounds').
    """
    # --- unpack exactly like before ---
    if 'wn' in fixed_param.keys():
        if len(params)==2:
            wn = fixed_param['wn']
            fknee, alpha = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    elif 'alpha' in fixed_param.keys():
        if len(params)==2:
            alpha = fixed_param['alpha']
            wn, fknee = params[0], params[1]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    elif len(fixed_param)==0:
        if len(params)==3:
            wn, fknee, alpha = params[0], params[1], params[2]
        else:
            raise ValueError('The number of fit parameters are invalid.')
    else:
        raise ValueError('"alpha" or "wn" can be a fixed parameter.')

    # --- minimal, model-internal constraints (since Nelder–Mead ignores bounds) ---
    f_safe = np.maximum(f, 1e-21)                # avoid div-by-zero
    # floor fknee at small fraction of the *lowest fitted* frequency
    fpos = f_safe[f_safe > 0]
    fmin_pos = float(np.nanmin(fpos)) if fpos.size else 1e-6
    fknee_floor = max(1e-6, 0.25 * fmin_pos)     # tweak 0.25 if desired (0.1–0.5)
    fknee_eff = fknee if fknee >= fknee_floor else fknee_floor

    # softly clip alpha to the same bounds the code intended (0..10)
    alpha_eff = np.clip(alpha, 0.0, 10.0)

    return wn**2 * (1 + (fknee_eff / f_safe) ** alpha_eff)


def noise_model_stable(f, params, **fixed_param):
    """
    Stable version of wn^2 * (1 + (fknee/f)^alpha), parameterized in log-space.

    params = [log_wn, log_fknee, alpha]    (or 2 params if one is fixed)
    """
    # Unpack (handle fixed params)
    if 'wn' in fixed_param:
        log_wn = np.log(fixed_param['wn'])
        log_fknee, alpha = params
    elif 'alpha' in fixed_param:
        alpha = fixed_param['alpha']
        log_wn, log_fknee = params
    else:
        log_wn, log_fknee, alpha = params

    # Guard f away from 0
    f_safe = np.maximum(f, 1e-8)

    # log(model) = 2*log_wn + log(1 + exp(alpha*(log_fknee - log f)))
    t = alpha * (log_fknee - np.log(f_safe))
    t = np.clip(t, -50, 50)                # avoid overflow in exp
    log_model = 2.0*log_wn + np.log1p(np.exp(t))
    return np.exp(log_model)


def neglnlike_stable(params, x, y, bin_size=1, **fixed_param):
    m = noise_model_stable(x, params, **fixed_param)
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        return 1e30
    out = np.sum((np.log(m) + y/m) * bin_size)
    if not np.isfinite(out):
        return 1e30
    return out
'''

def noise_model(
    f,
    params,
    *,
    fixed_param=None,           # None, or {'wn': <float>} or {'alpha': <float>}
    use_logspace=True,          # False -> linear params [wn, fknee, alpha]
                                # True  -> log-space params [log_wn, log_fknee, alpha]
    fknee_floor_frac=0.25,      # floor fknee at this fraction of lowest fitted f
    alpha_bounds=(0.0, 10.0),   # clip alpha into these bounds
):
    """
    Unified noise model:
        P(f) = wn^2 * [1 + (fknee / f)^alpha]

    Parameters
    ----------
    f : array-like
        Frequencies.
    params : array-like
        If use_logspace == False (linear):
            - fixed_param is None -> [wn, fknee, alpha]
            - fixed_param == {'wn': wn0} -> [fknee, alpha]
            - fixed_param == {'alpha': a0} -> [wn, fknee]
        If use_logspace == True (stable/log-space):
            - fixed_param is None -> [log_wn, log_fknee, alpha]
            - fixed_param == {'wn': wn0} -> [log_fknee, alpha]
            - fixed_param == {'alpha': a0} -> [log_wn, log_fknee]

    Keyword-only options
    --------------------
    fixed_param : dict or None
        Fix 'wn' or 'alpha' during evaluation.
    use_logspace : bool
        If True, interpret params in log-space (numerically stable).
    fknee_floor_frac : float
        Floor fknee at max(1e-6, fknee_floor_frac * f_min_pos) where f_min_pos is
        the smallest positive frequency used (protects against fknee -> 0).
    alpha_bounds : (float, float)
        Clip alpha to this range.

    Returns
    -------
    ndarray
        PSD model evaluated at f.
    """
    f = np.asarray(f, dtype=float)
    f_safe = np.maximum(f, 1e-21)  # avoid div-by-zero and log(0)
    fpos = f_safe[f_safe > 0]
    fmin_pos = float(np.nanmin(fpos)) if fpos.size else 1e-6
    fknee_floor = max(1e-6, fknee_floor_frac * fmin_pos)

    # --- Unpack parameters depending on space and fixed_param ---
    if use_logspace:
        if fixed_param is None:
            if len(params) != 3:
                raise ValueError("use_logspace=True expects [log_wn, log_fknee, alpha].")
            log_wn, log_fknee, alpha = params
        elif 'wn' in fixed_param:
            if len(params) != 2:
                raise ValueError("use_logspace=True with fixed wn expects [log_fknee, alpha].")
            log_fknee, alpha = params
            log_wn = np.log(float(fixed_param['wn']))
        elif 'alpha' in fixed_param:
            if len(params) != 2:
                raise ValueError("use_logspace=True with fixed alpha expects [log_wn, log_fknee].")
            log_wn, log_fknee = params
            alpha = float(fixed_param['alpha'])
        else:
            raise ValueError("fixed_param must be None, {'wn': val}, or {'alpha': val}.")

        # Guard / clip
        alpha_eff = float(np.clip(alpha, *alpha_bounds))
        fknee = float(np.exp(log_fknee))
        fknee_eff = fknee if fknee >= fknee_floor else fknee_floor
        log_fknee_eff = np.log(fknee_eff)

        # Stable computation in log space: log P = 2*log_wn + log(1 + exp(alpha*(log_fknee - log f)))
        t = alpha_eff * (log_fknee_eff - np.log(f_safe))
        t = np.clip(t, -50.0, 50.0)                # avoid overflow
        log_model = 2.0 * float(log_wn) + np.log1p(np.exp(t))
        P = np.exp(log_model)
        return P

    else:
        # Linear parameterization
        if fixed_param is None:
            if len(params) != 3:
                raise ValueError("Linear mode expects [wn, fknee, alpha].")
            wn, fknee, alpha = params
        elif 'wn' in fixed_param:
            if len(params) != 2:
                raise ValueError("Linear mode with fixed wn expects [fknee, alpha].")
            wn = float(fixed_param['wn'])
            fknee, alpha = params
        elif 'alpha' in fixed_param:
            if len(params) != 2:
                raise ValueError("Linear mode with fixed alpha expects [wn, fknee].")
            alpha = float(fixed_param['alpha'])
            wn, fknee = params
        else:
            raise ValueError("fixed_param must be None, {'wn': val}, or {'alpha': val}.")

        wn = float(wn)
        fknee = float(fknee)
        alpha_eff = float(np.clip(alpha, *alpha_bounds))
        fknee_eff = fknee if fknee >= fknee_floor else fknee_floor

        return wn**2 * (1.0 + (fknee_eff / f_safe) ** alpha_eff)


def neglnlike(
    params,
    x,
    y,
    *,
    bin_size=1.0,
    fixed_param=None,
    use_logspace=False,
    fknee_floor_frac=0.25,
    alpha_bounds=(0.0, 10.0),
    fail_value=1.0e30,
):
    """
    Negative log-likelihood for PSD data under the Whittle approximation:

        -ln L ~ sum_{bins} bin_size * [ ln M(f) + Pxx(f) / M(f) ]

    where M is the model P(f) and Pxx is the observed PSD.

    Parameters
    ----------
    params : array-like
        See `noise_model` for parameter expectations (depends on use_logspace / fixed_param).
    x : ndarray
        Frequencies.
    y : ndarray
        Observed PSD samples at x.
    bin_size : float
        Optional multiplicity/weight per bin (e.g., size of log-bins).
    fixed_param : dict or None
        Forwarded to `noise_model`.
    use_logspace : bool
        Forwarded to `noise_model`.
    fknee_floor_frac : float
        Forwarded to `noise_model`.
    alpha_bounds : (float, float)
        Forwarded to `noise_model`.
    fail_value : float
        Return value if numerical issues occur.

    Returns
    -------
    float
        The negative log-likelihood value.
    """
    try:
        m = noise_model(
            x,
            params,
            fixed_param=fixed_param,
            use_logspace=use_logspace,
            fknee_floor_frac=fknee_floor_frac,
            alpha_bounds=alpha_bounds,
        )
    except Exception:
        return fail_value

    if (not np.all(np.isfinite(m))) or np.any(m <= 0):
        return fail_value

    out = np.sum((np.log(m) + (y / m)) * bin_size)
    if not np.isfinite(out):
        return fail_value
    return float(out)


def get_psd_mask(aman, psd_mask=None, f=None,
                mask_hwpss=True, hwp_freq=None, max_hwpss_mode=10, hwpss_width=((-0.4, 0.6), (-0.2, 0.2)),
                mask_peak=False, peak_freq=None, peak_width=(-0.002, +0.002),
                merge=True, overwrite=True
):
    """
    Function to get masks for hwpss or single peak in PSD.

    Arguments
    ---------
        aman : AxisManager
            Axis manager which has 'nusamps' axis.
        psd_mask : numpy.ndarray or Ranges
            Existing psd_mask to be updated. If None, a new mask is created.
        f : nparray
            Frequency of PSD of signal. If None, aman.freqs are used.
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
    if f is None:
        f = aman.freqs
    if psd_mask is None:
        psd_mask = np.zeros(f.shape, dtype=bool)
    elif isinstance(psd_mask, so3g.RangesInt32):
        psd_mask = psd_mask.mask()

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
        if 'nusamps' not in list(aman._axes.keys()):
            aman.merge(core.AxisManager(core.OffsetAxis("nusamps", len(f))))
        aman.wrap("psd_mask", psd_mask, [(0,"nusamps")])
    return psd_mask

def get_binned_psd(
    aman,
    f=None,
    pxx=None,
    unbinned_mode=3,
    base=1.05,
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
            
    f_bin, bin_size = log_binning(f, unbinned_mode=unbinned_mode, base=base, return_bin_size=True)
    pxx_bin = log_binning(pxx, unbinned_mode=unbinned_mode, base=base, return_bin_size=False)

    if merge:
        aman.merge(core.AxisManager(core.OffsetAxis("nusamps_bin", len(f_bin))))
        if overwrite:
            if "freqs_bin" in aman:
                aman.move("freqs_bin", None)
            if "Pxx_bin" in aman:
                aman.move("Pxx_bin", None)
            if "bin_size" in aman:
                aman.move("bin_size", None)
        aman.wrap("freqs_bin", f_bin, [(0,"nusamps_bin")])
        aman.wrap("bin_size", bin_size, [(0,"nusamps_bin")])
        if pxx_bin.ndim > 2 and 'subscans' in aman and pxx_bin.shape[-1] == aman.subscans.count:
            aman.wrap("Pxx_bin", pxx_bin, [(0, "dets"), (1, "nusamps_bin"), (2, "subscans")])
        else:
            aman.wrap("Pxx_bin", pxx_bin, [(0,"dets"),(1,"nusamps_bin")])

    return f_bin, pxx_bin, bin_size


def _perdet_log_bounds_from_estimates(
    wn_est, fknee_est, alpha_est, f_max,
    fixed_param=None,
    # relative expansion around estimates:
    wn_factor_lo=0.25, wn_factor_hi=4.0,
    fknee_factor_lo=0.25, fknee_factor_hi=4.0,
    alpha_pad_lo=0.75, alpha_pad_hi=0.75,
    # absolute clamps:
    wn_global=(1e-12, 1e-2),
    fknee_global=(1e-4, None),  # upper bound will become f_max/2
    alpha_global=(0.5, 4.0),
):
    """
    Build per-detector bounds in log-space from per-det (or scalar) estimates.
    Returns arrays (N,2) for each parameter you are fitting.
    """
    import numpy as np

    wn_est    = np.asarray(wn_est,    dtype=float)
    fknee_est = np.asarray(fknee_est, dtype=float)
    alpha_est = np.asarray(alpha_est, dtype=float)
    ndet = max(wn_est.size, fknee_est.size, alpha_est.size)

    if wn_est.size    == 1: wn_est    = np.full(ndet, wn_est.item())
    if fknee_est.size == 1: fknee_est = np.full(ndet, fknee_est.item())
    if alpha_est.size == 1: alpha_est = np.full(ndet, alpha_est.item())

    # ---- derive linear bounds around estimates with multiplicative padding
    wn_lo_lin    = wn_est    * wn_factor_lo
    wn_hi_lin    = wn_est    * wn_factor_hi
    fknee_lo_lin = fknee_est * fknee_factor_lo
    fknee_hi_lin = fknee_est * fknee_factor_hi
    alpha_lo_lin = alpha_est - alpha_pad_lo
    alpha_hi_lin = alpha_est + alpha_pad_hi

    # ---- clamp to global limits
    wn_lo_lin    = np.maximum(wn_lo_lin, wn_global[0])
    wn_hi_lin    = np.minimum(wn_hi_lin, wn_global[1])

    fknee_upper  = f_max/2.0 if f_max is not None else None
    fknee_lo_lin = np.maximum(fknee_lo_lin, fknee_global[0])
    if fknee_upper is not None:
        fknee_hi_lin = np.minimum(fknee_hi_lin, fknee_upper)

    alpha_lo_lin = np.maximum(alpha_lo_lin, alpha_global[0])
    alpha_hi_lin = np.minimum(alpha_hi_lin, alpha_global[1])

    # Avoid inversions; if estimate was tiny/NaN, fall back to globals
    def _sanitize(lo, hi, lo_default, hi_default):
        lo2 = lo.copy()
        hi2 = hi.copy()
        bad = ~np.isfinite(lo2) | ~np.isfinite(hi2) | (hi2 <= lo2)
        lo2[bad] = lo_default
        hi2[bad] = hi_default
        return lo2, hi2

    wn_lo_lin,    wn_hi_lin    = _sanitize(wn_lo_lin,    wn_hi_lin,    wn_global[0],    wn_global[1])
    fknee_lo_lin, fknee_hi_lin = _sanitize(fknee_lo_lin, fknee_hi_lin, fknee_global[0], fknee_upper or 1.0)
    alpha_lo_lin, alpha_hi_lin = _sanitize(alpha_lo_lin, alpha_hi_lin, alpha_global[0], alpha_global[1])

    # Convert to log-space for wn,fknee
    log_wn_bounds    = np.stack([np.log(wn_lo_lin),    np.log(wn_hi_lin)],    axis=1)  # (N,2)
    log_fknee_bounds = np.stack([np.log(fknee_lo_lin), np.log(fknee_hi_lin)], axis=1)  # (N,2)
    alpha_bounds     = np.stack([alpha_lo_lin,         alpha_hi_lin],         axis=1)  # (N,2)

    # Drop the bound for any fixed parameter
    if fixed_param == 'wn':
        return None, log_fknee_bounds, alpha_bounds
    elif fixed_param == 'alpha':
        return log_wn_bounds, log_fknee_bounds, None
    else:
        return log_wn_bounds, log_fknee_bounds, alpha_bounds


'''
def fit_noise_model(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs={},
    fknee_est=1,
    wn_est=4E-5,
    alpha_est=3.4,
    merge_fit=False,
    lowf=None,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    mask=False,
    fixed_param=None,
    binning=False,
    unbinned_mode=3,
    base=1.05,
    freq_spacing=None,
    subscan=False,
    label_axis='dets'
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
    fknee_est : float or nparray
        Initial estimation value of knee frequency. If it is given in array,
        initial values are applied for each detector.
    wn_est : float or nparray
        Initial estimation value of white noise. If it is given in array,
        initial values are applied for each detector.
    alpha_est : float or nparray
        Initial estimation value of alpha. If it is given in array,
        initial values are applied for each detector.
    merge_fit : bool
        Merges fit and fit statistics into input axis manager.
    lowf : float
        Minimum frequency to include in the fitting.
        Default is None which selects lowf as the second index of f.
    f_max : float
        Maximum frequency to include in the fitting. This is particularly
        important for lowpass filtered data such as that post demodulation
        if the data is not downsampled after lowpass filtering.
    merge_name : bool
        If ``merge_fit`` is True then addes into axis manager with merge_name.
    merge_psd : bool
        If ``merge_psd`` is True then adds fres and Pxx to the axis manager.
    mask : bool
        If True, (nusamps,) mask is taken from aman.psd_mask, or calculated on the fly.
        Can also be a 1d (nusamps,) bool array True for good samples to keep.
        Can also be a Ranges in which case it will be inverted before application.

    fixed_param : str
        This accepts 'wn' or 'alpha' or None. If 'wn' ('alpha') is given, 
        white noise level (alpha) is fixed to the wn_est (alpha_est).
    binning : bool
        True to bin the psd before fitting.
        The binning is determined by the 'unbinned_mode' and 'base' params.
    unbinned_mode : int
        First Fourier modes up to this number are left un-binned.
    base : float (> 1)
        Base of the logspace bins.
    freq_spacing : float
        The approximate desired frequency spacing of the PSD. Passed to calc_psd.
    subscan : bool
        If True, fit noise on subscans.
    label_axis : str
        The name of LabelAxis in the input aman.
        Default is ``dets``.
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
        psdargs['noverlap'] = psdargs.get('noverlap', 0)
        f, pxx, nseg = calc_psd(
            aman,
            signal=signal,
            timestamps=aman.timestamps,
            freq_spacing=freq_spacing,
            merge=merge_psd,
            subscan=subscan,
            full_output=True,
            label_axis=label_axis,
            **psdargs,
        )
    if np.any(mask):
        if isinstance(mask, np.ndarray):
            pass
        elif isinstance(mask, Ranges):
            mask = ~mask.mask()
        elif mask == True:
            if 'psd_mask' in aman:
                mask = ~aman.psd_mask.mask()
            else: # Calculate on the fly
                mask = ~(get_psd_mask(aman, f=f, merge=False).mask())
        else:
            raise ValueError("mask should be an ndarray or True")
        f = f[mask]
        pxx = pxx[:, mask]

    if subscan:
        fit_noise_model_kwargs = {"fknee_est": fknee_est, "wn_est": wn_est, "alpha_est": alpha_est,
                                  "lowf": lowf, "f_max": f_max, "fixed_param": fixed_param,
                                  "binning": binning, "unbinned_mode": unbinned_mode, "base": base,
                                  "freq_spacing": freq_spacing}
        fitout, covout = _fit_noise_model_subscan(aman, signal,  f, pxx, fit_noise_model_kwargs)
        axis_map_fit = [(0, label_axis), (1, "noise_model_coeffs"), (2, aman.subscans)]
        axis_map_cov = [(0, label_axis), (1, "noise_model_coeffs"), (2, "noise_model_coeffs"), (3, aman.subscans)]
    else:
        eix = np.argmin(np.abs(f - f_max))
        if lowf is None:
            six = 1
        else:
            six = np.argmin(np.abs(f - lowf))
        f = f[six:eix]
        pxx = pxx[:, six:eix]
        bin_size = 1
        
        # --- sensible fknee bounds from the band we are actually fitting ---
        fpos = f[f > 0]
        fmin_pos = float(np.nanmin(fpos)) if fpos.size else 1e-6
        #fknee_lb = max(1e-6, 0.25 * fmin_pos)          # lower bound ~ fraction of lowest f
        fknee_lb = max(1e-6, 0.10 * fmin_pos)
        fknee_ub = None #f_max / 2.0 if np.isfinite(f_max) else None  # conservative upper bound
        
        # binning
        if binning == True:
            f, pxx, bin_size = get_binned_psd(aman, f=f, pxx=pxx, unbinned_mode=unbinned_mode,
                                              base=base, merge=False)
        fitout = np.zeros((aman[label_axis].count, 3))
        # This is equal to np.sqrt(np.diag(cov)) when doing curve_fit
        covout = np.zeros((aman[label_axis].count, 3, 3))
        if isinstance(wn_est, (int, float)):
            wn_est = np.full(aman[label_axis].count, wn_est)
        elif len(wn_est)!=aman[label_axis].count:
            print('Size of wn_est must be equal to aman.dets.count or a single value.')
            return
        if isinstance(fknee_est, (int, float)):
            fknee_est = np.full(aman[label_axis].count, fknee_est)
        elif len(fknee_est)!=aman[label_axis].count:
            print('Size of fknee_est must be equal to aman.dets.count or a single value.')
            return
        if isinstance(alpha_est, (int, float)):
            alpha_est = np.full(aman[label_axis].count, alpha_est)
        elif len(alpha_est)!=aman[label_axis].count:
            print('Size of alpha_est must be equal to aman.dets.count or a single value.')
            return
        if fixed_param == None:
            initial_params = np.array([wn_est, fknee_est, alpha_est])
            bounds = ((sys.float_info.min, None), (fknee_lb, fknee_ub), (0, 4.0)) #TODO add to args 0,4
        if fixed_param == "wn":
            initial_params = np.array([fknee_est, alpha_est])
            fixed = wn_est
            bounds = ((fknee_lb, fknee_ub), (0, 4.0)) #TODO add to args 0,4
        if fixed_param == "alpha":
            initial_params = np.array([wn_est, fknee_est])
            fixed = alpha_est
            bounds = ((sys.float_info.min, None), (fknee_lb, fknee_ub))

        for i in range(len(pxx)):
            p = pxx[i]
            #p0 = initial_params.T[i]
            #_fixed = {}
            #if fixed_param != None:
            #    _fixed = {fixed_param: fixed[i]}
            #res = minimize(lambda params: neglnlike(params, f, p, bin_size=bin_size, **_fixed),
            #               p0, bounds=bounds, method="Nelder-Mead")
            p0 = initial_params.T[i]
            # clip initial guess into bounds (keeps optimizer inside feasible region)
            def _clip(p, bnds):
                q = np.array(p, dtype=float)
                for j, (lo, hi) in enumerate(bnds):
                    if lo is not None: q[j] = max(q[j], lo)
                    if hi is not None: q[j] = min(q[j], hi)
                return q
            p0 = _clip(p0, bounds)

            _fixed = {}
            if fixed_param != None:
                _fixed = {fixed_param: fixed[i]}

            # bounded optimizer that actually respects bounds
            res = minimize(lambda params: neglnlike(params, f, p, bin_size=bin_size, **_fixed),
                           p0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 200})

            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p, bin_size=bin_size, **_fixed), full_output=True)
                    hessian_ndt, _ = Hfun(res["x"])
                    # Inverse of the hessian is an estimator of the covariance matrix
                    # sqrt of the diagonals gives you the standard errors.
                    #covout_i = np.linalg.inv(hessian_ndt)  
                    covout_i = np.linalg.pinv(hessian_ndt + 1e-12 * np.eye(hessian_ndt.shape[0]))          
                except np.linalg.LinAlgError:
                    print(
                        f"Cannot calculate Hessian for detector {aman[label_axis].vals[i]} skipping. (LinAlgError)"
                    )
                    covout_i = np.full((len(p0), len(p0)), np.nan)
                except IndexError:
                    print(
                        f"Cannot calculate Hessian for detector {aman[label_axis].vals[i]} skipping. (IndexError)"
                    )
                    covout_i = np.full((len(p0), len(p0)), np.nan)
                except RuntimeWarning as e:
                    covout_i = np.full((len(p0), len(p0)), np.nan)
                    print(f'RuntimeWarning: {e}\n Hessian failed because results are: {res["x"]}, for det: {aman[label_axis].vals[i]}')
            fitout_i = res.x
            if fixed_param == "wn":
                covout_i = np.insert(covout_i, 0, 0, axis=0)
                covout_i = np.insert(covout_i, 0, 0, axis=1)
                covout_i[0][0] = np.nan
                fitout_i = np.insert(fitout_i, 0, wn_est[i])
            elif fixed_param == "alpha":
                covout_i = np.insert(covout_i, 2, 0, axis=0)
                covout_i = np.insert(covout_i, 2, 0, axis=1)
                covout_i[2][2] = np.nan
                fitout_i = np.insert(fitout_i, 2, alpha_est[i])
            
            # Build a full 3x3 covariance matrix for convenience (even if a param is fixed)
            Cfull = np.full((3, 3), np.nan)

            if covout_i is None:
                pass  # leave NaNs
            elif covout_i.shape == (3, 3):
                # already padded to [wn, fknee, alpha]
                Cfull[:] = covout_i
            elif covout_i.shape == (2, 2):
                # unpadded; place it depending on which param is fixed
                if fixed_param == "wn":
                    # cov over [fknee, alpha] -> rows/cols 1..2
                    Cfull[1:, 1:] = covout_i
                elif fixed_param == "alpha":
                    # cov over [wn, fknee] -> rows/cols 0..1
                    Cfull[:2, :2] = covout_i
                else:
                    # unexpected but safe fallback: assume it's [fknee, alpha]
                    Cfull[1:, 1:] = covout_i
            else:
                # unexpected shape; keep NaNs
                pass

            # Now you can safely compute correlation etc.
            fknee_fit = float(fitout_i[1])
            alpha_fit = float(fitout_i[2])

            rho_fa = 0.0
            if (np.isfinite(Cfull[1,1]) and Cfull[1,1] > 0 and
                np.isfinite(Cfull[2,2]) and Cfull[2,2] > 0 and
                np.isfinite(Cfull[1,2])):
                rho_fa = Cfull[1,2] / np.sqrt(Cfull[1,1] * Cfull[2,2])

            at_floor   = (fknee_fit <= fknee_lb * (1 + 1e-4))
            steep_a    = (alpha_fit >= 3.5)  # tweak threshold if desired
            high_corr  = (abs(rho_fa) >= 0.9)

            if fixed_param == "wn" and at_floor and (steep_a or high_corr):
                #print(f"Detector {aman.dets.vals[i]}: refitting with wn free (was at fknee floor, alpha={alpha_fit:.2f}, rho={rho_fa:.2f})")
                # --- targeted second pass: narrow alpha around prior estimate ---
                alpha_lo = max(0.5, alpha_est[i] - 0.5)
                alpha_hi = min(4.0, alpha_est[i] + 0.5)
                bnds2 = ((fknee_lb, fknee_ub), (alpha_lo, alpha_hi))

                p0_2 = np.array([
                    np.clip(fknee_est[i], fknee_lb, fknee_ub),
                    np.clip(alpha_est[i], alpha_lo, alpha_hi)
                ], dtype=float)

                try:
                    res2 = minimize(lambda params: neglnlike(params, f, p, bin_size=bin_size, **{'wn': wn_est[i]}),
                                    p0_2, bounds=bnds2, method="L-BFGS-B", options={"maxiter": 200})
                    if (res2.fun < res.fun) or not np.isfinite(res.fun):
                        res = res2
                        fitout_i = np.insert(res.x, 0, wn_est[i])  # [wn_fixed, fknee, alpha]
                        # recompute covariance at the new optimum (use pinv for robustness)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("error")
                            Hfun = ndt.Hessian(lambda params: neglnlike(params, f, p, bin_size=bin_size, **{'wn': wn_est[i]}),
                                            full_output=True)
                            H2, _ = Hfun(res.x)
                            cov22 = np.linalg.pinv(H2 + 1e-12*np.eye(H2.shape[0]))
                        covout_i = np.full((3,3), np.nan)
                        covout_i[1:,1:] = cov22
                        Cfull[:] = covout_i  # keep Cfull in sync if you use it later
                except Exception:
                    pass  # keep first-pass result if second pass fails
            
            covout[i] = covout_i
            fitout[i] = fitout_i
        axis_map_fit = [(0, label_axis), (1, "noise_model_coeffs")]
        axis_map_cov = [(0, label_axis), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")]

    noise_model_coeffs = ["white_noise", "fknee", "alpha"]

    noise_fit_stats = core.AxisManager(
        aman[label_axis],
        core.LabelAxis(
            name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U11")
        ),
    )
    noise_fit_stats.wrap("fit", fitout, axis_map_fit)
    noise_fit_stats.wrap("cov", covout, axis_map_cov)

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats
'''

def fit_noise_model(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs={},
    fknee_est=1,
    wn_est=4E-5,
    alpha_est=3.4,
    merge_fit=False,
    lowf=None,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    mask=False,
    fixed_param=None,          # None | "wn" | "alpha"
    binning=False,
    unbinned_mode=3,
    base=1.05,
    freq_spacing=None,
    subscan=False,
    label_axis='dets',
    use_logspace=False,        # NEW: optimize in log-space if True
    method="L-BFGS-B",         # NEW: optimizer method (respects bounds)
    options=None               # NEW: scipy.optimize.minimize options
):
    """
    Fits noise model P(f) = wn^2 * [1 + (fknee / f)^alpha] to the PSD.

    If use_logspace=True, parameters are optimized in log space and
    neglnlike(..., use_logspace=True) is used internally.
    """
    if options is None:
        options = {"maxiter": 200}

    if signal is None:
        signal = aman.signal

    # --- PSD (compute if not provided) ---
    if f is None or pxx is None:
        psdargs['noverlap'] = psdargs.get('noverlap', 0)
        f, pxx, nseg = calc_psd(
            aman,
            signal=signal,
            timestamps=aman.timestamps,
            freq_spacing=freq_spacing,
            merge=merge_psd,
            subscan=subscan,
            full_output=True,
            label_axis=label_axis,
            **psdargs,
        )

    # --- Optional mask handling ---
    if np.any(mask):
        if isinstance(mask, np.ndarray):
            pass
        elif isinstance(mask, Ranges):
            mask = ~mask.mask()
        elif mask is True:
            if 'psd_mask' in aman:
                mask = ~aman.psd_mask.mask()
            else:
                mask = ~(get_psd_mask(aman, f=f, merge=False).mask())
        else:
            raise ValueError("mask should be an ndarray or True")
        f = f[mask]
        pxx = pxx[:, mask]

    # --- Subscan path delegates to helper (propagate use_logspace/method/options) ---
    if subscan:
        fit_noise_model_kwargs = {
            "fknee_est": fknee_est, "wn_est": wn_est, "alpha_est": alpha_est,
            "lowf": lowf, "f_max": f_max, "fixed_param": fixed_param,
            "binning": binning, "unbinned_mode": unbinned_mode, "base": base,
            "freq_spacing": freq_spacing, "use_logspace": use_logspace,
            "method": method, "options": options, "merge_psd": False,
            "label_axis": label_axis,
        }
        fitout, covout = _fit_noise_model_subscan(aman, signal, f, pxx, fit_noise_model_kwargs)
        axis_map_fit = [(0, label_axis), (1, "noise_model_coeffs"), (2, aman.subscans)]
        axis_map_cov = [(0, label_axis), (1, "noise_model_coeffs"), (2, "noise_model_coeffs"), (3, aman.subscans)]

    else:
        # --- Frequency window selection ---
        eix = np.argmin(np.abs(f - f_max))
        six = 1 if lowf is None else np.argmin(np.abs(f - lowf))
        f = f[six:eix]
        pxx = pxx[:, six:eix]

        # --- Optional log binning ---
        bin_size = 1.0
        if binning:
            f, pxx, bin_size = get_binned_psd(
                aman, f=f, pxx=pxx,
                unbinned_mode=unbinned_mode, base=base, merge=False
            )

        # --- fknee lower bound from fitted band ---
        fpos = f[f > 0]
        fmin_pos = float(np.nanmin(fpos)) if fpos.size else 1e-6
        fknee_lb = max(1e-6, 0.10 * fmin_pos)
        fknee_ub = None

        ndet = pxx.shape[0]
        fitout = np.zeros((ndet, 3), dtype=float)
        covout = np.zeros((ndet, 3, 3), dtype=float)

        def _as_arr(x, n, name):
            if isinstance(x, (int, float)): return np.full(n, float(x))
            x = np.asarray(x, dtype=float)
            if x.size != n:
                raise ValueError(f"{name} must be scalar or length {n}")
            return x

        def _clip1(v, lo, hi):
            v2 = float(v)
            if lo is not None: v2 = max(v2, lo)
            if hi is not None: v2 = min(v2, hi)
            return v2

        wn_est_arr    = _as_arr(wn_est,    ndet, "wn_est")
        fknee_est_arr = _as_arr(fknee_est, ndet, "fknee_est")
        alpha_est_arr = _as_arr(alpha_est, ndet, "alpha_est")

        for i in range(ndet):
            p = pxx[i]

            # --- Build initial guess and bounds in chosen space ---
            if not use_logspace:
                if fixed_param is None:
                    p0 = np.array([wn_est_arr[i], fknee_est_arr[i], alpha_est_arr[i]], float)
                    bnds = ((np.finfo(float).tiny, None), (fknee_lb, fknee_ub), (0.0, 4.0))
                elif fixed_param == "wn":
                    p0 = np.array([fknee_est_arr[i], alpha_est_arr[i]], float)
                    bnds = ((fknee_lb, fknee_ub), (0.0, 4.0))
                elif fixed_param == "alpha":
                    p0 = np.array([wn_est_arr[i], fknee_est_arr[i]], float)
                    bnds = ((np.finfo(float).tiny, None), (fknee_lb, fknee_ub))
                else:
                    raise ValueError("fixed_param must be None, 'wn', or 'alpha'")
                p0 = np.array([_clip1(p0[j], *bnds[j]) for j in range(len(p0))], float)
            else:
                log_wn0    = np.log(max(wn_est_arr[i],    1e-30))
                log_fknee0 = np.log(max(fknee_est_arr[i], 1e-30))
                if fixed_param is None:
                    p0 = np.array([log_wn0, log_fknee0, alpha_est_arr[i]], float)
                    log_wn_lo, log_wn_hi       = np.log(1e-12), np.log(1e-2)
                    log_fknee_lo, log_fknee_hi = np.log(fknee_lb), (np.log(fknee_ub) if fknee_ub else np.inf)
                    bnds = ((log_wn_lo, log_wn_hi), (log_fknee_lo, log_fknee_hi), (0.5, 4.0))
                elif fixed_param == "wn":
                    p0 = np.array([log_fknee0, alpha_est_arr[i]], float)
                    log_fknee_lo, log_fknee_hi = np.log(fknee_lb), (np.log(fknee_ub) if fknee_ub else np.inf)
                    bnds = ((log_fknee_lo, log_fknee_hi), (0.5, 4.0))
                elif fixed_param == "alpha":
                    p0 = np.array([log_wn0, log_fknee0], float)
                    log_wn_lo, log_wn_hi       = np.log(1e-12), np.log(1e-2)
                    log_fknee_lo, log_fknee_hi = np.log(fknee_lb), (np.log(fknee_ub) if fknee_ub else np.inf)
                    bnds = ((log_wn_lo, log_wn_hi), (log_fknee_lo, log_fknee_hi))
                else:
                    raise ValueError("fixed_param must be None, 'wn', or 'alpha'")
                p0 = np.array([_clip1(p0[j], *bnds[j]) for j in range(len(p0))], float)

            fixed_dict = {}
            if fixed_param == "wn":
                fixed_dict = {"wn": wn_est_arr[i]}
            elif fixed_param == "alpha":
                fixed_dict = {"alpha": alpha_est_arr[i]}

            # --- Optimize unified Whittle NLL in chosen space ---
            res = minimize(
                lambda params: neglnlike(
                    params, f, p,
                    bin_size=bin_size,
                    fixed_param=(fixed_dict if fixed_dict else None),
                    use_logspace=use_logspace,
                    fknee_floor_frac=0.25,
                    alpha_bounds=(0.0, 10.0),
                ),
                p0, bounds=bnds, method=method, options=options
            )

            # --- Map solution to linear [wn, fknee, alpha] ---
            if not use_logspace:
                if fixed_param is None:
                    wn, fknee, alpha = res.x
                elif fixed_param == "wn":
                    fknee, alpha = res.x
                    wn = wn_est_arr[i]
                else:
                    wn, fknee = res.x
                    alpha = alpha_est_arr[i]
            else:
                if fixed_param is None:
                    log_wn, log_fknee, alpha = res.x
                    wn, fknee = np.exp(log_wn), np.exp(log_fknee)
                elif fixed_param == "wn":
                    log_fknee, alpha = res.x
                    wn, fknee = wn_est_arr[i], np.exp(log_fknee)
                else:
                    log_wn, log_fknee = res.x
                    wn, fknee = np.exp(log_wn), np.exp(log_fknee)

            fitout[i] = [wn, fknee, alpha]

            # --- Covariance via Hessian in chosen space, then -> linear 3×3 ---
            Cfull = np.full((3, 3), np.nan, dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    Hfun = ndt.Hessian(lambda params: neglnlike(
                        params, f, p,
                        bin_size=bin_size,
                        fixed_param=(fixed_dict if fixed_dict else None),
                        use_logspace=use_logspace,
                        fknee_floor_frac=0.25,
                        alpha_bounds=(0.0, 10.0),
                    ), full_output=True)
                    H, _ = Hfun(res.x)
                    C_param = np.linalg.pinv(H + 1e-12*np.eye(H.shape[0]))

                if not use_logspace:
                    if fixed_param is None:
                        Cfull = C_param
                    elif fixed_param == "wn":
                        Cfull[1:, 1:] = C_param
                    else:
                        Cfull[:2, :2] = C_param
                else:
                    if fixed_param is None:
                        J = np.diag([wn, fknee, 1.0])  # d(wn)/d(log_wn)=wn, d(fknee)/d(log_fknee)=fknee
                        Cfull = J @ C_param @ J.T
                    elif fixed_param == "wn":
                        J = np.array([[fknee, 0.0],
                                      [0.0,   1.0]])
                        C_lin = J @ C_param @ J.T
                        Cfull[1:, 1:] = C_lin
                    else:
                        J = np.array([[wn,    0.0],
                                      [0.0, fknee]])
                        C_lin = J @ C_param @ J.T
                        Cfull[:2, :2] = C_lin
            except Exception:
                pass

            covout[i] = Cfull

        axis_map_fit = [(0, label_axis), (1, "noise_model_coeffs")]
        axis_map_cov = [(0, label_axis), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")]

    # --- Wrap outputs ---
    noise_model_coeffs = ["white_noise", "fknee", "alpha"]
    noise_fit_stats = core.AxisManager(
        aman[label_axis],
        core.LabelAxis(name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U11")),
    )
    noise_fit_stats.wrap("fit", fitout, axis_map_fit)
    noise_fit_stats.wrap("cov", covout, axis_map_cov)

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats



def fit_noise_model_test(
    aman,
    signal=None,
    f=None,
    pxx=None,
    psdargs={},
    fknee_est=1,
    wn_est=4E-5,
    alpha_est=3.4,
    merge_fit=False,
    lowf=None,
    f_max=100,
    merge_name="noise_fit_stats",
    merge_psd=True,
    mask=False,
    fixed_param=None,
    binning=False,
    unbinned_mode=3,
    base=1.05,
    freq_spacing=None,
    subscan=False,
    method="L-BFGS-B",   # or "Powell", "Nelder-Mead" (but L-BFGS-B respects bounds)
    options={"maxiter": 200}
):
    """
    Fits noise model with white and 1/f noise to the PSD of binned signal.
    Stable (log-space) formulation with per-detector bounds derived from estimates.

    Model: P(f) = wn^2 * [1 + (fknee / f)^alpha]
    We fit params in log-space: theta = [log_wn, log_fknee, alpha]
    """

    # ----------------------- stable helpers -----------------------
    def noise_model_stable(fx, params, **fixed_param_):
        """
        Stable version of wn^2 * (1 + (fknee/f)^alpha) using log-space params.
        params = [log_wn, log_fknee, alpha] (or 2-long if one param fixed).
        """
        if 'wn' in fixed_param_:
            log_wn = np.log(fixed_param_['wn'])
            log_fknee, alpha = params
        elif 'alpha' in fixed_param_:
            alpha = fixed_param_['alpha']
            log_wn, log_fknee = params
        else:
            log_wn, log_fknee, alpha = params

        f_safe = np.maximum(fx, 1e-8)
        t = alpha * (log_fknee - np.log(f_safe))
        t = np.clip(t, -50, 50)  # avoid overflow
        log_model = 2.0 * log_wn + np.log1p(np.exp(t))
        return np.exp(log_model)

    def neglnlike_stable(params, x, y, bin_size=1, **fixed_param_):
        m = noise_model_stable(x, params, **fixed_param_)
        if not np.all(np.isfinite(m)) or np.any(m <= 0):
            return 1e30
        out = np.sum((np.log(m) + y / m) * bin_size)
        if not np.isfinite(out):
            return 1e30
        return out

    def _perdet_log_bounds_from_estimates(
        wn_est_, fknee_est_, alpha_est_, f_max_,
        fixed_param_=None,
        # multiplicative padding around estimates
        wn_factor_lo=0.25, wn_factor_hi=4.0,
        fknee_factor_lo=0.25, fknee_factor_hi=4.0,
        alpha_pad_lo=0.75, alpha_pad_hi=0.75,
        # absolute clamps
        wn_global=(1e-12, 1e-2),
        fknee_global=(1e-4, None),  # upper bound will be f_max_/2
        alpha_global=(0.5, 4.0),
    ):
        wn_est_ = np.asarray(wn_est_, dtype=float)
        fknee_est_ = np.asarray(fknee_est_, dtype=float)
        alpha_est_ = np.asarray(alpha_est_, dtype=float)
        ndet = max(wn_est_.size, fknee_est_.size, alpha_est_.size)

        if wn_est_.size == 1: wn_est_ = np.full(ndet, wn_est_.item())
        if fknee_est_.size == 1: fknee_est_ = np.full(ndet, fknee_est_.item())
        if alpha_est_.size == 1: alpha_est_ = np.full(ndet, alpha_est_.item())

        # expand around estimates
        wn_lo_lin    = wn_est_    * wn_factor_lo
        wn_hi_lin    = wn_est_    * wn_factor_hi
        fknee_lo_lin = fknee_est_ * fknee_factor_lo
        fknee_hi_lin = fknee_est_ * fknee_factor_hi
        alpha_lo_lin = alpha_est_ - alpha_pad_lo
        alpha_hi_lin = alpha_est_ + alpha_pad_hi

        # clamp to globals
        wn_lo_lin = np.maximum(wn_lo_lin, wn_global[0])
        wn_hi_lin = np.minimum(wn_hi_lin, wn_global[1])

        fknee_upper = (f_max_/2.0) if f_max_ is not None else None
        fknee_lo_lin = np.maximum(fknee_lo_lin, fknee_global[0])
        if fknee_upper is not None:
            fknee_hi_lin = np.minimum(fknee_hi_lin, fknee_upper)

        alpha_lo_lin = np.maximum(alpha_lo_lin, alpha_global[0])
        alpha_hi_lin = np.minimum(alpha_hi_lin, alpha_global[1])

        # sanitize inversions/NaNs
        def _sanitize(lo, hi, lo_def, hi_def):
            lo2, hi2 = lo.copy(), hi.copy()
            bad = ~np.isfinite(lo2) | ~np.isfinite(hi2) | (hi2 <= lo2)
            lo2[bad] = lo_def
            hi2[bad] = hi_def
            return lo2, hi2

        wn_lo_lin, wn_hi_lin = _sanitize(wn_lo_lin, wn_hi_lin, wn_global[0], wn_global[1])
        fknee_lo_lin, fknee_hi_lin = _sanitize(fknee_lo_lin, fknee_hi_lin,
                                               fknee_global[0],
                                               fknee_upper if fknee_upper is not None else 1.0)
        alpha_lo_lin, alpha_hi_lin = _sanitize(alpha_lo_lin, alpha_hi_lin,
                                               alpha_global[0], alpha_global[1])

        # to log-space for wn/fknee
        log_wn_bnds    = np.stack([np.log(wn_lo_lin),    np.log(wn_hi_lin)],    axis=1)
        log_fknee_bnds = np.stack([np.log(fknee_lo_lin), np.log(fknee_hi_lin)], axis=1)
        alpha_bnds     = np.stack([alpha_lo_lin,         alpha_hi_lin],         axis=1)

        if fixed_param_ == 'wn':
            return None, log_fknee_bnds, alpha_bnds
        elif fixed_param_ == 'alpha':
            return log_wn_bnds, log_fknee_bnds, None
        else:
            return log_wn_bnds, log_fknee_bnds, alpha_bnds
    # --------------------- end helpers ----------------------------

    if signal is None:
        signal = aman.signal

    # PSD (compute if not provided)
    if f is None or pxx is None:
        psdargs['noverlap'] = psdargs.get('noverlap', 0)
        f, pxx, nseg = calc_psd(
            aman,
            signal=signal,
            timestamps=aman.timestamps,
            freq_spacing=freq_spacing,
            merge=merge_psd,
            subscan=subscan,
            full_output=True,
            **psdargs,
        )

    # Optional masking
    if np.any(mask):
        if isinstance(mask, np.ndarray):
            pass
        elif isinstance(mask, Ranges):
            mask = ~mask.mask()
        elif mask is True:
            if 'psd_mask' in aman:
                mask = ~aman.psd_mask.mask()
            else:
                mask = ~(get_psd_mask(aman, f=f, merge=False).mask())
        else:
            raise ValueError("mask should be an ndarray or True")
        f = f[mask]
        pxx = pxx[:, mask]

    # Subscan path (delegates to per-subscan calls to this function)
    if subscan:
        fit_noise_model_kwargs = {
            "fknee_est": fknee_est, "wn_est": wn_est, "alpha_est": alpha_est,
            "lowf": lowf, "f_max": f_max, "fixed_param": fixed_param,
            "binning": binning, "unbinned_mode": unbinned_mode, "base": base,
            "freq_spacing": freq_spacing, "merge_psd": False,
            "method": method, "options": options
        }
        fitout, covout = _fit_noise_model_subscan(aman, signal, f, pxx, fit_noise_model_kwargs)
        axis_map_fit = [(0, "dets"), (1, "noise_model_coeffs"), (2, aman.subscans)]
        axis_map_cov = [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs"), (3, aman.subscans)]
    else:
        # Frequency window
        eix = np.argmin(np.abs(f - f_max))
        six = 1 if lowf is None else np.argmin(np.abs(f - lowf))
        f = f[six:eix]
        pxx = pxx[:, six:eix]
        
        # Optional log binning
        bin_size = 1
        if binning:
            f, pxx, bin_size = get_binned_psd(aman, f=f, pxx=pxx,
                                              unbinned_mode=unbinned_mode,
                                              base=base, merge=False)

        ndet = pxx.shape[0]
        fitout = np.zeros((ndet, 3), dtype=float)
        covout = np.zeros((ndet, 3, 3), dtype=float)

        # Ensure per-det arrays for estimates
        def _as_array(x, name):
            if isinstance(x, (int, float)): return np.full(ndet, float(x))
            x = np.asarray(x, dtype=float)
            if x.size != ndet:
                raise ValueError(f"{name} must be scalar or length ndet ({ndet}), got {x.size}.")
            return x
        wn_est_arr    = _as_array(wn_est, "wn_est")
        fknee_est_arr = _as_array(fknee_est, "fknee_est")
        alpha_est_arr = _as_array(alpha_est, "alpha_est")

        # Per-detector bounds from estimates
        log_wn_bnds_arr, log_fknee_bnds_arr, alpha_bnds_arr = _perdet_log_bounds_from_estimates(
            wn_est_arr, fknee_est_arr, alpha_est_arr, f_max, fixed_param_=fixed_param,
            wn_factor_lo=0.25, wn_factor_hi=4.0,
            fknee_factor_lo=0.25, fknee_factor_hi=4.0,
            alpha_pad_lo=0.75, alpha_pad_hi=0.75,
            wn_global=(1e-12, 1e-2),
            fknee_global=(1e-4, None),
            alpha_global=(0.5, 4.0),
        )

        # Fit each detector robustly
        for i in range(ndet):
            p = pxx[i]

            # Default outputs (avoid UnboundLocalError)
            fit_linear = np.array([np.nan, np.nan, np.nan], dtype=float)
            C = np.full((3, 3), np.nan, dtype=float)

            # Initial guesses in log-space + per-det bounds
            try:
                if fixed_param is None:
                    p0 = np.array([np.log(wn_est_arr[i]),
                                   np.log(fknee_est_arr[i]),
                                   alpha_est_arr[i]], dtype=float)
                    # clip into bounds
                    p0[0] = np.clip(p0[0], log_wn_bnds_arr[i,0],    log_wn_bnds_arr[i,1])
                    p0[1] = np.clip(p0[1], log_fknee_bnds_arr[i,0], log_fknee_bnds_arr[i,1])
                    p0[2] = np.clip(p0[2], alpha_bnds_arr[i,0],     alpha_bnds_arr[i,1])
                    bnds = (tuple(log_wn_bnds_arr[i]),
                            tuple(log_fknee_bnds_arr[i]),
                            tuple(alpha_bnds_arr[i]))
                elif fixed_param == "wn":
                    p0 = np.array([np.log(fknee_est_arr[i]),
                                   alpha_est_arr[i]], dtype=float)
                    p0[0] = np.clip(p0[0], log_fknee_bnds_arr[i,0], log_fknee_bnds_arr[i,1])
                    p0[1] = np.clip(p0[1], alpha_bnds_arr[i,0],     alpha_bnds_arr[i,1])
                    bnds = (tuple(log_fknee_bnds_arr[i]),
                            tuple(alpha_bnds_arr[i]))
                elif fixed_param == "alpha":
                    p0 = np.array([np.log(wn_est_arr[i]),
                                   np.log(fknee_est_arr[i])], dtype=float)
                    p0[0] = np.clip(p0[0], log_wn_bnds_arr[i,0],    log_wn_bnds_arr[i,1])
                    p0[1] = np.clip(p0[1], log_fknee_bnds_arr[i,0], log_fknee_bnds_arr[i,1])
                    bnds = (tuple(log_wn_bnds_arr[i]),
                            tuple(log_fknee_bnds_arr[i]))
                else:
                    raise ValueError("fixed_param must be None, 'wn', or 'alpha'")
            except Exception as e:
                print(f"[fit init] det {aman.dets.vals[i]}: {e}")
                fitout[i] = fit_linear
                covout[i] = C
                continue

            # Fixed dict
            fixed_dict = {}
            if fixed_param is not None:
                fixed_dict = {fixed_param: (wn_est_arr[i] if fixed_param == 'wn' else alpha_est_arr[i])}

            # Optimize
            try:
                res = minimize(lambda params: neglnlike_stable(params, f, p,
                                                               bin_size=bin_size, **fixed_dict),
                               p0, bounds=bnds, method=method, options=options)
            except Exception as e:
                print(f"[opt fail] det {aman.dets.vals[i]}: {e}")
                fitout[i] = fit_linear
                covout[i] = C
                continue

            # Map solution back to linear space
            try:
                if fixed_param is None:
                    log_wn, log_fknee, alpha = res.x
                    wn, fknee = np.exp(log_wn), np.exp(log_fknee)
                    fit_linear = np.array([wn, fknee, alpha], dtype=float)
                elif fixed_param == "wn":
                    log_fknee, alpha = res.x
                    fknee = np.exp(log_fknee)
                    fit_linear = np.array([wn_est_arr[i], fknee, alpha], dtype=float)
                else:  # fixed_param == "alpha"
                    log_wn, log_fknee = res.x
                    wn, fknee = np.exp(log_wn), np.exp(log_fknee)
                    fit_linear = np.array([wn, fknee, alpha_est_arr[i]], dtype=float)
            except Exception as e:
                print(f"[map fail] det {aman.dets.vals[i]}: {e}")

            # Covariance via Hessian in log-space (optional; robust pinv)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    Hfun = ndt.Hessian(lambda params: neglnlike_stable(params, f, p,
                                                                       bin_size=bin_size, **fixed_dict),
                                       full_output=True)
                    H, _ = Hfun(res.x)
                    cov_log = np.linalg.pinv(H)

                if fixed_param is None:
                    # params=[log_wn,log_fknee,alpha] -> [wn,fknee,alpha]
                    J = np.diag([np.exp(res.x[0]), np.exp(res.x[1]), 1.0])
                    C = J @ cov_log @ J.T
                elif fixed_param == "wn":
                    # params=[log_fknee,alpha] -> [wn_fixed,fknee,alpha] (pad to 3x3)
                    J = np.array([[np.exp(res.x[0]), 0.0],
                                  [0.0,              1.0]])
                    cov_lin = J @ cov_log @ J.T
                    C = np.full((3,3), np.nan); C[1:,1:] = cov_lin
                else:  # fixed_param == "alpha"
                    # params=[log_wn,log_fknee] -> [wn,fknee,alpha_fixed] (pad to 3x3)
                    J = np.array([[np.exp(res.x[0]), 0.0],
                                  [0.0,              np.exp(res.x[1])]])
                    cov_lin = J @ cov_log @ J.T
                    C = np.full((3,3), np.nan); C[:2,:2] = cov_lin
            except Exception:
                # keep C as NaNs if Hessian fails
                pass

            fitout[i] = fit_linear
            covout[i] = C

        axis_map_fit = [(0, "dets"), (1, "noise_model_coeffs")]
        axis_map_cov = [(0, "dets"), (1, "noise_model_coeffs"), (2, "noise_model_coeffs")]

    # Wrap outputs
    noise_model_coeffs = ["white_noise", "fknee", "alpha"]
    noise_fit_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(name="noise_model_coeffs", vals=np.array(noise_model_coeffs, dtype="<U11")),
    )
    noise_fit_stats.wrap("fit", fitout, axis_map_fit)
    noise_fit_stats.wrap("cov", covout, axis_map_cov)

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

def log_binning(psd, unbinned_mode=3, base=1.05, mask=None,
                return_bin_size=False, drop_nan=False):
    """
    Function to bin PSD or frequency. First several Fourier modes are left un-binned.
    Fourier modes higher than that are averaged into logspace bins.

    Parameters
    ----------
    psd : numpy.ndarray
        PSD (or frequency) to be binned. Can be a 1D or 2D array.
    unbinned_mode : int, optional
        First Fourier modes up to this number are left un-binned. Defaults to 3.
    base : float, optional
        Base of the logspace bins. Must be greater than 1. Defaults to 1.05.
    mask : numpy.ndarray, optional
        Mask for psd. If all values in a bin are masked, the value becomes np.nan.
        Should be a 1D array.
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

    is_1d = psd.ndim == 1

    # Ensure psd is at least 2D for consistent processing
    psd = np.atleast_2d(psd)
    num_signals, num_samples = psd.shape[:2]
    
    if mask is not None:
        # Ensure mask is at least 2D and has the same shape as psd
        mask = np.atleast_2d(mask)
        if mask.shape[1] != num_samples:
            raise ValueError("Mask must have the same number of columns as psd")
        mask = np.tile(mask, (num_signals,) + psd.shape[2:] + (1,))
        mask = np.moveaxis(mask, -1, 1)
        psd = np.ma.masked_array(psd, mask=mask)
    
    # Initialize the binned PSD and optionally the bin sizes
    binned_psd = np.zeros((num_signals, unbinned_mode + 1) + psd.shape[2:])
    binned_psd[:, :unbinned_mode + 1] = psd[:, :unbinned_mode + 1]
    bin_size = np.ones((num_signals, unbinned_mode + 1)) if return_bin_size else None

    # Determine the number of bins and their indices
    N = int(np.ceil(np.emath.logn(base, num_samples - unbinned_mode)))
    binning_idx = np.unique(np.logspace(base, N, N, base=base, dtype=int) + unbinned_mode - 1)
    
    # Bin the PSD values for each signal
    new_binned_psd = []
    new_bin_size = []
    for start, end in zip(binning_idx[:-1], binning_idx[1:]):
        bin_mean = np.nanmean(psd[:, start:end], axis=1)
        new_binned_psd.append(bin_mean)
        if return_bin_size:
            new_bin_size.append(end - start)
    
    # Convert lists to numpy arrays and concatenate with initial values
    new_binned_psd = np.array(new_binned_psd)
    new_binned_psd = np.moveaxis(new_binned_psd, 0, 1)# Transpose to match dimensions
    binned_psd = np.hstack([binned_psd, new_binned_psd])
    if return_bin_size:
        new_bin_size = np.array(new_bin_size)
        bin_size = np.hstack([bin_size, np.tile(new_bin_size, (num_signals, 1))])

    if drop_nan:
        valid_indices = ~np.isnan(binned_psd).any(axis=0)
        binned_psd = binned_psd[:, valid_indices]
        if return_bin_size:
            bin_size = bin_size[:, valid_indices]

    if is_1d:
        binned_psd = binned_psd.flatten()
        if return_bin_size:
            bin_size = bin_size.flatten()

    if return_bin_size:
        return binned_psd, bin_size
    return binned_psd

'''
def _fit_noise_model_subscan(
    aman,
    signal,
    f,
    pxx,
    fit_noise_model_kwargs,
):
    """
    Fits noise model with white and 1/f noise to the PSD of signal subscans.
    Args are as for fit_noise_model.
    """
    fitout = np.empty((aman.dets.count, 3, aman.subscan_info.subscans.count))
    covout = np.empty((aman.dets.count, 3, 3, aman.subscan_info.subscans.count))

    per_subscan = {}
    for entry in ["fknee_est", "wn_est", "alpha_est"]:
        if (entry in fit_noise_model_kwargs):
            val = fit_noise_model_kwargs[entry]
            if isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[-1] == aman.subscan_info.subscans.count:
                per_subscan[entry] = fit_noise_model_kwargs.pop(entry)

    kwargs = fit_noise_model_kwargs.copy() if len(per_subscan) > 0 else fit_noise_model_kwargs
    for isub in range(aman.subscan_info.subscans.count):
        if np.all(np.isnan(pxx[...,isub])): # Subscan has been fully cut
            fitout[..., isub] = np.full((aman.dets.count, 3), np.nan)
            covout[..., isub] = np.full((aman.dets.count, 3, 3), np.nan)
        else:
            for entry in list(per_subscan.keys()):
                kwargs[entry] = per_subscan[entry][..., isub]
            noise_model = fit_noise_model(aman, f=f, pxx=pxx[...,isub], merge_fit=False, merge_psd=False, subscan=False, **kwargs)

            fitout[..., isub] = noise_model.fit
            covout[..., isub] = noise_model.cov

    return fitout, covout
'''

def _fit_noise_model_subscan(
    aman,
    signal,
    f,
    pxx,
    fit_noise_model_kwargs,
):
    """
    Fits noise model with white and 1/f noise to the PSD of signal subscans.
    Propagates use_logspace/method/options into per-subscan fits.
    """
    fitout = np.empty((aman.dets.count, 3, aman.subscan_info.subscans.count))
    covout = np.empty((aman.dets.count, 3, 3, aman.subscan_info.subscans.count))

    per_subscan = {}
    for entry in ["fknee_est", "wn_est", "alpha_est"]:
        if (entry in fit_noise_model_kwargs):
            val = fit_noise_model_kwargs[entry]
            if isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[-1] == aman.subscan_info.subscans.count:
                per_subscan[entry] = fit_noise_model_kwargs.pop(entry)

    kwargs = fit_noise_model_kwargs.copy() if len(per_subscan) > 0 else fit_noise_model_kwargs
    for isub in range(aman.subscan_info.subscans.count):
        if np.all(np.isnan(pxx[..., isub])):  # Subscan fully cut
            fitout[..., isub] = np.full((aman.dets.count, 3), np.nan)
            covout[..., isub] = np.full((aman.dets.count, 3, 3), np.nan)
        else:
            for entry in list(per_subscan.keys()):
                kwargs[entry] = per_subscan[entry][..., isub]
            # Ensure we do NOT merge PSD again or recurse into subscan
            noise_model_stats = fit_noise_model(
                aman, f=f, pxx=pxx[..., isub],
                merge_fit=False, merge_psd=False, subscan=False, **kwargs
            )
            fitout[..., isub] = noise_model_stats.fit
            covout[..., isub] = noise_model_stats.cov

    return fitout, covout


'''
def build_hpf_params_dict(
    filter_name,
    noise_fit=None,
    filter_params=None,
):
    """
    Build the filter parameter dictionary from a provided dictionary or from
    noise-fit results.

    Parameters
    ----------
    filter_name : str
        One of:
          - "counter_1_over_f" -> expects {"fk": ..., "n": ...}
          - "high_pass_butter4" -> expects {"fc": ...}
          - "high_pass_sine2"   -> expects {"cutoff": ..., "width": ...}
          - "low_pass_butter4"  -> expects {"fc": ...}
          - "low_pass_sine2"    -> expects {"cutoff": ..., "width": ...}
    noise_fit : AxisManager or None
        AxisManager with fields:
          - .fit: (ndet, 3) array ordered like noise_fit.noise_model_coeffs
          - .noise_model_coeffs.vals: list/array of names, e.g. ["white_noise","fknee","alpha"]
        If provided, mapped parameters are pulled from this.
    filter_params : dict or None
        Optional user-specified parameters. Used to:
          - Provide required params not derivable from noise_fit (e.g. "width").
          - Toggle per-detector output via 'each_detector': True/False.
          - For "counter_1_over_f": override field names via:
                'fk_field' (default "fknee"), 'n_field' (default "alpha"),
            and optionally scale fk via 'fk_scale' (default 1.0).

    Returns
    -------
    dict
        Parameter dict ready for the requested filter.

    Behavior
    --------
    - If noise_fit is None: returns filter_params unchanged (no inference).
    - If noise_fit is given:
        * Default: returns median parameters across detectors.
        * If filter_params['each_detector'] is True: returns per-detector vectors.
        * Special override for "counter_1_over_f":
            - If 'fk_field' or 'n_field' present, build from those columns (per-detector vectors);
              apply optional 'fk_scale' to fk values.
    """
    # Mapping from filter param names -> noise_fit model coefficient names
    pars_mapping = {
        "counter_1_over_f": {
            "fk": "fknee",    # knee frequency
            "n":  "alpha",    # 1/f slope
        },
        "high_pass_butter4": {
            "fc": "fknee",
        },
        "high_pass_sine2": {
            "cutoff": "fknee",
            "width":  None,   # must be supplied by user filter_params
        },
        "low_pass_butter4": {
            "fc": "fknee",
        },
        "low_pass_sine2": {
            "cutoff": "fknee",
            "width":  None,   # must be supplied by user filter_params
        },
    }

    if filter_name not in pars_mapping:
        raise NotImplementedError(
            f"{filter_name} params from noise fit are not implemented"
        )

    # If we cannot infer from noise_fit, just return provided params
    if noise_fit is None:
        if filter_params is None:
            raise ValueError(
                "noise_fit is None and no filter_params provided; nothing to build."
            )
        return filter_params

    if filter_params is None:
        filter_params = {}

    # Extract fit array and coefficient names
    coeff_names = list(noise_fit.noise_model_coeffs.vals)   # e.g., ["white_noise","fknee","alpha"]
    fit_arr     = np.asarray(noise_fit.fit)                 # shape (ndet, 3)

    # Helper: median or per-detector vectors from noise_fit
    each_detector = bool(filter_params.get("each_detector", False))
    if each_detector:
        # Return per-detector vectors for each coefficient name
        noise_fit_dict = {k: fit_arr[:, i] for i, k in enumerate(coeff_names)}
    else:
        # Return medians across detectors
        med = np.nanmedian(fit_arr, axis=0)
        noise_fit_dict = {k: med[i] for i, k in enumerate(coeff_names)}

    # Special HEAD feature: field-driven per-detector overrides for counter_1_over_f
    # If the user specifies fk_field/n_field, use those columns (always per-detector vectors)
    # and apply optional fk_scale.
    if filter_name == "counter_1_over_f" and (
        ("fk_field" in filter_params) or ("n_field" in filter_params)
    ):
        fk_field = filter_params.get("fk_field", "fknee")
        n_field  = filter_params.get("n_field",  "alpha")
        if fk_field not in coeff_names:
            raise ValueError(f"fk_field '{fk_field}' not found in noise_fit.noise_model_coeffs {coeff_names}")
        if n_field not in coeff_names:
            raise ValueError(f"n_field '{n_field}' not found in noise_fit.noise_model_coeffs {coeff_names}")

        fk_idx = coeff_names.index(fk_field)
        n_idx  = coeff_names.index(n_field)

        fk_vec = np.asarray(fit_arr[:, fk_idx], dtype=float)
        n_vec  = np.asarray(fit_arr[:, n_idx],  dtype=float)

        fk_scale = float(filter_params.get("fk_scale", 1.0))
        fk_vec = fk_vec * fk_scale

        params_dict = {}
        for k, v in pars_mapping[filter_name].items():
            if v is None:
                if k not in filter_params:
                    raise ValueError(
                        f"Required parameter '{k}' not found in filter_params and cannot be derived from noise_fit."
                    )
                params_dict[k] = filter_params[k]
            else:
                if v == "fknee":
                    params_dict[k] = fk_vec
                elif v == "alpha":
                    params_dict[k] = n_vec
                else:
                    raise ValueError(f"Unexpected mapping target '{v}' for {filter_name}")
        return params_dict

    # General path: build params from noise_fit_dict according to mapping
    # Optionally apply fk_scale to any parameter sourced from 'fknee'.
    fk_scale_global = float(filter_params.get("fk_scale", 1.0)) if "fk_scale" in filter_params else 1.0

    params_dict = {}
    for param_name, source_coeff in pars_mapping[filter_name].items():
        if source_coeff is None:
            # Must be provided by user
            if param_name not in filter_params:
                raise ValueError(
                    f"Required parameter '{param_name}' not found in filter_params and cannot be derived from noise_fit."
                )
            params_dict[param_name] = filter_params[param_name]
        else:
            if source_coeff not in noise_fit_dict:
                raise ValueError(
                    f"Coefficient '{source_coeff}' not present in noise_fit_dict keys {list(noise_fit_dict.keys())}"
                )
            val = noise_fit_dict[source_coeff]
            # Apply global scaling to fknee-derived values if requested
            if source_coeff == "fknee" and ("fk_scale" in filter_params):
                val = np.asarray(val, dtype=float) * fk_scale_global
            params_dict[param_name] = val

    return params_dict
'''

def build_hpf_params_dict(
    filter_name,
    noise_fit=None,
    filter_params=None,
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
            "counter_1_over_f": {
                "fk": "fknee", 
                "n": "alpha"
            },
            "high_pass_butter4": {
                "fc": "fknee",
            },
            "high_pass_sine2": {
                "cutoff": "fknee",
                "width": None
            },
            "low_pass_butter4": {
                "fc": "fknee",
            },
            "low_pass_sine2": {
                "cutoff": "fknee",
                "width": None
            },
        }

        if filter_name not in pars_mapping.keys():
            raise NotImplementedError(
                f"{filter_name} params from noise fit is not implemented"
            )
        
        noise_fit_array = noise_fit.fit
        noise_fit_params = noise_fit.noise_model_coeffs.vals

        each_detector = False
        if filter_params is not None:
            each_detector = filter_params.get('each_detector', False)
        if each_detector:
            noise_fit_dict = {
                k: noise_fit_array[:, i]
                for i, k in enumerate(noise_fit_params)
            }
        else:
            median_params = np.median(noise_fit_array, axis=0)
            noise_fit_dict = {
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
                params_dict[k] = noise_fit_dict[v]

        filter_params = params_dict
    
    return filter_params


def get_common_noise_params(
    aman,
    signal=None,
    merge=False,
    merge_name='common_mode',
    **fit_args
):
    """
    Fits noise model with white and 1/f noise to the PSD of common noise.
    This can be applied to any 1D data with same size as aman.samps.

    Args
    ----
    aman : AxisManager
        Axis manager which has samps axis aligned with signal.
    signal : nparray
        The pre-computed common mode sized 1 x nsamps, or
        the signal sized ndets x nsamps.
        When signal is given in 2D shape, the common mode across detectors
        is calculated with median method.
        
    merge : bool
        If True, ``aman_common`` is added into axis manager with merge_name.
    merge_name : str
        ``aman_common`` is added with this name.
    
    Returns
    -------
    noise_fit_stats : AxisManager
        The fit parameters of given common mode.

    """

    if signal is None:
        signal = np.nanmedian(aman.signal, axis=0)
    elif signal.shape == (aman.dets.count, aman.samps.count):
        signal = np.nanmedian(signal, axis=0)
    elif len(signal) != aman.samps.count:
        raise ValueError(
                        "The input signal is required to have same length as aman.samps."
                    )
    aman_common = core.AxisManager(core.LabelAxis('common', ['common']), aman.samps)
    aman_common.wrap('signal', np.array([signal]), [(0, 'common'), (1, 'samps')])
    aman_common.wrap('timestamps', aman.timestamps, [(0, 'samps')])
    noise_fit_stats = fit_noise_model(aman_common, signal=None, merge_fit=True, 
                                      label_axis='common', **fit_args)
    if merge:
        aman.wrap(merge_name, aman_common)
    return noise_fit_stats
