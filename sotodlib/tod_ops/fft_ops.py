"""FFTs and related operations
"""
from scipy.signal import welch
from scipy.fft import set_global_backend
import numpy as np
import pyfftw
import logging

import so3g

from . import detrend_data

logger = logging.getLogger(__name__)


def _get_num_threads():
    # Guess how many threads we should be using in FFT ops...
    return so3g.useful_info().get("omp_num_threads", 4)


# Setup scipy operations to use fftw
try:
    pyfftw.config.NUM_THREADS = _get_num_threads()
    set_global_backend(pyfftw.interfaces.scipy_fft)
    pyfftw.interfaces.cache.enable()
except ValueError:
    logger.debug("Failed to configure pyfftw as scipy fft backend")


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
            be 'linear', 'mean', or None.

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
        signal = detrend_data(
            aman, detrend, axis_name=axis_name, signal_name=signal_name
        )

    if other_idx is not None and other_idx != 0:
        signal = signal.transpose()

    if window is not None:
        signal = signal * window(axis.count)[None, :]

    if resize == "zero_pad":
        k = int(np.ceil(np.log(axis.count) / np.log(2)))
        n = 2 ** k
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
    best = p ** n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p ** n
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
    best = p ** n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p ** n
        best_friend = find_superior_integer(target / base, primes[1:])
        if (best_friend * base) <= best:
            best = best_friend * base
    return int(best)


def calc_psd(aman, signal=None, timestamps=None, **kwargs):
    """Calculates the power spectrum density of an input signal using signal.welch()
    Data defaults to aman.signal and times defaults to aman.timestamps
        Arguments:
            aman: AxisManager with (dets, samps) OR (channels, samps)axes.

            signal: data signal to pass to scipy.signal.welch()

            timestamps: timestamps associated with the data signal

            **kwargs: keyword args to be passed to signal.welch()

        Returns two numpy arrays:
            freqs:
                array of frequencies corresponding to PSD calculated from welch
            Pxx:
                array of PSD values
    """
    if signal is None:
        signal = aman.signal
    if timestamps is None:
        timestamps = aman.timestamps

    freqs, Pxx = welch(signal, 1 / np.median(np.diff(timestamps)), **kwargs)
    return freqs, Pxx


def calc_wn(aman, pxx=None, freqs=None, low_f=5, high_f=10):
    """
    Function that calculates the white noise level as a median PSD value between two frequencies.
    Defaults to calculation of white noise between 5 and 10Hz.
    Defaults frequency information to a wrapped "freqs" field in aman.
    Args:
    - aman: Axis manager
        Uses aman.freq as frequency information associated with the PSD, pxx.
    - pxx: Float array
        Psd information to calculate white noise. Defaults to aman.pxx
    - freqs: 1d Float array
        frequency information related to the psd. Defaults to aman.freqs
    - low_f: Floating point number
        low frequency cutoff to calculate median psd value. Defaults to 5Hz
    - high_f: Floating point number
        high frequency cutoff to calculate median psd value. Defaults to 10Hz
    Returns:
    wn: Float array
        array of white noise levels for each psd passed into argument.
    """
    if freqs is None:
        freqs = aman.freqs

    if pxx is None:
        pxx = aman.pxx

    fmsk = np.all([freqs >= low_f, freqs <= high_f], axis=0)
    if pxx.ndim == 1:
        wn2 = np.median(pxx[fmsk])
    else:
        wn2 = np.median(pxx[:, fmsk], axis=1)

    wn = np.sqrt(wn2)
    return wn
