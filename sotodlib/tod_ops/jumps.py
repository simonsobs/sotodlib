import concurrent.futures
import os
from typing import Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import scipy.ndimage as simg
import scipy.signal as sig
import scipy.stats as ss
from numpy.typing import NDArray
from pixell.utils import block_expand, block_reduce
from scipy.sparse import csr_array
from skimage.restoration import denoise_tv_chambolle
from so3g.proj import Ranges, RangesMatrix
from sotodlib.core import AxisManager

from ..flag_utils import _merge

NFUTURE = int(os.environ.get("NUM_FUTURES", min(32, int(os.cpu_count() or 0) + 4)))


def std_est(
    x: NDArray[np.floating],
    ds: int = 1,
    axis: int = -1,
    method: str = "median_unbiased",
) -> NDArray[np.floating]:
    """
    Estimate white noise standard deviation of data.
    More robust to jumps and 1/f then np.std()

    Arguments:

        x: Data to compute standard deviation of.

        ds: Downsample factor to use, does a naive slicing.

        axis: The axis to compute along.

        method: The method to pass to np.quantile.

    Returns:

        stdev: The estimated white noise standard deviation of x.
    """
    if ds > 2 * x.shape[axis]:
        ds = 1
    sl = [slice(None)] * len(x.shape)
    if ds > 1:
        sl[axis] = slice(None, None, ds)
    # Find ~1 sigma limits of differenced data
    lims = np.quantile(
        np.diff(x, axis=axis)[tuple(sl)],
        np.array([0.159, 0.841]),
        axis=axis,
        method=method,
    )
    # Convert to standard deviation
    return (lims[1] - lims[0]) / 8**0.5


def _jumpfinder(
    x: NDArray[np.floating],
    min_size: Optional[Union[float, NDArray[np.floating]]] = None,
    win_size: int = 20,
    nsigma: float = 25,
) -> NDArray[np.bool_]:
    """
    Matched filter jump finder.

    Arguments:

        x: Data to jumpfind on, expects 1D or 2D.

        min_size: The smallest jump size counted as a jump.

        win_size: Size of window used by SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

    Returns:

        jumps: Mask with the same shape as x that is True at jumps.
               Jumps within win_size of each other may not be distinguished.
    """
    if min_size is None:
        min_size = ss.iqr(x, -1)

    # Since this is intended for det data lets assume we either 1d or 2d data
    # and in the case of 2d data we find jumps along rows
    orig_shape = x.shape
    x = np.atleast_2d(x)

    jumps = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] < win_size:
        return jumps.reshape(orig_shape)

    size_msk = (np.max(x, axis=-1) - np.min(x, axis=-1)) < min_size
    if np.all(size_msk):
        return jumps.reshape(orig_shape)

    # If std is basically 0 no need to check for jumps
    std = np.std(x, axis=-1)
    std_msk = np.isclose(std, 0.0) + np.isclose(std_est(x, ds=win_size, axis=-1), std)

    msk = ~(size_msk + std_msk)
    if not np.any(msk):
        return jumps.reshape(orig_shape)

    # Take cumulative sum, this is equivalent to convolving with a step
    x_step = np.cumsum(x[msk], axis=-1)

    # Smooth and take the second derivative
    sg_x_step = np.abs(sig.savgol_filter(x_step, win_size, 2, deriv=2, axis=-1))

    # Peaks should be jumps
    # Doing the simple thing and looking for things much larger than the median
    peaks = (
        sg_x_step
        > (
            np.median(sg_x_step, axis=-1)
            + nsigma * std_est(sg_x_step, ds=win_size, axis=-1)
        )[..., None]
    )
    if not np.any(peaks):
        return jumps.reshape(orig_shape)

    # The peak may have multiple points above this criteria
    peak_idx = np.where(peaks)
    peak_idx_padded = peak_idx[1] + (x.shape[-1] + win_size) * peak_idx[0]
    gaps = np.diff(peak_idx_padded) >= win_size
    begins = np.insert(peak_idx_padded[1:][gaps], 0, peak_idx_padded[0])
    ends = np.append(peak_idx_padded[:-1][gaps], peak_idx_padded[-1])
    jump_idx = ((begins + ends) / 2).astype(int) + 1
    jump_rows = jump_idx // (x.shape[1] + win_size)
    jump_cols = jump_idx % (x.shape[1] + win_size)

    # Estimate jump heights and get better positions
    # TODO: Pad things to avoid np.diff annoyance
    half_win = int(win_size / 2)
    win_rows = np.repeat(jump_rows, 2 * half_win)
    win_cols = np.repeat(jump_cols, 2 * half_win) + np.tile(
        np.arange(-1 * half_win, half_win, dtype=int), len(jump_cols)
    )
    win_cols = np.clip(win_cols, 0, x.shape[-1] - 3)
    d2x_step = np.abs(np.diff(x_step, n=2, axis=-1))[win_rows, win_cols].reshape(
        (len(jump_idx), 2 * half_win)
    )
    jump_sizes = np.amax(d2x_step, axis=-1)
    jump_cols = (
        win_cols.reshape(d2x_step.shape)[
            np.arange(len(jump_idx)), np.argmax(d2x_step, axis=-1)
        ]
        + 2
    )

    # Make a jump size cut
    if isinstance(min_size, np.ndarray):
        _min_size = min_size[jump_rows]
    else:
        _min_size = min_size
    size_cut = jump_sizes > _min_size
    jump_rows = jump_rows[size_cut]
    jump_cols = jump_cols[size_cut]

    jumps[np.flatnonzero(msk)[jump_rows], jump_cols] = True

    return jumps.reshape(orig_shape)


def jumpfix_subtract_heights(
    x: NDArray[np.floating],
    jumps: Union[Ranges, RangesMatrix, NDArray[np.bool_]],
    inplace: bool = False,
    heights: Optional[Union[NDArray[np.floating], csr_array]] = None,
    **kwargs,
) -> NDArray[np.floating]:
    """
    Naive jump fixing routine where we subtract known heights between jumps.
    Note that you should exepect a glitch at the jump locations.
    Works best if you buffer the jumps mask by a bit.

    Arguments:

        x: Data to jumpfix on, expects 1D or 2D.

        jumps: Boolean mask or Ranges(Matrix) of jump locations.
               Should be the same shape at x.

        inplace: Whether of not x should be fixed inplace.

        heights: Array of jump heights, can be sparse.
                 If None will be computed.

        **kwargs: Additional arguments to pass to estimate_heights if heights is None.

    Returns:

        x_fixed: x with jumps removed.
                 If inplace is True this is just a reference to x.
    """

    def _fix(i, jump_ranges, heights, x_fixed):
        for j, jump_range in enumerate(jump_ranges):
            for start, end in jump_range.ranges():
                _heights = heights[i + j, start:end]
                height = _heights[np.argmax(np.abs(_heights))]
                x_fixed[i + j, int((start + end) / 2):] -= height

    x_fixed = x
    if not inplace:
        x_fixed = x.copy()
    orig_shape = x.shape
    x_fixed = np.atleast_2d(x_fixed)
    if isinstance(jumps, np.ndarray):
        jumps = RangesMatrix.from_mask(np.atleast_2d(jumps))
    elif isinstance(jumps, Ranges):
        jumps = RangesMatrix.from_mask(np.atleast_2d(jumps.mask()))

    if heights is None:
        heights = estimate_heights(x_fixed, jumps.mask(), **kwargs)
    elif isinstance(heights, csr_array):
        heights = heights.toarray()
    heights = cast(NDArray[np.floating], heights)

    nfuture = min(len(x_fixed), NFUTURE)
    slices = [slice(i * nfuture, (i + 1) * nfuture) for i in range(nfuture)]
    slices[-1] = slice(slices[-1].start, len(x_fixed))
    with concurrent.futures.ThreadPoolExecutor() as e:
        _ = [
            e.submit(_fix, i, jumps.ranges[s], heights[s], x_fixed[s])
            for i, s in enumerate(slices)
        ]

    return x_fixed.reshape(orig_shape)


def _make_step(signal: NDArray[np.floating], jumps: NDArray[np.bool_]):
    jumps = np.atleast_2d(jumps)
    jumps[:, [0, -1]] = False
    ranges = RangesMatrix.from_mask(jumps)
    signal_step = np.atleast_2d(signal.copy())
    samps = signal_step.shape[-1]
    for i, det in enumerate(ranges.ranges):
        for r in det.ranges():
            mid = int((r[0] + r[1]) / 2)
            signal_step[i, r[0] : mid] = signal_step[i, max(r[0] - 1, 0)]
            signal_step[i, mid : r[1]] = signal_step[i, min(r[1] + 1, samps - 1)]
    return signal_step.reshape(signal.shape)


def _diff_buffed(
    signal: NDArray[np.floating],
    jumps: Optional[NDArray[np.bool_]],
    win_size: int,
    make_step: bool,
) -> NDArray[np.floating]:
    win_size = int(win_size)
    pad = np.zeros((len(signal.shape), 2), dtype=int)
    half_win = int(win_size / 2)
    pad[-1, :] = half_win
    if jumps is not None and make_step:
        signal = _make_step(signal, jumps)
    padded = np.pad(signal, pad, mode="edge")
    diff_buffed = padded[..., win_size:] - padded[..., : (-1 * win_size)]

    return diff_buffed


def estimate_heights(
    signal: NDArray[np.floating],
    jumps: NDArray[np.bool_],
    win_size: int = 20,
    twopi: bool = False,
    make_step: bool = False,
    diff_buffed: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Simple jump estimation routine.

    Arguments:

        signal: The signal with jumps.

        jumps: Boolean mask of jump locations in signal.

        win_size: Number of samples to buffer when estimating heights.

        twopi: If True, heights will be rounded to the nearest 2*pi

        make_step: If True jump ranges will be turned into clean step functions.

        diff_buffed: Difference between signal and a signal shifted by win_size.
                     If None will be computed.

    Returns:

        heights: Array of jump heights.
    """
    if diff_buffed is None:
        diff_buffed = _diff_buffed(signal, jumps, win_size, make_step)

    jumps = np.atleast_2d(jumps)
    diff_buffed = np.atleast_2d(diff_buffed)
    if len(jumps.shape) > 2:
        raise ValueError("Only 1d and 2d arrays are supported")
    if twopi:
        diff_buffed = np.round(diff_buffed / (2 * np.pi)) * 2 * np.pi

    heights = np.zeros_like(jumps, dtype=float)
    heights[jumps] = diff_buffed[jumps]

    return heights


def _filter(
    x: NDArray[np.floating],
    medfilt: int = 0,
    gaussian_width: float = 0,
    tv_weight: float = 0,
) -> NDArray[np.floating]:
    if gaussian_width > 0 or tv_weight > 0 or medfilt > 0:
        x = x.copy()
    if medfilt > 0:
        _size = medfilt - 1 + (medfilt % 2)
        size = np.ones(len(x.shape), dtype=int)
        size[-1] = _size
        x = simg.median_filter(x, size)
    if gaussian_width > 0:
        x = simg.gaussian_filter1d(x, gaussian_width, axis=-1, output=x).astype(float)
    if tv_weight > 0:
        channel_axis = 0
        if len(x.shape) == 1:
            channel_axis = None
        x = denoise_tv_chambolle(x, tv_weight, channel_axis=channel_axis)
    return x


@overload
def twopi_jumps(
    aman,
    signal=...,
    win_size=...,
    nsigma=...,
    atol=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array, NDArray[np.floating]]:
    ...


@overload
def twopi_jumps(
    aman,
    signal=...,
    win_size=...,
    nsigma=...,
    atol=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array]:
    ...


def twopi_jumps(
    aman: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    win_size: int = 20,
    nsigma: float = 5.0,
    atol: Optional[Union[float, NDArray[np.floating]]] = None,
    fix: bool = True,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    name: str = "jumps_2pi",
    **filter_pars,
) -> Union[
    Tuple[RangesMatrix, csr_array], Tuple[RangesMatrix, csr_array, NDArray[np.floating]]
]:
    """
    Find and optionally fix jumps that are height ~N*2pi.
    TOD is expected to have detectors with high trends already cut.
    Data is assumed to have units of phase here.

    Arguments:

        aman: The axis manager containing signal to find jumps on.

        signal: Signal to jumpfind on. If None than aman.signal is used.

        win_size: Size of window to use when looking for jumps.
                  This should be set to something of order the width of the jumps.

        nsigma: How many multiples of the white noise level to set to use to compute atol.
                Only used if atol is None.

        atol: How close the jump height needs to be to N*2pi to count.
              If set to None, then nsigma times the WN level of the TOD is used.
              Note that in general this is faster than nsigma.

        fix: If True the jumps will be fixed by adding N*2*pi at the jump locations.

        inplace: If True jumps will be fixed inplace.

        merge: If True will wrap ranges matrix into ``aman.flags.<name>``

        overwrite: If True will overwrite existing content of ``aman.flags.<name>``

        name: String used to populate field in flagmanager if merge is True.

        **filter_pars: Parameters to pass to _filter

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               Buffered to win_size.

        heights: csr_array of jump heights.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = aman.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")
    if atol is None:
        atol = nsigma * std_est(signal.astype(float), ds=win_size)
        np.clip(atol, 1e-8, 1e-2)

    _signal = _filter(signal, **filter_pars)
    diff_buffed = _diff_buffed(_signal, None, win_size, False)

    if isinstance(atol, int):
        atol = float(atol)
    if isinstance(atol, float):
        ratio = np.abs(diff_buffed) / (2 * np.pi)
        jumps = (np.abs(ratio - np.round(ratio, 0)) <= atol) & (ratio >= 0.5)
        jumps[..., :win_size] = False
    elif isinstance(atol, np.ndarray):
        jumps = np.atleast_2d(np.zeros_like(signal, dtype=bool))
        diff_buffed = np.atleast_2d(diff_buffed)
        if len(atol) != len(jumps):
            raise ValueError(f"Non-scalar atol provided with length {len(atol)}")
        ratio = np.abs(diff_buffed / (2 * np.pi))
        jumps = (np.abs(ratio - np.round(ratio, 0)) <= atol[..., None]) & (ratio >= 0.5)
        jumps.reshape(signal.shape)
    else:
        raise TypeError(f"Invalid atol type: {type(atol)}")

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))
    jumps = jump_ranges.mask()
    heights = estimate_heights(
        signal, jumps, win_size=win_size, twopi=True, diff_buffed=diff_buffed
    )

    if merge:
        _merge(aman, jump_ranges, name, overwrite)

    if fix:
        fixed = jumpfix_subtract_heights(signal, jump_ranges, inplace, heights)
        return jump_ranges, csr_array(heights), fixed
    return jump_ranges, csr_array(heights)


@overload
def slow_jumps(
    aman,
    signal=...,
    win_size=...,
    thresh=...,
    abs_thresh=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array, NDArray[np.floating]]:
    ...


@overload
def slow_jumps(
    aman,
    signal=...,
    win_size=...,
    thresh=...,
    abs_thresh=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array]:
    ...


def slow_jumps(
    aman: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    win_size: int = 800,
    thresh: float = 20,
    abs_thresh: bool = True,
    fix: bool = True,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    name: str = "jumps_slow",
    **filter_pars,
) -> Union[
    Tuple[RangesMatrix, csr_array], Tuple[RangesMatrix, csr_array, NDArray[np.floating]]
]:
    """
    Find and optionally fix slow jumps.
    This is useful for catching things that aren't really jumps but
    change the DC level in a jumplike way (ie: a short unlock period).

    Arguments:

        aman: The axis manager containing signal to find jumps on.

        signal: Signal to jumpfind on. If None than aman.signal is used.

        win_size: Size of window to use when looking for jumps.
                  This should be set to something of order the width of the jumps.

        thresh: ptp value at which to flag things as jumps.
                Default value is in phase units.
                You can also pass a percentile to use here if abs_thresh is False.

        abs_thresh: If True thresh is an absolute threshold.
                    If False it is a percential in range [0, 1).

        fix: If True the jumps will be fixed by adding N*2*pi at the jump locations.

        inplace: If True jumps will be fixed inplace.

        merge: If True will wrap ranges matrix into ``aman.flags.<name>``

        overwrite: If True will overwrite existing content of ``aman.flags.<name>``

        name: String used to populate field in flagmanager if merge is True.

        **filter_pars: Parameters to pass to _filter

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               Buffered to win_size.

        heights: csr_array of jump heights.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = aman.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")

    _signal = _filter(signal, **filter_pars)

    # Block ptp
    bptp = block_reduce(_signal, win_size, op=np.ptp, inclusive=True)

    if not abs_thresh:
        thresh = float(np.quantile(bptp.ravel(), thresh))
    bptp = block_expand(bptp, win_size, _signal.shape[-1], inclusive=True)
    jumps = bptp > thresh

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))
    heights = estimate_heights(
        _signal, jump_ranges.mask(), win_size=win_size, make_step=True
    )

    if merge:
        _merge(aman, jump_ranges, name, overwrite)

    if fix:
        fixed = jumpfix_subtract_heights(signal, jump_ranges, inplace, heights)

        return jump_ranges, csr_array(heights), fixed
    return jump_ranges, csr_array(heights)


@overload
def find_jumps(
    aman,
    signal=...,
    max_iters=...,
    min_sigma=...,
    min_size=...,
    win_size=...,
    nsigma=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array]:
    ...


@overload
def find_jumps(
    aman,
    signal=...,
    max_iters=...,
    min_sigma=...,
    min_size=...,
    win_size=...,
    nsigma=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array, NDArray[np.floating]]:
    ...


def find_jumps(
    aman: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    max_iters: int = 1,
    min_sigma: Optional[float] = None,
    min_size: Optional[Union[float, NDArray[np.floating]]] = None,
    win_size: int = 20,
    nsigma: float = 25,
    fix: bool = False,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    name: str = "jumps",
    **filter_pars,
) -> Union[
    Tuple[RangesMatrix, csr_array], Tuple[RangesMatrix, csr_array, NDArray[np.floating]]
]:
    """
    Find jumps in aman.signal_name with a matched filter for edge detection.
    Expects aman.signal_name to be 1D of 2D.

    Arguments:

        aman: axis manager.

        signal: Signal to jumpfind on. If None than aman.signal is used.

        min_sigma: Number of standard deviations to count as a jump, note that
                   the standard deviation here is computed by std_est and is
                   the white noise standard deviation, so it doesn't include
                   contributions from jumps or 1/f.
                   If min_size is provided it will be used instead of this.

        min_size: The smallest jump size counted as a jump.
                  By default this is set to None and min_sigma is used instead,
                  if set this will override min_sigma.
                  If both min_sigma and min_size are None then the IQR is used as min_size.

        win_size: Size of window used by SG filter when peak finding.
                  Also used for height estimation, should be of order jump width.

        nsigma: Number of sigma above the mean for something to be a peak.

        fix: Set to True to fix.

        inplace: Whether of not signal should be fixed inplace.

        merge: If True will wrap ranges matrix into ``aman.flags.<name>``

        overwrite: If True will overwrite existing content of ``aman.flags.<name>``

        name: String used to populate field in flagmanager if merge is True.

        **filter_pars: Parameters to pass to _filter

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               There is some uncertainty on order of a few samples.
               Jumps within a few samples of each other may not be distinguished.

        heights: csr_array of jump heights.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = aman.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")

    orig_shape = signal.shape
    if len(orig_shape) > 2:
        raise ValueError("Jumpfinder only works on 1D or 2D data")

    if min_size is None and min_sigma is not None:
        min_size = min_sigma * std_est(signal, ds=win_size, axis=-1)
    if min_size is None:
        raise ValueError("min_size is somehow still None")
    if isinstance(min_size, np.ndarray) and np.ndim(min_size) > 1:  # type: ignore
        raise ValueError("min_size must be 1d or a scalar")
    elif isinstance(min_size, (float, int)):
        min_size = float(min_size) * np.ones(len(signal))

    _signal = _filter(signal, **filter_pars)
    if max_iters > 1:
        _signal = signal.copy()
    _signal = np.atleast_2d(_signal)
    # Median subtract, if we don't do this then when we cumsum we get floats
    # that are too big and lack the precicion to find jumps well
    _signal -= np.median(_signal, axis=-1)[..., None]

    nfuture = min(len(_signal), NFUTURE)
    slices = [slice(i * nfuture, (i + 1) * nfuture) for i in range(nfuture)]
    slices[-1] = slice(slices[-1].start, len(_signal))
    with concurrent.futures.ThreadPoolExecutor() as e:
        jump_futures = [
            e.submit(_jumpfinder, _signal[s], min_size[s], win_size, nsigma)
            for s in slices
        ]
    jumps = np.vstack([j.result() for j in jump_futures]).reshape(orig_shape)

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))
    jumps = jump_ranges.mask()
    heights = estimate_heights(signal, jumps, win_size=win_size)

    if merge:
        _merge(aman, jump_ranges, name, overwrite)

    if fix:
        fixed = jumpfix_subtract_heights(
            signal, jump_ranges, inplace=inplace, heights=heights
        )
        return jump_ranges, csr_array(heights), fixed
    return jump_ranges, csr_array(heights)


def jumps_aman(
    aman: AxisManager, jump_ranges: RangesMatrix, heights: csr_array
) -> AxisManager:
    """
    Helper to wrap the jumpfinder outputs into a AxisManager for use with preprocess.


    Arguments:

        aman: AxisManager to steam axis information from.

        jump_ranges: RangesMatrix containing the jump flag.

        heights: csr_array of jump heights.

    Returns:

        jumps_aman: AxisManager containing the jump information wrapped in
                    'jump_flag' and 'jump_heights'.
    """
    jump_aman = AxisManager(aman.dets, aman.samps)
    jump_aman.wrap("jump_flag", jump_ranges, [(0, "dets"), (1, "samps")])
    jump_aman.wrap("jump_heights", heights, [(0, "dets"), (1, "samps")])

    return jump_aman
