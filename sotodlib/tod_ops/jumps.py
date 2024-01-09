from functools import partial
from typing import Dict, Literal, Optional, Protocol, Tuple, Union, overload

import numpy as np
import scipy.ndimage as simg
import scipy.signal as sig
import scipy.stats as ss
from numpy.typing import NDArray
from scipy.sparse import csr_array, lil_array
from skimage.restoration import denoise_tv_chambolle
from so3g.proj import RangesMatrix
from sotodlib.core import AxisManager


def std_est(x: NDArray[np.floating], axis: int = -1) -> NDArray[np.floating]:
    """
    Estimate white noise standard deviation of data.
    More robust to jumps and 1/f then np.std()

    Arguments:

        x: Data to compute standard deviation of.

    Returns:

        stdev: The estimated white noise standard deviation of x.
    """
    # Find ~1 sigma limits of differenced data
    lims = np.quantile(np.diff(x, axis=axis), np.array([0.159, 0.841]), axis=axis)
    # Convert to standard deviation
    return (lims[1] - lims[0]) / 8**0.5


def _jumpfinder(
    x: NDArray[np.floating],
    min_chunk: int = 20,
    min_size: Optional[Union[float, NDArray[np.floating]]] = None,
    win_size: int = 20,
    nsigma: float = 25,
    max_depth: int = 1,
    depth: int = 0,
) -> NDArray[np.bool_]:
    """
    Recursive edge detection based jumpfinder.

    Note that the jumpfinder is very sensitive to changes in parameters
    and the parameters are not independant of each other,
    so it may take some playing around to get it to work properly.

    Arguments:

        x: Data to jumpfind on, expects 1D or 2D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Size of window used by SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        depth: The current recursion depth.

    Returns:

        jumps: Mask with the same shape as x that is True at jumps.
               Jumps within min_chunk of each other may not be distinguished.
    """
    if min_size is None:
        min_size = ss.iqr(x, -1)

    # Since this is intended for det data lets assume we either 1d or 2d data
    # and in the case of 2d data we find jumps along rows
    orig_shape = x.shape
    x = np.atleast_2d(x)

    jumps = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] < min_chunk:
        return jumps.reshape(orig_shape)

    size_msk = (np.max(x, axis=-1) - np.min(x, axis=-1)) < min_size
    if np.all(size_msk):
        return jumps.reshape(orig_shape)

    # If std is basically 0 no need to check for jumps
    std = np.std(x, axis=-1)
    std_msk = np.isclose(std, 0.0) + np.isclose(std_est(x, axis=-1), std)

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
        > (np.median(sg_x_step, axis=-1) + nsigma * std_est(sg_x_step, axis=-1))[
            ..., None
        ]
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

    # If no jumps found return
    if not np.any(jumps):
        return jumps.reshape(orig_shape)

    # If at max_depth then return
    if depth == max_depth:
        return jumps.reshape(orig_shape)

    # Recursively check for jumps between jumps
    # We lose the benefits of vectorization here, so high depths are slow.
    for row in range(len(x)):
        row_msk = jump_rows == row
        if not np.any(row_msk):
            continue
        _jumps = jump_cols[row_msk]
        _jumps = np.insert(jumps, 0, 0)
        _jumps = np.append(_jumps, len(x))
        for i in range(len(_jumps) - 1):
            sub_jumps = _jumpfinder(
                x[row, (_jumps[i]) : (_jumps[i + 1])],
                min_chunk,
                min_size,
                win_size,
                nsigma,
                max_depth,
                depth + 1,
            )
            jumps[row, (_jumps[i]) : (_jumps[i + 1])] += sub_jumps
    return jumps.reshape(orig_shape)


def jumpfinder_sliding_window(
    x: NDArray[np.floating], window_size: int = 10000, overlap: int = 1000, **kwargs
) -> NDArray[np.bool_]:
    """
    Run jumpfinder through a sliding window.
    This can help get jumps towards the edges of the data that may be missed.
    Nominally those jumps can be found if the jumpfinder reaches sufficient depth,
    but sometimes it takes tweaking of the parameters to catch them.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        window_size: Size of window to use.

        overlap: Overlap between adjacent windows.

        **kwargs: kwargs to pass to _jumpfinder

    Returns:

        jumps: Mask with the same shape as x that is True at jumps.
               Jumps within min_chunk of each other may not be distinguished.
    """
    jumps = np.zeros(x.shape, dtype=bool)
    tot = jumps.shape[-1]
    for i in range(tot // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, tot))
        _jumps = _jumpfinder(
            x=x[..., start:end],
            **kwargs,
        )
        jumps[..., start:end] += _jumps
    return jumps


class JumpFix(Protocol):
    def __call__(
        self,
        x: NDArray[np.floating],
        jumps: NDArray[np.bool_],
        inplace: bool = False,
        **kwargs,
    ) -> NDArray[np.floating]:
        ...


def jumpfix_subtract_heights(
    x: NDArray[np.floating],
    jumps: NDArray[np.bool_],
    inplace: bool = False,
    heights: Optional[csr_array] = None,
    **kwargs,
) -> NDArray[np.floating]:
    """
    Naive jump fixing routine where we subtract known heights between jumps.
    Note that you should exepect a glitch at the jump locations.
    Works best if you buffer the jumps mask by a bit.

    Arguments:

        x: Data to jumpfix on, expects 1D or 2D.

        jumps: Boolean mask of that is True at jump locations.
               Should be the same shape at x.

        inplace: Whether of not x should be fixed inplace.

        heights: Sparse array of jump heights.
                 If None will be computed.

        **kwargs: Additional arguments to pass to estimate_heights if heights is None.

    Returns:

        x_fixed: x with jumps removed.
                 If inplace is True this is just a reference to x.
    """
    x_fixed = x
    if not inplace:
        x_fixed = x.copy()
    orig_shape = x.shape
    x_fixed = np.atleast_2d(x_fixed)
    jumps = np.atleast_2d(jumps)
    jumps[:, [0, -1]] = False

    if heights is None:
        heights = estimate_heights(x_fixed, jumps, **kwargs)
    else:
        _heights = lil_array(jumps.shape, dtype=float)
        _heights[jumps] = heights[jumps]
        heights = _heights.tocsr()
    heights_flat = np.array(heights[jumps])

    rows, cols = np.nonzero(np.diff(~jumps, axis=-1))
    rows = rows[::2]
    cols = cols.reshape((-1, 2))

    diff = np.diff(cols, axis=-1).ravel()
    has_jumps = diff < x.shape[-1]
    rows = rows[has_jumps]
    cols = cols[has_jumps]

    n = 0
    for r, (c1, c2) in zip(rows, cols):
        m = n + (c2 - c1)
        _heights = heights_flat[n:m]
        height = _heights[np.argmax(np.abs(_heights))]
        x_fixed[r, int(np.mean([c1, c2])) :] -= height
        n = m

    return x_fixed.reshape(orig_shape)


def _diff_buffed(
    signal: NDArray[np.floating], win_size: int, medfilt: bool
) -> NDArray[np.floating]:
    win_size = int(win_size)
    if medfilt:
        _size = win_size - 1 + (win_size % 2)
        size = np.ones(len(signal.shape), dtype=int)
        size[-1] = _size
        signal = simg.median_filter(signal, size)
    pad = np.zeros((len(signal.shape), 2), dtype=int)
    pad[-1, 0] = win_size
    diff_buffed = signal - np.pad(signal, pad, mode="edge")[..., : (-1 * win_size)]

    return diff_buffed


def estimate_heights(
    signal: NDArray[np.floating],
    jumps: NDArray[np.bool_],
    win_size: int = 20,
    twopi: bool = False,
    medfilt: bool = False,
    diff_buffed: Optional[NDArray[np.floating]] = None,
) -> csr_array:
    """
    Simple jump estimation routine.

    Arguments:

        signal: The signal with jumps.

        jumps: Boolean mask of jump locations in signal.

        win_size: Number of samples to buffer when estimating heights.

        twopi: If True, heights will be rounded to the nearest 2*pi

        medfilt: If True, a median filter of size ~win_size will be applied.

        diff_buffed: Difference between signal and a signal shifted by win_size.
                     If None will be computed.

    Returns:

        heights: Sparse array of jump heights.
    """
    if diff_buffed is None:
        diff_buffed = _diff_buffed(signal, win_size, medfilt)

    jumps = np.atleast_2d(jumps)
    diff_buffed = np.atleast_2d(diff_buffed)
    if len(jumps.shape) > 2:
        raise ValueError("Only 1d and 2d arrays are supported")

    heights = lil_array(jumps.shape, dtype=diff_buffed.dtype)
    heights[jumps] = diff_buffed[jumps]
    heights = heights.tocsr()

    if twopi:
        heights = csr_array(np.round(heights / (2 * np.pi)) * 2 * np.pi)
    return heights


def _merge(tod: AxisManager, jump_ranges: RangesMatrix, flagname: str, overwrite: bool):
    if not isinstance(tod, AxisManager):
        print("TOD is not an AxisManager, not merging")
        return
    elif "dets" not in tod or "samps" not in tod:
        print("dets or samps axis not in TOD, not merging")
        return
    elif jump_ranges.shape != (tod.dets.count, tod.samps.count):
        print("Shape of jumps does not match that of TOD, not merging")
        return
    if flagname in tod.flags._fields:
        if overwrite:
            tod.flags.move(flagname, None)
        else:
            print("Flag already exists and overwrite is False")
    if "flags" not in tod._fields:
        flags = AxisManager(tod.dets, tod.samps)
        tod.wrap("flags", flags)
    tod.flags.wrap(flagname, jump_ranges, [(0, "dets"), (1, "samps")])


def _filter(
    x: NDArray[np.floating],
    gaussian_width: float,
    tv_weight: float,
    force_copy: bool = False,
) -> NDArray[np.floating]:
    if force_copy or gaussian_width > 0 or tv_weight > 0:
        x = x.copy()
    if gaussian_width > 0:
        x = simg.gaussian_filter1d(x, gaussian_width, axis=-1, output=x).astype(float)
    if tv_weight > 0:
        channel_axis = 1
        if len(x.shape) == 1:
            channel_axis = None
        x = denoise_tv_chambolle(x, tv_weight, channel_axis=channel_axis)
    return np.array(x, dtype=np.float32)


@overload
def twopi_jumps(
    tod,
    signal=...,
    win_size=...,
    atol=...,
    gaussian_width=...,
    tv_weight=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    flagname=...,
) -> Tuple[RangesMatrix, NDArray[np.floating]]:
    ...


@overload
def twopi_jumps(
    tod,
    signal=...,
    win_size=...,
    atol=...,
    gaussian_width=...,
    tv_weight=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    flagname=...,
) -> RangesMatrix:
    ...


def twopi_jumps(
    tod: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    win_size: int = 20,
    atol: Optional[Union[float, NDArray[np.floating]]] = None,
    gaussian_width: float = 0,
    tv_weight: float = 0,
    fix: bool = True,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    flagname: str = "jumps_2pi",
) -> Union[RangesMatrix, Tuple[RangesMatrix, NDArray[np.floating]]]:
    """
    Find and optionally fix jumps that are height ~N*2pi.
    TOD is expected to have detectors with high trends already cut.
    Data is assumed to have units of phase here.

    Arguments:

        tod: The axis manager containing signal to find jumps on.

        signal: Signal to jumpfind on. If None than tod.signal is used.

        win_size: Size of window to use when looking for jumps.
                  This should be set to something of order the width of the jumps.

        atol: How close the jump height needs to be to N*2pi to count.
              If set to None, then 3 times the WN level of the TOD is used.

        gaussian_width: Width of gaussian filter.
                        If <= 0, filter is not applied.

        tv_weight: Weight used by total variance filter.
                   If <= 0, filter is not applied.

        fix: If True the jumps will be fixed by adding N*2*pi at the jump locations.

        inplace: If True jumps will be fixed inplace.

        merge: If True will wrap ranges matrix into ``tod.flags.<flagname>``

        overwrite: If True will overwrite existing content of ``tod.flags.<flagname>``

        flagname: String used to populate field in flagmanager if merge is True.

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               Buffered to win_size.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = tod.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")
    if atol is None:
        atol = 3 * std_est(signal.astype(float))
        np.clip(atol, 1e-8, 1e-2)

    _signal = _filter(signal, gaussian_width, tv_weight)
    diff_buffed = _diff_buffed(_signal, win_size, False)

    if isinstance(atol, int):
        atol = float(atol)
    if isinstance(atol, float):
        jumps = (np.isclose(0, np.abs(diff_buffed) % (2 * np.pi), atol=atol)) & (
            (np.abs(diff_buffed) // (2 * np.pi)) >= 1
        )
        jumps[:win_size] = False
    elif isinstance(atol, np.ndarray):
        jumps = np.atleast_2d(np.zeros_like(signal, dtype=bool))
        diff_buffed = np.atleast_2d(diff_buffed)
        if len(atol) != len(jumps):
            raise ValueError(f"Non-scalar atol provided with length {len(atol)}")
        for i, _atol in enumerate(atol):
            jumps[i] = (
                np.isclose(0, np.abs(diff_buffed[i]) % (2 * np.pi), atol=_atol)
            ) & ((np.abs(diff_buffed[i]) // (2 * np.pi)) >= 1)
        jumps[:, :win_size] = False
        jumps.reshape(signal.shape)
    else:
        raise TypeError(f"Invalid atol type: {type(atol)}")

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))

    if merge:
        _merge(tod, jump_ranges, flagname, overwrite)

    if fix:
        jumps = jump_ranges.mask()
        heights = estimate_heights(signal, jumps, twopi=True, diff_buffed=diff_buffed)
        fixed = jumpfix_subtract_heights(signal, jumps, inplace, heights)

        return jump_ranges, fixed
    return jump_ranges


@overload
def find_jumps(
    tod,
    signal=...,
    max_iters=...,
    min_chunk=...,
    min_sigma=...,
    min_size=...,
    win_size=...,
    nsigma=...,
    max_depth=...,
    gaussian_width=...,
    tv_weight=...,
    window_args=...,
    fix: None = None,
    fix_kwargs=...,
    inplace=...,
    merge=...,
    overwrite=...,
    flagname=...,
) -> RangesMatrix:
    ...


@overload
def find_jumps(
    tod,
    signal=...,
    max_iters=...,
    min_chunk=...,
    min_sigma=...,
    min_size=...,
    win_size=...,
    nsigma=...,
    max_depth=...,
    gaussian_width=...,
    tv_weight=...,
    window_args=...,
    fix: JumpFix = ...,
    fix_kwargs=...,
    inplace=...,
    merge=...,
    overwrite=...,
    flagname=...,
) -> Tuple[RangesMatrix, NDArray[np.floating]]:
    ...


def find_jumps(
    tod: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    max_iters: int = 1,
    min_chunk: int = 20,
    min_sigma: Optional[float] = None,
    min_size: Optional[Union[float, NDArray[np.floating]]] = None,
    win_size: int = 20,
    nsigma: float = 25,
    max_depth: int = 0,
    gaussian_width: float = 0,
    tv_weight: float = 0,
    window_args: Optional[Dict] = None,
    fix: Optional[JumpFix] = None,
    fix_kwargs: Dict = {},
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    flagname: str = "jumps",
) -> Union[RangesMatrix, Tuple[RangesMatrix, NDArray[np.floating]]]:
    """
    Find jumps in tod.signal_name.
    Expects tod.signal_name to be 1D of 2D.

    Arguments:

        tod: axis manager.

        signal: Signal to jumpfind on. If None than tod.signal is used.

        max_iters: Maximum iterations of the jumpfind -> median sub -> jumpfind loop.
                   This is prefered over increasing depth in general.

        min_chunk: The smallest chunk of data to look for jumps in.

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

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        gaussian_width: Width of gaussian filter.
                        If <= 0, filter is not applied.

        tv_weight: Weight used by total variance filter.
                   If <= 0, filter is not applied.

        window_args: Arguments for sliding window.
                     Set to None to not use, set to a dict of arguments otherwise.

        fix: Method to use for jumpfixing.
             Set to None to not fix.

        fix_kwargs: kwargs other than inplace to pass to the jumpfixer.

        inplace: Whether of not signal should be fixed inplace.

        merge: If True will wrap ranges matrix into ``tod.flags.<flagname>``

        overwrite: If True will overwrite existing content of ``tod.flags.<flagname>``

        flagname: String used to populate field in flagmanager if merge is True.

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               There is some uncertainty on order of a few samples.
               Jumps within a few samples of each other may not be distinguished.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = tod.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")

    if len(signal.shape) > 2:
        raise ValueError("Jumpfinder only works on 1D or 2D data")

    if min_size is None and min_sigma is not None:
        min_size = min_sigma * std_est(signal, axis=-1)
    if np.ndim(min_size) > 1:  # type: ignore
        raise ValueError("min_size must be 1d or a scalar")
    elif np.ndim(min_size) == 1:  # type: ignore
        min_size = np.array(min_size)

    do_fix = fix is not None

    _signal = _filter(signal, gaussian_width, tv_weight, force_copy=True)
    # Median subtract, if we don't do this then when we cumsum we get floats
    # that are too big and lack the precicion to find jumps well
    _signal -= np.median(_signal, axis=-1)[..., None]

    if window_args is None:
        jumpfinder = _jumpfinder
    else:
        jumpfinder = partial(jumpfinder_sliding_window, **window_args)

    jumps = np.zeros(signal.shape, dtype=bool)
    msk = np.ones(len(jumps), dtype=bool)
    for _ in range(max_iters):
        if isinstance(min_size, np.ndarray):
            _min_size = min_size[msk]
        else:
            _min_size = min_size
        _jumps = jumpfinder(
            _signal[msk],
            min_chunk=min_chunk,
            min_size=_min_size,
            win_size=win_size,
            nsigma=nsigma,
            max_depth=max_depth,
        )
        if np.sum(_jumps) == 0:
            break

        jumps[msk] += _jumps
        _signal[msk] = jumpfix_subtract_heights(_signal[msk], _jumps, True)
        msk = np.any(jumps, axis=-1)

    # TODO: include heights in output

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(min_chunk / 2))

    if merge:
        _merge(tod, jump_ranges, flagname, overwrite)

    if do_fix:
        fixed = fix(signal, jumps, inplace=inplace, **fix_kwargs)
        return jump_ranges, fixed
    return jump_ranges
