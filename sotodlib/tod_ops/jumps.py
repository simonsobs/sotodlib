from typing import Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import scipy.ndimage as simg
import scipy.stats as ss
from numpy.typing import NDArray
from pixell.utils import block_expand, block_reduce, moveaxis
from scipy.sparse import csr_array
from skimage.restoration import denoise_tv_chambolle
from so3g import (
    matched_jumps,
    matched_jumps64,
    find_quantized_jumps,
    find_quantized_jumps64,
)
from so3g.proj import Ranges, RangesMatrix
from sotodlib.core import AxisManager

from ..flag_utils import _merge


def std_est(
    x: NDArray[np.floating],
    ds: int = 1,
    win_size: int = 20,
    axis: int = -1,
    method: str = "median_unbiased",
) -> NDArray[np.floating]:
    """
    Estimate white noise standard deviation of data.
    More robust to jumps and 1/f then np.std()

    Arguments:

        x: Data to compute standard deviation of.

        ds: Downsample factor to use, does a naive slicing in blocks of ``win_size``.

        win_size: Window size to downsample by.

        axis: The axis to compute along.

        method: The method to pass to np.quantile.

    Returns:

        stdev: The estimated white noise standard deviation of x.
    """
    if ds > 2 * x.shape[axis]:
        ds = 1
    if ds > 1:
        x = np.moveaxis(x, axis, -1)
        x = x[..., : -1 * (x.shape[-1] % win_size)]
        shape = list(x.shape) + [win_size]
        shape[-2] = -1
        x = x.reshape(tuple(shape))
        x = np.moveaxis(x, -2, 0)
        diff = np.diff(x[::ds], axis=-1)
        diff = moveaxis(diff, 0, -2)
        diff = diff.reshape(shape[:-1])
        diff = np.moveaxis(diff, -1, axis)
    else:
        diff = np.diff(x, axis=axis)
    # Find ~1 sigma limits of differenced data
    lims = np.quantile(
        diff,
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
    exact: bool = False,
) -> NDArray[np.bool_]:
    """
    Matched filter jump finder.

    Arguments:

        x: Data to jumpfind on, expects 1D or 2D.

        min_size: The smallest jump size counted as a jump.

        win_size: Size of window used when peak finding.

        exact: Flag only the jump locations if True.
               If False flag the whole window (cheaper).

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
    dtype = x.dtype.name
    if len(x.shape) > 2:
        raise ValueError("x may not have more than 2 dimensions")
    if dtype == "float32":
        matched_filt = matched_jumps
    elif dtype == "float64":
        matched_filt = matched_jumps64
    else:
        raise TypeError("x must be float32 or float64")

    jumps = np.zeros(x.shape, dtype=bool, order="C")
    if x.shape[-1] < win_size:
        return jumps.reshape(orig_shape)

    msk = np.ptp(x, axis=-1) > min_size
    if not np.any(msk):
        return jumps.reshape(orig_shape)

    # Flag with a matched filter
    win_size += win_size % 2  # Odd win size adds a wierd phasing issue
    _x = np.ascontiguousarray(x[msk])
    _jumps = np.ascontiguousarray(np.empty_like(_x), "int32")
    if isinstance(min_size, np.ndarray):
        _min_size = min_size[msk].astype(_x.dtype)
    elif min_size is None:
        raise TypeError("min_size is None")
    else:
        _min_size = (min_size * np.ones(len(_x))).astype(_x.dtype)
    matched_filt(_x, _jumps, _min_size, win_size)
    jumps[msk] = _jumps > 0

    if exact:
        structure = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        labels, _ = simg.label(jumps, structure)
        peak_idx = np.array(
            simg.maximum_position(
                np.diff(_x, axis=-1, prepend=np.zeros(len(_x))), labels
            )
        )
        jump_rows = [peak_idx[:, 0]]
        jump_cols = peak_idx[:, 1]
        jumps[:] = False
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

    def _fix(jump_ranges, heights, x_fixed):
        for j, jump_range in enumerate(jump_ranges):
            for start, end in jump_range.ranges():
                _heights = heights[j, start:end]
                height = _heights[np.argmax(np.abs(_heights))]
                x_fixed[j, int((start + end) / 2) :] -= height

    x_fixed = x
    if not inplace:
        x_fixed = x.copy()
    orig_shape = x.shape
    x_fixed = np.atleast_2d(x_fixed)
    if isinstance(jumps, np.ndarray):
        jumps = RangesMatrix.from_mask(np.atleast_2d(jumps))
    elif isinstance(jumps, Ranges):
        jumps = RangesMatrix.from_mask(np.atleast_2d(jumps.mask()))
    if not isinstance(jumps, RangesMatrix):
        raise TypeError("jumps not RangesMatrix or convertable to RangesMatrix")

    if heights is None:
        heights = estimate_heights(x_fixed, jumps.mask(), **kwargs)
    elif isinstance(heights, csr_array):
        heights = heights.toarray()
    heights = cast(NDArray[np.floating], heights)

    _fix(jumps.ranges, heights, x_fixed)

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
    win_size = int(win_size + win_size % 2)
    if jumps is not None and make_step:
        signal = _make_step(signal, jumps)
    diff_buffed = np.empty_like(signal)
    diff_buffed[..., :win_size] = 0
    diff_buffed[..., win_size:] = np.subtract(
        signal[..., win_size:],
        signal[..., : (-1 * win_size)],
        out=diff_buffed[..., win_size:],
    )

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
    max_tol=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    ds=...,
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
    max_tol=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    ds=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array]:
    ...


def twopi_jumps(
    aman: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    win_size: int = 20,
    nsigma: float = 5.0,
    atol: Optional[Union[float, NDArray[np.floating]]] = None,
    max_tol: float = 0.0314,
    fix: bool = True,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    name: str = "jumps_2pi",
    ds: int = 10,
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

        max_tol: Upper bound of the nsigma based thresh.
                 atol ignores this.

        fix: If True the jumps will be fixed by adding N*2*pi at the jump locations.

        inplace: If True jumps will be fixed inplace.

        merge: If True will wrap ranges matrix into ``aman.flags.<name>``

        overwrite: If True will overwrite existing content of ``aman.flags.<name>``

        name: String used to populate field in flagmanager if merge is True.

        ds: Downsample factor used when computing noise level, the actual factor used is `ds*win_size`.

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
        atol = nsigma * std_est(
            signal.astype(float), ds=win_size * ds, win_size=win_size
        )
        np.clip(atol, 1e-8, max_tol)

    _signal = _filter(signal, **filter_pars)
    _signal = np.atleast_2d(_signal)
    if isinstance(atol, int) or isinstance(atol, float):
        atol = np.ones(len(_signal), float) * float(atol)
    elif np.isscalar(atol):
        raise TypeError(f"Invalid atol type: {type(atol)}")
    if len(atol) != len(signal):
        raise ValueError(f"Non-scalar atol provided with length {len(atol)}")

    _signal = np.ascontiguousarray(_signal)
    heights = np.empty_like(_signal)
    atol = np.ascontiguousarray(atol, dtype=_signal.dtype)
    if _signal.dtype.name == "float32":
        find_quantized_jumps(_signal, heights, atol, win_size, 2 * np.pi)
    elif _signal.dtype.name == "float64":
        find_quantized_jumps64(_signal, heights, atol, win_size, 2 * np.pi)
    else:
        raise TypeError("signal must be float32 or float64")

    # Shift things by half the window
    heights = np.roll(heights, -1 * int(win_size / 2), -1)
    heights[:, (-1 * int(win_size / 2)) :] = 0

    jumps = heights != 0
    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))

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
    min_sigma=...,
    min_size=...,
    win_size=...,
    exact=...,
    fix: Literal[False] = False,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    ds=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array]:
    ...


@overload
def find_jumps(
    aman,
    signal=...,
    min_sigma=...,
    min_size=...,
    win_size=...,
    exact=...,
    fix: Literal[True] = True,
    inplace=...,
    merge=...,
    overwrite=...,
    name=...,
    ds=...,
    **filter_pars,
) -> Tuple[RangesMatrix, csr_array, NDArray[np.floating]]:
    ...


def find_jumps(
    aman: AxisManager,
    signal: Optional[NDArray[np.floating]] = None,
    min_sigma: Optional[float] = None,
    min_size: Optional[Union[float, NDArray[np.floating]]] = None,
    win_size: int = 20,
    exact: bool = False,
    fix: bool = False,
    inplace: bool = False,
    merge: bool = True,
    overwrite: bool = False,
    name: str = "jumps",
    ds: int = 10,
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

        win_size: Size of window used when peak finding.
                  Also used for height estimation, should be of order jump width.

        exact: If True search for the exact jump location.
               If False flag allow some undertainty within the window (cheaper).

        fix: Set to True to fix.

        inplace: Whether of not signal should be fixed inplace.

        merge: If True will wrap ranges matrix into ``aman.flags.<name>``

        overwrite: If True will overwrite existing content of ``aman.flags.<name>``

        name: String used to populate field in flagmanager if merge is True.

        ds: Downsample factor used when computing noise level, the actual factor used is `ds*win_size`.

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
    _signal = _filter(signal, **filter_pars)
    _signal = np.atleast_2d(_signal)

    if len(orig_shape) > 2:
        raise ValueError("Jumpfinder only works on 1D or 2D data")

    if min_size is None and min_sigma is not None:
        min_size = min_sigma * std_est(
            signal, ds=win_size * ds, win_size=win_size, axis=-1
        )
    if min_size is None:
        raise ValueError("min_size is somehow still None")
    if isinstance(min_size, np.ndarray) and np.ndim(min_size) > 1:  # type: ignore
        raise ValueError("min_size must be 1d or a scalar")
    elif isinstance(min_size, (float, int)):
        min_size = float(min_size) * np.ones(len(_signal))

    # Mean subtract, if we don't do this then when we cumsum we get floats
    # that are too big and lack the precicion to find jumps well
    _signal -= np.mean(_signal, axis=-1)[..., None]

    jumps = _jumpfinder(_signal, min_size, win_size, exact).reshape(orig_shape)

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
