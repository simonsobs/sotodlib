import numpy as np
import scipy.ndimage as simg
import scipy.signal as sig
import scipy.stats as ss
from skimage.restoration import denoise_tv_chambolle
from so3g.proj import RangesMatrix


def std_est(x, axis=-1):
    """
    Estimate white noise standard deviation of data.
    More robust to jumps and 1/f then np.std()

    Arguments:

        x: Data to compute standard deviation of.

    Returns:

        stdev: The estimated white noise standard deviation of x.
    """
    # Find ~1 sigma limits of differenced data
    lims = np.quantile(np.diff(x, axis=axis), [0.159, 0.841], axis=axis)
    # Convert to standard deviation
    return (lims[1] - lims[0]) / 8**0.5


def _jumpfinder(x, min_chunk, min_size, win_size, nsigma, max_depth=1, depth=0):
    """
    Recursive edge detection based jumpfinder.

    Note that the jumpfinder is very sensitive to changes in parameters
    and the parameters are not independant of each other,
    so it may take some playing around to get it to work properly.

    Arguments:

        x: Data to jumpfind on, expects 1D or 2D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        depth: The current recursion depth.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    if min_chunk is None:
        min_chunk = 20
    if min_size is None:
        min_size = ss.iqr(x, -1)
    if win_size is None:
        win_size = 20
    if nsigma is None:
        nsigma = 25

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


def get_jump_sizes(x, jumps, win_size):
    """
    Estimate jumps sizes.

    Arguments:

        x: Data with jumps, expects 1D.

        jumps: Indices of jumps in x.

        win_size: Number of samples to average over when checking jump size.

    Returns:

        sizes: Array of jump sizes, same order and jumps.
    """
    sizes = np.zeros(len(jumps))
    for i, j in enumerate(jumps):
        if i + 1 < len(jumps):
            right = min(j + win_size, len(x), int((jumps[i + 1] + j) / 2))
        else:
            right = min(j + win_size, len(x))
        right_height = np.median(x[int((j + right) / 2) : right])

        if i > 0:
            left = max(j - win_size, 0, int((jumps[i - 1] + j) / 2))
        else:
            left = max(j - win_size, 0)
        left_height = np.median(x[left : int((j + left) / 2)])

        sizes[i] = right_height - left_height
    return sizes.astype(float)


def jumpfinder_tv(
    x,
    min_chunk=None,
    min_size=None,
    win_size=None,
    nsigma=None,
    max_depth=1,
    weight=0.5,
):
    """
    Apply total variance filter and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        weight: Denoising weight. Higher weights denoise more, lower weights
                preserve the input signal better.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    if min_chunk is None:
        min_chunk = 5
    if min_size is None:
        min_size = 0.1
    if win_size is None:
        win_size = 5

    channel_axis = 1
    if len(x.shape) == 1:
        channel_axis = None
    x_filt = denoise_tv_chambolle(x, weight, channel_axis=channel_axis)
    return _jumpfinder(
        x_filt,
        min_chunk,
        min_size,
        win_size,
        nsigma,
        max_depth,
        0,
    )


def jumpfinder_gaussian(
    x,
    min_chunk=None,
    min_size=None,
    win_size=None,
    nsigma=None,
    max_depth=1,
    sigma=0.5,
):
    """
    Apply gaussian filter to data and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.
                  Note that this is in terms of the filtered data.

        win_size: Number of samples to average over when checking jump size.
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        sigma: Kernal size of the gaussian filter.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    if min_chunk is None:
        min_chunk = 20
    if min_size is None:
        min_size = 0.1
    if win_size is None:
        win_size = 10

    # Apply filter
    x_filt = simg.gaussian_filter1d(x, sigma, axis=-1)

    # Search for jumps in filtered data
    return _jumpfinder(
        x_filt,
        min_chunk,
        min_size,
        win_size,
        nsigma,
        max_depth,
        0,
    )


def jumpfinder_sliding_window(
    x,
    min_chunk=None,
    min_size=None,
    win_size=None,
    nsigma=None,
    max_depth=1,
    window_size=10000,
    overlap=1000,
    jumpfinder_func=jumpfinder_tv,
    **kwargs,
):
    """
    Run jumpfinder through a sliding window.
    This can help get jumps towards the edges of the data that may be missed.
    Nominally those jumps can be found if the jumpfinder reaches sufficient depth,
    but sometimes it takes tweaking of the parameters to catch them.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        window_size: Size of window to use.

        overlap: Overlap between adjacent windows.

        jumpfinder_func: Jumpfinding function to use.

        **kwargs: Additional keyword args to pass to jumpfinder.
                  Arguments that will ultimately be passed to scipy.signal.find_peaks
                  should be passed after arguments specific to the jumpfinder.
                  The additional arguments to pass for each jumpfinder are below:

                  * _jumpfinder: None
                  * jumpfinder_tv: weight
                  * jumpfinder_gaussian: sigma

                  See docstrings of each jumpfinder for more details.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    jumps = np.zeros(x.shape, dtype=bool)
    tot = jumps.shape[-1]
    for i in range(tot // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, tot))
        _jumps = jumpfinder_func(
            x=x[..., start:end],
            min_chunk=min_chunk,
            min_size=min_size,
            win_size=win_size,
            nsigma=nsigma,
            max_depth=max_depth,
            **kwargs,
        )
        jumps[..., start:end] += _jumps
    return jumps


def jumpfix_median_sub(x, jumps, inplace=False):
    """
    Naive jump fixing routine where we median subtract between jumps.
    Note that you should exepect a glitch at the jump locations.

    Arguments:

        x: Data to jumpfix on, expects 1D or 2D.

        jumps: Boolean mask of that is True at jump locations.
               Should be the same shape at x.

        inplace: Whether of not x should be fixed inplace.

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

    padded_shape = list(jumps.shape)
    padded_shape[-1] += 2
    jumps_padded = np.ones(padded_shape, dtype=bool)
    jumps_padded[..., 1:-1] = jumps

    rows, cols = np.nonzero(np.diff(~jumps_padded, axis=-1))
    rows = rows[::2]
    cols = cols.reshape((-1, 2))

    diff = np.diff(cols, axis=-1).ravel()
    has_jumps = diff < x.shape[-1]
    rows = rows[has_jumps]
    cols = cols[has_jumps]

    for r, (c1, c2) in zip(rows, cols):
        x_fixed[r, c1:c2] -= np.median(x_fixed[r, c1:c2])

    return x_fixed.reshape(orig_shape)


def jumpfix_subtract_heights(x, jumps, heights, inplace=False):
    """
    Naive jump fixing routine where we subtract known heights between jumps.
    Note that you should exepect a glitch at the jump locations.

    Arguments:

        x: Data to jumpfix on, expects 1D or 2D.

        jumps: Boolean mask of that is True at jump locations.
               Should be the same shape at x.

        heights: Array of jump heights.
                 Should be a flat array of length n_jumps.

        inplace: Whether of not x should be fixed inplace.

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
        _heights = heights[n:m]
        height = _heights[np.argmax(np.abs(_heights))]
        x_fixed[r, int(np.mean([c1, c2])) :] -= height
        n = m

    return x_fixed.reshape(orig_shape)


def twopi_jumps(tod, signal=None, win_size=20, atol=None, fix=True, inplace=False):
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

        fix: If True the jumps will be fixed by adding N*2*pi at the jump locations.

        inplace: If True jumps will be fixed inplace.

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               Buffered to win_size.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = tod.signal
    if atol is None:
        atol = 3 * std_est(signal)
        np.clip(atol, 1e-8, 1e-2)

    diff_buffed = signal - np.roll(signal, win_size, axis=-1)

    if np.isscalar(atol):
        jumps = (np.isclose(0, np.abs(diff_buffed) % (2 * np.pi), atol=atol)) & (
            (np.abs(diff_buffed) // (2 * np.pi)) >= 1
        )
        jumps[:win_size] = False
    else:
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

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(int(win_size / 2))

    if fix:
        jumps = jump_ranges.mask()
        jumps[:, -2:] = False
        heights = diff_buffed[jump_ranges.mask()]
        # Round heights to the nearest 2pi
        heights = np.round(heights / (2 * np.pi)) * 2 * np.pi

        fixed = jumpfix_subtract_heights(signal, jumps, heights, inplace)

        return jump_ranges, fixed
    return jump_ranges


def find_jumps(
    tod,
    signal=None,
    max_iters=1,
    buff_size=0,
    jumpfinder=_jumpfinder,
    min_chunk=None,
    min_sigma=None,
    min_size=None,
    win_size=None,
    nsigma=None,
    max_depth=0,
    fix=None,
    inplace=False,
    **kwargs,
):
    """
    Find jumps in tod.signal_name.
    Expects tod.signal_name to be 1D of 2D.

    Arguments:

        tod: axis manager.

        signal: Signal to jumpfind on. If None than tod.signal is used.

        buff_size: How many samples to flag around each jump in RangesMatrix.

        jumpfinder: Jumpfinding function to use.

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

        win_size: Number of samples to average over when checking jump size.
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        fix: Method to use for jumpfixing.
             Set to None to not fix.

        inplace: Whether of not signal should be fixed inplace.

        **kwargs: Additional keyword args to pass to jumpfinder.

                  * _jumpfinder: None
                  * jumpfinder_tv: weight
                  * jumpfinder_gaussian: sigma
                  * jumpfinder_sliding_window: window_size, overlap, jumpfinder_func

                  See docstrings of each jumpfinder for more details.

                  Note that jumpfinder_sliding_window accepts kwargs to pass
                  on to whichever jumpfinder it calls as well.

    Returns:

        jumps: RangesMatrix containing jumps in signal,
               if signal is 1D Ranges in returned instead.
               There is some uncertainty on order of a few samples.
               Jumps within a few samples of each other may not be distinguished.

        fixed: signal with jump fixed. Only returned if fix is set.
    """
    if signal is None:
        signal = tod.signal

    if len(signal.shape) > 2:
        raise ValueError("Jumpfinder only works on 1D or 2D data")

    if min_size is None and min_sigma is not None:
        min_size = min_sigma * std_est(signal, axis=-1)
    if np.ndim(min_size) > 1:  # type: ignore
        raise ValueError("min_size must be 1d or a scalar")
    elif np.ndim(min_size) == 1:  # type: ignore
        min_size = np.array(min_size)

    do_fix = fix is not None
    is_med = False
    if do_fix:
        is_med = fix.__name__ == "jumpfix_median_sub"
    _signal = signal
    if not (inplace and is_med):
        _signal = signal.copy()

    # Median subtract, if we don't do this then when we cumsum we get floats
    # that are too big and lack the precicion to find jumps well
    _signal -= np.median(_signal, axis=-1)[..., None]

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
            **kwargs,
        )
        if np.sum(_jumps) == 0:
            break

        jumps[msk] += _jumps
        _signal[msk] = jumpfix_median_sub(_signal[msk], _jumps, True)
        msk = np.any(jumps, axis=-1)

    # TODO: include heights in output

    jump_ranges = RangesMatrix.from_mask(jumps).buffer(buff_size)

    if do_fix:
        if is_med:
            fixed = _signal
        else:
            fixed = fix(_signal, jumps, inplace)
        return jump_ranges, fixed
    return jump_ranges
