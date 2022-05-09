import numpy as np
import scipy.signal as sig
import scipy.ndimage as simg
import scipy.optimize as sopt
from so3g.proj import Ranges, RangesMatrix
from skimage.restoration import denoise_tv_chambolle


def std_est(x):
    """
    Estimate white noise standard deviation of data.
    More robust to jumps and 1/f then np.std()

    Arguments:

        x: Data to compute standard deviation of.

    Returns:

        stdev: The estimated white noise standard deviation of x.
    """
    # Find ~1 sigma limits of differenced data
    lims = np.quantile(np.diff(x), [0.159, 0.841])
    # Convert to standard deviation
    return (lims[1] - lims[0]) / 8 ** 0.5


def _jumpfinder(x, min_chunk, min_size, win_size, max_depth=-1, depth=0, **kwargs):
    """
    Recursive edge detection based jumpfinder.

    Note that the jumpfinder is very sensitive to changes in parameters
    and the parameters are not independant of each other,
    so it may take some playing around to get it to work properly.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        depth: The current recursion depth.

        **kwargs: Arguments to pass to scipy.signal.find_peaks.

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

    if len(x) < min_chunk:
        return np.array([], dtype=int)

    # If std is basically 0 no need to check for jumps
    if np.isclose(x.std(), 0.0) or (std_est(x) == 0):
        return np.array([], dtype=int)

    # Scale data to have std of order 1
    _x = x / std_est(x)

    # Mean subtract to make the jumps in the steps below more prominant peaks
    _x -= _x.mean()

    # Take cumulative sum, this is equivalent to convolving with a step
    x_step = np.cumsum(_x)

    # Look for peaks to find jumps
    u_jumps, _ = sig.find_peaks(x_step, **kwargs)
    d_jumps, _ = sig.find_peaks(-1 * x_step, **kwargs)
    jumps = np.concatenate([u_jumps, d_jumps])
    jumps.sort()

    # Filter out jumps that are too small
    sizes = get_jump_sizes(x, jumps, win_size)
    jumps = jumps[abs(sizes) > min_size]

    # If no jumps found return
    if len(jumps) == 0:
        return jumps.astype(int)

    # If at max_depth then return
    if depth == max_depth:
        return jumps.astype(int)

    # Recursively check for jumps between jumps
    _jumps = np.insert(jumps, 0, 0)
    _jumps = np.append(_jumps, len(x))
    added = 0
    for i in range(len(_jumps) - 1):
        sub_jumps = _jumpfinder(
            x[(_jumps[i]) : (_jumps[i + 1])],
            min_chunk,
            min_size,
            win_size,
            max_depth,
            depth + 1,
            **kwargs
        )
        jumps = np.insert(jumps, i + added, sub_jumps + _jumps[i])
        added += len(sub_jumps)
    return jumps.astype(int)


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
    max_depth=-1,
    weight=1,
    height=1,
    prominence=1,
    **kwargs
):
    """
    Apply total variance filter and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        weight: Denoising weight. Higher weights denoise more, lower weights
                preserve the input signal better.

        height: Height of peaks to pass to scipy.signal.find_peaks.

        prominence: Prominence of peaks to pass to scipy.signal.find_peaks.

        **kwargs: Additional arguments to pass to scipy.signal.find_peaks.

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

    x_filt = denoise_tv_chambolle(x, weight)
    return _jumpfinder(
        x_filt,
        min_chunk,
        min_size,
        win_size,
        max_depth,
        0,
        height=height,
        prominence=prominence,
        **kwargs
    )


def jumpfinder_gaussian(
    x,
    min_chunk=None,
    min_size=None,
    win_size=None,
    max_depth=-1,
    sigma=5,
    height=1,
    prominence=1,
    **kwargs
):
    """
    Apply gaussian filter to data and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.
                  Note that this is in terms of the filtered data.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

        sigma: Sigma of gaussian kernal.

        height: Height of peaks to pass to scipy.signal.find_peaks.

        prominence: Prominence of peaks to pass to scipy.signal.find_peaks.

        **kwargs: Additional arguments to pass to scipy.signal.find_peaks.

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
    x_filt = simg.gaussian_filter(x, sigma, 0)

    # Search for jumps in filtered data
    return _jumpfinder(
        x_filt,
        min_chunk,
        min_size,
        win_size,
        max_depth,
        0,
        height=height,
        prominence=prominence,
        **kwargs
    )


def jumpfinder_sliding_window(
    x,
    min_chunk=None,
    min_size=None,
    win_size=None,
    max_depth=-1,
    window_size=10000,
    overlap=1000,
    jumpfinder_func=jumpfinder_tv,
    **kwargs
):
    """
    Run jumpfinder through a sliding window.
    This can help get jumps towards the edges of the data that may be missed.
    Nominally those jumps can be found if the jumpfinder reaches sufficient depth,
    but sometimes it takes tweaking of the parameters to catch them.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        window_size: Size of window to use.

        overlap: Overlap between adjacent windows.

        jumpfinder_func: Jumpfinding function to use.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smallest jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

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
    jumps = np.array([], dtype=int)
    for i in range(len(x) // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, len(x)))
        _jumps = jumpfinder_func(
            x[start:end],
            min_chunk=min_chunk,
            min_size=min_size,
            win_size=win_size,
            max_depth=max_depth,
            **kwargs
        )
        jumps = np.hstack((jumps, _jumps + start))
    return np.unique(jumps).astype(int)


def find_jumps(
    tod,
    signal=None,
    buff_size=0,
    jumpfinder=jumpfinder_tv,
    min_chunk=None,
    min_sigma=5,
    min_size=None,
    win_size=None,
    max_depth=-1,
    **kwargs
):
    """
    Find jumps in tod.signal_name.
    Expects tod.signal_name to be 1D of 2D.

    Arguments:

        tod: axis manager.

        signal: Signal to jumpfind on. If None than tod.signal is used.

        buff_size: How many samples to flag around each jump in RangesMatrix.

        jumpfinder: Jumpfinding function to use.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_sigma: Number of standard deviations to count as a jump, note that
                   the standard deviation here is computed by std_est and is
                   the white noise standard deviation, so it doesn't include
                   contributions from jumps or 1/f.
                   If min_size is provided it will be used instead of this.

        min_size: The smallest jump size counted as a jump.
                  By default this is set to None and min_sigma is used instead,
                  if set this will override min_sigma.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negative for infite depth and 0 for no recursion.

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
    """
    if signal is None:
        signal = tod.signal

    jump_mask = np.zeros(signal.shape, dtype=bool)

    if len(signal.shape) == 1:
        if min_size is None:
            min_size = min_sigma * std_est(signal)
        jumps = jumpfinder(
            signal,
            min_chunk=min_chunk,
            min_size=min_size,
            win_size=win_size,
            max_depth=max_depth,
            **kwargs
        )
        jump_mask[jumps] = True
    elif len(signal.shape) == 2:
        for i, _signal in enumerate(signal):
            if min_size is None:
                _min_size = min_sigma * std_est(_signal)
            else:
                _min_size = min_size
            jumps = jumpfinder(
                _signal,
                min_chunk=min_chunk,
                min_size=_min_size,
                win_size=win_size,
                max_depth=max_depth,
                **kwargs
            )
            jump_mask[i][jumps] = True
    else:
        raise ValueError("Jumpfinder only works on 1D or 2D data")
    return RangesMatrix.from_mask(jump_mask).buffer(buff_size)


def fit_jumps(x, jumps, sizes, win_size=10, jump_range=5):
    """
    Try to fit for a more precise jump size and position.
    The parameter that is minimized is the size of the peak in the matched filter.

    Arguments:

        x: Signal to fit for jump on.
           Note that currently this function expects a signal with a single jump,
           so the input should be sliced accordingly.

        jumps: Jump locations.

        sizes: Sizes of jumps, used as initial value in fit.

        win_size: Maximum number of samples to include on either side of jump in slice.

        jump_range: Number of samples on either side of jump to fit for optimal jump location in.

    Returns:

        fit_res: The fit poistions and sizes of the jumps.
    """
    # TODO: simultaneous fit of multiple jumps?
    def min_func(size, x, pos):
        _x = x.copy()
        _x[pos:] -= size
        _x -= _x.mean()
        _x = np.cumsum(_x)
        return _x.max() - _x.min()

    fit_res = np.zeros((len(sizes), 2))
    for i, _j in enumerate(jumps):
        # Ugly brute force fit for optimal jump location
        _jumps = np.arange(_j - jump_range, _j + jump_range)
        if jump_range == 0:
            _jumps = [_j]
        fit_sizes = np.zeros(len(_jumps))
        fit_fun = np.zeros(len(_jumps))
        for k, j in enumerate(_jumps):
            if i + 1 < len(jumps):
                right = min(j + win_size, len(x), int((jumps[i + 1] + j) / 2))
            else:
                right = min(j + win_size, len(x))

            if i > 0:
                left = max(j - win_size, 0, int((jumps[i - 1] + j) / 2))
            else:
                left = max(j - win_size, 0)
            res = sopt.minimize(min_func, sizes[i], (x[left:right], j - left))
            fit_sizes[k] = res.x
            fit_fun[k] = res.fun
        idx = np.argmin(fit_fun)
        fit_res[i] = (idx - 5 + _j, fit_sizes[idx])
    return fit_res


def jumpfix(x, jumps, sizes, min_size=0, win_size=10, fit=True, **kwargs):
    """
    Fix jumps.
    Currently does the extremely naive technique of just subtracting the provided size.
    Jumps of size ~2*n*pi will be assumed to be tracking jumps of size 2*n*pi.

    Arguments:

        x: Signal to fix jumps on.

        jumps: Array of jump locations.

        sizes: Size of each jump, if None get_jump_sizes is called.

        min_size: Smallest size jump to fix.

        win_size: Window size for get_jump_sizes, also used to slice if fit=True.

        fit: Call fit_jumps to try to fit for a more accurate jump position and size.

        **kwargs: Arguments to pass on to np.isclose.

    Returns:

        _x: Signal with jumps fixed.
            Because jump positions may be off by a few samples, there are often
            glitches at the jump positions.
    """
    _x = x.copy()

    if sizes is None:
        sizes = get_jump_sizes(_x, jumps, win_size=win_size)

    if fit:
        fit_res = fit_jumps(_x, jumps, sizes).T
        jumps = fit_res[0].astype(int)
        sizes = fit_res[1]

    for j, s in zip(jumps, sizes):
        if abs(s) < min_size:
            continue
        # Check for tracking jump
        # TODO: Figure out correct atol and rtol for this
        if np.isclose(s / (2 * np.pi), np.round(s / (2 * np.pi)), **kwargs):
            s = 2 * np.pi * np.round(s / (2 * np.pi))
        _x[j:] -= s
    return _x


def fix_jumps(tod, signal=None, jumps=None, sizes=None):
    """
    Interface for jump fixing.
    Currently only supports very naive jump fixing but more to come.

    Arguments:

        tod: Axis manager.

        signal: Signal to jumpfix on. If None than tod.signal is used.

        jumps: Locations of jumps. Can be the following types:

                * ndarray: 1d or 2d array of jump locations.
                           If 1d signal must also be 1d.
                           If 2d signal must also be 2d and ordered the same.
                * Ranges: Ranges of jump location, currently doesn't assume any
                          buffering and takes the first index of each range to be
                          a jump. Signal must be 1d.
                * list: List of ndarrays or Ranges which will be treated as above.
                        Signal must be 2d and ordered the same.
                        If list contains ndarrays they must be 1d.
                * RangesMatrix: Each range in RangesMatrix is treated as above.
                                Signal must be 2d and ordered the same.

        sizes: Jump sizes.
               If signal is 1d this should be an iterable where each element is a size.
               If signal is 2d this should be an iterable where each element is an
               iterable containing sizes for the corresponding row in signal.
               Sizes should have the same ordering as jumps.

        Returns:

            fixed_signal: Signal with jump fixing applied.
                          Note that because jump positions may be off by a few samples,
                          there are often glitches at the jump positions in this output.
    """
    if signal is None:
        signal = tod.signal

    ndim = len(signal.shape)
    if ndim not in (1, 2):
        raise ValueError("Jumpfixer only works on 1D or 2D data")

    if isinstance(jumps, np.ndarray):
        jump_locs = jumps
        jdim = jump_locs.ndim
    elif isinstance(jumps, Ranges):
        jdim = 1
        jump_locs = jumps.ranges().flatten()[::2]
    elif isinstance(jumps, list):
        jdim = 2
        jump_locs = []
        for i, _jumps in enumerate(jumps):
            if isinstance(_jumps, np.ndarray):
                jump_locs.append(_jumps)
            elif isinstance(_jumps, Ranges):
                jump_locs.append(_jumps.ranges().flatten()[::2])
            else:
                raise ValueError("Jumps provided in invalid format")
    elif isinstance(jumps, RangesMatrix):
        jdim = 2
        jump_locs = []
        for _jumps in jumps.ranges:
            jump_locs.append(_jumps.ranges().flatten()[::2])
    else:
        raise ValueError("Jumps provided in invalid format")

    if jdim != ndim:
        raise ValueError("Jumps and signal don't have the same number of dimensions")

    if ndim == 1:
        if jump_locs.shape != sizes.shape:
            raise ValueError("Number of sizes does not match number of jumps")
        return jumpfix(signal, jump_locs, sizes)

    signal_fixed = np.zeros(signal.shape)
    for i, _jump_locs in enumerate(jump_locs):
        if _jump_locs.shape != sizes[i].shape:
            raise ValueError("Number of sizes does not match number of jumps")
        signal_fixed[i] = jumpfix(signal[i], _jump_locs, sizes[i])
    return signal_fixed
