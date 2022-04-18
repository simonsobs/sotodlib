import numpy as np
import scipy.signal as sig
import scipy.ndimage as simg
from so3g.proj import RangesMatrix
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
    if len(x) < min_chunk:
        return np.array([], dtype=int)

    # If std is basically 0 no need to check for jumps
    if np.isclose(x.std(), 0.0):
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
    sizes = get_jump_sizes(x, jumps, min_chunk, win_size)
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


def get_jump_sizes(x, jumps, min_chunk, win_size):
    """
    Estimate jumps sizes.

    Arguments:

        x: Data with jumps, expects 1D.

        jumps: Indices of jumps in x.

        min_chunk: The smallest chunk of data to look for jumps in.

        win_size: Number of samples to average over when checking jump size.

    Returns:

        sizes: Array of jump sizes, same order and jumps.
    """
    sizes = np.zeros(len(jumps))
    for i, j in enumerate(jumps):
        if i + 1 < len(jumps):
            right = min(j + win_size, len(x), jumps[i + 1] - min_chunk)
        else:
            right = min(j + win_size, len(x))
        right_height = np.median(x[j + min_chunk : right])

        if i > 0:
            left = max(j - win_size, 0, jumps[i - 1] + min_chunk)
        else:
            left = max(j - win_size, 0)
        left_height = np.median(x[left : j - min_chunk])

        sizes[i] = right_height - left_height
    return sizes.astype(int)


def jumpfinder_tv(
    x,
    min_chunk=10,
    min_size=0.5,
    win_size=20,
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
    min_chunk=10,
    min_size=0.1,
    win_size=20,
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
    min_chunk=10,
    min_size=0.1,
    win_size=20,
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
    min_chunk=10,
    min_sigma=5,
    min_size=None,
    win_size=20,
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
