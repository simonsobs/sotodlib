import numpy as np
import scipy.signal as sig
import scipy.ndimage as simg
from so3g.proj import RangesMatrix
from skimage.restoration import denoise_tv_chambolle


def _jumpfind(x, min_chunk, min_size, win_size, max_depth=-1, depth=0, **kwargs):
    """
    Recursive edge detection based jumpfinder.

    Note that the jumpfinder is very sensitive to changes in parameters
    and the parameters are not independant of each other,
    so it may take some playing around to get it to work properly.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smalled jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negetive for infite depth and 0 for no recursion.

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

    # Mean subtract to make the jumps in the steps below more prominant peaks
    _x = x - x.mean()

    # Scale data to have std of order 1
    if _x.std() > 10:
        _x = _x / (10 ** (int(np.log10(_x.std()))))

    # Take cumulative sum, this is equivalent to convolving with a step
    x_step = np.cumsum(_x)

    # Look for peaks to find jumps
    u_jumps, _ = sig.find_peaks(x_step, **kwargs)
    d_jumps, _ = sig.find_peaks(-1 * x_step, **kwargs)
    jumps = np.concatenate([u_jumps, d_jumps])
    jumps.sort()

    # Filter out jumps that are too small
    j_i = []
    for i, j in enumerate(jumps):
        if i + 1 < len(jumps):
            right = min(j + win_size, len(x), jumps[i + 1] - min_chunk)
        else:
            right = min(j + win_size, len(x))
        if i > 0:
            left = max(j - win_size, 0, jumps[i - 1] + min_chunk)
        else:
            left = max(j - win_size, 0)

        if (
            abs(
                np.median(x[j + min_chunk : right]) - np.median(x[left : j - min_chunk])
            )
            > min_size
        ):
            j_i.append(i)
    jumps = jumps[j_i]

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
        sub_jumps = _jumpfind(
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


def jumpfind_tv(
    x,
    weight=1,
    min_chunk=10,
    min_size=0.5,
    win_size=20,
    max_depth=-1,
    height=1,
    prominence=1,
    **kwargs
):
    """
    Apply total variance filter and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        weight: Denoising weight. Higher weights denoise more, lower weights
                preserve the input signal better.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smalled jump size counted as a jump.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negetive for infite depth and 0 for no recursion.

        height: Height of peaks to pass to scipy.signal.find_peaks.

        prominence: Prominence of peaks to pass to scipy.signal.find_peaks.

        **kwargs: Additional arguments to pass to scipy.signal.find_peaks.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    x_filt = denoise_tv_chambolle(x, weight)
    return _jumpfind(
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


def jumpfind_gaussian(
    x,
    sigma=5,
    min_chunk=10,
    min_size=0.1,
    win_size=20,
    max_depth=-1,
    height=1,
    prominence=1,
    **kwargs
):
    """
    Apply gaussain filter to data and then search for jumps.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        sigma: Sigma of gaussain kernal.

        min_chunk: The smallest chunk of data to look for jumps in.

        min_size: The smalled jump size counted as a jump.
                  Note that this is in terms of the filtered data.

        win_size: Number of samples to average over when checking jump size.

        max_depth: The maximum recursion depth.
                   Set negetive for infite depth and 0 for no recursion.

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
    return _jumpfind(
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


def jumpfind_recursive_gaussian(x, sensitivity=2.0, max_depth=-1, depth=0):
    """
    Calculate (semi) intelligent default parameters and jumpfind with them
    The data is gaussain filtered before jumpfinding.

    The presence of large spikes can have a negetive effect on this function,
    please mask/slice/interpolate them out before using this functon.

    Note that the difference between this and jumpfind_gaussian is that in this
    function the filter is applied in each segment at it recurses.
    It also attempts to compute parameters based on the std of the data.

    Arguments:

        x: Data to jumpfind on, expects 1D.

        sensitivity: Sensitivity of the jumpfinder, roughly correlates with
                     1/(jump size) but since data is filtered, detrended, and
                     rescaled during jumpfinding it is better to think of it as
                     something non-physical.

        max_depth: The maximum recursion depth.
                   Set negetive for infite depth and 0 for no recursion.

        depth: The current recursion depth.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of a few samples.
               Jumps within 20 samples of each other may not be distinguished.
    """
    _x = sig.detrend(x)
    if _x.std() > 10:
        _x = _x / (10 ** (int(np.log10(_x.std()))))
    if np.isclose(_x.std(), 0.0):
        return np.array([])

    jumps = jumpfind_gaussian(
        _x,
        2 * _x.std(),
        10,
        1.0 / sensitivity,
        20,
        0,
        height=1,
        prominence=1,
    )

    # If no jumps found return
    if len(jumps) == 0:
        return jumps

    # If at max_depth then return
    if depth == max_depth:
        return jumps

    # Recursively check for jumps between jumps
    _jumps = np.insert(jumps, 0, 0)
    _jumps = np.append(_jumps, len(x))
    added = 0
    for i in range(len(_jumps) - 1):
        sub_jumps = jumpfind_recursive_gaussian(
            x[(_jumps[i]) : (_jumps[i + 1])], sensitivity, max_depth, depth + 1
        )
        jumps = np.insert(jumps, i + added, sub_jumps + _jumps[i])
        added += len(sub_jumps)
    return jumps


def jumpfind_sliding_window(
    x, window_size=10000, overlap=1000, jumpfinder=jumpfind_tv, **kwargs
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

        jumpfinder: Jumpfinding function to use.

        **kwargs: Keyword args to pass to jumpfinder.

    Returns:

        jumps: The indices of jumps in x.
               There is some uncertainty on order of 1 sample.
               Jumps within min_chunk of each other may not be distinguished.
    """
    jumps = np.array([], dtype=int)
    for i in range(len(x) // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, len(x)))
        _jumps = jumpfinder(x[start:end], **kwargs) + start
        jumps = np.hstack((jumps, _jumps))
    return np.unique(jumps).astype(int)


def jumpfind(tod, signal=None, buff_size=10, jumpfinder=jumpfind_tv, **kwargs):
    """
    Find jumps in tod.signal_name.
    Expects tod.signal_name to be 1D of 2D.

    Arguments:

        tod: axis manager.

        signal: Signal to jumpfind on. If None than tod.signal is used.

        buff_size: How many samples to flag around each jump in RangesMatrix.

        jumpfinder: Jumpfinding function to use.

        **kwargs: Keyword args to pass to jumpfinder.

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
        jumps = jumpfinder(signal, **kwargs)
        jump_mask[jumps] = True
    elif len(signal.shape) == 2:
        for i, _signal in enumerate(signal):
            jumps = jumpfinder(_signal, **kwargs)
            jump_mask[i][jumps] = True
    else:
        raise ValueError("Jumpfinder only works on 1D or 2D data")
    return RangesMatrix.from_mask(jump_mask).buffer(buff_size)
