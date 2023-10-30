import numpy as np
import scipy.ndimage as simg
import scipy.signal as sig
from skimage.restoration import denoise_tv_chambolle
from so3g.proj import RangesMatrix


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
    return (lims[1] - lims[0]) / 8**0.5


def _jumpfinder(x, min_chunk, min_size, win_size, nsigma, max_depth=-1, depth=0):
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
        min_size = 0.1
    if win_size is None:
        win_size = 10
    if nsigma is None:
        nsigma = 5

    if len(x) < min_chunk:
        return np.array([], dtype=int)

    if np.max(x) - np.min(x) < min_size:
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

    # Smooth and take the second derivative
    sg_x_step = np.abs(sig.savgol_filter(x_step, win_size, 2))

    # Peaks should be jumps
    # Doing the simple thing and looking for things much larger than the median
    peaks = np.where(sg_x_step > np.median(sg_x_step) + nsigma * std_est(sg_x_step))[0]
    # The peak may have multiple points above this criteria
    gaps = np.diff(peaks) > win_size
    begins = np.insert(peaks[1:][gaps], 0, peaks[0])
    ends = np.append(peaks[:-1][gaps], peaks[-1])
    jumps = ((begins + ends) / 2).astype(int) + 1

    # Filter out jumps that are too small
    # TODO: There must be a way to get jump size from x_step...
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
            nsigma,
            max_depth,
            depth + 1,
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
    nsigma=None,
    max_depth=1,
    weight=1,
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

    x_filt = denoise_tv_chambolle(x, weight)
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
    sigma=5,
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
    **kwargs
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
    jumps = np.array([], dtype=int)
    for i in range(len(x) // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, len(x)))
        _jumps = jumpfinder_func(
            x[start:end],
            min_chunk=min_chunk,
            min_size=min_size,
            win_size=win_size,
            nsigma=nsigma,
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
    nsigma=None,
    max_depth=1,
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
                  Also used to apply the SG filter when peak finding.

        nsigma: Number of sigma above the mean for something to be a peak.

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

    # TODO: Move the bool mask creation to _jumpfinder so that this can be vectorized.
    jump_mask = np.zeros(signal.shape, dtype=bool)

    if len(signal.shape) == 1:
        if min_size is None:
            min_size = min_sigma * std_est(signal)
        jumps = jumpfinder(
            signal,
            min_chunk=min_chunk,
            min_size=min_size,
            win_size=win_size,
            nsigma=nsigma,
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
                nsigma=nsigma,
                max_depth=max_depth,
                **kwargs
            )
            jump_mask[i][jumps] = True
    else:
        raise ValueError("Jumpfinder only works on 1D or 2D data")
    # TODO: include heights in output
    return RangesMatrix.from_mask(jump_mask).buffer(buff_size)
