import numpy as np


def detrend_tod(
    tod,
    method='linear',
    axis_name='samps',
    signal_name='signal',
    in_place=True,
    wrap_name=None,
    count=10
):
    """Returns detrended data. Detrends data in place by default but pass
    in_place=False if you would like a copied array (such as if you're just
    looking to use this in an FFT).

    Using this with method ='mean' and axis_name='dets' will remove a
    common mode from the detectors
    Using this with method ='median' and axis_name='dets' will remove a
    common mode from the detectors with the median rather than the mean

    Arguments
    ---------
        tod: axis manager
        method: str
            method of detrending can be 'linear', 'mean', or median
        axis_name: str
            the axis along which to detrend. default is 'samps'
        signal_name: str
            the name of the signal to detrend. defaults to 'signal'. Can have
            any shape as long as axis_name can be resolved.
        in_place: bool.
            If False it makes a copy of signal before detrending and returns
            the copy.
        wrap_name: str or None.
            If not None, wrap the detrended data into tod with this name.
        count: int
            Number of samples to use, on each end, when measuring mean level
            for 'linear' detrend.  Values larger than 1 suppress the influence
            of white noise.

    Returns
    -------
        signal: array of type tod[signal_name]
            Detrended signal. Done in place or on a copy depend on in_place
            argument.
    """

    signal = tod[signal_name]
    if not in_place:
        signal = signal.copy()
    dtype_in = signal.dtype

    axis_idx = list(tod._assignments[signal_name]).index(axis_name)
    n_samps = signal.shape[axis_idx]

    # Ensure last axis is the one to detrend.
    # note that any axis re-ordering is not the slow part of this function
    if axis_idx != signal.ndim - 1:
        # Note this reordering is its own inverse; e.g. [0, 1, -1, 3, 4, 2]
        axis_reorder = list(range(signal.ndim))
        axis_reorder[axis_idx], axis_reorder[-1] = -1, axis_idx
        signal = signal.transpose(tuple(axis_reorder))

    if method == 'mean':
        signal = signal - np.mean(signal, axis=-1)[..., None]
    elif method == 'median':
        signal = signal - np.median(signal, axis=-1)[..., None]
    elif method == 'linear':
        x = np.linspace(0, 1, n_samps, dtype=dtype_in)
        count = max(1, min(count, signal.shape[-1] // 2))
        slopes = (signal[..., -count:].mean(axis=-1, dtype=dtype_in) -
                  signal[..., :count].mean(axis=-1, dtype=dtype_in))

        # the 2d loop is significantly faster if possible
        if len(signal.shape) == 2:
            for i in range(signal.shape[0]):
                signal[i, :] -= slopes[i]*x
        else:
            signal -= slopes[..., None] * x
        signal -= np.mean(signal, axis=-1)[..., None]
    else:
        raise ValueError("method flag must be linear, mean, or median")

    if axis_idx != signal.ndim - 1:
        signal = signal.transpose(tuple(axis_reorder))

    assert signal.dtype == dtype_in

    if wrap_name is not None:
        axis_map = [(
            i, x) for i, x in enumerate(tod._assignments[signal_name])
        ]
        tod.wrap(wrap_name, signal, axis_map)

    return signal
