import numpy as np

def detrend_data(tod, method='linear', axis_name='samps', 
                 signal_name='signal', count=10):
    
    """Returns detrended data. Decide for yourself if it goes
        into the axis manager. Generally intended for use before filter.
        Using this with method ='mean' and axis_name='dets' will remove a 
        common mode from the detectors
        Using this with method ='median' and axis_name='dets' will remove a 
        common mode from the detectors with the median rather than the mean
    
    Arguments:
    
        tod: axis manager
    
        method: method of detrending can be 'linear', 'mean', or median
        
        axis_name: the axis along which to detrend. default is 'samps'
        
        signal_name: the name of the signal to detrend. defaults to 'signal'.
            Can have any shape as long as axis_name can be resolved.

        count: Number of samples to use, on each end, when measuring
            mean level for 'linear' detrend.  Values larger than 1
            suppress the influence of white noise.

    Returns:
        
        detrended signal: does not actually detrend the data in the axis
            manager, let's you decide if you want to do that.

    """
    signal = tod[signal_name]
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
        signal = signal - np.mean(signal, axis=-1)[...,None]
    elif method == 'median':
        signal = signal - np.median(signal, axis=-1)[...,None]
    elif method == 'linear':
        x = np.linspace(0, 1, n_samps)
        count = max(1, min(count, signal.shape[-1] // 2))
        slopes = signal[...,-count:].mean(axis=-1)-signal[...,:count].mean(axis=-1)
        if len(signal.shape)>2:
            signal -= slopes[...,None] * x
        ## the 2d loop is significantly faster if possible
        else:
            for i in range(signal.shape[0]):
                signal[i,:] -= slopes[i]*x
        signal -= np.mean(signal, axis=-1)[...,None]
    else:
        raise ValueError("method flag must be linear, mean, or median")

    if axis_idx != signal.ndim - 1:
        signal = signal.transpose(tuple(axis_reorder))

    if signal.dtype != dtype_in:
        # .astype takes awhile on long arrays, only cast if necessary
        return signal.astype(dtype_in)
    else:
        return signal

def detrend_tod(tod, method='linear', signal_name='signal', axis_name='samps', count=10, out_name=None):
    """simple wrapper: to be more verbose"""
    if out_name is None:
        out_name = signal_name
        
    signal = detrend_data(tod, method=method, 
                          axis_name=axis_name, 
                          signal_name=signal_name,
                          count=count)

    tod[out_name] = signal
    
    return tod
