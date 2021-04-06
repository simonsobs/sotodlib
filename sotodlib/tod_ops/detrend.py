import numpy as np

def detrend_data(tod, method='linear', axis_name='samps', 
                 signal_name='signal', count=10):
    
    """Returns detrended data. Decide for yourself if it goes
        into the axis manager. Generally intended for use before filter.
        Using this with method ='mean' and axis_name='dets' will remove a 
        common mode from the detectors
    
    Arguments:
    
        tod: axis manager
    
        method: method of detrending can be 'linear' or 'mean'
        
        axis_name: the axis along which to detrend. default is 'samps'
        
        signal_name: the name of the signal to detrend. defaults to 'signal'
            if it isn't 2D it is made to be.

        count: Number of samples to use, on each end, when measuring
            mean level for 'linear' detrend.  Values larger than 1
            suppress the influence of white noise.

    Returns:
        
        detrended signal: does not actually detrend the data in the axis
            manager, let's you decide if you want to do that.

    """
    assert len(tod._assignments[signal_name]) <= 2
        
    signal = np.atleast_2d(getattr(tod, signal_name))
    axis = getattr(tod, axis_name)
    
    if len(tod._assignments[signal_name])==1:
        ## will have gotten caught by atleast_2d
        idx = 1
        other_idx = None
        
    elif len(tod._assignments[signal_name])==2:
        checks = np.array([x==axis_name for x in tod._assignments[signal_name]],dtype='bool')
        idx = np.where(checks)[0][0]
        other_idx = np.where(~checks)[0][0]
    
    if other_idx is not None and other_idx == 1:
        signal = signal.transpose()
        
    if method == 'mean':
        signal = signal - np.mean(signal, axis=1)[:,None]
    elif method == 'linear':
        x = np.linspace(0,1, axis.count)
        count = max(1, min(count, signal.shape[-1] // 2))
        slopes = signal[:,-count:].mean(axis=-1)-signal[:,:count].mean(axis=-1)
        signal = signal - slopes[:,None]*x 
        signal -= np.mean(signal, axis=1)[:,None]
    else:
        raise ValueError("method flag must be linear or mean")

    if other_idx is not None and other_idx == 1:
        signal = signal.transpose()
    return signal

def detrend_tod(tod, method='linear', signal_name='signal', axis_name='samps', out_name=None):
    """simple wrapper: to be more verbose"""
    if out_name is None:
        out_name = signal_name
        
    signal = detrend_data(tod, method=method, 
                          axis_name=axis_name, 
                          signal_name=signal_name)

    tod[out_name] = signal
    
    return tod
