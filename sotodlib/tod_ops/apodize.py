import numpy as np

def get_apodize_window_for_ends(aman, apodize_samps=1600):
    """
    Generate an apodization window using a cosine taper at the beginning and end.

    Args:
        aman: An axismanager
        apodize_samps (int): Number of samples to apply the cosine taper to at each end.

    Returns:
        numpy.ndarray: An array representing the apodization window.
    """
    w = np.ones(aman.samps.count)
    cosedge = np.cos(np.linspace(0, np.pi/2, apodize_samps))
    w[-apodize_samps:] = cosedge
    w[:apodize_samps] = np.flip(cosedge)
    return w
    
def get_apodize_window_from_flags(aman, flags, apodize_samps=200):
    """
    Generate an apodization window based on flag values. Apply cosine tapering every 
    continuous portion of data between flagged region.

    Args:
        aman: An axismanager
        flags (str or RangesMatrix): Flags of mask in RangesMatrix. If provided by 
            a string, 'aman.flags[flags]' is used for the flags.
        apodize_samps (int): Number of samples to apply the cosine taper.

    Returns:
        numpy.ndarray: An array representing the apodization window.
    """
    if isinstance(flags, str):
        flags = aman.flags[flags]
    apodizer = ~flags.mask()
    apodizer = apodizer.astype(float)
    cosedge = np.cos(np.linspace(0, np.pi/2, apodize_samps))
    
    for di, det in enumerate(aman.dets.vals):
        idxes_left = np.where(np.diff(apodizer[di]) == -1)[0]
        idxes_right = np.where(np.diff(apodizer[di]) == 1)[0]

        for _i, (_left, _right) in enumerate(zip(idxes_left, idxes_right)):            
            _apo_idxes_left = (_left-apodize_samps+1, _left+1)
            _apo_idxes_right = (_right-1, _right+apodize_samps-1)
            
            if _i == 0:
                if _apo_idxes_left[0] < 0:
                    apodizer[di][:_apo_idxes_left[1]] = 0
                else:
                    apodizer[di][_apo_idxes_left[0]:_apo_idxes_left[1]] = cosedge
                apodizer[di][_apo_idxes_right[0]:_apo_idxes_right[1]] = np.flip(cosedge)
                
            elif _i == len(idxes_left) - 1:
                apodizer[di][_apo_idxes_left[0]:_apo_idxes_left[1]] = cosedge
                if _apo_idxes_right[1] > aman.samps.count:                    
                    apodizer[di][_apo_idxes_right[0]:] = 0
                else:
                    apodizer[di][_apo_idxes_right[0]:_apo_idxes_right[1]] = np.flip(cosedge)
            else:
                apodizer[di][_apo_idxes_left[0]:_apo_idxes_left[1]] = cosedge
                apodizer[di][_apo_idxes_right[0]:_apo_idxes_right[1]] = np.flip(cosedge)
    return apodizer
    
def apodize_cosine(aman, signal='signal', apodize_samps=1600, in_place=True,
                   apo_axis='apodized', window=None):
    """
    Function to smoothly filter the timestream to 0's on the ends with a
    cosine function. If window is 

    Args:
        signal (str): Axis to apodize
        apodize_samps (int): Number of samples on tod ends to apodize.
        in_place (bool): writes over signal with apodized version
        apo_axis (str): Axis to store the apodized signal if not in place.
        window (numpy.ndarray): Precomputed apodization window.
    """
    if window is None:
        w = get_apodize_window_for_ends(aman, apodize_samps)
        
    if in_place:
        aman[signal] *= w
    else:
        aman.wrap_new(apo_axis, dtype='float32', shape=('dets', 'samps'))
        aman[apo_axis] = aman[signal]*w
    return
