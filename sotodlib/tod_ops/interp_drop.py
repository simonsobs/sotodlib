import numpy as np
from sotodlib import core
from so3g.proj import Ranges

def interp_drop_array(arr, drop_idxes):
    """
    Function to interpolate numpy.array.
    Insert the mean of left and right samples at the dropped indexes
    
    Args:
        arr: array contains data drops.
        drop_idxes: Indexes of dropped samples.
        
    Return:
        new_arr: array that drops are filled.
    """
    
    new_arr = np.copy(arr)
    for  i in range(len(drop_idxes)):
        drop_idx = drop_idxes[i] + i # '+i' means index shift by previous insertions
        if arr.ndim >= 3:
            raise ValueError(f'Unsupported data dimensions (arr.ndim={arr.ndim})')
        else:
            inserted_data = (new_arr[..., drop_idx-1] + new_arr[..., drop_idx])/2.
            new_arr = np.insert(new_arr, drop_idx, inserted_data, axis=-1)
 
    return new_arr

def interp_drop_single_aman(aman, drop_idxes):
    """
    Function to interpolate axismanager.
    Insert the mean of left and right samples at the dropped indexes.
    
    Args:
        aman: axismanager object.
        drop_idxes: Indexes of dropped samples.
    Return:
        new_aman: axismanager object that drops are filled.
    """
    new_samps = core.OffsetAxis(name='samps', 
                                count=aman.samps.count + len(drop_idxes), 
                                offset=aman.samps.offset)
    
    other_axes = []
    for axis_name, axis_obj in aman._axes.items():
        if axis_name != 'samps':
            other_axes.append(axis_obj)
            
    new_aman = core.AxisManager(new_samps, *other_axes)
    
    for field, axis_names in aman._assignments.items():
        axes = aman._assignments[field]
        data = aman[field]
        
        if 'samps' in axes:
            if type(data) == np.ndarray:
                new_data = interp_drop_array(data, drop_idxes)
                new_aman.wrap(field, new_data, [(i, axis) for i,axis in enumerate(axes)])
            else:
                # If the data is aman or flagman, recursively derive new data
                new_data = interp_drop_single_aman(data, drop_idxes)
                new_aman.wrap(field, new_data)
        else:
            new_aman.wrap(field, data)
    return new_aman

def interp_drop(aman):
    """
    Function to interpolate axismanager.
    Uses interp_drop_single_aman to enable recursive iteration.
    It checks 'aman.primary.FrameCounter', which is primary timestamps of smurf,
    as an indicator of data drop, and if there is drops, interpolate the dropped sample
    with the mean of data of the both side.
    After making a new axismaneger object, add 'interp_drop' flag to aman.flags.
    
    Args:
        aman: axismanager object.
    Return:
        new_aman: axismanager object that drops are filled.
    """
    delta_frame_counter = np.median(np.diff(aman.primary.FrameCounter))
    drop_idxes = np.where(np.diff(aman.primary.FrameCounter) > 1.1*delta_frame_counter)[0] + 1
    
    consecutive_drop_idxes = np.where(np.diff(aman.primary.FrameCounter)>2.1*delta_frame_counter)[0] + 1
    if consecutive_drop_idxes.size != 0:
        raise ValueError(f'Index {consecutive_drop_idxes} drops more than 2 points consecutively')
    
    if len(drop_idxes) == 0:
        print('no interpolation is applied')
        return aman
    
    new_aman = interp_drop_single_aman(aman, drop_idxes)
    
    new_drop_idxes = drop_idxes + np.arange(0, len(drop_idxes))
    mask_interp_drop = np.zeros(len(new_aman.timestamps), dtype='bool')
    mask_interp_drop[new_drop_idxes] = True
    flag = Ranges.from_bitmask(mask_interp_drop)
    new_aman.flags.wrap('interp_drop', flag)
    
    return new_aman
