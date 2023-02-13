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
    data_dim = len(np.shape(arr))
    new_arr = np.copy(arr)
    
    for  i in range(len(drop_idxes)):
        drop_idx = drop_idxes[i] + i
        if data_dim == 1:
            # Such as azimuth or elevation
            inserted_data = (new_arr[drop_idx-1] + new_arr[drop_idx])/2.
            new_arr = np.insert(new_arr, drop_idx, inserted_data)
            
        elif data_dim == 2:
            inserted_data = (new_arr[:, drop_idx-1] + new_arr[:, drop_idx])/2.
            new_arr = np.insert(new_arr, drop_idx, inserted_data, axis=1)
        else:
            raise ValueError(f'Unsupported data dimensions ({field})')
 
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

def interp_drop(aman, overwrite=True):
    """
    Function to interpolate axismanager.
    Uses interp_drop_single_aman to enable recursive iteration.
    Add 'interp_drop' flag to aman.flags.
    
    Args:
        aman: axismanager object.
        overwrite: If True, aman is update to drop-filled one. 
        Else returns new drop-filled axismanager object.
    Return:
        new_aman: axismanager object that drops are filled.
    """
    drop_idxes = np.where(np.diff(aman.primary.FrameCounter) > np.median(np.diff(aman.primary.FrameCounter)))[0] + 1
    
    if len(drop_idxes) == 0:
        print('no interpolation is applied')
        return
    new_aman = interp_drop_single_aman(aman, drop_idxes)
    
    new_drop_idxes = drop_idxes + np.arange(0, len(drop_idxes))
    mask_interp_drop = np.zeros(len(new_aman.timestamps), dtype='bool')
    mask_interp_drop[new_drop_idxes] = True
    
    flag = Ranges.from_bitmask(mask_interp_drop)
    new_aman.flags.wrap('interp_drop', flag)
    
    if overwrite==True:
        aman = new_aman
        return aman
    else:
        return new_aman