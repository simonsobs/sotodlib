import numpy as np 
import sotodlib.flags as flags
import logging
logger = logging.getLogger(__name__)

def subscan_polyfilter(aman, degree, signal='signal', exclude_turnarounds=True, mask=None):
    """
    Apply polynomial filtering to subscan segments in a data array.

    This function applies polynomial filtering to subscan segments within aman[`signal`] for each detector.
    Subscan segments are defined based on the presence of flags such as 'left_scan' and 'right_scan'. Polynomial filtering
    is used to remove low-degree polynomial trends within each subscan segment.

    Parameters:
    - aman (object): AxisManager object
    - degree (int): The degree of the polynomial to be removed.
    - signal (str, optional): The name of the signal in 'aman' to which polynomial filtering is applied. Default is 'signal'.
    - exclude_turnarounds (bool, optional): If True, turnarounds are excluded from subscan identification. Default is True.
    - mask (str or RangesMatrix, optional): A mask used to select specific data points for filtering. Default is None.
        If None, no mask is applied. If the mask is given in str, aman.flags[`mask`] is used as mask. Arbitrary mask can be 
        specified in the style of RangesMatrix.
    """
    if exclude_turnarounds:
        if ("left_scan" not in aman.flags) or ("turnarounds" not in aman.flags):
            raise ValueError('Flag turnarounds,left_scan, and right_scan by `sotodlib.flags.get_turnaround_flags`')
        valid_scan = np.logical_and(np.logical_or(aman.flags["left_scan"].mask(), aman.flags["right_scan"].mask()),
                                    ~aman.flags["turnarounds"].mask())
        subscan_indices = _get_subscan_range_index(valid_scan)
    else:
        if ("left_scan" not in aman.flags):
            raise ValueError('Flag left_scan and right_scan by `sotodlib.flags.get_turnaround_flags`')
        subscan_indices_l = _get_subscan_range_index(aman.flags["left_scan"].mask())
        subscan_indices_r = _get_subscan_range_index(aman.flags["right_scan"].mask())
        subscan_indices = np.vstack([subscan_idx_l, subscan_idx_r])
        subscan_indices= subscan_indices[np.argsort(subscan_indices[:, 0])]
    
    if mask is None:
        mask_array = np.zeros(aman.samps.count, dtype=bool)
    elif type(mask) is str:
        mask_array = aman.flags[mask].mask()
    else:
        mask_array = mask.mask()
    is_matrix = len(mask_array.shape) > 1
    
    t = aman.timestamps - aman.timestamps[0]
    for i_det in range(aman.dets.count):
        if is_matrix:
            each_det_mask = mask_array[i_det]
        else:
            each_det_mask = mask_array

        for start, end in subscan_indices:
            if np.count_nonzero(~each_det_mask[start:end+1]) < degree:
                # If degree of freedom is lower than zero, just subtract mean
                aman[signal][i_det, start:end+1] -= np.mean(aman[signal][i_det, start:end+1])
            else:
                t_mean = np.mean(t[start:end+1])
                pars = np.ma.polyfit(
                        np.ma.array(t[start:end+1]-t_mean, mask=each_det_mask[start:end+1]),
                        np.ma.array(aman[signal][i_det,start:end+1], mask=each_det_mask[start:end+1]),
                        deg=degree)

                aman[signal][i_det,start:end+1] -= np.polyval(pars, t[start:end+1]-t_mean)
    return
                
def _get_subscan_range_index(scan_flag,_min=0):
    """
    Get the indices of subscans in a binary flag array.
    This function identifies subscan ranges within a binary flag array, where subscans are defined as consecutive
    sequences of 'True' values in the input 'scan_flag' array.

    Parameters:
    - scan_flag (numpy.ndarray): A 1-dimensional binary flag array indicating the presence of subscans.
    - _min (int, optional): The minimum length of a subscan range to consider. Default is 0.

    Returns:
    - numpy.ndarray: A 2-column array containing the start and end indices of subscan ranges in 'scan_flag' where
      the length of each subscan is greater than or equal to '_min'.
    """
    ones = np.where(scan_flag)[0]
    diff = np.diff(ones)
    
    starts = np.insert(ones[np.where(diff != 1)[0]+1], 0, ones[0])
    ends = np.append(ones[np.where(diff != 1)[0] ], ones[-1])
    indices = list(zip(starts, ends))
    indices = np.array([(start, end) for start, end in indices if (end - start + 1 >= _min)])
    return indices
 