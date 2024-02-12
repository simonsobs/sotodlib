import numpy as np
import logging
from . import flags
logger = logging.getLogger(__name__)

def subscan_polyfilter(aman, degree, signal=None, exclude_turnarounds=False, mask=None, in_place=True, wrap=None):
    """
    Apply polynomial filtering to subscan segments in a data array.
    This function applies polynomial filtering to subscan segments within signal for each detector.
    Subscan segments are defined based on the presence of flags such as 'left_scan' and 'right_scan'. Polynomial filtering
    is used to remove low-degree polynomial trends within each subscan segment.

    Arguments
    ---------
    aman : AxisManager
    degree : int
        The degree of the polynomial to be removed.
    signal : array-like, optional
        The TOD signal to use. If not provided, `aman.signal` will be used.
    exclude_turnarounds : bool
        Optional. If True, turnarounds are excluded from subscan identification. Default is False.
    mask : str or RangesMatrix
        Optional. A mask used to select specific data points for filtering. Default is None.
        If None, no mask is applied. If the mask is given in str, ``aman.flags['mask']`` is used as mask.
        Arbitrary mask can be specified in the style of RangesMatrix.
    in_place: bool
        Optional. If True, `aman.signal` is overwritten with the processed signal.
    wrap: None or str
        Optional. Only used when in_place is False. If not None, the filtered TOD is wraped into aman[wrap].

    Returns
    -------
    signal : array-like
        The processed signal.
    """
    if signal is None:
        signal = aman.signal

    if not(in_place):
        signal = signal.copy()
        
    if exclude_turnarounds:
        if ("left_scan" not in aman.flags) or ("turnarounds" not in aman.flags):
            logger.warning('aman does not have left/right scan or turnarounds flag. `sotodlib.flags.get_turnaround_flags` will be ran with default parameters')
            _ = flags.get_turnaround_flags(aman)
        valid_scan = np.logical_and(np.logical_or(aman.flags["left_scan"].mask(), aman.flags["right_scan"].mask()),
                                    ~aman.flags["turnarounds"].mask())
        subscan_indices = _get_subscan_range_index(valid_scan)
    else:
        if ("left_scan" not in aman.flags):
            logger.warning('aman does not have left/right scan. `sotodlib.flags.get_turnaround_flags` will be ran with default parameters')
            _ = flags.get_turnaround_flags(aman)
        subscan_indices_l = _get_subscan_range_index(aman.flags["left_scan"].mask())
        subscan_indices_r = _get_subscan_range_index(aman.flags["right_scan"].mask())
        subscan_indices = np.vstack([subscan_indices_l, subscan_indices_r])
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
                signal[i_det, start:end+1] -= np.mean(signal[i_det, start:end+1])
            else:
                t_mean = np.mean(t[start:end+1])
                pars = np.ma.polyfit(
                        np.ma.array(t[start:end+1]-t_mean, mask=each_det_mask[start:end+1]),
                        np.ma.array(signal[i_det,start:end+1], mask=each_det_mask[start:end+1]),
                        deg=degree)

                signal[i_det,start:end+1] -= np.polyval(pars, t[start:end+1]-t_mean)
    if in_place:
        aman.signal = signal
    if wrap is not None:
        aman.wrap(wrap, signal, [(0, 'dets'), (1, 'samps')])
    return signal
                
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
 
