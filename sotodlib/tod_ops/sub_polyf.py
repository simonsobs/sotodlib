import numpy as np
import logging
from scipy.special import eval_legendre
from . import flags
logger = logging.getLogger(__name__)

def subscan_polyfilter(aman, degree, signal=None, exclude_turnarounds=False, mask=None, method="legendre", in_place=True):
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
    method : str
        Optioal. Method to model the baseline of TOD.
        In `legendre` method, baseline model is constructed using orthonormality of Legendre function.
        In `polyfit` method, numpy.polyfit is used.
        `legendre` is faster. Default is `legendre`.
    in_place: bool
        Optional. If True, `aman.signal` is overwritten with the processed signal.

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
            _ = flags.get_turnaround_flags(aman,truncate=True)
        valid_scan = np.logical_and(np.logical_or(aman.flags["left_scan"].mask(), aman.flags["right_scan"].mask()),
                                    ~aman.flags["turnarounds"].mask())
        subscan_indices = _get_subscan_range_index(valid_scan)
    else:
        if ("left_scan" not in aman.flags):
            logger.warning('aman does not have left/right scan. `sotodlib.flags.get_turnaround_flags` will be ran with default parameters')
            _ = flags.get_turnaround_flags(aman,truncate=True)
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


    if method not in ["polyfit","legendre"] :
        raise ValueError("Only polyfit and legendre are acceptable.")

        
    elif method == "polyfit":
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

                    
    elif method == "legendre":

        ### we calculate model in for loop, so we need additional +1 .
        degree_corr = degree + 1


        time = np.copy(aman["timestamps"])

        for subscan in subscan_indices:

            ### Normalization constant of legendre function 
            norm_vector = np.arange(degree_corr)
            norm_vector = 2./(2.*norm_vector+1)
            
            # Get each subscan to be filtered & subtract mean
            tod_mat = signal[:,subscan[0]:subscan[1]+1]
            means = np.mean(tod_mat,axis=1)[:,np.newaxis]
            tod_mat -= means

            # Scale time range into [-1,1]
            x = np.linspace(-1, 1, tod_mat.shape[1])
            dx = np.mean(np.diff(x))
            sub_time = time[subscan[0]:subscan[1]+1]

            # Generate legendre functions of each degree and store them in an array
            arr_legendre = []
            for deg in range(degree_corr) :
                each_legendre = eval_legendre(deg, x)
                arr_legendre.append(each_legendre)
            arr_legendre = np.array(arr_legendre)

            # flag to know if the result is matrix formation
            flag_matrix = False

            # Modify TODs if mask is defined
            if mask is None :
                pass
            else :
                # if mask is matrix like, we should interpolate TOD det by det.
                if is_matrix :
                    # Is maksed range overlapped with this subscan?
                    if np.sum((mask_array[:,subscan[0]:subscan[1]+1]).astype(np.int32)) > 0 :
                        
                        msk_indx = mask_array[:,subscan[0]:subscan[1]+1]
                        
                        for idet in range(tod_mat.shape[0]) : 
                            n_intep =  np.sum((mask_array[idet,subscan[0]:subscan[1]+1]).astype(np.int32))
                            if n_intep > 0:
                                if n_intep == tod_mat.shape[1] : continue
                                interped = np.interp(np.flatnonzero(msk_indx[idet]),np.flatnonzero(~msk_indx[idet]), tod_mat[idet][~msk_indx[idet]])
                                tod_mat[idet,msk_indx[idet]] = interped
                            
                    # If mask does not affect this range, just go through.
                    else :
                        pass
                # If mask is array like, same ranges of each det will be interpolated.
                else :
                    n_intep =  np.sum((mask_array[subscan[0]:subscan[1]+1]).astype(np.int32))
                    if n_intep > 0 :
                        
                        if n_intep == tod_mat.shape[1] : continue
                        msk_indx = mask_array[subscan[0]:subscan[1]+1]
                        
                        for idet in range(tod_mat.shape[0]) : 
                            interped = np.interp(np.flatnonzero(msk_indx),np.flatnonzero(~msk_indx), tod_mat[idet][~msk_indx])
                            tod_mat[idet,msk_indx] = interped
                    else :
                        pass
            
            # Make model to be subtracted
            coeffs = np.dot(arr_legendre, tod_mat.T)
            model = np.dot((coeffs/norm_vector[:, np.newaxis]).T,arr_legendre)*dx

            # We subtracted the mean at the start of this script. This should be corrected.
            model += means
            tod_mat += means

            signal[:,subscan[0]:subscan[1]+1] = tod_mat
            signal[:,subscan[0]:subscan[1]+1] -= model

    if in_place:
        aman.signal = signal

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
 
