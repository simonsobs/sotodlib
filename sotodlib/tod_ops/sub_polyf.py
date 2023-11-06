import numpy as np 

from sotodlib import flags


def subscan_polyfilter(aman, degree, signal='signal', subscan_def=None, mask=None) : 
    
    """
    
    Args:
        aman: AxisManager object
        degree: Degree of polynomial
        signal: the name of the signal to filtered. defaults is 'signal'.
        subscan_def: the name of Ranges object with which subscan ranges 
                     are defined. (False ranges will be filtered)
                     if None, left/right scan flag will be used.
        mask: Ranges/RangesMatrix object which will be 
                    rejected from polyfiltering ranges.
    Returns:
        -
    """
    
    # Case with right/left scan flag 
    if subscan_def == None : 
        # Check if aman has left/right_flag 
        if  ("left_scan" in aman.flags) and ("right_scan" in aman.flags) :
            pass
        else :
            flag.get_turnaround_flags(aman, merge=False)
            
        ta_list_l = _get_subscan_range_index(aman.flags["left_scan"].mask())
        ta_list_r = _get_subscan_range_index(aman.flags["right_scan"].mask())
        
        # First part can be overlapped. Delete overlapped part.
        if ta_list_l[0][0] == 0 and ta_list_r[0][0] == 0 :
            if ta_list_l[0][1] > ta_list_r[0][1] : 
                del ta_list_r[0]
            else : 
                del ta_list_l[0]
        
        ta_list_l.extend(ta_list_r)
        ta_list = ta_list_l
            
    # Case with given definition 
    else : 
        ta_flag = subscan_def 
        ta_list = _get_subscan_range_index(ta_flag.mask())

    time = aman.timestamps
    time = time - time[0]
    
    # Get boolean array. 
    is_matrix = False
    if not mask is None : 
        masked_flag = mask.mask()
        if len(masked_flag.shape) > 1 : is_matrix = True
        else : masked_flag = masked_flag.reshape([1,len(masked_flag)])
        
        assert aman.signal.shape == masked_flag.shape, "Mask shape is not consistent with signal shape."
        
    else : 
        masked_flag = np.zeros([1,len(time)],dtype=bool)
    
    for i_det in range(aman[signal].shape[0]) : 
        
        # In case Ranges object is given, re-shape and read data of index 0.
        # If RangesMatrix object is given, read data of index i_det
        if is_matrix == False : 
            read = 0
        else : 
            read = i_det
        
        for ta in ta_list :
            start = ta[0]
            end = ta[1]
            
            if np.any(masked_flag[read, start:end+1]) : 
                if np.all(masked_flag[read, start:end+1] == True): continue
                    
                t_mean = np.mean(time[start:end+1])
                pars,_,_,_,_ = np.ma.polyfit(
                    np.ma.array(time[start:end+1]-t_mean, mask=masked_flag[read, start:end+1]),
                    np.ma.array(aman[signal][i_det,start:end+1], mask=masked_flag[read, start:end+1]),
                    deg=degree,
                    full=True,
                )         
                
                aman.signal[i_det,start:end+1] = aman.signal[i_det,start:end+1] - np.polyval(pars, time[start:end+1]-t_mean)
                
            else : 
                t_mean = np.mean(time[start:end+1])
                pars = np.polyfit(time[start:end+1]-t_mean,aman[signal][i_det,start:end+1],deg=degree)
                f = np.poly1d(pars)        
                
                aman.signal[i_det,start:end+1] = aman[signal][i_det,start:end+1] - f(time[start:end+1]-t_mean) 
                


def _get_subscan_range_index(ta_flag,_min=0):
    
    # starts and ends of st. parts
    ones = np.where(ta_flag==False)[0]
    diff = np.diff(ones)
    
    starts = np.insert(ones[np.where(diff != 1)[0]+1], 0, ones[0])
    ends = np.append(ones[np.where(diff != 1)[0] ], ones[-1])
    indices = list(zip(starts, ends))
    #indices = [(start, end) for start, end in indices if (end - start + 1 >= _min)]
    indices = [(start, end) for start, end in indices if (end - start + 1 >= _min)]

    return indices 
 
    

