import numpy as np 

from sotodlib import flags



def get_leftright_array(
    tod, az=None
):
    if az is None:
        az = tod.boresight.az
    
    daz = np.copy(np.diff(tod.boresight.az))
    daz = np.append(daz,daz[-1])
    signs = np.sign(daz)
    
    # positive range
    positive_indices = np.where(signs > 0)[0]
    ranges = np.split(positive_indices, np.where(np.diff(positive_indices) != 1)[0]+1)
    index_ranges_p = [(rng[0], rng[-1]) for rng in ranges]
    
    # negative range
    negative_indices = np.where(signs <= 0)[0]
    ranges = np.split(negative_indices, np.where(np.diff(negative_indices) != 1)[0]+1)
    index_ranges_n = [(rng[0], rng[-1]) for rng in ranges]
    

    index_ranges_p.extend(index_ranges_n)
    
    return index_ranges_p



def get_range_list(ta_flag,_min=0):
    
    # starts and ends of non-turn-around parts
    ones = np.where(ta_flag==False)[0]
    diff = np.diff(ones)
    
    starts = np.insert(ones[np.where(diff != 1)[0]], 0, ones[0])
    ends = np.append(ones[np.where(diff != 1)[0] - 1], ones[-1])
    indices = list(zip(starts, ends))
    indices = [(start, end) for start, end in indices if (end - start + 1 >= _min)]

    return indices


def subscan_polyfilter_tamasked(aman, degree, signal='signal', default_flag=False, wrap_flag=True) : 
    
    """
    subscan polyfilter for only non turn-around part.
    
    Args:
        aman: AxisManager object
        degree: Degree of polynomial
        default_flag: If True, turnaround part is looked for using original method.
                        Otherwise, turnaround part is looked for by looking at daz/dt

    Returns:
        -
    """
    
    if default_flag : ta_flag = flags.get_turnaround_flags(aman,merge=False)
    else : ta_flag = flags.get_turnaround_flags_by_dazdt(aman, merge=False)
    ta_list = get_range_list(ta_flag.mask())
    
    time = aman.timestamps
    time = time - time[0]
    
    for i_det in range(aman[signal].shape[0]) : 
        for ta in ta_list :
            start = ta[0]
            end = ta[1]
            
            t_mean = np.mean(time[start:end+1])
            
            pars = np.polyfit(time[start:end+1]-t_mean,aman[signal][i_det,start:end+1],deg=degree)
            f = np.poly1d(pars)
            
            aman.signal[i_det,start:end+1] = aman[signal][i_det,start:end+1] - f(time[start:end+1]-t_mean)

        

def subscan_polyfilter(aman, degree, signal='signal') :
    
    """
    subscan polyfilter
    In this function, subscan is defined by the sign of scan speed. 
    
    Args:
        aman: AxisManager object
        degree: Degree of polynomial
        signal: Which tod is used

    Returns:
        -
    """   
    
    time = aman.timestamps
    time = time - time[0]
    
    ranges_list = flags.get_leftright_array(aman)
    
    for i_det in range(aman[signal].shape[0]) :  
        for subscan in ranges_list :
            start = subscan[0]
            end = subscan[1]

            ### if the number of points is small, just subtract the mean
            if end - start < degree : 
                aman[signal][i_det,start:end+1] = aman[signal][i_det,start:end+1] - np.mean(aman[signal][i_det,start:end+1])
                continue

            ### mean of time is subtracted to make fit easier, 
            ### then polynomial fit is applied.
            t_mean = np.mean(time[start:end+1])
            pars = np.polyfit(time[start:end+1]-t_mean,aman[signal][i_det,start:end+1],deg=degree)
            f = np.poly1d(pars)
        
            aman[signal][i_det,start:end+1] = aman[signal][i_det,start:end+1] - f(time[start:end+1]-t_mean)

    
