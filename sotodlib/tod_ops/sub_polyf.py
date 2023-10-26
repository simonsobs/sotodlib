import numpy as np 
import matplotlib.pyplot as plt 

from sotodlib import flags



def get_turnaround_flags_by_dazdt(
    tod, qlim=5.,az=None
):

    """
    Args:
        tod: AxisManager object
        qlimp: percentage threshold applied to positive daz/dt
        qlimn: percentage threshold applied to negative daz/dt
        az: The azimuth signal to look for turnarounds. If None it defaults to
            tod.boresight.az

    Returns:
        flag: boolean array of turn-arounds
    """
    
    daz = np.copy(np.diff(tod.boresight.az))
    daz = np.append(daz,daz[-1])
    # rough selection of stationary part
    lo,hi = np.min(daz)*(100-5)*0.01,np.max(daz)*(100-5)*0.01
    
    mean_hi = np.mean(daz[daz>hi])
    mean_lo = np.mean(daz[daz<lo])
    
    hi = mean_hi*(100.-qlim)*0.01
    lo = mean_lo*(100.-qlim)*0.01
    # to avoid oscillation of daz
    #lo = mean_lo+2.*(mean_lo-np.min(daz[daz<lo]))
    
    
    m = np.logical_and(daz > lo, daz < hi)
    
    return m


# Copy of flags.get_turnaround_flags, editted to return boolean numpy array
def get_turnaround_flags( # Copy of flags.get_turnaround_flags, editted to return boolean numpy array
    tod, qlim=1, az=None
):
    """Flag the scan turnaround times.

    Args:
        tod: AxisManager object
        qlim: percentile used to find turnaround
        az: The azimuth signal to look for turnarounds. If None it defaults to
            tod.boresight.az

    Returns:
        flag: Ranges object of turn-arounds
    """
    
    if az is None:
        az = tod.boresight.az
    lo, hi = np.percentile(az, [qlim, 100 - qlim])
    m = np.logical_or(az < lo, az > hi)

    return m


def get_ta_list(ta_flag):
    
    # starts and ends of st. parts
    num1,num2 = 5000,7000
    ones = np.where(ta_flag==False)[0]
    diff = np.diff(ones)
    
    starts = np.insert(ones[np.where(diff != 1)[0]], 0, ones[0])
    ends = np.append(ones[np.where(diff != 1)[0] - 1], ones[-1])
    indices = list(zip(starts, ends))
    indices = [(start, end) for start, end in indices if (end - start + 1 >= num1) and (end - start + 1 <= num2)]

    return indices 



# subscan ranges are obtained based on the sign of az speed 
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
 
    
def subscan_polyfilter_onlystat(aman, degree, signal='signal', default_flag=False, wrap_flag=True) : 

    """
    subscan polyfilter for only stationary part.
    
    Args:
        aman: AxisManager object
        degree: Degree of polynomial
        default_flag: If True, turnaround part is looked for using 'get_turnaround_flags()'
                        Otherwise, turnaround part is looked for by looking at daz/dt (get_turnaround_flags_by_dazdt())
        wrap_flag: This function can make the jumps in tod at each turnaround part.
                        If wrap_flag is True, a mask that veto turnaround part will be wrapped in aman. 

    Returns:
        -
    """
   
    if default_flag : ta_flag = get_turnaround_flags(aman,qlim=5)
    else : ta_flag = get_turnaround_flags_by_dazdt(aman,qlim=5)
    ta_list = get_ta_list(ta_flag)
    
    time = aman.timestamps
    time = time - time[0]
    new_mask = np.zeros_like(time,dtype=bool)
    
    for i_det in range(aman[signal]shape[0]) : 
        for ta in ta_list :
            start = ta[0]
            end = ta[1]
            
            t_mean = np.mean(time[start:end+1])
            
            pars = np.polyfit(time[start:end+1]-t_mean,aman[signal][i_det,start:end+1],deg=degree)
            f = np.poly1d(pars)
            aman[signal][i_det,start:end+1] = aman[signal][i_det,start:end+1] - f(time[start:end+1]-t_mean)
            new_mask[start:end+1] = True
        ]
    if wrap_flag : 
        print("non-turn-around part is wrapped.")
        aman.wrap('stationary_part', new_mask, [(0, 'samps')])

    

def subscan_polyfilter(aman, degree, signal='signal') :

    """
    subscan polyfilter
    In this function, subscan is defined looking at the sign of scan speed.

    Args:
        aman: AxisManager object
        degree: Degree of polynomial
        signal: Which tod is used

    Returns:
        -
    """

    ### Get left scan ranges and right scan ranges
    subscans = get_leftright_array(aman)

    time = aman.timestamps
    time = time - time[0]

    for i_det in range(aman[signal].shape[0]) :
        for subscan in subscans :
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

