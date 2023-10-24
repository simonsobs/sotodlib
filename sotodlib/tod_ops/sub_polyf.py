import numpy as np 
import matplotlib.pyplot as plt 

from sotodlib import flags



def get_turnaround_flags_by_dazdt(
    tod, qlim=3.,az=None
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
    # to avoid oscillation of daz
    lo = mean_lo+2.*(mean_lo-np.min(daz[daz<lo]))
    
    
    m = np.logical_and(daz > lo, daz < hi)
    
    return m


def get_turnaround_flags( # Copy of flags.get_turnaround_flags, editted
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
 
    
def subscan_polyfilter_onlystat(aman, degree, default_flag=False, wrap_flag=True) : 
    
    ta_flag = get_turnaround_flags_by_dazdt(aman,3.)
    ta_list = get_ta_list(ta_flag)
    
    time = aman.timestamps
    time = time - time[0]
    new_mask = np.zeros_like(time,dtype=bool)
    
    for i_det in range(aman.signal.shape[0]) : 
        for ta in ta_list :
            start = ta[0]
            end = ta[1]
            
            t_mean = np.mean(time[start:end+1])
            
            pars = np.polyfit(time[start:end+1]-t_mean,aman.signal[i_det,start:end+1],deg=degree)
            f = np.poly1d(pars)
            aman.signal[i_det,start:end+1] = aman.signal[i_det,start:end+1] - f(time[start:end+1]-t_mean)
            new_mask[start:end+1] = True
        
    if wrap_flag : 
        print("non-turn-around part is wrapped.")
        aman.wrap('stationary_part', new_mask, [(0, 'samps')])

    
    
