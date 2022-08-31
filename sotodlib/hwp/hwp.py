import scipy.interpolate
from scipy.optimize import curve_fit
from spt3g import core
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)

def func_hwpss_2f4f(x,A2,A4,phi2,phi4):
    return A2*np.cos(2*x+phi2) + A4*np.cos(4*x+phi4)
    
def func_hwpss_2f(x,A2,phi2):
    return A2*np.cos(2*x+phi2)
                                                 
def func_hwpss_4f(x,A4,phi4):
    return A4*np.cos(4*x+phi4)

def func_hwpss_all(x,A1,A2,A3,A4,A5,phi1,phi2,phi3,phi4,phi5):
    return A1*np.cos(x+phi1) + A2*np.cos(2*x+phi2) + A3*np.cos(3*x+phi3) + A4*np.cos(4*x+phi4) + A5*np.cos(5*x+phi5)

def subtract_hwpss(aman, hwp_angle=None, mode=0, name = 'subtract_hwpss', signal='signal', prefilt_cutoff=0.1, bins=361):    
    """ 
    ** HWPSS extraction function **
    Args ---
    aman: AxisManager
    mode: 0 = 2f&4f, 1 = 2f only, 2 = 4f only, 3 = all up to 5f
    name: aman output name
    signal: target timestream data
    Return ---
    Adding aman.hwpss
    """
    filt = tod_ops.filters.high_pass_sine2(cutoff=prefilt_cutoff)
    signal_prefilt = np.array(tod_ops.fourier_filter(aman, filt, detrend=None, signal_name=signal))
        
    hwpss_denom = np.histogram(aman.hwp_angle, bins=bins, range=[0, 2*np.pi])[0]
    hwp_angle_bins = np.linspace(0,2*np.pi,bins)
    hwpss=[]
    for i in range(aman.dets.count):
        hwpss.append(np.histogram(aman.hwp_angle, bins=bins, range=[0, 2*np.pi], weights=signal_prefilt[i])[0] / np.where(hwpss_denom==0, 1, hwpss_denom))
        idx = np.argwhere(hwpss[i]!=0).flatten()
        if i == 0: hwp_angle_bins = hwp_angle_bins[idx]
        hwpss[i] = hwpss[i][idx]
    
    if 'subtract_hwpss' in aman.keys(): aman.move('subtract_hwpss','')
    aman.wrap_new( 'subtract_hwpss', ('dets', 'samps'))
    if mode == 0: func = func_hwpss_2f4f
    elif mode == 1: func = func_hwpss_2f
    elif mode == 2: func = func_hwpss_4f
    elif mode == 3: func = func_hwpss_all
    for i in range(aman.dets.count):
        popt, pcov = curve_fit(func, hwp_angle_bins, hwpss[i], maxfev=1000000)
        aman.ext_hwpss[i] += func(aman.hwp_angle, *popt)


def demod(aman, hwp_angle=None, bpf_width=0.5, lpf_cut=0.5, signal='signal'):
    """ 
    ** Simple demoduation function **
    Args ---
    aman: AxisManager
    hwp_angle: HWP reconstructed angle, do not need to input if aman already has this
    bpf_width: Width of pre-bandpass filter
    lpf_cut: cut off of low pass filter after applying demod. factor
    Return ---
    Adding aman.signal_demod
    """
    if hwp_angle is not None: 
        if 'hwp_angle' in aman.keys(): aman.move('hwp_angle', '') 
        aman.wrap('hwp_angle', hwp_angle)
    if 'signal_demod_prelfilt' in aman.keys(): aman.move('signal_demod_prelfilt','')
    if 'signal_demod_prebfilt' in aman.keys(): aman.move('signal_demod_prebfilt','')
    if 'signal_demod_wo_lfilt' in aman.keys(): aman.move('signal_demod_wo_lfilt','')
    if 'signal_demod' in aman.keys():aman.move('signal_demod','')

    speed = (np.sum(np.diff(aman.hwp_angle) % (2*np.pi)) / (aman.timestamps[-1] - aman.timestamps[0])) / (2*np.pi)
    prelfilt = tod_ops.filters.low_pass_butter4(fc=4*speed+bpf_width)
    aman.wrap('signal_demod_prelfilt', tod_ops.fourier_filter(aman, prelfilt, detrend=None, signal_name=signal), [(0,'dets'), (1,'samps')] ) 
    prehfilt = tod_ops.filters.high_pass_butter4(fc=4*speed-bpf_width)
    aman.wrap('signal_demod_prebfilt', tod_ops.fourier_filter(aman, prehfilt, detrend=None, signal_name='signal_demod_prelfilt'), [(0,'dets'), (1,'samps')] ) 

    aman.wrap('signal_demod_wo_lfilt', (aman.signal_demod_prebfilt * np.exp(-1j*(4*aman.hwp_angle))).real, [(0,'dets'), (1,'samps')] ) 
    lfilt = tod_ops.filters.low_pass_butter4(fc=lpf_cut)
    aman.wrap('signal_demod', tod_ops.fourier_filter(aman, lfilt, detrend=None, signal_name='signal_demod_wo_lfilt'), [(0,'dets'), (1,'samps')] ) 
    
    aman.move('signal_demod_prelfilt','')
    aman.move('signal_demod_prebfilt','')
    aman.move('signal_demod_wo_lfilt','')