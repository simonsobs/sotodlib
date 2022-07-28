import os
import numpy as np
import scipy.interpolate
from scipy.optimize import curve_fit
import so3g
from spt3g import core
from sotodlib import core, flags, tod_ops, sim_flags

def I_to_P_param(band=0, loading=10, inc_ang=5):
    """ 
    ** I to P leakage simulation function **
    - Mueller matirices of three layer achromatic HWP for each frequency and each incident angle are derived by RCWA simulation.
    - AR coating based on Mullite-Duroid coating for SAT1 at UCSD.　Thickness are used fabricated values.
    - 2f and 4f amplitudes and phases are extracted by HWPSS fitting of MIQ and MIU.
    - Amplitudes and phases are fitted by polynominal functions w.r.t each incident angle.
    Args ---
    band: MF1=0, MF2=1
    loading: intensity [Kcmb]
    inc_ang: incident angle [deg]
    Return ---
    amp_2f/4f: 2f/4f amplitude [mKcmb]
    phi_2f/4f: 2f/4f phase shift [rad]
    """
    if band == 0:
        poly_A2_ang = np.array([-7.01e-06, -6.87e-05,  3.435e-02])
        poly_A4_ang = np.array([1.61e-07, 1.20e-05, 3.42e-06, 1.23e-07])
        phi_2f, phi_4f = 66.46, 55.57 #[deg]
    elif band == 1:
        poly_A2_ang = np.array([-1.11e-05,  2.58e-05,  3.40e-02])    
        poly_A4_ang = np.array([4.38e-08, 8.90e-06, 1.85e-06, 1.18e-07])
        phi_2f, phi_4f = 71.62, 54.43 #[deg]
    amp_2f = loading * np.poly1d(poly_A2_ang)(inc_ang)
    amp_4f = loading * np.poly1d(poly_A4_ang)(inc_ang)
    amp_2f, amp_4f = amp_2f * 1e3, amp_4f * 1e3
    phi_2f, phi_4f = np.deg2rad(phi_2f), np.deg2rad(phi_4f)
    if inc_ang == 0: amp_4f, phi_4f = 0, 0
    return amp_2f, amp_4f, phi_2f, phi_4f

def sim_hwpss(aman, hwp_freq=2., band=0, loading=10., inc_ang=5.):
    """ 
    ** HWPSS simulation function **
    - 2f and 4f amp and phase are from I_to_P_param
    Args ---
    aman: AxisManager
    hwp_freq: 2 [Hz]
    band: MF1=0, MF2=1
    loading: intensity [Kcmb]
    inc_ang: incident angle [deg]
    Return ---
    Adding aman.hwpss
    """
    if 'hwpss' in aman.keys(): aman.move('hwpss','')
    aman.wrap_new( 'hwpss', ('dets', 'samps'))
    hwp_angle = np.mod(2*np.pi * aman.timestamps * hwp_freq, 2*np.pi)
    if not 'hwp_angle' in aman.keys(): aman.wrap('hwp_angle', hwp_angle, [(0,'samps')])
    else: aman.hwp_angle = hwp_angle
    amp_2f, amp_4f, phi_2f, phi_4f = I_to_P_param(inc_ang=15)
    amp = [0, 0, amp_2f, 0, amp_4f]
    amp_r = [1, 5, amp_2f*0.05, 1, amp_4f*0.05]
    phi = [0, 0, phi_2f, 0, phi_4f]
    phi_r = [0, np.deg2rad(1), phi_2f*0.1, np.deg2rad(1), phi_4f*0.1]
    nf = len(amp)
    for n in [2,4]:
        amp_rnd = np.random.normal(loc=amp[n], scale=amp_r[n], size=(aman.dets.count,))
        phase_rnd = np.mod(np.random.normal(loc=phi[n], scale=phi_r[n],size=(aman.dets.count,)), 2*np.pi)
        aman.hwpss += amp_rnd[:,None]*np.cos(n*aman.hwp_angle + phase_rnd[:,None])
                                                 
def func_hwpss_2f4f(x,A2,A4,phi2,phi4):
    return A2*np.cos(2*x+phi2) + A4*np.cos(4*x+phi4)
    
def func_hwpss_2f(x,A2,phi2):
    return A2*np.cos(2*x+phi2)
                                                 
def func_hwpss_4f(x,A4,phi4):
    return A4*np.cos(4*x+phi4)

def func_hwpss_all(x,A1,A2,A3,A4,A5,phi1,phi2,phi3,phi4,phi5):
    return A1*np.cos(x+phi1) + A2*np.cos(2*x+phi2) + A3*np.cos(3*x+phi3) + A4*np.cos(4*x+phi4) + A5*np.cos(5*x+phi5)

def ext_hwpss(aman, hwp_angle=None, mode=0, name = 'ext_hwpss', signal='signal'):    
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
    filt = tod_ops.filters.high_pass_sine2(cutoff=0.1)
    signal_prefilt = np.array(tod_ops.fourier_filter(aman, filt, detrend=None, signal_name=signal))
    
    bins=361
    hwpss_denom = np.histogram(aman.hwp_angle, bins=bins, range=[0, 2*np.pi])[0]
    hwp_angle_bins = np.linspace(0,2*np.pi,bins)
    hwpss=[]
    for i in range(aman.dets.count):
        hwpss.append(np.histogram(aman.hwp_angle, bins=bins, range=[0, 2*np.pi], weights=signal_prefilt[i])[0] / np.where(hwpss_denom==0, 1, hwpss_denom))
        idx = np.argwhere(hwpss[i]!=0).flatten()
        if i == 0: hwp_angle_bins = hwp_angle_bins[idx]
        hwpss[i] = hwpss[i][idx]
    
    if 'ext_hwpss' in aman.keys(): aman.move('ext_hwpss','')
    aman.wrap_new( 'ext_hwpss', ('dets', 'samps'))
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