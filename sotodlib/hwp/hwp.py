import numpy as np
from scipy.optimize import curve_fit
from spt3g import core
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)


def func_hwpss_2f4f(x, A2, A4, phi2, phi4):
    return A2 * np.cos(2 * x + phi2) + A4 * np.cos(4 * x + phi4)


def func_hwpss_2f(x, A2, phi2):
    return A2 * np.cos(2 * x + phi2)


def func_hwpss_4f(x, A4, phi4):
    return A4 * np.cos(4 * x + phi4)


def func_hwpss_all(x, A1, A2, A3, A4, A5, phi1, phi2, phi3, phi4, phi5):
    return A1*np.cos(x+phi1) + A2*np.cos(2*x+phi2) + \
            A3*np.cos(3*x+phi3) + A4*np.cos(4*x+phi4) + A5*np.cos(5*x+phi5)


def extract_hwpss(aman, signal='signal', name='hwpss_ext',
                  hwp_angle=None, mode=0, 
                  prefilt_cutoff=0.1, bins=3601):
    """
    * Extract HWP synchronous signal from given tod (AxisManager)
    * Estimated HWPSS will be added to AxisManager as aman['name']

    Args
    -----
        aman: AxisManager
            AxisManager including target timestream data
        signal: str
            name of target timestream data in AxisManager
        name: str
            output timestram name into AxisManager
        mode: int 
            0 = 2f&4f, 1 = 2f only, 2 = 4f only, 3 = all up to 5f
        prefit_cutoff: float
            cut off value of pre-lowpass filter
        bins: int
            number of bins for hwpss histogram
    """
    
    filt = tod_ops.filters.high_pass_sine2(cutoff=prefilt_cutoff)
    signal_prefilt = np.array(tod_ops.fourier_filter(
        aman, filt, detrend=None, signal_name=signal))

    hwpss_denom = np.histogram(aman.hwp_angle, bins=bins, range=[0, 2 * np.pi])[0]
    hwp_angle_bins = np.linspace(0, 2 * np.pi, bins)
    hwpss = []
    for i in range(aman.dets.count):
        hwpss.append(np.histogram(aman.hwp_angle,bins=bins,range=[0,2*np.pi],
                     weights=signal_prefilt[i])[0]/np.where(hwpss_denom==0, 1, hwpss_denom))
        idx = np.argwhere(hwpss[i] != 0).flatten()
        if i == 0:
            hwp_angle_bins = hwp_angle_bins[idx]
        hwpss[i] = hwpss[i][idx]

    if name in aman.keys():
        aman.move(name, None)
    aman.wrap_new(name, ('dets', 'samps'))
    if mode == 0:
        func = func_hwpss_2f4f
    elif mode == 1:
        func = func_hwpss_2f
    elif mode == 2:
        func = func_hwpss_4f
    elif mode == 3:
        func = func_hwpss_all
    for i in range(aman.dets.count):
        popt, pcov = curve_fit(func, hwp_angle_bins, hwpss[i], maxfev=1000000)
        aman[name][i] = func(aman.hwp_angle, *popt)


def demod(aman, signal='signal', name='signal_demod',
          hwp_angle=None, bpf_center=None, bpf_width=0.5, lpf_cut=0.5):
    """
    * Simple demoduation function for AxisManager
    * Demod timstream will be added to AxisManager as aman['name']
    
    Args
    -----
    aman: AxisManager
        target AxisManager
    signal: str
        name of target timestream
    name: str
        output timestram name into AxisManager
    hwp_angle: list or None
        HWP reconstructed angle, 
        do not need to input if aman already has this
    bpf_center: float or None
        Center frequency of pre-bandpass filter
        If not specified, it becomes 4*f_HWP
    bpf_width: float
        Width of pre-bandpass filter
    lpf_cut: float
        cut off of low pass filter after applying demod. factor
    """
    if hwp_angle is not None:
        if 'hwp_angle' in aman.keys():
            aman.move('hwp_angle', None)
        aman.wrap('hwp_angle', hwp_angle)
    if name + '_prelfilt' in aman.keys():
        aman.move(name + '_prelfilt', None)
    if name + '_prebfilt' in aman.keys():
        aman.move(name + '_prebfilt', None)
    if name + '_wo_lfilt' in aman.keys():
        aman.move(name + '_wo_lfilt', None)
    if name in aman.keys():
        aman.move(name, None)
    
    if bpf_center is None:
        speed = (np.sum(np.abs((np.diff(aman.hwp_angle) + np.pi)%(2*np.pi) - np.pi)) / 
                 (aman.timestamps[-1] - aman.timestamps[0]))/ (2 * np.pi)
        bpf_center = 4 * speed
        
    prelfilt = tod_ops.filters.low_pass_butter4(fc=bpf_center + bpf_width)
    aman.wrap(name + '_prelfilt', 
              tod_ops.fourier_filter(aman, prelfilt, 
                                     detrend=None, 
                                     signal_name=signal), 
              [(0, 'dets'), (1, 'samps')])
    prehfilt = tod_ops.filters.high_pass_butter4(fc=bpf_center - bpf_width)
    aman.wrap(name + '_prebfilt', 
              tod_ops.fourier_filter(aman, prehfilt, 
                                     detrend=None, 
                                     signal_name=name + '_prelfilt'), 
              [(0, 'dets'), (1, 'samps')])

    aman.wrap(name + '_wo_lfilt', 
              (aman[name + '_prebfilt'] * np.exp(-1j * (4 * aman.hwp_angle))).real, 
              [(0, 'dets'), (1, 'samps')])
    lfilt = tod_ops.filters.low_pass_butter4(fc=lpf_cut)
    aman.wrap(name,
              tod_ops.fourier_filter(aman, lfilt, 
                                     detrend=None, 
                                     signal_name=name+'_wo_lfilt'), 
              [(0, 'dets'), (1, 'samps')])

    aman.move(name + '_prelfilt', None)
    aman.move(name + '_prebfilt', None)
    aman.move(name + '_wo_lfilt', None)
