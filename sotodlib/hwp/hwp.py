import numpy as np
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)

def extract_hwpss(aman, signal=None, hwp_angle=None,
                  bin_signal=True, bins=360, 
                  lin_reg=True, modes=[2, 4, 6, 8],
                  apply_prefilt=True, prefilt_cutoff=0.5,
                  mask_flags=True, add_to_aman=True, name='hwpss_extract'):
    
    if signal is None:
        if apply_prefilt: 
            filt = tod_ops.filters.high_pass_sine2(cutoff=prefilt_cutoff)
            signal = np.array(tod_ops.fourier_filter(aman, filt, detrend='linear', signal_name='signal'))
        else:
            signal = aman.signal
            
    if hwp_angle is None:
        hwp_angle = aman.hwp_angle
            
    if bin_signal:
        x, ys, yerrs = binning_signal(aman, signal, hwp_angle=None, bins=bins, mask_flags=mask_flags)
        if lin_reg:
            fitsig_binned, coeffs, covars, redchi2s = hwpss_linreg(x, ys, yerrs, modes)
        else:
            Params_init = guess_hwpss_params(x, ys, modes)
            fitsig_binned, coeffs, covars, redchi2s = hwpss_curvefit(x, ys, yerrs, modes, Params_init=Params_init)
            
        # tod template
        fitsig_tod = harms_func(hwp_angle, modes, coeffs)
        
    else:
        if mask_flags:
            m = ~aman.flags.glitches.mask()
        else:
            m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
        x = hwp_angle
        ys = signal
        yerrs = np.std(signal, axis=-1) #!!!!!! FIX ME !!!!!!!!
        
        if lin_reg:
            fitsig_tod, coeffs, covars, redchi2s = hwpss_linreg(x, ys, yerrs, modes)
            
        else:
            raise ValueError('Curve-fitting for TOD are specified.' + \
                             'It will take too long time and return meaningless result.' + \
                             'Specify (bin_signal, lin_reg) = (True, True) or (True, False) or (False, True)')
    
    return fitsig_tod, coeffs, covars, redchi2s

def binning_signal(aman, signal=None, hwp_angle=None,
                   bins=360, mask_flags=False):
    """
    Bin time-ordered data by the HWP angle and return the binned signal and its standard deviation.

    Parameters
    ----------
    aman : TOD
        The Axismanager object to be binned.
    signal : str, optional
        The name of the signal to be binned. Defaults to aman.signal if not specified.
    hwp_angle : str, optional
        The name of the timestream of hwp_angle. Defaults to aman.hwp_angle if not specified.
    bins : int, optional
        The number of HWP angle bins to use. Default is 360.
    mask_flags : bool, optional
        Flag indicating whether to exclude flagged samples when binning the signal. Default is False.

    Returns
    ---
    aman_proc:
        The AxisManager object which contains
        * center of each bin of hwp_angle
        * binned hwp synchrounous signal
        * estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if hwp_angle is None:
        hwp_angle = aman.hwp_angle
    
    # binning hwp_angle tod
    hwpss_denom, hwp_angle_bins = np.histogram(hwp_angle, bins=bins, range=[0, 2 * np.pi])
    
    # convert bin edges into bin centers
    hwp_angle_bin_centers = (hwp_angle_bins[1]-hwp_angle_bins[0])/2 + hwp_angle_bins[:-1]
    
    # prepare binned signals
    binned_hwpss = np.zeros((aman.dets.count, bins), dtype='float32')
    binned_hwpss_squared_mean = np.zeros((aman.dets.count, bins), dtype='float32')
    binned_hwpss_sigma = np.zeros((aman.dets.count, bins), dtype='float32')
    
    # get mask from aman
    if mask_flags:
        m = ~aman.flags.glitches.mask()
    else:
        m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
        
    # binning tod
    for i in range(aman.dets.count):
        binned_hwpss[i][:] = np.histogram(hwp_angle[m[i]], bins=bins, range=[0,2*np.pi],
                               weights=signal[i][m[i]])[0] / np.where(hwpss_denom==0, 1, hwpss_denom)
    
        binned_hwpss_squared_mean[i][:] = np.histogram(hwp_angle[m[i]], bins=bins, range=[0,2*np.pi],
                                   weights=signal[i][m[i]]**2)[0] / np.where(hwpss_denom==0, 1, hwpss_denom)
    
    # get sigma of each bin
    binned_hwpss_sigma = np.sqrt( np.abs(binned_hwpss_squared_mean - binned_hwpss**2)) / np.sqrt(np.where(hwpss_denom==0, 1, hwpss_denom))
    # use median of sigma of each bin as uniform sigma for a detector
    hwpss_sigma = np.median(binned_hwpss_sigma, axis=-1)
    
    return hwp_angle_bin_centers, binned_hwpss, hwpss_sigma
    
def hwpss_linreg(x, ys, yerrs, modes):
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    I = np.linalg.inv(np.tensordot(vects, vects, (1,1)))
    coeffs = np.matmul(ys, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    fitsig = np.matmul(vects.T, coeffs.T).T
    
    # covariance of coefficients
    covars = np.zeros((ys.shape[0], 2*len(modes), 2*len(modes)))    
    for det_idx in range(ys.shape[0]):
        covars[det_idx, :, :] = I * yerrs[det_idx]
    
    # reduced chi-square
    redchi2s = np.sum(((ys - fitsig)/yerrs[:, np.newaxis])**2, axis=-1) / (x.shape[0] - 2*len(modes))
    
    return fitsig, coeffs, covars, redchi2s


def wrapper_harms_func(x, modes, *args):
    coeffs = np.array(args[0])
    return harms_func(x, modes, coeffs) 

def harms_func(x, modes, coeffs):
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)
        
    harmonics = np.matmul(vects.T, coeffs.T).T
    return harmonics

def guess_hwpss_params(x, ys, modes):
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)
    Params_init = 2 * np.matmul(ys,vects.T) / x.shape[0]
    return Params_init

def hwpss_curvefit(x, ys, yerrs, modes, Params_init=None):
    N_dets = ys.shape[0]
    N_samps = ys.shape[-1]
    N_modes = len(modes)
    
    if Params_init is None:
        Params_init = np.zeros((N_dets, 2*N_modes))
    
    coeffs = np.zeros((N_dets, 2*len(modes)))
    covars = np.zeros((N_dets, 2*len(modes), 2*len(modes)))
    redchi2s = np.zeros(N_dets)
    fitsig = np.zeros((N_dets, N_samps))
    
    for det_idx in range(N_dets):
        params_init = Params_init[det_idx]
        coeff, covar = curve_fit(lambda x, *params_init: wrapper_harms_func(x, modes, params_init),
                               x, ys[det_idx], p0=params_init, sigma=yerrs[det_idx] * np.ones_like(ys[det_idx]), 
                               absolute_sigma=True)
        
        coeffs[det_idx, :] = coeff
        covars[det_idx, :] = covar
        
        yfit = harms_func(x, modes, coeff)
        fitsig[det_idx, :] = yfit
        redchi2s[det_idx] = np.sum( ((ys[det_idx] - yfit) / yerrs[det_idx])**2 ) / (x.shape[0] - 2*len(modes))
        
    return fitsig, coeffs, covars, redchi2s

def subtract_hwpss(aman, signal=None, hwpss_template=None,
                   subtract_name='hwpss_remove'):
    """
    Subtract the hwpss template from the signal in an axis manager.
    """
    if signal is None:
        signal = aman.signal
    if hwpss_template is None:
        hwpss_template = aman.hwpss_ext

    aman.wrap(subtract_name, np.subtract(signal, hwpss_template), [(0,'dets'), (1,'samps')])


def demod_tod(aman, signal='hwpss_remove', 
              fc_lpf=2., lpf_sin2_width=0.5, 
              bpf=False, bpf_width=2., bpf_center=None, 
              bpf_type='sine2', bpf_sine2_width=None):
    """
    Simple demodulation function. Wraps new axes demodQ (real part from hwp
    demodulation), demodU (imag part from hwp demodulation), and dsT (lowpass
    filtered version of raw signal) into input axis manager ``aman``.
    Args
    ----
    aman (AxisManager): Axis manager to perform fit.
    signal (str): Axis to demodulate
    fc_lpf (float): low pass filter cutoff
    lpf_sin2_width (float): width of sine^2 low pass filter
    bpf (bool): Apply bandpass filter before demodulation.
    bpf_width (float): Width of bandpass filter in Hz.
    bpf_center (float): Center of bandpass filter, if not passed will
                        estimate 4*f_HWP from hwp_angles in aman.
    bpf_type (str): Either 'butter' or 'sine2' type filters.
    bpf_sine2_width (float): Width parameter if using 'sine2' bpf.
    """
    aman.wrap_new('demodQ', dtype='float32', shape=('dets', 'samps'))
    aman.wrap_new('demodU', dtype='float32', shape=('dets', 'samps'))
    aman.wrap_new('dsT', dtype='float32', shape=('dets', 'samps'))

    phasor = np.exp(4.j * aman.hwp_angle)
    filt = tod_ops.filters.low_pass_sine2(fc_lpf, width=lpf_sin2_width)

    if bpf:
        if bpf_center is None:
            speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
                     (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
            bpf_center = 4 * speed

        if bpf_type == 'butter':
            bpf_filt = tod_ops.filters.low_pass_butter4(fc=bpf_center + bpf_width)*\
                       tod_ops.filters.high_pass_butter4(fc=bpf_center - bpf_width)
        if bpf_type == 'sine2':
            bpf_filt = tod_ops.filters.low_pass_sine2(cutoff=bpf_center + bpf_width,
                                                      width=bpf_sine2_width)*\
                       tod_ops.filters.high_pass_sine2(cutoff=bpf_center - bpf_width,
                                                       width=bpf_sine2_width)

        demod = tod_ops.fourier_filter(aman, bpf_filt, detrend=None,
                                       signal_name=signal) * phasor
    else:
        demod = aman[signal] * phasor
    aman.dsT = aman[signal]

    aman['demodQ'] = demod.real
    aman['demodQ'] = tod_ops.fourier_filter(aman, filt, signal_name='demodQ', detrend=None) * 2
    aman['demodU'] = demod.imag
    aman['demodU'] = tod_ops.fourier_filter(aman, filt, signal_name='demodU', detrend=None) * 2
    aman['dsT'] = tod_ops.fourier_filter(aman, filt, signal_name='dsT', detrend=None)
