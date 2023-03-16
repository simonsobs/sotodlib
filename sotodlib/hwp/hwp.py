import numpy as np
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)

def extract_hwpss(aman, signal=None,
                  bin_signal=True, bins=360, 
                  linear_regression=True, modes=[2, 4, 6, 8], 
                  mask_flags=True, add_to_aman=True, name='hwpss_extract'):
    
    if signal is None:
        if apply_prefilt: 
            filt = tod_ops.filters.high_pass_sine2(cutoff=prefilt_cutoff)
            signal = np.array(tod_ops.fourier_filter(aman, filt, detrend='linear', signal_name='signal'))
        else:
            signal = aman.signal
            
    if bin_signal:
        aman_proc = binning_signal(aman, signal, hwp_angle=None, bins=bins, mask_flags=mask_flags)
        if linear_regression:
            #hoge
        else:
            #hoge
        
    else:
        if linear_regression:
            #hoge
        else:
            #hoge
    
    return #aman_proc

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
    
    # get aman_proc
    aman_proc = core.AxisManager(aman.dets, core.IndexAxis('bin_samps', count=bins))
    aman_proc.wrap('hwp_angle_bin_centers', hwp_angle_bin_centers, [(0, 'bin_samps')])
    aman_proc.wrap('binned_hwpss', binned_hwpss, [(0, 'dets'), (1, 'bin_samps')])
    aman_proc.wrap('hwpss_sigma', hwpss_sigma, [(0, 'dets')])
    
    return aman_proc
    
def hwpss_linreg(x, ys, yerrs):
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    I = np.linalg.inv(np.tensordot(vects, vects, (1,1)))
    coeffs = np.matmul(ys, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    fitsig = np.matmul(vects.T, coeffs.T).T
    
    # covariance of coefficients
    covar = np.zeros((ys.shape[0], 2*len(modes), 2*len(modes)))    
    for det_idx in range(ys.shape[0]):
        covar[det_idx, :, :] = I * yerrs[det_idx]
    
    # reduced chi-square
    redchi2 = np.sum(((ys - fitsig)/yerrs[:, np.newaxis])**2, axis=-1) / (x.shape[0] - 2*len(modes))
    
    return fitsig, coeffs, covar, redchi2


def get_hwpss_guess(aman, signal='signal', modes=[2, 4, 6, 8]):
    """
    Projects out fourier modes, ``modes`` from ``aman`` and returns them. This
    can be used as an initial guess for the fitting function.

    Args
    ----
    aman (AxisManager):
    """
    As = np.asarray([np.sum(aman[signal]*np.sin(m*aman.hwp_angle)\
                    * np.diff(aman.timestamps).mean(), axis=1)\
                    / (np.ptp(aman.timestamps)/2) for m in modes])
    Bs = np.asarray([np.sum(aman[signal]*np.cos(m*aman.hwp_angle)\
                    * np.diff(aman.timestamps).mean(), axis=1)\
                    / (np.ptp(aman.timestamps)/2) for m in modes])
    proc_aman = core.AxisManager().wrap('A_guess', As.T, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                                          (1, core.IndexAxis('modes'))])
    proc_aman.wrap('B_guess', Bs.T, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                     (1, core.IndexAxis('modes'))])
    return proc_aman


def extract_hwpss_template(aman, As, Bs, signal='signal', name='hwpss_ext',
                           mask_flags=True, modes=[2, 4, 6, 8],
                           add_to_aman=True):
    """
    Function for fitting out the hwpss template given a guess of the fourier
    coefficients that match the size of the list of ``modes`` intended to fit.
    This uses ``scipy.optimize.curve_fit`` to fit the template so that you can
    fit masked versions of the signal to avoid biasing the template estimations
    by the glitches in the data.
    Args
    ----
    aman (AxisManager):
    As (float list):
    Bs (float list):
    signal (str):
    name (str):
    mask_flags (bool):
    bins (int):
    """
    N_modes = len(modes)
    new_As = np.zeros((aman.dets.count, N_modes))
    new_Bs = np.zeros((aman.dets.count, N_modes))
    new_A0 = np.zeros(aman.dets.count)
    new_modes = np.zeros((aman.dets.count, N_modes))
    sins = np.zeros((N_modes, aman.samps.count))
    coss = np.zeros((N_modes, aman.samps.count))

    for i, nm in enumerate(modes):
        sins[i, :] = np.sin(nm*aman.hwp_angle)
        coss[i, :] = np.cos(nm*aman.hwp_angle)

    def wrapper_fit_func(x, N, sins, coss, *args):
        a, b, m, a0 = list(args[0][:N]), list(args[0][N:2*N]),\
                      list(args[0][2*N:3*N]), args[0][-1]
        return fit_func(x, a0, a, b, m, N, sins, coss)

    def fit_func(x, A0, As, Bs, N_mode, N, sins, coss):
        y = A0*np.ones(len(x))
        for i, [a, b] in enumerate(list(zip(As, Bs))):
            y += a*sins[i] + b*coss[i]
        return y

    if mask_flags:
        m = ~aman.flags.glitches.mask()
    else:
        m = np.full((aman.dets.count, aman.samps.count), True)

    for i in range(aman.dets.count):
        params_0 = list(As[:, i]) + list(Bs[:, i]) + list(modes)
        params_0.append(np.nanmedian(aman[signal][i, m[i]]))
        popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(aman.hwp_angle[m[i]], N_modes, sins[:,m[i]], coss[:,m[i]], params_0),
                               aman.hwp_angle[m[i]], aman[signal][i,m[i]], p0=params_0)

        new_As[i], new_Bs[i], new_modes[i], new_A0[i] = popt[0:N_modes], popt[N_modes:2*N_modes], popt[2*N_modes:3*N_modes], popt[-1]

    aman_proc = core.AxisManager().wrap('A_sine', new_As, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                                           (1, core.IndexAxis('modes'))])
    aman_proc.wrap('A_cos', new_Bs, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                     (1, core.IndexAxis('modes'))])
    aman_proc.wrap('modes', new_modes, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                        (1, core.IndexAxis('modes'))])
    aman_proc.wrap('A0', new_A0, [(0, core.LabelAxis('dets', aman.dets.vals))])
    if add_to_aman:
        construct_hwpss_template(aman, aman_proc, name=name)

    return aman_proc


def construct_hwpss_template(aman, template_coeff_aman, name='hwpss_ext', overwrite=True):
    """
    Adds an axis corresponding to the hwpss template signal to an axis manager
    given an axis manager with hwp angles and a axis manager with the fitted
    hwpss template coefficients and modes.
    """
    def wrapper_fit_func(x, N, sins, coss, *args):
        a, b, m, a0 = list(args[0][:N]), list(args[0][N:2*N]),\
                      list(args[0][2*N:3*N]), args[0][-1]
        return fit_func(x, a0, a, b, m, N, sins, coss)

    def fit_func(x, A0, As, Bs, N_mode, N, sins, coss):
        y = A0*np.ones(len(x))
        for i, [a, b] in enumerate(list(zip(As, Bs))):
            y += a*sins[i] + b*coss[i]
        return y

    N_modes = len(template_coeff_aman['modes'][0])
    sins = np.zeros((N_modes, aman.samps.count))
    coss = np.zeros((N_modes, aman.samps.count))
    for i, nm in enumerate(template_coeff_aman['modes'][0]):
        sins[i, :] = np.sin(nm*aman.hwp_angle)
        coss[i, :] = np.cos(nm*aman.hwp_angle)

    if name in aman.keys() and overwrite:
        aman.move(name, None)
    aman.wrap_new(name, ('dets', 'samps'))

    for i in range(aman.dets.count):
        aman[name][i] = wrapper_fit_func(aman.hwp_angle, N_modes, sins, coss,
                                         np.append(np.append(np.append(template_coeff_aman['A_sine'][i],
                                                   template_coeff_aman['A_cos'][i]),
                                                   template_coeff_aman['modes'][i]),
                                                   template_coeff_aman['A0'][i]))


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
