import numpy as np
from scipy.optimize import curve_fit
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
        speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
                 (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
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


# Functions to test with preprocessing pipeline added by Max S-F below
def hwpss_linreg(aman, signal='signal', modes=[2, 4, 6, 8], mask_flags=True,
                 add_to_aman=True, name='hwpss_extract', return_chi2=True):
    """
    Fit out half-waveplate synchronous signal (harmonics of rotation frequency),
    The harmonics to fit out are defined by ``modes`` and the operation is
    performed on the ``signal`` axis of the ``aman`` axis manager.

    Args
    ----
    aman (AxisManager): Axis manager to perform fit.
    signal (string): Axis within aman to perform fit on.
    modes (int list): List of hwp rotation frequency harmonics to fit.
    mask_flags (bool): Mask flagged samples in fit.
    add_to_aman (bool): Add fitted hwpss to aman.
    name (string): Axis name for fitted signal if ``add_to_aman`` is True.
    return_chi2 (bool): If true calculates the reduced chi^2 of the fit
                        and returns it in aman_proc.

    Returns
    -------
    aman_proc (AxisManager): Axis manager containing fitted ceofficients.
    """
    vects = np.ones( (aman.samps.count,1) )
    for mode in modes:
        vects = np.hstack((vects,
                           np.array( [np.sin(mode*aman.hwp_angle),
                           np.cos(mode*aman.hwp_angle)]).transpose()))
    vects = vects.T

    coeffs = np.empty((aman.dets.count, len(vects)))

    if mask_flags:
        m = ~aman.flags.glitches.mask()
        vects = vects[:,m]
    else:
        vects = vects

    I = np.linalg.inv(np.tensordot(vects, vects, (1,1)))

    for k in range(aman.dets.count):
        d = aman['signal'][k]
        if mask_flags:
            d = d[m]
        coeffs[k,:] = np.dot(vects, d)

    coeffs = np.dot(I, coeffs.T).T

    fitsig = np.matmul(vects.T, coeffs.T)
    aman_proc = core.AxisManager().wrap('hwpss_coeffs', coeffs,
                                        [(0, core.LabelAxis('dets', aman.dets.vals)),
                                         (1, core.IndexAxis('modes'))])
    aman_proc.wrap('hwpss_vects', vects, [(0, core.IndexAxis('modes')),
                                          (1, core.OffsetAxis('samps'))])
    aman_proc.wrap('hwpss_harms', np.asarray(modes), [(0, core.IndexAxis('modes'))])

    if add_to_aman:
        if name in aman.keys() and overwrite:
            aman.move(name, None)
        aman.wrap(name, fitsig.T, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                   (1, core.OffsetAxis('samps'))])
    if return_chi2:
        rss = np.sum((aman[signal]-fitsig.T)**2, axis=1)
        _ = tod_ops.fft_ops.calc_psd(aman, merge=True, nperseg=aman.samps.count//2)
        wn = tod_ops.fft_ops.calc_wn(aman, low_f=20, high_f=30)
        red_chi2 = rss/wn/(aman.samps.count-aman_proc.modes.count)
        aman_proc.wrap('hwpss_fit_red_chi2', red_chi2, [(0, core.LabelAxis('dets', aman.dets.vals))])

    return aman_proc


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


def demod_tod(aman, signal='hwpss_remove', fc_lpf=2., width=0.5, bpf=False,
              bpf_width=2., bpf_center=None, bpf_type='sine2', 
              sine2_width=None):
    """
    Simple demodulation function. Wraps new axes demodQ (real part from hwp
    demodulation), demodU (imag part from hwp demodulation), and dsT (lowpass
    filtered version of raw signal) into input axis manager ``aman``.
    Args
    ----
    aman (AxisManager): Axis manager to perform fit.
    signal (str): Axis to demodulate
    fc_lpf (float): low pass filter cutoff
    width (float): width of sine^2 low pass filter
    bpf (bool): Apply bandpass filter before demodulation.
    bpf_width (float): Width of bandpass filter in Hz.
    bpf_center (float): Center of bandpass filter, if not passed will
                        estimate 4*f_HWP from hwp_angles in aman.
    bpf_type (str): Either 'butter' or 'sine2' type filters.
    sine2_width (float): Width parameter if using 'sine2' bpf.
    """
    aman.wrap_new('demodQ', dtype='float32', shape=('dets', 'samps'))
    aman.wrap_new('demodU', dtype='float32', shape=('dets', 'samps'))
    aman.wrap_new('dsT', dtype='float32', shape=('dets', 'samps'))

    phasor = np.exp(4.j * aman.hwp_angle)
    filt = tod_ops.filters.low_pass_sine2(fc_lpf, width=width)

    if bpf:
        if bpf_center is None:
            speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
                     (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
            bpf_center = 4 * speed

        if bpf_type == 'butter':
            bpf_filt = tod_ops.filters.low_pass_butter4(fc=bpf_center + bpf_width)*\
                       tod_ops.filters.high_pass_butter4(fc=bpf_center - bpf_width)
        if bpf_type == 'sine2':
            bpf_filt = tod_ops.filters.low_pass_sine2(fc=bpf_center + bpf_width,
                                                      width=sine2_width)*\
                       tod_ops.filters.high_pass_sine2(fc=bpf_center - bpf_width,
                                                       width=sine2_width)

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
