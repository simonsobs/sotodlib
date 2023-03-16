import numpy as np
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)

# Functions to test with preprocessing pipeline added by Max S-F below
def hwpss_linreg(aman, signal='signal', modes=[2, 4, 6, 8], mask_flags=True,
                 add_to_aman=True, name='hwpss_extract', return_fitstats=True):
    """
    Fit out half-waveplate synchronous signal (harmonics of rotation frequency),
    The harmonics to fit out are defined by ``modes`` and the operation is
    performed on the ``signal`` axis of the ``aman`` axis manager.

    Fitted function is:
    y = a_0 + a_1*sin(modes[0]*aman.hwp_angle) + a_2*cos(modes[0]*aman.hwp_angle) + 
        a_3*sin(modes[1]*aman.hwp_angle) + a_4*cos(modes[1]*aman.hwp_angle)) + ...
        a_(2n+1)*sin(modes[n]*aman.hwp_angle) + a_(2n+2)*cos(modes[n+1]*aman.hwp_angle)

    The returned proc_aman['coeffs'] is an array shaped n_dets x (n_modes + 1)
    containing the a_0, a_1, ... a_(2n+2) coeffients of the harmonic expansion y.

    Args
    ----
    aman (AxisManager): Axis manager to perform fit.
    signal (string): Axis within aman to perform fit on.
    modes (int list): List of hwp rotation frequency harmonics to fit.
    mask_flags (bool): Mask flagged samples in fit.
    add_to_aman (bool): Add fitted hwpss to aman.
    name (string): Axis name for fitted signal if ``add_to_aman`` is True.
    return_fitstats (bool): If true calculates the covariance matrix of the fit,
                            reduced chi^2 of the fit, and R^2 of the fit.

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
    modes_out = np.zeros(2*len(modes)+1)
    modes_out[0] = 0
    modes_out[1::2] = modes
    modes_out[2::2] = modes
    aman_proc.wrap('hwpss_harms', modes_out, [(0, core.IndexAxis('modes'))])

    if add_to_aman:
        if name in aman.keys() and overwrite:
            aman.move(name, None)
        aman.wrap(name, fitsig.T, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                   (1, core.OffsetAxis('samps'))])
    if return_fitstats:
        rss = np.sum((aman[signal]-fitsig.T)**2, axis=1)
        varreg = rss/(aman.samps.count-aman_proc.modes.count)
        vardat = np.var(aman[signal], axis=1)
        r2 = 1-rss/(aman.samps.count-1)/vardat
        redchi2 = varreg
        cov = np.zeros((aman.dets.count,aman_proc.modes.count,aman_proc.modes.count))
        for v in range(len(varreg)):
            cov[v,:,:] = I*varreg[v]
        aman_proc.wrap('hwpss_fit_covariance', cov, [(0, core.LabelAxis('dets', aman.dets.vals)),
                                                     (1, core.IndexAxis('modes')),
                                                     (2, core.IndexAxis('modes'))])
        aman_proc.wrap('hwpss_redchi2', redchi2, [(0,core.LabelAxis('dets', aman.dets.vals))])
        aman_proc.wrap('hwpss_r2', r2, [(0,core.LabelAxis('dets', aman.dets.vals))])

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
