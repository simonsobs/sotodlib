"""Functions for estimating and subtracting the temperature to polarization leakage.
"""
import numpy as np
from sotodlib import core
from sotodlib.tod_ops import filters, apodize
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from scipy.odr import ODR, Model, RealData
from lmfit import Model as LmfitModel


def t2p_fit(aman, T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU',
            sigma_T=None, sigma_demod=None, flag_name=None, ds_factor=100):
    """

    Compute the leakage coefficients from temperature (T) to polarization (Q and U)
    individually.  Return an axismanager of coefficients with their 
    statistical uncertainties and reduced chi-squared values for the fit.

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    T_sig_name : str
        Name of the temperature signal in `aman`. Default is 'dsT'.
    Q_sig_name : str
        Name of the Q polarization signal in `aman`. Default is 'demodQ'.
    U_sig_name : str
        Name of the U polarization signal in `aman`. Default is 'demodU'.
    flag_name : str or None
        Name of the flag field in `aman` to use for masking data. If None,
        no masking is applied.
    sigma_T : array-like
        Array of standard deviations for the signal TOD used fitting.
        If None, all are set to unity.
    sigma_demod : array-like or None
        Array of standard deviations for the Q and U TODs used as weights
        while fitting.  If None, all are set to unity.
    ds_factor : float
        Factor by which to downsample the TODs prior to fitting. Default is 100. 

    Returns
    -------
    out_aman : AxisManager
        An AxisManager containing leakage coefficients, their errors, and reduced
        chi-squared statistics.
    """

    def leakage_model(params, x):
        return params[0] * x + params[1]

    if sigma_T is None:
        sigma_T = np.ones_like(aman.dets.count)

    if sigma_demod is None:
        sigma_demod = np.ones_like(aman.dets.count)

    if flag_name is None:
        mask = np.ones_like(aman[T_sig_name], dtype='bool')
    elif flag_name in aman.flags._fields.keys():
        mask = ~aman.flags[flag_name].mask()
    else:
        raise ValueError('flag_name should be in aman.flags')

    coeffsQ = np.zeros(aman.dets.count)
    errorsQ = np.zeros(aman.dets.count)
    redchi2sQ = np.zeros(aman.dets.count)
    coeffsU = np.zeros(aman.dets.count)
    errorsU = np.zeros(aman.dets.count)
    redchi2sU = np.zeros(aman.dets.count)

    for di, det in enumerate(aman.dets.vals):        
        T_ds_det = aman[T_sig_name][di][mask[di]][::ds_factor]
        Q_ds_det = aman[Q_sig_name][di][mask[di]][::ds_factor]
        U_ds_det = aman[U_sig_name][di][mask[di]][::ds_factor]

        # fitting for Q
        try:
            model = Model(leakage_model)
            data = RealData(x=T_ds_det, 
                            y=Q_ds_det, 
                            sx=np.ones_like(T_ds_det) * sigma_T[di], 
                            sy=np.ones_like(T_ds_det) * sigma_demod[di])
            odr = ODR(data, model, beta0=[np.mean(Q_ds_det), 1e-3])
            output = odr.run()
            coeffsQ[di] = output.beta[0]
            errorsQ[di] = output.sd_beta[0]
            redchi2sQ[di] = output.sum_square / (len(T_ds_det) - 2)
        except:
            coeffsQ[di] = np.nan
            errorsQ[di] = np.nan
            redchi2sQ[di] = np.nan

        # fitting for U
        try:
            model = Model(leakage_model)
            data = RealData(x=T_ds_det, 
                            y=U_ds_det, 
                            sx=np.ones_like(T_ds_det) * sigma_T[di], 
                            sy=np.ones_like(T_ds_det) * sigma_demod[di])
            odr = ODR(data, model, beta0=[np.mean(U_ds_det), 1e-3])
            output = odr.run()
            coeffsU[di] = output.beta[0]
            errorsU[di] = output.sd_beta[0]
            redchi2sU[di] = output.sum_square / (len(T_ds_det) - 2)
        except:
            coeffsU[di] = np.nan
            errorsU[di] = np.nan
            redchi2sU[di] = np.nan

    out_aman = core.AxisManager(aman.dets, aman.samps)
    out_aman.wrap('coeffsQ', coeffsQ, [(0, 'dets')])
    out_aman.wrap('errorsQ', errorsQ, [(0, 'dets')])
    out_aman.wrap('redchi2sQ', redchi2sQ, [(0, 'dets')])

    out_aman.wrap('coeffsU', coeffsU, [(0, 'dets')])
    out_aman.wrap('errorsU', errorsU, [(0, 'dets')])
    out_aman.wrap('redchi2sU', redchi2sU, [(0, 'dets')])

    return out_aman

def t2p_joint_fit(aman, T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU',
                  sigma_demod=None, flag_name=None, ds_factor=100):
    """
    Compute the leakage coefficients from temperature (T) to polarization (Q and U)
    by performing a joint fit of both simultaneously. Return an axismanager of the 
    coefficients with their statistical uncertainties and reduced chi-squared values 
    for the fit.

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    T_sig_name : str
        Name of the temperature signal in `aman`. Default is 'dsT'.
    Q_sig_name : str
        Name of the Q polarization signal in `aman`. Default is 'demodQ'.
    U_sig_name : str
        Name of the U polarization signal in `aman`. Default is 'demodU'.
    flag_name : str or None
        Name of the flag field in `aman` to use for masking data. If None,
        no masking is applied.
    sigma_demod : array-like or None
        Array of standard deviations for the Q and U TOD used as weights
        while fitting. If None, all are set to unity.
    ds_factor : float
        Factor by which to downsample the TODs prior to fitting. Default is 100. 

    Returns
    -------
    out_aman : AxisManager
        An AxisManager containing leakage coefficients, their errors, and reduced
        chi-squared statistics.
    """

    def leakage_model(dT, AQ, AU, lamQ, lamU):
        return AQ + lamQ * dT + 1.j * (AU + lamU * dT)

    if sigma_demod is None:
        sigma_demod = np.ones_like(aman.dets.count)

    if flag_name is None:
        mask = np.ones_like(aman[T_sig_name], dtype='bool')
    elif flag_name in aman.flags._fields.keys():
        mask = ~aman.flags[flag_name].mask()
    else:
        raise ValueError('flag_name should be in aman.flags')

    A_Q_array = np.empty([aman.dets.count])
    A_U_array = np.empty_like(A_Q_array)
    lambda_Q_array = np.empty_like(A_Q_array)
    lambda_U_array = np.empty_like(A_Q_array)

    A_Q_error = np.empty_like(A_Q_array)
    A_U_error = np.empty_like(A_Q_array)
    lambda_Q_error = np.empty_like(A_Q_array)
    lambda_U_error = np.empty_like(A_Q_array)

    redchi2s_array = np.empty_like(A_Q_array)

    for di, det in enumerate(aman.dets.vals):
        x = aman[T_sig_name][di][mask[di]][::ds_factor]
        yQ = aman[Q_sig_name][di][mask[di]][::ds_factor]
        yU = aman[U_sig_name][di][mask[di]][::ds_factor]

        try:
            model = LmfitModel(leakage_model, independent_vars=['dT'], 
                          weights=np.ones_like(x)/sigma_demod[di])
            params = model.make_params(AQ=np.median(yQ), AU=np.median(yU), 
                                       lamQ=0., lamU=0.)
            result = model.fit(yQ + 1j * yU, params, dT=x)
            A_Q_array[di] = result.params['AQ'].value
            A_U_array[di] = result.params['AU'].value
            lambda_Q_array[di] = result.params['lamQ'].value
            lambda_U_array[di] = result.params['lamU'].value

            A_Q_error[di] = result.params['AQ'].stderr
            A_U_error[di] = result.params['AU'].stderr
            lambda_Q_error[di] = result.params['lamQ'].stderr
            lambda_U_error[di] = result.params['lamU'].stderr
            redchi2s_array[di] = result.redchi
        except:
            A_Q_array[di] = np.nan
            A_U_array[di] = np.nan
            lambda_Q_array[di] = np.nan
            lambda_U_array[di] = np.nan
            
            A_Q_error[di] = np.nan
            A_U_error[di] = np.nan
            lambda_Q_array[di] = np.nan
            lambda_U_error[di] = np.nan
            redchi2s_array[di] = np.nan

    out_aman = core.AxisManager(aman.dets, aman.samps)
    out_aman.wrap('AQ', A_Q_array, [(0, 'dets')])
    out_aman.wrap('AU', A_U_array, [(0, 'dets')])
    out_aman.wrap('lamQ', lambda_Q_array, [(0, 'dets')])
    out_aman.wrap('lamU', lambda_U_array, [(0, 'dets')])

    out_aman.wrap('AQ_error', A_Q_error, [(0, 'dets')])
    out_aman.wrap('AU_error', A_U_error, [(0, 'dets')])
    out_aman.wrap('lamQ_error', lambda_Q_error, [(0, 'dets')])
    out_aman.wrap('lamU_error', lambda_U_error, [(0, 'dets')])
    out_aman.wrap('redchi2s', redchi2s_array, [(0, 'dets')])

    return out_aman

def get_t2p_coeffs(aman, T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU',
                   joint_fit=True, wn_demod=None, f_lpf_cutoff=2.0, flag_name=None,
                   ds_factor=100, subtract_sig=False, merge_stats=True, 
                   t2p_stats_name='t2p_stats'):
    """
    Compute the leakage coefficients from temperature (T) to polarization (Q and U) by
    either a joint fit of both or individually. Optionally subtract this leakage. Return 
    an axismanager of the coefficients with their statistical uncertainties and reduced 
    chi-squared values for the fit.

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    T_sig_name : str
        Name of the temperature signal in `aman`. Default is 'dsT'.
    Q_sig_name : str
        Name of the Q polarization signal in `aman`. Default is 'demodQ'.
    U_sig_name : str
        Name of the U polarization signal in `aman`. Default is 'demodU'.
    joint_fit : bool
        Whether to fit Q and U leakage coefficients as parameters in a single model or
        fit independently. Default is True.
    wn_demod : float or None
        Precomputed white noise level for demodulated signals. If None, it will be calculated.
    f_lpf_cutoff : float
        Cutoff frequency of low pass filter in demodulation. Used for error bar estimation by
        combination with wn_demod. Default is 2.0.
    flag_name : str
        Name of the flag field in `aman` to use for masking data. If None, no masking is applied.
    ds_factor: float or None
        Factor by which to downsample the TODs prior to fitting. If None, the low pass filter
        frequency is used to estimate the factor.
    subtract_sig : bool
        Whether to subtract the calculated leakage from the polarization signals. Default is False.
    merge_stats : bool
        Whether to merge the calculated statistics back into `aman`. Default is True.
    t2p_stats_name : str
        Name under which to wrap the output AxisManager containing statistics. Default is 't2p_stats'.

    Returns
    -------
    out_aman : AxisManager
                An AxisManager containing leakage coefficients, their errors, and reduced
                chi-squared statistics.
    """

    # get white noise level of demod for error estimation
    if wn_demod is None:
        freqs, Pxx_demod = calc_psd(aman, signal=aman[Q_sig_name], merge=False)
        wn_demod = calc_wn(aman, pxx=Pxx_demod, freqs=freqs, low_f=0.5, high_f=1.5)

    # integrate the white noise level over frequencies to get error bar of each point
    sigma_demod = wn_demod * np.sqrt(f_lpf_cutoff)
    sigma_T = sigma_demod / np.sqrt(2)

    if ds_factor is None:
        ds_factor = int(np.mean(1. / np.diff(aman.timestamps)) / (f_lpf_cutoff))

    if joint_fit:
        out_aman = t2p_joint_fit(aman, T_sig_name, Q_sig_name, U_sig_name,
                                 sigma_demod, flag_name, ds_factor)
    else:
        out_aman = t2p_fit(aman, T_sig_name, Q_sig_name, U_sig_name,
                           sigma_T, sigma_demod, flag_name, ds_factor)

    if subtract_sig:
        subtract_t2p(aman, out_aman)
    if merge_stats:
        aman.wrap(t2p_stats_name, out_aman)

    return out_aman

def subtract_t2p(aman, t2p_aman, T_signal=None):
    """
    Subtract T to P leakage.

    Parameters
    ----------
    aman : AxisManager
        The tod.
    t2p_aman : AxisManager
        Axis manager with Q and U leakage coeffients.
        If joint fitting was used in get_t2p_coeffs, Q coeffs are in 
        fields ``lamQ`` and ``AQ`` and U coeffs are in ``lamU`` and 
        ``AU``. Otherwise Q coeff is in field ``coeffsQ`` and U coeff 
        in ``coeffsU``.
    T_signal : array
        Temperature signal to scale and subtract from Q/U.
        Default is ``aman['dsT']``.
    """

    if T_signal is None:
        T_signal = aman['dsT']

    if 'AQ' in t2p_aman._assignments:
        aman.demodQ -= (T_signal * t2p_aman.lamQ[:, np.newaxis] + t2p_aman.AQ[:, np.newaxis])
        aman.demodU -= (T_signal * t2p_aman.lamU[:, np.newaxis] + t2p_aman.AU[:, np.newaxis])
    elif 'coeffsQ' in t2p_aman._assignments:
        aman.demodQ -= np.multiply(T_signal.T, t2p_aman.coeffsQ).T
        aman.demodU -= np.multiply(T_signal.T, t2p_aman.coeffsU).T
    else:
        raise ValueError('no leakage coefficients found in axis manager')