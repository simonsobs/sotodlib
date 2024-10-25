"""Functions for estimating and subtracting the temperature to polarization leakage.
"""
import numpy as np
from sotodlib import core
from sotodlib.tod_ops import filters, apodize
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from lmfit import Model
    
def leakage_model(dT, AQ, AU, lamQ, lamU):
    return AQ + lamQ * dT + 1.j * (AU + lamU * dT)

def get_t2p_coeffs(aman, T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU',
                   wn_demod=None, f_lpf_cutoff=2.0, mask=None, ds_factor=100, 
                   subtract_sig=False, merge_stats=True, t2p_stats_name='t2p_stats'):
    """
    Compute the leakage coefficients from temperature (T) to polarization (Q and U).
    Optionally subtract this leakage and return axismanager of coefficients.

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
    subtract_sig : bool
        Whether to subtract the calculated leakage from the polarization signals. Default is False.
    merge_stats : bool
        Whether to merge the calculated statistics back into `aman`. Default is True.
    t2p_stats_name : str
        Name under which to wrap the output AxisManager containing statistics. Default is 't2p_stats'.

    Returns
    -------
    out_aman : AxisManager
        An AxisManager containing leakage coefficients.
    """
    
    # get white noise level of demod for error estimation
    if wn_demod is None:
        freqs, Pxx_demod = calc_psd(aman, signal=aman[Q_sig_name], merge=False)
        wn_demod = calc_wn(aman, pxx=Pxx_demod, freqs=freqs, low_f=0.5, high_f=1.5)
        
    # integrate the white noise level over frequencies to get error bar of each point
    sigma_demod = wn_demod * np.sqrt(f_lpf_cutoff)
    sigma_T = sigma_demod/np.sqrt(2)
    
    if mask is None:
        mask = np.ones_like(aman[T_sig_name], dtype='bool')
        
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
            model = Model(leakage_model, independent_vars=['dT'], weights=np.ones_like(x)/sigma_demod[di])
            params = model.make_params(AQ=np.median(yQ), AU=np.median(yU), lamQ=0., lamU=0.)
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
        Q coeffs are in fields ``lamQ`` and ``AQ`` and U coeffs are in fields 
        ``lamU`` and ``AU``.
    T_signal : array
        Temperature signal to scale and subtract from Q/U.
        Default is ``aman['dsT']``.

    """
    
    if T_signal is None:
        T_signal = aman['dsT']

    aman.demodQ -= (T_signal * t2p_aman.lamQ[:, np.newaxis] + t2p_aman.AQ[:, np.newaxis])
    aman.demodU -= (T_signal * t2p_aman.lamU[:, np.newaxis] + t2p_aman.AU[:, np.newaxis])