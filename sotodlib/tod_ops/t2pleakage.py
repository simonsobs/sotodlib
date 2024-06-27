"""Functions for estimating and subtracting the temperature to polarization leakage.
"""
import numpy as np
from sotodlib import core
from sotodlib.tod_ops import filters, apodize
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from scipy.odr import ODR, Model, RealData

def get_t2p_coeffs(aman, 
                   T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU', wn_demod=None,
                   apodize_samps=2000, lpf_cfgs=None, flag_name=None, 
                   subtract_sig=False, merge_stats=True, t2p_stats_name='t2p_stats'):
    """
    Apply a lowpass filter to the temperature and polarization signals, apodize them,
    and compute the leakage coefficients from temperature (T) to polarization (Q and U).
    Optionally subtract this leakage and return axismanager of coefficients with their 
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
    wn_demod : float or None
        Precomputed white noise level for demodulated signals. If None, it will be calculated.
    apodize_samps : int
        Number of samples over which to apodize at the start and end of the signals. Default is 2000.
    lpf_cfgs : dict or None
        Configuration for the lowpass filter. If None, defaults to {'type':'sine2', 'cutoff':1.0, 'trans_width':0.01}.
    flag_name : str or None
        Name of the flag field in `aman` to use for masking data. If None, no masking is applied.
    subtract_sig : bool
        Whether to subtract the calculated leakage from the polarization signals. Default is False.
    merge_stats : bool
        Whether to merge the calculated statistics back into `aman`. Default is True.
    t2p_stats_name : str
        Name under which to wrap the output AxisManager containing statistics. Default is 't2p_stats'.

    Returns
    -------
    out_aman : AxisManager
        An AxisManager containing leakage coefficients, their errors, and reduced chi-squared statistics.
    """
    if lpf_cfgs is None:
        lpf_cfgs = {'type':'sine2', 'cutoff':1.0, 'trans_width':0.01}
    filt = filters.get_lpf(lpf_cfgs)
    
    # apply lowpass filter with apodization
    apodize.apodize_cosine(aman, signal=T_sig_name, apodize_samps=apodize_samps, in_place=False, apo_axis='apodized')
    T_lpf = filters.fourier_filter(aman, filt, signal_name='apodized')
    aman.move('apodized', None)
    apodize.apodize_cosine(aman, signal=Q_sig_name, apodize_samps=apodize_samps, in_place=False, apo_axis='apodized')
    Q_lpf = filters.fourier_filter(aman, filt, signal_name='apodized')
    aman.move('apodized', None)
    apodize.apodize_cosine(aman, signal=U_sig_name, apodize_samps=apodize_samps, in_place=False, apo_axis='apodized')
    U_lpf = filters.fourier_filter(aman, filt, signal_name='apodized')
    aman.move('apodized', None)
    
    # get white noise level of demod for error estimation
    if wn_demod is None:
        freqs, Pxx_demod = calc_psd(aman, signal=aman[Q_sig_name], merge=False)
        wn_demod = calc_wn(aman, pxx=Pxx_demod, freqs=freqs, low_f=0.5, high_f=1.5)
        
    # integrate the white noise level over frequencies to get error bar of each point
    sigma_demod = wn_demod * np.sqrt(lpf_cfgs['cutoff']) 
    sigma_T = sigma_demod/np.sqrt(2)
    
    # get downsampled data
    ds_factor = int(np.mean(1/np.diff(aman.timestamps)) / (2 * lpf_cfgs['cutoff']) )
    ds_slice = slice(apodize_samps, -apodize_samps, ds_factor)
    ts_ds = aman.timestamps[ds_slice]
    T_lpf_ds = T_lpf[:, ds_slice]
    Q_lpf_ds = Q_lpf[:, ds_slice]
    U_lpf_ds = U_lpf[:, ds_slice]

    if flag_name is None:
        mask_ds = np.ones_like(T_lpf_ds, dtype=bool)
    elif flag_name in aman.flags._fields.keys():
        mask_ds = ~aman.flags[flag_name].mask()[:, ds_slice]
    else:
        raise ValueError('flag_name should be in aman.flags')
    
    def linear_model(params, x):
        return params[0] * x + params[1]
    
    coeffsQ = np.zeros(aman.dets.count)
    errorsQ = np.zeros(aman.dets.count)
    redchi2sQ = np.zeros(aman.dets.count)
    coeffsU = np.zeros(aman.dets.count)
    errorsU = np.zeros(aman.dets.count)
    redchi2sU = np.zeros(aman.dets.count)
    
    for di, det in enumerate(tqdm(aman.dets.vals)):
        mask_ds_det = mask_ds[di]
        ts_ds_det = ts_ds[mask_ds_det]
        T_lpf_ds_det = T_lpf_ds[di, mask_ds_det]
        Q_lpf_ds_det = Q_lpf_ds[di, mask_ds_det]
        U_lpf_ds_det = U_lpf_ds[di, mask_ds_det]
        
        # fitting for Q
        try:
            model = Model(linear_model)
            data = RealData(x=T_lpf_ds_det, 
                            y=Q_lpf_ds_det, 
                            sx=np.ones_like(T_lpf_ds_det) * sigma_T[di], 
                            sy=np.ones_like(T_lpf_ds_det) * sigma_demod[di])
            odr = ODR(data, model, beta0=[np.mean(Q_lpf_ds_det), 1e-3])
            output = odr.run()
            coeffsQ[di] = output.beta[0]
            errorsQ[di] = output.sd_beta[0]
            redchi2sQ[di] = output.sum_square / (len(T_lpf_ds_det) - 2)
        except:
            coeffsQ[di] = np.nan
            errorsQ[di] = np.nan
            redchi2sQ[di] = np.nan
        
        # fitting for U
        try:
            model = Model(linear_model)
            data = RealData(x=T_lpf_ds_det, 
                            y=U_lpf_ds_det, 
                            sx=np.ones_like(T_lpf_ds_det) * sigma_T[di], 
                            sy=np.ones_like(T_lpf_ds_det) * sigma_demod[di])
            odr = ODR(data, model, beta0=[np.mean(U_lpf_ds_det), 1e-3])
            output = odr.run()
            coeffsU[di] = output.beta[0]
            errorsU[di] = output.sd_beta[0]
            redchi2sU[di] = output.sum_square / (len(T_lpf_ds_det) - 2)
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
    
    if subtract_sig:
        subtract_t2p(aman, out_aman)
    if merge_stats:
        aman.wrap(t2p_stats_name, out_aman, [(0, 'dets')])
    
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
        Q coeff in field ``coeffsQ`` and U coeff in field ``coeffsU``.
    T_signal : array
        Temperature signal to scale and subtract from Q/U.
        Default is ``aman['dsT']``.

    """
    if T_signal is None:
        T_signal = aman['dsT']
    aman.demodQ -= np.multiply(T_signal.T, t2p_aman.coeffsQ).T
    aman.demodU -= np.multiply(T_signal.T, t2p_aman.coeffsU).T
    
