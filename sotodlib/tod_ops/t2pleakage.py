"""Functions for estimating and subtracting the temperature to polarization leakage.
"""
from sotodlib import core
from sotodlib.tod_ops import filters, detrend
import numpy as np

def get_t2p_coeffs(aman, T_sig_name='dsT', Q_sig_name='demodQ', 
                   U_sig_name='demodU', subtract_sig=False, trim_samps=2000,
                   lpf_cfgs={'type':'sine2', 'cutoff':0.5, 'trans_width':0.1}):
    """
    Project Q and U onto T after lowpass filtering the signals so that the projection
    is dominantly on the low-f atmospheric signal. Then use the coefficients to
    subtract the leaked T from Q/U.

    Parameters
    ----------
    aman : AxisManager
        The tod
    T_sig_name : str
        Temperature signal. Default is ``dsT``.
    Q_sig_name : str
        Q polarization signal. Default is ``demodQ``.
    U_sig_name : str
        U polarization signal. Default is ``demodU``.
    subtract_sig : bool
        Whether to subtract the T to P leakage from ``aman[T_sig_name]``.
        Default is True.
    trim_samps : int
        Number of samples to trim from the ends of the TOD after lowpass filtering.
    lpf_cfgs : dict
        See ``sotodlib.tod_ops.filters.get_lpf`` for the structure.

    Returns
    -------
    out_aman : AxisManager
        Output axis manager with the Q/U leakage coefficients wrapped in.
    """
    # prepare T/Q/U for estimating leakage coefficients
    # Lowpass filter so estimation is mostly on low-f atmospheric component
    filt = filters.get_lpf(lpf_cfgs)
    T_lpf = filters.fourier_filter(aman, filt, signal_name=T_sig_name)
    Q_lpf = filters.fourier_filter(aman, filt, signal_name=Q_sig_name)
    U_lpf = filters.fourier_filter(aman, filt, signal_name=U_sig_name)
    aman.wrap('T_lpf', T_lpf, axis_map=[(0,'dets'), (1,'samps')])
    aman.wrap('Q_lpf', Q_lpf, axis_map=[(0,'dets'), (1,'samps')])
    aman.wrap('U_lpf', U_lpf, axis_map=[(0,'dets'), (1,'samps')])
    # Restrict ends of tod 
    aman.restrict('samps', (aman.samps.offset+trim_samps, 
                            aman.samps.offset+aman.samps.count-trim_samps))
    # Remove mean
    detrend.detrend_tod(aman, method='mean', signal_name=Q_sig_name)
    detrend.detrend_tod(aman, method='mean', signal_name=U_sig_name)
    detrend.detrend_tod(aman, method='mean', signal_name='Q_lpf')
    detrend.detrend_tod(aman, method='mean', signal_name='U_lpf')
    detrend.detrend_tod(aman, method='mean', signal_name='T_lpf')
    # Prepare output arrays
    coeffsQ = np.zeros(aman.dets.count)
    coeffsU = np.zeros(aman.dets.count)
    # Project out coefficients
    for di in range(aman.dets.count):
        I = np.linalg.inv(np.tensordot(np.atleast_2d(aman.T_lpf[di]), 
                                       np.atleast_2d(aman.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(aman.Q_lpf[di]), 
                      np.atleast_2d(aman.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsQ[di] = c[0]
        I = np.linalg.inv(np.tensordot(np.atleast_2d(aman.T_lpf[di]), 
                                       np.atleast_2d(aman.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(aman.U_lpf[di]), 
                      np.atleast_2d(aman.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsU[di] = c[0]
    # Wrap coefficients into output axis manager
    out_aman = core.AxisManager(aman.dets, aman.samps)
    out_aman.wrap('coeffsQ', coeffsQ, [(0, 'dets')])
    out_aman.wrap('coeffsU', coeffsU, [(0, 'dets')])
    # Delete lowpass signals used to estimate coefficients
    aman.move('T_lpf', None)
    aman.move('Q_lpf', None)
    aman.move('U_lpf', None)
    if subtract_sig:
        subtract_t2p(aman, out_aman)
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
    
