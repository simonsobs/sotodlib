import numpy as np
from numpy.polynomial import legendre as L
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sotodlib import core, tod_ops
from sotodlib.tod_ops import bin_signal
import logging

logger = logging.getLogger(__name__)

def bin_by_az(aman, signal=None, az=None, bin_range=[-np.pi, np.pi], bins=3600, flags=None):
    """
    Bins a signal by azimuth angle.

    Parameters
    ----------
    aman: TOD
        core.AxisManager
    signal: array-like, optional
        numpy array of signal to be binned. If None, the signal is taken from aman.signal.
    az: array-like, optional
        A 1D numpy array representing the azimuth angles. If not provided, the azimuth angles are taken from aman.boresight.az attribute.
    bin_range: array-like, optional
        A list specifying the range of azimuth angles to consider for binning. Defaults to [-np.pi, np.pi].
        If None, [min(az), max(az)] will be used for binning.
    bins: integer
        The number of bins to divide the azimuth angle range into. Defaults to 360.
    flags: RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.

    Returns
    -------
    binning_dict: A dictionary containing the binned signal values.
        * 'bin_centers': center of each bin of azimuth
        * 'binned_signal': binned signal
        * 'binned_signal_sigma': estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if az is None:
        az = aman.boresight.az
    binning_dict = bin_signal(aman, bin_by=az, signal=signal,
                               bin_range=bin_range, bins=bins, flags=flags)
    return binning_dict
    
    
def get_sss(aman, signal=None, az=None, bin_range=[-np.pi, np.pi], bins=3600, flags=None, 
            method='interpolate', nmodes=None,
            merge_stats=True, sss_stats_name='sss_stats',
            merge_model=True, sss_model_name='sss_model'):
    """
    Derive SSS (Scan Synchronous Signal) statistics and model from the given AMAN data.

    Parameters
    ----------
    aman: TOD
        core.AxisManager
    signal: array-like, optional
        A numpy array representing the signal to be used for SSS extraction. If not provided, the signal is taken from aman.signal.
    az: array-like, optional
        A 1D numpy array representing the azimuth angles. If not provided, the azimuth angles are taken from aman.boresight.az.
    bin_range: list, optinal
        A list specifying the range of azimuth angles to consider for binning. Defaults to [-np.pi, np.pi].
        If None, [min(az), max(az)] will be used for binning.
    bins: integer
        The number of bins to divide the azimuth angle range into. Defaults to 360.
    flags : RangesMatrix, optinal
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.
    method: str
        The method to use for SSS modeling. Options are 'interpolate' and 'fit'. 
        In 'interpolate', binned signal is used directly.
        In 'fitting', fitting is applied to the binned signal.
        Defaults to 'interpolate'.
    nmodes: integer, optinal
        The number of Legendre modes to use for SSS when method is 'fit'. Required when method is 'fit'.
    merge_stats: boolean, optional
        Boolean flag indicating whether to merge the SSS statistics with aman. Defaults to True.
    sss_stats_name: string, optional
        The name to assign to the merged SSS statistics. Defaults to 'sss_stats'.
    merge_model: boolean, optional
        Boolean flag indicating whether to merge the SSS model with the aman. Defaults to True.
    sss_model_name: string, optional
        The name to assign to the merged SSS model. Defaults to 'sss_model'.

    Returns
    -------
        tuple: A tuple containing the SSS statistics and the SSS model.
            - sss_stats: An instance of core.AxisManager containing the SSS statistics.
            - model_sig_tod: A numpy array representing the SSS model as a function of time.
    """
    if signal is None:
        signal = aman.signal
    if az is None:
        az = aman.boresight.az
        
    # do binning
    binning_dict = bin_by_az(aman, signal=signal, az=az, bin_range=bin_range, bins=bins, flags=flags)
    bin_centers = binning_dict['bin_centers']
    binned_signal = binning_dict['binned_signal']
    binned_signal_sigma = binning_dict['binned_signal_sigma']
    uniform_binned_signal_sigma = np.nanmedian(binned_signal_sigma, axis=-1)
    
    sss_stats = core.AxisManager(aman.dets)
    sss_stats.wrap('binned_az', bin_centers, [(0, core.IndexAxis('bin_samps', count=bins))])
    sss_stats.wrap('binned_signal', binned_signal, [(0, 'dets'), (1, 'bin_samps')])
    sss_stats.wrap('binned_signal_sigma', binned_signal_sigma, [(0, 'dets'), (1, 'bin_samps')])
    
    if method == 'fit':
        bin_width = bin_centers[1] - bin_centers[0]
        m = ~np.isnan(binned_signal[0]) # masks bins without counts
        if np.count_nonzero(m) < nmodes:
            raise ValueError('Number of valid bins is smaller than mode of Legendre function')
        
        if bin_range==None:
            az_min = np.min(bin_centers[m]) - bin_width/2
            az_max = np.max(bin_centers[m]) + bin_width/2
        else:
            az_min = -np.pi
            az_max = np.pi
        
        x_Legendre = ( 2*az - (az_min+az_max) ) / (az_max - az_min)
        x_Legendre_bin_centers = ( 2*bin_centers - (az_min+az_max) ) / (az_max - az_min)
        x_Legendre_bin_centers = np.where(~m, np.nan, x_Legendre_bin_centers)
        
        mode_names = []
        for mode in range(nmodes+1):
            mode_names.append(f'Legendre{mode}')
        
        coeffs = L.legfit(x_Legendre_bin_centers[m], binned_signal[:, m].T, nmodes)
        coeffs = coeffs.T
        binned_model = L.legval(x_Legendre_bin_centers, coeffs.T)
        binned_model = np.where(~m, np.nan, binned_model)
        sum_of_squares = np.sum(((binned_signal[:, m] - binned_model[:,m])**2), axis=-1)
        redchi2s = sum_of_squares/uniform_binned_signal_sigma**2 /  ( len(x_Legendre_bin_centers[m]) - nmodes - 1)
        
        sss_stats.wrap('binned_model', binned_model, [(0, 'dets'), (1, 'bin_samps')])
        sss_stats.wrap('x_Legendre_bin_centers', x_Legendre_bin_centers, [(0, 'bin_samps')])
        sss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, core.LabelAxis(name='modes', vals=np.array(mode_names, dtype='<U10')))])
        sss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])
        
        model_sig_tod = L.legval(x_Legendre, coeffs.T)
        
    if method == 'interpolate':
        f_template = interp1d(bin_centers, binned_signal, fill_value='extrapolate')
        model_sig_tod = f_template(aman.boresight.az)
    
    if merge_stats:
        aman.wrap(sss_stats_name, sss_stats)
    if merge_model:
        aman.wrap(sss_model_name, model_sig_tod, [(0, 'dets'), (1, 'samps')])
    
    return sss_stats, model_sig_tod

def subtract_sss(aman, signal=None, sss_template=None,
                 subtract_name='sss_remove'):
    """
    Subtract the scan synchronous signal (SSS) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal to which the SSS template will
        be applied.
    signal : ndarray, optional
        The signal from which the SSS template will be subtracted. If `signal` is
        None (default), the signal contained in the axis manager will be used.
    sss_template : ndarray, optional
        The SSS template to be subtracted from the signal. If `sss_template`
        is None (default), the SSS template stored in the axis manager under
        the key 'sss_extract' will be used.
    subtract_name : str, optional
        The name of the output axis manager field that will contain the SSS-subtracted 
        signal. Defaults to 'sss_remove'.

    Returns
    -------
    None
    """
    if signal is None:
        signal = aman.signal
    if sss_template is None:
        sss_template = aman['sss_model']

    aman.wrap(subtract_name, np.subtract(
              signal, sss_template), [(0, 'dets'), (1, 'samps')])
