"""Module for estimating Azimuth Synchronous Signal (azss)"""
import numpy as np
from operator import attrgetter
from numpy.polynomial import legendre as L
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sotodlib import core, tod_ops
from sotodlib.tod_ops import bin_signal, apodize, filters
from so3g.proj import Ranges, RangesMatrix
import logging

logger = logging.getLogger(__name__)

def bin_by_az(aman, signal=None, az=None, frange=None, bins=100, flags=None, 
              apodize_edges=True, apodize_edges_samps=1600, 
              apodize_flags=True, apodize_flags_samps=200):
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
    frange: array-like, optional
        A list specifying the range of azimuth angles to consider for binning. Defaults to None.
        If None, [min(az), max(az)] will be used for binning.
    bins: int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If ``bins`` is a sequence, ``bins`` overwrite ``frange``.
    flags: RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.
    apodize_edges : bool, optional
        If True, applies an apodization window to the edges of the signal. Defaults to True.
    apodize_edges_samps : int, optional
        The number of samples over which to apply the edge apodization window. Defaults to 1600.
    apodize_flags : bool, optional
        If True, applies an apodization window based on the flags. Defaults to True.
    apodize_flags_samps : int, optional
        The number of samples over which to apply the flags apodization window. Defaults to 200.

    Returns
    -------
    binning_dict: A dictionary containing the binned signal values.
        * 'bin_edges' : float array of bin edges
        * 'bin_centers': center of each bin of azimuth
        * 'bin_counts': counts of binned samples
        * 'binned_signal': binned signal
        * 'binned_signal_sigma': estimated sigma of binned signal
    """
    if apodize_edges:
        weight_for_signal = apodize.get_apodize_window_for_ends(aman, apodize_samps=apodize_edges_samps)
        if (flags is not None) and apodize_flags:
            flags_mask = flags.mask()
            # check the flags dimension
            if flags_mask.ndim == 1:
                flag_is_1d = True
            else:
                all_columns_same = np.all(np.all(flags_mask == flags_mask[0, :], axis=0))
                if all_columns_same:
                    flag_is_1d = True
                    flags_mask = flags_mask[0]
                else:
                    flag_is_1d = False
                    
            if flag_is_1d:
                weight_for_signal = weight_for_signal * apodize.get_apodize_window_from_flags(aman, 
                                                                                              flags=flags,
                                                                                              apodize_samps=apodize_flags_samps)
            else:
                weight_for_signal = weight_for_signal[np.newaxis, :] * apodize.get_apodize_window_from_flags(aman, 
                                                                                                             flags=flags, 
                                                                                                             apodize_samps=apodize_flags_samps)
    else:
        if (flags is not None) and apodize_flags:
            weight_for_signal = apodize.get_apodize_window_from_flags(aman, flags=flags, apodize_samps=apodize_flags_samps)
        else:
            weight_for_signal = None
    binning_dict = bin_signal(aman, bin_by=az, signal=signal,
                              range=frange, bins=bins, flags=flags, weight_for_signal=weight_for_signal)
    return binning_dict

def fit_azss(az, azss_stats, max_mode, fit_range=None):
    """
    Function for fitting Legendre polynomials to signal binned in azimuth.

    Parameters
    ----------
    az: array-like
        azimuth array from main axis manager
    azss_stats: AxisManager
        Axis manager containing binned signal and azimuth used for fitting.
        Created by ``get_azss`` function.
    max_mode: integer
        Highest order Legendre polynomial to include in the fit.
    fit_range: list
        Azimuth range used to renormalized to the [-1,1] range spanned
        by the Legendre polynomials for fitting. Default is the max-min
        span in the ``binned_az`` array passed in via ``azss_stats``.

    Returns
    -------
    azss_stats: AxisManager
        Returns updated azss_stats with added fit information
    model_sig_tod: array-like
        Model fit for each detector size ndets x n_samps
    """
    bin_width = azss_stats.binned_az[1] - azss_stats.binned_az[0]
    m = ~np.isnan(azss_stats.binned_signal[0]) # masks bins without counts
    if np.count_nonzero(m) < max_mode + 1:
        raise ValueError('Number of valid bins is smaller than mode of Legendre function')
    
    if fit_range==None:
        az_min = np.min(azss_stats.binned_az[m]) - bin_width/2
        az_max = np.max(azss_stats.binned_az[m]) + bin_width/2
    else:
        az_min, az_max = fit_range[0], fit_range[1]
    
    x_legendre = ( 2*az - (az_min+az_max) ) / (az_max - az_min)
    x_legendre_bin_centers = ( 2*azss_stats.binned_az - (az_min+az_max) ) / (az_max - az_min)
    x_legendre_bin_centers = np.where(~m, np.nan, x_legendre_bin_centers)
    
    mode_names = []
    for mode in range(max_mode+1):
        mode_names.append(f'legendre{mode}')
    
    coeffs = L.legfit(x_legendre_bin_centers[m], azss_stats.binned_signal[:, m].T, max_mode)
    coeffs = coeffs.T
    binned_model = L.legval(x_legendre_bin_centers, coeffs.T)
    binned_model = np.where(~m, np.nan, binned_model)
    sum_of_squares = np.sum(((azss_stats.binned_signal[:, m] - binned_model[:,m])**2), axis=-1)
    redchi2s = sum_of_squares/azss_stats.uniform_binned_signal_sigma**2 / ( len(x_legendre_bin_centers[m]) - max_mode - 1)
    
    azss_stats.wrap('binned_model', binned_model, [(0, 'dets'), (1, 'bin_az_samps')])
    azss_stats.wrap('x_legendre_bin_centers', x_legendre_bin_centers, [(0, 'bin_az_samps')])
    azss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, core.LabelAxis(name='modes', vals=np.array(mode_names, dtype='<U10')))])
    azss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])
    
    return azss_stats, L.legval(x_legendre, coeffs.T)
    
def _prepare_azss_stats(aman, signal, az, frange=None, bins=100, flags=None, 
                        apodize_edges=True, apodize_edges_samps=40000, apodize_flags=True,
                        apodize_flags_samps=200, method='interpolate', max_mode=None):
    """
    Helper function to collect initial info for azss_stats AxisManager.
    """
    # do binning
    binning_dict = bin_by_az(aman, signal=signal, az=az, frange=frange, bins=bins, flags=flags,
                            apodize_edges=apodize_edges, apodize_edges_samps=apodize_edges_samps,
                            apodize_flags=apodize_flags, apodize_flags_samps=apodize_flags_samps,)
    bin_centers = binning_dict['bin_centers']
    bin_counts = binning_dict['bin_counts']
    binned_signal = binning_dict['binned_signal']
    binned_signal_sigma = binning_dict['binned_signal_sigma']
    uniform_binned_signal_sigma = np.nanmedian(binned_signal_sigma, axis=-1)

    azss_stats = core.AxisManager(aman.dets)
    azss_stats.wrap('binned_az', bin_centers, [(0, core.IndexAxis('bin_az_samps', count=bins))])
    azss_stats.wrap('bin_counts', bin_counts, [(0, 'dets'), (1, 'bin_az_samps')])
    azss_stats.wrap('binned_signal', binned_signal, [(0, 'dets'), (1, 'bin_az_samps')])
    azss_stats.wrap('binned_signal_sigma', binned_signal_sigma, [(0, 'dets'), (1, 'bin_az_samps')])
    azss_stats.wrap('uniform_binned_signal_sigma', uniform_binned_signal_sigma, [(0, 'dets')])
    azss_stats.wrap('method', method)
    if not frange is None:
        azss_stats.wrap('frange_min', frange[0])
        azss_stats.wrap('frange_max', frange[1])
    if max_mode:
        azss_stats.wrap('max_mode', max_mode)
    return azss_stats

def get_azss(aman, signal='signal', az=None, frange=None, bins=100, flags=None,
            apodize_edges=True, apodize_edges_samps=1600, apodize_flags=True, apodize_flags_samps=200,
            apply_prefilt=True, prefilt_cfg=None, prefilt_detrend='linear',
            method='interpolate', max_mode=None, subtract_in_place=False,
            merge_stats=True, azss_stats_name='azss_stats', turnaround_info=None,
            merge_model=True, azss_model_name='azss_model', left_right=False):
    """
    Derive azss (Azimuth Synchronous Signal) statistics and model from the given axismanager data.
    **NOTE:** This function does not modify the ``signal`` unless ``subtract_in_place = True``.

    Parameters
    ----------
    aman: TOD
        core.AxisManager
    signal: array-like, optional
        A numpy array representing the signal to be used for azss extraction. If not provided, the signal is taken from aman.signal.
    az: array-like, optional
        A 1D numpy array representing the azimuth angles. If not provided, the azimuth angles are taken from aman.boresight.az.
    frange: list, optional
        A list specifying the range of azimuth angles to consider for binning. Defaults to [-np.pi, np.pi].
        If None, [min(az), max(az)] will be used for binning.
    bins: int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If `bins` is a sequence, `bins` overwrite `frange`.
    flags : RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.
    apodize_edges : bool, optional
        If True, applies an apodization window to the edges of the signal. Defaults to True.
    apodize_edges_samps : int, optional
        The number of samples over which to apply the edge apodization window. Defaults to 1600.
    apodize_flags : bool, optional
        If True, applies an apodization window based on the flags. Defaults to True.
    apodize_flags_samps : int, optional
        The number of samples over which to apply the flags apodization window. Defaults to 200.
    apply_prefilt : bool, optional
        If True, applies a pre-filter to the signal before azss extraction. Defaults to True.
    prefilt_cfg : dict, optional
        Configuration for the pre-filter. Defaults to {'type': 'sine2', 'cutoff': 0.005, 'trans_width': 0.005}.
    prefilt_detrend : str, optional
        Method for detrending before filtering. Defaults to 'linear'.
    method: str
        The method to use for azss modeling. Options are 'interpolate' and 'fit'. 
        In 'interpolate', binned signal is used directly.
        In 'fit', fitting is applied to the binned signal.
        Defaults to 'interpolate'.
    max_mode: integer, optional
        The number of Legendre modes to use for azss when method is 'fit'. Required when method is 'fit'.
    subtract_in_place: bool
        If True, it subtract the modeled tod from original signal. The aman.signal will be modified.
    merge_stats: boolean, optional
        Boolean flag indicating whether to merge the azss statistics with aman. Defaults to True.
    azss_stats_name: string, optional
        The name to assign to the merged azss statistics. Defaults to 'azss_stats'.
    merge_model: boolean, optional
        Boolean flag indicating whether to merge the azss model with the aman. Defaults to True.
    azss_model_name: string, optional
        The name to assign to the merged azss model. Defaults to 'azss_model'.
    left_right: bool
        Default False. If True estimate (and subtract) the AzSS template for left and right subscans
        separately.
    turnaround_info: FlagManager or AxisManager
        Optional, default is aman.flags.

    Returns
    -------
    Tuple:
        - azss_stats: core.AxisManager
            - azss statistics including: azumith bin centers, bin counts, binned signal, std of each detector-az bin, std of each detector.
            - If ``method=fit`` then also includes: binned legendre model, legendre bin centers, fit coefficients, reduced chi2.
        - model_sig_tod: numpy.array
            - azss model as a function of time either from fits or interpolation depending on ``method`` argument.
    """
    if prefilt_cfg is None:
        prefilt_cfg = {'type': 'sine2', 'cutoff': 0.005, 'trans_width': 0.005}
    prefilt = filters.get_hpf(prefilt_cfg)

    if signal is None:
        #signal_name variable to be deleted when tod_ops.fourier_filter is updated
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if apply_prefilt:
        # This requires signal to be a string.
        signal = np.array(tod_ops.fourier_filter(
                aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name)
                )

    if az is None:
        az = aman.boresight.az

    if flags is None:
        flags = Ranges.from_mask(np.zeros(aman.samps.count).astype(bool))
    else: 
        flags = aman.flags.reduce(flags=flags, method='union', wrap=False)

    if left_right:
        if turnaround_info is None:
            turnaround_info = aman.flags
        if isinstance(turnaround_info, str):
            _f = attrgetter(turnaround_info)
            turnaround_info = _f(aman)
        if not isinstance(turnaround_info, (core.AxisManager, core.FlagManager)):
            raise TypeError('turnaround_info must be AxisManager or FlagManager')

        if "valid_right_scans" not in turnaround_info: 
            left_mask = turnaround_info.left_scan
            right_mask = turnaround_info.right_scan
        else:
            left_mask = turnaround_info.valid_left_scans
            right_mask = turnaround_info.valid_right_scans

        azss_left = _prepare_azss_stats(aman, signal, az, frange, bins, flags+~left_mask, apodize_edges,
                                        apodize_edges_samps, apodize_flags, apodize_flags_samps,
                                        method=method, max_mode=max_mode)
        azss_left.add_axis(aman.samps)
        azss_left.add_axis(aman.dets)
        azss_left.wrap('mask', left_mask, [(0, 'dets'), (1, 'samps')])
        azss_right = _prepare_azss_stats(aman, signal, az, frange, bins, flags+~right_mask, apodize_edges,
                                        apodize_edges_samps, apodize_flags, apodize_flags_samps,
                                        method=method, max_mode=max_mode)
        azss_right.add_axis(aman.samps)
        azss_right.add_axis(aman.dets)
        azss_right.wrap('mask', right_mask, [(0, 'dets'), (1, 'samps')])
        
        azss_stats = core.AxisManager(aman.dets)
        azss_stats.wrap('azss_stats_left', azss_left)
        azss_stats.wrap('azss_stats_right', azss_right)
        azss_stats.wrap('left_right', left_right)
    else:
        azss_stats = _prepare_azss_stats(aman, signal, az, frange, bins, flags, apodize_edges,
                                        apodize_edges_samps, apodize_flags, apodize_flags_samps,
                                        method=method, max_mode=max_mode)
        azss_stats.wrap('left_right', left_right)

    model_left, model_right, model = None, None, None
    if merge_model or subtract_in_place:
        if left_right:
            azss_stats, model_left, model_right = get_model_sig_tod(aman, azss_stats, az)
            if merge_model:
                aman.wrap(azss_model_name+'_left', model_left, [(0, 'dets'), (1, 'samps')])
                aman.wrap(azss_model_name+'_right', model_right, [(0, 'dets'), (1, 'samps')])
            if subtract_in_place:
                if signal_name is None:
                    lmask = left_mask.mask()
                    signal[lmask] -= model_left[lmask].astype(signal.dtype)
                    rmask = right_mask.mask()
                    signal[rmask] -= model_right[rmask].astype(signal.dtype)
                else:
                    lmask = left_mask.mask()
                    aman[signal_name][lmask] -= model_left[lmask].astype(aman[signal_name].dtype)
                    rmask = right_mask.mask()
                    aman[signal_name][rmask] -= model_right[rmask].astype(aman[signal_name].dtype)
        else:
            azss_stats, model, _ = get_model_sig_tod(aman, azss_stats, az)
            if merge_model:
                aman.wrap(azss_model_name, model, [(0, 'dets'), (1, 'samps')])
            if subtract_in_place:
                if signal_name is None:
                    signal -= model.astype(signal.dtype)
                else:
                    aman[signal_name] -= model.astype(aman[signal_name].dtype)

    if merge_stats:
        aman.wrap(azss_stats_name, azss_stats)

    if left_right:
        return azss_stats, model_left, model_right
    else:
        return azss_stats, model

def get_model_sig_tod(aman, azss_stats, az=None):
    """
    Function to return the azss template for subtraction given the azss_stats AxisManager
    """
    if az is None:
        az = aman.boresight.az

    if azss_stats.left_right:
        model = []
        for fld in ['azss_stats_left', 'azss_stats_right']:
            _azss_stats = azss_stats[fld]
            if 'frange_min' in _azss_stats:
                frange = (_azss_stats.frange_min, _azss_stats.frange_max)
            else:
                frange = None
            if _azss_stats.method == 'fit':
                if type(_azss_stats.max_mode) is not int:
                    raise ValueError('max_mode is not provided as integer')

                _azss_stats, _model = fit_azss(az=az, azss_stats=_azss_stats,
                                                   max_mode=_azss_stats.max_mode,
                                                   fit_range=frange)
                azss_stats.wrap(fld, _azss_stats, overwrite=True)
                model.append(_model)

            if _azss_stats.method == 'interpolate':
                good_az = np.logical_and(_azss_stats.binned_az >= np.min(az), _azss_stats.binned_az <= np.max(az))
                f_template = interp1d(_azss_stats.binned_az[good_az],
                                      _azss_stats.binned_signal[:, good_az], fill_value='extrapolate')
                _model = f_template(az)
                model.append(_model)
        for ii in range(len(model)):
            model[ii][~np.isfinite(model[ii])] = 0
        return azss_stats, model[0], model[1]

    else:
        if azss_stats.method == 'fit':
            if type(azss_stats.max_mode) is not int:
                    raise ValueError('max_mode is not provided as integer')
            if 'frange_min' in azss_stats:
                frange = (azss_stats.frange_min, azss_stats.frange_max)
            else:
                frange = None
            azss_stats, model = fit_azss(az=az, azss_stats=azss_stats,
                                         max_mode=azss_stats.max_mode,
                                         fit_range=frange)
        if azss_stats.method == 'interpolate':
            good_az = np.logical_and(azss_stats.binned_az >= np.min(az), azss_stats.binned_az <= np.max(az))
            f_template = interp1d(azss_stats.binned_az[good_az], azss_stats.binned_signal[:, good_az], fill_value='extrapolate')
            model = f_template(az)
            model[~np.isfinite(model)] = 0
        return azss_stats, model, None

def subtract_azss(aman, azss_stats, signal='signal', subtract_name='azss_remove',
                  in_place=False):
    """
    Subtract the scan synchronous signal (azss) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal and the azss template.
    azss_stats: AxisManager
        Contains AxisManager from get_azss.
    signal : str, optional
        The name of the field in the axis manager containing the signal to be processed.
        Defaults to 'signal'.
    subtract_name : str, optional
        The name of the field in the axis manager that will store the azss-subtracted signal.
        Only used if in_place is False. Defaults to 'azss_remove'.
    in_place : bool, optional
        If True, the subtraction is done in place, modifying the original signal in the axis manager.
        If False, the result is stored in a new field specified by subtract_name. Defaults to False.

    Returns
    -------
    None
    """
    if signal is None:
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        if np.shape(signal) != (aman.dets.count, aman.samps.count):
            raise ValueError("When passing signal as ndarray shape must match (n_dets x n_samps).")
        signal_name = None
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    azss_stats, model_left, model_right = get_model_sig_tod(aman, azss_stats, az=None)

    if in_place:
        if signal_name is None:
            if azss_stats.left_right:
                for model, azss_fld in zip([model_left, model_right], ['azss_stats_left', 'azss_stats_right']):
                    mask = azss_stats[azss_fld]['mask'].mask()
                    signal[:, mask] -= model[:, mask].astype(signal.dtype)
            else:
                signal -= model_left.astype(signal.dtype)
        else:
            if azss_stats.left_right:
                for model, azss_fld in zip([model_left, model_right], ['azss_stats_left', 'azss_stats_right']):
                    mask = azss_stats[azss_fld]['mask'].mask()
                    aman[signal_name][mask] -= model[mask].astype(aman[signal_name].dtype)
            else:
                aman[signal_name] -= model_left.astype(aman[signal_name].dtype)
    else:
        if azss_stats.left_right:
            wrap_sig = np.copy(signal)
            for model, azss_fld in zip([model_left, model_right], ['azss_stats_left', 'azss_stats_right']):
                mask = azss_stats[azss_fld]['mask'].mask()
                wrap_sig[:, mask] -= model[:, mask].astype(signal.dtype)
            aman.wrap(subtract_name, wrap_sig, [(0, 'dets'), (1, 'samps')])
        else:
            aman.wrap(subtract_name, 
                      np.subtract(aman[signal_name], model_left, dtype='float32'),
                      [(0, 'dets'), (1, 'samps')])
