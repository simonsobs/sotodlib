"""Module for estimating Azimuth Synchronous Signal (azss)"""
import numpy as np
from numpy.polynomial import legendre as L
from scipy.interpolate import interp1d
from sotodlib import core, tod_ops
from sotodlib.tod_ops import bin_signal, apodize, filters, pca
import logging

logger = logging.getLogger(__name__)


def bin_by_az(aman, signal=None, az=None, range=None, bins=100, flags=None,
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
    range: array-like, optional
        A list specifying the range of azimuth angles to consider for binning. Defaults to None.
        If None, [min(az), max(az)] will be used for binning.
    bins: int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If ``bins`` is a sequence, ``bins`` overwrite ``range``.
    flags: RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        If provided by a string, `aman.flags.get(flags)` is used for the flags.
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
        if isinstance(flags, str):
            flags = aman.flags.get(flags)
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
                              range=range, bins=bins, flags=flags, weight_for_signal=weight_for_signal)
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
    model_sig_tod: array-like
        Model fit for each detector size ndets x n_samps
    """
    bin_width = azss_stats.binned_az[1] - azss_stats.binned_az[0]
    m = ~np.isnan(azss_stats.binned_signal[0])  # masks bins without counts
    if np.count_nonzero(m) < max_mode + 1:
        raise ValueError('Number of valid bins is smaller than mode of Legendre function')

    if fit_range is None:
        az_min = np.min(azss_stats.binned_az[m]) - bin_width / 2
        az_max = np.max(azss_stats.binned_az[m]) + bin_width / 2
    else:
        az_min, az_max = fit_range[0], fit_range[1]

    x_legendre = (2 * az - (az_min+az_max)) / (az_max - az_min)
    x_legendre_bin_centers = (2 * azss_stats.binned_az - (az_min+az_max)) / (az_max - az_min)
    x_legendre_bin_centers = np.where(~m, np.nan, x_legendre_bin_centers)

    coeffs = L.legfit(x_legendre_bin_centers[m], azss_stats.binned_signal[:, m].T, max_mode)
    coeffs = coeffs.T
    binned_model = L.legval(x_legendre_bin_centers, coeffs.T)
    binned_model = np.where(~m, np.nan, binned_model)
    sum_of_squares = np.sum(((azss_stats.binned_signal[:, m] - binned_model[:, m])**2), axis=-1)
    redchi2s = sum_of_squares/azss_stats.uniform_binned_signal_sigma**2 / (len(x_legendre_bin_centers[m]) - max_mode - 1)

    mode_names = np.array([f'legendre{mode}' for mode in range(max_mode + 1)], dtype='<U10')
    modes_axis = core.LabelAxis(name='azss_modes', vals=mode_names)
    azss_stats.add_axis(modes_axis)
    azss_stats.wrap('binned_model', binned_model, [(0, 'dets'), (1, 'bin_az_samps')])
    azss_stats.wrap('x_legendre_bin_centers', x_legendre_bin_centers, [(0, 'bin_az_samps')])
    azss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'azss_modes')])
    azss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    return L.legval(x_legendre, coeffs.T)


def get_azss(aman, signal='signal', az=None, range=None, bins=100, flags=None, scan_flags=None,
             apodize_edges=True, apodize_edges_samps=1600, apodize_flags=True, apodize_flags_samps=200,
             apply_prefilt=True, prefilt_cfg=None, prefilt_detrend='linear',
             method='interpolate', max_mode=None, subtract_in_place=False,
             merge_stats=True, azss_stats_name='azss_stats',
             merge_model=True, azss_model_name='azss_model'):
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
    range: list, optional
        A list specifying the range of azimuth angles to consider for binning. Defaults to [-np.pi, np.pi].
        If None, [min(az), max(az)] will be used for binning.
    bins: int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (100, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        If `bins` is a sequence, `bins` overwrite `range`.
    flags : str or Rannges or RangesMatrix, optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        Default is no mask applied.
    scan_flags : str or Ranges, optional
        Subtract in the scan/time region specified by flags.
        Typically `flags.left_scan` or `flags.right_scan` will be used.
        If we estimate and subtract azss in left going scan only, then
        `flags` should be a "union of glitch_flags and flags.right_scan (or ~flags.left_scan)", and
        `scan_flags` should be `flags.left_scan`.
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
        # signal_name variable to be deleted when tod_ops.fourier_filter is updated
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if scan_flags is None:
        scan_flags = np.ones(aman.samps.count, dtype=bool)
    elif isinstance(scan_flags, str):
        scan_flags = aman.flags.get(scan_flags).mask()
    else:
        scan_flags = scan_flags.mask()

    if apply_prefilt:
        # This requires signal to be a string.
        signal = np.array(tod_ops.fourier_filter(
                aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name)
                )

    if az is None:
        az = aman.boresight.az

    # do binning
    binning_dict = bin_by_az(aman, signal=signal, az=az, range=range, bins=bins, flags=flags,
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

    model_sig_tod = get_azss_model(aman, azss_stats, az, method, max_mode, range)

    if merge_stats:
        aman.wrap(azss_stats_name, azss_stats)
    if merge_model:
        aman.wrap(azss_model_name, model_sig_tod, [(0, 'dets'), (1, 'samps')])
    if subtract_in_place:
        aman[signal_name][:, scan_flags] -= model_sig_tod.astype(signal.dtype)[:, scan_flags]
    return azss_stats, model_sig_tod


def get_azss_model(aman, azss_stats, az=None, method='interpolate', max_mode=None, range=None):
    """
    Function to return the azss template for subtraction given the azss_stats AxisManager
    """
    if az is None:
        az = aman.boresight.az

    if method == 'fit':
        if not isinstance(max_mode, int):
            raise ValueError('max_mode is not provided as integer')
        model = fit_azss(az=az, azss_stats=azss_stats, max_mode=max_mode, fit_range=range)

    if method == 'interpolate':
        # mask az bins that has no data and extrapolate
        mask = ~np.any(np.isnan(azss_stats.binned_signal), axis=0)
        f_template = interp1d(azss_stats.binned_az[mask], azss_stats.binned_signal[:, mask], fill_value='extrapolate')
        model = f_template(az)

    if np.any(~np.isfinite(model)):
        logger.warning('azss model has nan. set zero to nan but this may make glitch')
        model[~np.isfinite(model)] = 0
    return model


def subtract_azss(aman, azss_stats, signal='signal', method='interpolate', max_mode=None, range=None,
                  scan_flags=None, subtract_name='azss_remove', in_place=False, remove_template=False):
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
    method: str
        The method to use for azss modeling. Options are 'interpolate' and 'fit'.
    max_mode: integer, optinal
        The number of Legendre modes to use for azss when method is 'fit'. Required when method is 'fit'.
    range:
        Azimuth range used for 'fit' method.
    scan_flags: str or Ranges, optional
        Subtract in the scan/time region specified by flags.
        Typically `flags.left_scan` or `flags.right_scan` will be used.
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

    model = get_azss_model(aman, azss_stats, method=method, max_mode=max_mode, range=range)

    if scan_flags is None:
        scan_flags = np.ones(aman.samps.count, dtype=bool)
    elif isinstance(scan_flags, str):
        scan_flags = aman.flags.get(scan_flags).mask()
    else:
        scan_flags = scan_flags.mask()

    if in_place:
        if signal_name is None:
            signal[:, scan_flags] -= model.astype(signal.dtype)[:, scan_flags]
        else:
            aman[signal_name][:, scan_flags] -= model.astype(aman[signal_name].dtype)[:, scan_flags]
    else:
        subtracted = signal[:, scan_flags] - model.astype(signal.dtype)[:, scan_flags]
        aman.wrap(subtract_name, subtracted, [(0, 'dets'), (1, 'samps')])


def subtract_azss_template(
    aman,
    signal='signal',
    azss='azss_stats',
    method='interpolate',
    scan_flags=None,
    pca_modes=None,
    subtract=True,
):
    """
    Make azss template model that is "common" to all detectors and subtract
    If pca_modes are specified, make template by pca, otherwise make template by weighted mean.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the signal and the azss template.
    signal: str or array-like
        numpy array of signal to be binned. If None, the signal is taken from aman.signal.
    azss: str or azss_stats AxisManager
        azss_stats AxisManager generated by `azss.get_azss`
    method : str
        The method to use for azss modeling. Option is 'interpolate' only now.
    scan_flags: str or flags, optional
        Subtract template model in the time region specified by flags.
        Typically `flags.left_scan` or `flags.right_scan` will be used.
    pca_modes: integer, optinal
        Number of pca modes for making azss template
    subtract: boolean, optional
        If true subtract azss template from signal

    Returns
    -------
    azss_template_model

    """
    if isinstance(signal, str):
        signal = aman.get(signal)
    if isinstance(azss, str):
        azss = aman.get(azss)
    if scan_flags is None:
        scan_flags = np.ones(aman.samps.count, dtype=bool)
    elif isinstance(scan_flags, str):
        scan_flags = aman.flags.get(scan_flags).mask()
    else:
        scan_flags = scan_flags.mask()

    mask = ~np.any(np.isnan(azss.binned_signal), axis=0)

    if pca_modes is not None:
        # make pca model
        template = np.full((azss.dets.count, azss.bin_az_samps.count), np.nan)
        if aman.dets.count < pca_modes:
            raise ValueError(f'The number of pca modes {pca_modes} is '
                             f'larger than the number of detectors {aman.dets.count}.')
        p = pca.get_pca(aman, signal=azss.binned_signal[:, mask])
        R = p.R[:, :pca_modes]
        template[:, mask] = R @ R.T @ azss.binned_signal[:, mask]

    else:
        # make commom mode template
        weighted_azss = np.sum(azss.binned_signal * azss.binned_signal_sigma**-2, axis=0)
        weight = np.sum(azss.binned_signal_sigma**-2, axis=0)
        template = weighted_azss / weight

        # Least square fit the amplitude of azss per each detector
        coefs = np.sum(template[np.newaxis, mask] * azss.binned_signal[:, mask] * azss.binned_signal_sigma[:, mask]**-2, axis=1)
        coefs /= np.sum(template[np.newaxis, mask]**2 * azss.binned_signal_sigma[:, mask]**-2, axis=1)
        template = coefs[:, np.newaxis] * template[np.newaxis, :]

    if method == 'interpolate':
        f_template = interp1d(azss.binned_az[mask], template[:, mask], fill_value='extrapolate')
        model = f_template(aman.boresight.az)
    else:
        raise ValueError(f'{method} is not supported yet')

    if np.any(~np.isfinite(model)):
        logger.warning('azss model has nan. set zero to nan but this may make glitch')
        model[~np.isfinite(model)] = 0

    if subtract:
        signal[:, scan_flags] -= model[:, scan_flags]
    return model
