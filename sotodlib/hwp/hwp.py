import numpy as np
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
from sotodlib.tod_ops import bin_signal, filters, apodize
import logging

logger = logging.getLogger(__name__)


def get_hwpss(aman, signal_name=None, hwp_angle=None, bin_signal=True, bins=360,
              lin_reg=True, modes=[1, 2, 3, 4, 5, 6, 7, 8], apply_prefilt=True,
              prefilt_cfg=None, prefilt_detrend='linear', flags=None,
              apodize_edges=True, apodize_edges_samps=1600, 
              apodize_flags=True, apodize_flags_samps=200,
              merge_stats=True, hwpss_stats_name='hwpss_stats',
              merge_model=True, hwpss_model_name='hwpss_model'):
    """
    Extracts HWP synchronous signal (HWPSS) from a time-ordered data (TOD)
    using linear regression or curve-fitting. The curve-fitting or linear
    regression are either run on the full time ordered data vs hwp angle
    or the time ordered data binned in hwp_angle. If the curve-fitting
    option is used it must be performed on the binned data.

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract HWPSS from.
    signal_name : str
        The field name in the axis manager to use for the TOD signal.
        If not provided, ``signal`` will be used.
    hwp_angle : array-like, optional
        The HWP angle for each sample in `aman`. If not provided, `aman.hwp_angle` will be used.
    bin_signal : bool, optional
        Whether to bin the TOD signal into HWP angle bins before extracting HWPSS. Default is `True`.
    bins : int, optional
        The number of HWP angle bins to use if `bin_signal` is `True`. Default is 360.
    lin_reg : bool, optional
        Whether to use linear regression to extract HWPSS from the binned signal. If `False`, curve-fitting will be used instead.
        Default is `True`.
    modes : list of int, optional
        The HWPSS harmonic modes to extract. Default is [1, 2, 3, 4, 6, 8].
    apply_prefilt : bool, optional
        Whether to apply a high-pass filter to signal before extracting HWPSS. Default is `True`.
        If run through preprocess and `signal` is not `aman.signal` then default to `False`.
    prefilt_cfg : dict, optional
        The configuration of the high-pass filter, in Hz. Only used if `apply_prefilt` is `True`.
        Default is sine2 filter of with cutoff frequency of 1.0 Hz and trans_width of 1.0 Hz.
    prefilt_detrend: str or None
        Method of detrending when you apply prefilter. Default is `linear`. If data is already detrended or you do not want to detrend,
        set it to `None`.
    flags : RangesMatrix, optional
        Flags to be masked out before extracting HWPSS. If Default is None, and no mask will be applied.
    merge_stats : bool, optional
        Whether to add the extracted HWPSS statistics to `aman` as new axes. Default is `True`.
    hwpss_stats_name : str, optional
        The name to use for the new field containing the HWPSS statistics if `merge_stats` is `True`. Default is 'hwpss_stats'.
    merge_extract : bool, optional
        Whether to add the extracted HWPSS to `aman` as a new signal field. Default is `True`.
    hwpss_extract_name : str, optional
        The name to use for the new signal field containing the extracted HWPSS if `merge_extract` is `True`. Default is 'hwpss_extract'.

    Returns
    -------
    hwpss_stats : AxisManager object
        The extracted HWPSS and its statistics. The statistics include:

            - **coeffs** (n_dets x n_modes) : coefficients of the model

            .. math::
                \sum_n \mathrm{coeffs}[2n]\sin{(\mathrm{modes}[n] \chi_{\mathrm{hwp}})} + \mathrm{coeffs}[2n+1]\cos{(\mathrm{modes}[n] \chi_{\mathrm{hwp}})}

            where the sum on n range(len(modes)). **Note**: n_modes is 2*len(modes)

            - **covars** (n_dets x n_modes x n_modes) : variance covariance matrix of the fitted coefficients for each detector.
            - **redchi2** (n_dets) : reduced chi^2 of the fit for each detector.
        
            **In the binned case the following are returned:**
            
            - **binned_angle** (n_bins) : binned version of hwp_angle in range (0, 2pi] with number of bins set by bins argument.
            - **bin_counts** (n_dets x n_bins): sample counts of each bin for each detector.
            - **binned_signal** (n_dets x n_bins) : binned signal for each detector.
            - **sigma_bin** (n_dets) : average over all bins of the standard deviation of the signal within each bin.
        
            **In the non-binned case the following are returned:**

            - **sigma_tod** (n_dets) : estimate of the standard deviation of the signal using function ``estimate_sigma_tod``
    """
    if prefilt_cfg is None:
        prefilt_cfg = {'type': 'sine2', 'cutoff': 1.0, 'trans_width': 1.0}

    prefilt = filters.get_hpf(prefilt_cfg)

    if signal_name is None:
        if apply_prefilt:
            signal = np.array(tod_ops.fourier_filter(
                aman, prefilt, detrend=prefilt_detrend, signal_name='signal'))
        else:
            signal = aman.signal
    else:
        if apply_prefilt:
            signal = np.array(tod_ops.fourier_filter(
                aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name))
        else:
            signal = aman[signal_name]

    if hwp_angle is None:
        hwp_angle = aman.hwp_angle

    # define hwpss_stats
    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    hwpss_stats = core.AxisManager(aman.dets, core.LabelAxis(
        name='modes', vals=np.array(mode_names, dtype='<U3')))
    if bin_signal:
        hwp_angle_bin_centers, bin_counts, binned_hwpss, hwpss_sigma_bin = get_binned_hwpss(
            aman, signal, hwp_angle=None, bins=bins, flags=flags, 
            apodize_edges=apodize_edges, apodize_edges_samps=apodize_edges_samps, 
            apodize_flags=apodize_flags, apodize_flags_samps=apodize_flags_samps,)
        
        # check bin count
        num_invalid_bins = np.count_nonzero(np.isnan(binned_hwpss[0][:]))
        if num_invalid_bins > 0:
            logger.warning(f'There are {num_invalid_bins} bins with zero samples. ' + 
                             'You maybe using simulation data whose hwp speed is perfectly constant, ' + 
                             'or your specification of number of bins is too large.')
        
        # wrap
        hwpss_stats.wrap('binned_angle', hwp_angle_bin_centers, [
                       (0, core.IndexAxis('bin_samps', count=bins))])
        hwpss_stats.wrap('bin_counts', bin_counts, [
                       (0, 'dets'), (1, 'bin_samps')])
        hwpss_stats.wrap('binned_signal', binned_hwpss, [
                       (0, 'dets'), (1, 'bin_samps')])
        hwpss_stats.wrap('sigma_bin', hwpss_sigma_bin, [(0, 'dets')])

        if lin_reg:
            fitsig_binned, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin, modes=modes)
        else:
            params_init = guess_hwpss_params(
                x=hwp_angle_bin_centers, ys=binned_hwpss, modes=modes)
            fitsig_binned, coeffs, covars, redchi2s = hwpss_curvefit(x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin,
                                                                     modes=modes, params_init=params_init)
        # tod template
        fitsig_tod = harms_func(hwp_angle, modes, coeffs)

        # wrap the optimal values and stats
        hwpss_stats.wrap('binned_model', fitsig_binned,
                       [(0, 'dets'), (1, 'bin_samps')])
        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    else:
        if flags is None:
            m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
        else:
            m = ~flags.mask()

        hwpss_sigma_tod = estimate_sigma_tod(signal, hwp_angle)
        hwpss_stats.wrap('sigma_tod', hwpss_sigma_tod, [(0, 'dets')])

        if lin_reg:
            fitsig_tod, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle, ys=signal, yerrs=hwpss_sigma_tod, modes=modes)

        else:
            raise ValueError('Curve-fitting for TOD are specified.' +
                             'It will take too long time and return meaningless result.' +
                             'Specify (bin_signal, lin_reg) = (True, True) or (True, False) or (False, True)')

        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])
    
    if merge_stats:
        aman.wrap(hwpss_stats_name, hwpss_stats)
    if merge_model:
        aman.wrap(hwpss_model_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return hwpss_stats


def get_binned_hwpss(aman, signal=None, hwp_angle=None,
                     bins=360, flags=None, 
                     apodize_edges=True, apodize_edges_samps=1600,
                     apodize_flags=True, apodize_flags_samps=200):
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
    flags : None or RangesMatrix
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
    aman_proc:
        The AxisManager object which contains
        * center of each bin of hwp_angle
        * binned hwp synchrounous signal
        * estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if hwp_angle is None:
        hwp_angle = aman['hwp_angle']
        
    if apodize_edges:
        weight_for_signal = apodize.get_apodize_window_for_ends(aman, apodize_samps=apodize_edges_samps)
        if (flags is not None) and apodize_flags:
            flag_mask = flags.mask()
            if flag_mask.ndim == 1:
                flag_is_1d = True
            else:
                all_columns_same = np.all(np.all(flags_mask == flags_mask[0, :], axis=0))
                if all_columns_same:
                    flag_is_1d = True
                    flag_mask = flags_mask[0]
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
    
    binning_dict = bin_signal(aman, bin_by=hwp_angle, range=[0, 2*np.pi],
                              bins=bins, signal=signal, flags=flags, weight_for_signal=weight_for_signal)
    
    bin_centers = binning_dict['bin_centers']
    bin_counts = binning_dict['bin_counts']
    binned_hwpss = binning_dict['binned_signal']
    binned_hwpss_sigma = binning_dict['binned_signal_sigma']
    
    # use median of sigma of each bin as uniform sigma for a detector
    hwpss_sigma = np.nanmedian(binned_hwpss_sigma, axis=-1)
    
    return bin_centers, bin_counts, binned_hwpss, hwpss_sigma


def hwpss_linreg(x, ys, yerrs, modes):
    """
    Performs a linear regression of the input data ys as a function of x, using a set of sine and cosine
    basis functions defined by the input modes. Returns the fitted signal, the coefficients of the
    basis functions, their covariance matrix, and the reduced chi-square.

    Parameters
    -----------
    x : numpy.ndarray
        The independent variable values of the data points to fit.
    ys : numpy.ndarray
        The dependent variable values of the data points to fit.
    yerrs : numpy.ndarray
        The error estimates of the dependent variable values.
    modes : list of int
        The frequencies of the sine and cosine basis functions to use.

    Returns
    -------
    fitsig : numpy.ndarray
        The fitted signal, obtained by evaluating the model with the optimal coefficients.
    coeffs : numpy.ndarray
        The coefficients of the sine and cosine basis functions that best fit the data.
    covars : numpy.ndarray
        The covariance matrix of the coefficients, estimated from the data errors.
    redchi2s : numpy.ndarray
        The reduced chi-square statistic of the fit, computed for each data point.
    """
    m = np.isnan(ys[0][:])
    xn = np.copy(x)
    x = x[~m]
    ys = ys[:, ~m]

    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(ys, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    fitsig = harms_func(x, modes, coeffs)

    # covariance of coefficients
    covars = np.zeros((ys.shape[0], 2*len(modes), 2*len(modes)))
    for det_idx in range(ys.shape[0]):
        covars[det_idx, :, :] = I * yerrs[det_idx]**2

    # reduced chi-square
    redchi2s = np.sum(
        ((ys - fitsig)/yerrs[:, np.newaxis])**2, axis=-1) / (x.shape[0] - 2*len(modes))

    fitsig = harms_func(xn, modes, coeffs)
    return fitsig, coeffs, covars, redchi2s


def harms_func(x, modes, coeffs):
    """
    calculates the harmonics function given the input values, modes and coefficients.

    Parameters
    ----------
    x (numpy.ndarray): Input values
    modes (list): List of modes to be used in the harmonics function
    coeffs (numpy.ndarray): Coefficients of the harmonics function

    Returns
    -------
    numpy.ndarray: The calculated harmonics function.
    """
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    if coeffs is None:
        return vects
    else:
        harmonics = np.matmul(coeffs, vects)
        return harmonics


def guess_hwpss_params(x, ys, modes):
    """
    Compute initial guess for the coefficients of a harmonics-based fit to data.

    Parameters
    ----------
    x : array-like of shape (nsamps,)
    ys : array-like of shape (ndets, nsamps)
    modes : array-like of shape (nmodes,)
        List of modes to use in the fit.

    Returns
    -------
    Params_init : ndarray of shape (m, 2*p)
        Initial guess for the coefficients of a harmonics-based fit to the data.
    """
    m = np.isnan(ys[0][:])
    x = x[~m]
    ys = ys[:, ~m]

    vects = harms_func(x, modes, coeffs=None)
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)
    params_init = 2 * np.matmul(ys, vects.T) / x.shape[0]
    return params_init

def wrapper_harms_func(x, modes, *args):
    """
    A wrapper function for the harmonics function to be used for fitting data using Scipy's curve-fitting algorithm.
    Parameters
    ----------
    x : array-like
        The x-values of the data points to be fitted.
    modes : array-like
        An array of integers representing the modes of the harmonics function.
    *args : tuple
        A tuple of arguments. The first argument should be an array of coefficients used to calculate the harmonics function.
    Returns
    -------
    y : array-like
        An array of the same length as x representing the values of the harmonics function evaluated at x using the given 
        modes and coefficients.
    """
    coeffs = np.array(args[0])
    return harms_func(x, modes, coeffs)

def hwpss_curvefit(x, ys, yerrs, modes, params_init=None):
    """
    Fit harmonics to input data using scipy's curve_fit method.

    Parameters
    ----------
    x : array_like
        1-D array of x values.
    ys : array_like
        2-D array of y values for each detector.
    yerrs : array_like
        1-D array of the standard deviation of the y values for each detector.
    modes : array_like
        1-D array of mode numbers to be fitted.
    params_init : array_like, optional
        2-D array of initial parameter values for each detector. Default is None.

    Returns
    -------
    fitsig : ndarray
        2-D array of the fitted values for each detector.
    coeffs : ndarray
        2-D array of the fitted coefficients for each detector.
    covars : ndarray
        3-D array of the covariance matrix of the fitted coefficients for each detector.
    redchi2s : ndarray
        1-D array of the reduced chi-square values for each detector.

    Notes
    -----
    This function fits a set of harmonic functions to the input data using scipy's curve_fit method.
    The `modes` parameter specifies the mode numbers to be fitted.
    The `params_init` parameter can be used to provide initial guesses for the fit parameters.
    """
    N_dets = ys.shape[0]
    N_samps = ys.shape[-1]
    N_modes = len(modes)
    
    # Handle binned data w/ 0 counts in a bin.
    m = np.isnan(ys[0][:])
    xn = np.copy(x)
    x = x[~m]
    ys = ys[:, ~m]
  
    if params_init is None:
        params_init = np.zeros((N_dets, 2*N_modes))

    coeffs = np.zeros((N_dets, 2*len(modes)))
    covars = np.zeros((N_dets, 2*len(modes), 2*len(modes)))
    redchi2s = np.zeros(N_dets)
    fitsig = np.zeros((N_dets, N_samps))

    for det_idx in range(N_dets):
        p0 = params_init[det_idx]
        coeff, covar = curve_fit(lambda x, *p0: wrapper_harms_func(x, modes, p0),
                                 x, ys[det_idx], p0=p0, sigma=yerrs[det_idx] *
                                 np.ones_like(ys[det_idx]),
                                 absolute_sigma=True)

        coeffs[det_idx, :] = coeff
        covars[det_idx, :] = covar

        yfit = harms_func(x, modes, coeff)
        fitsig[det_idx, :] = harms_func(xn, modes, coeff)
        redchi2s[det_idx] = np.sum(
            ((ys[det_idx] - yfit) / yerrs[det_idx])**2) / (x.shape[0] - 2*len(modes))

    return fitsig, coeffs, covars, redchi2s


def estimate_sigma_tod(signal, hwp_angle):
    """
    Estimate the noise level of a signal in a time-ordered data (TOD) using a half-wave plate (HWP) modulation.

    Parameters
    ----------
    signal : ndarray
        A 2D numpy array of shape (n_dets, n_samps) containing the TOD of each detector.
    hwp_angle : ndarray
        A 1D numpy array containing the HWP angles in degrees.

    Returns
    -------
    hwpss_sigma_tod : ndarray
        A 1D numpy array containing the estimated noise level for each detector.

    Notes
    -----
    This function computes the mean of the signal in each period of HWP rotation and multiplies it
    by the square root of the number of samples in that period. The standard deviation of the
    resulting values for all periods is then computed and returned as the estimated sigma of each data point.
    """
    hwp_zeros_idxes = np.where(np.abs(np.diff(hwp_angle)) > 5)[0][:] + 1
    hwpss_sigma_tod = np.zeros((signal.shape[0], hwp_zeros_idxes.shape[0] - 1))

    for i, (init_idx, end_idx) in enumerate(zip(hwp_zeros_idxes[:-1], hwp_zeros_idxes[1:])):
        hwpss_sigma_tod[:, i] = np.mean(
            signal[:, init_idx:end_idx], axis=-1) * np.sqrt(end_idx - init_idx)
    hwpss_sigma_tod = np.std(hwpss_sigma_tod, axis=-1)
    return hwpss_sigma_tod


def subtract_hwpss(aman, signal_name='signal', hwpss_template_name='hwpss_model',
                   subtract_name='hwpss_remove', in_place=False, remove_template=True):
    """
    Subtract the half-wave plate synchronous signal (HWPSS) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal and the HWPSS template.
    signal_name : str, optional
        The name of the field in the axis manager containing the signal to be processed.
        Defaults to 'signal'.
    hwpss_template_name : str, optional
        The name of the field in the axis manager containing the HWPSS template.
        Defaults to 'hwpss_model'.
    subtract_name : str, optional
        The name of the field in the axis manager that will store the HWPSS-subtracted signal.
        Only used if in_place is False. Defaults to 'hwpss_remove'.
    in_place : bool, optional
        If True, the subtraction is done in place, modifying the original signal in the axis manager.
        If False, the result is stored in a new field specified by subtract_name. Defaults to False.
    remove_template : bool, optional
        If True, the HWPSS template field is removed from the axis manager after subtraction.
        Defaults to True.

    Returns
    -------
    None
    """
    if in_place:
        aman[signal_name] = np.subtract(aman[signal_name], aman[hwpss_template_name], dtype='float32')
    else:
        if subtract_name in aman._fields:
            aman[subtract_name] = np.subtract(aman[signal_name], aman[hwpss_template_name], dtype='float32')
        else:
            aman.wrap(subtract_name, np.subtract(
                    aman[signal_name], aman[hwpss_template_name], dtype='float32'),
                    [(0, 'dets'), (1, 'samps')])
    
    if remove_template:
        aman.move(hwpss_template_name, None)


def demod_tod(aman, signal_name='signal', demod_mode=4,
              bpf_cfg=None, lpf_cfg=None):
    """
    Demodulate TOD based on HWP angle

    Parameters
    ----------
    aman : AxisManager
        The AxisManager object
    signal_name : str, optional
        Axis name of the demodulated signal in aman. Default is 'signal'.
    demod_mode : int, optional
        Demodulation mode. Default is 4.
    bpf_cfg : dict
        Configuration for Band-pass filter applied to the TOD data before demodulation.
        If not specified, a 4th-order Butterworth filter of 
        (demod_mode * HWP speed) +/- 0.95*(HWP speed) is used.
        Example) bpf_cfg = {'type': 'butter4', 'center': 8.0, 'width': 3.8}
        See filters.get_bpf for details.
    lpf_cfg : dict
        Configuration for Low-pass filter applied to the demodulated TOD data. If not specified,
        a 4th-order Butterworth filter with a cutoff frequency of 0.95*(HWP speed)
        is used.
        Example) lpf_cfg = {'type': 'butter4', 'cutoff': 1.9}
        See filters.get_lpf for details.

    Returns
    -------
    None
        The demodulated TOD data is added to the input `aman` container as new signals:
        'dsT' for the original signal filtered with `lpf`, 'demodQ' for the demodulated
        signal real component filtered with `lpf` and multiplied by 2, and 'demodU' for
        the demodulated signal imaginary component filtered with `lpf` and multiplied by 2.

    """
    # HWP speed in Hz
    speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
            (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
    
    if bpf_cfg is None:
        bpf_center = demod_mode * speed
        bpf_width = speed * 2. * 0.95
        bpf_cfg = {'type': 'sine2',
                   'center': bpf_center,
                   'width': bpf_width,
                   'trans_width': 0.1}
    bpf = filters.get_bpf(bpf_cfg)
    
    if lpf_cfg is None:
        lpf_cutoff = speed * 0.95
        lpf_cfg = {'type': 'sine2',
                   'cutoff': lpf_cutoff,
                   'trans_width': 0.1}
    lpf = filters.get_lpf(lpf_cfg)
        
    phasor = np.exp(demod_mode * 1.j * aman.hwp_angle)
    demod = tod_ops.fourier_filter(aman, bpf, detrend=None,
                                   signal_name=signal_name) * phasor
    
    # dsT
    aman.wrap_new('dsT', dtype='float32', shape=('dets', 'samps'))
    aman.dsT = aman[signal_name]
    aman['dsT'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='dsT', detrend=None)
    # demodQ
    aman.wrap_new('demodQ', dtype='float32', shape=('dets', 'samps'))
    aman['demodQ'] = demod.real
    aman['demodQ'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='demodQ', detrend=None) * 2.
    # demodU
    aman.wrap_new('demodU', dtype='float32', shape=('dets', 'samps'))
    aman['demodU'] = demod.imag
    aman['demodU'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='demodU', detrend=None) * 2.
