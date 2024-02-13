"""FFTs and related operations
"""
from scipy.signal import welch
from scipy.optimize import minimize
import pyfftw
import numpy as np
import numdifftools as ndt

import so3g

from sotodlib import core
from . import detrend_tod

def _get_num_threads():
    # Guess how many threads we should be using in FFT ops...
    return so3g.useful_info().get('omp_num_threads', 4)

def rfft(aman, detrend='linear', resize='zero_pad', window=np.hanning,
         axis_name='samps', signal_name='signal', delta_t=None):
    """Return the real fft of aman.signal_name along the axis axis_name. 
        Does not change the data in the axis manager.
    
    Arguments:
    
        aman: axis manager
        
        detrend: Method of detrending to be done before ffting. Can
            be 'linear', 'mean', or None. Note that detrending here can be slow
            for large arrays.
            
        resize: How to resize the axis to increase fft speed. 'zero_pad' 
            will increase to the next 2**N. 'trim' will cut out so the 
            factorization of N contains only low primes. None will not 
            change the axis length and might be quite slow.
            
        window: a function that takes N are returns an fft window
            Can be None if no windowing
        
        axis_name: name of axis you would like to fft along
        
        signal_name: name of the variable in aman to fft
        
        delta_t: if none, it will look for 'timestamps' in the axis manager 
                and will otherwise assume 1. if not None, it should be the 
                sampling rate along the axis you're ffting
        
    Returns:
    
        fft: the fft'd data
        
        freqs: the frequencies it is value at (since resizing is an option)
    """
    if len(aman._assignments[signal_name]) >2:
        raise ValueError('rfft only works for 1D or 2D data streams')
        
    axis = getattr(aman, axis_name)
    
    if len(aman._assignments[signal_name])==1:
        n_det = 1
        main_idx = 0
        other_idx = None
        
    elif len(aman._assignments[signal_name])==2:
        checks = np.array([x==axis_name for x in aman._assignments[signal_name]],dtype='bool')
        main_idx = np.where(checks)[0][0]
        other_idx = np.where(~checks)[0][0]
        other_axis = getattr(aman, aman._assignments[signal_name][other_idx])
        n_det = other_axis.count
    
    if detrend is None:
        signal = np.atleast_2d(getattr(aman, signal_name))
    else:
        signal = detrend_tod(aman, detrend, axis_name=axis_name, 
                             signal_name=signal_name, in_place=True)
    
    if other_idx is not None and other_idx != 0:
        signal = signal.transpose()
     
    if window is not None:
        signal = signal*window(axis.count)[None,:]
    
    
    if resize == 'zero_pad':
        k = int(np.ceil(np.log(axis.count)/np.log(2)))
        n = 2**k 
    elif resize == 'trim':
        n = find_inferior_integer(axis.count)
    elif resize is None:
        n = axis.count
    else:
        raise ValueError('resize must be "zero_pad", "trim", or None')

    a, b, t_fun = build_rfft_object(n_det, n, 'FFTW_FORWARD')
    if resize == 'zero_pad':
        a[:,:axis.count] = signal
        a[:,axis.count:] = 0
    elif resize == 'trim':
        a[:] = signal[:,:n]
    else:
        a[:] = signal[:]
    
    t_fun();
    
    if delta_t is None:
        if 'timestamps' in aman:
            delta_t = (aman.timestamps[-1]-aman.timestamps[0])/axis.count
        else:
            delta_t = 1
    freqs = np.fft.rfftfreq(n, delta_t)
    
    if other_idx is not None and other_idx != 0:
        return b.transpose(), freqs
    
    return b, freqs
    
    
def build_rfft_object(n_det, n, direction='FFTW_FORWARD', **kwargs):
    """Build PyFFTW object for fft-ing
    
    Arguments:
        
        n_det: number of detectors (or just the arr.shape[0] for the 
            array you are going to fft)
            
        n: number of samples in timestream
        
        direction: fft direction. Can be FFTW_FORWARD, FFTW_BACKWARD, or BOTH
        
        kwargs: additional arguments to pass to pyfftw.FFTW
        
    Returns:
        
        a: array for the real valued side of the fft
        
        b: array for the the complex side of the fft
        
        t_fun: function for performing FFT (two are returned if direction=='BOTH')
    """
    fftargs = {'threads': _get_num_threads(), 'flags': ['FFTW_ESTIMATE']}
    fftargs.update(kwargs)
    
    a = pyfftw.empty_aligned((n_det,n), dtype='float32')
    b = pyfftw.empty_aligned((n_det,(n+2)//2), dtype='complex64')
    if direction == 'FFTW_FORWARD':
        t_fun = pyfftw.FFTW(a, b, direction=direction, **fftargs)
    elif direction == 'FFTW_BACKWARD':
        t_fun = pyfftw.FFTW(b, a, direction=direction, **fftargs)
    elif direction == 'BOTH':
        t_1 = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', **fftargs)
        t_2 = pyfftw.FFTW(b, a, direction='FFTW_BACKWARD', **fftargs)
        return a, b, t_1, t_2
    else:
        raise ValueError('direction must be FFTW_FORWARD or FFTW_BACKWARD')

    return a, b, t_fun

def find_inferior_integer(target, primes=[2,3,5,7,11,13]):
    """Find the largest integer less than or equal to target whose prime
    factorization contains only the integers listed in primes.

    """
    p = primes[0]
    n = np.floor(np.log(target) / np.log(p))
    best = p**n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p**n
        best_friend = find_inferior_integer(target / base, primes[1:])
        if (best_friend * base) >= best:
            best = best_friend * base
    return int(best)

def find_superior_integer(target, primes=[2,3,5,7,11,13]):
    """Find the smallest integer less than or equal to target whose prime
    factorization contains only the integers listed in primes.

    """
    p = primes[0]
    n = np.ceil(np.log(target) / np.log(p))
    best = p**n
    if len(primes) == 1:
        return int(best)
    while n > 0:
        n -= 1
        base = p**n
        best_friend = find_superior_integer(target / base, primes[1:])
        if (best_friend * base) <= best:
            best = best_friend * base
    return int(best)

def calc_psd(
    aman, 
    signal=None, 
    timestamps=None, 
    max_samples=2**18,
    prefer='center',
    merge=False, 
    overwrite=True, 
    **kwargs
):
    """Calculates the power spectrum density of an input signal using signal.welch()
    Data defaults to aman.signal and times defaults to aman.timestamps
        
    Arguments:
        aman (AxisManager): with (dets, samps) OR (channels, samps)axes.
        signal (float ndarray): data signal to pass to scipy.signal.welch()
        timestamps (float ndarray): timestamps associated with the data signal         
        max_samples (int): maximum samples along sample axis to send to welch
        prefer (str): One of ['left', 'right', 'center'], indicating what
            part of the array we would like to send to welch if cuts are
            required
        merge (bool): if true merge results into axismanager
        overwrite (bool): if true will overwrite f, pxx axes.
        **kwargs: keyword args to be passed to signal.welch()

    Returns:
        freqs: array of frequencies corresponding to PSD calculated from welch
        Pxx: array of PSD values 
    """
    if signal is None:
        signal = aman.signal
    if timestamps is None:
        timestamps = aman.timestamps
    
    n_samps = signal.shape[-1]
    if n_samps <= max_samples:
        start = 0 
        stop = n_samps
    else:
        offset = n_samps-max_samples
        if prefer == 'left':
            offset=0
        elif prefer == 'center':
            offset //= 2
        elif prefer == 'right':
            pass
        else:
            raise ValueError(f"Invalid choise prefer='{prefer}'")
        start = offset
        stop = offset + max_samples
    
            
    freqs, Pxx = welch( 
        signal[:,start:stop], 
        1/np.nanmedian(np.diff(timestamps[start:stop])), 
        **kwargs
    )
    if merge:
        aman.merge( core.AxisManager(core.OffsetAxis("fsamps", len(freqs))))
        if overwrite:
            if "freqs" in aman._fields:
                aman.move("freqs", None)
            if "Pxx" in aman._fields:
                aman.move("Pxx", None)
        aman.wrap("freqs", freqs, [(0,"fsamps")])
        aman.wrap("Pxx", Pxx, [(0,"dets"),(1,"fsamps")])
    return freqs, Pxx

def calc_wn(aman, pxx=None, freqs=None, low_f=5, high_f=10):
    """
    Function that calculates the white noise level as a median PSD value between
    two frequencies. Defaults to calculation of white noise between 5 and 10Hz.
    Defaults frequency information to a wrapped "freqs" field in aman.
    
    Arguments
    ---------
        aman (AxisManager): 
            Uses aman.freq as frequency information associated with the PSD, pxx.
        
        pxx (Float array): 
            Psd information to calculate white noise. Defaults to aman.pxx
        
        freqs (1d Float array): 
            frequency information related to the psd. Defaults to aman.freqs
        
        low_f (Float): 
            low frequency cutoff to calculate median psd value. Defaults to 5Hz
        
        high_f (float): 
            high frequency cutoff to calculate median psd value. Defaults to 10Hz
    
    Returns
    -------
        wn: Float array of white noise levels for each psd passed into argument.
    """
    if freqs is None:
        freqs = aman.freqs

    if pxx is None:
        pxx = aman.Pxx
    
    fmsk = np.all([freqs >= low_f, freqs <= high_f], axis=0 )
    if pxx.ndim == 1:
        wn2 = np.median(pxx[fmsk])
    else:
        wn2 = np.median(pxx[:,fmsk], axis=1)
    
    wn = np.sqrt(wn2)
    return wn

def noise_model(f, p):
    """
    Noise model for power spectrum with white noise, and 1/f noise.
    """
    fknee, w, alpha = p[0], p[1], p[2]
    return w*(1 + (fknee/f)**alpha)

def neglnlike(params, x, y):
    model = noise_model(x, params)
    output = np.sum(np.log(model) + y/model)
    if not np.isfinite(output):
        return 1.0e30
    return output

def fit_noise_model(aman, signal=None, f=None, pxx=None, psdargs=None,
                    fwhite=(10,100), lowf=1, merge_fit=False,
                    f_max = 100, merge_name='noise_fit_stats', 
                    merge_psd=True):
    """
    Fits noise model with white and 1/f noise to the PSD of signal.
    This uses a MLE method that minimizes a log likelihood. This is
    better for chi^2 distributed data like the PSD.

    Reference: http://keatonb.github.io/archivers/powerspectrumfits

    Args
    ----
    aman : AxisManager
        Axis manager which has samps axis aligned with signal.
    signal : nparray
        Signal sized ndets x nsamps to fit noise model to.
        Default is None which corresponds to aman.signal.
    f : nparray
        Frequency of PSD of signal.
        Default is None which calculates f, pxx from signal.
    pxx : nparray
        PSD sized ndets x len(f) which is fit to with model.
        Default is None which calculates f, pxx from signal.
    psdargs : dict
        Dictionary of optional argument for ``scipy.signal.welch``
    fwhite : tuple
        Low and high frequency used to estimate white noise for initial
        guess passed to ``scipy.signal.curve_fit``.
    lowf : tuple
        Frequency below which estimate of 1/f noise index and knee are estimated
        for initial guess passed to ``scipy.signal.curve_fit``.
    merge_fit : bool
        Merges fit and fit statistics into input axis manager.
    f_max : float
        Maximum frequency to include in the fitting. This is particularly
        important for lowpass filtered data such as that post demodulation
        if the data is not downsampled after lowpass filtering.
    merge_name : bool
        If ``merge_fit`` is True then addes into axis manager with merge_name.
    merge_psd : bool
        If ``merg_psd`` is True then adds fres and Pxx to the axis manager.
    Returns
    -------
    noise_fit_stats : AxisManager
        If merge_fit is False then axis manager with fit and fit statistics
        is returned otherwise nothing is returned and axis manager is wrapped
        into input aman.
    """
    if signal is None:
        signal = aman.signal

    if f is None or pxx is None:
        if psdargs is None:
            f, pxx = calc_psd(aman, signal=signal, timestamps=aman.timestamps,
                              merge=merge_psd)
        else:
            f, pxx = calc_psd(aman, signal=signal, timestamps=aman.timestamps,
                              merge=merge_psd, **psdargs)
    eix = np.argmin(np.abs(f-f_max))
    f = f[1:eix]
    pxx = pxx[:,1:eix]

    fitout = np.zeros((aman.dets.count, 3))
    # This is equal to np.sqrt(np.diag(cov)) when doing curve_fit
    covout = np.zeros((aman.dets.count, 3, 3))
    for i in range(aman.dets.count):
        p = pxx[i]
        wnest = np.median(p[((f>fwhite[0]) & (f<fwhite[1]))])
        pfit = np.polyfit(np.log10(f[f<lowf]), np.log10(p[f<lowf]), 1)
        fidx = np.argmin(np.abs(10**np.polyval(pfit, np.log10(f)) - wnest))
        p0 = [f[fidx], wnest, -pfit[0]]
        res = minimize(neglnlike, p0, args=(f, p), method='Nelder-Mead')
        try:
            Hfun = ndt.Hessian(lambda params : neglnlike(params, f, p), full_output=True)
            hessian_ndt, _ = Hfun(res['x'])
            # Inverse of the hessian is an estimator of the covariance matrix
            # sqrt of the diagonals gives you the standard errors.
            covout[i] = np.linalg.inv(hessian_ndt)
        except np.linalg.LinAlgError:
            print(f'Cannot calculate Hessian for detector {aman.dets.vals[i]} skipping.')
            covout[i] = np.full((3,3), np.nan)
        fitout[i] = res.x


    noise_model_coeffs = ['fknee', 'white_noise', 'alpha']
    noise_fit_stats = core.AxisManager(aman.dets, core.LabelAxis(
        name='noise_model_coeffs', vals=np.array(noise_model_coeffs, dtype='<U8')))
    noise_fit_stats.wrap('fit', fitout, [(0, 'dets'), (1, 'noise_model_coeffs')])
    noise_fit_stats.wrap('cov', covout, [(0, 'dets'), (1, 'noise_model_coeffs'),
                                         (2, 'noise_model_coeffs')])

    if merge_fit:
        aman.wrap(merge_name, noise_fit_stats)
    return noise_fit_stats
