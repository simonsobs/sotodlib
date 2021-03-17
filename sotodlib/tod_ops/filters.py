import numpy as np
import pyfftw
import inspect
import scipy.signal as signal

from . import detrend_data
from .fft_ops import build_rfft_object

def fourier_filter(tod, filt_function,
                   detrend='linear', resize='zero_pad',
                   axis_name='samps', signal_name='signal', 
                   time_name='timestamps',
                   **kwargs):
    
    """Return a filtered tod.signal_name along the axis axis_name. 
        Does not change the data in the axis manager.
    
    Arguments:
    
        tod: axis manager
        
        filt_function: function( freqs, tod ) function that takes a set of 
            frequencies and the axis manager and returns the filter in 
            fouier space
        
        detrend: Method of detrending to be done before ffting. Can
            be 'linear', 'mean', or None.
            
        resize: How to resize the axis to increase fft speed. 'zero_pad' 
            will increase to the next 2**N. 'trim' will cut out so the 
            factorization of N contains only low primes. None will not 
            change the axis length and might be quite slow. Trim will be 
            kinda weird here, because signal will not be returned as the same
            size as it is input

        axis_name: name of axis you would like to fft along
        
        signal_name: name of the variable in tod to fft
        
        time_name: name for getting time of data (in seconds) from tod
        
    Returns:
    
        signal: filtered tod.signal_name 
        
    """
    if len(tod._assignments[signal_name]) >2:
        raise ValueError('fouier_filter only works for 1D or 2D data streams')
        
    axis = getattr(tod, axis_name)
    times = getattr(tod, time_name)
    delta_t = (times[-1]-times[0])/axis.count
    
    if len(tod._assignments[signal_name])==1:
        n_det = 1
        ## signal will be at least 2D
        main_idx = 1
        other_idx = None
        
    elif len(tod._assignments[signal_name])==2:
        checks = np.array([x==axis_name for x in tod._assignments[signal_name]],dtype='bool')
        main_idx = np.where(checks)[0][0]
        other_idx = np.where(~checks)[0][0]
        other_axis = getattr(tod, tod._assignments[signal_name][other_idx])
        n_det = other_axis.count
    
    if detrend is None:
        signal = np.atleast_2d(getattr(tod, signal_name))
    else:
        signal = detrend_data(tod, detrend, axis_name=axis_name, 
                             signal_name=signal_name)
    
    if other_idx is not None and other_idx != 0:
        ## so that code can be written always along axis 1
        signal = signal.transpose()
    
    if resize == 'zero_pad':
        k = int(np.ceil(np.log(axis.count)/np.log(2)))
        n = 2**k 
    elif resize == 'trim':
        n = fft.find_inferior_integer(axis.count)
    elif resize is None:
        n = axis.count
    else:
        raise ValueError('resize must be "zero_pad", "trim", or None')

    a, b, t_1, t_2 = build_rfft_object(n_det, n, 'BOTH')
    if resize == 'zero_pad':
        a[:,:axis.count] = signal
        a[:,axis.count:] = 0
    elif resize == 'trim':
        a[:] = signal[:,:n]
    else:
        a[:] = signal[:]
    
    ## FFT Signal
    t_1();
    
    ## Get Filter
    freqs = np.fft.rfftfreq(n, delta_t)
    filt_function.apply(freqs, tod, b, **kwargs)

    ## FFT Back
    t_2();
    
    if resize == 'zero_pad':
        signal = a[:,:axis.count]
    else:
        signal = a[:]   
        
    if other_idx is not None and other_idx != 0:
        return signal.transpose()
    
    return signal

################################################################

# Base class... provides that a * b always returns a FilterChain.
class _chainable:
    @staticmethod
    def _preference(f):
        return getattr(f, 'preference', 'compose')
    def __mul__(self, other):
        return FilterChain([self, other])

class FilterFunc(_chainable):
    """Class to support chaining of Fourier filters.

    FilterFunc.deco may be used to decorate functions with signatures
    like::

       function_name(freqs, tod, *args, **kwargs)

    """
    # Note self._fun is set in the subclass, as a class variable,
    # e.g.: _fun = staticmethod(gaussian_filter)
    _fun_nargs = 2
    preference = 'compose'
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def __call__(self, freqs, tod):
        return self._fun(freqs, tod, *self.args, **self.kwargs)
    def apply(self, freqs, tod, target):
        target *= self(freqs, tod)
    @classmethod
    def deco(cls, fun):
        class filter_func(cls):
            _fun = staticmethod(fun)
        # get help from someone
        filter_func.__doc__ = fun.__doc__
        # get arguments from someone after removing the partial args
        args = list(inspect.signature(fun).parameters.values())[cls._fun_nargs:]
        filter_func.__signature__ = inspect.Signature(parameters=args)
        return filter_func

class FilterApplyFunc(FilterFunc):
    """Class to support chaining of Fourier filters.

    @FilterApplyFunc.deco may be used to decorate functions with signatures
    like:

       function_name(target, freqs, tod, *, **)

    Such filter functions must return a transfer function array, if
    target=None.  But if target is a fourier transform array, the
    function must apply the transfer function to that array and return
    None.

    """
    _fun_nargs = 3
    preference = 'apply'
    def __call__(self, freqs, tod):
        return self._fun(None, freqs, tod, *self.args, **self.kwargs)
    def apply(self, freqs, tod, target):
        return self._fun(target, freqs, tod, *self.args, **self.kwargs)

class FilterChain(_chainable):
    """A chain of Fourier filters."""
    def __init__(self, items):
        super().__init__()
        self.links = []
        for a in items:
            if isinstance(a, FilterChain):
                self.links.extend(a.links)
            else:
                self.links.append(a)

    def __call__(self, freqs, tod):
        # We could do better by handling self.links out of order.
        filt = self.links[0](freqs, tod)
        for f in self.links[1:]:
            if self._preference(f) == 'apply' and filt.ndim == 2:
                f.apply(freqs, tod, filt)
            else:
                _filt = f(freqs, tod)
                # Swap those to help broadcasting work...
                if _filt.ndim > filt.ndim:
                    filt, _filt = _filt, filt
                filt *= _filt
                del _filt
        return filt

    def apply(self, freqs, tod, target):
        filt = None
        for f in self.links:
            if self._preference(f) == 'apply':
                f.apply(freqs, tod, target)
            elif filt is None:
                filt = f(freqs, tod)
            else:
                _filt = f(freqs, tod)
                # Swap those to help broadcasting work...
                if _filt.ndim > filt.ndim:
                    filt, _filt = _filt, filt
                filt *= _filt
                del _filt
        if filt is not None:
            target *= filt

# Alias the decorators...
fft_filter = FilterFunc.deco
fft_apply_filter = FilterApplyFunc.deco


# Filtering Functions
#################
@fft_filter
def low_pass_butter4(freqs, tod, lp_fc=1):
    """ 4th order lowpass filter at with cutoff lp_fc
        """
    b, a = signal.butter(4, 2*np.pi*lp_fc, 'lowpass', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def high_pass_butter4(freqs, tod, hp_fc=1):
    """ 4th order lowpass filter at with cutoff hp_fc
    """
    b, a = signal.butter(4, 2*np.pi*hp_fc, 'highpass', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def tau_filter(freqs, tod, tau_name='timeconst', do_inverse=True):
    """tau_filter is deprecated, use timeconst_filter."""
    logging.warning('tau_filter is deprecated; use timeconst_filter.')
    taus = getattr(tod, tau_name)
    
    filt = 1 + 2.0j*np.pi*taus[:,None]*freqs[None,:]
    if not do_inverse:
        return 1.0/filt
    return filt

@fft_apply_filter
def timeconst_filter(target, freqs, tod, timeconst=None, invert=False):
    """One-pole time constant filter for fourier_filter.

    Builds filter for applying or removing time constants from signal
    data.

    Args:

      timeconst: Array of time constant values (one per detector).
        Alternately, a string indicating what member of tod to use for
        the time constants array.  Defaults to 'timeconst'.
      invert (bool): If true, returns the inverse transfer function,
        to deconvolve the time constants.

    Example::

      # Deconvolve time constants.
      fourier_filter(tod, timeconst_filter(invert=True),
                     detrend='linear', resize='zero_pad')

    """
    if timeconst is None:
        timeconst = 'timeconst'
    if isinstance(timeconst, str):
        timeconst = tod[timeconst]

    if target is None:
        filt = 1 + 2.0j*np.pi*timeconst[:,None]*freqs[None,:]
        if invert:
            return filt
        return 1.0/filt

    # Apply filter directly to FFT in target.
    assert(len(timeconst) == len(target))  # safe zip.
    if invert:
        for tau, dest in zip(timeconst, target):
            dest *= 1.+ 2.j*np.pi*tau*freqs
    else:
        for tau, dest in zip(timeconst, target):
            dest /= 1.+ 2.j*np.pi*tau*freqs

@fft_filter
def timeconst_filter_single(freqs, tod, timeconst, invert=False):
    """One-pole time constant filter for fourier_filter.

    This version accepts a single time constant value, in seconds.  To
    use different time constants for each detector, see
    timeconst_filter.

    Example::

      # Apply a 1ms time constant.
      fourier_filter(tod, timeconst_filter_single(timeconst=0.001),
                     detrend=None, resize='zero_pad')

    """
    if invert:
        return 1. + 2.j * np.pi * timeconst * freqs
    return 1. / (1. + 2.j * np.pi * timeconst * freqs)

@fft_filter
def gaussian_filter(freqs, tod, t_sigma=None, f_sigma=None, gain=1.0, fc=0.):
    """Gaussian filter

    Borrowed from moby2.tod.filters
    """
    if t_sigma is not None and f_sigma is not None:
        raise ValueError("cannot specify both time and frec sigmas. Using t_sigma.")
    if t_sigma is not None:
        sigma = 1.0 / (2*np.pi*t_sigma)
    elif f_sigma is not None:
        sigma = f_sigma
    else:
        sigma = 1.0
    return gain * np.exp(-0.5*(np.abs(freqs)-fc)**2/sigma**2)

@fft_filter
def low_pass_sine2(freqs, tod, lp_fc=1.0, df=0.1):
    """low-pass by sine squared

    Borrowed from moby2.tod.filters
    """
    f = np.abs(freqs)
    filt = np.zeros_like(f)
    filt[f < lp_fc - df/2] = 1.0
    sel = (f > lp_fc - df/2)*(f < lp_fc + df/2)
    filt[sel] = np.sin(np.pi/2*(1 - 1/df*(f[sel] - hp_fc + df/2)))**2
    return filt

@fft_filter
def high_pass_sine2(freqs, tod, hp_fc=1.0, df=0.1):
    """high-pass by sine-squared

    Borrowed from moby2.tod.filters
    """
    f = np.abs(freqs)
    filt = np.zeros_like(f)
    filt[f > hp_fc + df/2] = 1.0
    sel = (f > hp_fc - df/2)*(f < hp_fc + df/2)
    filt[sel] = np.sin(np.pi/2/df*(f[sel] - hp_fc + df/2))**2
    return filt

@fft_filter
def iir_filter(freqs, tod, b=None, a=None, fscale=1., iir_params=None, invert=False):
    """Infinite impulse response (IIR) filter.  This sort of filter is
    used in digital applications as a low-pass filter prior to
    decimation.  The Smurf and MCE readout filters can both be
    expressed in this form.

    Args:
      b: numerator polynomial filter coefficients (z^0,z^1, ...)
      a: denominator coefficients
      fscale: scalar used to compute z = exp(-2j*pi*freqs*fscale).
        This will generally correspond to the sampling frequency of
        the original signal (before decimation).
      iir_params: IIR filter params as described below; or a string
        name under which to look up those params in tod; defaults to
        'iir_params'.  Note that if `a` and `b` are passed explicitly
        then no attempt is made to resolve this argument.
      invert: If true, returns denom/num instead of num/denom.

    Notes:
      The `b` and `a` coefficients are as implemented in
      scipy.signal.freqs, scipy.signal.butter, etc.  The "angular
      frequencies", `w`, are computed as 2*pi*freqs*fscale.

      To pass in all parameters at once, set iir_params (or
      tod[iir_params]) to a (3, n) array.  This will be expanded to
      a=P[0,:], b=P[1,:], fscale=P[2,0].

    """
    if a is None:
        # Get params from TOD?
        if iir_params is None:
            iir_params = 'iir_params'
        if isinstance(iir_params, str):
            iir_params = tod[iir_params]
        a, b, fscale = iir_params  # must be (3, n)
        fscale = fscale[0]
    z = np.exp(-2j*np.pi*freqs * fscale)
    B, A = np.polyval(b[::-1], z), np.polyval(a[::-1], z)
    if invert:
        return A / B
    return B / A
