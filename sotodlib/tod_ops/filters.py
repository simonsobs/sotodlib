import numpy as np
import pyfftw
import inspect
import scipy.signal as signal
from operator import attrgetter

import logging

from . import detrend_tod
from . import fft_ops
from sotodlib import core

logger = logging.getLogger(__name__)


def fourier_filter(tod, filt_function,
                   detrend=None, resize='zero_pad',
                   axis_name='samps', signal_name='signal', 
                   time_name='timestamps',
                   **kwargs):
    """Return a filtered tod.signal_name along the axis axis_name. 
        Does not change the data in the axis manager.
    
    Arguments:
    
        tod: axis manager
        
        filt_function: function( freqs, tod ) function that takes a set of 
            frequencies and the axis manager and returns the filter in 
            fourier space
        
        detrend: Method of detrending to be done before ffting. Can
            be 'linear', 'mean', or None. Note that detrending here can be slow
            for large arrays
            
        resize: How to resize the axis to increase fft
            speed. 'zero_pad' will increase to the next nice number (a
            product of small primes compatible with the FFT
            implementation).  'trim' will eliminate samples from the
            end so that axis has a nice length for FFTs.  None will not
            change the axis length and might be quite slow. Trim will
            be kinda weird here, because signal will not be returned
            as the same size as it is input

        axis_name: name of axis you would like to fft along
        
        signal_name: name of the variable in tod to fft
        
        time_name: name for getting time of data (in seconds) from tod
        
    Returns:
    
        signal: filtered tod.signal_name 
        
    """
    if len(tod._assignments[signal_name]) >2:
        raise ValueError('fourier_filter only works for 1D or 2D data streams')
        
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
    
    if resize == 'zero_pad':
        n = fft_ops.find_superior_integer(axis.count)
        logger.info('fourier_filter: padding %i -> %i' % (axis.count, n))
    elif resize == 'trim':
        n = fft_ops.find_inferior_integer(axis.count)
        logger.info('fourier_filter: trimming %i -> %i' % (axis.count, n))
    elif resize is None:
        n = axis.count
    else:
        raise ValueError('resize must be "zero_pad", "trim", or None')

    if detrend is not None:
        logger.info('fourier_filter: detrending.')
        signal = detrend_tod(tod, detrend, axis_name=axis_name,
                             signal_name=signal_name, in_place=False)
    else:
        signal = tod[signal_name]
    signal = np.atleast_2d(signal)

    if isinstance(filt_function, identity_filter):
        logger.info('fourier_filter: filt_function is identity; skipping FFT.')
        signal = signal.copy()

    else:
        logger.info('fourier_filter: initializing rfft object.')
        a, b, t_1, t_2 = fft_ops.build_rfft_object(n_det, n, 'BOTH')

        if other_idx is not None and other_idx != 0:
            ## so that code can be written always along axis 1
            signal = signal.transpose()

        # This copy is valid for all modes of "resize"
        logger.info('fourier_filter: copying in data.')
        a[:,:min(n, axis.count)] = signal[:,:min(n, axis.count)]
        a[:,min(n, axis.count):] = 0

        ## FFT Signal
        logger.info('fourier_filter: FFT.')
        t_1()

        ## Get Filter
        logger.info('fourier_filter: applying filter.')
        freqs = np.fft.rfftfreq(n, delta_t)
        filt_function.apply(freqs, tod, b, **kwargs)

        ## FFT Back
        logger.info('fourier_filter: IFFT.')
        t_2()

        # Un-pad?
        signal = a[:,:min(n, axis.count)]

        if other_idx is not None and other_idx != 0:
            return signal.transpose()

    if other_idx is None:
        return signal[0]
    
    return signal

def fft_trim(tod, axis='samps', prefer='right'):
    """Restrict AxisManager sample range so that FFTs are efficient.  This
    uses the find_inferior_integer function.

    Args:
      tod (AxisManager): Target, which is modified in place.
      axis (str): Axis to target.
      prefer (str): One of ['left', 'right', 'center'], indicating
        whether to trim away samples from the end, the beginning, or
        !equally at the beginning and end (respectively).

    Returns:
      The (start, stop) indices to use to slice an array and get these
      samples.

    """
    axis_obj = tod[axis]
    old_size = axis_obj.count
    new_size = fft_ops.find_inferior_integer(old_size)

    offset = old_size - new_size
    if prefer == 'left':
        offset = 0
    elif prefer == 'center':
        offset //= 2
    elif prefer == 'right':
        pass
    else:
        raise ValueError(f'Invalid choice prefer="{prefer}"')

    start_stop = (offset, offset+new_size)

    if isinstance(axis_obj, core.OffsetAxis):
        # Account for special indexing of OffsetAxis.
        offset += axis_obj.offset

    tod.restrict(axis, (offset, offset+new_size))
    return start_stop


################################################################

# Base class... provides that a * b always returns a FilterChain.
class _chainable:
    @staticmethod
    def _preference(f):
        return getattr(f, 'preference', 'compose')
    def __mul__(self, other):
        if isinstance(other, identity_filter):
            return self
        if isinstance(self, identity_filter):
            return other
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
def counter_1_over_f(freqs, tod, fk, n):
    """
    Counter 1/f filter for noise w/ PSD that follows:
    
    w*(1 + (fk/f)**n) 
    where w is the white noise level, fk is the knee frequency, and
    n is the 1/f index.
    """
    return 1/(1+(fk/freqs)**n)

@fft_filter
def identity_filter(freqs, tod, invert=False):
    """Identity filter (gain=1 at all frequencies).

    This filter has some special handling -- the FFT will not take
    place if your only intention is to apply an identify filter; also
    identity_filter() * other_filter() will simply return the
    other_filter().

    """
    return np.ones(len(freqs))

@fft_filter
def gain(freqs, tod, gain=1.):
    """Filter that simply applies a simple gain (which could be complex)
    to the entire spectrum.

    This can be used (with gain=1) as an identity filter for testing
    the filter preprocessing.

    """
    return gain * np.ones(len(freqs))

@fft_filter
def low_pass_butter4(freqs, tod, fc):
    """4th-order low-pass filter with f3db at fc (Hz).

    """
    b, a = signal.butter(4, 2*np.pi*fc, 'lowpass', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def high_pass_butter4(freqs, tod, fc):
    """4th-order high-pass filter with f3db at fc (Hz).

    """
    b, a = signal.butter(4, 2*np.pi*fc, 'highpass', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def band_pass_butter4(freqs, tod, fc_low, fc_high):
    """4th-order band-pass filter with f3db at fc (Hz).

    """
    b, a = signal.butter(4, [2*np.pi*fc_low, 2*np.pi*fc_high],
                         'bandpass', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def band_stop_butter4(freqs, tod, fc_low, fc_high):
    """4th-order band-stop filter with f3db at fc (Hz).

    """
    b, a = signal.butter(4, [2*np.pi*fc_low, 2*np.pi*fc_high],
                         'bandstop', analog=True)
    return np.abs(signal.freqs(b, a, 2*np.pi*freqs)[1])

@fft_filter
def tau_filter(freqs, tod, tau_name='timeconst', do_inverse=True):
    """tau_filter is deprecated; use timeconst_filter."""
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
        # attrgetter used to retrieve a field multiple layers deep.
        _f = attrgetter(timeconst)
        timeconst = _f(tod)

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
def gaussian_filter(freqs, tod, fc=0., f_sigma=None, gain=1.0, t_sigma=None):
    """Gaussian bandpass filter

    Parameters:
        fc (float0): Central frequency of the filter (peak of
            passband), in Hz.
        f_sigma (float): Standard deviation of the filter kernel, in
            Hz.
        gain (float): Gain of the filter.
        t_sigma (float): Instead of f_sigma, set t_sigma and f_sigma =
            1/(2 pi t_sigma) will be used.

    The filter kernel has the shape of a normal distribution, centered
    on fc with standard deviation f_sigma, and peak height gain.

    """
    if t_sigma is not None and f_sigma is not None:
        raise ValueError("User must not specify both f_sigma and t_sigma.")
    if t_sigma is not None:
        f_sigma = 1.0 / (2*np.pi*t_sigma)
    if f_sigma is None:
        raise ValueError('User must specify either f_sigma or t_sigma.')
    return gain * np.exp(-0.5*(np.abs(freqs)-fc)**2/f_sigma**2)

@fft_filter
def low_pass_sine2(freqs, tod, cutoff, width=None):
    """Low-pass filter.  Response falls from 1 to 0 between frequencies
    (cutoff - width/2, cutoff + width/2), with a sine-squared shape.

    """
    if width is None:
        width = cutoff * 2
    phase = np.pi * np.clip((abs(freqs) - cutoff) / width, -0.5, 0.5)
    return 0.5 - 0.5 * np.sin(phase)

@fft_filter
def high_pass_sine2(freqs, tod, cutoff, width=None):
    """High-pass filter.  Response rises from 0 to 1 between frequencies
    (cutoff - width/2, cutoff + width/2), with a sine-squared shape.

    """
    if width is None:
        width = cutoff * 2
    phase = np.pi * np.clip((abs(freqs) - cutoff) / width, -0.5, 0.5)
    return 0.5 + 0.5 * np.sin(phase)

@fft_filter
def band_stop_sine2(freqs, tod, fc_low, fc_high, width=None):
    """Band-stop filter.  Response falls/rises from 1/0 to 0/1 between frequencies
    (fc_low/fc_high - width/2, fc_low/fc_high + width/2), with a sine-squared shape.

    """
    return low_pass_sine2._fun(freqs, tod, fc_low, width) + \
        high_pass_sine2._fun(freqs, tod, fc_high, width)

@fft_filter
def iir_filter(freqs, tod, b=None, a=None, fscale=1., iir_params=None,
               invert=False):
    """Infinite impulse response (IIR) filter.  This sort of filter is
    used in digital applications as a low-pass filter prior to
    decimation.  The Smurf and MCE readout filters can both be
    expressed in this form.

    Args:
      b: numerator polynomial filter coefficients (z^0,z^1, ...)
      a: denominator coefficients
      fscale: scalar used to compute z = exp(-2j*pi*freqs*fscale).
        In general this should be equal to 1/f_orig, where f_orig
        is the original sampling frequency (before downsampling).
      iir_params: Alternative way to specify b, a, and fscale (see
        notes).
      invert: If true, returns denom/num instead of num/denom.

    Notes:
      The `b` and `a` coefficients are as implemented in
      scipy.signal.freqs, scipy.signal.butter, etc.  The "angular
      frequencies", `w`, are computed as 2*pi*freqs*fscale.

      If the filter parameters (b, a, fscale) are not passed in
      explicitly, they will be extracted from an AxisManager based on
      the argument iir_params, which must be a dict or AxisManager
      with keys "b", "a", and "fscale", or an AxisManager including 
      the sub-iir_params of each stream_id. In the later case, if 
      the filter parameters of each stream_id is different,
      raises an error.
      
      But note that:

      - If iir_params is a string, tod[iir_params] is used (and must
        be an AxisManager or dict).
      - If iir_params is None, that's the same as passing
        iir_params='iir_params'.

    """
    if a is None:
        # Get params from TOD?
        if iir_params is None:
            iir_params = 'iir_params'
        if isinstance(iir_params, str):
            iir_params = tod[iir_params]
        if (isinstance(iir_params, core.AxisManager)
            and 'a' not in iir_params._fields):
            # Check iir_param's uniformity
            _a = None
            for _field, _sub_iir_params in iir_params._fields.items():
                if isinstance(_sub_iir_params, core.AxisManager) and 'a' in _sub_iir_params._fields:
                    sub_iir_params = _sub_iir_params
                    if _a is None:
                        _a, _b, _fscale = sub_iir_params['a'], sub_iir_params['b'], sub_iir_params['fscale']
                    else:
                        if np.any(np.hstack([sub_iir_params['a'] != _a,
                                             sub_iir_params['b'] != _b,
                                             sub_iir_params['fscale'] != _fscale,])):
                            raise ValueError('iir parameters are not uniform.')
            iir_params = sub_iir_params
            # check if iir_params from axis manager are None
            if iir_params['a'] is None or iir_params['b'] is None:
                raise ValueError('axis manager iir parameters are empty')
        try:
            a = iir_params['a']
            b = iir_params['b']
            fscale = iir_params['fscale']
        except Exception as e:
            raise ValueError("Failed to extract filter parameters from "
                             f"iir_params={iir_params}.")

    z = np.exp(-2j*np.pi*freqs * fscale)
    B, A = np.polyval(b[::-1], z), np.polyval(a[::-1], z)
    if invert:
        return A / B
    return B / A


# Functions to derive low/high/band pass filter from configuration
##################################################################
def get_lpf(cfg):
    """
    Returns a low-pass filter based on the configuration.

    Args:
        cfg (dict): A dictionary containing the low-pass filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of low-pass filter. Supported values are "identity", "butter4" and "sine2".
            - "cutoff": A float specifying the cutoff frequency of the low-pass filter.
            - "trans_width": A float specifying the transition width of the low-pass filter (only for "sine2" type).

    Returns:
        filters.fourier_filter: the low-pass filter.
    """
    if cfg['type'] == 'identity':
        return identity_filter()
    elif cfg['type'] == 'butter4':
        cutoff = cfg['cutoff']
        return low_pass_butter4(fc=cutoff)
    elif cfg['type'] == 'sine2':
        cutoff = cfg['cutoff']
        trans_width = cfg['trans_width']
        return low_pass_sine2(cutoff=cutoff, width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `identity`, `butter4` and `sine2`')


def get_hpf(cfg):
    """
    Returns a high-pass filter based on the configuration.

    Args:
        cfg (dict): A dictionary containing the high-pass filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of high-pass filter. Supported values are "identity", "butter4" and "sine2".
            - "cutoff": A float specifying the cutoff frequency of the high-pass filter.
            - "trans_width": A float specifying the transition width of the high-pass filter (only for "sine2" type).

    Returns:
        filters.fourier_filter: the high-pass filter.
    """
    if cfg['type'] == 'identity':
        return identity_filter()
    elif cfg['type'] == 'butter4':
        cutoff = cfg['cutoff']
        return high_pass_butter4(fc=cutoff)
    elif cfg['type'] == 'sine2':
        cutoff = cfg['cutoff']
        trans_width = cfg['trans_width']
        return high_pass_sine2(cutoff=cutoff, width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `identity`, `butter4` and `sine2`')
        

def get_bpf(cfg):
    """
    Returns a band-pass filter based on the configuration.

    Args:
        cfg (dict): A dictionary containing the band-pass filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of band-pass filter. Supported values are "identity", "butter4" and "sine2".
            - "center": A float specifying the center frequency of the band-pass filter.
            - "width": A float specifying the width of the band-pass filter.
            - "trans_width": A float specifying the transition width of the band-pass filter (only for "sine2" type).

    Returns:
        filters.fourier_filter: the band-pass filter.
    """
    if cfg['type'] == 'identity':
        return identity_filter()
    elif cfg['type'] == 'butter4':
        center = cfg['center']
        width = cfg['width']
        return band_pass_butter4(fc_low=center - width/2., fc_high=center + width/2.)
    elif cfg['type'] == 'sine2':
        center = cfg['center']
        width = cfg['width']
        trans_width = cfg['trans_width']
        return low_pass_sine2(cutoff=center + width/2., width=trans_width)*\
                high_pass_sine2(cutoff=center - width/2., width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `identity`, `butter4` and `sine2`')


def get_bsf(cfg):
    """
    Returns a band-stop filter based on the configuration.

    Args:
        cfg (dict): A dictionary containing the band-stop filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of band-stop filter. Supported values are "identity", "butter4" and "sine2".
            - "center": A float specifying the center frequency of the band-stop filter.
            - "width": A float specifying the width of the band-stop filter.
            - "trans_width": A float specifying the transition width of the band-stop filter (only for "sine2" type).

    Returns:
        filters.fourier_filter: the band-stop filter.
    """
    if cfg['type'] == 'identity':
        return identity_filter()
    elif cfg['type'] == 'butter4':
        center = cfg['center']
        width = cfg['width']
        return band_stop_butter4(fc_low=center - width / 2., fc_high=center + width / 2.)
    elif cfg['type'] == 'sine2':
        center = cfg['center']
        width = cfg['width']
        trans_width = cfg['trans_width']
        return band_stop_sine2(fc_low=center - width / 2., fc_high=center + width / 2, width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `identity`, `butter4` and `sine2`')
