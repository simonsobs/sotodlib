################################
####### Noise model stuff ######
################################
import numpy as np
import so3g
from pixell import fft, utils, bunch
import warnings

from .utils import *

class Nmat:
    def __init__(self):
        """Initialize the noise model. In subclasses this will typically set up parameters, but not
        build the details that depend on the actual time-ordered data"""
        self.ivar  = np.ones(1, dtype=np.float32)
        self.ready = True
    def build(self, tod, **kwargs):
        """Measure the noise properties of the given time-ordered data tod[ndet,nsamp], and
        return a noise model object tailored for that specific tod. The returned object
        needs to provide the .apply(tod) method, which multiplies the tod by the inverse noise
        covariance matrix. Usually the returned object will be of the same class as the one
        we call .build(tod) on, just with more of the internal state initialized."""
        return self
    def apply(self, tod):
        """Multiply the time-ordered data tod[ndet,nsamp] by the inverse noise covariance matrix.
        This is done in-pace, but the result is also returned."""
        self.check_ready()
        return tod*self.ivar
    def white(self, tod):
        """Like apply, but without detector or time correlations"""
        self.check_ready()
        return tod*self.ivar
    def write(self, fname):
        self.check_ready()
        bunch.write(fname, bunch.Bunch(type="Nmat"))
    @staticmethod
    def from_bunch(data): return Nmat()
    def check_ready(self):
        if not self.ready:
            raise ValueError("Attempt to use partially constructed %s. Typically one gets a fully constructed one from the return value of nmat.build(tod)" % type(self).__name__)

class NmatUncorr(Nmat):
    def __init__(self, spacing="exp", nbin=100, nmin=10, window=2, bins=None, ips_binned=None, ivar=None, nwin=None):
        self.spacing    = spacing
        self.nbin       = nbin
        self.nmin       = nmin
        self.bins       = bins
        self.ips_binned = ips_binned
        self.ivar       = ivar
        self.window     = window
        self.nwin       = nwin
        self.ready      = bins is not None and ips_binned is not None and ivar is not None

    def build(self, tod, srate, **kwargs):
        # Apply window while taking fft
        nwin  = utils.nint(self.window*srate)
        apply_window(tod, nwin)
        ft    = fft.rfft(tod)
        # Unapply window again
        apply_window(tod, nwin, -1)
        return self.build_fourier(ft, tod.shape[1], srate, nwin=nwin)

    def build_fourier(self, ftod, nsamp, srate, nwin=0):
        ps = np.abs(ftod)**2
        del ftod
        if self.spacing == "exp":
            bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        elif self.spacing == "lin":
            bins = utils.linbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        else:
            raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
        ps_binned  = utils.bin_data(bins, ps) / nsamp
        ips_binned = 1/ps_binned
        # Compute the representative inverse variance per sample
        ivar = np.zeros(len(ps))
        for bi, b in enumerate(bins):
            ivar += ips_binned[:,bi]*(b[1]-b[0])
        ivar /= bins[-1,1]-bins[0,0]
        return NmatUncorr(spacing=self.spacing, nbin=len(bins), nmin=self.nmin, bins=bins, ips_binned=ips_binned, ivar=ivar, window=self.window, nwin=nwin)

    def apply(self, tod, inplace=False, exp=1):
        self.check_ready()
        if inplace: tod = np.array(tod)
        if self.nwin > 0: apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        self.apply_fourier(ftod, tod.shape[1], exp=exp)
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        return tod

    def apply_fourier(self, ftod, nsamp, exp=1):
        self.check_ready()
        # Candidate for speedup in C
        for bi, b in enumerate(self.bins):
            ftod[:,b[0]:b[1]] *= (self.ips_binned[:,None,bi])**exp/nsamp
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed

    def white(self, tod, inplace=True):
        self.check_ready()
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        tod *= self.ivar[:,None]
        apply_window(tod, self.nwin)
        return tod

    def write(self, fname):
        self.check_ready()
        data = bunch.Bunch(type="NmatUncorr")
        for field in ["spacing", "nbin", "nmin", "bins", "ips_binned", "ivar", "window", "nwin"]:
            data[field] = getattr(self, field)
        bunch.write(fname, data)

    @staticmethod
    def from_bunch(data):
        return NmatUncorr(spacing=data.spacing, nbin=data.nbin, nmin=data.nmin, bins=data.bins, ips_binned=data.ips_binned, ivar=data.ivar, window=data.window, nwin=data.nwin)

class NmatDetvecs(Nmat):
    def __init__(self, bin_edges="act_default", nbin=100, nmin=10, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20], downweight=[], 
                window=2, nwin=None, verbose=False, bins=None, D=None, V=None, iD=None, iV=None, s=None, ivar=None, bmin_eigvec=1000,
                skip_mean=True, mp_significance=None, wnoise_band=None, max_noise_modes=None, detrend_order=None):
        if bin_edges == "act_default":
            bin_edges = np.array([
                0.16, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
                1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 3.80,
                4.60, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 11.0,
                12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
                24.0, 26.0, 28.0, 30.0, 32.0, 36.5, 41.0,
                45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 90.0,
                100., 110., 120., 130., 140., 150., 160., 170.,
                180., 190.
            ])
        elif bin_edges in ["exp", "lin"]:
            pass # Computed later in build_fourier
        elif isinstance(bin_edges, (list, np.ndarray)):
            bin_edges = np.ascontiguousarray(bin_edges)
        else:
            raise ValueError(
                f"bin_edges must be 'act_default', 'exp', 'lin', a list, or a numpy array, got {bin_edges!r}"
            )
        self.bin_edges = bin_edges
        self.nbin       = nbin
        self.nmin       = nmin
        self.mode_bins = mode_bins
        if eig_lim is None:
            self.eig_lim = None
        else:
            self.eig_lim   = np.zeros(len(mode_bins)) + eig_lim
        if single_lim is None:
            self.single_lim = None
        else:
            self.single_lim= np.zeros(len(mode_bins)) + single_lim
        self.verbose   = verbose
        self.downweight= downweight
        self.bins = bins
        self.window = window
        self.nwin   = nwin
        self.bmin_eigvec = bmin_eigvec
        self.skip_mean   = skip_mean
        self.mp_significance = mp_significance
        self.wnoise_band = wnoise_band # Expected format: [fmin, fmax]
        self.max_noise_modes = max_noise_modes
        if detrend_order is not None and detrend_order > 2:
            warnings.warn(f"detrend_order={detrend_order} is not supported. Defaulting to 2 (parabola).")
            self.detrend_order = 2
        else:
            self.detrend_order = detrend_order
        self.D, self.V, self.iD, self.iV, self.s, self.ivar = D, V, iD, iV, s, ivar
        self.ready      = all([a is not None for a in [D, V, iD, iV, s, ivar]])

    def build(self, tod, srate, **kwargs):
        # Detrend the data before windowing & FFT to prevent spectral leakage
        self.detrend_inplace(tod)
        # Apply window before measuring noise model
        nwin  = utils.nint(self.window*srate)
        apply_window(tod, nwin)
        ftod  = fft.rfft(tod)
        # Unapply window again
        apply_window(tod, nwin, -1)
        return self.build_fourier(ftod, tod.shape[1], srate, nwin=nwin, **kwargs)

    def build_fourier(self, ftod, nsamp, srate, nwin=0, **kwargs):
        ndet, nfreq = ftod.shape
        dtype       = utils.real_dtype(ftod.dtype)
        # First define our eigenmode bins.
        # Default mode_bins=[0.25,4.0,20] builds them in two bins: the first from 0.25 to 4 Hz the second from 4 to 20 Hz
        mode_bins = makebins(self.mode_bins, srate, nfreq, nmin=self.bmin_eigvec, rfun=np.ceil, cap=False)
        if np.any(np.diff(mode_bins) < 0):
            raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
        
        # If MP thresholding enabled, estimate white noise level.
        noise_sigmas = None
        
        if self.mp_significance is not None:
            if self.wnoise_band is None:
                raise ValueError("To use Marchenko-Pastur thresholding, you must provide 'wnoise_band' (e.g.,[0.2, 1.8])\
                                 to specify the frequency range for white noise estimation.")
                
            # Construct the physical frequency array for the rfft-ed data
            freqs = np.linspace(0, srate / 2.0, nfreq)
            
            # Find integer indices for the start and stop of the white noise band
            idx_start = np.searchsorted(freqs, self.wnoise_band[0])
            idx_stop  = np.searchsorted(freqs, self.wnoise_band[1], side='right')
            del freqs
            if idx_start >= idx_stop:
                raise ValueError(f"Invalid noise band {self.wnoise_band}. No frequency bins fall in this range.")
            
            # Calculate white noise power per detector
            band_data = ftod[:, idx_start:idx_stop]
            sigma_sq = np.mean(np.real(band_data * np.conj(band_data)), axis=1)
            noise_sigmas = np.sqrt(sigma_sq)

        # Then use these to get our set of basis vectors
        vecs = find_modes_jon(
            ftod,
            mode_bins, 
            eig_lim=self.eig_lim, 
            single_lim=self.single_lim, 
            skip_mean=self.skip_mean, 
            verbose=self.verbose,
            mp_significance=self.mp_significance,
            noise_sigmas=noise_sigmas,
            max_noise_modes=self.max_noise_modes
        )
        nmode= vecs.shape[1]
        if nmode == 0:
            if self.verbose:
                print("NmatDetvecs: Could not find any noise modes, defaulting to uncorrelated noise model")
            noise_uncorr = NmatUncorr(window=self.window)
            return noise_uncorr.build_fourier(ftod, nsamp, srate, nwin=nwin)
        # Compute frequency bins when needed
        if self.bin_edges in ["exp", "lin"]:
             if self.bin_edges == "exp":
                bins = utils.expbin(nfreq, nbin=self.nbin, nmin=self.nmin)
             else:
                bins = utils.linbin(nfreq, nbin=self.nbin, nmin=self.nmin)
        else:
            # When provided a custom array cut bins that extend beyond our max frequency
            bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
            bins      = makebins(bin_edges, srate, nfreq, nmin=5, rfun=np.round)
        nbin      = len(bins)
        # Now measure the power of each basis vector in each bin. The residual
        # noise will be modeled as uncorrelated
        E  = np.zeros([nbin,nmode])
        D  = np.zeros([nbin,ndet])
        Nd = np.zeros([nbin,ndet])
        # Build the projection operator to solve for the per noise mode power profiles
        pinv_vecs = np.linalg.pinv(vecs)
        for bi, b in enumerate(bins):
            # Skip the DC mode, since it's unmeasurable and filtered away
            b = np.maximum(1,b)
            ft_bin = ftod[:, b[0]:b[1]]
            E[bi], D[bi], Nd[bi] = measure_detvecs(ft_bin, vecs, pinv_vecs)
        # Optionally downweight the lowest frequency bins
        if self.downweight != None and len(self.downweight) > 0:
            D[:len(self.downweight)] /= np.array(self.downweight)[:,None]
        # Instead of VEV' we can have just VV' if we bake sqrt(E) into V
        V = vecs[None]*E[:,None]**0.5
        # At this point we have a model for the total noise covariance as
        # N = D + VV'. But since we're doing inverse covariance weighting
        # we need a similar representation for the inverse iN. The function
        # woodbury_invert computes iD, iV, s such that iN = iD + s iV iV'
        # where s usually is -1, but will become +1 if one inverts again
        iD, iV, s = woodbury_invert(D, V)
        # Also compute a representative white noise level
        bsize = bins[:,1]-bins[:,0]
        ivar  = np.sum(iD*bsize[:,None],0)/np.sum(bsize)
        # What about units? I haven't applied any fourier unit factors so far,
        # so we're in plain power units. From the uncorrelated model I found
        # that factor of tod.shape[1] is needed
        iD   *= nsamp
        iV   *= nsamp**0.5
        ivar *= nsamp
        # Fix dtype
        bins = np.ascontiguousarray(bins.astype(np.int32))
        D    = np.ascontiguousarray(D.astype(dtype))
        V    = np.ascontiguousarray(V.astype(dtype))
        iD   = np.ascontiguousarray(iD.astype(dtype))
        iV   = np.ascontiguousarray(iV.astype(dtype))

        return NmatDetvecs(bin_edges=self.bin_edges, eig_lim=self.eig_lim, single_lim=self.single_lim,
                window=self.window, nwin=nwin, downweight=self.downweight, verbose=self.verbose,
                mode_bins= self.mode_bins, bins=bins, D=D, V=V, iD=iD, iV=iV, s=s, ivar=ivar)

    def apply(self, tod, inplace=True, slow=False):
        self.check_ready()
        if not inplace: tod = np.array(tod)        
        # Detrending needs to be part of the operations to ensure:
        # 1. We are properly marginalizing over the data trends to not bias the sky signal estimate
        # 2. Trends in the current sky signal estimate are not causing spectral leakage
        # We also need to apply the operations symmetrically to ensure the system solved through CG is strictly symmetric.
        self.detrend_inplace(tod)
        apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        self.apply_fourier(ftod, tod.shape[1], slow=slow)
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        self.detrend_inplace(tod)
        return tod

    def apply_fourier(self, ftod, nsamp, slow=False):
        self.check_ready()
        dtype= utils.real_dtype(ftod.dtype)
        if slow:
            for bi, b in enumerate(self.bins):
                # Want to multiply by iD + siViV'
                ft    = ftod[:,b[0]:b[1]]
                iD    = self.iD[bi]/nsamp
                iV    = self.iV[bi]/nsamp**0.5
                ft[:] = iD[:,None]*ft + self.s*iV.dot(iV.T.dot(ft))
        else:
            so3g.nmat_detvecs_apply(ftod.view(dtype), self.bins, self.iD, self.iV, float(self.s), float(nsamp))
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed

    def _get_detrend_basis(self, n_samps, dtype):
        """Lazily initialize and cache the orthogonal polynomials based on detrend_order."""
        if not hasattr(self, '_detrend_cache') or self._detrend_cache[0] != n_samps:
            v1, v1_norm, v2, v2_norm = None, None, None, None
            # Only compute slope basis if order >= 1
            if self.detrend_order is not None and self.detrend_order >= 1:
                v1 = np.linspace(-0.5, 0.5, n_samps, dtype=dtype)
                v1_norm = np.dot(v1, v1)
            # Only compute parabola basis if order >= 2
            if self.detrend_order is not None and self.detrend_order >= 2:
                x2 = v1 ** 2
                v2 = x2 - np.mean(x2)
                v2_norm = np.dot(v2, v2)
            self._detrend_cache = (n_samps, v1, v1_norm, v2, v2_norm)
        return self._detrend_cache[1:]

    def detrend_inplace(self, tod):
        """Symmetric, in-place projection up to detrend_order."""
        # Skip entirely if requested
        if self.detrend_order is None or self.detrend_order < 0:
            return tod
        n_dets, n_samps = tod.shape
        # 0th Order (Mean). We do this first to center the data, 
        # protecting the float32 precision of the subsequent dot products
        means = np.mean(tod, axis=-1, keepdims=True)
        tod -= means 
        if self.detrend_order == 0:
            return tod
        # Get cached basis vectors
        v1, v1_norm, v2, v2_norm = self._get_detrend_basis(n_samps, tod.dtype)
        # 1st Order (Slope)
        if self.detrend_order == 1:
            slopes = np.dot(tod, v1) / v1_norm
            # This is not vectorized but saves memory & cache locality
            for i in range(n_dets):
                tod[i] -= slopes[i] * v1
            return tod
        # 2nd Order (Parabola)
        if self.detrend_order == 2:
            slopes = np.dot(tod, v1) / v1_norm
            paras  = np.dot(tod, v2) / v2_norm
            for i in range(n_dets):
                tod[i] -= (slopes[i] * v1 + paras[i] * v2)
            return tod
            
    def white(self, tod, inplace=True):
        self.check_ready()
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        tod *= self.ivar[:,None]
        apply_window(tod, self.nwin)
        return tod

    def write(self, fname):
        self.check_ready()
        data = bunch.Bunch(type="NmatDetvecs")
        for field in ["bin_edges", "eig_lim", "single_lim", "window", "nwin", "downweight",
                "bins", "D", "V", "iD", "iV", "s", "ivar", "mode_bins", "verbose", "bmin_eigvec",
                "skip_mean", "mp_significance", "wnoise_band", "max_noise_modes", "detrend_order"]:
            val = getattr(self, field)
            if val is not None:
                data[field] = val
        try:
            bunch.write(fname, data)
        except Exception as e:
            msg = f"Failed to write {fname}: {e}"
            raise RuntimeError(msg)

@staticmethod
def from_bunch(data):
    return NmatDetvecs(
        bin_edges   = getattr(data, "bin_edges", None),
        eig_lim     = getattr(data, "eig_lim", None),
        single_lim  = getattr(data, "single_lim", None),
        window      = getattr(data, "window", None),
        nwin        = getattr(data, "nwin", None),
        downweight  = getattr(data, "downweight", None),
        bins        = getattr(data, "bins", None),
        D           = getattr(data, "D", None),
        V           = getattr(data, "V", None),
        iD          = getattr(data, "iD", None),
        iV          = getattr(data, "iV", None),
        s           = getattr(data, "s", None),
        ivar        = getattr(data, "ivar", None),
    )

# Combination of NmatDetvecs and NmatUncorr. The first handles the correlations while the
# latter handles the variance
class NmatScaledvecs(Nmat):
    def __init__(self, nmat_uncorr, nmat_detvecs, window=0, nwin=0):
        self.nmat_uncorr  = nmat_uncorr
        self.nmat_detvecs = nmat_detvecs
        self.window       = window
        self.nwin         = nwin
    @property
    def ready(self): return self.nmat_uncorr.ready and self.nmat_detvecs.ready
    @property
    def ivar(self): return self.nmat_uncorr.ivar
    def build(self, tod, srate, **kwargs):
        # Apply window before measuring noise model
        nsamp = tod.shape[1]
        nwin  = utils.nint(self.window*srate)
        apply_window(tod, nwin)
        ftod  = fft.rfft(tod)
        nmat_uncorr = self.nmat_uncorr.build_fourier(ftod, nsamp, srate)
        # Whiten the tod before building the correlated model
        nmat_uncorr.apply_fourier(ftod, nsamp, exp=0.5)
        nmat_detvecs= self.nmat_detvecs.build_fourier(ftod, nsamp, srate)
        # Unapply window again
        apply_window(tod, nwin, -1)
        return NmatScaledvecs(nmat_uncorr, nmat_detvecs, window=self.window, nwin=self.nwin)

    def apply(self, tod, inplace=True):
        self.check_ready()
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        # Our model is N" = Nu"**0.5 Nd" Nu**0.5. But since both models are
        # fourier-diagonal, this is equal to Nd" Nu"
        self.nmat_uncorr .apply_fourier(ftod, tod.shape[1])
        self.nmat_detvecs.apply_fourier(ftod, tod.shape[1])
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        return tod

    def white(self, tod, inplace=True):
        self.check_ready()
        return self.nmat_uncorr.white(tod, inplace=inplace)

    def write(self, fname):
        self.check_ready()
        raise NotImplementedError

    @staticmethod
    def from_bunch(data):
        raise NotImplementedError

class NmatWhite(Nmat):
    def __init__(self, ivar=None):
        """
        This is a white noise model for the mapmaker.
        The white noise model is characterized by 
        (1) no correlations between detectors 
        (2) a white (flat) spectrum per detector  
        (3) the weights, if not passed in through ivar, are computed simply
        from the inverse variance of the timestream.

        Parameters
        ----------
        ivar : numpy.ndarray or None, optional
            Overwrite the inverse variance per detector

        Returns
        -------
        noise_model : An Nmat object with the noise model

        """
        self.ivar = ivar
    def build(self, tod, srate, **kwargs):
        if np.any(np.logical_not(np.isfinite(tod))):
            raise ValueError(f"There is a nan when calculating the white noise !!!")
        ivar = 1/np.var(tod,1)
        return NmatWhite(ivar)
    def apply(self, tod, inplace=False):
        self.check_ready()
        if inplace: tod = np.array(tod)
        tod *= self.ivar[:,None]
        return tod
    def white(self, tod, inplace=True):
        self.check_ready()
        return self.apply(tod, inplace=inplace)
    def write(self, fname):
        self.check_ready()
        data = bunch.Bunch(type="NmatWhite")
        for field in ["ivar"]:
            data[field] = getattr(self, field)
        bunch.write(fname, data)
    @staticmethod
    def from_bunch(data):
        return NmatWhite(ivar=data.ivar)
    @property
    def ready(self):
        return self.ivar is not None

class NmatUnit(Nmat):
    """
    This is a noise model that does nothing, equivalent to multiply by a 
    unit noise matrix
    """

    def __init__(self, ivar=None):
        self.ivar  = ivar
        self.ready = ivar is not None
    def build(self, tod, **kwargs):
        ndet, nsamps = tod.shape
        ivar = np.ones(ndet)
        return NmatUnit(ivar=ivar)
    def apply(self, tod):
        # the tod is returned intact
        self.check_ready()
        return tod
    def white(self, tod):
        # the tod is returned intact
        self.check_ready()
        return tod
    def write(self, fname):
        self.check_ready()
        data = bunch.Bunch(type="NmatUnit")
        for field in ["ivar"]:
            data[field] = getattr(self, field)
        bunch.write(fname, data)
    @staticmethod
    def from_bunch(data): 
        return NmatUnit(ivar=data.ivar)

def write_nmat(fname, nmat):
    nmat.write(fname)

def read_nmat(fname):
    data = bunch.read(fname)
    typ  = data.type.decode()
    if   typ == "NmatDetvecs": return NmatDetvecs.from_bunch(data)
    elif typ == "NmatUncorr":  return NmatUncorr .from_bunch(data)
    elif typ == "NmatWhite":   return NmatWhite  .from_bunch(data)
    elif typ == "NmatUnit":    return NmatUnit   .from_bunch(data)
    elif typ == "Nmat":        return Nmat       .from_bunch(data)
    else: raise IOError("Unrecognized noise matrix type '%s' in '%s'" % (str(typ), fname))
