################################
####### Noise model stuff ######
################################
import time
import numpy as np
import so3g
from pixell import fft, utils, bunch

from .utils import *

class Nmat:
    def __init__(self):
        """Initialize the noise model. In subclasses this will typically set up parameters, but not
        build the details that depend on the actual time-ordered data"""
        self.ivar  = np.ones(1, dtype=np.float32)
        self.ready = True
    def build(self, tod, srate, **kwargs):
        """Measure the noise properties of the given time-ordered data tod[ndet,nsamp], and
        return a noise model object tailored for that specific tod. The returned object
        needs to provide the .apply(tod) method, which multiplies the tod by the inverse noise
        covariance matrix. Usually the returned object will be of the same class as the one
        we call .build(tod) on, just with more of the internal state initialized."""
        _ = tod, srate, kwargs
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
        _ = kwargs
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
        if   self.spacing == "exp": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        elif self.spacing == "lin": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        else: raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
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
    def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
            downweight=[], window=2, nwin=None, verbose=False, bins=None,
            D=None, V=None, iD=None, iV=None, s=None, ivar=None, bmin_eigvec=1000):
        # This is all taken from act, not tuned to so yet
        if bin_edges is None: bin_edges = np.array([
            0.16, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
            1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 3.80,
            4.60, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 11.0,
            12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            24.0, 26.0, 28.0, 30.0, 32.0, 36.5, 41.0,
            45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 90.0,
            100., 110., 120., 130., 140., 150., 160., 170.,
            180., 190.
        ])
        self.bin_edges = bin_edges
        self.mode_bins = mode_bins
        self.eig_lim   = np.zeros(len(mode_bins))+eig_lim
        self.single_lim= np.zeros(len(mode_bins))+single_lim
        self.verbose   = verbose
        self.downweight= downweight
        self.bins = bins
        self.window = window
        self.nwin   = nwin
        self.bmin_eigvec = bmin_eigvec
        self.D, self.V, self.iD, self.iV, self.s, self.ivar = D, V, iD, iV, s, ivar
        self.ready      = all([a is not None for a in [D, V, iD, iV, s, ivar]])

    def build(self, tod, srate, **kwargs):
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
        # First build our set of eigenvectors in two bins. The first goes from
        # 0.25 to 4 Hz the second from 4Hz and up
        mode_bins = makebins(self.mode_bins, srate, nfreq, nmin=self.bmin_eigvec, rfun=np.ceil, cap=False)
        if np.any(np.diff(mode_bins) < 0):
            raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
        # Then use these to get our set of basis vectors
        vecs = find_modes_jon(ftod, mode_bins, eig_lim=self.eig_lim, single_lim=self.single_lim, verbose=self.verbose)
        nmode= vecs.shape[1]
        if vecs.size == 0: raise errors.ModelError("Could not find any noise modes")
        # Cut bins that extend beyond our max frequency
        bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
        bins      = makebins(bin_edges, srate, nfreq, nmin=5, rfun=np.round)
        nbin      = len(bins)
        # Now measure the power of each basis vector in each bin. The residual
        # noise will be modeled as uncorrelated
        E  = np.zeros([nbin,nmode])
        D  = np.zeros([nbin,ndet])
        Nd = np.zeros([nbin,ndet])
        for bi, b in enumerate(bins):
            # Skip the DC mode, since it's it's unmeasurable and filtered away
            b = np.maximum(1,b)
            E[bi], D[bi], Nd[bi] = measure_detvecs(ftod[:,b[0]:b[1]], vecs)
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
                bins=bins, D=D, V=V, iD=iD, iV=iV, s=s, ivar=ivar)

    def apply(self, tod, inplace=True, slow=False):
        self.check_ready()
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        self.apply_fourier(ftod, tod.shape[1], slow=slow)
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
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
                "bins", "D", "V", "iD", "iV", "s", "ivar"]:
            data[field] = getattr(self, field)
        try:
            bunch.write(fname, data)
        except Exception as e:
            msg = f"Failed to write {fname}: {e}"
            raise RuntimeError(msg)

    @staticmethod
    def from_bunch(data):
        return NmatDetvecs(bin_edges=data.bin_edges, eig_lim=data.eig_lim, single_lim=data.single_lim,
                window=data.window, nwin=data.nwin, downweight=data.downweight,
                bins=data.bins, D=data.D, V=data.V, iD=data.iD, iV=data.iV, s=data.s, ivar=data.ivar)

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
        _ = kwargs
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
        _ = srate, kwargs
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

# See https://github.com/amaurea/sogma/blob/master/python/nmat.py#L323 for the original
class NmatAdaptive(Nmat):
    def __init__(self, eig_lim=16, single_lim=0.55, window=2, sampvar=1e-2,
            bsmooth=50, atm_res=2**(1/4), maxmodes=100, dev=None,
            hbmps=None, bsize_mean=None, bsize_per=None,
            bins=None, iD=None, iE=None, V=None, Kh=None, ivar=None, nwin=None,
            normexp=-1):
        self.eig_lim    = eig_lim
        self.single_lim = single_lim
        self.window     = window
        self.sampvar    = sampvar
        self.bsmooth    = bsmooth
        self.atm_res    = atm_res
        self.maxmodes   = maxmodes
        self.normexp    = normexp
        # These are usually computed in build()
        self.bsize_mean = bsize_mean
        self.bsize_per  = bsize_per
        self.bins       = bins
        self.nwin       = nwin
        self.ready = all([a is not None for a in [bsize_mean,bsize_per,hbmps,bins,iD,iE,V,Kh,ivar,nwin]])
        if self.ready:
            self.iD   = np.ascontiguousarray(iD)
            self.ivar = np.ascontiguousarray(ivar)
            self.hbmps= np.ascontiguousarray(hbmps)
            self.V    = [np.ascontiguousarray(v) for v in V]
            self.Kh   = [np.ascontiguousarray(kh) for kh in Kh]
            self.iE   = iE
            self.maxbin = np.max(self.bins[:,1]-self.bins[:,0])

    def build(self, tod, srate, **kwargs):
        _ = kwargs
        # Improve on NmatDetvecs in two ways:
        # 1. Factorize out per-detector power
        # 2. Use the shape of the power spectrum to define
        #    both where to measure the eigenvectors, and
        #    the bins for the eigenvalues.
        nwin = utils.nint(self.window*srate)
        _, nsamp = tod.shape
        # Apply window in-place, to prefpare for fft
        apply_window(tod, nwin)
        ftod = fft.rfft(tod)
        # Normalize to avoid 32-bit overflow in atmospheric region.
        # Make sure apply and ivar are consistent
        # ftod *= nsamp**self.normexp
        # Undo window
        apply_window(tod, nwin, -1)
        return self.build_fourier(ftod, srate, nsamp)

    def build_fourier(self, ftod, srate, nsamp):
        ndet, nfreq = ftod.shape
        dtype = utils.real_dtype(ftod.dtype)
        nwin  = utils.nint(self.window*srate)
        # data-type to use in mode-finding. At some point it seemed like
        # this needed to be float64, but float32 turned out to be fine
        # after fixing some issues with overfit correlations
        wtype = np.float32
        # [Step 1]: Build a detector-uncorrelated noise model
        # 1a. Measure power spectra
        ps  = np.abs(ftod)**2  # full-res ps
        mps = np.mean(ps,0)    # mean det power

        # Currently unused, 
        # mtod= np.mean(ftod,0)
        # psm = np.abs(mtod)**2  # power of common mode
        # freqs = np.linspace(0, srate, nfreq)

        # The relative sample variance is 2/nsamp, which we want to be better than self.sampvar.
        # This requires nsamp > 2/self.sampvar
        bsize_mean = max(1,utils.ceil(2/self.sampvar/ndet))
        bsize_per  = max(1,utils.ceil(2/self.sampvar))
        # Simpler if the bin sizes are multiples of each other
        bsize_per  = utils.ceil(bsize_per, bsize_mean)
        bmps       = utils.block_reduce(mps, bsize_mean)
        # Precompute whitening version
        hbmps      = bmps**-0.5
        # 1b. Divide out from the tod. After this ftod is whitened, except
        # for the detector correlations
        block_scale(ftod, hbmps, bsize=bsize_mean, inplace=True)
        # [Step 2]: Build the frequency bins
        # 2a. Measure the smooth background behind the peaks. The bin size should
        # be wider then any spike we want to ignore. A good size would be the 1/scan_period,
        # but we don't have az here. Can't use too wide bins, or we won't find the
        # first spikes
        bsize_smooth = self.bsmooth
        smooth_bps   = utils.block_reduce(mps, bsize_smooth, op=np.median, inclusive=False)
        smooth_ps    = logint(smooth_bps,np.arange(nfreq)/bsize_smooth)
        # 2b. Find our spikes
        rel_ps       = mps/smooth_ps
        bins_spike   = find_spikes(rel_ps)
        # If we detect a spike at the very beginning, in the extrapolated region, then it's
        # unreliable
        if len(bins_spike) > 0 and bins_spike[0,0] == 0: bins_spike = bins_spike[1:]
        # 2c. Find atmospheric regions using smooth_bps
        bins_atm     = find_atm_bins(smooth_bps, bsize=bsize_smooth, step=self.atm_res)
        # Add a final all-the-rest bin
        bins_atm     = np.concatenate([bins_atm,[[bins_atm[-1,1],nfreq]]],0)
        # Want to exclude the spikes from the atmospheric model.
        # Since we model the freq-bins as independent and zero-mean,
        # we can just zero out the spike regions after measuring
        # them, and then compensate for the loss of power aftewards.
        # This lets us avoid splitting the atm-bins when measuring.
        # 3a. Find modes in spike bins
        # This is a bit ad-hoc, but it gives high but not completely dominating
        # weight to the low-l atmospheric region
        weight = (mps/np.mean(mps))**0.5
        spike_power = None
        if len(bins_spike) > 0:
            spike_power = noise_modes_hybrid(ftod, bins_spike, weight=weight,
                eig_lim=self.eig_lim, single_lim=self.single_lim, nmax=self.maxmodes, wtype=wtype)
        # 3c. Find atm modes
        mask = np.ones(ftod.shape[-1], np.int32)
        for bi, (b1,b2) in enumerate(bins_spike):
            mask[b1:b2] = 0
        atm_power = noise_modes_hybrid(ftod, bins_atm, weight=weight, mask=mask,
            eig_lim=self.eig_lim, single_lim=self.single_lim, nmax=self.maxmodes, wtype=wtype)
        # 4. Interleave bins, unless we don't need to
        if len(bins_spike) > 0 and spike_power is not None:
            bins, srcs, sinds = override_bins([bins_atm, bins_spike])
            bins  = np.array(bins)
            power = bunch.Bunch()
            for key in atm_power:
                power[key] = pick_data([atm_power[key], spike_power[key]], srcs, sinds)
        else:
            bins  = np.array(bins_atm)
            power = atm_power
        # 5. Precompute Kh and iD
        nbin = len(bins)
        iD = 1/np.array(power.Ds).astype(wtype, copy=False) # [nbin,ndet]
        iE = [1/e.astype(wtype, copy=False) for e in power.Es]       # [nbin][nmode]
        # Precompute Kh = (E" + V'D"V)**-0.5. [nbin][nmode,nmode]
        Kh = []
        for bi in range(nbin):
            V  = power.Vs[bi].astype(wtype, copy=False)
            iK = np.diag(iE[bi]) + V.T.dot(iD[bi,:,None] * V)
            Kh.append(np.linalg.cholesky(np.linalg.inv(iK)))
        # Convert to target precsion
        iD = iD.astype(dtype, copy=False)
        iE = [a.astype(dtype, copy=False) for a in iE]
        Kh = [a.astype(dtype, copy=False) for a in Kh]
        # 6. nsamp normalization of hbmps
        hbmps *= nsamp**self.normexp
        # 7. Also compute a representative white noise level
        # bsize = np.array(bins[:,1]-bins[:,0])
        ivar  = np.mean(1/utils.block_reduce(ps, bsize_per, op=np.median, inclusive=False), -1)
        # (nsamp**normexp)**2 * nsamp = nsamp**(2*normexp+1)
        ivar *= nsamp**(2*self.normexp+1)
        # 8. Construct the full noise model
        return NmatAdaptive(
            eig_lim=self.eig_lim, single_lim=self.single_lim, window=self.window,
            sampvar=self.sampvar, bsmooth=self.bsmooth, atm_res=self.atm_res,
            maxmodes=self.maxmodes, 
            hbmps=hbmps, bsize_mean=bsize_mean, bsize_per=bsize_per,
            bins=bins, iD=iD, iE=iE, V=power.Vs, Kh=Kh, ivar=ivar, nwin=nwin, normexp=self.normexp)

    def apply(self, tod, inplace=True, nofft=False):
        self.check_ready()
        if not inplace: tod = tod.copy()
        if not nofft: apply_window(tod, self.nwin)
        if not nofft: ft = fft.rfft(tod)
        else: ft = tod
        # If we don't cast to real here, we get the same result but much slower
        # real_dtype needed for the nofft case
        rft = ft.view(utils.real_dtype(tod.dtype))
        # Apply the high-resolution, non-detector-correlated part of the model.
        # The *2 compensates for the cast to real
        block_scale(rft, self.hbmps, bsize=self.bsize_mean*2, inplace=True)
        # Then handle the detector-correlation part.
        # First set up work arrays. Safe to overwrite tod array here,
        # since we'll overwrite it with the ifft afterwards anyway
        ndet     = len(tod)
        maxnmode = max([V.shape[1] for V in self.V])
        # nbin     = len(self.bins)
        # Tmp must be big enough to hold a full bin's worth of data
        tmp    = np.empty([maxnmode,2*self.maxbin],dtype=rft.dtype)
        vtmp   = np.empty([ndet,maxnmode],         dtype=rft.dtype)
        divtmp = np.empty([ndet,maxnmode],         dtype=rft.dtype)
        apply_vecs2(rft, self.iD, self.V, self.Kh, self.bins, tmp, vtmp, divtmp, out=rft)
        # Second half of high-resolution part
        block_scale(rft, self.hbmps, bsize=self.bsize_mean*2, inplace=True)
        # And finish
        if not nofft: fft.irfft(ft, tod)
        if not nofft: apply_window(tod, self.nwin)
        return tod

    def white(self, tod, inplace=True):
        self.check_ready()
        if not inplace: tod = tod.copy()
        apply_window(tod, self.nwin)
        tod *= self.ivar[:,None]
        apply_window(tod, self.nwin)
        return tod

    # Debug functions below. These are not part of the
    # Nmat interface.
    def eval_cov(self, d1=None, d2=None, finds=None):
        """This debug function evaluates the covariance between detector sets d1 and d2
        at the given frequency indices."""
        self.check_ready()
        nbin, ndet = self.iD.shape
        # nfreq      = self.bins[-1,1]
        if d1    is None: d1 = np.arange(ndet)
        if d2    is None: d2 = np.arange(ndet)
        # if finds is None: fidns = self.dev.np.arange(nfreq)
        # Evaluate the core cov = D+VEV' for each bin. Our model is
        C = np.zeros((len(d1),len(d2),nbin), self.iD.dtype)
        for bi, _ in enumerate(self.bins):
            D  = np.diag(1/self.iD[bi])[d1][:,d2]
            V1 = self.V[bi][d1]
            V2 = self.V[bi][d2]
            C[:,:,bi] = D + V1.dot(1/self.iE[bi][:,None]*V2.T)
        # Read this off at the requested indices
        binds = np.searchsorted(self.bins[:,1], finds, side="right")
        C = C[:,:,binds]
        # Modify by the high-res scale
        C *= self.hbmps[finds//self.bsize_mean]**-2
        return C
    def eval_var(self, d=None, finds=None):
        """This debug function evaluates the variance for det set d
        at the given frequency indices."""
        self.check_ready()
        nbin, ndet = self.iD.shape
        # nfreq      = self.bins[-1,1]
        if d     is None: d  = np.arange(ndet)
        # if finds is None: fidns = np.arange(nfreq)
        # Evaluate the core cov = D+VEV' for each bin. Our model is
        ps = np.zeros((len(d),nbin), self.iD.dtype)
        for bi, _ in enumerate(self.bins):
            ps[:,bi] = 1/self.iD[bi,d] + np.sum(self.V[bi][d]**2/self.iE[bi],-1)
        # Read this off at the requested indices
        binds = np.searchsorted(self.bins[:,1], finds, side="right")
        ps  = ps[:,binds]
        # Modify by the high-res scale
        ps *= self.hbmps[finds//self.bsize_mean]**-2
        return ps
    def det_slice(self, sel):
        self.check_ready()
        ivar = self.ivar[sel]
        iD   = self.iD[:,sel]
        V    = [v[sel] for v in self.V]
        Kh   = []
        for bi, _ in enumerate(self.bins):
            iK = np.diag(self.iE[bi]) + V[bi].T.dot(iD[bi,:,None] * V[bi])
            Kh.append(np.linalg.cholesky(np.linalg.inv(iK)))
        return NmatAdaptive(
            eig_lim=self.eig_lim, single_lim=self.single_lim, window=self.window,
            sampvar=self.sampvar, bsmooth=self.bsmooth, atm_res=self.atm_res,
            maxmodes=self.maxmodes,  hbmps=self.hbmps,
            bsize_mean=self.bsize_mean, bsize_per=self.bsize_per,
            bins=self.bins, iD=iD, iE=self.iE, V=V,
            Kh=Kh, ivar=ivar, nwin=self.nwin, normexp=self.normexp)
    def inv(self):
        self.check_ready()
        iE = [1/ie for ie in self.iE]
        iD = 1/self.iD
        Kh = []
        for bi, _ in enumerate(self.bins):
            iK = np.diag(iE[bi]) + self.V[bi].T.dot(iD[bi,:,None] * self.V[bi])
            Kh.append(np.linalg.cholesky(np.linalg.inv(iK)))
        ivar  = 1/self.ivar
        hbmps = 1/self.hbmps
        return NmatAdaptive(
            eig_lim=self.eig_lim, single_lim=self.single_lim, window=self.window,
            sampvar=self.sampvar, bsmooth=self.bsmooth, atm_res=self.atm_res,
            maxmodes=self.maxmodes, hbmps=hbmps,
            bsize_mean=self.bsize_mean, bsize_per=self.bsize_per,
            bins=self.bins, iD=iD, iE=iE, V=self.V,
            Kh=Kh, ivar=ivar, nwin=self.nwin, normexp=self.normexp)

class NmatUnit(Nmat):
    """
    This is a noise model that does nothing, equivalent to multiply by a 
    unit noise matrix
    """

    def __init__(self, ivar=None):
        self.ivar  = ivar
        self.ready = ivar is not None
    def build(self, tod, srate, **kwargs):
        _ = srate, kwargs
        ndet, _ = tod.shape
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
