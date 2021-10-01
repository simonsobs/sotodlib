# Module with functions for mapmaking. It could have been split into many sub-modules,
# e.g. noise_model.py, pointing_matrix.py, utilities, mlmapmaker.py etc. Maybe we will
# do that later, but for now I don't think that split makes things easier for the user.

import numpy as np, sys, time, warnings
import so3g
from . import coords
from . import tod_ops
from pixell import enmap, utils, fft, bunch, wcsutils, mpi

##########################################
##### Maximum likelihood mapmaking #######
##########################################

class MLMapmaker:
    def __init__(self, shape, wcs, comps="T", noise_model=None, dtype_tod=np.float32,
            dtype_map=np.float64, comm=mpi.COMM_WORLD, recenter=None, verbose=False):
        if shape is not None:
            shape = shape[-2:]
        if noise_model is None:
            noise_model = NmatUncorr()
        self.shape = shape
        self.wcs   = wcs
        self.comm  = comm
        self.comps = comps
        self.ncomp = len(comps)
        self.dtype_tod = dtype_tod
        self.dtype_map = dtype_map
        self.verbose   = verbose
        if shape is None:
            self.map_rhs   = None
            self.map_div   = None
            self.auto_grow = True
        else:
            self.map_rhs   = enmap.zeros((self.ncomp,)          +self.shape, self.wcs, self.dtype_map)
            self.map_div   = enmap.zeros((self.ncomp,self.ncomp)+self.shape, self.wcs, self.dtype_map)
            self.auto_grow = False
        self.map_idiv  = None
        self.noise_model = noise_model
        self.observations = []
        # Cut-related stuff
        self.junk_offs= None
        self.junk_rhs = []
        self.junk_div = []
        # Set up the degrees of freedom we will solve for
        self.dof   = MultiZipper(comm=comm)
        #self.dof.add(MapZipper(self.map_rhs.shape, self.map_rhs.wcs))
        self.ready    = False
        self.recenter = recenter
    def build_obs(self, id, obs, noise_model=None):
        # Signal must have the right dtype, or the pmat we build will break later
        t1     = time.time()
        tod    = obs.signal.astype(self.dtype_tod, copy=False)
        ctime  = obs.timestamps
        # Set up cuts handling
        pcut   = PmatCut(obs.glitch_flags)
        # Build the local geometry and pointing matrix for this observation
        if self.recenter:
            rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter,
                ctime=ctime[len(ctime)//2], geom=(self.shape, self.wcs), site=SITE))
        else: rot = None
        # Ideally we would include cuts in the pmat. It would slightly simplify PmatCut, which
        # would skip the "clear" step, and it would make the map_div calculation simpler.
        # However, doing so changes the result, which should be investigated.
        pmat    = coords.pmat.P.for_tod(obs, comps=self.comps, geom=(self.shape, self.wcs), rot=rot, threads="domdir")
        t2 = time.time()
        # Build the noise model
        if noise_model is None: noise_model = self.noise_model
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        nmat   = self.noise_model.build(tod, srate=srate)
        t3 = time.time()
        tod      = nmat.apply(tod)
        t4 = time.time()
        map_rhs  = enmap.zeros((self.ncomp,)+self.shape, self.wcs, self.dtype_map)
        junk_rhs = np.zeros(pcut.njunk, self.dtype_tod)
        ## FIXME
        #so3g.test_cuts(pcut.cuts.ranges)
        pcut.backward(tod, junk_rhs)
        pmat.to_map(dest=map_rhs, signal=tod)
        t5 = time.time()
        # After this we don't need the tod values any more, so we are free to mess with them.
        map_div    = enmap.zeros((self.ncomp,self.ncomp)+self.shape, self.wcs, self.dtype_map)
        junk_div   = np.ones(pcut.njunk, self.dtype_tod)
        tod[:]     = 0
        pcut.forward(tod, junk_div)
        tod       *= nmat.ivar[:,None]
        pcut.backward(tod, junk_div)
        #pmat.to_weights(dest=map_div, det_weights=nmat.ivar.astype(self.dtype_tod))
        # Full manual build of map_div
        for i in range(self.ncomp):
            map_div[i]   = 0
            map_div[i,i] = 1
            tod[:]       = 0
            pmat.from_map(map_div[i], dest=tod)
            pcut.clear(tod)
            tod *= nmat.ivar[:,None]
            map_div[i]   = 0
            pmat.to_map(signal=tod, dest=map_div[i])
        t6 = time.time()
        if np.any(map_div[0,0,0,:]!=0) or np.any(map_div[0,0,-1,:] != 0) or np.any(map_div[0,0,:,0] != 0) or np.any(map_div[0,0,:,-1] != 0):
            warnings.warn("Local work space was too small - data truncated")
        # And return the ML data for this observation
        data = bunch.Bunch(id=id, ndet=obs.dets.count, nsamp=len(ctime), dets=obs.dets.vals,
                shape=self.shape, wcs=self.wcs, pmat=pmat, pcut=pcut, nmat=nmat, map_rhs=map_rhs,
                map_div=map_div, junk_rhs=junk_rhs, junk_div=junk_div)
        print("build %-70s : Pbuild %8.3f Nbuild %8.3f Pw' %8.3f N %8.3f Pm' %8.3f  %3d %6d" % (id, t2-t1, t3-t2, t6-t5, t4-t3, t5-t4, data.ndet, data.nsamp))
        return data
    def add_obs(self, data):
        assert not self.ready, "Adding more data after preparing to solve is not supported"
        if self.auto_grow:
            # Grow rhs and div to hold new obs?
            if self.shape is None:
                self.shape, self.wcs = data.shape, data.wcs
                self.map_rhs, self.map_div = data.map_rhs.copy(), data.map_div.copy()
            else:
                new_shape, new_wcs = coords.get_supergeom((self.shape, self.wcs), (data.shape, data.wcs))
                if (new_shape != self.shape):
                    for attr in ['map_rhs', 'map_div']:
                        submap = getattr(self, attr)
                        lead_shape = submap.shape[:-2]
                        newmap = enmap.zeros(lead_shape + new_shape, dtype=submap.dtype, wcs=new_wcs)
                        newmap.insert(submap)
                        setattr(self, attr, newmap)
                    self.shape, self.wcs = new_shape, new_wcs
                # Add data.rhs and data.div to our full rhs and div
                self.map_rhs.insert(data.map_rhs, op=np.ndarray.__iadd__)
                self.map_div.insert(data.map_div, op=np.ndarray.__iadd__)
            # Add the rest to self.observations
            data = data.copy(); del data.map_rhs; del data.map_div
        else:
            self.map_rhs.insert(data.map_rhs, op=np.ndarray.__iadd__)
            self.map_div.insert(data.map_div, op=np.ndarray.__iadd__)
        # Handle the cut samples
        self.junk_rhs.append(data.junk_rhs)
        self.junk_div.append(data.junk_div)
        del data.junk_rhs, data.junk_div
        self.observations.append(data)
    def prepare(self):
        if self.ready: return
        if self.auto_grow:
            # Promote everything to full-size maps.
            all_geoms = self.comm.gather((self.shape, self.wcs), root=0)
            if self.comm.rank == 0:
                new_shape, new_wcs = coords.get_supergeom(*all_geoms)
            else:
                new_shape, new_wcs = None, None
            new_shape, new_wcs = self.comm.bcast((new_shape, new_wcs), root=0)
            for attr in ['map_rhs', 'map_div']:
                submap = getattr(self, attr)
                lead_shape = submap.shape[:-2]
                newmap = enmap.zeros(lead_shape + new_shape, dtype=submap.dtype, wcs=new_wcs)
                newmap.insert(submap)
                setattr(self, attr, newmap)
                del submap
            self.shape, self.wcs = new_shape, new_wcs

        print("rank %3d ntod %2d" % (self.comm.rank, len(self.junk_rhs)))
        self.comm.Barrier()

        self.map_rhs  = utils.allreduce(self.map_rhs, self.comm)
        self.map_div  = utils.allreduce(self.map_div, self.comm)
        self.map_idiv = safe_invert_div(self.map_div)

        self.junk_offs= utils.cumsum([r.size for r in self.junk_rhs], endpoint=True)
        self.junk_rhs = np.concatenate(self.junk_rhs)
        self.junk_div = np.concatenate(self.junk_div)

        # Add the RHS now that the shape is finalized.
        self.dof.add(MapZipper(self.map_rhs.shape, self.map_rhs.wcs))
        self.dof.add(ArrayZipper(self.junk_rhs.shape), distributed=True)

        self.ready = True

    def A(self, x):
        imap, ijunk = self.dof.unzip(x)
        # This is necessary because multizipper reduces everything to a single array, and
        # hence can't maintain the separation between map_dtype and tod_dtype
        ijunk = ijunk.astype(self.dtype_tod)
        omap, ojunk = imap*0, ijunk*0
        for di, data in enumerate(self.observations):
            j1, j2 = self.junk_offs[di:di+2]
            tod = np.zeros([data.ndet, data.nsamp], self.dtype_tod)
            wmap = imap.extract(data.shape, data.wcs)*1
            t1 = time.time()
            data.pmat.from_map(wmap, dest=tod)
            data.pcut.forward(tod, ijunk[j1:j2])
            t2 = time.time()
            data.nmat.apply(tod)
            t3 = time.time()
            wmap[:] = 0
            data.pcut.backward(tod, ojunk[j1:j2])
            data.pmat.to_map(signal=tod, dest=wmap)
            t4 = time.time()
            omap.insert(wmap, op=np.ndarray.__iadd__)
            print("A %-70s P %8.3f N %8.3f P' %8.3f  %3d %6d" % (data.id, t2-t1, t3-t2, t4-t3, data.ndet, data.nsamp))
        omap = utils.allreduce(omap,self.comm)
        return self.dof.zip(omap, ojunk)
    def M(self, x):
        map, junk = self.dof.unzip(x)
        map = enmap.map_mul(self.map_idiv, map)
        return self.dof.zip(map, junk/self.junk_div)
    def solve(self, maxiter=500, maxerr=1e-6):
        self.prepare()
        rhs    = self.dof.zip(self.map_rhs, self.junk_rhs)
        solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot)
        while True or solver.i < maxiter and solver.err > maxerr:
            solver.step()
            yield bunch.Bunch(i=solver.i, err=solver.err, x=self.dof.unzip(solver.x)[0])

class ArrayZipper:
    def __init__(self, shape):
        self.shape = shape
        self.ndof  = int(np.product(shape))
    def zip(self, arr):  return arr.reshape(-1)
    def unzip(self, x):  return x.reshape(self.shape)
    def dot(self, a, b): return np.sum(a*b)

class MapZipper:
    def __init__(self, shape, wcs):
        self.shape, self.wcs = shape, wcs
        self.ndof  = int(np.product(shape))
    def zip(self, map): return np.asarray(map.reshape(-1))
    def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs)
    def dot(self, a, b): return np.sum(a*b)

class MultiZipper:
    def __init__(self, comm=mpi.COMM_WORLD):
        self.comm    = comm
        self.zippers = []
        self.dist    = []
        self.ndof    = 0
        self.bins    = []
    def add(self, zipper, distributed=False):
        self.zippers.append(zipper)
        self.dist.append(distributed)
        self.bins.append([self.ndof, self.ndof+zipper.ndof])
        self.ndof += zipper.ndof
    def zip(self, *objs):
        return np.concatenate([zipper.zip(obj) for zipper, obj in zip(self.zippers, objs)])
    def unzip(self, x):
        res = []
        for zipper, (b1,b2) in zip(self.zippers, self.bins):
            res.append(zipper.unzip(x[b1:b2]))
        return res
    def dot(self, a, b):
        res_dist, res_undist = 0,0
        for (b1,b2), dist in zip(self.bins, self.dist):
            work = np.sum(a[b1:b2]*b[b1:b2])
            if dist: res_dist   += work
            else:    res_undist += work
        res_dist = self.comm.allreduce(res_dist)
        return res_dist + res_undist

class PmatCut:
    """Implementation of cuts-as-extra-degrees-of-freedom for a single obs."""
    def __init__(self, cuts, model="full", params={"resolution":100, "nmax":100}):
        self.cuts   = cuts
        self.model  = model
        self.params = params
        self.njunk  = so3g.process_cuts(self.cuts.ranges, "measure", self.model, self.params, None, None)
    def forward(self, tod, junk):
        """Project from the cut parameter (junk) space for this scan to tod."""
        so3g.process_cuts(self.cuts.ranges, "insert", self.model, self.params, tod, junk)
    def backward(self, tod, junk):
        """Project from tod to cut parameters (junk) for this scan."""
        so3g.process_cuts(self.cuts.ranges, "extract", self.model, self.params, tod, junk)
        self.clear(tod)
    def clear(self, tod):
        junk = np.empty(self.njunk, tod.dtype)
        so3g.process_cuts(self.cuts.ranges, "clear", self.model, self.params, tod, junk)

def inject_map(obs, map, recenter=None):
    # Infer the stokes components
    map = map.preflat
    if map.shape[0] not in [1,2,3]:
        raise ValueError("Map to inject must have either 1, 2 or 3 components, corresponding to T, QU and TQU.")
    comps = infer_comps(map.shape[0])
    # Support recentering the coordinate system
    if recenter is not None:
        ctime  = obs.timestamps
        rot    = recentering_to_quat_lonlat(*evaluate_recentering(recenter, ctime=ctime[len(ctime)//2], geom=(map.shape, map.wcs), site=SITE))
    else: rot = None
    # Set up our pointing matrix for the map
    pmat  = coords.pmat.P.for_tod(obs, comps=comps, geom=(map.shape, map.wcs), rot=rot, threads="domdir")
    # And perform the actual injection
    pmat.from_map(map.extract(shape, wcs), dest=obs.signal)

def safe_invert_div(div, lim=1e-2):
    hit = div[0,0] != 0
    # Get the condition number of each pixel
    work    = np.ascontiguousarray(div[:,:,hit].T)
    E, V    = np.linalg.eigh(work)
    cond    = E[:,0]/E[:,-1]
    good    = cond >= lim
    # Invert the good ones
    inv_good= np.einsum("...ij,...j,...kj->...ik", V[good], 1/E[good], V[good])
    # Treat the bad ones as being purely T
    inv_bad = work[~good]*0
    inv_bad[:,0,0] = 1/work[~good,0,0]
    # Copy back
    work[good]  = inv_good
    work[~good] = inv_bad
    # And put into final output
    idiv = div*0
    idiv[:,:,hit] = work.T
    return idiv

################################
####### Noise model stuff ######
################################

class Nmat:
    def __init__(self): self.ivar = 1.0
    def build(self, tod, **kwargs): pass
    def apply(self, tod): return tod

class NmatUncorr(Nmat):
    def __init__(self, spacing="exp", nbin=100, nmin=10, bins=None, ips_binned=None, ivar=None):
        self.spacing    = spacing
        self.nbin       = nbin
        self.nmin       = nmin
        self.bins       = bins
        self.ips_binned = ips_binned
        self.ivar       = ivar
    def build(self, tod, **kwargs):
        ps = np.abs(fft.rfft(tod))**2
        if   self.spacing == "exp": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        elif self.spacing == "lin": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        else: raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
        ps_binned  = utils.bin_data(bins, ps)
        ips_binned = 1/ps_binned
        # Compute the representative inverse variance per sample
        ivar = np.zeros(len(tod))
        for bi, b in enumerate(bins):
            ivar += ips_binned[:,bi]*(b[1]-b[0])
        ivar /= bins[-1,1]-bins[0,0]
        ivar *= tod.shape[1]
        return NmatUncorr(spacing=self.spacing, nbin=len(bins), nmin=self.nmin, bins=bins, ips_binned=ips_binned, ivar=ivar)
    def apply(self, tod, inplace=False):
        if inplace: tod = np.array(tod)
        ftod = fft.rfft(tod)
        # Candidate for speedup in C
        norm = tod.shape[1]
        for bi, b in enumerate(self.bins):
            ftod[:,b[0]:b[1]] *= self.ips_binned[:,None,bi]/norm
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed
        fft.irfft(ftod, tod)
        return tod

class NmatDetvecs(Nmat):
    def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
            downweight=None, window=0, nwin=None, verbose=False, bins=None, D=None, V=None, iD=None, iV=None, s=None, ivar=None):
        # This is all taken from act, not tuned to so yet
        if bin_edges is None: bin_edges = np.array([
            0.10, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
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
        self.D, self.V, self.iD, self.iV, self.s, self.ivar = D, V, iD, iV, s, ivar
    def build(self, tod, srate, **kwargs):
        # Apply window before measuring noise model
        nwin  = utils.nint(self.window/srate)
        apply_window(tod, nwin)
        ft    = fft.rfft(tod)
        # Unapply window again
        apply_window(tod, nwin, -1)
        ndet, nfreq = ft.shape
        nsamp = tod.shape[1]
        # First build our set of eigenvectors in two bins. The first goes from
        # 0.25 to 4 Hz the second from 4Hz and up
        mode_bins = makebins(self.mode_bins, srate, nfreq, 1000, rfun=np.round)[1:]
        # Then use these to get our set of basis vectors
        vecs = find_modes_jon(ft, mode_bins, eig_lim=self.eig_lim, single_lim=self.single_lim, verbose=self.verbose)
        nmode= vecs.shape[1]
        if vecs.size == 0: raise errors.ModelError("Could not find any noise modes")
        # Cut bins that extend beyond our max frequency
        bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
        bins      = makebins(bin_edges, srate, nfreq, nmin=2*nmode, rfun=np.round)
        nbin      = len(bins)
        # Now measure the power of each basis vector in each bin. The residual
        # noise will be modeled as uncorrelated
        E  = np.zeros([nbin,nmode])
        D  = np.zeros([nbin,ndet])
        Nd = np.zeros([nbin,ndet])
        for bi, b in enumerate(bins):
            # Skip the DC mode, since it's it's unmeasurable and filtered away
            b = np.maximum(1,b)
            E[bi], D[bi], Nd[bi] = measure_detvecs(ft[:,b[0]:b[1]], vecs)
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
        D    = np.ascontiguousarray(iD.astype(tod.dtype))
        V    = np.ascontiguousarray(iV.astype(tod.dtype))
        iD   = np.ascontiguousarray(D.astype(tod.dtype))
        iV   = np.ascontiguousarray(V.astype(tod.dtype))

        return NmatDetvecs(bin_edges=self.bin_edges, eig_lim=self.eig_lim, single_lim=self.single_lim,
                window=self.window, nwin=nwin, downweight=self.downweight, verbose=self.verbose,
                bins=bins, D=D, V=V, iD=iD, iV=iV, s=s, ivar=ivar)
    def apply(self, tod, inplace=False, slow=False):
        if inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        norm = tod.shape[1]
        if slow:
            for bi, b in enumerate(self.bins):
                # Want to multiply by iD + siViV'
                ft    = ftod[:,b[0]:b[1]]
                iD    = self.iD[bi]/norm
                iV    = self.iV[bi]/norm**0.5
                ft[:] = iD[:,None]*ft + self.s*iV.dot(iV.T.dot(ft))
        else:
            so3g.nmat_detvecs_apply(ftod.view(tod.dtype), self.bins, self.iD, self.iV, self.s, norm)
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        return tod

def measure_cov(d, nmax=10000):
    d = d[:,::max(1,d.shape[1]//nmax)]
    n,m = d.shape
    step  = 10000
    res = np.zeros((n,n),d.dtype)
    for i in range(0,m,step):
        sub = mycontiguous(d[:,i:i+step])
        res += np.real(sub.dot(np.conj(sub.T)))
    return res/m
def project_out(d, modes): return d-modes.T.dot(modes.dot(d))
def project_out_from_matrix(A, V):
    # Used Woodbury to project out the given vectors from the covmat A
    if V.size == 0: return A
    Q = A.dot(V)
    return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))
def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))
def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
    binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
    if nmin > 0:
        binds2 = [binds[0]]
        for b in binds:
            if b-binds2[-1] >= nmin: binds2.append(b)
        binds = binds2
    return np.array([np.concatenate([[0],binds]),np.concatenate([binds,[nfreq]])]).T
def mycontiguous(a):
    # I used this in act for some reason, but not sure why. I vaguely remember ascontiguousarray
    # causing weird failures later in lapack
    b = np.zeros(a.shape, a.dtype)
    b[...] = a[...]
    return b
def find_modes_jon(ft, bins, eig_lim=None, single_lim=0, skip_mean=False, verbose=False):
    ndet = ft.shape[0]
    vecs = np.zeros([ndet,0])
    if not skip_mean:
        # Force the uniform common mode to be included. This
        # assumes all the detectors have accurately measured gain.
        # Forcing this avoids the possibility that we don't find
        # any modes at all.
        vecs = np.concatenate([vecs,np.full([ndet,1],ndet**-0.5)],1)
    for bi, b in enumerate(bins):
        d    = ft[:,b[0]:b[1]]
        cov  = measure_cov(d)
        cov  = project_out_from_matrix(cov, vecs)
        e, v = np.linalg.eig(cov)
        e, v = e.real, v.real
        #e, v = e[::-1], v[:,::-1]
        accept = np.full(len(e), True, bool)
        if eig_lim is not None:
            # Compute median, exempting modes we don't have enough data to measure
            nsamp    = b[1]-b[0]+1
            median_e = np.median(np.sort(e)[::-1][:nsamp])
            accept  &= e/median_e >= eig_lim[bi]
        if verbose: print("bin %d: %4d modes above eig_lim" % (bi, np.sum(accept)))
        if single_lim is not None and e.size:
            # Reject modes too concentrated into a single mode. Since v is normalized,
            # values close to 1 in a single component must mean that all other components are small
            singleness = np.max(np.abs(v),0)
            accept    &= singleness < single_lim[bi]
        if verbose: print("bin %d: %4d modes also above single_lim" % (bi, np.sum(accept)))
        e, v = e[accept], v[:,accept]
        vecs = np.concatenate([vecs,v],1)
    return vecs
def measure_detvecs(ft, vecs):
    # Measure amps when we have non-orthogonal vecs
    rhs  = vecs.T.dot(ft)
    div  = vecs.T.dot(vecs)
    amps = np.linalg.solve(div,rhs)
    E    = np.mean(np.abs(amps)**2,1)
    # Project out modes for every frequency individually
    dclean = ft - vecs.dot(amps)
    # The rest is assumed to be uncorrelated
    Nu = np.mean(np.abs(dclean)**2,1)
    # The total auto-power
    Nd = np.mean(np.abs(ft)**2,1)
    return E, Nu, Nd
def sichol(A):
    iA = np.linalg.inv(A)
    try: return np.linalg.cholesky(iA), 1
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(-iA), -1
def safe_inv(a):
    with utils.nowarn():
        res = 1/a
        res[~np.isfinite(res)] = 0
    return res
def woodbury_invert(D, V, s=1):
    """Given a compressed representation C = D + sVV', compute a
    corresponding representation for inv(C) using the Woodbury
    formula."""
    V, D = map(np.asarray, [V,D])
    ishape = D.shape[:-1]
    # Flatten everything so we can be dimensionality-agnostic
    D = D.reshape(-1, D.shape[-1])
    V = V.reshape(-1, V.shape[-2], V.shape[-1])
    I = np.eye(V.shape[2])
    # Allocate our output arrays
    iD = safe_inv(D)
    iV = V*0
    # Invert each
    for i in range(len(D)):
        core = I*s + (V[i].T*iD[i,None,:]).dot(V[i])
        core, sout = sichol(core)
        iV[i] = iD[i,:,None]*V[i].dot(core)
    sout = -sout
    return iD, iV, sout
def apply_window(tod, nsamp, exp=1):
    """Apply a cosine taper to each end of the TOD."""
    if nsamp <= 0: return
    taper   = 0.5*(1-np.cos(np.arange(1,nsamp+1)*np.pi/nsamp))
    taper **= exp
    tod[...,:nsamp]  *= taper
    tod[...,-nsamp:] *= taper[::-1]

########################
###### Utilities #######
########################

def get_ids(query):
    try:
        with open(query, "r") as fname:
            return [line.split()[0] for line in fname]
    except IOError:
        return context.obsdb.query(query)['obs_id']

def infer_comps(ncomp): return ["T","QU","TQU"][ncomp-1]

def parse_recentering(desc):
    """Parse an object centering description, as provided by the --center-at argument.
    The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]
    from: specifies which point is to be centered. Given as either
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object (e.g. Saturn), which should not move
        appreciably in celestial coordinates during a TOD
    to: the point at which to recenter. Optional. Given as either
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object
      Defaults to ra=0,dec=0 or ra=0,dec=90, depending on the projection
    up: which direction should point up after recentering. Optional. Given as either
      * the name of a coordinate system (e.g. hor, cel, gal), in which case
        up will point towards the north pole of that system
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object
      Defualts to the celestial north pole

    Returns "info", a bunch representing the recentering specification in more python-friendly
    terms. This can later be passed to evaluate_recentering to get the actual euler angles that perform
    the recentering.

    Examples:
      * 120.2:-13.8
        Centers on ra = 120.2°, dec = -13.8°, with up being celestial north
      * Saturn
        Centers on Saturn, with up being celestial north
      * Uranus,up=hor
        Centers on Uranus, but up is up in horizontal coordinates. Appropriate for beam mapping
      * Uranus,up=hor,to=0:90
        As above, but explicitly recenters on the north pole
    """
    # If necessary the syntax above could be extended with from_sys, to_sys and up-sys, which
    # so one could specify galactic coordiantes for example. Or one could generalize
    # from ra:dec to phi:theta[:sys], where sys would default to cel. But for how I think
    # this is enough.
    args = desc.split(",")
    info  = {"to":"auto", "up":"cel", "from_sys":"cel", "to_sys":"cel", "up_sys":"cel"}
    for ai, arg in enumerate(args):
        # Split into key,value
        toks = arg.split("=")
        if ai == 0 and len(toks) == 1:
            key, val = "from", toks[0]
        elif len(toks) == 2:
            key, val = toks
        else:
            raise ValueError("parse_recentering wants key=value format, but got %s" % (arg))
        # Handle the values
        if ":" in val:
            val = [float(w)*utils.degree for w in val.split(":")]
        info[key] = val
    if "from" not in info:
        raise ValueError("parse_recentering needs at least the from argument")
    return info

def evaluate_recentering(info, ctime, geom=None, site=None, weather="typical"):
    """Evaluate the quaternion that performs the coordinate recentering specified in
    info, which can be obtained from parse_recentering."""
    import ephem
    # Get the coordinates of the from, to and up points. This was a bit involved...
    def to_cel(lonlat, sys, ctime=None, site=None, weather=None):
        # Convert lonlat from sys to celestial coorinates. Maybe polish and put elswhere
        if sys == "cel" or sys == "equ":
            return lonlat
        elif sys == "hor":
            return so3g.proj.CelestialSightLine.az_el(ctime, lonlat[0], lonlat[1], site=site, weather=weather).coords()[0,:2]
        else:
            raise NotImplementedError
    def get_pos(name, ctime, sys=None):
        if isinstance(name, str):
            if name in ["hor", "cel", "equ", "gal"]:
                return to_cel([0,np.pi/2], name, ctime, site, weather)
            elif name == "auto":
                return np.array([0,0]) # would use geom here
            else:
                obj = getattr(ephem, name)()
                djd = ctime/86400 + 40587.0 + 2400000.5 - 2415020
                obj.compute(djd)
                return np.array([obj.a_ra, obj.a_dec])
        else:
            return to_cel(name, sys, ctime, site, weather)
    p1 = get_pos(info["from"], ctime, info["from_sys"])
    p2 = get_pos(info["to"],   ctime, info["to_sys"])
    pu = get_pos(info["up"],   ctime, info["up_sys"])
    return [p1,p2,pu]

def recentering_to_quat_lonlat(p1, p2, pu):
    """Return the quaternion that represents the rotation that takes point p1
    to p2, with the up direction pointing towards the point pu, all given as lonlat pairs"""
    from so3g.proj import quat
    # 1. First rotate our point to the north pole: Ry(-(90-dec1))Rz(-ra1)
    # 2. Apply the same rotation to the up point.
    # 3. We want the up point to be upwards, so rotate it to ra = 180°: Rz(pi-rau2)
    # 4. Apply the same rotation to the real point
    # 5. Rotate the point to its target position: Rz(ra2)Ry(90-dec2)
    ra1, dec1 = p1
    ra2, dec2 = p2
    rau, decu = pu
    qu    = quat.rotation_lonlat(rau, decu)
    R     = ~quat.rotation_lonlat(ra1, dec1)
    rau2  = quat.decompose_lonlat(R*qu)[0]
    R     = quat.euler(2, ra2)*quat.euler(1, np.pi/2-dec2)*quat.euler(2, np.pi-rau2)*R
    a = quat.decompose_lonlat(R*quat.rotation_lonlat(ra1,dec1))
    return R

def highpass(tod, fknee=1e-2, alpha=3):
    ft   = fft.rfft(tod)
    freq = fft.rfftfreq(tod.shape[1])
    ft  /= 1 + (freq/fknee)**-alpha
    return fft.irfft(ft, tod, normalize=True)

def find_boresight_jumps(vals, width=20, tol=0.1):
    # median filter array to get reference behavior
    bad   = np.zeros(vals.size,dtype=bool)
    width = int(width)//2*2+1
    fvals = utils.block_mean_filter(vals, width)
    bad  |= np.abs(vals-fvals) > tol
    return bad

def robust_unwind(a, period=2*np.pi, cut=None, tol=1e-3, mask=None):
    """Like utils.unwind, but only registers something as an angle jump if
    it is of just the right shape. If cut is specified, it should be a list
    of valid angle cut positions, which will further restrict when jumps are
    allowed. Only 1d input is supported."""
    # Find places where a jump would be acceptable
    period = float(period)
    diffs  = (a[1:]-a[:-1])/period
    valid  = np.abs(np.abs(diffs)-1) < tol/period
    diffs *= valid
    jumps  = np.concatenate([[0],np.round(diffs)])
    if mask is not None:
        jumps[mask] = 0
        jumps[:-1][mask[1:]] = 0
    if cut is not None:
        near_cut = np.zeros(a.size, bool)
        for cutval in cut:
            near_cut |= np.abs((a - cutval + period/2) % period + period/2) < tol
        jumps[~near_cut] = 0
    # Then correct our values
    return a - np.cumsum(jumps)*period

def find_elevation_outliers(el, tol=0.5*utils.degree):
    typ = np.median(el[::100])
    return np.abs(el-typ)>tol

def freq2ind(freqs, srate, nfreq, rfun=None):
    """Returns the index of the first fourier mode with greater than freq
    frequency, for each freq in freqs."""
    if freqs is None: return freqs
    if rfun  is None: rfun = np.ceil
    return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def rangemat_sum(rangemat):
    res = np.zeros(len(rangemat))
    for i, r in enumerate(rangemat):
        ra = r.ranges()
        res[i] = np.sum(ra[:,1]-ra[:,0])
    return res

def find_usable_detectors(obs, maxcut=0.1):
    ncut  = rangemat_sum(obs.glitch_flags)
    good  = ncut < obs.samps.count * maxcut
    return obs.dets.vals[good]

def fix_boresight_glitches(obs, ang_tol=0.1*utils.degree, t_tol=1):
    az   = robust_unwind(obs.boresight.az)
    bad  = find_boresight_jumps(az,               tol=ang_tol)
    bad |= find_boresight_jumps(obs.boresight.el, tol=ang_tol)
    bad |= find_boresight_jumps(obs.timestamps,   tol=t_tol)
    bcut = so3g.RangesInt32.from_mask(bad)
    obs.boresight.az[:] = az
    tod_ops.get_gap_fill_single(obs.timestamps,   bcut, swap=True)
    tod_ops.get_gap_fill_single(obs.boresight.az, bcut, swap=True)
    tod_ops.get_gap_fill_single(obs.boresight.el, bcut, swap=True)
