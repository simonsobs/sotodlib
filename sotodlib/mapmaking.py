# Module with functions for mapmaking. It could have been split into many sub-modules,
# e.g. noise_model.py, pointing_matrix.py, utilities, mlmapmaker.py etc. Maybe we will
# do that later, but for now I don't think that split makes things easier for the user.

from time import time
import sys
import warnings

import numpy as np
import so3g
from pixell import enmap, utils, fft, bunch, tilemap, resample

from . import core
from . import coords
from . import tod_ops

##########################################
##### Maximum likelihood mapmaking #######
##########################################

def deslope_el(tod, el, srate, inplace=False):
    if not inplace: tod = tod.copy()
    utils.deslope(tod, w=1, inplace=True)
    f     = fft.rfftfreq(tod.shape[-1], 1/srate)
    fknee = 3.0
    with utils.nowarn():
        iN = (1+(f/fknee)**-3.5)**-1
    b  = 1/np.sin(el)
    utils.deslope(b, w=1, inplace=True)
    Nb = fft.irfft(iN*fft.rfft(b),b*0)
    amp= np.sum(tod*Nb,-1)/np.sum(b*Nb)
    tod-= amp[:,None]*b
    return tod

class MLMapmaker:
    def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False):
        """Initialize a Maximum Likelihood Mapmaker.
        Arguments:
        * signals: List of Signal-objects representing the models that will be solved
          jointly for. Typically this would be the sky map and the cut samples. NB!
          The way the cuts currently work, they *MUST* be the first signal specified.
          If not, the equation system will be inconsistent and won't converge.
        * noise_model: A noise model constructor which will be used to initialize the
          noise model for each observation. Can be overriden in add_obs.
        * dtype: The data type to use for the time-ordered data. Only tested with float32
        * verbose: Whether to print progress messages. Not implemented"""
        if noise_model is None:
            noise_model = NmatUncorr()
        self.signals      = signals
        self.dtype        = dtype
        self.verbose      = verbose
        self.noise_model  = noise_model
        self.data         = []
        self.dof          = MultiZipper()
        self.ready        = False

    def add_obs(self, id, obs, deslope=True, noise_model=None):
        # Prepare our tod
        ctime  = obs.timestamps
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        tod    = obs.signal.astype(self.dtype, copy=False)
        if deslope:
            utils.deslope(tod, w=5, inplace=True)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None: noise_model = self.noise_model
        # Build the noise model from the obs unless a fully
        # initialized noise model was passed
        if noise_model.ready:
            nmat = noise_model
        else:
            try:
                nmat = noise_model.build(tod, srate=srate)
            except Exception as e:
                msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
                raise RuntimeError(msg)
        # And apply it to the tod
        tod    = nmat.apply(tod)
        # Add the observation to each of our signals
        for signal in self.signals:
            signal.add_obs(id, obs, nmat, tod)
        # Save what we need about this observation
        self.data.append(bunch.Bunch(id=id, ndet=obs.dets.count, nsamp=len(ctime),
            dets=obs.dets.vals, nmat=nmat))

    def prepare(self):
        if self.ready: return
        for signal in self.signals:
            signal.prepare()
            self.dof.add(signal.dof)
        self.ready = True

    def A(self, x):
        # unzip goes from flat array of all the degrees of freedom to individual maps, cuts etc.
        # to_work makes a scratch copy and does any redistribution needed
        #t0 = time()
        #t1 = time()
        iwork = [signal.to_work(m) for signal,m in zip(self.signals,self.dof.unzip(x))]
        #t2 = time(); print(f" A    iwork : {t2-t1:8.3f}s", flush=True)
        owork = [w*0 for w in iwork]
        #t1 = time(); print(f" A    owork : {t1-t2:8.3f}s", flush=True)
        #t_forward = 0
        #t_apply = 0
        #t_backward = 0
        for di, data in enumerate(self.data):
            tod = np.zeros([data.ndet, data.nsamp], self.dtype)
            #t1 = time()
            for si, signal in reversed(list(enumerate(self.signals))):
                signal.forward(data.id, tod, iwork[si])
            #t2 = time()
            #t_forward += t2 - t1
            data.nmat.apply(tod)
            #t1 = time()
            #t_apply += t1 - t2
            for si, signal in enumerate(self.signals):
                signal.backward(data.id, tod, owork[si])
            #t2 = time()
            #t_backward += t2 - t1
        #print(f" A  forward : {t_forward:8.3f}s", flush=True)
        #print(f" A    apply : {t_apply:8.3f}s", flush=True)
        #print(f" A backward : {t_backward:8.3f}s", flush=True)
        #t1 = time()
        result = self.dof.zip(*[signal.from_work(w) for signal,w in zip(self.signals,owork)])
        #t2 = time(); print(f" A      zip : {t2-t1:8.3f}s", flush=True)
        #print(f" A    TOTAL : {t2-t0:8.3f}s", flush=True)
        return result

    def M(self, x):
        #t1 = time()
        iwork = self.dof.unzip(x)
        #t2 = time(); print(f" M    iwork : {t2-t1:8.3f}s", flush=True)
        result = self.dof.zip(*[signal.precon(w) for signal, w in zip(self.signals, iwork)])
        #t1 = time(); print(f" M      zip : {t1-t2:8.3f}s", flush=True)
        return result

    def solve(self, maxiter=500, maxerr=1e-6):
        self.prepare()
        rhs    = self.dof.zip(*[signal.rhs for signal in self.signals])
        solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot)
        while solver.i < maxiter and solver.err > maxerr:
            solver.step()
            yield bunch.Bunch(i=solver.i, err=solver.err, x=self.dof.unzip(solver.x))

class Signal:
    """This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""
    def __init__(self, name, ofmt, output, ext):
        """Initialize a Signal. It probably doesn't make sense to construct a generic signal
        directly, though. Use one of the subclasses.
        Arguments:
        * name: The name of this signal, e.g. "sky", "cut", etc.
        * ofmt: The format used when constructing output file prefix
        * output: Whether this signal should be part of the output or not.
        * ext: The extension used for the files.
        """
        self.name   = name
        self.ofmt   = ofmt
        self.output = output
        self.ext    = ext
        self.dof    = None
        self.ready  = False
    def add_obs(self, id, obs, nmat, Nd): pass
    def prepare(self): self.ready = True
    def forward (self, id, tod, x): pass
    def backward(self, id, tod, x): pass
    def precon(self, x): return x
    def to_work  (self, x): return x.copy()
    def from_work(self, x): return x
    def write   (self, prefix, tag, x): pass

class SignalMap(Signal):
    """Signal describing a non-distributed sky map."""
    def __init__(self, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False,
            interpol="nearest"):
        """Signal describing a sky map in the coordinate system given by "sys", which defaults
        to equatorial coordinates. If tiled==True, then this will be a distributed map with
        the given tile_shape, otherwise it will be a plain enmap."""
        Signal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
        self.interpol = interpol
        self.data  = {}
        ncomp      = len(comps)
        shape      = tuple(shape[-2:])
        if tiled:
            geo = tilemap.geometry(shape, wcs, tile_shape=tile_shape)
            self.rhs = tilemap.zeros(geo.copy(pre=(ncomp,)),      dtype=dtype)
            self.div = tilemap.zeros(geo.copy(pre=(ncomp,ncomp)), dtype=dtype)
            self.hits= tilemap.zeros(geo.copy(pre=()),            dtype=dtype)
        else:
            self.rhs = enmap.zeros((ncomp,)     +shape, wcs, dtype=dtype)
            self.div = enmap.zeros((ncomp,ncomp)+shape, wcs, dtype=dtype)
            self.hits= enmap.zeros(              shape, wcs, dtype=dtype)

    def add_obs(self, id, obs, nmat, Nd, pmap=None):
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        Nd     = Nd.copy() # This copy can be avoided if build_obs is split into two parts
        ctime  = obs.timestamps
        pcut   = PmatCut(obs.glitch_flags) # could pass this in, but fast to construct
        if pmap is None:
            # Build the local geometry and pointing matrix for this observation
            if self.recenter:
                rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter,
                    ctime=ctime[len(ctime)//2], geom=(self.rhs.shape, self.rhs.wcs), site=unarr(obs.site)))
            else: rot = None
            pmap = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry,
                rot=rot, threads="domdir", weather=unarr(obs.weather), site=unarr(obs.site),
                interpol=self.interpol)
        # Build the RHS for this observation
        pcut.clear(Nd)
        obs_rhs = pmap.zeros()
        pmap.to_map(dest=obs_rhs, signal=Nd)
        # Build the per-pixel inverse covmat for this observation
        obs_div    = pmap.zeros(super_shape=(self.ncomp,self.ncomp))
        for i in range(self.ncomp):
            obs_div[i]   = 0
            obs_div[i,i] = 1
            Nd[:]        = 0
            pmap.from_map(obs_div[i], dest=Nd)
            pcut.clear(Nd)
            Nd = nmat.white(Nd)
            obs_div[i]   = 0
            pmap.to_map(signal=Nd, dest=obs_div[i])
        # Build hitcount
        Nd[:] = 1
        pcut.clear(Nd)
        obs_hits = pmap.to_map(signal=Nd)
        del Nd

        # Update our full rhs and div. This works for both plain and distributed maps
        self.rhs = self.rhs.insert(obs_rhs, op=np.ndarray.__iadd__)
        self.div = self.div.insert(obs_div, op=np.ndarray.__iadd__)
        self.hits= self.hits.insert(obs_hits[0],op=np.ndarray.__iadd__)
        # Save the per-obs things we need. Just the pointing matrix in our case.
        # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
        self.data[id] = bunch.Bunch(pmap=pmap, obs_geo=obs_rhs.geometry)

    def prepare(self):
        """Called when we're done adding everything. Sets up the map distribution,
        degrees of freedom and preconditioner."""
        if self.ready: return
        if self.tiled:
            self.geo_work = self.rhs.geometry
            self.rhs  = tilemap.redistribute(self.rhs, self.comm)
            self.div  = tilemap.redistribute(self.div, self.comm)
            self.hits = tilemap.redistribute(self.hits,self.comm)
            self.dof  = TileMapZipper(self.rhs.geometry, dtype=self.dtype, comm=self.comm)
        else:
            if self.comm is not None:
                self.rhs  = utils.allreduce(self.rhs, self.comm)
                self.div  = utils.allreduce(self.div, self.comm)
                self.hits = utils.allreduce(self.hits,self.comm)
            self.dof  = MapZipper(*self.rhs.geometry, dtype=self.dtype)
        self.idiv  = safe_invert_div(self.div)
        self.ready = True

    @property
    def ncomp(self): return len(self.comps)

    def forward(self, id, tod, map, tmul=1, mmul=1):
        """map2tod operation. For tiled maps, the map should be in work distribution,
        as returned by unzip. Adds into tod."""
        if id not in self.data: return # Should this really skip silently like this?
        if tmul != 1: tod *= tmul
        if mmul != 1: map = map*mmul
        self.data[id].pmap.from_map(dest=tod, signal_map=map, comps=self.comps)

    def backward(self, id, tod, map, tmul=1, mmul=1):
        """tod2map operation. For tiled maps, the map should be in work distribution,
        as returned by unzip. Adds into map"""
        if id not in self.data: return
        if tmul != 1: tod  = tod*tmul
        if mmul != 1: map *= mmul
        self.data[id].pmap.to_map(signal=tod, dest=map, comps=self.comps)

    def precon(self, map):
        if self.tiled: return tilemap.map_mul(self.idiv, map)
        else: return enmap.map_mul(self.idiv, map)

    def to_work(self, map):
        if self.tiled: return tilemap.redistribute(map, self.comm, self.geo_work.active)
        else: return map.copy()

    def from_work(self, map):
        if self.tiled:
            return tilemap.redistribute(map, self.comm, self.rhs.geometry.active)
        else:
            if self.comm is None: return map
            else: return utils.allreduce(map, self.comm)

    def write(self, prefix, tag, m):
        if not self.output: return
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if self.tiled:
            tilemap.write_map(oname, m, self.comm)
        else:
            if self.comm is None or self.comm.rank == 0:
                enmap.write_map(oname, m)
        return oname

class SignalCut(Signal):
    def __init__(self, comm, name="cut", ofmt="{name}_{rank:02}", dtype=np.float32,
            output=False, cut_type=None):
        """Signal for handling the ML solution for the values of the cut samples."""
        Signal.__init__(self, name, ofmt, output, ext="hdf")
        self.comm  = comm
        self.data  = {}
        self.dtype = dtype
        self.cut_type = cut_type
        self.off   = 0
        self.rhs   = []
        self.div   = []

    def add_obs(self, id, obs, nmat, Nd):
        """Add and process an observation. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data."""
        Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
        pcut    = PmatCut(obs.glitch_flags, model=self.cut_type)
        # Build our RHS
        obs_rhs = np.zeros(pcut.njunk, self.dtype)
        pcut.backward(Nd, obs_rhs)
        # Build our per-pixel inverse covmat
        obs_div = np.ones(pcut.njunk, self.dtype)
        Nd[:]     = 0
        pcut.forward(Nd, obs_div)
        Nd       *= nmat.ivar[:,None]
        pcut.backward(Nd, obs_div)
        self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.njunk)
        self.off += pcut.njunk
        self.rhs.append(obs_rhs)
        self.div.append(obs_div)

    def prepare(self):
        """Process the added observations, determining our degrees of freedom etc.
        Should be done before calling forward and backward."""
        if self.ready: return
        self.rhs = np.concatenate(self.rhs)
        self.div = np.concatenate(self.div)
        self.dof = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
        self.ready = True

    def forward(self, id, tod, junk):
        if id not in self.data: return
        d = self.data[id]
        d.pcut.forward(tod, junk[d.i1:d.i2])

    def precon(self, junk):
        return junk/self.div

    def backward(self, id, tod, junk):
        if id not in self.data: return
        d = self.data[id]
        d.pcut.backward(tod, junk[d.i1:d.i2])

    def write(self, prefix, tag, m):
        if not self.output: return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank
        oname = self.ofmt.format(name=self.name, rank=rank)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        with h5py.File(oname, "w") as hfile:
            hfile["data"] = m
        return oname

class ArrayZipper:
    def __init__(self, shape, dtype, comm=None):
        self.shape = shape
        self.ndof  = int(np.product(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, arr):  return arr.reshape(-1)

    def unzip(self, x):  return x.reshape(self.shape).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else self.comm.allreduce(np.sum(a*b))

class MapZipper:
    def __init__(self, shape, wcs, dtype, comm=None):
        self.shape, self.wcs = shape, wcs
        self.ndof  = int(np.product(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, map): return np.asarray(map.reshape(-1))

    def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b),self.comm)

class TileMapZipper:
    def __init__(self, geo, dtype, comm):
        self.geo   = geo
        self.comm  = comm
        self.dtype = dtype
        self.ndof  = geo.size

    def zip(self, map):
        return np.asarray(map.reshape(-1))

    def unzip(self, x):
        return tilemap.TileMap(x.reshape(self.geo.pre+(-1,)).astype(self.dtype, copy=False), self.geo)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b),self.comm)

class MultiZipper:
    def __init__(self):
        self.zippers = []
        self.ndof    = 0
        self.bins    = []

    def add(self, zipper):
        self.zippers.append(zipper)
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
        res = 0
        for (b1,b2), dof in zip(self.bins, self.zippers):
            res += dof.dot(a[b1:b2],b[b1:b2])
        return res

class PmatCut:
    """Implementation of cuts-as-extra-degrees-of-freedom for a single obs."""
    def __init__(self, cuts, model=None, params={"resolution":100, "nmax":100}):
        self.cuts   = cuts
        self.model  = model or "full"
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

def inject_map(obs, map, recenter=None, interpol="nearest"):
    # Infer the stokes components
    map = map.preflat
    if map.shape[0] not in [1,2,3]:
        raise ValueError("Map to inject must have either 1, 2 or 3 components, corresponding to T, QU and TQU.")
    comps = infer_comps(map.shape[0])
    # Support recentering the coordinate system
    if recenter is not None:
        ctime  = obs.timestamps
        rot    = recentering_to_quat_lonlat(*evaluate_recentering(recenter, ctime=ctime[len(ctime)//2], geom=(map.shape, map.wcs), site=unarr(obs.site)))
    else: rot = None
    # Set up our pointing matrix for the map
    pmat  = coords.pmat.P.for_tod(obs, comps=comps, geom=(map.shape, map.wcs), rot=rot, threads="domdir", interpol=self.interpol)
    # And perform the actual injection
    pmat.from_map(map.extract(shape, wcs), dest=obs.signal)

def safe_invert_div(div, lim=1e-2, lim0=np.finfo(np.float32).tiny**0.5):
    try:
        # try setting up a context manager that limits the number of threads
        from threadpoolctl import threadpool_limitse
        cm = threadpool_limits(limits=1, user_api="blas")
    except:
        # threadpoolctl not available, need a dummy context manager
        import contextlib
        cm = contextlib.nullcontext()
    with cm:
        hit = div[0,0] > lim0
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
        return tod*self.ivar
    def white(self, tod):
        """Like apply, but without detector or time correlations"""
        return tod*self.ivar
    def write(self, fname):
        bunch.write(fname, bunch.Bunch(type="Nmat"))
    @staticmethod
    def from_bunch(data): return Nmat()

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
        ps = np.abs(ft)**2
        del ft
        if   self.spacing == "exp": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        elif self.spacing == "lin": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
        else: raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
        ps_binned  = utils.bin_data(bins, ps) / tod.shape[1]
        ips_binned = 1/ps_binned
        # Compute the representative inverse variance per sample
        ivar = np.zeros(len(tod))
        for bi, b in enumerate(bins):
            ivar += ips_binned[:,bi]*(b[1]-b[0])
        ivar /= bins[-1,1]-bins[0,0]
        return NmatUncorr(spacing=self.spacing, nbin=len(bins), nmin=self.nmin, bins=bins, ips_binned=ips_binned, ivar=ivar, window=self.window, nwin=nwin)

    def apply(self, tod, inplace=False):
        if inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        ftod = fft.rfft(tod)
        # Candidate for speedup in C
        norm = tod.shape[1]
        for bi, b in enumerate(self.bins):
            ftod[:,b[0]:b[1]] *= self.ips_binned[:,None,bi]/norm
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        return tod

    def white(self, tod, inplace=True):
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        tod *= self.ivar[:,None]
        apply_window(tod, self.nwin)
        return tod

    def write(self, fname):
        data = bunch.Bunch(type="NmatUncorr")
        for field in ["spacing", "nbin", "nmin", "bins", "ips_binned", "ivar", "window", "nwin"]:
            data[field] = getattr(self, field)
        bunch.write(fname, data)

    @staticmethod
    def from_bunch(data):
        return NmatUncorr(spacing=data.spacing, nbin=data.nbin, nmin=data.nmin, bins=data.bins, ips_binned=data.ips_binned, ivar=data.ivar, window=window, nwin=nwin)

class NmatDetvecs(Nmat):
    def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
            downweight=[], window=2, nwin=None, verbose=False, bins=None, D=None, V=None, iD=None, iV=None, s=None, ivar=None):
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
        self.D, self.V, self.iD, self.iV, self.s, self.ivar = D, V, iD, iV, s, ivar
        self.ready      = all([a is not None for a in [D, V, iD, iV, s, ivar]])

    def build(self, tod, srate, **kwargs):
        # Apply window before measuring noise model
        nwin  = utils.nint(self.window*srate)
        apply_window(tod, nwin)
        ft    = fft.rfft(tod)
        # Unapply window again
        apply_window(tod, nwin, -1)
        ndet, nfreq = ft.shape
        nsamp = tod.shape[1]
        # First build our set of eigenvectors in two bins. The first goes from
        # 0.25 to 4 Hz the second from 4Hz and up
        mode_bins = makebins(self.mode_bins, srate, nfreq, 1000, rfun=np.round)[1:]
        if np.any(np.diff(mode_bins) < 0):
            raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
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

    def apply(self, tod, inplace=True, slow=False):
        if not inplace: tod = np.array(tod)
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
            so3g.nmat_detvecs_apply(ftod.view(tod.dtype), self.bins, self.iD, self.iV, float(self.s), float(norm))
        # I divided by the normalization above instead of passing normalize=True
        # here to reduce the number of operations needed
        fft.irfft(ftod, tod)
        apply_window(tod, self.nwin)
        return tod

    def white(self, tod, inplace=True):
        if not inplace: tod = np.array(tod)
        apply_window(tod, self.nwin)
        tod *= self.ivar[:,None]
        apply_window(tod, self.nwin)
        return tod

    def write(self, fname):
        data = bunch.Bunch(type="NmatDetvecs")
        for field in ["bin_edges", "eig_lim", "single_lim", "window", "nwin", "downweight",
                "bins", "D", "V", "iD", "iV", "s", "ivar"]:
            data[field] = getattr(self, field)
        bunch.write(fname, data)

    @staticmethod
    def from_bunch(data):
        return NmatDetvecs(bin_edges=data.bin_edges, eig_lim=data.eig_lim, single_lim=data.single_lim,
                window=data.window, nwin=data.nwin, downweight=data.downweight,
                bins=data.bins, D=data.D, V=data.V, iD=data.iD, iV=data.iV, s=data.s, ivar=data.ivar)

def write_nmat(fname, nmat):
    nmat.write(fname)

def read_nmat(fname):
    data = bunch.read(fname)
    typ  = data.type.decode()
    if   typ == "NmatDetvecs": return NmatDetvecs.from_bunch(data)
    elif typ == "NmatUncorr":  return NmatUncorr .from_bunch(data)
    elif typ == "Nmat":        return Nmat       .from_bunch(data)
    else: raise IOError("Unrecognized noise matrix type '%s' in '%s'" % (str(typ), fname))

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
    # Translate from frequency to index
    binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
    # Make sure no bins have two few entries
    if nmin > 0:
        binds2 = [binds[0]]
        for b in binds:
            if b-binds2[-1] >= nmin: binds2.append(b)
        binds = binds2
    # Cap at nfreq and eliminate any resulting empty bins
    binds = np.unique(np.minimum(np.concatenate([[0],binds,[nfreq]]),nfreq))
    # Go from edges to [:,{from,to}]
    bins  = np.array([binds[:-1],binds[1:]]).T
    return bins

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

def get_ids(query, context=None):
    try:
        with open(query, "r") as fname:
            return [line.split()[0] for line in fname]
    except IOError:
        return context.obsdb.query(query or "1")['obs_id']

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

def unarr(a): return np.array(a).reshape(-1)[0]

def downsample_ranges(ranges, down):
    """Downsample either an array of ranges [:,{from,to}]
    or an so3gRangesInt32 object, by the integer factor down.
    The downsampling is inclusive: The output ranges will be
    as small as possible while fully encompassing the input ranges."""
    if isinstance(ranges, so3g.RangesInt32):
        return so3g.RangesInt32.from_array(downsample_ranges(ranges.ranges(), down), (ranges.count+down-1)//down)
    oranges = ranges.copy()
    # Lower range is simple
    oranges[:,0] //= down
    # End should be one above highest impacted index.
    oranges[:,1] = (oranges[:,1]-1)//down+1
    return oranges

def downsample_cut(cut, down):
    """Given an integer RangesMatrix respresenting samples to cut,
    return a new such RangesMatrix that describes which samples to
    cut if the timestream were to be downsampled by the integer
    factor down."""
    return so3g.proj.ranges.RangesMatrix([downsample_ranges(r,down) for r in cut.ranges])

def downsample_obs(obs, down):
    """Downsample AxisManager obs by the integer factor down.

    This implementation is quite specific and probably needs
    generalization in the future, but it should work correctly
    and efficiently for ACT-like data at least. In particular
    it uses fourier-resampling when downsampling the detector
    timestreams to avoid both aliasing noise and introducing
    a transfer function."""
    assert down == utils.nint(down), "Only integer downsampling supported, but got '%.8g'" % down
    # Compute how many samples we will end up with
    onsamp = (obs.samps.count+down-1)//down
    # Set up our output axis manager
    res    = core.AxisManager(obs.dets, core.IndexAxis("samps", onsamp))
    # Stuff without sample axes
    for key, axes in obs._assignments.items():
        if "samps" not in axes:
            val = getattr(obs, key)
            if isinstance(val, core.AxisManager):
                res.wrap(key, val)
            else:
                axdesc = [(k,v) for k,v in enumerate(axes) if v is not None]
                res.wrap(key, val, axdesc)
    # The normal sample stuff
    res.wrap("timestamps", obs.timestamps[::down], [(0, "samps")])
    bore = core.AxisManager(core.IndexAxis("samps", onsamp))
    for key in ["az", "el", "roll"]:
        bore.wrap(key, getattr(obs.boresight, key)[::down], [(0, "samps")])
    res.wrap("boresight", bore)
    res.wrap("signal", resample.resample_fft_simple(obs.signal, onsamp), [(0,"dets"),(1,"samps")])
    # The cuts
    for key in ["glitch_flags", "source_flags"]:
        res.wrap(key, downsample_cut(getattr(obs, key), down), [(0,"dets"),(1,"samps")])
    # Not sure how to deal with flags. Some sort of or-binning operation? But it
    # doesn't matter anyway
    return res
