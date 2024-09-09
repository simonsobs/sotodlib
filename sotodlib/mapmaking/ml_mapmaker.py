import numpy as np
from pixell import bunch, enmap, tilemap, utils

from .. import coords
from .pointing_matrix import *
from .utilities import *


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

    def add_obs(self, id, obs, deslope=True, noise_model=None, signal_estimate=None):
        # Prepare our tod
        ctime  = obs.timestamps
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        tod    = obs.signal.astype(self.dtype, copy=False)
        # Subtract an existing estimate of the signal before estimating
        # the noise model, if available
        if signal_estimate is not None: tod -= signal_estimate
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
        # Add back signal estimate
        if signal_estimate is not None:
            tod += signal_estimate
            # The signal estimate might not be desloped, so
            # adding it back can reintroduce a slope. Fix that here.
            if deslope:
                utils.deslope(tod, w=5, inplace=True)
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

    def solve(self, maxiter=500, maxerr=1e-6, x0=None):
        self.prepare()
        rhs    = self.dof.zip(*[signal.rhs for signal in self.signals])
        if x0 is not None: x0 = self.dof.zip(*x0)
        solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot, x0=x0)
        while solver.i < maxiter and solver.err > maxerr:
            solver.step()
            yield bunch.Bunch(i=solver.i, err=solver.err, x=self.dof.unzip(solver.x))

    def translate(self, other, x):
        """Translate degrees of freedom x from some other mapamaker to the current one.
        The other mapmaker must have the same list of signals, except that they can have
        different sample rate etc. than this one. See the individual Signal-classes
        translate methods for details. This is used in multipass mapmaking."""
        xnew = []
        for ssig, osig, oval in zip(self.signals, other.signals, x):
            xnew.append(ssig.translate(osig, oval))
        return xnew

    def transeval(self, id, obs, other, x, tod=None):
        """Evaluate degrees of freedom x for the given tod after translating
        it from those used by another, similar mapmaker. This will have the same
        signals, but possibly with different sample rates etc."""
        if tod is None: tod = np.zeros([obs.dets.count, obs.samps.count], self.dtype)
        for si, (ssig, osig, oval), in reversed(list(enumerate(zip(self.signals,other.signals,x)))):
            ssig.transeval(id, obs, osig, oval, tod=tod)
        return tod


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
    def translate(self, other, x): return x
    def transeval(self, id, obs, other, x, tod): pass

class SignalMap(Signal):
    """Signal describing a non-distributed sky map."""
    def __init__(self, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False,
            interpol=None):
        """Signal describing a sky map in the coordinate system given by "sys", which defaults
        to equatorial coordinates. If tiled==True, then this will be a distributed map with
        the given tile_shape, otherwise it will be a plain enmap. interpol controls the
        pointing matrix interpolation mode. See so3g's Projectionist docstring for details."""
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

    def add_obs(self, id, obs, nmat, Nd, pmap=None, glitch_flags:str ="flags.glitch_flags"):
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        Nd     = Nd.copy() # This copy can be avoided if build_obs is split into two parts
        ctime  = obs.timestamps
        pcut   = PmatCut(get_flags_from_path(obs, glitch_flags)) # could pass this in, but fast to construct
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
                self.hits = utils.allreduce(self.hits, self.comm)
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

    def _checkcompat(self, other):
        # Check if other is compatible with us. For SignalMap, we currently
        # only support direct equivalence
        if other.sys != self.sys or other.recenter != self.recenter:
            raise ValueError("Coordinate system mismatch")
        if other.tiled != self.tiled:
            raise ValueError("Tiling mismatch")
        if self.tiled:
            if other.rhs.geometry.shape != self.rhs.geometry.shape:
                raise ValueError("Geometry mismatch")
            # Tiling is not set up yet by the time transeval is called.
            # Transeval doesn't need the tiling to match, though
            #if other.rhs.ntile != self.rhs.ntile or other.rhs.nactive != self.rhs.nactive:
            #    raise ValueError("Tiling mismatch")
        else:
            if other.rhs.shape != self.rhs.shape:
                raise ValueError("Geometry mismatch")

    def translate(self, other, map):
        """Translate map from another SignalMap representation to the current,
        returning a new map. The new map may be a reference to the original."""
        # Currently we don't support any actual translation, but could handle
        # resolution changes in the future (probably not useful though)
        self._checkcompat(other)
        return map

    def transeval(self, id, obs, other, map, tod):
        """Translate map from SignalMap other to the current SignalMap,
        and then evaluate it for the given observation, returning a tod.
        This is used when building a signal-free tod for the noise model
        in multipass mapmaking."""
        # Currently we don't support any actual translation, but could handle
        # resolution changes in the future (probably not useful though)
        self._checkcompat(other)
        # Build the local geometry and pointing matrix for this observation
        if self.recenter:
            rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter,
                ctime=ctime[len(ctime)//2], geom=(self.rhs.shape, self.rhs.wcs), site=unarr(obs.site)))
        else: rot = None
        pmap = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry,
            rot=rot, threads="domdir", weather=unarr(obs.weather), site=unarr(obs.site),
            interpol=self.interpol)
        # Build the RHS for this observation
        pmap.from_map(dest=tod, signal_map=map, comps=self.comps)
        return tod

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

    def add_obs(self, id, obs, nmat, Nd, glitch_flags="flags.glitch_flags"):
        """Add and process an observation. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data."""
        Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
        pcut    = PmatCut(obs.flags.glitch_flags, model=self.cut_type)
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

    def _checkcompat(self, other):
        if other.cut_type != self.cut_type:
            raise ValueError("Cut type mismatch")

    def translate(self, other, junk):
        """Translate junk degrees of freedom from one SignalCut representation
        to another, e.g. changing sample rate. Returns the result"""
        self._checkcompat(other)
        res = np.full(self.off, -1e10, self.dtype)
        for id in self.data:
            sdata = self .data[id]
            odata = other.data[id]
            so3g.translate_cuts(odata.pcut.cuts, sdata.pcut.cuts, sdata.pcut.model, sdata.pcut.params, junk[odata.i1:odata.i2], res[sdata.i1:sdata.i2])
        return res

    def transeval(self, id, obs, other, junk, tod, glitch_flags: str="flags.glitch_flags"):
        """Translate data junk from SignalCut other to the current SignalCut,
        and then evaluate it for the given observation, returning a tod.
        This is used when building a signal-free tod for the noise model
        in multipass mapmaking."""
        self._checkcompat(other)
        # We have to make a pointing matrix from scratch because add_obs
        # won't have been called yet at this point
        spcut = PmatCut(get_flags_from_path(obs, glitch_flags), model=self.cut_type)
        # We do have one for other though, since that will be the output
        # from the previous round of multiplass mapmaking.
        odata = other.data[id]
        sjunk = np.zeros(spcut.njunk, junk.dtype)
        # Translate the cut degrees of freedom. The sample rate could have
        # changed, for example.
        so3g.translate_cuts(odata.pcut.cuts, spcut.cuts, spcut.model, spcut.params, junk[odata.i1:odata.i2], sjunk)
        # And project onto the tod
        spcut.forward(tod, sjunk)
        return tod
