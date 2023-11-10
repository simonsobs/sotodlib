import numpy as np
from pixell import enmap, utils, tilemap, bunch

from .. import coords
from .utilities import *
from .pointing_matrix import *

class DemodMapmaker:
    def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False, comps='TQU'):
        """Initialize a FilterBin Mapmaker for demodulated data 
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
            noise_model = NmatWhite()
        self.signals      = signals
        self.dtype        = dtype
        self.verbose      = verbose
        self.noise_model  = noise_model
        self.data         = []
        self.dof          = MultiZipper()
        self.ready        = False
        self.ncomp        = len(comps)

    def add_obs(self, id, obs, noise_model=None, deslope=False):
        # Prepare our tod
        ctime  = obs.timestamps
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        # now we have 3 signals, dsT / demodQ / demodU. We pack them into an array with shape (3,...)
        tod    = np.array([obs.dsT.astype(self.dtype, copy=False), obs.demodQ.astype(self.dtype, copy=False), obs.demodU.astype(self.dtype, copy=False)])
        if deslope:
            for i in range(self.ncomp):
                utils.deslope(tod[i], w=5, inplace=True)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None: noise_model = self.noise_model
        # Build the noise model from the obs unless a fully
        # initialized noise model was passed
        if noise_model.ready:
            nmat = noise_model
        else:
            try:
                # we build the noise model from demodQ. For now we will apply it to Q and U also, but this will change
                nmat = noise_model.build(tod[1], srate=srate) # I have to define how the noise model will be build
            except Exception as e:
                msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
                raise RuntimeError(msg)
        # And apply it to the tod
        for i in range(self.ncomp):
            tod[i]    = nmat.apply(tod[i])
        # Add the observation to each of our signals
        for signal in self.signals:
            signal.add_obs(id, obs, nmat, tod)
        # Save what we need about this observation
        self.data.append(bunch.Bunch(id=id, ndet=obs.dets.count, nsamp=len(ctime), dets=obs.dets.vals, nmat=nmat))

class DemodSignal:
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
    def to_work  (self, x): return x.copy()
    def from_work(self, x): return x
    def write   (self, prefix, tag, x): pass

class DemodSignalMap(DemodSignal):
    """Signal describing a non-distributed sky map."""
    def __init__(self, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False):
        """Signal describing a sky map in the coordinate system given by "sys", which defaults
        to equatorial coordinates. If tiled==True, then this will be a distributed map with
        the given tile_shape, otherwise it will be a plain enmap."""
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
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

    def add_obs(self, id, obs, nmat, Nd, pmap=None, wrong_definition=False):
        # Nd will have 3 components, corresponding to ds_T, demodQ, demodU with the noise model applied
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        for i in range(self.ncomp):
            Nd[i]     = Nd[i].copy() # This copy can be avoided if build_obs is split into two parts
        ctime  = obs.timestamps
        if pmap is None:
            # Build the local geometry and pointing matrix for this observation
            if self.recenter:
                rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter,
                    ctime=ctime[len(ctime)//2], geom=(self.rhs.shape, self.rhs.wcs), site=unarr(obs.site)))
            else: rot = None
            # we handle cuts here through obs.glitch_flags
            pmap = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry, rot=rot, threads="domdir", weather=unarr(obs.weather), site=unarr(obs.site), cuts=obs.glitch_flags)
        # Build the RHS for this observation
        obs_rhs = pmap.zeros() # this is the final RHS, we will fill it at the end
        
        obs_rhs_T = pmap.zeros(super_shape=(1),comps='T')
        pmap.to_map(dest=obs_rhs_T, signal=Nd[0], comps='T') # this is only the RHS for T
        
        # RHS for QU. We save it to dummy maps
        obs_rhs_demodQ = pmap.to_map(signal=Nd[1], comps='QU') # Do I need to pass tod=obs ?
        obs_rhs_demodU = pmap.to_map(signal=Nd[2], comps='QU') # Do I need to pass tod=obs ?
        obs_rhs_demodQU = pmap.zeros(super_shape=(2),comps='QU',)
        if wrong_definition == True:
            # CAUTION: Here the definition of mQU_weighted uses a wrong way of definition, as toast simulation defines that in the wrong way.
            obs_rhs_demodQU[0][:] = obs_rhs_demodQ[0] - obs_rhs_demodU[1]
            # (= Q_{flipped detector coord}*cos(2 theta_pa) - U_{flipped detector coord}*sin(2 theta_pa) )
            obs_rhs_demodQU[1][:] = obs_rhs_demodQ[1] + obs_rhs_demodU[0]
            # (= Q_{flipped detector coord}*sin(2 theta_pa) + U_{flipped detector coord}*cos(2 theta_pa) )
        else:
            #### In field, you should use instead ####
            obs_rhs_demodQU[0][:] = obs_rhs_demodQ[0] + obs_rhs_demodU[1]
            # (= Q_{flipped detector coord}*cos(2 theta_pa) + U_{flipped detector coord}*sin(2 theta_pa) )
            obs_rhs_demodQU[1][:] = -obs_rhs_demodQ[1] + obs_rhs_demodU[0] 
            # (= -Q_{flipped detector coord}*sin(2 theta_pa) + U_{flipped detector coord}*cos(2 theta_pa) )
        # we write into the obs_rhs. 
        obs_rhs[0] = obs_rhs_T[0]
        obs_rhs[1] = obs_rhs_demodQU[0]
        obs_rhs[2] = obs_rhs_demodQU[1]
        obs_div    = pmap.zeros(super_shape=(self.ncomp,self.ncomp))
        # Build the per-pixel inverse covmat for this observation
        #obs_div = enmap.zeros((3, 3) + pmap.geom.shape, wcs=pmap.geom.wcs)
        #det_weights = 1/np.std(obs.demodQ, axis=1)**2
        wT = pmap.to_weights(obs, signal=obs.dsT, comps='T', det_weights=nmat.ivar,)
        wQU = pmap.to_weights(obs, signal=obs.demodQ, comps='T', det_weights=nmat.ivar,)
        obs_div[0,0] = wT
        obs_div[1,1] = wQU
        obs_div[2,2] = wQU
        # Build hitcount
        Nd[0,:] = 1
        obs_hits = pmap.to_map(signal=Nd[0],)
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

    def to_work(self, map):
        if self.tiled: return tilemap.redistribute(map, self.comm, self.geo_work.active)
        else: return map.copy()

    def from_work(self, map):
        if self.tiled: return tilemap.redistribute(map, self.comm, self.rhs.geometry.active)
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