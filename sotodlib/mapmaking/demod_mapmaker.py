"""
This submodule contains classes to perform filter+bin mapmaking for 
demodulated data. The DemodMapmaker class creates a mapmaker object.
DemodSignalMap creates a signal you want to solve for, over which 
you accumulate observations into the div and rhs maps. For examples
how to use look at docstring of DemodMapmaker.
"""
__all__ = ['DemodMapmaker','DemodSignal','DemodSignalMap', 'DemodSignalMapHealpix']
import numpy as np
from pixell import enmap, utils, tilemap, bunch
import so3g.proj

from .. import coords
from .utilities import recentering_to_quat_lonlat, evaluate_recentering, MultiZipper, unarr, safe_invert_div
from .utilities import import_optional, get_flags, process_detweight_str
from .noise_model import NmatWhite
from types import SimpleNamespace

hp = import_optional('healpy')
h5py = import_optional('h5py')

class DemodMapmaker:
    def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False, comps='TQU', singlestream=False):
        """
        Initialize a FilterBin Mapmaker for demodulated data 
        
        Arguments
        ---------
        signals : list 
            List of Signal-objects representing the models that will be 
            solved jointly for. Currently this would be a DemodSignal
        noise_model : sotodlib.mapmaking.Nmat or None
            A noise model constructor which will be used to initialize the
            noise model for each observation. Can be overriden in add_obs.
            Noises other than NmatWhite not implemented. If None, a white 
            noise model is used.
        dtype : numpy.dtype
            The data type to use for the time-ordered data. Only tested
            with float32
        verbose : Bool
            Whether to print progress messages. Not implemented
        comps : str
            String with the components to solve for. Not implemented for 
            anything other than TQU
        singlestream : Bool
            If True, do not perform demodulated filter+bin mapmaking but 
            rather regular filter+bin mapmaking, i.e. map from obs.signal
            rather than from obs.dsT, obs.demodQ, obs.demodU
        
        Example usage :: 
            signal_map = mapmaking.DemodSignalMap(shape, wcs, comm)
            signals    = [signal_map]
            mapmaker   = mapmaking.DemodMapmaker(signals, 
                         noise_model=noise_model)
        
        """
        
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
        self.singlestream = singlestream

    def add_obs(self, id, obs, noise_model=None, split_labels=None, det_weights='ivar', qp_kwargs={}):
        """
        This function will accumulate an obs into the DemodMapmaker object, i.e. will add to 
        a RHS and div map.
        
        Arguments
        ---------
        id : str
            String that identifies an observation to accumulate. Typically will be something 
            like obs_id:wafer_slot:band
        obs : AxisManager
            An observation (axismanager) object to be accumulated.
        noise_model : sotodlib.mapmaking.Nmat or None, optional
            If a noise model is passed, the model in the Demodmapmaker object will be overriden,
            if None, the default model defined in the Demodmapmaker object is used.
        split_labels: list or None, optional
            A list of strings with the splits requested. If None then no splits were asked for,
            i.e. we will produce one map 

        """
        ctime  = obs.timestamps
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        if not(self.singlestream):
            # now we have 3 signals, dsT / demodQ / demodU. We pack them into an array with shape (3,...)
            tod    = np.array([obs.dsT.astype(self.dtype, copy=False), obs.demodQ.astype(self.dtype, copy=False), obs.demodU.astype(self.dtype, copy=False)])
        else:
            tod = obs.signal.astype(self.dtype, copy=False)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None: noise_model = self.noise_model
        # Build the noise model from the obs unless a fully
        # initialized noise model was passed
        if noise_model.ready:
            nmat = noise_model
        else:
            try:
                if not(self.singlestream):               
                    nmat = noise_model.build(tod[1], srate=srate) # Here we are building the model from demodQ
                else:
                    nmat = noise_model.build(tod, srate=srate)
            except Exception as e:
                msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
                raise RuntimeError(msg)
        # Add the observation to each of our signals
        for signal in self.signals:
            signal.add_obs(id, obs, nmat, tod, split_labels=split_labels, det_weights=det_weights, qp_kwargs=qp_kwargs)
        # Save what we need about this observation
        self.data.append(bunch.Bunch(id=id, ndet=obs.dets.count, nsamp=len(ctime), dets=obs.dets.vals, nmat=nmat))

class DemodSignal:
    def __init__(self, name, ofmt, output, ext):
        """
        This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc.
        
        Arguments
        ---------
        name : str
            The name of this signal, e.g. "sky", "cut", etc.
        ofmt : str
            The format used when constructing output file prefix
        output : Bool
            Whether this signal should be part of the output or not.
        ext : str
            The extension used for the files.

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
    def __init__(self, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False, Nsplits=1, singlestream=False):
        """
        Signal describing a non-distributed sky map. Signal describing a sky map in the coordinate 
        system given by "sys", which defaults to equatorial coordinates. If tiled==True, then this 
        will be a distributed map with the given tile_shape, otherwise it will be a plain enmap.
        
        Arguments
        ---------
        shape : numpy.ndarray
            Shape of the output map geometry
        wcs : wcs
            WCS of the output map geometry
        comm : MPI.comm
            MPI communicator
        comps : str, optional
            Components to map
        name : str, optional
            The name of this signal, e.g. "sky", "cut", etc.
        ofmt : str, optional
            The format used when constructing output file prefix
        output : Bool, optional
            Whether this signal should be part of the output or not.
        ext : str, optional
            The extension used for the files.
        dtype : numpy.dtype
            The data type to use for the time-ordered data.
        sys : str or None, optional
            The coordinate system to map. Defaults to equatorial
        recenter : str or None
            String to make object-centered maps, such as Moon/Sun/Planet centered maps.
            Look at sotodlib.mapmaking.parse_recentering for details.
        tile_shape : list, optional
            List with the shape of the tiles when using tiled maps from pixell
        tiled : Bool, optional
            If True, use tiled maps from pixell. If False, enmaps will be used.
        Nsplits : int, optional
            Number of splits that you will map simultaneously. By default is 1 when no
            splits are requested.
        singlestream : Bool, optional
            If True, do not perform demodulated filter+bin mapmaking but rather regular 
            filter+bin mapmaking, i.e. map from obs.signal rather than from obs.dsT, 
            obs.demodQ, obs.demodU
        
        """
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
        self.wrapper = lambda x : x
        ncomp      = len(comps)
        shape      = tuple(shape[-2:])
        if tiled:
            geo = tilemap.geometry(shape, wcs, tile_shape=tile_shape)
            self.rhs = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,)),      dtype=dtype)
            self.div = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,ncomp)), dtype=dtype)
            self.hits= tilemap.zeros(geo.copy(pre=(Nsplits,)),            dtype=dtype)
        else:
            self.rhs = enmap.zeros((Nsplits, ncomp)     +shape, wcs, dtype=dtype)
            self.div = enmap.zeros((Nsplits,ncomp,ncomp)+shape, wcs, dtype=dtype)
            self.hits= enmap.zeros((Nsplits,)+shape, wcs, dtype=dtype)

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None, det_weights='ivar', qp_kwargs={}):
        # Nd will have 3 components, corresponding to ds_T, demodQ, demodU with the noise model applied
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        ctime  = obs.timestamps
        for n_split in range(self.Nsplits):
            if pmap is None:
                # Build the local geometry and pointing matrix for this observation
                if self.recenter:
                    rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter, ctime=ctime[len(ctime)//2], geom=(self.rhs.shape, self.rhs.wcs), site=unarr(obs.site)))
                else: rot = None
                if split_labels is None:
                    flagnames = ['glitch_flags']
                    # this is the case with no splits
                else:
                    flagnames = ['glitch_flags', split_labels[n_split]]
                cuts = get_flags(obs, flagnames)
                threads='domdir'
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry, rot=rot, threads=threads, weather=unarr(obs.weather), site=unarr(obs.site), cuts=cuts, hwp=True, pix_scheme="rectpix", qp_kwargs=qp_kwargs)
            else:
                pmap_local = pmap

            det_weightsT, det_weightsQU = process_detweight_str(det_weights, nmat)
            if not(self.singlestream):
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap_local, obs.dsT, obs.demodQ, obs.demodU, det_weightsT, det_weightsQU, self.ncomp, self.wrapper)
            else:
                obs_rhs, obs_div, obs_hits = project_all_single(pmap_local, Nd, det_weightsT, self.ncomp, self.wrapper)
            # Update our full rhs and div. This works for both plain and distributed maps
            self.rhs[n_split] = self.rhs[n_split].insert(obs_rhs, op=np.ndarray.__iadd__)
            self.div[n_split] = self.div[n_split].insert(obs_div, op=np.ndarray.__iadd__)
            self.hits[n_split] = self.hits[n_split].insert(obs_hits[0],op=np.ndarray.__iadd__)
            # Save the per-obs things we need. Just the pointing matrix in our case.
            # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
            self.data[(id,n_split)] = bunch.Bunch(pmap=pmap_local, obs_geo=obs_rhs.geometry)
        del Nd

    def prepare(self):
        """Called when we're done adding everything. Sets up the map distribution,
        degrees of freedom and preconditioner."""
        if self.ready: return
        if self.tiled:
            self.geo_work = self.rhs.geometry
            self.rhs  = tilemap.redistribute(self.rhs, self.comm)
            self.div  = tilemap.redistribute(self.div, self.comm)
            self.hits = tilemap.redistribute(self.hits,self.comm)
        else:
            if self.comm is not None:
                self.rhs  = utils.allreduce(self.rhs, self.comm)
                self.div  = utils.allreduce(self.div, self.comm)
                self.hits = utils.allreduce(self.hits,self.comm)
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

class DemodSignalMapHealpix(DemodSignal):
    def __init__(self, nside, nside_tile=None, comm=None, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits.gz", dtype=np.float32, Nsplits=1, singlestream=False):
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.dtype = dtype
        self.tiled = (nside_tile is not None)
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
        ncomp      = len(comps)
        ordering = 'NEST' # Only NEST allowed for tiled maps
        self.hp_geom = SimpleNamespace(nside=nside, nside_tile=nside_tile, ordering=ordering)
        npix = 12 * nside**2
        self.rhs = np.zeros((Nsplits, ncomp, npix), dtype=dtype)
        self.div = np.zeros((Nsplits, ncomp, ncomp, npix), dtype=dtype)
        self.hits = np.zeros((Nsplits, npix), dtype=dtype)
        if self.tiled:
            self.wrapper = coords.healpix_utils.untile_healpix
        else:
            self.wrapper = lambda x:x

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None, det_weights='ivar', qp_kwargs={}):
        for n_split in range(self.Nsplits):
            if pmap is None:
                # Build the local geometry and pointing matrix for this observation
                # we handle cuts here through obs.flags
                if split_labels is None:
                    # this is the case with no splits
                    flagnames = ['glitch_flags'] # None
                else:
                    flagnames = ['glitch_flags', split_labels[n_split]]
                cuts = get_flags(obs, flagnames)
                threads = ["tiles", "simple"][self.hp_geom.nside_tile is None] # 'simple' is likely to perform very poorly but no other method implemented for untiled healpix
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.hp_geom, threads=threads, weather=unarr(obs.weather), site=unarr(obs.site), cuts=cuts, hwp=True, pix_scheme="healpix", qp_kwargs=qp_kwargs)
            else:
                pmap_local = pmap

            det_weightsT, det_weightsQU = process_detweight_str(det_weights, nmat)
            # Build the RHS for this observation
            if not(self.singlestream):
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap_local, obs.dsT, obs.demodQ, obs.demodU, det_weightsT, det_weightsQU, self.ncomp, self.wrapper)
            else:
                obs_rhs, obs_div, obs_hits = project_all_single(pmap_local, Nd, det_weightsT, self.ncomp, self.wrapper) # Should there be det_weights here?

            # Update our full rhs and div. This works for both plain and distributed maps
            self.rhs[n_split] = obs_rhs
            self.div[n_split] = obs_div
            self.hits[n_split] = obs_hits[0]
            # Save the per-obs things we need. Just the pointing matrix in our case.
            # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
            self.data[(id,n_split)] = bunch.Bunch(pmap=pmap_local)
            self.ready = True
        del Nd ## You can prob get rid of Nd entirely; keeping for compatibility

    def prepare(self):
        ## For now need this method for compatibility
        ## In the future may support handling of partial maps through a scheme similar to pixell tilemap
        return

    @property
    def ncomp(self): return len(self.comps)

    def write(self, prefix, tag, m, write_partial=False):
        if not self.output: return
        assert (self.comm is None or self.comm.rank == 0)
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if write_partial and self.ext not in ['fits', 'fits.gz']:
            raise NotImplementedError("write_partial only supported for fits")

        if self.ext in ["fits", "fits.gz"]:
            if hp is None:
                raise ValueError("Cannot save healpix map as fits; healpy could not be imported. Install healpy or save as .npy or .h5")
            if m.ndim > 2:
                m = np.reshape(m, (np.product(m.shape[:-1]), m.shape[-1]), order='C') # Flatten wrapping axes; healpy.write_map can't handle >2d array
            hp.write_map(oname, m.view(self.dtype), nest=(self.hp_geom.ordering=='NEST'), partial=write_partial, overwrite=True) ## TODO Replace hard-coded nest
        elif self.ext == "npy":
            np.save(oname, m.view(self.dtype))
        elif self.ext in ['h5', 'hdf5']:
            if h5py is None:
                raise ValueError("Cannot save healpix map as hdf5; h5py could not be imported. Install h5py or save as .npy or .fits")
            with h5py.File(oname, 'w') as f:
                dset = f.create_dataset("data", m.shape, dtype=self.dtype, data=m)
        else:
            raise ValueError(f"Unknown extension {self.ext}")

def project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper=lambda x:x):
    zeros = lambda *args, **kwargs : wrapper(pmap.zeros(*args, **kwargs))
    to_map = lambda *args, **kwargs : wrapper(pmap.to_map(*args, **kwargs))

    rhs = zeros()
    rhs_T = to_map(signal=signalT, comps='T', det_weights=det_weightsT)
    rhs_demodQ = to_map(signal=signalQ, comps='QU', det_weights=det_weightsQU)
    rhs_demodU = to_map(signal=signalU, comps='QU', det_weights=det_weightsQU)
    rhs_demodQU = zeros(super_shape=(2), comps='QU',)

    rhs_demodQU[0][:] = rhs_demodQ[0] - rhs_demodU[1]
    rhs_demodQU[1][:] = rhs_demodQ[1] + rhs_demodU[0]
    del rhs_demodQ, rhs_demodU

    # we write into the rhs.
    rhs[0] = rhs_T[0]
    rhs[1] = rhs_demodQU[0]
    rhs[2] = rhs_demodQU[1]
    return rhs

def project_div_demod(pmap, det_weightsT, det_weightsQU, ncomp, wrapper=lambda x:x):
    zeros = lambda *args, **kwargs : wrapper(pmap.zeros(*args, **kwargs))
    to_weights = lambda *args, **kwargs : wrapper(pmap.to_weights(*args, **kwargs))

    div = zeros(super_shape=(ncomp, ncomp))
    # Build the per-pixel inverse covmat for this observation
    wT = to_weights(comps='T', det_weights=det_weightsT)
    wQU = to_weights(comps='T', det_weights=det_weightsQU)
    div[0,0] = wT
    div[1,1] = wQU
    div[2,2] = wQU
    return div

def project_all_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, ncomp, wrapper=lambda x:x):
    rhs =  project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper)
    div = project_div_demod(pmap, det_weightsT, det_weightsQU, ncomp, wrapper)
    hits = wrapper(pmap.to_map(signal=np.ones_like(signalT))) ## Note hits is *not* weighted by det_weights
    return rhs, div, hits

def project_all_single(pmap, Nd, det_weights, ncomp, wrapper=lambda x:x):
    rhs = wrapper(pmap.to_map(signal=Nd, comps='TQU', det_weights=det_weights))
    div = pmap.zeros(super_shape=(ncomp, ncomp))
    pmap.to_weights(dest=div, comps='TQU', det_weights=det_weights)
    div = wrapper(div)
    hits = wrapper(pmap.to_map(np.ones_like(Nd)))
    return rhs, div, hits
