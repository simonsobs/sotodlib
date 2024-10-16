"""
This submodule contains classes to perform filter+bin mapmaking for 
demodulated data. The DemodMapmaker class creates a mapmaker object.
DemodSignalMap creates a signal you want to solve for, over which 
you accumulate observations into the div and rhs maps. For examples
how to use look at docstring of DemodMapmaker.
"""
__all__ = ['DemodMapmaker','DemodSignal','DemodSignalMap']
import numpy as np
from pixell import enmap, utils, tilemap, bunch
import so3g.proj

from .. import coords
from .utilities import recentering_to_quat_lonlat, evaluate_recentering, MultiZipper, unarr, safe_invert_div
from .utilities import import_optional, get_flags
from .noise_model import NmatWhite

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

    def add_obs(self, id, obs, noise_model=None, split_labels=None):
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
            signal.add_obs(id, obs, nmat, tod, split_labels=split_labels)
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
    def __init__(self, shape=None, wcs=None, comm=None, comps="TQU", name="sky", ofmt="{name}", output=True,
                 ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False, Nsplits=1, singlestream=False,
                 nside=None, nside_tile=None):
        """
        Parent constructor; use for_rectpix or for_healpix instead. Args described there.
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

        self.pix_scheme = "rectpix" if (shape is not None) else "healpix"
        if self.pix_scheme == "healpix":
            self.tiled = (nside_tile is not None)
            self.hp_geom = coords.healpix_utils.get_geometry(nside, nside_tile, ordering='NEST')
            npix = 12 * nside**2
            self.rhs = np.zeros((Nsplits, ncomp, npix), dtype=dtype)
            self.div = np.zeros((Nsplits, ncomp, ncomp, npix), dtype=dtype)
            self.hits = np.zeros((Nsplits, npix), dtype=dtype)

            if self.tiled:
                self.wrapper = coords.healpix_utils.tiled_to_full
        else:
            shape = tuple(shape[-2:])
            if tiled:
                geo = tilemap.geometry(shape, wcs, tile_shape=tile_shape)
                self.rhs = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,)),      dtype=dtype)
                self.div = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,ncomp)), dtype=dtype)
                self.hits= tilemap.zeros(geo.copy(pre=(Nsplits,)),            dtype=dtype)
            else:
                self.rhs = enmap.zeros((Nsplits, ncomp)     +shape, wcs, dtype=dtype)
                self.div = enmap.zeros((Nsplits,ncomp,ncomp)+shape, wcs, dtype=dtype)
                self.hits= enmap.zeros((Nsplits,)+shape, wcs, dtype=dtype)

    @classmethod
    def for_rectpix(cls, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
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
        return cls(shape=shape, wcs=wcs, comm=comm, comps=comps, name=name, ofmt=ofmt, output=output,
                   ext=ext, dtype=dtype, sys=sys, recenter=recenter, tile_shape=tile_shape, tiled=tiled,
                   Nsplits=Nsplits, singlestream=singlestream, nside=None, nside_tile=None)

    @classmethod
    def for_healpix(cls, nside, nside_tile=None, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits.gz", dtype=np.float32, Nsplits=1, singlestream=False):
        """
        Signal describing a sky map in healpix pixelization, NEST ordering.

        Arguments
        ---------
        nside : int
            Nside of the output map. Should be a power of 2
        nside_tile: int, str or None, optional
            Nside of the tiling scheme. Should be a power of 2 and smaller than nside.
            May also be 'auto' to set automatically, or None to use no tiling.
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
            May be 'fits', 'fits.gz', 'h5', 'h5py', or 'npy'.
        dtype : numpy.dtype
            The data type to use for the time-ordered data.
        Nsplits : int, optional
            Number of splits that you will map simultaneously. By default is 1 when no
            splits are requested.
        singlestream : Bool, optional
            If True, do not perform demodulated filter+bin mapmaking but rather regular
            filter+bin mapmaking, i.e. map from obs.signal rather than from obs.dsT,
            obs.demodQ, obs.demodU
        """
        return cls(shape=None, wcs=None, comm=None, comps=comps, name=name, ofmt=ofmt, output=output,
                   ext=ext, dtype=dtype, sys=None, recenter=None, tile_shape=None, tiled=False,
                   Nsplits=Nsplits, singlestream=singlestream, nside=nside, nside_tile=nside_tile)

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None):
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
                    # this is the case with no splits
                    flagnames = ['glitch_flags']
                else:
                    flagnames = ['glitch_flags', split_labels[n_split]]
                cuts = get_flags(obs, flagnames)
                if self.pix_scheme == "rectpix":
                    threads='domdir'
                    geom = self.rhs.geometry
                else:
                    threads = ["tiles", "simple"][self.hp_geom.nside_tile is None]
                    geom = self.hp_geom
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=geom, rot=rot, threads=threads, weather=unarr(obs.weather), site=unarr(obs.site), cuts=cuts, hwp=True)
            else:
                pmap_local = pmap

            if not(self.singlestream):
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap=pmap_local, signalT=obs.dsT, signalQ=obs.demodQ, signalU=obs.demodU,
                                                               det_weightsT=2*nmat.ivar, det_weightsQU=nmat.ivar, ncomp=self.ncomp, wrapper=self.wrapper)
            else:
                obs_rhs, obs_div, obs_hits = project_all_single(pmap=pmap_local, Nd=Nd, det_weights=nmat.ivar, comps='TQU', wrapper=self.wrapper)
            # Update our full rhs and div. This works for both plain and distributed maps
            if self.pix_scheme == "rectpix":
                self.rhs[n_split] = self.rhs[n_split].insert(obs_rhs, op=np.ndarray.__iadd__)
                self.div[n_split] = self.div[n_split].insert(obs_div, op=np.ndarray.__iadd__)
                self.hits[n_split] = self.hits[n_split].insert(obs_hits[0],op=np.ndarray.__iadd__)
                obs_geo = obs_rhs.geometry
            else:
                self.rhs[n_split] = obs_rhs
                self.div[n_split] = obs_div
                self.hits[n_split] = obs_hits[0]
                obs_geo = self.hp_geom

            # Save the per-obs things we need. Just the pointing matrix in our case.
            # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
            self.data[(id,n_split)] = bunch.Bunch(pmap=pmap_local, obs_geo=obs_geo)
        del Nd

    def prepare(self):
        """Called when we're done adding everything. Sets up the map distribution,
        degrees of freedom and preconditioner."""
        if self.pix_scheme == "healpix": return
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
        if self.pix_scheme == "rectpix":
            if self.tiled:
                tilemap.write_map(oname, m, self.comm)
            else:
                if self.comm is None or self.comm.rank == 0:
                    enmap.write_map(oname, m)
        else:
            if self.ext in ["fits", "fits.gz"]:
                if hp is None:
                    raise ImportError("Cannot save healpix map as fits; healpy could not be imported. Install healpy or save as npy or h5")
                if m.ndim > 2:
                    m = np.reshape(m, (np.prod(m.shape[:-1]), m.shape[-1]), order='C') # Flatten wrapping axes; healpy.write_map can't handle >2d array
                hp.write_map(oname, m.view(self.dtype), nest=(self.hp_geom.ordering=='NEST'), overwrite=True)
            elif self.ext == "npy":
                np.save(oname, {'nside': self.hp_geom.nside, 'ordering':self.hp_geom.ordering, 'data':m.view(self.dtype)}, allow_pickle=True)
            elif self.ext in ['h5', 'hdf5']:
                if h5py is None:
                    raise ValueError("Cannot save healpix map as hdf5; h5py could not be imported. Install h5py or save as npy or fits")
                with h5py.File(oname, 'w') as f:
                    dset = f.create_dataset("data", m.shape, dtype=self.dtype, data=m)
                    dset.attrs['ordering'] = self.hp_geom.ordering
                    dset.attrs['nside'] = self.hp_geom.nside
            else:
                raise ValueError(f"Unknown extension {self.ext}")

        return oname

def project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper=lambda x:x):
    """
    Project demodulated T, Q, U timestreams into weighted maps.

    Arguments
    ---------
    pmap : sotodlib.coords.pmat.P
        Projection matrix.
    signalT, signalQ, signalU : np.ndarray (ndets, nsamps)
        T, Q, U timestreams after demodulation.
    det_weightsT, det_weightsQU : np.ndarray (ndets,) or None
        Array of detector weights for T and QU. If None, unit weights are used.
    wrapper : Function with single argument
        Wrapper function for output of pmap operations.
    """
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
    """
    Make weight maps for demodulated data.

    Arguments
    ---------
    pmap : sotodlib.coords.pmat.P
        Projection matrix.
    det_weightsT, det_weightsQU : np.ndarray (ndets,) or None
        Array of detector weights for T and QU. If None, unit weights are used.
    ncomp : int
        Number of map components. e.g. 3 for TQU.
    wrapper : Function with single argument
        Wrapper function for output of pmap operations.
    """
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
    """
    Get weighted signal, weight, and hits maps for demodulated data.
    See project_rhs_demod for description of args.
    """
    rhs =  project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper)
    div = project_div_demod(pmap, det_weightsT, det_weightsQU, ncomp, wrapper)
    hits = wrapper(pmap.to_map(signal=np.ones_like(signalT))) ## Note hits is *not* weighted by det_weights
    return rhs, div, hits

def project_all_single(pmap, Nd, det_weights, comps, wrapper=lambda x:x):
    """
    Get weighted signal, weight and hits maps from a single (non-demodulated) timestream.

    Arguments
    ---------
    pmap : sotodlib.coords.pmat.P
        Projection matrix.
    Nd : np.ndarray (ndets, nsamps)
        Input timestream to map.
    det_weights : np.ndarray (ndets,) or None
        Array of detector weights. If None, unit weights are used.
    comps : str
        Components to use in mapmaking. 'TQU', 'T', or 'QU'.
    wrapper : Function with single argument
        Wrapper function for output of pmap operations.
    """
    ncomp = len(comps)
    rhs = wrapper(pmap.to_map(signal=Nd, comps=comps, det_weights=det_weights))
    div = pmap.zeros(super_shape=(ncomp, ncomp))
    pmap.to_weights(dest=div, comps=comps, det_weights=det_weights)
    div = wrapper(div)
    hits = wrapper(pmap.to_map(signal=np.ones_like(Nd)))
    return rhs, div, hits
