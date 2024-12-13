"""
This submodule contains classes to perform filter+bin mapmaking for 
demodulated data. The DemodMapmaker class creates a mapmaker object.
DemodSignalMap creates a signal you want to solve for, over which 
you accumulate observations into the div and rhs maps. For examples
how to use look at docstring of DemodMapmaker.
"""
__all__ = ['DemodMapmaker','DemodSignal','DemodSignalMap','make_demod_map']
import numpy as np, os
from pixell import enmap, utils as putils, tilemap, bunch, mpi
import so3g.proj

from .. import core
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
            The data type to use for the maps.
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
                if self.Nsplits == 1:
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
                self.rhs  = putils.allreduce(self.rhs, self.comm)
                self.div  = putils.allreduce(self.div, self.comm)
                self.hits = putils.allreduce(self.hits,self.comm)
        self.ready = True

    @property
    def ncomp(self): return len(self.comps)

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

def setup_demod_map(noise_model, shape=None, wcs=None, nside=None,
                    comm=mpi.COMM_WORLD, comps='TQU', split_labels=None,
                    singlestream=False, dtype_tod=np.float32,
                    dtype_map=np.float32, recenter=None, verbose=0):
    """
    Setup the classes for demod mapmaking and return
    a DemodMapmmaker object
    """
    if shape is not None and wcs is not None:
        Nsplits = len(split_labels)
        signal_map = DemodSignalMap.for_rectpix(shape, wcs, comm, comps=comps,
                                              dtype=dtype_map, tiled=False,
                                              ofmt="", Nsplits=Nsplits,
                                              singlestream=singlestream,
                                              recenter=recenter)
    elif nside is not None:
        Nsplits = len(split_labels)
        signal_map = DemodSignalMap.for_healpix(nside, nside_tile='auto', 
                                    comps=comps, dtype=dtype_map,
                                    ofmt="", Nsplits=Nsplits,
                                    singlestream=singlestream,
                                    ext="fits.gz")
    signals    = [signal_map]
    mapmaker   = DemodMapmaker(signals, noise_model=noise_model,
                                         dtype=dtype_tod,
                                         verbose=verbose>0,
                                         singlestream=singlestream)
    return mapmaker

def write_demod_maps(prefix, data, split_labels=None):
    """
    Write maps from data into files
    """
    Nsplits = len(split_labels)
    for n_split in range(Nsplits):
        data.signal.write(prefix, "%s_wmap"%split_labels[n_split],
                          data.wmap[n_split])
        data.signal.write(prefix, "%s_weights"%split_labels[n_split],
                          data.weights[n_split])
        data.signal.write(prefix, "%s_hits"%split_labels[n_split],
                          data.signal.hits[n_split])

def write_demod_info(oname, info, split_labels=None):
    putils.mkdir(os.path.dirname(oname))
    Nsplits = len(split_labels)
    for n_split in range(Nsplits):
        bunch.write(oname+'_%s_info.hdf'%split_labels[n_split], info[n_split])

def make_demod_map(context, obslist, noise_model, info,
                    preprocess_config, prefix, shape=None, wcs=None,
                    nside=None, comm=mpi.COMM_WORLD, comps="TQU", t0=0,
                    dtype_tod=np.float32, dtype_map=np.float32,
                    tag="", verbose=0, split_labels=None, L=None,
                    site='so_sat3', recenter=None, singlestream=False):
    """
    Make a demodulated map from the list of observations in obslist.

    Arguments
    ---------
    context : str
        File path to context used to load obs from.
    obslist : dict
        The obslist which is the output of the
        mapmaking.obs_grouping.build_obslists, contains the information of the
        single or multiple obs to map.
    noise_model : sotodlib.mapmaking.Nmat
        Noise model to pass to DemodMapmaker.
    info : list
        Information for the database, will be written as a .hdf file.
    preprocess_config : dict
        Dictionary with the config yaml file for the preprocess database.
    prefix : str
        Prefix for the output files
    shape : tuple, optional
        Shape of the geometry to use for mapping.
    wcs : dict, optional
        WCS kernel of the geometry to use for mapping.
    nside : int, optional
        Nside for healpix pixelization
    comps : str, optional
        Which components to map, only TQU supported for now.
    t0 : int, optional
        Ctime to use as the label in the map files.
    dtype_tod : numpy.dtype, optional
        The data type to use for the time-ordered data. Only tested
        with float32.
    dtype_map : numpy.dtype, optional
        The data type to use for the maps.
    tag : str, optional
        Prefix tag for the logger.
    verbose : bool, optional
    split_labels : list or None, optional
        A list of strings with the splits requested. If None then no splits
        were asked for, i.e. we will produce one map.
    L : logger, optional
        Logger for printing on the screen.
    site : str, optional
        Platform name for the pointing matrix.
    recenter : str or None
        String to make object-centered maps, such as Moon/Sun/Planet centered maps.
        Look at sotodlib.mapmaking.parse_recentering for details.
    singlestream : Bool
        If True, do not perform demodulated filter+bin mapmaking but
        rather regular filter+bin mapmaking, i.e. map from obs.signal
        rather than from obs.dsT, obs.demodQ, obs.demodU.

    Returns
    -------
    errors : list
        List of errors from preprocess database. To be used in cleanup_mandb.
    outputs : list
        List of outputs from preprocess database. To be used in cleanup_mandb.
    """
    from ..preprocess import preprocess_util
    context = core.Context(context)
    if L is None:
        L = preprocess_util.init_logger("Demod filterbin mapmaking")
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    mapmaker = setup_demod_map(noise_model, shape=shape, wcs=wcs, nside=nside,
                    comm=comm, comps=comps, split_labels=split_labels,
                    singlestream=singlestream, dtype_tod=dtype_tod,
                    dtype_map=dtype_map, recenter=recenter, verbose=verbose)

    if comm.rank == 0: L.info(pre + "Building RHS")
    # And feed it with our observations
    nobs_kept  = 0
    errors = [] ; outputs = []; # PENDING: do an allreduce of these.
                                # not needed for atomic maps, but needed for
                                # depth-1 maps
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        error, output, obs = preprocess_util.preproc_or_load_group(obs_id,
                            configs=preprocess_config,
                            dets={'wafer_slot':detset, 'wafer.bandpass':band}, 
                            logger=L, context=context, overwrite=False)
        errors.append(error) ; outputs.append(output) ;
        if error not in [None,'load_success']:
            L.info('tod %s:%s:%s failed in the prepoc database'%(obs_id,detset,band))
            continue
        obs.wrap("weather", np.full(1, "toco"))
        obs.wrap("site",    np.full(1, site))
        mapmaker.add_obs(name, obs, split_labels=split_labels)
        L.info('Done with tod %s:%s:%s'%(obs_id,detset,band))
        nobs_kept += 1
    nobs_kept = comm.allreduce(nobs_kept)
    # if we skip all the obs then we return error and output
    if nobs_kept == 0:
        return errors, outputs

    for signal in mapmaker.signals:
        signal.prepare()
    if comm.rank == 0: L.info(pre + "Writing F+B outputs")
    wmap = []
    weights = []
    # mapmaker.signals[0] is signal_map
    for n_split in range(mapmaker.signals[0].Nsplits):
        wmap.append( mapmaker.signals[0].rhs[n_split] )
        div = np.diagonal(mapmaker.signals[0].div[n_split], axis1=0, axis2=1)
        div = np.moveaxis(div, -1, 0) # this moves the last axis to the 0th position
        weights.append(div)
    mapdata = bunch.Bunch(wmap=wmap, weights=weights, signal=mapmaker.signals[0], t0=t0)

    info = add_weights_to_info(info, weights, split_labels)

    # output to files
    write_demod_maps(prefix, mapdata, split_labels=split_labels, )
    write_demod_info(prefix, info, split_labels=split_labels )
    return errors, outputs

def add_weights_to_info(info, weights, split_labels):
    Nsplits = len(split_labels)
    for isplit in range(Nsplits):
        sub_info = info[isplit]
        sub_weights = weights[isplit]
        # Assuming weights are TT, QQ, UU
        if sub_weights.shape[0] != 3:
            raise ValueError(f"sub_weights has unexpected shape {sub_weights.shape}. First axis should be (3,) for TT, QQ, UU")
        mean_qu = np.mean(sub_weights[1:], axis=0)
        positive = np.where(mean_qu > 0)
        sumweights = np.sum(mean_qu[positive])
        meanweights = np.mean(mean_qu[positive])
        medianweights = np.median(mean_qu[positive])
        sub_info['total_weight_qu'] = sumweights
        sub_info['mean_weight_qu'] = meanweights
        sub_info['median_weight_qu'] = medianweights
        info[isplit] = sub_info
    return info

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
