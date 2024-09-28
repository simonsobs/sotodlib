"""
This submodule contains classes to perform filter+bin mapmaking for 
demodulated data. The DemodMapmaker class creates a mapmaker object.
DemodSignalMap creates a signal you want to solve for, over which 
you accumulate observations into the div and rhs maps. For examples
how to use look at docstring of DemodMapmaker.
"""
__all__ = ['DemodMapmaker','DemodSignal','DemodSignalMap','make_atomic_map']
import numpy as np
from pixell import enmap, utils, tilemap, bunch
import so3g.proj

from .. import coords
from .. import site_pipeline
from .utilities import recentering_to_quat_lonlat, evaluate_recentering, MultiZipper, unarr, safe_invert_div
from .noise_model import NmatWhite

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
            signal_map = mapmaking.DemodSignalMap(shape, wcs)
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
    def __init__(self, shape, wcs, comps="TQU", name="sky", ofmt="{name}", output=True,
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
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
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
                    pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry, rot=rot, threads="domdir", weather=unarr(obs.weather), site=unarr(obs.site), cuts=obs.flags.glitch_flags, hwp=True)
                else:
                    # this is the case where we are processing a split. We need to figure out what type of split it is (detector, samples), build the RangesMatrix mask and create the pmap.
                    if split_labels[n_split] in ['det_left','det_right','det_in','det_out','det_upper','det_lower']:
                        # then we are in a detector fixed in time split.
                        cuts = obs.flags.glitch_flags + obs.det_flags[split_labels[n_split]]
                    elif split_labels[n_split] == 'scan_left':
                        cuts = obs.flags.glitch_flags + obs.flags.left_scan
                    elif split_labels[n_split] == 'scan_right':
                        cuts = obs.flags.glitch_flags + obs.flags.right_scan
                    pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry, rot=rot, threads="domdir", weather=unarr(obs.weather), site=unarr(obs.site), cuts=cuts, hwp=True)
            else:
                pmap_local = pmap
                    
            # Build the RHS for this observation
            obs_rhs = pmap_local.zeros() # this is the final RHS, we will fill it at the end
            
            if not(self.singlestream):
                obs_rhs_T = pmap_local.to_map(tod=obs, signal=obs.dsT, comps='T', det_weights=2*nmat.ivar)
                
                obs_rhs_demodQ = pmap_local.to_map(tod=obs, signal=obs.demodQ, comps='QU', det_weights=nmat.ivar)
                obs_rhs_demodU = pmap_local.to_map(tod=obs, signal=obs.demodU, comps='QU', det_weights=nmat.ivar)
                obs_rhs_demodQU = pmap_local.zeros(super_shape=(2), comps='QU',)
                
                obs_rhs_demodQU[0][:] = obs_rhs_demodQ[0] - obs_rhs_demodU[1]
                obs_rhs_demodQU[1][:] = obs_rhs_demodQ[1] + obs_rhs_demodU[0]
                del obs_rhs_demodQ, obs_rhs_demodU
                
                # we write into the obs_rhs. 
                obs_rhs[0] = obs_rhs_T[0]
                obs_rhs[1] = obs_rhs_demodQU[0]
                obs_rhs[2] = obs_rhs_demodQU[1]
                del obs_rhs_demodQU, obs_rhs_T
                                
                obs_div    = pmap_local.zeros(super_shape=(self.ncomp, self.ncomp))
                # Build the per-pixel inverse covmat for this observation
                wT = pmap_local.to_weights(obs, signal=obs.dsT, comps='T', det_weights=2*nmat.ivar,)
                wQU = pmap_local.to_weights(obs, signal=obs.demodQ, comps='T', det_weights=nmat.ivar,)
                obs_div[0,0] = wT
                obs_div[1,1] = wQU
                obs_div[2,2] = wQU
                # Build hitcount
                Nd[0,:] = 1
                obs_hits = pmap_local.to_map(signal=Nd[0],)
            else:
                pmap_local.to_map(dest=obs_rhs, signal=Nd, comps='TQU')
                obs_div    = pmap_local.zeros(super_shape=(self.ncomp,self.ncomp))
                pmap_local.to_weights(dest=obs_div, signal=Nd, comps='TQU')
                Nd[:] = 1
                obs_hits = pmap_local.to_map(signal=Nd,)
                
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
        self.ready = True

    @property
    def ncomp(self): return len(self.comps)

    def write(self, prefix, tag, m):
        if not self.output: return
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if self.tiled:
            tilemap.write_map(oname, m)
        else:
            enmap.write_map(oname, m)
        return oname

def make_atomic_map(context, obslist, shape, wcs, noise_model, L,
                    preprocess_config, comps="TQU", t0=0,
                    dtype_tod=np.float32, dtype_map=np.float32,
                    tag="", verbose=0, split_labels=None,
                    site='so_sat3', recenter=None, outs=None, singlestream=False):
    """
        Initialize a FilterBin Mapmaker for demodulated data

        Arguments
        ---------
        context : sotodlib.core.Context
            Context object used to load obs from.
        obslist : dict
            The obslist which is the output of the
            mapmaking.obs_grouping.build_obslists, contains the information of the
            single or multiple obs to map.
        shape : tuple
            Shape of the geometry to use for mapping.
        wcs : dict
            WCS kernel of the geometry to use for mapping.
        noise_model : sotodlib.mapmaking.Nmat
            Noise model to pass to DemodMapmaker.
        L : logger
            Logger for printing on the screen.
        preprocess_config : dict
            Dictionary with the config yaml file for the preprocess database.
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
        site : str, optional
            Plataform name for the pointing matrix.
        recenter : str or None
            String to make object-centered maps, such as Moon/Sun/Planet centered maps.
            Look at sotodlib.mapmaking.parse_recentering for details.
        outs : list, optional
            List to accumulate the outputs from preproc_or_load_group. This is
            necessary if you end up preprocessing an obs here and want to save
            to the database so you won't have to re-process again.
        singlestream : Bool
            If True, do not perform demodulated filter+bin mapmaking but
            rather regular filter+bin mapmaking, i.e. map from obs.signal
            rather than from obs.dsT, obs.demodQ, obs.demodU.

        Example usage ::
        """

    pre = "" if tag is None else tag + " "
    L.info(pre + "Initializing equation system")

    # Set up our mapmaking equation
    if split_labels==None:
        # this is the case where we did not request any splits at all
        signal_map = DemodSignalMap(shape, wcs, comps=comps,
                                              dtype=dtype_map, tiled=False,
                                              ofmt="", singlestream=singlestream,
                                              recenter=recenter )
    else:
        # this is the case where we asked for at least 2 splits (1 split set).
        # We count how many split we'll make, we need to define the Nsplits
        # maps inside the DemodSignalMap
        Nsplits = len(split_labels)
        signal_map = DemodSignalMap(shape, wcs, comps=comps,
                                              dtype=dtype_map, tiled=False,
                                              ofmt="", Nsplits=Nsplits,
                                              singlestream=singlestream,
                                              recenter=recenter)
    signals    = [signal_map]
    mapmaker   = DemodMapmaker(signals, noise_model=noise_model,
                                         dtype=dtype_tod,
                                         verbose=verbose>0,
                                         singlestream=singlestream)
    L.info(pre + "Building RHS")
    # And feed it with our observations
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        error, output, obs = site_pipeline.preprocess_tod.preproc_or_load_group(obs_id, configs=preprocess_config,
                            dets={'wafer_slot':detset, 'wafer.bandpass':band}, 
                            logger=L, context=context, overwrite=False)

        obs.wrap("weather", np.full(1, "toco"))
        obs.wrap("site",    np.full(1, site))
        obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds 
                       + obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags, )
        
        if error not in [None,'load_success']:
            L.info('tod %s:%s:%s failed in the prepoc database'%(obs_id,detset,band))
            continue
        outs.append(error)
        outs.append(output)
        # And add it to the mapmaker
        if split_labels==None:
            # this is the case of no splits
            mapmaker.add_obs(name, obs)
        else:
            # this is the case of having splits. We need to pass the split_labels
            # at least. If we have detector splits fixed in time, then we pass the
            # masks in det_split_masks. Otherwise, det_split_masks will be None.
            mapmaker.add_obs(name, obs, split_labels=split_labels)
        L.info('Done with tod %s:%s:%s'%(obs_id,detset,band))

    for signal in signals:
        signal.prepare()
    L.info(pre + "Writing F+B outputs")
    wmap = []
    weights = []
    for n_split in range(signal_map.Nsplits):
        wmap.append( signal_map.rhs[n_split] )
        div = np.diagonal(signal_map.div[n_split], axis1=0, axis2=1)
        div = np.moveaxis(div, -1, 0) # this moves the last axis to the 0th position
        weights.append(div)
    return bunch.Bunch(wmap=wmap, weights=weights, signal=signal_map, t0=t0)