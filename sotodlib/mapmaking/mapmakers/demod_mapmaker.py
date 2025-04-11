"""
This submodule contains classes to perform filter+bin mapmaking for 
demodulated data. The DemodMapmaker class creates a mapmaker object.
DemodSignalMap creates a signal you want to solve for, over which 
you accumulate observations into the div and rhs maps. For examples
how to use look at docstring of DemodMapmaker.
"""
import numpy as np
from pixell import bunch, mpi

from ..noise_model import NmatWhite
from ..signals import DemodSignalMap, add_weights_to_info
from ..utils import MultiZipper, atomic_db_aux


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
        dtype : numpy.dtype, optional
            The data type to use for the time-ordered data. Only tested
            with float32
        verbose : Bool, optional
            Whether to print progress messages. Not implemented
        comps : str, optional
            String with the components to solve for. Not implemented for 
            anything other than TQU
        singlestream : Bool, optional
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
            tod = np.array([obs.dsT.astype(self.dtype, copy=False),
                            obs.demodQ.astype(self.dtype, copy=False),
                            obs.demodU.astype(self.dtype, copy=False)])
        else:
            tod = obs.signal.astype(self.dtype, copy=False)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None: 
            noise_model = self.noise_model
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


def setup_demod_map(noise_model, shape=None, wcs=None, nside=None,
                    comm=mpi.COMM_WORLD, comps='TQU', split_labels=None,
                    singlestream=False, dtype_tod=np.float32,
                    dtype_map=np.float64, recenter=None, verbose=0):
    """
    Setup the classes for demod mapmaking and return
    a DemodMapmmaker object
    """
    if wcs is not None:
        Nsplits = len(split_labels)
        signal_map = DemodSignalMap.for_rectpix(shape, wcs, comm, comps=comps,
                                              dtype_map=dtype_map, dtype_tod=dtype_tod, tiled=False,
                                              ofmt="", Nsplits=Nsplits,
                                              singlestream=singlestream,
                                              recenter=recenter)
    elif nside is not None:
        Nsplits = len(split_labels)
        signal_map = DemodSignalMap.for_healpix(nside, nside_tile='auto', 
                                    comps=comps, dtype_map=dtype_map, dtype_tod=dtype_tod,
                                    ofmt="", Nsplits=Nsplits,
                                    singlestream=singlestream,
                                    ext="fits.gz")
    signals    = [signal_map]
    mapmaker   = DemodMapmaker(signals, noise_model=noise_model,
                                         dtype=dtype_tod,
                                         verbose=verbose>0,
                                         singlestream=singlestream)
    return mapmaker

def write_demod_maps(prefix, data, info, unit='K', split_labels=None, atomic_db=None):
    """
    Write maps from data into files
    """
    Nsplits = len(split_labels)
    for n_split in range(Nsplits):
        if np.all(data.wmap[n_split] == 0.0):
            if atomic_db is not None:
                atomic_db_aux(atomic_db, info[n_split], valid=False)
            continue
        data.signal.write(prefix, "%s_wmap"%split_labels[n_split],
                          data.wmap[n_split], unit=unit+'^-1')
        data.signal.write(prefix, "%s_weights"%split_labels[n_split],
                          data.weights[n_split], unit=unit+'^2')
        data.signal.write(prefix, "%s_hits"%split_labels[n_split],
                          data.signal.hits[n_split], unit='hits')
        if atomic_db is not None:
            atomic_db_aux(atomic_db, info[n_split], valid=True)

def make_demod_map(context, obslist, noise_model, info,
                    preprocess_config, prefix, shape=None, wcs=None,
                    nside=None, comm=mpi.COMM_WORLD, comps="TQU", t0=0,
                    dtype_tod=np.float32, dtype_map=np.float32,
                    tag="", verbose=0, split_labels=None, L=None,
                    site='so_sat3', recenter=None, singlestream=False,
                    atomic_db=None, unit='K'):
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
    preprocess_config : list of dict
        List of dictionaries with the config yaml file for the preprocess database.
        If two, then a multilayer preprocessing is to be used.
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
    atomic_db : str, optional
        Path to the atomic map data base. Maps created will be added to it.

    Returns
    -------
    errors : list
        List of errors from preprocess database. To be used in cleanup_mandb.
    outputs : list
        List of outputs from preprocess database. To be used in cleanup_mandb.
    """
    from ..preprocess import preprocess_util

    #context = core.Context(context)
    if L is None:
        L = preprocess_util.init_logger("Demod filterbin mapmaking")
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: 
        L.info(pre + "Initializing equation system")
    mapmaker = setup_demod_map(noise_model, shape=shape, wcs=wcs, nside=nside,
                    comm=comm, comps=comps, split_labels=split_labels,
                    singlestream=singlestream, dtype_tod=dtype_tod,
                    dtype_map=dtype_map, recenter=recenter, verbose=verbose)

    if comm.rank == 0: 
        L.info(pre + "Building RHS")
    # And feed it with our observations
    nobs_kept  = 0
    
    # PENDING: do an allreduce of these, not needed for atomic maps, but needed for
    # depth-1 maps
    errors = [] 
    outputs = []
    
    if len(preprocess_config)==1:
        preproc_init = preprocess_config[0]
        preproc_proc = None
    else:
        preproc_init = preprocess_config[0]
        preproc_proc = preprocess_config[1]

    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        error, output_init, output_proc, obs = preprocess_util.preproc_or_load_group(obs_id,
                                                configs_init=preproc_init,
                                                configs_proc=preproc_proc,
                                                dets={'wafer_slot':detset, 'wafer.bandpass':band},
                                                logger=L,
                                                overwrite=False)
        errors.append(error) ; outputs.append((output_init, output_proc))
        if error not in [None,'load_success']:
            L.info('tod %s:%s:%s failed in the preproc database'%(obs_id,detset,band))
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
    if comm.rank == 0:
        L.info(pre + "Writing F+B outputs")
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
    write_demod_maps(prefix, mapdata, info, split_labels=split_labels, atomic_db=atomic_db, unit=unit)

    return errors, outputs
