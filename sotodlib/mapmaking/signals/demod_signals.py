import numpy as np
from pixell import bunch, enmap, tilemap
from pixell import utils as putils

from ... import coords
from ..utils import (evaluate_recentering, import_optional,
                     recentering_to_quat_lonlat, unarr)
from .base_signal import Signal

hp = import_optional('healpy')
h5py = import_optional('h5py')


class DemodSignalMap(Signal):
    def __init__(self, shape=None, wcs=None, comm=None, comps="TQU", name="sky", ofmt="{name}", output=True,
                 ext="fits", dtype_map=np.float64, dtype_tod=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False, Nsplits=1, singlestream=False,
                 nside=None, nside_tile=None):
        """
        Parent constructor; use for_rectpix or for_healpix instead. Args described there.
        """
        Signal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype_map = dtype_map
        self.dtype_tod = dtype_tod
        self.tile_shape = tile_shape        
        self.tiled = tiled
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
        self.wrapper = lambda x : x
        self.wcs = wcs
        ncomp      = len(comps)

        self.pix_scheme = "rectpix" if (wcs is not None) else "healpix"
        if self.pix_scheme == "healpix":
            self.tiled = (nside_tile is not None)
            self.hp_geom = coords.healpix_utils.get_geometry(nside, nside_tile, ordering='NEST')
            npix = 12 * nside**2
            self.rhs = np.zeros((Nsplits, ncomp, npix), dtype=dtype_map)
            self.div = np.zeros((Nsplits, ncomp, ncomp, npix), dtype=dtype_map)
            self.hits = np.zeros((Nsplits, npix), dtype=dtype_map)

            if self.tiled:
                self.wrapper = coords.healpix_utils.tiled_to_full
        else:
            if shape is None:
                # We will set shape, wcs from wcs_kernel on loading the first obs                
                self.rhs = None
                self.div = None
                self.hits = None
            else:
                self.init_maps_rectpix(shape, wcs)

    @classmethod
    def for_rectpix(cls, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype_map=np.float64, dtype_tod=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False, Nsplits=1, singlestream=False):
        """
        Signal describing a non-distributed sky map. Signal describing a sky map in the coordinate 
        system given by "sys", which defaults to equatorial coordinates. If tiled==True, then this 
        will be a distributed map with the given tile_shape, otherwise it will be a plain enmap.
        
        Arguments
        ---------
        shape : numpy.ndarray
            Shape of the output map geometry. If None, computed from coords.get_footprint on first add_obs.
        wcs : wcs
            WCS of the output map geometry (or wcs kernel).
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
        dtype_map : numpy.dtype, optional
            The data type to use for the maps.
        dtype_tod : numpy.dtype, optional
            The data type to use for the time ordered data
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
                   ext=ext, dtype_map=dtype_map, dtype_tod=dtype_tod, sys=sys, recenter=recenter, tile_shape=tile_shape, tiled=tiled,
                   Nsplits=Nsplits, singlestream=singlestream, nside=None, nside_tile=None)

    @classmethod
    def for_healpix(cls, nside, nside_tile=None, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits.gz", dtype_map=np.float64, dtype_tod=np.float32, Nsplits=1, singlestream=False):
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
        dtype_map : numpy.dtype, optional
            The data type to use for the maps.
        dtype_tod : numpy.dtype, optional
            The data type to use for the time ordered data
        Nsplits : int, optional
            Number of splits that you will map simultaneously. By default is 1 when no
            splits are requested.
        singlestream : Bool, optional
            If True, do not perform demodulated filter+bin mapmaking but rather regular
            filter+bin mapmaking, i.e. map from obs.signal rather than from obs.dsT,
            obs.demodQ, obs.demodU
        """
        return cls(shape=None, wcs=None, comm=None, comps=comps, name=name, ofmt=ofmt, output=output,
                   ext=ext, dtype_map=dtype_map, dtype_tod=dtype_tod, sys=None, recenter=None, tile_shape=None, tiled=False,
                   Nsplits=Nsplits, singlestream=singlestream, nside=nside, nside_tile=nside_tile)

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None):
        # Nd will have 3 components, corresponding to ds_T, demodQ, demodU with the noise model applied
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        ctime = obs.timestamps
        for n_split in range(self.Nsplits):
            if pmap is None:
                # Build the local geometry and pointing matrix for this observation
                rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter,
                                                                       ctime=ctime[len(ctime)//2],
                                                                       geom=(self.rhs.shape, self.rhs.wcs),
                                                                       site=unarr(obs.site))) if self.recenter else None
                if self.Nsplits == 1:
                    # this is the case with no splits
                    cuts = obs.flags.glitch_flags
                else:
                    # remember that the dets or samples you want to keep should be false, hence we negate
                    cuts = obs.flags.glitch_flags + ~obs.preprocess.split_flags.cuts[split_labels[n_split]]
                if self.pix_scheme == "rectpix":
                    threads='domdir'
                    if self.rhs is None: # Still need to initialize the geometry
                        geom = None
                        wcs_kernel = self.wcs
                    else:
                        geom = self.rhs.geometry
                        wcs_kernel = None
                else:
                    threads = ["tiles", "simple"][self.hp_geom.nside_tile is None]
                    geom = self.hp_geom
                    wcs_kernel = None
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=geom,
                                                   rot=rot, wcs_kernel=wcs_kernel, threads=threads,
                                                   weather=unarr(obs.weather), site=unarr(obs.site),
                                                   cuts=cuts, hwp=True)
            else:
                pmap_local = pmap

            if self.rhs is None: # Set the geometry now from the pmat
                shape, wcs = pmap_local.geom
                self.init_maps_rectpix(shape, wcs)
                self.wcs = wcs

            if not(self.singlestream):
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap=pmap_local, signalT=obs.dsT.astype(self.dtype_tod), 
                                                               signalQ=obs.demodQ.astype(self.dtype_tod), signalU=obs.demodU.astype(self.dtype_tod),
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
        if self.pix_scheme == "healpix" or self.ready:
            return
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

    def write(self, prefix, tag, m, unit='K'):
        if not self.output:
            return
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if self.pix_scheme == "rectpix":
            if self.tiled:
                tilemap.write_map(oname, m, self.comm, extra={'BUNIT':unit})
            else:
                if self.comm is None or self.comm.rank == 0:
                    enmap.write_map(oname, m, extra={'BUNIT':unit})
        else:
            if self.ext in ["fits", "fits.gz"]:
                if hp is None:
                    raise ImportError("Cannot save healpix map as fits; healpy could not be imported. Install healpy or save as npy or h5")
                if m.ndim > 2:
                    m = np.reshape(m, (np.prod(m.shape[:-1]), m.shape[-1]), order='C') # Flatten wrapping axes; healpy.write_map can't handle >2d array
                hp.write_map(oname, m.view(self.dtype_map), nest=(self.hp_geom.ordering=='NEST'), overwrite=True, column_units=unit)
            elif self.ext == "npy":
                np.save(oname, {'nside': self.hp_geom.nside, 'ordering':self.hp_geom.ordering, 'data':m.view(self.dtype_map)}, allow_pickle=True)
            elif self.ext in ['h5', 'hdf5']:
                if h5py is None:
                    raise ValueError("Cannot save healpix map as hdf5; h5py could not be imported. Install h5py or save as npy or fits")
                with h5py.File(oname, 'w') as f:
                    dset = f.create_dataset("data", m.shape, dtype=self.dtype_map, data=m)
                    dset.attrs['ordering'] = self.hp_geom.ordering
                    dset.attrs['nside'] = self.hp_geom.nside
            else:
                raise ValueError(f"Unknown extension {self.ext}")

        return oname

    def init_maps_rectpix(self, shape, wcs):
        """ Initialize tilemaps or enmaps rhs, div, hits for given shape and wcs"""
        shape = tuple(shape[-2:])
        Nsplits, ncomp = self.Nsplits, self.ncomp
        
        if self.tiled:
            geo = tilemap.geometry(shape, wcs, tile_shape=self.tile_shape)
            rhs = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,)),      dtype=self.dtype_map)
            div = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,ncomp)), dtype=self.dtype_map)
            hits= tilemap.zeros(geo.copy(pre=(Nsplits,)),            dtype=self.dtype_map)
        else:
            rhs = enmap.zeros((Nsplits, ncomp)     +shape, wcs, dtype=self.dtype_map)
            div = enmap.zeros((Nsplits,ncomp,ncomp)+shape, wcs, dtype=self.dtype_map)
            hits= enmap.zeros((Nsplits,)+shape, wcs, dtype=self.dtype_map)
        self.rhs = rhs
        self.div = div
        self.hits = hits
        return rhs, div, hits
    
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
        sub_info.total_weight_qu = sumweights
        sub_info.mean_weight_qu = meanweights
        sub_info.median_weight_qu = medianweights
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
