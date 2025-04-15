
import warnings

import h5py
import numpy as np
import so3g
from pixell import bunch, enmap, tilemap
from pixell import utils as putils

from ... import coords
from ..pointing_matrix import PmatCut
from ..utils import (ArrayZipper, MapZipper, TileMapZipper,
                     evaluate_recentering, recentering_to_quat_lonlat,
                     safe_invert_div, unarr)
from .base_signal import Signal


class SignalMap(Signal):
    """Signal describing a non-distributed sky map."""

    def __init__(
        self,
        shape,
        wcs,
        comm,
        comps="TQU",
        name="sky",
        ofmt="{name}",
        output=True,
        ext="fits",
        dtype=np.float32,
        sys=None,
        recenter=None,
        tile_shape=(500, 500),
        tiled=False,
        interpol=None,
        cut_field= "flags.glitch_flags",
    ):
        """Signal describing a sky map in the coordinate system given by "sys", which defaults
        to equatorial coordinates. If tiled==True, then this will be a distributed map with
        the given tile_shape, otherwise it will be a plain enmap. interpol controls the
        pointing matrix interpolation mode. See so3g's Projectionist docstring for details.
        """
        Signal.__init__(self, name, ofmt, output, ext)
        self.comm = comm
        self.comps = comps
        self.sys = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
        self.interpol = interpol
        self.data = {}
        self.cut_field = cut_field
        ncomp = len(comps)
        shape = tuple(shape[-2:])

        if tiled:
            geo = tilemap.geometry(shape, wcs, tile_shape=tile_shape)
            self.rhs = tilemap.zeros(geo.copy(pre=(ncomp,)), dtype=dtype)
            self.div = tilemap.zeros(geo.copy(pre=(ncomp, ncomp)), dtype=dtype)
            self.hits = tilemap.zeros(geo.copy(pre=()), dtype=dtype)
        else:
            self.rhs = enmap.zeros((ncomp,) + shape, wcs, dtype=dtype)
            self.div = enmap.zeros((ncomp, ncomp) + shape, wcs, dtype=dtype)
            self.hits = enmap.zeros(shape, wcs, dtype=dtype)

    def add_obs(self, id, obs, nmat, Nd, pmap=None, cuts=None):
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        Nd     = Nd.copy()  # This copy can be avoided if build_obs is split into two parts
        ctime  = obs.timestamps
        if cuts is None:
            cuts = obs[self.cut_field]
        pcut   = PmatCut(cuts)  # could pass this in, but fast to construct
        if pmap is None:
            # Build the local geometry and pointing matrix for this observation
            if self.recenter:
                rot = recentering_to_quat_lonlat(
                    *evaluate_recentering(
                        self.recenter,
                        ctime=ctime[len(ctime) // 2],
                        geom=self.rhs.geometry,
                        site=unarr(obs.site),
                    )
                )
            else:
                rot = None
            pmap = coords.pmat.P.for_tod(
                obs,
                comps=self.comps,
                geom=self.rhs.geometry,
                rot=rot,
                threads="domdir",
                weather=unarr(obs.weather),
                site=unarr(obs.site),
                interpol=self.interpol,
            )
        # Build the RHS for this observation
        pcut.clear(Nd)
        obs_rhs = pmap.zeros()
        pmap.to_map(dest=obs_rhs, signal=Nd)
        # Build the per-pixel inverse covmat for this observation
        obs_div = pmap.zeros(super_shape=(self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            obs_div[i] = 0
            obs_div[i, i] = 1
            Nd[:] = 0
            pmap.from_map(obs_div[i], dest=Nd)
            pcut.clear(Nd)
            Nd = nmat.white(Nd)
            obs_div[i] = 0
            pmap.to_map(signal=Nd, dest=obs_div[i])
        # Build hitcount
        Nd[:] = 1
        pcut.clear(Nd)
        obs_hits = pmap.to_map(signal=Nd)
        del Nd

        # Update our full rhs and div. This works for both plain and distributed maps
        self.rhs = self.rhs.insert(obs_rhs, op=np.ndarray.__iadd__)
        self.div = self.div.insert(obs_div, op=np.ndarray.__iadd__)
        self.hits = self.hits.insert(obs_hits[0], op=np.ndarray.__iadd__)
        # Save the per-obs things we need. Just the pointing matrix in our case.
        # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
        self.data[id] = bunch.Bunch(pmap=pmap, obs_geo=obs_rhs.geometry)

    def prepare(self):
        """Called when we're done adding everything. Sets up the map distribution,
        degrees of freedom and preconditioner."""
        if self.ready:
            return
        if self.tiled:
            self.geo_work = self.rhs.geometry
            self.rhs = tilemap.redistribute(self.rhs, self.comm)
            self.div = tilemap.redistribute(self.div, self.comm)
            self.hits = tilemap.redistribute(self.hits, self.comm)
            self.dof = TileMapZipper(
                self.rhs.geometry, dtype=self.dtype, comm=self.comm
            )
        else:
            if self.comm is not None:
                self.rhs = putils.allreduce(self.rhs, self.comm)
                self.div = putils.allreduce(self.div, self.comm)
                self.hits = putils.allreduce(self.hits, self.comm)
            self.dof = MapZipper(*self.rhs.geometry, dtype=self.dtype)
        self.idiv = safe_invert_div(self.div)
        self.ready = True

    @property
    def ncomp(self):
        return len(self.comps)

    def forward(self, id, tod, map, tmul=1, mmul=1):
        """map2tod operation. For tiled maps, the map should be in work distribution,
        as returned by unzip. Adds into tod."""
        if id not in self.data:
            return  # Should this really skip silently like this?
        if tmul != 1:
            tod *= tmul
        if mmul != 1:
            map = map * mmul
        self.data[id].pmap.from_map(dest=tod, signal_map=map, comps=self.comps)

    def backward(self, id, tod, map, tmul=1, mmul=1):
        """tod2map operation. For tiled maps, the map should be in work distribution,
        as returned by unzip. Adds into map"""
        if id not in self.data:
            return
        if tmul != 1:
            tod = tod * tmul
        if mmul != 1:
            map *= mmul
        self.data[id].pmap.to_map(signal=tod, dest=map, comps=self.comps)

    def precon(self, map):
        if self.tiled:
            return tilemap.map_mul(self.idiv, map)
        else:
            return enmap.map_mul(self.idiv, map)

    def to_work(self, map):
        if self.tiled:
            return tilemap.redistribute(map, self.comm, self.geo_work.active)
        else:
            return map.copy()

    def from_work(self, map):
        if self.tiled:
            return tilemap.redistribute(map, self.comm, self.rhs.geometry.active)
        else:
            if self.comm is None:
                return map
            else:
                return putils.allreduce(map, self.comm)

    def wzeros(self):
        """Like to_work, but zeroed instead of containing the signal. Much cheaper"""
        return tilemap.zeros(self.geo_work, self.rhs.dtype) if self.tiled else enmap.zeros(*self.rhs.geometry,  self.rhs.dtype)

    def write(self, prefix, tag, m, unit='K'):
        if not self.output:
            return
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if self.tiled:
            tilemap.write_map(oname, m, self.comm, extra={'BUNIT':unit})
        else:
            if self.comm is None or self.comm.rank == 0:
                enmap.write_map(oname, m, extra={'BUNIT':unit})
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

class SignalCut(Signal):
    def __init__(
        self,
        comm,
        name="cut",
        ofmt="{name}_{rank:02}",
        dtype=np.float32,
        output=False,
        cut_type=None,
        cut_field = "flags.glitch_flags",
    ):
        """Signal for handling the ML solution for the values of the cut samples."""
        Signal.__init__(self, name, ofmt, output, ext="hdf")
        self.comm = comm
        self.data = {}
        self.dtype = dtype
        self.cut_type = cut_type
        self.off = 0
        self.rhs = []
        self.div = []
        self.cut_field = cut_field

    def add_obs(self, id, obs, nmat, Nd, cuts=None):
        """Add and process an observation. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        Nd = Nd.copy()  # This copy can be avoided if build_obs is split into two parts
        if cuts is None: cuts = obs[self.cut_field]
        pcut = PmatCut(cuts, model=self.cut_type)
        # Build our RHS
        obs_rhs = np.zeros(pcut.njunk, self.dtype)
        pcut.backward(Nd, obs_rhs)
        # Build our per-pixel inverse covmat
        obs_div = np.ones(pcut.njunk, self.dtype)
        Nd[:] = 0
        pcut.forward(Nd, obs_div)
        Nd *= nmat.ivar[:, None]
        pcut.backward(Nd, obs_div)
        self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off + pcut.njunk)
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
        return junk / self.div

    def backward(self, id, tod, junk):
        if id not in self.data: return
        d = self.data[id]
        d.pcut.backward(tod, junk[d.i1:d.i2])

    def wzeros(self): return np.zeros_like(self.rhs)

    def write(self, prefix, tag, m):
        if not self.output: return
        rank = 0 if self.comm is None else self.comm.rank
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
        res = np.full(self.off, np.nan, self.dtype)
        for id in self.data:
            sdata = self.data[id]
            odata = other.data[id]
            so3g.translate_cuts(
                odata.pcut.cuts,
                sdata.pcut.cuts,
                sdata.pcut.model,
                sdata.pcut.params,
                junk[odata.i1:odata.i2],
                res [sdata.i1:sdata.i2],
            )
        # This check should be cheap enough to be worth it.
        # It's only called once every mapmaking pass anyway.
        # It should not trigger during normal circumstances,
        # but is useful for catching bugs when changing how
        # the cuts work.
        if np.any(~np.isfinite(res)):
            warnings.warn("Incompatible cuts in translate(): Probably caused by differing numbers of cut ranges. Skipping translate for these cuts")
            res[:] = 0
        return res

class SignalSrcsamp(SignalCut):
    def __init__(
        self,
        comm,
        mask,
        sys=None,
        recenter=None,
        name="srcsamp",
        ofmt="{name}_{rank:02}",
        dtype=np.float32,
        map_dtype=np.float64,
        output=False,
        # Prior parameters
        eps_edge=10, eps_core=0.01, redge=2*putils.arcmin,
    ):
        """Signal for eliminating model error near small areas in the map with bright objects"""
        SignalCut.__init__(self, comm, name=name, ofmt=ofmt, dtype=dtype, output=output, cut_type="full")
        self.map_dtype= map_dtype
        # Mask should have shape (1,ny,nx).
        # Non-tiled, so mask should be low-res! 4 arcmin should work fine.
        # That's just 25 MB for our patch
        mask          = mask.preflat[0]
        self.distmap  = mask.distance_transform().astype(map_dtype)
        self.mask     = mask.astype(map_dtype)
        self.sys      = sys
        self.recenter = recenter
        # Prior stuff. The prior breaks the degeneracy between the map
        # and these extra degrees offreedom. The prior strength changes
        # smoothly around the edge of the mask to prevent discontinuities
        # in the final map. This is controlled by eps_core, eps_edge and redge
        self.eps_edge = eps_edge
        self.eps_core = eps_core
        self.redge    = redge
        # Distance of of our extra degrees of freedom
        self.distsamps  = []

    def add_obs(self, id, obs, nmat, Nd):
        """Add and process an observation. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        # First scan our mask to find which samples need this
        # treatment
        ctime  = obs.timestamps
        if self.recenter:
            rec = evaluate_recentering(self.recenter, ctime=ctime[len(ctime) // 2],
                    geom=(self.mask.shape, self.mask.wcs), site=unarr(obs.site))
            rot = recentering_to_quat_lonlat(*rec)
        else: rot = None
        pmap = coords.pmat.P.for_tod(obs, comps="T", geom=self.mask.geometry,
            rot=rot, threads="domdir", weather=unarr(obs.weather),
            site=unarr(obs.site), interpol="nearest")
        tod = np.zeros((obs.dets.count, obs.samps.count), Nd.dtype)
        pmap.from_map(self.mask, dest=tod)
        cuts= so3g.proj.RangesMatrix.from_mask(tod>0.5)
        # Then build our RHS as normal
        tod[:]  = Nd.copy()
        pcut    = PmatCut(cuts, model=self.cut_type)
        obs_rhs = np.zeros(pcut.njunk, self.dtype)
        pcut.backward(tod, obs_rhs)
        # Build our per-pixel inverse covmat
        # This assumes cut_type="full"!
        obs_div = np.zeros(pcut.njunk, self.dtype)
        tod[:]  = nmat.ivar[:,None]
        pcut.backward(tod, obs_div)
        # Get the distance of each of our samples from the mask
        # This assumes cut_type="full"!
        obs_dist= np.zeros(pcut.njunk, self.dtype)
        pmap.from_map(self.distmap, dest=tod)
        pcut.backward(tod, obs_dist)
        # Finished!
        self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off + pcut.njunk)
        self.off += pcut.njunk
        self.rhs.append(obs_rhs)
        self.div.append(obs_div)
        self.distsamps.append(obs_dist)

    def prepare(self):
        """Process the added observations, determining our degrees of freedom etc.
        Should be done before calling forward and backward."""
        if self.ready: return
        SignalCut.prepare(self)
        self.distsamps= np.concatenate(self.distsamps)
        x             = np.minimum(self.distsamps/self.redge, 1)
        self.epsilon  = np.exp(np.log(self.eps_edge) * (1-x) + np.log(self.eps_core) * x)
        self.epsilon *= self.div
        # Can now delete distansamps to save memory. epsilon is the
        # thing we'll actually use in the prior
        self.distsamps= None

    def prior(self, xin, xout):
        xout += xin * self.epsilon

    def translate(self, other, junk):
        """Translate junk degrees of freedom from one SignalSrcsamp representation
        to another, e.g. changing sample rate. Currently simply initializes the
        new ones to zero, since so3g.translate_cuts can't handle it when the number
        of cut ranges per detector has changed, which can happen with the way we build
        the srcsamp cuts. It could be supported if necessary, but it's not high priority."""
        return np.zeros(self.off, self.dtype)

