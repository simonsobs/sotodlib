import os
import warnings

import numpy as np
import h5py
import so3g
from typing import Optional
from pixell import bunch, enmap, tilemap
from pixell import utils as putils

from . import utils as smutils
from .. import coords
from .pointing_matrix import PmatCut
from .noise_model import NmatUncorr

class MLMapmaker:
    def __init__(
        self,
        signals=[],
        noise_model=None,
        dtype=np.float32,
        verbose=False,
        glitch_flags: str = "flags.glitch_flags",
    ):
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
        self.signals = signals
        self.dtype = dtype
        self.verbose = verbose
        self.noise_model = noise_model
        self.data = []
        self.dof = smutils.MultiZipper()
        self.ready = False
        self.glitch_flags_path = glitch_flags

    def add_obs(self, id, obs, deslope=True, noise_model=None, signal_estimate=None):
        # Prepare our tod
        ctime = obs.timestamps
        srate = (len(ctime) - 1) / (ctime[-1] - ctime[0])
        tod = obs.signal.astype(self.dtype, copy=False)
        # Subtract an existing estimate of the signal before estimating
        # the noise model, if available
        if signal_estimate is not None:
            tod -= signal_estimate
        if deslope:
            putils.deslope(tod, w=5, inplace=True)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None:
            noise_model = self.noise_model
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
                putils.deslope(tod, w=5, inplace=True)
        # And apply it to the tod
        tod = nmat.apply(tod)
        # Add the observation to each of our signals
        for signal in self.signals:
            signal.add_obs(id, obs, nmat, tod)
        # Save what we need about this observation
        self.data.append(
            bunch.Bunch(
                id=id,
                ndet=obs.dets.count,
                nsamp=len(ctime),
                dets=obs.dets.vals,
                nmat=nmat,
            )
        )

    def prepare(self):
        if self.ready:
            return
        for signal in self.signals:
            signal.prepare()
            self.dof.add(signal.dof)
        self.ready = True

    def evaluator(self, x_zip):
        """Return a helper object that lets one evaluate the data model
        Px for the zipped solution x for a TOD"""
        return MLEvaluator(x_zip, self.signals, self.dof, dtype=self.dtype)
    def accumulator(self):
        """Return a helper object that lets one accumulate P'd for
        for a TOD"""
        return MLAccumulator(self.signals, self.dof)

    def A(self, x_zip):
        # unzip goes from flat array of all the degrees of freedom to individual maps, cuts etc.
        # to_work makes a scratch copy and does any redistribution needed
        evaluator   = self.evaluator(x_zip)
        accumulator = self.accumulator()
        for di, data in enumerate(self.data):
            tod = evaluator.evaluate(data)
            data.nmat.apply(tod)
            accumulator.accumulate(data, tod)
        return accumulator.finish()

    def M(self, x_zip):
        iwork = self.dof.unzip(x_zip)
        result = self.dof.zip(
            *[signal.precon(w) for signal, w in zip(self.signals, iwork)]
        )
        return result

    def solve(
        self,
        maxiter=500,
        maxerr=1e-9,
        x0=None,
        fname_checkpoint=None,
        checkpoint_interval=1,
    ):
        self.prepare()
        rhs = self.dof.zip(*[signal.rhs for signal in self.signals])

        solver = putils.CG(self.A, rhs, M=self.M, dot=self.dof.dot, x0=x0)
        # If there exists a checkpoint, restore solver state
        if fname_checkpoint is None:
            checkpoint = False
            restart = False
        else:
            checkpoint = True
            outdir = os.path.dirname(fname_checkpoint)
            if len(outdir) != 0:
                os.makedirs(outdir, exist_ok=True)
            if os.path.isfile(fname_checkpoint):
                solver.load(fname_checkpoint)
                restart = True
            else:
                restart = False
        while restart or (solver.i < maxiter and solver.err > maxerr):
            if restart:
                # When restarting, do not step
                restart = False
            else:
                solver.step()
                if checkpoint and solver.i % checkpoint_interval == 0:
                    # Avoid checkpoint corruption by making a copy of the previous checkpoint
                    if os.path.isfile(fname_checkpoint):
                        os.replace(fname_checkpoint, fname_checkpoint + ".old")
                    # Write a checkpoint
                    solver.save(fname_checkpoint)
            # x is the unzipped solution. It's a list of one object per signal we solve for.
            # x_zip is the raw solution, as a 1d vector.
            yield bunch.Bunch(i=solver.i, err=solver.err, x=self.dof.unzip(solver.x), x_zip=solver.x)

    def translate(self, other, x_zip):
        """Translate degrees of freedom x from some other mapamaker to the current one.
        The other mapmaker must have the same list of signals, except that they can have
        different sample rate etc. than this one. See the individual Signal-classes
        translate methods for details. This is used in multipass mapmaking.
        """
        x     = other.dof.unzip(x_zip)
        x_new = []
        for ssig, osig, oval in zip(self.signals, other.signals, x):
            x_new.append(ssig.translate(osig, oval))
        return self.dof.zip(*x_new)

class MLEvaluator:
    """Helper for MLMapmaker that represents the action of P in the model d = Px+n."""
    def __init__(self, x_zip, signals, dof, dtype=np.float32):
        self.signals = signals
        self.x_zip   = x_zip
        self.dof     = dof
        self.dtype   = dtype
        self.iwork = [signal.to_work(m) for signal, m in zip(self.signals, self.dof.unzip(x_zip))]
    def evaluate(self, data, tod=None):
        """Evaluate Px for one tod"""
        if tod is None: tod = np.zeros([data.ndet, data.nsamp], self.dtype)
        for si, signal in reversed(list(enumerate(self.signals))):
            signal.forward(data.id, tod, self.iwork[si])
        return tod

class MLAccumulator:
    """Helper for MLMapmaker that represents the action of P.T in the model d = Px+n."""
    def __init__(self, signals, dof):
        self.signals = signals
        self.dof     = dof
        self.owork = [signal.wzeros() for signal in signals]
    def accumulate(self, data, tod):
        """Accumulate P'd for one tod"""
        for si, signal in enumerate(self.signals):
            signal.backward(data.id, tod, self.owork[si])
    def finish(self):
        """Return the full P'd based on the previous accumulation"""
        return self.dof.zip(
            *[signal.from_work(w) for signal, w in zip(self.signals, self.owork)]
        )

class Signal:
    """This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""

    def __init__(self, name, ofmt, output, ext, **kwargs):
        """Initialize a Signal. It probably doesn't make sense to construct a generic signal
        directly, though. Use one of the subclasses.
        Arguments:
        * name: The name of this signal, e.g. "sky", "cut", etc.
        * ofmt: The format used when constructing output file prefix
        * output: Whether this signal should be part of the output or not.
        * ext: The extension used for the files.
        * **kwargs: additional keyword based parameters, accessible as class parameters
        """
        self.name = name
        self.ofmt = ofmt
        self.output = output
        self.ext = ext
        self.dof = None
        self.ready = False
        self.__dict__.update(kwargs)

    def add_obs(self, id, obs, nmat, Nd, **kwargs):
        pass

    def prepare(self):
        self.ready = True

    def forward(self, id, tod, x):
        pass

    def backward(self, id, tod, x):
        pass

    def precon(self, x):
        return x

    def to_work(self, x):
        return x.copy()

    def from_work(self, x):
        return x

    def wzeros(self):
        return 0

    def write(self, prefix, tag, x):
        pass

    def translate(self, other, x):
        return x

    def prior(self, xin, xout):
        pass


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
        if cuts is None: cuts = obs[self.cut_field]
        pcut   = PmatCut(cuts)  # could pass this in, but fast to construct
        if pmap is None:
            # Build the local geometry and pointing matrix for this observation
            if self.recenter:
                rot = smutils.recentering_to_quat_lonlat(
                    *smutils.evaluate_recentering(
                        self.recenter,
                        ctime=ctime[len(ctime) // 2],
                        geom=self.rhs.geometry,
                        site=smutils.unarr(obs.site),
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
                weather=smutils.unarr(obs.weather),
                site=smutils.unarr(obs.site),
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
            self.dof = smutils.TileMapZipper(
                self.rhs.geometry, dtype=self.dtype, comm=self.comm
            )
        else:
            if self.comm is not None:
                self.rhs = putils.allreduce(self.rhs, self.comm)
                self.div = putils.allreduce(self.div, self.comm)
                self.hits = putils.allreduce(self.hits, self.comm)
            self.dof = smutils.MapZipper(*self.rhs.geometry, dtype=self.dtype)
        self.idiv = smutils.safe_invert_div(self.div)
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
        if self.tiled: return tilemap.zeros(self.geo_work, self.rhs.dtype)
        else:          return enmap.zeros(*self.rhs.geometry,  self.rhs.dtype)

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
        self.dof = smutils.ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
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
        if self.recenter:
            rec = smutils.evaluate_recentering(self.recenter, ctime=ctime[len(ctime) // 2],
                    geom=(self.mask.shape, self.mask.wcs), site=smutils.unarr(obs.site))
            rot = smutils.recentering_to_quat_lonlat(*rec)
        else: rot = None
        pmap = coords.pmat.P.for_tod(obs, comps="T", geom=self.mask.geometry,
            rot=rot, threads="domdir", weather=smutils.unarr(obs.weather),
            site=smutils.unarr(obs.site), interpol="nearest")
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

# Needed for multi-beam support in SignalMap. These should probably be moved somewhere
# eventually. Maybe Projectionist should be generalized to support multiple beams so
# shit somewhat fragile construction isn't needed in the first place.
class MultiProj(so3g.proj.Projectionist):
    def to_map(self, signal, assemblies, output=None, det_weights=None, threads=None, comps=None):
        for ai, assembly in enumerate(assemblies):
            output = super().to_map(signal, assembly, output=output, det_weights=det_weights,
                threads=threads[ai] if threads is not None else None, comps=comps)
        return output
    def to_weights(self, assemblies, output=None, det_weights=None, threads=None, comps=None):
        for ai, assembly in enumerate(assemblies):
            output = super().to_weights(signal, assembly, output=output, det_weights=det_weights,
                threads=threads[ai] if threads is not None else None, comps=comps)
        return output
    def from_map(self, src_map, assemblies, signal=None, comps=None):
        for ai, assembly in enumerate(assemblies):
            signal = super().from_map(src_map, assembly, signal=signal, comps=comps)
        return signal
    def get_active_tiles(self, assemblies, assign=False):
        infos = []
        for ai, assembly in enumerate(assemblies):
            infos.append(super().get_active_tiles(assembly, assign=assign))
        active = list(np.unique(np.concatenate([info["active_tiles"] for info in infos])))
        # Mapping from old active to new
        amap   = [putils.find(active, info["active_tiles"]) for info in infos]
        hits   = np.zeros(len(active),int)
        for i, info in enumerate(infos):
            hits[amap[i]] += info["hit_counts"]
        # group_ranges and group_tiles are harder. We skip them for now.
        # They are only used by the "tiles" thread partitioning, not the
        # standard "domdir"
        return {"active_tiles":active, "hit_counts":hits}
    def assign_threads(self, assemblies, method='domdir', n_threads=None):
        return [super().assign_threads(assembly, method=method, n_threads=n_threads) for assembly in assemblies]
    def assign_threads_from_map(self, assembly, tmap, n_threads=None):
        return [super().assign_threads_from_map(assembly, tmap, n_threads=n_threads) for assembly in assemblies]

class PmatMultibeam(coords.pmat.P):
    def __init__(self, fps, **kwargs):
        super().init(fp=fps[0], **kwargs)
        self.fps = fps
    @classmethod
    def for_tod(cls, tod, focal_planes, threads=None, **kwargs):
        tmp = super().for_tod(tod, **kwargs)
        fps = [coords.helpers.get_fplane(tod, focal_plane=fplane) for fplane in focal_planes]
        return cls(sight=tmp.sight, fps=fps, geom=tmp.geom, comps=tmp.comps,
            cuts=tmp.cuts, threads=threads, det_weights=det_weights, interpol=intepol)
    def _get_proj(self):
        if self.geom is None:
            raise ValueError("Can't project without a geometry!")
        # Backwards compatibility for old so3g
        interpol_kw = _get_interpol_args(self.interpol)
        if self.tiled:
            return MultiProj.for_tiled(
                self.geom.shape, self.geom.wcs, self.geom.tile_shape,
                active_tiles=self.active_tiles, **interpol_kw)
        else:
            return MultiProj.for_geom(self.geom.shape, self.geom.wcs, **interpol_kw)
    def _get_asm(self):
        return [so3g.proj.Assembly.attach(self.sight, fp) for fp in self.fps]

# Need a way to get PmatMultibeam into SignalMap. Either SignalMap must
# construct it itself, or it must be passed to its add_obs. We don't have
# direct access to its add_obs here though, and it's not really the
# mapmaking class' business to deal with pointing details like this.
# Cleanest solution is probably to modify the way SignalMap.add_obs
# allocates its pointing matrix, making it make a normal P unless obs
# is populated with multibeam info (though, since multibeam is a superset
# of single-beam, one could just use multibeam always. Longer term it might
# be best to just modify so3g to have multibeam support from the start.
# But for now let's keep it here.
#
# How should the multibeam info be recorded in obs? Each beam is fully
# described by a focal_plane entry. Could modify that to be a list, but
# this might break things, and it could still be good to keep the standard
# responses available. So how about making a separate multibeam structure
# that containes this more detailed description of the beam? It would then
# be populated from simpler leakage beam parameters. Let's go with that for now.

def get_pmap(obs, geom, recenter=None, **kwargs):
    if recenter:
        t0  = obs.timestamps[obs.samps.count//2]
        rot = recentering_to_quat_lonlat(*evaluate_recentering(recenter,
            ctime=t0, geom=geom, site=smutils.unarr(obs.site)))
    else: rot = None
    if "multibeam" in obs:
        return PmatMultibeam.for_tod(obs, focal_planes=obs.multibeam, geom=geom, rot=rot, **kwargs)
    else:
        return coords.pmat.P.for_tod(obs, geom=geom, rot=rot, **kwargs)
