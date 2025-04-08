import os

import numpy as np
from pixell import bunch
from pixell import utils as putils

from ..noise_model import NmatUncorr
from ..utils import MultiZipper


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
        self.dof = MultiZipper()
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


# COMMENTED OUT FOR NOW

# # Needed for multi-beam support in SignalMap. These should probably be moved somewhere
# # eventually. Maybe Projectionist should be generalized to support multiple beams so
# # shit somewhat fragile construction isn't needed in the first place.
# class MultiProj(so3g.proj.Projectionist):
#     def to_map(self, signal, assemblies, output=None, det_weights=None, threads=None, comps=None):
#         for ai, assembly in enumerate(assemblies):
#             output = super().to_map(signal, assembly, output=output, det_weights=det_weights,
#                 threads=threads[ai] if threads is not None else None, comps=comps)
#         return output
#     def to_weights(self, signal, assemblies, output=None, det_weights=None, threads=None, comps=None):
#         for ai, assembly in enumerate(assemblies):
#             output = super().to_weights(signal, assembly, output=output, det_weights=det_weights,
#                 threads=threads[ai] if threads is not None else None, comps=comps)
#         return output
#     def from_map(self, src_map, assemblies, signal=None, comps=None):
#         for ai, assembly in enumerate(assemblies):
#             signal = super().from_map(src_map, assembly, signal=signal, comps=comps)
#         return signal
#     def get_active_tiles(self, assemblies, assign=False):
#         infos = []
#         for ai, assembly in enumerate(assemblies):
#             infos.append(super().get_active_tiles(assembly, assign=assign))
#         active = list(np.unique(np.concatenate([info["active_tiles"] for info in infos])))
#         # Mapping from old active to new
#         amap   = [putils.find(active, info["active_tiles"]) for info in infos]
#         hits   = np.zeros(len(active),int)
#         for i, info in enumerate(infos):
#             hits[amap[i]] += info["hit_counts"]
#         # group_ranges and group_tiles are harder. We skip them for now.
#         # They are only used by the "tiles" thread partitioning, not the
#         # standard "domdir"
#         return {"active_tiles":active, "hit_counts":hits}
#     def assign_threads(self, assemblies, method='domdir', n_threads=None):
#         return [super().assign_threads(assembly, method=method, n_threads=n_threads) for assembly in assemblies]
#     def assign_threads_from_map(self, assemblies, tmap, n_threads=None):
#         return [super().assign_threads_from_map(assembly, tmap, n_threads=n_threads) for assembly in assemblies]

# class PmatMultibeam(coords.pmat.P):
#     def __init__(self, fps, **kwargs):
#         super().init(fp=fps[0], **kwargs)
#         self.fps = fps
#     @classmethod
#     def for_tod(cls, tod, focal_planes, threads=None, **kwargs):
#         tmp = super().for_tod(tod, **kwargs)
#         fps = [coords.helpers.get_fplane(tod, focal_plane=fplane) for fplane in focal_planes]
#         return cls(sight=tmp.sight, fps=fps, geom=tmp.geom, comps=tmp.comps,
#             cuts=tmp.cuts, threads=threads, det_weights=det_weights, interpol=intepol)
#     def _get_proj(self):
#         if self.geom is None:
#             raise ValueError("Can't project without a geometry!")
#         # Backwards compatibility for old so3g
#         interpol_kw = coords._get_interpol_args(self.interpol)
#         if self.tiled:
#             return MultiProj.for_tiled(
#                 self.geom.shape, self.geom.wcs, self.geom.tile_shape,
#                 active_tiles=self.active_tiles, **interpol_kw)
#         else:
#             return MultiProj.for_geom(self.geom.shape, self.geom.wcs, **interpol_kw)
#     def _get_asm(self):
#         return [so3g.proj.Assembly.attach(self.sight, fp) for fp in self.fps]

# # Need a way to get PmatMultibeam into SignalMap. Either SignalMap must
# # construct it itself, or it must be passed to its add_obs. We don't have
# # direct access to its add_obs here though, and it's not really the
# # mapmaking class' business to deal with pointing details like this.
# # Cleanest solution is probably to modify the way SignalMap.add_obs
# # allocates its pointing matrix, making it make a normal P unless obs
# # is populated with multibeam info (though, since multibeam is a superset
# # of single-beam, one could just use multibeam always. Longer term it might
# # be best to just modify so3g to have multibeam support from the start.
# # But for now let's keep it here.
# #
# # How should the multibeam info be recorded in obs? Each beam is fully
# # described by a focal_plane entry. Could modify that to be a list, but
# # this might break things, and it could still be good to keep the standard
# # responses available. So how about making a separate multibeam structure
# # that containes this more detailed description of the beam? It would then
# # be populated from simpler leakage beam parameters. Let's go with that for now.

# def get_pmap(obs, geom, recenter=None, **kwargs):
#     if recenter:
#         t0  = obs.timestamps[obs.samps.count//2]
#         rot = recentering_to_quat_lonlat(*evaluate_recentering(recenter,
#             ctime=t0, geom=geom, site=unarr(obs.site)))
#     else: rot = None
#     if "multibeam" in obs:
#         return PmatMultibeam.for_tod(obs, focal_planes=obs.multibeam, geom=geom, rot=rot, **kwargs)
#     else:
#         return coords.pmat.P.for_tod(obs, geom=geom, rot=rot, **kwargs)
