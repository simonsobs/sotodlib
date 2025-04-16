# Copyright (c) 2020-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Operator for interfacing with the Maximum Likelihood Mapmaker.

"""

import os

import numpy as np
import traitlets
from astropy import units as u
from pixell import enmap, tilemap, fft, utils, bunch

import toast
from toast.traits import trait_docs, Unicode, Int, Instance, Bool, Float, List
from toast.ops import Operator
from toast.utils import Logger, Environment, rate_from_times
from toast.timing import function_timer, Timer
from toast.observation import default_values as defaults
from toast.fft import FFTPlanReal1DStore
from toast.instrument_coords import xieta_to_quat, quat_to_xieta

import so3g

# The mapmaking tools import pixell.mpi, which will try to import
# mpi4py if it is available, even if running on a login node.
# Check to see if MPI is enabled in toast and if not, disable here.
if not toast.mpi.use_mpi:
    os.environ["DISABLE_MPI"] = "true"

from ... import mapmaking as mm
from ...core import AxisManager, IndexAxis, OffsetAxis, LabelAxis, FlagManager


@trait_docs
class MLMapmaker(Operator):
    """Operator which accumulates data to the Maximum Likelihood Mapmaker.

    """

    # Class traits

    # SKN note: I think this style is *much* more readable, but I won't change it for now
    #API       = Int(0, help="Internal interface version for this operator")
    #out_dir   = Unicode(".", help="The output directory")
    #area      = Unicode(None, allow_none=True, help="Load the enmap geometry from this file")
    #center_at = Unicode(None, allow_none=True, help="The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]")
    #comps     = Unicode("T", help="Components (must be 'T', 'QU' or 'TQU')")
    #Nmat      = Instance(allow_none=True, klass=mm.Nmat, help="The noise matrix to use")
    #dtype_map = Instance(klass=np.dtype, args=(np.float64,), help="Numpy dtype of map products")
    #times     = Unicode(obs_names.times, help="Observation shared key for timestamps")
    #boresight = Unicode(obs_names.boresight_azel, help="Observation shared key for boresight Az/El")
    #det_data  = Unicode(obs_names.det_data, help="Observation detdata key for the timestream data")
    #det_flags = Unicode(None, allow_none=True, help="Observation detdata key for flags to use")
    #det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")
    #shared_flags  = Unicode(None, allow_none=True, help="Observation shared key for telescope flags to use")
    #shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")
    #view      = Unicode(None, allow_none=True, help="Use this view of the data in all observations")
    #noise_model  = Unicode("noise_model", help="Observation key containing the noise model")
    #purge_det_data = Bool(False, help="If True, clear all observation detector data after accumulating")
    #tiled     = Bool(False, help="If True, the map will be represented as distributed tiles in memory. For large maps this is faster and more memory efficient, but for small maps it has some overhead due to extra communication.")
    #verbose   = Int(1, allow_none=True, help="Set verbosity in MLMapmaker.  If None, use toast loglevel")
    #weather   = Unicode("typical", help="Weather to assume when making maps")
    #site      = Unicode("so",      help="Site to use when making maps")


    API = Int(0, help="Internal interface version for this operator")

    out_dir = Unicode(".", help="The output directory")

    area = Unicode(None, allow_none=True, help="Load the enmap geometry from this file")

    center_at = Unicode(
        None,
        allow_none=True,
        help="The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]",
    )

    interpol = Unicode(
        "nearest",
        help="Either one or a comma-separated list of interpolation modes",
    )

    downsample = List(
        [1],
        help="Downsample TOD by these factors.",
    )

    comps = Unicode("TQU", help="Components (must be 'T', 'QU' or 'TQU')")

    Nmat = Instance(klass=mm.Nmat, allow_none=True, help="The noise matrix to use")

    nmat_type = Unicode(
        "NmatDetvecs",
        help="Noise matrix type is either `NmatDetvecs`, `NmatUncorr` or `Nmat`",
    )

    nmat_mode = Unicode(
        "build",
        help="How to initialize the noise matrix.  "
        "'build': Always build from data in obs.  "
        "'cache': Use if available in nmat_dir, otherwise build and save.  "
        "'load': Load from nmat_dir, error if missing.  "
        "'save': Build from obs data and save."
    )

    nmat_dir = Unicode(
        None,
        allow_none=True,
        help="Where to read/write/cache noise matrices. See nmat_mode. "
        "If None, write to {out_dir}/nmats"
    )

    srcsamp = Unicode(
        None,
        allow_none=True,
        help="path to mask file where True regions indicate where bright object "
        "mitigation should be applied. Mask is in equatorial coordinates. "
        "Not tiled, so should be low-res to not waste memory."
    )

    dtype_map = Unicode("float64", help="Numpy dtype of map products")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    boresight = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight Az/El"
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    purge_det_data = Bool(
        False,
        help="If True, clear all observation detector data after accumulating",
    )

    checkpoint_interval = Int(
        0,
        help="If greater than zero, the CG solver will store its state and"
        "restart from a checkpoint when available.",
    )

    skip_existing = Bool(
        False,
        help="If True, the mapmaker will not write any map products that "
        "already exist on disk.  See `checkpoint`."
    )

    tiled = Bool(
        True,
        help="If True, the map will be represented as distributed tiles in memory. "
        "For large maps this is faster and more memory efficient, but for small "
        "maps it has some overhead due to extra communication."
    )

    deslope = Bool(
        True,
        help="If True, each observation will have the mean and slope removed.",
    )

    verbose = Int(
        1,
        allow_none=True,
        help="Set verbosity in MLMapmaker.  If None, use toast loglevel",
    )

    weather = Unicode("vacuum", help="Weather to assume when making maps")
    site    = Unicode("so",     help="Site to use when making maps")

    maxiter = List(
        [500],
        help="List of maximum number of CG iterations for each pass.",
    )

    maxerr = Float(1e-6, help="Maximum error in the CG solver")

    truncate_tod = Bool(
        True,
        help="Truncate TOD to an easily factorizable length to ensure efficient FFT.",
    )

    write_div = Bool(True, help="Write out the noise weight map")
    write_hits= Bool(True, help="Write out the hitcount map")

    write_rhs = Bool(
        True, help="Write out the right hand side of the mapmaking equation"
    )

    write_bin = Bool(True, help="Write out the binned map")

    write_iter_map = Int(
        10, help="Number of iterations between saved maps.  Set to zero to disable."
    )

    @traitlets.validate("comps")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["T", "QU", "TQU"]:
            raise traitlets.TraitError("Invalid comps (must be 'T', 'QU' or 'TQU')")
        return check

    @traitlets.validate("checkpoint_interval")
    def _check_checkpoint_interval(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Invalid checkpoint_interval. Must be non-negative.")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("dtype_map")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check not in ["float", "float64"]:
            raise traitlets.TraitError(
                "Map data type must be float64 until so3g.ProjEng supports "
                "other map data types."
            )
        return check

    @traitlets.validate("verbose")
    def _check_params(self, proposal):
        check = proposal["value"]
        if check is None:
            # Set verbosity from the toast loglevel
            env = Environment.get()
            level = env.log_level()
            if level == "VERBOSE":
                check = 3
            elif level == "DEBUG":
                check = 2
            else:
                check = 1
        return check

    @traitlets.validate("maxerr")
    def _check_maxerr(self, proposal):
        check = proposal["value"]
        if check <= 0:
            raise traitlets.TraitError("Maxerr should be greater than zero")
        return check

    @traitlets.validate("write_iter_map")
    def _check_maxerr(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("write_iter_map must be nonnegative")
        return check

    @traitlets.validate("nmat_type")
    def _check_nmat_type(self, proposal):
        check = proposal["value"]
        allowed = ["NmatUncorr", "NmatDetvecs", "Nmat"]
        if check not in allowed:
            msg = f"nmat_type must be one of {allowed}, not {check}"
            raise traitlets.TraitError(msg)
        return check

    def __init__(self, **kwargs):
        self.shape = None
        self.wcs = None
        self.recenter = None
        self.signal_map = None
        self.mapmaker = None
        super().__init__(**kwargs)

    @function_timer
    def setup_passes(self):
        tmp = bunch.Bunch()
        tmp.downsample = self.downsample
        tmp.maxiter = self.maxiter
        tmp.interpol = self.interpol.split(",")
        # The entries may have different lengths. We use the max
        # and then pad the others by repeating the last element.
        # The final output will be a list of bunches
        npass = max([len(tmp[key]) for key in tmp])
        passes = []
        for i in range(npass):
            entry = bunch.Bunch()
            for key in tmp:
                entry[key] = tmp[key][min(i, len(tmp[key]) - 1)]
            passes.append(entry)
        return passes

    @function_timer
    def _load_noise_model(self, ob, npass, ipass, gcomm):
        # Maybe load precomputed noise model
        log = Logger.get()
        if self.nmat_dir is None:
            nmat_dir  = os.path.join(self.out_dir, "nmats")
        else:
            nmat_dir  = self.nmat_dir
        if npass != 1:
            nmat_dir += f"_pass{ipass + 1}"
        nmat_file = os.path.join(nmat_dir, f"nmat_{ob.name}.hdf")
        there = os.path.isfile(nmat_file)
        if self.nmat_mode == "load" and not there:
            raise RuntimeError(
                f"Nmat mode is 'load' but {nmat_file} does not exist."
            )
        if self.nmat_mode == "load" or (self.nmat_mode == "cache" and there):
            log.debug_rank(f"Loading noise model from '{nmat_file}'", comm=gcomm)
            try:
                nmat = mm.read_nmat(nmat_file)
            except Exception as e:
                if self.nmat_mode == "cache":
                    log.info_rank(
                        f"Failed to load noise model from '{nmat_file}'"
                        f" : '{e}'. Will cache a new one",
                        comm=gcomm,
                    )
                    nmat = None
                else:
                    msg = f"Failed to load noise model from '{nmat_file}' : {e}"
                    raise RuntimeError(msg)
        else:
            nmat = None
        return nmat, nmat_file

    @function_timer
    def _save_noise_model(self, mapmaker, nmat, nmat_file, gcomm):
        # Maybe save the noise model we built (only if we actually built one rather than
        # reading one in)
        log = Logger.get()
        if self.nmat_mode in ["save", "cache"] and nmat is None:
            log.debug_rank(f"Writing noise model to '{nmat_file}'", comm=gcomm)
            nmat_dir = os.path.dirname(nmat_file)
            os.makedirs(nmat_dir, exist_ok=True)
            mm.write_nmat(nmat_file, mapmaker.data[-1].nmat)
        return

    @function_timer
    def _wrap_obs(self, ob, dets, passinfo):
        """ Prepare data for the mapmaker """

        # Get the focalplane for this observation
        fp = ob.telescope.focalplane

        # Get the sample rate from the data.  We also have nominal sample rates
        # from the noise model and also from the focalplane.
        # (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
        #     ob.shared[self.times].data
        # )

        axdets = LabelAxis("dets", dets)
        nsample = int(ob.n_local_samples)

        axsamps = OffsetAxis(
            "samps",
            count=nsample,
            offset=ob.local_index_offset,
            origin_tag=ob.name,
        )

        # Convert the data view into a RangesMatrix
        ranges = so3g.proj.ranges.RangesMatrix.zeros((len(dets), nsample))
        if self.view is not None:
             view_ranges = np.array(
                 [[x.first, min(x.last, nsample) + 1] for x in ob.intervals[self.view]]
             )
             ranges += so3g.proj.ranges.Ranges.from_array(view_ranges, nsample)

        # Convert the focalplane offsets into the expected form
        det_to_row = {y["name"]: x for x, y in enumerate(fp.detector_data)}
        det_quat = np.array([fp.detector_data["quat"][det_to_row[x]] for x in dets])
        xi, eta, gamma = quat_to_xieta(det_quat)

        axfp = AxisManager()
        axfp.wrap("xi", xi, axis_map=[(0, axdets)])
        axfp.wrap("eta", eta, axis_map=[(0, axdets)])
        axfp.wrap("gamma", gamma, axis_map=[(0, axdets)])

        # Convert Az/El quaternion of the detector back into
        # angles from the simulation.
        theta, phi, pa = toast.qarray.to_iso_angles(ob.shared[self.boresight])

        # Azimuth is measured in the opposite direction from longitude
        az = 2 * np.pi - phi
        el = np.pi / 2 - theta
        roll = pa

        axbore = AxisManager()
        axbore.wrap("az", az, axis_map=[(0, axsamps)])
        axbore.wrap("el", el, axis_map=[(0, axsamps)])
        axbore.wrap("roll", roll, axis_map=[(0, axsamps)])

        axobs = AxisManager()
        axobs.wrap("focal_plane", axfp)
        axobs.wrap("timestamps", ob.shared[self.times], axis_map=[(0, axsamps)])
        axobs.wrap(
            "signal",
            np.vstack(ob.detdata[self.det_data][dets, :]),
            axis_map=[(0, axdets), (1, axsamps)],
        )
        axobs.wrap("boresight", axbore)
        axobs.wrap('flags', FlagManager.for_tod(axobs))
        axobs.flags.wrap("glitch_flags", ranges, axis_map=[(0, axdets), (1, axsamps)])
        axobs.wrap("weather", np.full(1, self.weather))
        axobs.wrap("site",    np.full(1, "so"))

        if self.truncate_tod:
            # FFT-truncate for faster fft ops
            axobs.restrict("samps", [0, fft.fft_len(axobs.samps.count)])

        # MLMapmaker.add_obs will apply deslope
        if self.deslope:
            utils.deslope(axobs.signal, w=5, inplace=True)

        if self.downsample != 1:
            axobs = mm.downsample_obs(axobs, passinfo.downsample)

        # NOTE:  Expected contents look like:
        # >>> tod
        # AxisManager(signal[dets,samps], timestamps[samps], readout_filter_cal[dets],
        # mce_filter_params[6], iir_params[3,5], flags*[samps], boresight*[samps],
        # array_data*[dets], pointofs*[dets], focal_plane*[dets], abscal[dets],
        # timeconst[dets], glitch_flags[dets,samps], source_flags[dets,samps],
        # relcal[dets], dets:LabelAxis(63), samps:OffsetAxis(372680))
        # >>> tod.focal_plane
        # AxisManager(xi[dets], eta[dets], gamma[dets], dets:LabelAxis(63))
        # >>> tod.boresight
        # AxisManager(az[samps], el[samps], roll[samps], samps:OffsetAxis(372680))

        return axobs

    @function_timer
    def _init_mapmaker(
            self, mapmaker, signal_map, mapmaker_prev, eval_prev, comm, gcomm, prefix,
    ):
        """
        This function is run at the end of the pass, runs the prepare, writes the maps
        and sets up the initial condition
        """
        log = Logger.get()
        timer = Timer()
        timer.start()

        mapmaker.prepare()

        if self.tiled:
            # Each group reports how many tiles they are using
            geo_work = mapmaker.signals[1].geo_work
            nactive = len(geo_work.active)
            ntile = np.prod(geo_work.shape[-2:])
            log.debug_rank(f"{nactive} / {ntile} tiles active", comm=gcomm)

        if comm is not None:
            comm.barrier()
        log.info_rank(
            f"MLMapmaker finished prepare in",
            comm=comm,
            timer=timer,
        )

        # This will need to be modified for more general cases where we don't solve for
        # a sky map, or where we solve for multiple sky maps. The mapmaker itself supports it,
        # the problem is the direct access to the rhs, div and idiv members
        if self.write_rhs:
            fname = f"{prefix}sky_rhs.fits"
            if self.skip_existing and os.path.isfile(fname):
                log.info_rank(f"Skipping existing rhs in {fname}", comm=comm)
            else:
                fname = signal_map.write(prefix, "rhs", signal_map.rhs)
                log.info_rank(f"Wrote rhs to {fname}", comm=comm)

        if self.write_div:
            fname = f"{prefix}sky_div.fits"
            if self.skip_existing and os.path.isfile(fname):
                log.info_rank(f"Skipping existing div in {fname}", comm=comm)
            else:
                # FIXME : only writing the TT variance to avoid integer overflow in communication
                fname = signal_map.write(prefix, "div", signal_map.div)
                # fname = signal_map.write(prefix, "div", signal_map.div[0, 0])
                log.info_rank(f"Wrote div to {fname}", comm=comm)

        if self.write_hits:
            fname = f"{prefix}sky_hits.fits"
            if self.skip_existing and os.path.isfile(fname):
                log.info_rank(f"Skipping existing div in {fname}", comm=comm)
            else:
                fname = signal_map.write(prefix, "hits", signal_map.hits)
                log.info_rank(f"Wrote hits to {fname}", comm=comm)

        mmul = tilemap.map_mul if self.tiled else enmap.map_mul
        if self.write_bin:
            fname = f"{prefix}sky_bin.fits"
            if self.skip_existing and os.path.isfile(fname):
                log.info_rank(f"Skipping existing bin in {fname}", comm=comm)
            else:
                fname = signal_map.write(
                    prefix, "bin", mmul(signal_map.idiv, signal_map.rhs)
                )
                log.info_rank(f"Wrote bin to {fname}", comm=comm)

        if comm is not None:
            comm.barrier()
        log.info_rank(f"MLMapmaker finished writing rhs, div, bin in", comm=comm, timer=timer)

        # Set up initial condition
        if eval_prev is None:
            # this will be the first pass
            x0 = None
        else:
            x0 = mapmaker.translate(mapmaker_prev, eval_prev.x_zip)
        return x0

    @function_timer
    def _apply_mapmaker(self, mapmaker, x0, passinfo, prefix, comm):
        log = Logger.get()
        timer = Timer()
        timer.start()
        tstep = Timer()
        tstep.start()

        if self.checkpoint_interval > 0:
            fname_checkpoint = f"{prefix}checkpoint.{comm.rank:04}.hdf"
            there = os.path.isfile(fname_checkpoint)
            if there:
                log.info_rank(f"Checkpoint detected. Will start from previous solver state", comm=comm)
        else:
            fname_checkpoint = None

        for step in mapmaker.solve(
                maxiter=passinfo.maxiter,
                maxerr=self.maxerr,
                x0=x0,
                fname_checkpoint=fname_checkpoint,
                checkpoint_interval=self.checkpoint_interval,
        ):
            if self.write_iter_map < 1:
                dump = False
            else:
                dump = step.i % self.write_iter_map == 0
            msg = f"CG step {step.i:4d} {step.err:15.7e} write={dump}"
            log.info_rank(f"MLMapmaker   {msg} ", comm=comm, timer=tstep)
            if dump:
                for signal, val in zip(mapmaker.signals, step.x):
                    if signal.output:
                        fname = signal.write(prefix, f"map{step.i:04}", val)
                        log.info_rank(f"Wrote signal to {fname} in", comm=comm, timer=tstep)

        log.info_rank(f"MLMapmaker finished solve in", comm=comm, timer=timer)

        for signal, val in zip(mapmaker.signals, step.x):
            if signal.output:
                fname = signal.write(prefix, "map", val)
                log.info_rank(f"Wrote {fname}", comm=comm)

        if comm is not None:
            comm.barrier()
        log.info_rank(f"MLMapmaker wrote map in", comm=comm, timer=timer)

        return mapmaker, mapmaker.evaluator(step.x_zip)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        comm = data.comm.comm_world
        gcomm = data.comm.comm_group
        timer.start()

        if comm is None and self.tiled:
            log.info("WARNING: Tiled mapmaking not supported without MPI.")
            self.tiled = False

        for trait in ["area"]:
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(
                    f"You must set `{trait}` before running MLMapmaker"
                )

        # nmat_type is guaranteed to be a valid Nmat class
        if self.nmat_type == 'NmatDetvecs':
            noise_model = getattr(mm, self.nmat_type)(downweight=[1e-4, 0.25, 0.50], window=0)
        else:
            noise_model = getattr(mm, self.nmat_type)()

        shape, wcs = enmap.read_map_geometry(self.area)

        if self.center_at is None:
            recenter = None
        else:
            recenter = mm.parse_recentering(self.center_at)
        dtype_tod = np.float32
        dtype_map = np.dtype(self.dtype_map)

        prefix = os.path.join(self.out_dir, f"{self.name}_")

        passes = self.setup_passes()
        npass = len(passes)
        mapmaker_prev = None
        eval_prev = None

        if self.srcsamp is None:
            srcsamp_mask = None
        else:
            if not os.path.isfile(self.srcsamp):
                raise RuntimeError(f"Source mask does not exist: {self.srcsamp}")
            srcsamp_mask = enmap.read_map(self.srcsamp)

        for ipass, passinfo in enumerate(passes):
            # The multipass mapmaking loop
            log.info_rank(
                f"Starting pass {ipass + 1}/{npass}, maxit={passinfo.maxiter} "
                f"down={passinfo.downsample}, interp={passinfo.interpol}",
                comm=comm,
            )
            if npass == 1:
                pass_prefix = prefix
            else:
                pass_prefix = f"{prefix}pass{ipass + 1}_"

            signal_cut = mm.SignalCut(comm, dtype=dtype_tod)
            signal_map = mm.SignalMap(
                shape,
                wcs,
                comm,
                comps=self.comps,
                dtype=dtype_map,
                recenter=recenter,
                tiled=self.tiled,
                interpol=passinfo.interpol,
            )
            signals = [signal_cut, signal_map]
            if srcsamp_mask is not None:
                signal_srcsamp = mm.SignalSrcsamp(comm, srcsamp_mask, dtype=dtype_tod)
                signals.append(signal_srcsamp)
            mapmaker = mm.MLMapmaker(
                signals, noise_model=noise_model, dtype=dtype_tod, verbose=self.verbose
            )

            for ob in data.obs:
                # Get the detectors we are using locally for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue

                nmat, nmat_file = self._load_noise_model(ob, npass, ipass, gcomm)

                # wrap_obs finishes in line 250 of make_ml_map.py, at the downsampling
                axobs = self._wrap_obs(ob, dets, passinfo)

                if ipass > 0:
                    # Evaluate the final model of the previous pass' mapmaker
                    # for this observation.
                    signal_estimate = eval_prev.evaluate(mapmaker_prev.data[len(mapmaker.data)])
                    # Resample this to the current downsampling level
                    signal_estimate = mm.resample.resample_fft_simple(signal_estimate, axobs.samps.count)
                else:
                    signal_estimate = None

                mapmaker.add_obs(
                    ob.name,
                    axobs,
                    deslope=self.deslope,
                    noise_model=nmat,
                    signal_estimate=signal_estimate,
                )
                del axobs
                del signal_estimate

                self._save_noise_model(mapmaker, nmat, nmat_file, gcomm)

                # Optionally delete the input detector data to save memory, if
                # the calling code knows that no additional operators will be
                # used afterwards.
                if ipass == npass - 1 and self.purge_det_data:
                    del ob.detdata[self.det_data]

            if comm is not None:
                comm.barrier()
            log.info_rank(
                f"MLMapmaker wrapped observations in",
                comm=comm,
                timer=timer,
            )

            # _init_mapmaker covers lines 293-303 of make_ml_map.py
            x0 = self._init_mapmaker(
                mapmaker,
                signal_map,
                mapmaker_prev,
                eval_prev,
                comm,
                gcomm,
                pass_prefix,
            )
            # _apply_mapmaker covers lines 305-320 of make_ml_map.py
            mapmaker_prev, eval_prev = self._apply_mapmaker(mapmaker, x0, passinfo, pass_prefix, comm)

            # Save metadata, may get dropped later

            self.shape = shape
            self.wcs = wcs
            self.recenter = recenter
            self.signal_map = signal_map
            self.mapmaker = mapmaker

        return

    @function_timer
    def _finalize(self, data, **kwargs):
        pass

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "shared": [
                self.times,
            ],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"detdata": list()}
        if self.det_out is not None:
            prov["detdata"].append(self.det_out)
        return prov

    def _accelerators(self):
        return list()
