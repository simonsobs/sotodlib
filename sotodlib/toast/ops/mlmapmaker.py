# Copyright (c) 2020-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Operator for interfacing with the Maximum Likelihood Mapmaker.

"""

import os

import numpy as np
import traitlets
from astropy import units as u
from pixell import enmap, tilemap, fft

import toast
from toast.traits import trait_docs, Unicode, Int, Instance, Bool, Float
from toast.ops import Operator
from toast.utils import Logger, Environment, rate_from_times
from toast.timing import function_timer, Timer
from toast.observation import default_values as defaults
from toast.fft import FFTPlanReal1DStore
from toast.instrument_coords import xieta_to_quat, quat_to_xieta

import so3g

from ... import mapmaking as mm
from ...core import AxisManager, IndexAxis, OffsetAxis, LabelAxis


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

    comps = Unicode("T", help="Components (must be 'T', 'QU' or 'TQU')")

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

    tiled = Bool(
        False,
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

    maxiter = Int(500, help="Maximum number of CG iterations")

    maxerr = Float(1e-6, help="Maximum error in the CG solver")

    truncate_tod = Bool(
        False,
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

    @traitlets.validate("maxiter")
    def _check_maxiter(self, proposal):
        check = proposal["value"]
        if check <= 0:
            raise traitlets.TraitError("Maxiter should be greater than zero")
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
        super().__init__(**kwargs)
        self._mapmaker = None

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ["area"]:
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(
                    f"You must set `{trait}` before running MLMapmaker"
                )

        # nmat_type is guaranteed to be a valid Nmat class
        self.Nmat = getattr(mm, self.nmat_type)()

        comm = data.comm.comm_world
        gcomm = data.comm.comm_group

        if self._mapmaker is None:
            # First call- create the mapmaker instance.
            # Get the timestream dtype from the first observation
            self._dtype_tod = data.obs[0].detdata[self.det_data].dtype

            self._shape, self._wcs = enmap.read_map_geometry(self.area)

            self._recenter = None
            if self.center_at is not None:
                self._recenter = mm.parse_recentering(self.center_at)

            dtype_tod    = np.float32
            signal_cut   = mm.SignalCut(comm, dtype=dtype_tod)

            signal_map   = mm.SignalMap(self._shape, self._wcs, comm, comps=self.comps,
                               dtype=np.dtype(self.dtype_map), recenter=self._recenter, tiled=self.tiled)
            signals      = [signal_cut, signal_map]
            self._mapmaker = mm.MLMapmaker(signals, noise_model=self.Nmat, dtype=dtype_tod, verbose=self.verbose)
            # Store this to be able to output rhs and div later
            self._signal_map = signal_map

        for ob in data.obs:
            # Get the detectors we are using locally for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Get the sample rate from the data.  We also have nominal sample rates
            # from the noise model and also from the focalplane.
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )

            # Get the focalplane for this observation
            fp = ob.telescope.focalplane

            # Prepare data for the mapmaker.

            axdets = LabelAxis("dets", dets)

            nsample = int(ob.n_local_samples)
            ind = slice(None)
            ncut = nsample - fft.fft_len(nsample)
            if ncut != 0:
                if self.truncate_tod:
                    log.info_rank(
                        f"Truncating {ncut} / {nsample} samples ({100 * ncut / nsample:.3f}%) from "
                        f"{ob.name} for better FFT performance.",
                        comm=gcomm,
                    )
                    nsample -= ncut
                    ind = slice(nsample)
                else:
                    log.warning_rank(
                        f"{ob.name} length contains large prime factors.  "
                        F"FFT performance may be degrared.  Recommend "
                        f"truncating {ncut} / {nsample} samples ({100 * ncut / nsample:.3f}%).",
                        comm=gcomm,
                    )

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
            theta, phi, pa = toast.qarray.to_iso_angles(ob.shared[self.boresight][ind])

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
            axobs.wrap("timestamps", ob.shared[self.times][ind], axis_map=[(0, axsamps)])
            axobs.wrap(
                "signal",
                ob.detdata[self.det_data][dets, ind],
                axis_map=[(0, axdets), (1, axsamps)],
            )
            axobs.wrap("boresight", axbore)
            axobs.wrap("glitch_flags", ranges, axis_map=[(0, axdets), (1, axsamps)])
            axobs.wrap("weather", np.full(1, self.weather))
            axobs.wrap("site",    np.full(1, "so"))

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

            # Maybe load precomputed noise model
            if self.nmat_dir is None:
                nmat_dir  = os.path.join(self.outdir, "nmats")
            else:
                nmat_dir  = self.nmat_dir
            nmat_file = nmat_dir + "/nmat_%s.hdf" % ob.name
            there = os.path.isfile(nmat_file)
            if self.nmat_mode == "load" and not there:
                raise RuntimeError(
                    f"Nmat mode is 'load' but {nmat_file} does not exist."
                )
            if self.nmat_mode == "load" or (self.nmat_mode == "cache" and there):
                log.info_rank(f"Loading noise model from '{nmat_file}'", comm=gcomm)
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

            self._mapmaker.add_obs(
                ob.name, axobs, deslope=self.deslope, noise_model=nmat
            )
            del axobs

            # Maybe save the noise model we built (only if we actually built one rather than
            # reading one in)
            if self.nmat_mode in ["save", "cache"] and nmat is None:
                log.info_rank(f"Writing noise model to '{nmat_file}'", comm=gcomm)
                os.makedirs(nmat_dir, exist_ok=True)
                mm.write_nmat(nmat_file, self._mapmaker.data[-1].nmat)

            # Optionally delete the input detector data to save memory, if
            # the calling code knows that no additional operators will be
            # used afterwards.
            if self.purge_det_data:
                del ob.detdata[self.det_data]

        return

    @function_timer
    def _finalize(self, data, **kwargs):
        # After multiple calls to exec, the finalize step will solve for the map.
        log = Logger.get()
        timer = Timer()
        comm = data.comm.comm_world
        gcomm = data.comm.comm_group
        timer.start()

        self._mapmaker.prepare()

        if self.tiled:
            geo_work = self._mapmaker.signals[1].geo_work
            nactive = len(geo_work.active)
            ntile = np.prod(geo_work.shape[-2:])
            log.info_rank(f"{nactive} / {ntile} tiles active", comm=gcomm)

        log.info_rank(
            f"MLMapmaker finished prepare in",
            comm=comm,
            timer=timer,
        )

        prefix = os.path.join(self.out_dir, f"{self.name}_")

        # This will need to be modified for more general cases where we don't solve for
        # a sky map, or where we solve for multiple sky maps. The mapmaker itself supports it,
        # the problem is the direct access to the rhs, div and idiv members
        if self.write_rhs:
            fname = self._signal_map.write(prefix, "rhs", self._signal_map.rhs)
            log.info_rank(f"Wrote rhs to {fname}", comm=comm)

        if self.write_div:
            #self._signal_map.write(prefix, "div", self._signal_map.div)
            # FIXME : only writing the TT variance to avoid integer overflow in communication
            fname = self._signal_map.write(prefix, "div", self._signal_map.div[0, 0])
            log.info_rank(f"Wrote div to {fname}", comm=comm)

        if self.write_hits:
            fname = self._signal_map.write(prefix, "hits", self._signal_map.hits)
            log.info_rank(f"Wrote hits to {fname}", comm=comm)

        mmul = tilemap.map_mul if self.tiled else enmap.map_mul
        if self.write_bin:
            fname = self._signal_map.write(
                prefix, "bin", mmul(self._signal_map.idiv, self._signal_map.rhs)
            )
            log.info_rank(f"Wrote bin to {fname}", comm=comm)

        if comm is not None:
            comm.barrier()
        log.info_rank(f"MLMapmaker finished writing rhs, div, bin in", comm=comm, timer=timer)

        tstep = Timer()
        tstep.start()

        for step in self._mapmaker.solve(maxiter=self.maxiter, maxerr=self.maxerr):
            if self.write_iter_map < 1:
                dump = False
            else:
                dump = step.i % self.write_iter_map == 0
            dstr = ""
            if dump:
                dstr = "(write)"
            msg = f"CG step {step.i:4d} {step.err:15.7e} {dstr}"
            log.info_rank(f"MLMapmaker   {msg} ", comm=comm, timer=tstep)
            if dump:
                for signal, val in zip(self._mapmaker.signals, step.x):
                    if signal.output:
                        fname = signal.write(prefix, "map%04d" % step.i, val)
                        log.info_rank(f"Wrote signal to {fname}", comm=comm)

        log.info_rank(f"MLMapmaker finished solve in", comm=comm, timer=timer)

        for signal, val in zip(self._mapmaker.signals, step.x):
            if signal.output:
                fname = signal.write(prefix, "map", val)
                log.info_rank(f"Wrote {fname}", comm=comm)

        if comm is not None:
            comm.barrier()
        log.info_rank(f"MLMapmaker wrote map in", comm=comm, timer=timer)

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


# class NmatToast(mm.Nmat):
#     """Noise matrix class that uses a TOAST noise model.

#     This takes an existing TOAST noise model and uses it for a MLMapmaker compatible
#     noise matrix.

#     Args:
#         model (toast.Noise):  The toast noise model.
#         det_order (dict):  The mapping from detector order in the AxisManager
#             to name in the Noise object.

#     """
#     def __init__(self, model, n_sample, det_order):
#         self.model = model
#         self.det_order = det_order
#         self.n_sample = n_sample

#         # Compute the radix-2 FFT length to use
#         self.fftlen = 2
#         while self.fftlen <= self.n_sample:
#             self.fftlen *= 2
#         self.npsd = self.fftlen // 2 + 1

#         # Compute the time domain offset that centers our data within the
#         # buffer
#         self.padded_start = (self.fftlen - self.n_sample) // 2

#         # Compute the common frequency values
#         self.nyquist = model.freq(model.keys[0])[-1].to_value(u.Hz)
#         self.rate = 2 * self.nyquist
#         self.freqs = np.fft.rfftfreq(self.fftlen, 1 / self.rate))

#         # Interpolate the PSDs to desired spacing and store for later
#         # application.

#     def build(self, tod, **kwargs):
#         """Build method is a no-op, we do all set up in the constructor."""
#         return self

#     def apply(self, tod, inplace=False):
#         """Apply our noise filter to the TOD.

#         We use our pre-built Fourier domain kernels.

#         """
#         return tod
