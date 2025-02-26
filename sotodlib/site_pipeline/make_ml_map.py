from sotodlib.site_pipeline import util
def get_parser():
    parser = util.ArgumentParser()
    # Remember: util.ArgumentParser adds an implicit --config-file argument
    # If this argument is given, it points to a yaml-file that will be used
    # to override the defaults given below. The actual arguments passed
    # override those in turn. Even required arguments like "query" become
    # optional if specified in the yaml file
    parser.add_argument("query")
    parser.add_argument("area")
    parser.add_argument("odir")
    parser.add_argument("prefix", nargs="?")
    parser.add_argument(      "--comps",   type=str, default="T",  help="List of components to solve for. T, QU or TQU, but only TQU is consistent with the actual data")
    parser.add_argument("-W", "--wafers",  type=str, default=None, help="Detector wafer subsets to map with. ,-sep")
    parser.add_argument("-B", "--bands",   type=str, default=None, help="Bandpasses to map. ,-sep")
    parser.add_argument("-C", "--context", type=str, default="/mnt/so1/shared/todsims/pipe-s0001/v4/context.yaml")
    parser.add_argument(      "--tods",    type=str, default=None, help="Arbitrary slice to apply to the list of tods to analyse")
    parser.add_argument("-n", "--ntod",    type=int, default=None, help="Keep at most this many tods")
    parser.add_argument("-N", "--nmat",    type=str, default="corr", help="Noise model to use. corr or uncorr")
    parser.add_argument(      "--max-dets",type=int, default=None,   help="Keep at most this many detectors")
    parser.add_argument("-S", "--site",    type=str, default="so_lat")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet",   action="count", default=0)
    parser.add_argument("-@", "--center-at", type=str, default=None)
    parser.add_argument("-w", "--window",  type=float, default=0.0)
    parser.add_argument("-i", "--inject",  type=str,   default=None, help="Path to map to inject. Equatorial coordinates")
    parser.add_argument(      "--nocal",   action="store_true", help="Disable calibration. Useful for sims")
    parser.add_argument(      "--nmat-dir",  type=str, default="{odir}/nmats")
    parser.add_argument(      "--nmat-mode", type=str, default="build", help="How to build the noise matrix. 'build': Always build from tod. 'cache': Use if available in nmat-dir, otherwise build and save. 'load': Load from nmat-dir, error if missing. 'save': Build from tod and save.")
    parser.add_argument("-d", "--downsample", type=str, default="1",  help="Downsample TOD by this factor. ,-sep")
    parser.add_argument(      "--maxiter",    type=str, default="500",help="Max number of CG steps per pass. ,-sep")
    parser.add_argument(      "--interpol",   type=str, default="nearest", help="Pmat interpol per pass. ,-sep")
    parser.add_argument("-T", "--tiled"  ,   type=int, default=1, help="0: untiled maps. Nonzero: tiled maps")
    parser.add_argument(      "--srcsamp",   type=str, default=None, help="path to mask file where True regions indicate where bright object mitigation should be applied. Mask is in equatorial coordinates. Not tiled, so should be low-res to not waste memory.")
    return parser

if __name__ == "__main__":
    # We do this all the way up here so we can report argument
    # errors right away, instead of having to potentially wait for
    # many seconds for heavy modules to import. This relies on
    # sotodlib.site_pipeline_util being light-weight. If it stops
    # being so, then ArgumentParser should be split out into a
    # separate module
    parser = get_parser()
    args   = parser.parse_args()

import numpy as np, sys, time, warnings, os, so3g
from sotodlib.core import Context, AxisManager, IndexAxis
from sotodlib.io import metadata   # PerDetectorHdf5 work-around
from sotodlib import tod_ops, mapmaking, core
from sotodlib.tod_ops import filters
from pixell import enmap, utils, fft, bunch, wcsutils, mpi, bench
from enlib import log

try: import moby2.analysis.socompat
except ImportError: warnings.warn("Can't import moby2.analysis.socompat. ACT data input probably won't work")
class DataMissing(Exception): pass

def main(**args):
    warnings.simplefilter('ignore')
    args    = bunch.Bunch(**args)

    SITE    = args.site
    verbose = args.verbose - args.quiet
    comm    = mpi.COMM_WORLD
    shape, wcs = enmap.read_map_geometry(args.area)

    # Reconstruct that wcs in case default fields have changed; otherwise
    # we risk adding information in MPI due to reconstruction, and that
    # can cause is_compatible failures.
    wcs = wcsutils.WCS(wcs.to_header())
    # Set shape to None to allow map to fit these TODs exactly.
    #shape = None

    comps = args.comps
    ncomp = len(comps)
    dtype_tod = np.float32
    dtype_map = np.float64
    nmat_dir  = args.nmat_dir.format(odir=args.odir)
    prefix= args.odir + "/"
    if args.prefix: prefix += args.prefix + "_"
    utils.mkdir(args.odir)
    L = log.init(level=log.DEBUG, rank=comm.rank)

    recenter = None
    if args.center_at:
            recenter = mapmaking.parse_recentering(args.center_at)

    with bench.mark('context'):
            context = Context(args.context)

    wafers  = args.wafers.split(",") if args.wafers else None
    bands   = args.bands .split(",") if args.bands  else None
    sub_ids = mapmaking.get_subids(args.query, context=context)
    sub_ids = mapmaking.filter_subids(sub_ids, wafers=wafers, bands=bands)

    # restrict tod selection further. E.g. --tods [0], --tods[:1], --tods[::100], --tods[[0,1,5,10]], etc.
    if args.tods:
            sub_ids = eval("sub_ids" + args.tods)
    if args.ntod is not None:
            sub_ids = sub_ids[:args.ntod]

    if len(sub_ids) == 0:
            if comm.rank == 0:
                    print("No tods found!")
            sys.exit(1)
    L.info("Found %d tods" % (len(sub_ids)))

    if args.inject:
            map_to_inject = enmap.read_map(args.inject).astype(dtype_map)

    if args.srcsamp:
            srcsamp_mask  = enmap.read_map(args.srcsamp)

    passes = mapmaking.setup_passes(downsample=args.downsample, maxiter=args.maxiter, interpol=args.interpol)
    for ipass, passinfo in enumerate(passes):
            L.info("Starting pass %d/%d maxit %d down %d interp %s" % (ipass+1, len(passes), passinfo.maxiter, passinfo.downsample, passinfo.interpol))
            pass_prefix = prefix + "pass%d_" % (ipass+1)
            # Multipass mapmaking
            # Will do multiple passes over the data, using the previous pass both as
            # the initial condition and to temporarily subtract when estimating the noise
            # model. Typically the different passes will differ by sample rate. This means
            # that e.g. SignalCut will be different for each pass, and the degrees of freedom
            # will need to be translated between them. Rough sketch:
            # 1. each pass defines new signals and new mapmaker
            # 2. loop through tods and preprare them as normal
            # 3. just before mapmaker.add_obs, translate the old solution and evaluate it to a tod for
            #    this observation. Pass this as an optional parameter to add_obs, since that's where the
            #    noise model is built. In add_obs, this is subtracted from the tod before the noise model is
            #    built, and then added back again.
            # 4. Separately, build up the transated degrees of freedom. How can we do this without
            #    hardcoding stuff for specific signals? Some signals have per-tod degrees of freedom,
            #    others don't. Each signal should probably have an x = signal.translate(signal_old, x_old) method.
            #    SignalCut already knows about the tod size and cut ranges. Then the mapmaker can also have
            #    an x = mapmaker.translate(mapmaker_old, x_old) method, which assumes a compatible mapmaker,
            #    and basically does
            #      xvals = []
            #      for signal, signal_old, xvals_old in zip(mapmaker.signals, mapmaker_old.signals, mapmaker_old.dof.unzip(x_old)):
            #         xvals.append(signal.translate(signal_old, xvals))
            #      return mapmaker.dof.zip(*xvals)
            #    Except we can't do this before we've added all the tods, because thew new mapmaker's dofs won't have
            #    been set up yet! Well, just do it afterwards, then. That's fine for #4, but it means we can't
            #    use this to implement #3.
            #
            # I need a way to translate and evaluate a signal for a single tod, before the whole signal has
            # been constructed, or this tod has even been added to it (since the adding happens after
            # the noise model is built). Probably best to just make a separate function for this, which doesn't
            # need to know about the new degrees of freedom, which haven't been finalized at this point. This
            # would end up doing some redundant operations, e.g. defining the map pointing matrix twice, but
            # I don't see a cleaner design than this.
            # This could have an interface like tod = signal.transeval(signal_old, id, obs, xval), or it could be
            # split into two parts signal.translate_single and signal.forward_single). But I don't think those
            # building blocks would be very reusable, and the full thing is more general.
            if   args.nmat == "uncorr": noise_model = mapmaking.NmatUncorr()
            elif args.nmat == "corr":   noise_model = mapmaking.NmatDetvecs(verbose=verbose>1, downweight=[1e-4, 0.25, 0.50], window=args.window)
            else: raise ValueError("Unrecognized noise model '%s'" % args.nmat)

            signal_cut = mapmaking.SignalCut(comm, dtype=dtype_tod)
            signal_map = mapmaking.SignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, recenter=recenter, tiled=args.tiled>0, interpol=args.interpol)
            signals    = [signal_cut, signal_map]
            if args.srcsamp:
                    signal_srcsamp = mapmaking.SignalSrcsamp(comm, srcsamp_mask, dtype=dtype_tod)
                    signals.append(signal_srcsamp)
            mapmaker   = mapmaking.MLMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)

            nkept = 0
            # TODO: Fix the task distribution. The current one doesn't care which mpi
            # task gets which tods, which sabotages the pixel-saving effects of tiled maps!
            # To be able to distribute the tods sensibly, we need a rough estimate of where
            # on the sky each tod is. We should be able to get this using the central
            # ctime, az and el for each tod.
            for ind in range(comm.rank, len(sub_ids), comm.size):
                    # Detsets correspond to separate files, so treat them as separate TODs.
                    sub_id = sub_ids[ind]
                    obs_id, wafer, band = sub_id.split(":")
                    name = sub_id.replace(":", "_")
                    L.debug("Processing %s" % sub_id)
                    try:
                            meta = context.get_meta(sub_id)
                            # Optionally restrict to maximum number of detectors. This is mainly
                            # useful for doing fast debug runs. Before doing this we make sure to
                            # sort the detctor list so we chop off a deterministic subset of detectors.
                            meta.restrict("dets", np.sort(meta.dets.vals))
                            my_dets = meta['dets'].vals
                            if args.max_dets is not None:
                                    meta.restrict('dets', meta['dets'].vals[:args.max_dets])
                            if len(my_dets) == 0: raise DataMissing("no dets left")
                            # Actually read the data
                            with bench.mark("read_obs %s" % sub_id):
                                    obs = context.get_obs(sub_id, meta=meta)

                            # Fix boresight
                            mapmaking.fix_boresight_glitches(obs)
                            # Get our sample rate. Would have been nice to have this available in the axisman
                            srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])

                            # Add site and weather, since they're not in obs yet
                            obs.wrap("weather", np.full(1, "typical"))
                            obs.wrap("site",    np.full(1, "so"))

                            # Prepare our data. FFT-truncate for faster fft ops
                            obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])

                            # Desolope to make it periodic. This should be done *before*
                            # dropping to single precision, to avoid unnecessary loss of precision due
                            # to potential high offses in the raw tod.
                            utils.deslope(obs.signal, w=5, inplace=True)
                            obs.signal = obs.signal.astype(dtype_tod)

                            if "flags" not in obs:
                                    obs.wrap("flags", core.AxisManager(obs.dets, obs.samps))

                            if "glitch_flags" not in obs.flags:
                                    if "glitch_flags" in obs:
                                            obs.flags.wrap("glitch_flags", obs.glitch_flags, axis_map=((0,obs.dets),(1,obs.samps)))
                                    else:
                                            obs.flags.wrap_new('glitch_flags', shape=('dets', 'samps'),
                                                            cls=so3g.proj.RangesMatrix.zeros)

                            # Optionally skip all the calibration. Useful for sims.
                            if not args.nocal:
                                    # Disqualify overly cut detectors
                                    good_dets = mapmaking.find_usable_detectors(obs)
                                    obs.restrict("dets", good_dets)
                                    if obs.dets.count == 0:
                                            L.debug("Skipped %s (all dets cut)" % (sub_id))
                                            continue
                                    # Gapfill glitches. This function name isn't the clearest
                                    tod_ops.get_gap_fill(obs, flags=obs.flags.glitch_flags, swap=True)
                                    # Gain calibration
                                    gain  = 1
                                    for gtype in ["relcal","abscal"]:
                                            gain *= obs[gtype][:,None]
                                    obs.signal *= gain
                                    # Fourier-space calibration
                                    fsig  = fft.rfft(obs.signal)
                                    freq  = fft.rfftfreq(obs.samps.count, 1/srate)
                                    # iir filter
                                    iir_filter  = filters.iir_filter()(freq, obs)
                                    fsig       /= iir_filter
                                    gain       /= iir_filter[0].real # keep track of total gain for our record
                                    fsig       /= filters.timeconst_filter(None)(freq, obs)
                                    fft.irfft(fsig, obs.signal, normalize=True)
                                    del fsig

                                    # Apply pointing correction.
                                    #obs.focal_plane.xi    += obs.boresight_offset.xi
                                    #obs.focal_plane.eta   += obs.boresight_offset.eta
                                    #obs.focal_plane.gamma += obs.boresight_offset.gamma
                                    obs.focal_plane.xi    += obs.boresight_offset.dx
                                    obs.focal_plane.eta   += obs.boresight_offset.dy
                                    obs.focal_plane.gamma += obs.boresight_offset.gamma

                            # Injecting at this point makes us insensitive to any bias introduced
                            # in the earlier steps (mainly from gapfilling). The alternative is
                            # to inject it earlier, and then anti-calibrate it.
                            # Might want to make the interpol used here separate from the one in
                            # the main mapmaking.
                            if args.inject:
                                    mapmaking.inject_map(obs, map_to_inject, recenter=recenter, interpol=args.interpol)
                            utils.deslope(obs.signal, w=5, inplace=True)

                            if args.downsample != 1:
                                    obs = mapmaking.downsample_obs(obs, passinfo.downsample)

                            # Maybe load precomputed noise model.
                            # FIXME: How to handle multipass here?
                            nmat_file = nmat_dir + "/nmat_%s.hdf" % name
                            if args.nmat_mode == "load" or args.nmat_mode == "cache" and os.path.isfile(nmat_file):
                                    print("Reading noise model %s" % nmat_file)
                                    nmat = mapmaking.read_nmat(nmat_file)
                            else: nmat = None

                            # And add it to the mapmaker
                            with bench.mark("add_obs %s" % sub_id):
                                    if ipass > 0:
                                            # Evaluate the final model of the previous pass' mapmaker
                                            # for this observation.
                                            signal_estimate = eval_prev.evaluate(mapmaker_prev.data[len(mapmaker.data)])
                                            # Resample this to the current downsampling level
                                            signal_estimate = mapmaking.resample.resample_fft_simple(
                                                    signal_estimate, obs.samps.count)
                                    else: signal_estimate = None
                                    mapmaker.add_obs(sub_id, obs, noise_model=nmat, signal_estimate=signal_estimate)
                                    del signal_estimate
                            del obs
                            nkept += 1

                            # Maybe save the noise model we built (only if we actually built one rather than
                            # reading one in)
                            if args.nmat_mode in ["save", "cache"] and nmat is None:
                                    print("Writing noise model %s" % nmat_file)
                                    utils.mkdir(nmat_dir)
                                    mapmaking.write_nmat(nmat_file, mapmaker.data[-1].nmat)
                    except (DataMissing,IndexError,ValueError) as e:
                            L.debug("Skipped %s (%s)" % (sub_id, str(e)))
                            continue

            nkept = comm.allreduce(nkept)
            if nkept == 0:
                    if comm.rank == 0:
                            L.info("All tods failed. Giving up")
                    sys.exit(1)

            L.info("Done building")

            with bench.mark("prepare"):
                    mapmaker.prepare()

            L.info("Done preparing")

            signal_map.write(pass_prefix, "rhs", signal_map.rhs)
            signal_map.write(pass_prefix, "div", signal_map.div)
            signal_map.write(pass_prefix, "bin", enmap.map_mul(signal_map.idiv, signal_map.rhs))

            L.info("Wrote rhs, div, bin")

            # Set up initial condition
            x0 = None if ipass == 0 else mapmaker.translate(mapmaker_prev, eval_prev.x_zip)

            t1 = time.time()
            for step in mapmaker.solve(maxiter=passinfo.maxiter, x0=x0):
                    t2 = time.time()
                    dump = step.i % 10 == 0
                    L.info("CG step %4d %15.7e %8.3f %s" % (step.i, step.err, t2-t1, "" if not dump else "(write)"))
                    if dump:
                            for signal, val in zip(signals, step.x):
                                    if signal.output:
                                            signal.write(pass_prefix, "map%04d" % step.i, val)
                    t1 = time.time()

            L.info("Done")
            for signal, val in zip(signals, step.x):
                    if signal.output:
                            signal.write(pass_prefix, "map", val)
            comm.Barrier()

            mapmaker_prev = mapmaker
            eval_prev     = mapmaker.evaluator(step.x_zip)

if __name__ == "__main__":
    main(**vars(args))
