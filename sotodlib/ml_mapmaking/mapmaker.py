import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-Y", "--yaml", type=str, default='config.yaml', help="Mapmaker config.yaml file")
cfg = yaml.safe_load(open('config.yaml','r'))
parser.add_argument('-Q', "--query", type=str, default=cfg['query'])
parser.add_argument("-f", "--freq", type=str, default=cfg['freq'])
parser.add_argument("-a", "--area", type=str, default=cfg['area'])
parser.add_argument("-o", "--odir", type=str, default=cfg['odir'])
parser.add_argument("-p", "--prefix", type=str, default=cfg['prefix'],
                    nargs="?")
parser.add_argument("-C", "--comps",   type=str, default="T")
parser.add_argument("-c", "--context", type=str, default=cfg['context'])
parser.add_argument("-n", "--ntod",    type=int, default=None)
parser.add_argument(      "--tods",    type=str, default=None)
parser.add_argument(      "--nset",    type=int, default=None)
parser.add_argument("-N", "--nmat",    type=str, default=cfg['nmat'])
parser.add_argument(      "--max-dets",type=int, default=None)
parser.add_argument("-S", "--site",    type=str, default=cfg['site'])
parser.add_argument("-v", "--verbose", action="count", default=0)
parser.add_argument("-q", "--quiet",   action="count", default=0)
parser.add_argument("-@", "--center-at", type=str, default=None)
parser.add_argument("-w", "--window",  type=float, default=0.0)
parser.add_argument("-i", "--inject",  type=str, default=None)
parser.add_argument(      "--nocal",   action="store_true", default=cfg['nocal'])
parser.add_argument(      "--nmat-dir", type=str, default="{odir}/nmats")
parser.add_argument(      "--nmat-mode", type=str, default=cfg['nmat-mode'], help="How to build the noise matrix. 'build': Always build from tod. 'cache': Use if available in nmat-dir, otherwise build and save. 'load': Load from nmat-dir, error if missing. 'save': Build from tod and save.")
parser.add_argument("-d", "--downsample", type=int, default=cfg['downsample'], help="Downsample TOD by this factor")
parser.add_argument(      "--maxiter",    type=int, default=cfg['maxiter'])
parser.add_argument("-T", "--tiled"  ,    type=int, default=cfg['tiled'])
parser.add_argument("-W", "--wafer"  ,   type=str, default=None, nargs='+', help="Detector wafer subset to map with")

args = parser.parse_args()
import numpy as np, sys, time, warnings, os, so3g
from sotodlib.core import Context, AxisManager, IndexAxis
from sotodlib.io import metadata   # PerDetectorHdf5 work-around
from sotodlib import tod_ops, mapmaking
from sotodlib.tod_ops import filters
from pixell import enmap, utils, fft, bunch, wcsutils, mpi
from enlib import bench, log
import yaml
import moby2.analysis.socompat

warnings.simplefilter('ignore')
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

#ids = context.obsdb.query(args.query)['obs_id']
ids = mapmaking.get_ids(args.query, context = context)

# restrict tod selection further. E.g. --tods [0], --tods[:1], --tods[::100], --tods[[0,1,5,10]], etc.
if args.tods:
    ids2 = eval("ids" + args.tods)
    if type(ids2) != type(ids): ids2 = np.array([ids2])
    ids = ids2
# This one is just a special case of the much more general one above
if args.ntod is not None:
    ids = ids[:args.ntod]

if len(ids) == 0:
    if comm.rank == 0:
        print("No tods found!")
    sys.exit(1)
L.info("Reading %d tods" % (len(ids)))

if args.inject:
    map_to_inject = enmap.read_map(args.inject).astype(dtype_map)

if   args.nmat == "uncorr": noise_model = mapmaking.NmatUncorr()
elif args.nmat == "corr":   noise_model = mapmaking.NmatDetvecs(verbose=verbose>1, downweight=[1e-4, 0.25, 0.50], window=args.window)
else: raise ValueError("Unrecognized noise model '%s'" % args.nmat)

signal_cut = mapmaking.SignalCut(comm, dtype=dtype_tod)
signal_map = mapmaking.SignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, recenter=recenter, tiled=args.tiled > 0)
signals    = [signal_cut, signal_map]
mapmaker   = mapmaking.MLMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)

#mapmaker = mapmaking.MLMapmaker(shape, wcs, comps=comps, noise_model=noise_model, dtype_tod=dtype_tod, dtype_map=dtype_map, comm=comm, recenter=recenter, verbose=verbose>0)

# First feed our mapmaker with data
nset_kept_tot = 0
for ind in range(comm.rank, len(ids), comm.size):
    # Detsets correspond to separate files, so treat them as separate TODs.
    obs_id = ids[ind]
    detsets = context.obsfiledb.get_detsets(obs_id)
    nset_kept = 0

    for detset in detsets:
        if args.nset is not None and nset_kept >= args.nset: continue
        name = "%s_%s" % (obs_id, detset)
        L.debug("Processing %s" % (name))

        # Cut out detector wafers we're not interested in, if args.wafer is specified
        if args.wafer is not None: 
            wafer_list = args.wafer
            dets_dict = {'dets:wafer_slot':wafer_list}
        else: dets_dict ={} 
        
        dets_dict['band'] = args.freq
        # Get the resolved list of detectors, to keep it below args.max_dets.
        meta = context.get_meta(obs_id=obs_id, dets=dets_dict)
        dets = meta['dets'].vals
        if args.max_dets is not None:
            meta.restrict('dets', meta['dets'].vals[:args.max_dets])
        if len(dets) == 0:
            L.debug("Skipped %s (no dets left)" % (name))
            continue

        with bench.mark("read_obs %s" % name):
            obs = context.get_obs(obs_id=obs_id, meta=meta)

        # Fix boresight
        mapmaking.fix_boresight_glitches(obs)
        # Get our sample rate. Would have been nice to have this available in the axisman
        srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])

        # Add site and weather, since they're not in obs yet
        obs.wrap("weather", np.full(1, "vacuum"))
        obs.wrap("site",    np.full(1, "so"))

        # Prepare our data. FFT-truncate for faster fft ops
        obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])

        # Desolope to make it periodic. This should be done *before*
        # dropping to single precision, to avoid unnecessary loss of precision due
        # to potential high offses in the raw tod.
        utils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)

        if "glitch_flags" not in obs:
            obs.wrap_new('glitch_flags', shape=('dets', 'samps'),
                    cls=so3g.proj.RangesMatrix.zeros)

        # Optionally skip all the calibration. Useful for sims.
        if not args.nocal:
            # Disqualify overly cut detectors
            good_dets = mapmaking.find_usable_detectors(obs)
            obs.restrict("dets", good_dets)
            if obs.dets.count == 0:
                L.debug("Skipped %s (all dets cut)" % (name))
                continue
            # Gapfill glitches. This function name isn't the clearest
            tod_ops.get_gap_fill(obs, flags=obs.glitch_flags, swap=True)
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
        # to inject it earlier, and then anti-calibrate it
        if args.inject:
            mapmaking.inject_map(obs, map_to_inject, recenter=recenter)
        utils.deslope(obs.signal, w=5, inplace=True)

        if args.downsample != 1:
            obs = mapmaking.downsample_obs(obs, args.downsample)

        # Maybe load precomputed noise model
        nmat_file = nmat_dir + "/nmat_%s.hdf" % name
        if args.nmat_mode == "load" or args.nmat_mode == "cache" and os.path.isfile(nmat_file):
            print("Reading noise model %s" % nmat_file)
            nmat = mapmaking.read_nmat(nmat_file)
        else: nmat = None

        # And add it to the mapmaker
        with bench.mark("add_obs %s" % name):
            mapmaker.add_obs(name, obs, noise_model=nmat)
        del obs
        nset_kept += 1

        # Maybe save the noise model we built (only if we actually built one rather than
        # reading one in)
        if args.nmat_mode in ["save", "cache"] and nmat is None:
            print("Writing noise model %s" % nmat_file)
            utils.mkdir(nmat_dir)
            mapmaking.write_nmat(nmat_file, mapmaker.data[-1].nmat)
    nset_kept_tot += nset_kept

nset_kept_tot = comm.allreduce(nset_kept_tot)
if nset_kept_tot == 0:
    if comm.rank == 0:
        L.info("All tods failed. Giving up")
    sys.exit(1)

L.info("Done building")

with bench.mark("prepare"):
    mapmaker.prepare()

L.info("Done preparing")

signal_map.write(prefix, "rhs", signal_map.rhs)
signal_map.write(prefix, "div", signal_map.div)
signal_map.write(prefix, "bin", enmap.map_mul(signal_map.idiv, signal_map.rhs))

L.info("Wrote rhs, div, bin")

t1 = time.time()
for step in mapmaker.solve(maxiter=args.maxiter):
    t2 = time.time()
    dump = step.i % 10 == 0
    L.info("CG step %4d %15.7e %8.3f %s" % (step.i, step.err, t2-t1, "" if not dump else "(write)"))
    if dump:
        for signal, val in zip(signals, step.x):
            if signal.output:
                signal.write(prefix, "map%04d" % step.i, val)
    t1 = time.time()

L.info("Done")
for signal, val in zip(signals, step.x):
    if signal.output:
        signal.write(prefix, "map", val)
comm.Barrier()