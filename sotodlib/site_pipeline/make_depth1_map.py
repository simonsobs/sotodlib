from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging
from sotodlib import tod_ops, coords, mapmaking
from sotodlib.core import Context, AxisManager, IndexAxis, FlagManager
from sotodlib.io import metadata   # PerDetectorHdf5 work-around
from sotodlib.tod_ops import filters, detrend_tod
from pixell import enmap, utils, fft, bunch, wcsutils, mpi, colors, memory
import yaml

defaults = {"query": "1",
            "odir": "./outputs",
            "comps": "T",
            "ntod": None,
            "tods": None,
            "nset": None,
            "site": 'so_lat',
            "nmat": "corr",
            "max_dets": None,
            "verbose": 0,
            "quiet": 0,
            "center_at": None,
            "window": 0.0,
            "nocal": True,
            "nmat_dir": "/nmats",
            "nmat_mode": "build",
            "downsample": 1,
            "maxiter": 100,
            "tiled": 1,
            "wafer": None,
            "freq": None,
            "tasks_per_group":1,
            "cont":False,
           }

def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to mapmaker config.yaml file")

    parser.add_argument("--query", type=str)
    parser.add_argument("--area", type=str, help="Path to FITS file describing the mapping geometry")
    parser.add_argument("--odir", type=str, help="Directory for saving output maps")
    parser.add_argument('-C', "--comps",   type=str, help="T,Q, and/or U")
    parser.add_argument("-c", "--context", type=str, help="Context containing TODs")
    parser.add_argument("-n", "--ntod",    type=int, help="Special case of `tods` above. Implemented as follows: [:ntod]")
    parser.add_argument(      "--tods",    type=str, help="Restrict TOD selections by index")
    parser.add_argument(      "--nset",    type=int, help="Number of detsets kept")
    parser.add_argument("-N", "--nmat",    type=str, help="'corr' or 'uncorr'")
    parser.add_argument(      "--max-dets",type=int, help="Maximum number of dets kept")
    parser.add_argument("-S", "--site",    type=str, help="Observatory site")
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument("-q", "--quiet",   action="count")
    parser.add_argument(      "--cont",  action="store_true", default=False, help="continue a run")
    parser.add_argument("-@", "--center-at", type=str)
    parser.add_argument("-w", "--window",  type=float)
    parser.add_argument(      "--nocal",   action="store_true", default=True, help="No relcal or abscal")
    parser.add_argument(      "--nmat-dir", type=str, help="Directory to where nmats are loaded from/saved to")
    parser.add_argument(      "--nmat-mode", type=str, help="How to build the noise matrix. 'build': Always build from tod. 'cache': Use if available in nmat-dir, otherwise build and save. 'load': Load from nmat-dir, error if missing. 'save': Build from tod and save.")
    parser.add_argument("-d", "--downsample", type=int, help="Downsample TOD by this factor")
    parser.add_argument(      "--maxiter",    type=int, help="Maximum number of iterative steps")
    parser.add_argument("-T", "--tiled"  ,    type=int)
    parser.add_argument("-W", "--wafer"  ,   type=str, nargs='+', help="Detector wafer subset to map with")
    parser.add_argument(      "--freq" ,  type=str, nargs='+', help="Frequency band to map with")
    parser.add_argument("-g", "--tasks-per-group", type=int, help="number of tasks per group. By default it is 1, but can be higher if you want more than one MPI job working on a depth-1 map, e.g. if you don't have enough memory for so many MPI jobs")
    return parser

def _get_config(config_file):
    return yaml.safe_load(open(config_file,'r'))

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, colors={'DEBUG':colors.reset,'INFO':colors.lgreen,'WARNING':colors.lbrown,'ERROR':colors.lred, 'CRITICAL':colors.lpurple}):
        logging.Formatter.__init__(self, msg)
        self.colors = colors
    def format(self, record):
        try:
            col = self.colors[record.levelname]
        except KeyError:
            col = colors.reset
        return col + logging.Formatter.format(self, record) + colors.reset

class LogInfoFilter(logging.Filter):
    def __init__(self, rank=0):
        self.rank = rank
        try:
            # Try to get actual time since task start if possible
            import os, psutil
            p = psutil.Process(os.getpid())
            self.t0 = p.create_time()
        except ImportError:
            # Otherwise measure from creation of this filter
            self.t0 = time.time()
    def filter(self, record):
        record.rank  = self.rank
        record.wtime = time.time()-self.t0
        record.wmins = record.wtime/60.
        record.whours= record.wmins/60.
        record.mem   = memory.current()/1024.**3
        record.resmem= memory.resident()/1024.**3
        record.memmax= memory.max()/1024.**3
        return record

def handle_empty(prefix, tag, comm, e):
    # This happens if we ended up with no valid tods for some reason
    if comm.rank == 0:
        L.info("%s Skipped: %s" % (tag, str(e)))
        utils.mkdir(os.path.dirname(prefix))
        with open(prefix + ".empty", "w") as ofile: ofile.write("\n")

def tele2equ(coords, ctime, detoffs=[0,0], site="so_sat1"):
    # Broadcast and flatten input arrays
    coords, ctime = utils.broadcast_arrays(coords, ctime, npre=(1,0))
    cflat = utils.to_Nd(coords, 2, axis=-1)
    tflat = utils.to_Nd(ctime,  1, axis=-1)
    dflat, dshape = utils.to_Nd(detoffs, 2, axis=-1, return_inverse=True)
    nsamp, ndet = cflat.shape[1], dflat.shape[1]
    assert cflat.shape[1:] == tflat.shape, "tele2equ coords and ctime have incompatible shapes %s vs %s" % (str(coords.shape), str(ctime.shape))
    # Set up the transform itself
    sight  = so3g.proj.CelestialSightLine.az_el(tflat, cflat[0], cflat[1],
            roll=cflat[2] if len(cflat) > 2 else 0, site=site, weather="toco")
    # To support other coordiante systems I would add
    # if rot is not None: sight.Q = rot * sight.Q
    dummy  = np.arange(ndet)
    fp     = so3g.proj.FocalPlane.from_xieta(dummy, dflat[0], dflat[1],
            dflat[2] if len(dflat) > 2 else 0)
    asm    = so3g.proj.Assembly.attach(sight, fp)
    proj   = so3g.proj.Projectionist()
    res    = np.zeros((ndet,nsamp,4))
    # And actually perform it
    proj.get_coords(asm, output=res)
    # Finally unflatten
    res    = res.reshape(dshape[1:]+coords.shape[1:]+(4,))
    return res

def find_scan_profile(context, my_tods, my_infos, comm=mpi.COMM_WORLD, npoint=100):
    # Pre-allocate empty profile since other tasks need a receive buffer
    profile = np.zeros([2,npoint])
    # Who has the first valid tod?
    first   = np.where(comm.allgather([len(my_tods)]))[0][0]
    if comm.rank == first:
        tod, info = my_tods[0], my_infos[0]
        # Find our array's central pointing offset. 
        fp   = tod.focal_plane
        xi0  = np.mean(utils.minmax(fp.xi))
        eta0 = np.mean(utils.minmax(fp.eta))
        # Build a boresight corresponding to a single az sweep at constant time
        azs  = info.az_center + np.linspace(-info.az_throw/2, info.az_throw/2, npoint)
        els  = np.full(npoint, info.el_center)
        profile[:] = tele2equ(np.array([azs, els])*utils.degree, info.timestamp, detoffs=[xi0, eta0]).T[1::-1] # dec,ra
    comm.Bcast(profile, root=first)
    return profile

def find_footprint(context, tods, ref_wcs, comm=mpi.COMM_WORLD, return_pixboxes=False, pad=1):
    # Measure the pixel bounds of each observation relative to our
    # reference wcs
    pixboxes = []
    for tod in tods:
        my_shape, my_wcs = coords.get_footprint(tod, ref_wcs)
        my_pixbox = enmap.pixbox_of(ref_wcs, my_shape, my_wcs)
        pixboxes.append(my_pixbox)
    pixboxes = utils.allgatherv(pixboxes, comm)
    if len(pixboxes) == 0: raise DataMissing("No usable obs to estimate footprint from")
    # Handle sky wrapping. This assumes cylindrical coordinates with sky-wrapping
    # in the x-direction, and that there's an integer number of pixels around the sky.
    # Could be done more generally, but would be much more involved, and this should be
    # good enough
    nphi     = utils.nint(np.abs(360/ref_wcs.wcs.cdelt[0]))
    widths   = pixboxes[:,1,0]-pixboxes[:,0,0]
    pixboxes[:,0,0] = utils.rewind(pixboxes[:,0,0], ref=pixboxes[0,0,0], period=nphi)
    pixboxes[:,1,0] = pixboxes[:,0,0] + widths
    # It's now safe to find the total pixel bounding box
    union_pixbox = np.array([np.min(pixboxes[:,0],0)-pad,np.max(pixboxes[:,1],0)+pad])
    # Use this to construct the output geometry
    shape = union_pixbox[1]-union_pixbox[0]
    wcs   = ref_wcs.deepcopy()
    wcs.wcs.crpix -= union_pixbox[0,::-1]
    if return_pixboxes: return shape, wcs, pixboxes
    else: return shape, wcs

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD, no_signal=False, site='so'):
    my_tods = []
    my_inds = []
    if inds is None: inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            tod = context.get_obs(obs_id, dets={"wafer_slot":detset, "band":band}, no_signal=no_signal)
            tod = calibrate_obs(tod, site=site)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError: continue
    return my_tods, my_inds

def calibrate_obs(obs, site='so', dtype_tod=np.float32, nocal=True):
    # The following stuff is very redundant with the normal mapmaker,
    # and should probably be factorized out
    mapmaking.fix_boresight_glitches(obs)
    srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])
    # Add site and weather, since they're not in obs yet
    obs.wrap("weather", np.full(1, "toco"))
    if "site" not in obs:
        obs.wrap("site",    np.full(1, site))
    # Prepare our data. FFT-truncate for faster fft ops
    obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])

    # add dummy glitch flags
    if 'flags' not in obs._fields:
        obs.wrap('flags', FlagManager.for_tod(obs))
    if "glitch_flags" not in obs.flags:
        obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape),[(0,'dets'),(1,'samps')])

    try:
        obs.signal
        has_signal = True
    except KeyError:
        has_signal = False
    
    #if obs.signal is not None:
    if has_signal:
        detrend_tod(obs, method='linear')
        utils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)
    
    #if (not nocal) and (obs.signal is not None):
    if (not nocal) and (has_signal):
        # Disqualify overly cut detectors
        good_dets = mapmaking.find_usable_detectors(obs)
        obs.restrict("dets", good_dets)
        if len(good_dets) > 0:
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
    return obs

def make_depth1_map(context, obslist, shape, wcs, noise_model, L, comps="TQU", t0=0, dtype_tod=np.float32, dtype_map=np.float64, comm=mpi.COMM_WORLD, tag="", niter=100, site='so', tiled=0, verbose=0, downsample=1):
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    # Set up our mapmaking equation
    signal_cut = mapmaking.SignalCut(comm, dtype=dtype_tod)
    signal_map = mapmaking.SignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=tiled > 0, ofmt="")
    signals    = [signal_cut, signal_map]
    mapmaker   = mapmaking.MLMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)
    if comm.rank == 0: L.info(pre + "Building RHS")
    time_rhs   = signal_map.rhs*0
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time
        obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "band":band})
        obs = calibrate_obs(obs, site=site)
        #bools, nod_indices, obs = find_el_nods(obs)

        if downsample != 1:
            obs = mapmaking.downsample_obs(obs, downsample)
        if obs.dets.count == 0:
            print('Skipping ', (obs_id, detset, band))
            continue
        # And add it to the mapmaker
        try:
            mapmaker.add_obs(name, obs)
        except KeyError:
            print('rank = %i' % comm.rank, (obs_id, detset, band) )
        # Also build the RHS for the per-pixel timestamp. First
        # make a white noise weighted timestamp per sample timestream
        Nt  = np.zeros_like(obs.signal, dtype=dtype_tod)
        Nt += obs.timestamps - t0
        Nt *= mapmaker.data[-1].nmat.ivar[:,None]
        signal_cut.data[name].pcut.clear(Nt)
        # Bin into pixels
        pmap = signal_map.data[name].pmap
        obs_time_rhs = pmap.zeros()
        pmap.to_map(dest=obs_time_rhs, signal=Nt)
        # Accumulate into output array
        time_rhs = time_rhs.insert(obs_time_rhs, op=np.ndarray.__iadd__)
        del obs, pmap, Nt, obs_time_rhs
        nobs_kept += 1

    nobs_kept = comm.allreduce(nobs_kept)
    if nobs_kept == 0: raise DataMissing("All data cut")
    mapmaker.prepare()
    # mapmaker doesn't know about time_rhs, so handle it manually
    if signal_map.tiled: time_rhs = tilemap.redistribute(time_rhs, comm)
    else:                time_rhs = utils.allreduce     (time_rhs, comm)    
    if signal_map.tiled: bin = tilemap.map_mul(signal_map.idiv,signal_map.rhs)
    else: bin = enmap.map_mul(signal_map.idiv,signal_map.rhs)
    if comm.rank == 0: L.info(pre + "Solving")
    t1 = time.time()
    for step in mapmaker.solve(maxiter=niter):
        t2 = time.time()
        if comm.rank == 0:
            L.info("%s CG step %5d %15.7e %6.1f %6.3f" % (tag, step.i, step.err, (t2-t1), (t2-t1)/nobs_kept))
        t1 = time.time()
    map  = step.x[1]
    ivar = signal_map.div[0,0]
    with utils.nowarn(): tmap = utils.remove_nan(time_rhs / ivar)
    return bunch.Bunch(map=map, ivar=ivar, tmap=tmap, signal=signal_map, t0=t0, bin=bin)

def write_depth1_map(prefix, data, dtype = np.float32 ):
    data.signal.write(prefix, "map",  data.map.astype(dtype))
    data.signal.write(prefix, "ivar", data.ivar.astype(dtype))
    data.signal.write(prefix, "time", data.tmap.astype(dtype))
    data.signal.write(prefix, "rhs", data.signal.rhs.astype(dtype))
    data.signal.write(prefix, "bin", data.bin.astype(dtype))

def write_depth1_info(oname, info):
    utils.mkdir(os.path.dirname(oname))
    bunch.write(oname, info)

def main(config_file=None, defaults=defaults, **args):
    
    cfg = dict(defaults)
    
    # Update the default dict with values provided from a config.yaml file
    if config_file is not None:
        cfg_from_file = _get_config(config_file)
        cfg.update({k: v for k, v in cfg_from_file.items() if v is not None})
    else:
        print("No config file provided, assuming default values") 

    # Merge flags from config file and defaults with any passed through CLI
    cfg.update({k: v for k, v in args.items() if v is not None})
    
    # Certain fields are required. Check if they are all supplied here
    required_fields = ['area','context']
    for req in required_fields:
        if req not in cfg.keys():
            raise KeyError("{} is a required argument. Please supply it in a config file or via the command line".format(req))
    args = cfg

    warnings.simplefilter('ignore')
    # Set up our communicators
    comm       = mpi.COMM_WORLD
    comm_intra = comm.Split(comm.rank // args['tasks_per_group'])
    comm_inter = comm.Split(comm.rank  % args['tasks_per_group'])
    
    SITE    = args['site']
    verbose = args['verbose'] - args['quiet']
    shape, wcs = enmap.read_map_geometry(args['area'])
    # Reconstruct that wcs in case default fields have changed; otherwise
    # we risk adding information in MPI due to reconstruction, and that
    # can cause is_compatible failures.
    wcs = wcsutils.WCS(wcs.to_header())
    # Set shape to None to allow map to fit these TODs exactly.
    #shape = None
    
    comps = args['comps']
    ncomp = len(comps)
    dtype_tod = np.float32
    dtype_map = np.float64
    meta_only  = False
    nmat_dir = os.path.join(args['odir'],args['nmat_dir'])
    utils.mkdir(args['odir'])

    # Set up logging.
    L   = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter( "%(rank)3d " + "%3d %3d" % (comm_inter.rank, comm.rank) + " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter(comm_intra.rank))
    L.addHandler(ch)

    recenter = None
    if args['center_at']:
        recenter = mapmaking.parse_recentering(args['center_at'])

    with mapmaking.mark('context'):
        context = Context(args['context'])

    if   args['nmat'] == "uncorr": noise_model = mapmaking.NmatUncorr()
    elif args['nmat'] == "corr":   noise_model = mapmaking.NmatDetvecs(verbose=verbose>1, downweight=[1e-4, 0.25, 0.50], window=args['window'])
    else: raise ValueError("Unrecognized noise model '%s'" % args['nmat'])

    obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(context, args['query'], mode='depth_1', nset=args['nset'], ntod=args['ntod'], tods=args['tods'], freq=args['freq'])
    
    for oi in range(comm_inter.rank, len(obskeys), comm_inter.size):
        pid, detset, band = obskeys[oi]
        obslist = obslists[obskeys[oi]]
        t       = utils.floor(periods[pid,0])
        t5      = ("%05d" % t)[:5]
        prefix  = "%s/%s/depth1_%010d_%s_%s" % (args['odir'], t5, t, detset, band)
        tag     = "%5d/%d" % (oi+1, len(obskeys))
        utils.mkdir(os.path.dirname(prefix))
        meta_done = os.path.isfile(prefix + "_info.hdf")
        maps_done = os.path.isfile(prefix + ".empty") or (
            os.path.isfile(prefix + "_time.fits") and
            os.path.isfile(prefix + "_map.fits") and
            os.path.isfile(prefix + "_ivar.fits"))
        if args['cont'] and meta_done and (maps_done or meta_only): continue
        if comm_intra.rank == 0:
            L.info("%s Proc period %4d dset %s:%s @%.0f dur %5.2f h with %2d obs" % (tag, pid, detset, band, t, (periods[pid,1]-periods[pid,0])/3600, len(obslist)))
        try:
            # 1. read in the metadata and use it to determine which tods are
            #    good and estimate how costly each is
            my_tods, my_inds = read_tods(context, obslist, comm=comm_intra, no_signal=True, site=SITE)
            my_costs  = np.array([tod.samps.count*len(mapmaking.find_usable_detectors(tod)) for tod in my_tods])
            # 2. prune tods that have no valid detectors
            valid     = np.where(my_costs>0)[0]
            my_tods, my_inds, my_costs = [[a[vi] for vi in valid] for a in [my_tods, my_inds, my_costs]]
            all_inds  = utils.allgatherv(my_inds,     comm_intra)
            all_costs = utils.allgatherv(my_costs,    comm_intra)
            if len(all_inds)  == 0: raise DataMissing("No valid tods")
            if sum(all_costs) == 0: raise DataMissing("No valid detectors in any tods")
            # 2. estimate the scan profile and footprint. The scan profile can be done
            #    with a single task, but that task might not be the first one, so just
            #    make it mpi-aware like the footprint stuff
            my_infos = [obs_infos[obslist[ind][3]] for ind in my_inds]
            profile  = find_scan_profile(context, my_tods, my_infos, comm=comm_intra)
            subshape, subwcs = find_footprint(context, my_tods, wcs, comm=comm_intra)
            # 3. Write out the depth1 metadata
            d1info = bunch.Bunch(profile=profile, pid=pid, detset=detset.encode(), band=band.encode(),
                    period=periods[pid], ids=np.char.encode([obslist[ind][0] for ind in all_inds]),
                    box=enmap.corners(subshape, subwcs), t=t)
            if comm_intra.rank == 0:
                write_depth1_info(prefix + "_info.hdf", d1info)
        except DataMissing as e:
            # This happens if we ended up with no valid tods for some reason
            handle_empty(prefix, tag, comm_intra, e)
            continue
        # 4. redistribute the valid tasks. Tasks with nothing to do don't continue
        # past here.
        my_inds   = all_inds[utils.equal_split(all_costs, comm_intra.size)[comm_intra.rank]]
        comm_good = comm_intra.Split(len(my_inds) > 0)
        if len(my_inds) == 0: continue
        try:
            # 5. make the maps
            mapdata = make_depth1_map(context, [obslist[ind] for ind in my_inds],
                    subshape, subwcs, noise_model, L, comps=comps, t0=t, comm=comm_good, tag=tag,
                    niter=args['maxiter'], dtype_map=dtype_map, dtype_tod=dtype_tod, site=SITE, tiled=args['tiled']>0, verbose=verbose>0, downsample=args['downsample'] )
            # 6. write them
            write_depth1_map(prefix, mapdata, dtype=dtype_tod)
        except DataMissing as e:
            handle_empty(prefix, tag, comm_good, e)
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
