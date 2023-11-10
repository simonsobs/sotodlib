from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging
from sotodlib.core import Context,  metadata as metadata_core
from sotodlib.io import metadata   # PerDetectorHdf5 work-around
from sotodlib import tod_ops, coords, mapmaking
from sotodlib.tod_ops import filters
from sotodlib.hwp import hwp
from pixell import enmap, utils, fft, bunch, wcsutils, tilemap, colors, memory, mpi
from scipy import ndimage

from . import util

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("context", help='context file')    
    parser.add_argument("query", help='query, can be a file (list of obs_id) or selection string')
    parser.add_argument("area", help='wcs geometry')
    parser.add_argument("odir", help='output directory')
    parser.add_argument("--mode", type=str, default='per_obs')
    parser.add_argument("--comps",   type=str, default="TQU")
    
    # detector position splits
    parser.add_argument("--det-in-out", action="store_true")
    parser.add_argument("--det-left-right", action="store_true")
    parser.add_argument("--det-upper-lower", action="store_true")
    
    parser.add_argument("--ntod",    type=int, default=None)
    parser.add_argument("--tods",    type=str, default=None)
    parser.add_argument("--nset",    type=int, default=None)
    parser.add_argument("--max-dets",type=int, default=None)
    parser.add_argument("--fixed-time", type=int, default=None)
    parser.add_argument("--mindur", type=int, default=None)
    parser.add_argument("--tasks-per-group", type=int,   default=1)
    parser.add_argument("--site",    type=str, default="so_sat1")
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("--quiet",   action="count", default=0)
    parser.add_argument("--window",  type=float, default=5.0)
    parser.add_argument("--cont",    action="store_true")
    parser.add_argument("--dtype_tod",    default=np.float32)
    parser.add_argument("--dtype_map",    default=np.float64)
    
    return parser

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
        # CARLOS: change throw with span for now
        azs  = info.az_nom + np.linspace(-info.az_span/2, info.az_span/2, npoint)
        els  = np.full(npoint, info.el_nom)
        profile[:] = tele2equ(np.array([azs, els])*utils.degree, info.timestamp, detoffs=[xi0, eta0]).T[1::-1] # dec,ra
    comm.Bcast(profile, root=first)
    return profile

class DataMissing(Exception): pass

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

def calibrate_obs(obs, dtype_tod=np.float32):
    # The following stuff is very redundant with the normal mapmaker,
    # and should probably be factorized out
    mapmaking.fix_boresight_glitches(obs)
    srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])
    # Add site and weather, since they're not in obs yet
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, "so"))
    # CARLOS: We have to add glitch_flags by hand
    obs.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape),[(0, 'dets'), (1, 'samps')])
    obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])
    # CARLOS: I had to do this because for some reason when I load a tod with no_signal=True obs.signal raises a keyerror instead of being None
    try:
        obs.signal
        has_signal = True
    except KeyError:
        has_signal = False
    if has_signal == True:
    #if obs.signal is not None:
        utils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)
    # Disqualify overly cut detectors
    good_dets = mapmaking.find_usable_detectors(obs)
    obs.restrict("dets", good_dets)
    #if obs.signal is not None and len(good_dets) > 0:
    if has_signal and len(good_dets) > 0:
        # Gapfill glitches. This function name isn't the clearest
        tod_ops.get_gap_fill(obs, flags=obs.glitch_flags, swap=True)
        # Gain calibration
        gain  = 1
        # CARLOS: no calibration in sims for the moment, so we skip everything for now
        #for gtype in ["relcal","abscal"]:
        #    gain *= obs[gtype][:,None]
        obs.signal *= gain
        # Fourier-space calibration
        #fsig  = fft.rfft(obs.signal)
        #freq  = fft.rfftfreq(obs.samps.count, 1/srate)
        # iir filter
        #iir_filter  = filters.iir_filter()(freq, obs)
        #fsig       /= iir_filter
        #gain       /= iir_filter[0].real # keep track of total gain for our record
        #fsig       /= filters.timeconst_filter(None)(freq, obs)
        #fft.irfft(fsig, obs.signal, normalize=True)
        #del fsig
    # Apply pointing correction.
    # CARLOS: we have no pointing correction yet
    #obs.focal_plane.xi    += obs.boresight_offset.dx
    #obs.focal_plane.eta   += obs.boresight_offset.dy
    #obs.focal_plane.gamma += obs.boresight_offset.gamma
    return obs

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD, no_signal=False, dtype_tod=np.float32):
    my_tods = []
    my_inds = []
    if inds is None: inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            # CARLOS: I change how the wafers are loaded
            tod = context.get_obs(obs_id, dets={"wafer_slot" : [detset], "band":band}, no_signal=no_signal )
            #tod = context.get_obs(obs_id, dets={"detset":detset+'_'+band, "band":band}, no_signal=no_signal)
            #det_mask = np.logical_and(0 < np.degrees(tod.focal_plane.gamma), np.degrees(tod.focal_plane.gamma)< 10 )
            #det_mask = np.logical_and(50 < np.degrees(tod.focal_plane.gamma), np.degrees(tod.focal_plane.gamma)< 55 )
            #tod = context.get_obs(obs_id, dets={"detset":detset+'_'+band, "band":band, 'readout_id':tod.det_info.readout_id[det_mask]}, no_signal=no_signal)
            #tod.focal_plane.gamma *= -1
            
            tod = calibrate_obs(tod, dtype_tod=dtype_tod)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError: continue
    return my_tods, my_inds

def make_depth1_map(context, obslist, shape, wcs, noise_model, comps="TQU", t0=0, dtype_tod=np.float32, dtype_map=np.float64, comm=mpi.COMM_WORLD, tag="", verbose=0):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    # Set up our mapmaking equation
    signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=True, ofmt="")
    signals    = [signal_map]
    mapmaker   = mapmaking.DemodMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)
    if comm.rank == 0: L.info(pre + "Building RHS")
    time_rhs   = signal_map.rhs*0
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time.
        # CARLOS: modify how the wafer is read
        obs = context.get_obs(obs_id, dets={"wafer_slot" : [detset], "band":band})
        #obs = context.get_obs(obs_id, dets={"detset":detset+'_'+band, "band":band})
        #det_mask = np.logical_and(0 < np.degrees(obs.focal_plane.gamma), np.degrees(obs.focal_plane.gamma)<10)
        # = np.logical_and(50 < np.degrees(obs.focal_plane.gamma), np.degrees(obs.focal_plane.gamma)<55)
        #obs = context.get_obs(obs_id, dets={"detset":detset+'_'+band, "band":band, 'readout_id':obs.det_info.readout_id[det_mask]})
        #obs.focal_plane.gamma *= -1
        
        obs = calibrate_obs(obs, dtype_tod=dtype_tod)
        
        # demodulate
        hwp.demod_tod(obs)
        # flip demodU
        #obs.demodU *= -1
        
        # filter 
        #obs.dsT    = filters.fourier_filter(obs, filters.high_pass_butter4(0.5), signal_name='dsT')
        #obs.demodQ = filters.fourier_filter(obs, filters.high_pass_butter4(0.5), signal_name='demodQ')
        #obs.demodU = filters.fourier_filter(obs, filters.high_pass_butter4(0.5), signal_name='demodU')
                
        if obs.dets.count == 0: continue
        # And add it to the mapmaker
        mapmaker.add_obs(name, obs)
        # Also build the RHS for the per-pixel timestamp. First
        # make a white noise weighted timestamp per sample timestream
        Nt  = np.zeros_like(obs.signal, dtype=dtype_tod)
        Nt += obs.timestamps - t0
        Nt *= mapmaker.data[-1].nmat.ivar[:,None]
        # Bin into pixels
        pmap = signal_map.data[name].pmap
        obs_time_rhs = pmap.zeros()
        pmap.to_map(dest=obs_time_rhs, signal=Nt,)
        # Accumulate into output array
        time_rhs = time_rhs.insert(obs_time_rhs, op=np.ndarray.__iadd__)
        del obs, pmap, Nt, obs_time_rhs
        nobs_kept += 1
        
        L.info('Done with tod %s:%s:%s'%(obs_id,detset,band))

    nobs_kept = comm.allreduce(nobs_kept)
    if nobs_kept == 0: raise DataMissing("All data cut")
    for signal in signals:
        signal.prepare()
    # mapmaker doesn't know about time_rhs, so handle it manually
    if signal_map.tiled: time_rhs = tilemap.redistribute(time_rhs, comm)
    else:                time_rhs = utils.allreduce     (time_rhs, comm)
    
    if comm.rank == 0: L.info(pre + "Writing F+B map")
    map  = tilemap.map_mul(signal_map.idiv,signal_map.rhs)
    ivar = signal_map.div[0,0]
    with utils.nowarn(): tmap = utils.remove_nan(time_rhs / ivar)
    return bunch.Bunch(map=map, ivar=ivar, tmap=tmap, signal=signal_map, t0=t0)

def write_depth1_map(prefix, data):
    data.signal.write(prefix, "map",  data.map)
    data.signal.write(prefix, "ivar", data.ivar)
    data.signal.write(prefix, "time", data.tmap)
    data.signal.write(prefix, "hits", data.signal.hits)

def write_depth1_info(oname, info):
    utils.mkdir(os.path.dirname(oname))
    bunch.write(oname, info)

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
    
def main(context=None, query=None, area=None, odir=None, mode='per_obs', comps='TQU', det_in_out=False, det_left_right=False, det_upper_lower=False, ntod=None, tods=None, nset=None, max_dets=None, fixed_time=None, mindur=None, tasks_per_group=1, site='so_sat1', verbose=0, quiet=0, window=5.0, cont=False, dtype_tod=np.float32, dtype_map=np.float64):
    warnings.simplefilter('ignore')
    # Set up our communicators
    comm       = mpi.COMM_WORLD
    comm_intra = comm.Split(comm.rank // tasks_per_group)
    comm_inter = comm.Split(comm.rank  % tasks_per_group)

    SITE       = site
    verbose    = verbose - quiet
    shape, wcs = enmap.read_map_geometry(area)
    wcs        = wcsutils.WCS(wcs.to_header())

    noise_model = mapmaking.NmatWhite()
    #comps      = comps
    ncomp      = len(comps)
    meta_only  = False
    utils.mkdir(odir)

    # Set up logging.
    L   = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter( "%(rank)3d " + "%3d %3d" % (comm_inter.rank, comm.rank) + " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter(comm_intra.rank))
    L.addHandler(ch)

    context = Context(context)
    obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(context, query, mode=mode, nset=nset, ntod=ntod, tods=tods, fixed_time=fixed_time, mindur=mindur)

    # Loop over obslists and map them
    for oi in range(comm_inter.rank, len(obskeys), comm_inter.size):
        pid, detset, band= obskeys[oi]
        obslist = obslists[obskeys[oi]]
        t       = utils.floor(periods[pid,0])
        t5      = ("%05d" % t)[:5]
        prefix  = "%s/%s/depth1_%010d_%s_%s" % (odir, t5, t, detset, band)
        tag     = "%5d/%d" % (oi+1, len(obskeys))
        utils.mkdir(os.path.dirname(prefix))
        meta_done = os.path.isfile(prefix + "_info.hdf")
        maps_done = os.path.isfile(prefix + ".empty") or (
            os.path.isfile(prefix + "_time.fits") and
            os.path.isfile(prefix + "_map.fits") and
            os.path.isfile(prefix + "_ivar.fits"))
        if cont and meta_done and (maps_done or meta_only): continue
        if comm_intra.rank == 0:
            L.info("%s Proc period %4d dset %s:%s @%.0f dur %5.2f h with %2d obs" % (tag, pid, detset, band, t, (periods[pid,1]-periods[pid,0])/3600, len(obslist)))
        try:
            # 1. read in the metadata and use it to determine which tods are
            #    good and estimate how costly each is
            my_tods, my_inds = read_tods(context, obslist, comm=comm_intra, no_signal=True, dtype_tod=dtype_tod)
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
        #try:
            # 5. make the maps
        mapdata = make_depth1_map(context, [obslist[ind] for ind in my_inds], subshape, subwcs, noise_model, t0=t, comm=comm_good, tag=tag, dtype_map=dtype_map, dtype_tod=dtype_tod, comps=comps, verbose=verbose)
            # 6. write them
        write_depth1_map(prefix, mapdata)
        #except DataMissing as e:
        #    handle_empty(prefix, tag, comm_good, e)
    comm.Barrier()
    if comm.rank == 0:
        print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
