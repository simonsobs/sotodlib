from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, sqlite3, itertools
from sotodlib import coords, mapmaking
from sotodlib.core import Context,  metadata as metadata_core, FlagManager
from sotodlib.core.flagman import has_any_cuts, has_all_cut
from sotodlib.io import metadata
from sotodlib.tod_ops import flags, jumps, gapfill, filters, detrend_tod, apodize, pca
from sotodlib.hwp import hwp
from pixell import enmap, utils, fft, bunch, wcsutils, tilemap, colors, memory, mpi
from scipy import ndimage, interpolate
from . import util

defaults = {"query": "1",
            "odir": "./output",
            "comps": "TQU",
            "mode": "per_obs",
            "ntod": None,
            "tods": None,
            "nset": None,
            "wafer": None, # not implemented yet
            "center_at": None,
            "site": 'so_sat1',
            "max_dets": None, # not implemented yet
            "verbose": 0,
            "quiet": 0,
            "tiled": 0, # not implemented yet
            "singlestream": False,
            "only_hits": False,
            "det_in_out": False,
            "det_left_right":False,
            "det_upper_lower":False,
            "scan_left_right":False,
            "tasks_per_group":1,
            "window":0.0, # not implemented yet
            "dtype_tod": np.float32,
            "dtype_map": np.float64,
            "atomic_db": "atomic_maps.db",
            "fixed_time": None,
            "mindur": None,
           }

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to mapmaker config.yaml file")
    
    parser.add_argument("--context", help='context file')    
    parser.add_argument("--query", help='query, can be a file (list of obs_id) or selection string')
    parser.add_argument("--area", help='wcs geometry')
    parser.add_argument("--odir", help='output directory')
    parser.add_argument("--mode", type=str, )
    parser.add_argument("--comps",   type=str,)
    parser.add_argument("--singlestream", action="store_true")
    parser.add_argument("--only_hits", action="store_true") # this will work only when we don't request splits, since I want to avoid loading the signal
    
    # detector position splits (fixed in time)
    parser.add_argument("--det_in_out", action="store_true")
    parser.add_argument("--det_left_right", action="store_true")
    parser.add_argument("--det_upper_lower", action="store_true")
    
    # time samples splits
    parser.add_argument("--scan_left_right", action="store_true")
    
    parser.add_argument("--ntod",    type=int, )
    parser.add_argument("--tods",    type=str, )
    parser.add_argument("--nset",    type=int, )
    parser.add_argument("--max-dets",type=int, )
    parser.add_argument("--fixed_ftime", type=int, )
    parser.add_argument("--mindur", type=int, )
    parser.add_argument("--tasks_per_group", type=int, )
    parser.add_argument("--site",    type=str, )
    parser.add_argument("--verbose", action="count", )
    parser.add_argument("--quiet",   action="count", )
    parser.add_argument("--window",  type=float, )
    parser.add_argument("--dtype_tod",  )
    parser.add_argument("--dtype_map",  )
    parser.add_argument("--atomic_db", help='name of the atomic map database, will be saved where make_filterbin_map is being run')
    return parser

def _get_config(config_file):
    return yaml.safe_load(open(config_file,'r'))

def get_ra_ref(obs, site='so_sat1'):
    # pass an AxisManager of the observation, and return two ra_ref @ dec=-40 deg.   
    # 
    #t = [obs.obs_info.start_time, obs.obs_info.start_time, obs.obs_info.stop_time, obs.obs_info.stop_time]
    t_start = obs.obs_info.start_time
    t_stop = obs.obs_info.stop_time
    az = np.arange((obs.obs_info.az_center-0.5*obs.obs_info.az_throw)*utils.degree, (obs.obs_info.az_center+0.5*obs.obs_info.az_throw)*utils.degree, 0.5*utils.degree)
    el = obs.obs_info.el_center*utils.degree
    
    csl = so3g.proj.CelestialSightLine.az_el(t_start*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    #spline = interpolate.CubicSpline(dec_, ra_, bc_type='not-a-knot')
    ra_ref_start = np.interp(-40*utils.degree, dec_, ra_)
    #ra_ref_start = spline(-40*utils.degree, nu=0)
    
    csl = so3g.proj.CelestialSightLine.az_el(t_stop*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    #spline = interpolate.CubicSpline(dec_, ra_, bc_type='not-a-knot')
    #ra_ref_stop = spline(-40*utils.degree, nu=0)
    ra_ref_stop = np.interp(-40*utils.degree, dec_, ra_)
    return ra_ref_start, ra_ref_stop

def tele2equ(coords, ctime, detoffs=[0,0], site="so_sat1"):
    # Broadcast and flatten input arrays
    coords, ctime = utils.broadcast_arrays(coords, ctime, npre=(1,0))
    cflat = utils.to_Nd(coords, 2, axis=-1)
    tflat = utils.to_Nd(ctime,  1, axis=-1)
    dflat, dshape = utils.to_Nd(detoffs, 2, axis=-1, return_inverse=True)
    nsamp, ndet = cflat.shape[1], dflat.shape[1]
    assert cflat.shape[1:] == tflat.shape, "tele2equ coords and ctime have incompatible shapes %s vs %s" % (str(coords.shape), str(ctime.shape))
    # Set up the transform itself
    sight  = so3g.proj.CelestialSightLine.az_el(tflat, cflat[0], cflat[1], roll=cflat[2] if len(cflat) > 2 else 0, site=site, weather="toco")
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
        azs  = info.az_center + np.linspace(-info.az_throw/2, info.az_throw/2, npoint)
        els  = np.full(npoint, info.el_center)
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


def calibrate_obs_new(obs, dtype_tod=np.float32, site='so_sat1', det_left_right=False, det_in_out=False, det_upper_lower=False):
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))
    obs.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0, 'dets'), (1, 'samps')])
    # Restrict non optical detectors, which have nans in their focal plane coordinates and will crash the mapmaking operation.
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    
    if obs.signal is not None:
        # this will happen in the read_tod call, where we will add the detector flags
        if det_left_right or det_in_out or det_upper_lower:
            # we add a flagmanager for the detector flags
            obs.wrap('det_flags', FlagManager.for_tod(obs))
            if det_left_right or det_in_out:
                xi = obs.focal_plane.xi
                # sort xi 
                xi_median = np.median(xi)    
            if det_upper_lower or det_in_out:
                eta = obs.focal_plane.eta
                # sort eta
                eta_median = np.median(eta)
            if det_left_right:
                mask = xi <= xi_median
                obs.det_flags.wrap_dets('det_left', np.logical_not(mask))
                mask = xi > xi_median
                obs.det_flags.wrap_dets('det_right', np.logical_not(mask))
            if det_upper_lower:
                mask = eta <= eta_median
                obs.det_flags.wrap_dets('det_lower', np.logical_not(mask))
                mask = eta > eta_median
                obs.det_flags.wrap_dets('det_upper', np.logical_not(mask))
            if det_in_out:
                # the bounding box is the center of the detset
                xi_center = np.min(xi) + 0.5 * (np.max(xi) - np.min(xi))
                eta_center = np.min(eta) + 0.5 * (np.max(eta) - np.min(eta))
                radii = np.sqrt((xi_center-xi)**2 + (eta_center-eta)**2)
                radius_median = np.median(radii)
                mask = radii <= radius_median
                obs.det_flags.wrap_dets('det_in', np.logical_not(mask))
                mask = radii > radius_median
                obs.det_flags.wrap_dets('det_out', np.logical_not(mask))
                
    # this is from Max's notebook
    if obs.signal is not None:
        flags.get_turnaround_flags(obs)
        flags.get_det_bias_flags(obs, rfrac_range=(0.05, 0.9), psat_range=(0, 20))
        bad_dets = has_all_cut(obs.flags.det_bias_flags)
        obs.restrict('dets', obs.dets.vals[~bad_dets])
        detrend_tod(obs, method='median')
        hwp.get_hwpss(obs)
        hwp.subtract_hwpss(obs)
        flags.get_trending_flags(obs, max_trend=2.5, n_pieces=10)
        tdets = has_any_cuts(obs.flags.trends)
        obs.restrict('dets', obs.dets.vals[~tdets])
        jflags, _, jfix = jumps.twopi_jumps(obs, signal=obs.hwpss_remove, fix=True, overwrite=True)
        obs.hwpss_remove = jfix
        gfilled = gapfill.fill_glitches(obs, nbuf=10, use_pca=False, modes=1, signal=obs.hwpss_remove, glitch_flags=obs.flags.jumps_2pi)
        obs.hwpss_remove = gfilled
        gflags = flags.get_glitch_flags(obs, t_glitch=1e-5, buffer=10, signal_name='hwpss_remove', hp_fc=1, n_sig=10, overwrite=True)
        #gdets = has_any_cuts(obs.flags.glitches)
        gstats = obs.flags.glitches.get_stats()
        obs.restrict('dets', obs.dets.vals[np.asarray(gstats['intervals']) < 10])
        detrend_tod(obs, method='median', signal_name='hwpss_remove')
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW).T
        obs.hwpss_remove = np.multiply(obs.hwpss_remove.T, obs.det_cal.phase_to_pW).T
        
        #LPF and PCA
        if True:
            filt = filters.low_pass_sine2(1, width=0.1)
            sigfilt = filters.fourier_filter(obs, filt, signal_name='hwpss_remove')
            obs.wrap('lpf_hwpss_remove', sigfilt, [(0,'dets'),(1,'samps')])
            obs.restrict('samps',(10*200, -10*200))
            pca_out = pca.get_pca(obs,signal=obs.lpf_hwpss_remove)
            pca_signal = pca.get_pca_model(obs, pca_out, signal=obs.lpf_hwpss_remove)
            median = np.median(pca_signal.weights[:,0])
            obs.signal = np.divide(obs.signal.T, pca_signal.weights[:,0]/median).T
        apodize.apodize_cosine(obs, apodize_samps=800)  
    return obs

def calibrate_obs_after_demod(obs, dtype_tod=np.float32):
    # project out T
    filt = filters.low_pass_sine2(0.5, width=0.1)
    T_lpf = filters.fourier_filter(obs, filt, signal_name='dsT')
    Q_lpf = filters.fourier_filter(obs, filt, signal_name='demodQ')
    U_lpf = filters.fourier_filter(obs, filt, signal_name='demodU')
    obs.wrap('T_lpf', T_lpf, axis_map=[(0,'dets'), (1,'samps')])
    obs.wrap('Q_lpf', Q_lpf, axis_map=[(0,'dets'), (1,'samps')])
    obs.wrap('U_lpf', U_lpf, axis_map=[(0,'dets'), (1,'samps')])
    
    obs.restrict('samps', (obs.samps.offset+10*200, obs.samps.offset+obs.samps.count-10*200))
    
    detrend_tod(obs, method='mean', signal_name='demodQ')
    detrend_tod(obs, method='mean', signal_name='demodU')
    detrend_tod(obs, method='mean', signal_name='Q_lpf')
    detrend_tod(obs, method='mean', signal_name='U_lpf')
    detrend_tod(obs, method='mean', signal_name='T_lpf')

    coeffsQ = np.zeros(obs.dets.count)
    coeffsU = np.zeros(obs.dets.count)

    for di in range(obs.dets.count):
        I = np.linalg.inv(np.tensordot(np.atleast_2d(obs.T_lpf[di]), np.atleast_2d(obs.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(obs.Q_lpf[di]), np.atleast_2d(obs.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsQ[di] = c[0]
        I = np.linalg.inv(np.tensordot(np.atleast_2d(obs.T_lpf[di]), np.atleast_2d(obs.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(obs.U_lpf[di]), np.atleast_2d(obs.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsU[di] = c[0]
    obs.demodQ -= np.multiply(obs.T_lpf.T, coeffsQ).T
    obs.demodU -= np.multiply(obs.T_lpf.T, coeffsU).T
    
    obs.move('hwpss_model', None)
    obs.move('hwpss_remove', None)
    obs.move('gap_filled', None)
    obs.move('lpf_hwpss_remove', None)
    
    hpf = filters.counter_1_over_f(0.1, 2)
    hpf_Q = filters.fourier_filter(obs, hpf, signal_name='demodQ')
    hpf_U = filters.fourier_filter(obs, hpf, signal_name='demodU')
    obs.demodQ = hpf_Q
    obs.demodU = hpf_U
    
    # cut 5% of higher ivar detectors
    ivar = 1.0/np.var(obs.demodQ, axis=-1)
    mask_det = ivar > np.percentile(ivar, 95)
    obs.restrict('dets', obs.dets.vals[~mask_det])
    
    return obs


def calibrate_obs(obs, dtype_tod=np.float32):
    # The following stuff is very redundant with the normal mapmaker,
    # and should probably be factorized out
    #mapmaking.fix_boresight_glitches(obs)
    srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])
    # Add site and weather, since they're not in obs yet
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, "so_sat1"))
    # CARLOS: We have to add glitch_flags by hand
    obs = obs.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0, 'dets'), (1, 'samps')])
    obs = obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])
    
    if obs.signal is not None:
        # FLAGS
        flags.get_turnaround_flags(obs)
        #tod_ops.flags.get_det_bias_flags(obs)
        #obs.wrap('flags.det_bias_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0, 'dets'), (1, 'samps')])
        
        # DETRENDING
        tod_ops.detrend.detrend_tod(obs, in_place=True)
        utils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)
    # Disqualify overly cut detectors
    good_dets = mapmaking.find_usable_detectors(obs)
    #obs.restrict("dets", good_dets)
    
    # remove detectors with an std greater than 5
    if obs.signal is not None:
        mask = np.std(obs.signal, axis=1) > 0.01
        mask = np.repeat(mask[:,None], int(obs.samps.count), axis=1)
        obs = obs.wrap("flags_stuck", so3g.proj.RangesMatrix.from_mask(mask),[(0, 'dets'), (1, 'samps')])
        #good_dets = obs.dets.vals[mask]
        #obs.restrict("dets", good_dets)

    if obs.signal is not None and len(good_dets) > 0:
        # Gain calibration
        #gain  = 1
        # CARLOS: no calibration in sims for the moment, so we skip everything for now
        #for gtype in ["relcal","abscal"]:
        #for gtype in ["phase_to_pW"]:
        #    gain *= obs.det_cal[gtype][:,None]
        #obs.signal *= gain
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW).T
        # since some of the calibrations are nans, we make a flags
        mask_notfinite = np.logical_not(np.isfinite(obs.signal)) # we want the nan detectors to be true in this mask
        obs = obs.wrap("flags_notfinite", so3g.proj.RangesMatrix.from_mask(mask_notfinite),[(0, 'dets'), (1, 'samps')])
        
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

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD, no_signal=False, dtype_tod=np.float32, only_hits=False, site='so_sat1' ):
    my_tods = []
    my_inds = []
    my_ra_ref = []
    if inds is None: inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            tod = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, no_signal=no_signal)
            tod = calibrate_obs_new(tod, dtype_tod=dtype_tod, site=site)
            if only_hits==False:
                ra_ref_start, ra_ref_stop = get_ra_ref(tod)
                my_ra_ref.append((ra_ref_start/utils.degree, ra_ref_stop/utils.degree))
            else:
                my_ra_ref.append(None)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError: continue
    return my_tods, my_inds, my_ra_ref

def write_hits_map(context, obslist, shape, wcs, t0=0, comm=mpi.COMM_WORLD, tag="", verbose=0, site='so_sat1'):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    
    hits = enmap.zeros(shape, wcs, dtype=np.float64)
    
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time.
        obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, no_signal=True)
        obs = calibrate_obs_new(obs, site=site)
        rot = None
        pmap_local = coords.pmat.P.for_tod(obs, comps='T', geom=hits.geometry, rot=rot, threads="domdir", weather=mapmaking.unarr(obs.weather), site=mapmaking.unarr(obs.site), hwp=True)
        obs_hits = pmap_local.to_weights(obs, comps='T', )
        #print('shape of hits', hits.shape)
        #print('shape of obs_hits', obs_hits.shape)
        
        hits = hits.insert(obs_hits[0,0], op=np.ndarray.__iadd__)
    return bunch.Bunch(hits=hits)

def make_depth1_map(context, obslist, shape, wcs, noise_model, comps="TQU", t0=0, dtype_tod=np.float32, dtype_map=np.float64, comm=mpi.COMM_WORLD, tag="", verbose=0, split_labels=None, singlestream=False, det_in_out=False, det_left_right=False, det_upper_lower=False, site='so_sat1', recenter=None):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    # Set up our mapmaking equation
    if split_labels==None:
        # this is the case where we did not request any splits at all
        signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=False, ofmt="", singlestream=singlestream, recenter=recenter )
    else:
        # this is the case where we asked for at least 2 splits (1 split set). We count how many split we'll make, we need to define the Nsplits maps inside the DemodSignalMap
        Nsplits = len(split_labels)
        signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=False, ofmt="", Nsplits=Nsplits, singlestream=singlestream, recenter=recenter)
    signals    = [signal_map]
    mapmaker   = mapmaking.DemodMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0, singlestream=singlestream)
    if comm.rank == 0: L.info(pre + "Building RHS")
    time_rhs   = signal_map.rhs*0 # this has an extra axis now for different splits, because signal_map.rhs does
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time.
        obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, )
        
        #obs.hwp_angle = np.mod(-1*np.unwrap(obs.hwp_solution.hwp_angle_ver3_1) + np.deg2rad(1.66-360*255/1440-90), 2*np.pi)
        
        obs = calibrate_obs_new(obs, dtype_tod=dtype_tod, det_in_out=det_in_out, det_left_right=det_left_right, det_upper_lower=det_upper_lower, site=site)
        
        # demodulate
        if singlestream == False:
            hwp.demod_tod(obs)
            obs = calibrate_obs_after_demod(obs, dtype_tod=dtype_tod)
        
        # filter 
        #if singlestream:
        #    obs.signal = filters.fourier_filter(obs, filters.high_pass_sine2(0.5))
        #else:
        #    obs.dsT    = filters.fourier_filter(obs, filters.high_pass_sine2(0.3), signal_name='dsT')
        #    obs.demodQ = filters.fourier_filter(obs, filters.high_pass_sine2(0.3), signal_name='demodQ')
        #    obs.demodU = filters.fourier_filter(obs, filters.high_pass_sine2(0.3), signal_name='demodU')
                
        if obs.dets.count == 0: continue
        
        # And add it to the mapmaker
        if split_labels==None:
            # this is the case of no splits
            mapmaker.add_obs(name, obs)
        else:
            # this is the case of having splits. We need to pass the split_labels at least. If we have detector splits fixed in time, then we pass the masks in det_split_masks. Otherwise, det_split_masks will be None
            mapmaker.add_obs(name, obs, split_labels=split_labels)
        
        if split_labels==None:
            # Case of no splits 
            # Also build the RHS for the per-pixel timestamp. First
            # make a white noise weighted timestamp per sample timestream
            Nt  = np.zeros_like(obs.signal, dtype=dtype_tod)
            Nt += obs.timestamps - t0
            Nt *= mapmaker.data[-1].nmat.ivar[:,None] # this is the data in the mapmaker object, which is simply a list 
            # Bin into pixels
            pmap = signal_map.data[(name,0)].pmap
            obs_time_rhs = pmap.zeros()
            pmap.to_map(dest=obs_time_rhs, signal=Nt,)
            # Accumulate into output array
            time_rhs[0] = time_rhs[0].insert(obs_time_rhs, op=np.ndarray.__iadd__)
        else:
            for n_split in range(Nsplits):
                # Also build the RHS for the per-pixel timestamp. First
                # make a white noise weighted timestamp per sample timestream
                Nt  = np.zeros_like(obs.signal, dtype=dtype_tod)
                Nt += obs.timestamps - t0
                Nt *= mapmaker.data[-1].nmat.ivar[:,None] # this is the data in the mapmaker object, which is simply a list 
                # Bin into pixels
                pmap = signal_map.data[(name,n_split)].pmap
                obs_time_rhs = pmap.zeros()
                pmap.to_map(dest=obs_time_rhs, signal=Nt,)
                # Accumulate into output array
                time_rhs[n_split] = time_rhs[n_split].insert(obs_time_rhs, op=np.ndarray.__iadd__)
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
    
    if comm.rank == 0: L.info(pre + "Writing F+B outputs")
    #map = [] ; ivar = [] ; tmap =[] ; 
    wmap = []
    weights = []
    for n_split in range(signal_map.Nsplits):
        #if signal_map.tiled: 
            #map.append( tilemap.map_mul(signal_map.idiv[n_split], signal_map.rhs[n_split]) )
        #else: 
            #map.append( enmap.map_mul(signal_map.idiv[n_split], signal_map.rhs[n_split]) )
        #ivar.append( signal_map.div[n_split] )
        wmap.append( signal_map.rhs[n_split] )
        weights.append(signal_map.div[n_split])
        #with utils.nowarn(): tmap.append( utils.remove_nan(time_rhs[n_split] / ivar[-1][0,0]) )
    #return bunch.Bunch(map=map, ivar=ivar, tmap=tmap, wmap=wmap, signal=signal_map, t0=t0 )
    return bunch.Bunch(wmap=wmap, weights=weights, signal=signal_map, t0=t0 )

def write_depth1_map(prefix, data, split_labels=None):
    if split_labels==None:
        # we have no splits, so we save index 0 of the lists
        #data.signal.write(prefix, "full_map",  data.map[0])
        #data.signal.write(prefix, "full_ivar", data.ivar[0])
        data.signal.write(prefix, "full_wmap", data.wmap[0])
        data.signal.write(prefix, "full_weights", data.weights[0])
        data.signal.write(prefix, "full_hits", data.signal.hits)
    else:
        # we have splits
        Nsplits = len(split_labels)
        for n_split in range(Nsplits):
            #data.signal.write(prefix, "%s_map"%split_labels[n_split],  data.map[n_split])
            #data.signal.write(prefix, "%s_ivar"%split_labels[n_split], data.ivar[n_split])
            data.signal.write(prefix, "%s_wmap"%split_labels[n_split], data.wmap[n_split])
            data.signal.write(prefix, "%s_weights"%split_labels[n_split], data.weights[n_split])
            data.signal.write(prefix, "%s_hits"%split_labels[n_split], data.signal.hits[n_split])

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
    required_fields = ['context','area']
    for req in required_fields:
        if req not in cfg.keys():
            raise KeyError("{} is a required argument. Please supply it in a config file or via the command line".format(req))
    args = cfg
    warnings.simplefilter('ignore')
    
    # Set up our communicators
    comm       = mpi.COMM_WORLD
    comm_intra = comm.Split(comm.rank // args['tasks_per_group'])
    comm_inter = comm.Split(comm.rank  % args['tasks_per_group'])

    verbose = args['verbose'] - args['quiet']
    shape, wcs = enmap.read_map_geometry(args['area'])
    wcs        = wcsutils.WCS(wcs.to_header())

    noise_model = mapmaking.NmatWhite()
    ncomp      = len(args['comps'])
    meta_only  = False
    utils.mkdir(args['odir'])
    
    recenter = None
    if args['center_at']:
        recenter = mapmaking.parse_recentering(args['center_at'])
    
    # Set up logging.
    L   = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter( "%(rank)3d " + "%3d %3d" % (comm_inter.rank, comm.rank) + " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter(comm_intra.rank))
    L.addHandler(ch)

    context = Context(args['context'])
    obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(context, args['query'], mode=args['mode'], nset=args['nset'], ntod=args['ntod'], tods=args['tods'], fixed_time=args['fixed_time'], mindur=args['mindur'])
    tags = []
    cwd = os.getcwd()
    
    split_labels = []
    if args['det_in_out']:
        split_labels.append('det_in')
        split_labels.append('det_out')
    if args['det_left_right']:
        split_labels.append('det_left')
        split_labels.append('det_right')
    if args['det_upper_lower']:
        split_labels.append('det_upper')
        split_labels.append('det_lower')
    if args['scan_left_right']:
        split_labels.append('scan_left')
        split_labels.append('scan_right')
    if not split_labels:
        #if the list 
        split_labels = None
    
    # we open the data base for checking
    if os.path.isfile('./'+args['atomic_db']) and not args['only_hits']:
        conn = sqlite3.connect('./'+args['atomic_db']) # open the connector, in reading mode only
        cursor = conn.cursor()
        # Now we have obslists and splits ready, we look through the data base to remove the maps we already have from it
        for key, value in  obslists.items():
            if split_labels == None:
                # we want to run only full maps
                query_ = 'SELECT * from atomic where obs_id="%s" and telescope="%s" and freq_channel="%s" and wafer="%s" and split_label="full"'%(value[0][0], obs_infos[value[0][3]].telescope, key[2], key[1] )
                res = cursor.execute(query_)
                matches = res.fetchall()
                if len(matches)>0:
                    # this means the map (key,value) is already in the data base, so we have to remove it to not run it again
                    # it seems that removing the maps from the obskeys is enough.
                    obskeys.remove(key) 
            else:
                # we are asking for splits
                missing_split = False
                for split_label in split_labels:
                    query_ = 'SELECT * from atomic where obs_id="%s" and telescope="%s" and freq_channel="%s" and wafer="%s" and split_label="%s"'%(value[0][0], obs_infos[value[0][3]].telescope, key[2], key[1], split_label )
                    res = cursor.execute(query_)
                    matches = res.fetchall()
                    if len(matches)==0:
                        # this means one of the requested splits is missing in the data base
                        missing_split = True
                        break
                if missing_split == False:
                    # this means we have all the splits we requested for the particular obs_id/telescope/freq/wafer
                    obskeys.remove(key)
        conn.close() # I close since I only wanted to read
    
    # Loop over obslists and map them
    for oi in range(comm_inter.rank, len(obskeys), comm_inter.size):
        pid, detset, band = obskeys[oi]
        obslist = obslists[obskeys[oi]]
        t       = utils.floor(periods[pid,0])
        t5      = ("%05d" % t)[:5]
        prefix  = "%s/%s/atomic_%010d_%s_%s" % (args['odir'], t5, t, detset, band)
        
        tag     = "%5d/%d" % (oi+1, len(obskeys))
        utils.mkdir(os.path.dirname(prefix))
        meta_done = os.path.isfile(prefix + "_full_info.hdf")
        maps_done = os.path.isfile(prefix + ".empty") or (
            os.path.isfile(prefix + "_full_map.fits") and
            os.path.isfile(prefix + "_full_ivar.fits") and 
            os.path.isfile(prefix + "_full_hits.fits")
        )
        #if cont and meta_done and (maps_done or meta_only): continue
        if comm_intra.rank == 0:
            L.info("%s Proc period %4d dset %s:%s @%.0f dur %5.2f h with %2d obs" % (tag, pid, detset, band, t, (periods[pid,1]-periods[pid,0])/3600, len(obslist)))
        try:
            # 1. read in the metadata and use it to determine which tods are
            #    good and estimate how costly each is
            my_tods, my_inds, my_ra_ref = read_tods(context, obslist, comm=comm_intra, no_signal=True, dtype_tod=args['dtype_tod'], only_hits=args['only_hits'] )
            # after read_tods the detector flags will be added to the axis manager
            my_costs  = np.array([tod.samps.count*len(mapmaking.find_usable_detectors(tod)) for tod in my_tods])
            # 2. prune tods that have no valid detectors
            valid     = np.where(my_costs>0)[0]
            my_tods, my_inds, my_costs = [[a[vi] for vi in valid] for a in [my_tods, my_inds, my_costs]]
            
            # this is for the tags
            if not args['only_hits']:
                if split_labels is None:
                    # this means the mapmaker was run without any splits requested
                    tags.append( (obslist[0][0], obs_infos[obslist[0][3]].telescope, band, detset, int(t), 'full', 'full', cwd+'/'+prefix+'_full', obs_infos[obslist[0][3]].el_center, obs_infos[obslist[0][3]].az_center, my_ra_ref[0][0], my_ra_ref[0][1], 0.0) )
                else:
                    # splits were requested and we loop over them
                    for split_label in split_labels:
                        tags.append( (obslist[0][0], obs_infos[obslist[0][3]].telescope, band, detset, int(t), split_label, '', cwd+'/'+prefix+'_%s'%split_label, obs_infos[obslist[0][3]].el_center, obs_infos[obslist[0][3]].az_center, my_ra_ref[0][0], my_ra_ref[0][1], 0.0) )
            
            all_inds  = utils.allgatherv(my_inds,     comm_intra)
            all_costs = utils.allgatherv(my_costs,    comm_intra)
            if len(all_inds)  == 0: raise DataMissing("No valid tods")
            if sum(all_costs) == 0: raise DataMissing("No valid detectors in any tods")
            # 2. estimate the scan profile and footprint. The scan profile can be done
            #    with a single task, but that task might not be the first one, so just
            #    make it mpi-aware like the footprint stuff
            my_infos = [obs_infos[obslist[ind][3]] for ind in my_inds]
            if recenter is None:
                profile  = find_scan_profile(context, my_tods, my_infos, comm=comm_intra)
                subshape, subwcs = find_footprint(context, my_tods, wcs, comm=comm_intra)
            else:
                profile = None
                subshape = shape
                subwcs = wcs
            # 3. Write out the depth1 metadata
            #d1info = bunch.Bunch(profile=profile, pid=pid, detset=detset.encode(), band=band.encode(),
            #        period=periods[pid], ids=np.char.encode([obslist[ind][0] for ind in all_inds]),
            #        box=enmap.corners(subshape, subwcs), t=t)
            #if comm_intra.rank == 0:
            #    write_depth1_info(prefix + "_info.hdf", d1info)
        except DataMissing as e:
            # This happens if we ended up with no valid tods for some reason
            handle_empty(prefix, tag, comm_intra, e)
            continue
        # 4. redistribute the valid tasks. Tasks with nothing to do don't continue
        # past here.
        my_inds   = all_inds[utils.equal_split(all_costs, comm_intra.size)[comm_intra.rank]]
        comm_good = comm_intra.Split(len(my_inds) > 0)
        if len(my_inds) == 0: continue
        if not args['only_hits']:
            # 5. make the maps
            mapdata = make_depth1_map(context, [obslist[ind] for ind in my_inds], subshape, subwcs, noise_model, t0=t, comm=comm_good, tag=tag, recenter=recenter, dtype_map=args['dtype_map'], dtype_tod=args['dtype_tod'], comps=args['comps'], verbose=args['verbose'], split_labels=split_labels, singlestream=args['singlestream'], det_in_out=args['det_in_out'], det_left_right=args['det_left_right'], det_upper_lower=args['det_upper_lower'], site=args['site'])
                # 6. write them
            write_depth1_map(prefix, mapdata, split_labels=split_labels, )
            #except DataMissing as e:
            #    handle_empty(prefix, tag, comm_good, e)
        else:
            mapdata = write_hits_map(context, [obslist[ind] for ind in my_inds], subshape, subwcs, t0=t, comm=comm_good, tag=tag, verbose=args['verbose'],)
            if comm_intra.rank == 0:
                oname = "%s_%s.%s" % (prefix, "full_hits", 'fits')
                enmap.write_map(oname, mapdata.hits)
    comm.Barrier()
    # gather the tags for writing into the sqlite database
    tags_total = comm_inter.gather(tags, root=0)
    if comm_inter.rank == 0 and not args['only_hits']:
        tags_total = list(itertools.chain.from_iterable(tags_total)) # this is because tags_total is a list of lists of tuples, and we want a list of tuples
        # Write into the atomic map database.
        conn = sqlite3.connect('./'+args['atomic_db']) # open the conector, if the database exists then it will be opened, otherwise it will be created
        cursor = conn.cursor()
        
        # Check if the table exists, if not create it
        # the tags will be telescope, frequency channel, wafer, ctime, split_label, split_details, prefix_path, elevation, pwv
        cursor.execute("""CREATE TABLE IF NOT EXISTS atomic (
                          obs_id TEXT,
                          telescope TEXT, 
                          freq_channel TEXT, 
                          wafer TEXT, 
                          ctime INTEGER,
                          split_label TEXT,
                          split_detail TEXT,
                          prefix_path TEXT,
                          elevation REAL,
                          azimuth REAL,
                          RA_ref_start REAL,
                          RA_ref_stop REAL,
                          pwv REAL
                          )""")
        conn.commit()
        
        for tuple_ in tags_total:
            cursor.execute("INSERT INTO atomic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple_)
        conn.commit()
        
        conn.close()
        print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)