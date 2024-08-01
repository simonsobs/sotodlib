from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, sqlite3, itertools
from sotodlib import coords, mapmaking
from sotodlib.core import Context,  metadata as metadata_core, FlagManager, AxisManager, OffsetAxis
from sotodlib.core.flagman import has_any_cuts, has_all_cut
from sotodlib.io import metadata, hk_utils
from sotodlib.tod_ops import flags, jumps, gapfill, filters, detrend_tod, apodize, pca, fft_ops, sub_polyf
from sotodlib.hwp import hwp, hwp_angle_model
from sotodlib.obs_ops import splits
from sotodlib.site_pipeline import preprocess_tod
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from pixell import enmap, utils, fft, bunch, wcsutils, tilemap, colors, memory, mpi
from scipy.signal import welch
from . import util

defaults = {"query": "1",
            "odir": "./output",
            "preprocess_config": None,
            "comps": "TQU",
            "mode": "per_obs",
            "ntod": None,
            "tods": None,
            "nset": None,
            "wafer": None,
            "freq": None,
            "center_at": None,
            "site": 'so_sat3',
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
            "window":0.0, # not implemented yet
            "dtype_tod": np.float32,
            "dtype_map": np.float64,
            "atomic_db": "atomic_maps.db",
            "fixed_time": None,
            "mindur": None,
            "calc_hpf_params": False,
            "l2_data_path": "/global/cfs/cdirs/sobs/untracked/data/site/hk",
           }

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to mapmaker config.yaml file")
    
    parser.add_argument("--context",
                        help='context file')    
    parser.add_argument("--query",
                        help='query, can be a file (list of obs_id) or selection string')
    parser.add_argument("--area",
                        help='wcs geometry')
    parser.add_argument("--odir",
                        help='output directory')
    parser.add_argument("--preprocess_config", type=str,
                        help='file with the config file to run the preprocessing pipeline')
    parser.add_argument("--mode", type=str, )
    parser.add_argument("--comps", type=str,)
    parser.add_argument("--singlestream", action="store_true")
    parser.add_argument("--only_hits", action="store_true") # this will work only when we don't request splits, since I want to avoid loading the signal
    
    # detector position splits (fixed in time)
    parser.add_argument("--det_in_out", action="store_true")
    parser.add_argument("--det_left_right", action="store_true")
    parser.add_argument("--det_upper_lower", action="store_true")
    
    # time samples splits
    parser.add_argument("--scan_left_right", action="store_true")
    
    parser.add_argument("--ntod", type=int, )
    parser.add_argument("--tods", type=str, )
    parser.add_argument("--nset", type=int, )
    parser.add_argument("--wafer", type=str,
                        help="Detector set to map with")
    parser.add_argument("--freq", type=str,
                        help="Frequency band to map with")
    parser.add_argument("--max-dets", type=int, )
    parser.add_argument("--fixed_ftime", type=int, )
    parser.add_argument("--mindur", type=int, )
    parser.add_argument("--site", type=str, )
    parser.add_argument("--verbose", action="count", )
    parser.add_argument("--quiet", action="count", )
    parser.add_argument("--window", type=float, )
    parser.add_argument("--dtype_tod", )
    parser.add_argument("--dtype_map", )
    parser.add_argument("--atomic_db",
                        help='name of the atomic map database, will be saved where make_filterbin_map is being run')
    parser.add_argument("--calc_hpf_params", action="store_true",
                        help='Set to calculate the parameters used in the high-pass-filter from the data')
    parser.add_argument("--l2_data_path",
                        help='Path to level-2 data')
    return parser


def _get_config(config_file):
    return yaml.safe_load(open(config_file,'r'))


def get_ra_ref(obs, site='so_sat3'):
    # pass an AxisManager of the observation, and return two
    # ra_ref @ dec=-40 deg.   
    # 
    #t = [obs.obs_info.start_time, obs.obs_info.start_time, obs.obs_info.stop_time, obs.obs_info.stop_time]
    t_start = obs.obs_info.start_time
    t_stop = obs.obs_info.stop_time
    az = np.arange((obs.obs_info.az_center-0.5*obs.obs_info.az_throw)*utils.degree,
                   (obs.obs_info.az_center+0.5*obs.obs_info.az_throw)*utils.degree, 0.5*utils.degree)
    el = obs.obs_info.el_center*utils.degree
    
    csl = so3g.proj.CelestialSightLine.az_el(t_start*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_start = np.interp(-40*utils.degree, dec_, ra_)
    
    csl = so3g.proj.CelestialSightLine.az_el(t_stop*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_stop = np.interp(-40*utils.degree, dec_, ra_)
    return ra_ref_start, ra_ref_stop


def tele2equ(coords, ctime, detoffs=[0,0], site="so_sat3"):
    # Broadcast and flatten input arrays
    coords, ctime = utils.broadcast_arrays(coords, ctime, npre=(1,0))
    cflat = utils.to_Nd(coords, 2, axis=-1)
    tflat = utils.to_Nd(ctime,  1, axis=-1)
    dflat, dshape = utils.to_Nd(detoffs, 2, axis=-1, return_inverse=True)
    nsamp, ndet = cflat.shape[1], dflat.shape[1]
    assert cflat.shape[1:] == tflat.shape, "tele2equ coords and ctime have incompatible shapes %s vs %s" % (str(coords.shape), str(ctime.shape))
    # Set up the transform itself
    sight  = so3g.proj.CelestialSightLine.az_el(tflat, cflat[0],
                                                cflat[1],
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
    # in the x-direction, and that there's an integer number of pixels around
    # the sky. Could be done more generally, but would be much more involved,
    # and this should be good enough.
    nphi     = utils.nint(np.abs(360/ref_wcs.wcs.cdelt[0]))
    widths   = pixboxes[:,1,0]-pixboxes[:,0,0]
    pixboxes[:,0,0] = utils.rewind(pixboxes[:,0,0],
                                   ref=pixboxes[0,0,0],
                                   period=nphi)
    pixboxes[:,1,0] = pixboxes[:,0,0] + widths
    # It's now safe to find the total pixel bounding box
    union_pixbox = np.array([np.min(pixboxes[:,0],0)-pad,np.max(pixboxes[:,1],0)
                             +pad])
    # Use this to construct the output geometry
    shape = union_pixbox[1]-union_pixbox[0]
    wcs   = ref_wcs.deepcopy()
    wcs.wcs.crpix -= union_pixbox[0,::-1]
    if return_pixboxes: return shape, wcs, pixboxes
    else: return shape, wcs

def wrap_info(obs, site):
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))

    if 'flags' not in obs._fields:
        obs.wrap('flags', FlagManager.for_tod(obs))

    # Union of flags into glitch_flags.
    # Use this if not using pre-process database:
    # (glitch_flags full of 0s when we don't have the preprocess data base)
    obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),
                   [(0, 'dets'), (1, 'samps')])
    # else:
    #obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds
    #               + obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags, )
    ## TODO add --preprocess to args
    return

def get_detector_cuts_flags(obs, rfrac_range=(0.05,0.9), psat_range=(0,20)):
    try:
        flags.get_turnaround_flags(obs) 
    except Exception as e:
        return False
    flags.get_det_bias_flags(obs, rfrac_range=rfrac_range, psat_range=psat_range)
    bad_dets = has_all_cut(obs.flags.det_bias_flags)
    obs.restrict('dets', obs.dets.vals[~bad_dets])
    if obs.dets.count<=1: return False # check if I cut all the detectors after the det bias flags
    return True

def select_data(obs):
    # Restrict non optical detectors, which have nans in their focal plane
    # coordinates and will crash the mapmaking operation.
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])

    status = get_detector_cuts_flags(obs)
    if not(status):
        return False

    return True

def subtract_hwpss(obs):
    hwp.get_hwpss(obs)
    hwp.subtract_hwpss(obs)
    # if I subtract the hwpss I replace the obs.signal
    obs.signal = obs.hwpss_remove
    obs.move('hwpss_remove', None)
    
def get_trending_cuts(obs):
    # Note: n_pieces should be dependent on the length of the observation
    flags.get_trending_flags(obs,
                             n_pieces=10,
                             max_trend=2.5) #,max_samples=obs.samps.count)
    tdets = has_any_cuts(obs.flags.trends)
    obs.restrict('dets', obs.dets.vals[~tdets])
    
def get_jumps(obs):
    try:
        jflags, _, jfix = jumps.twopi_jumps(obs,
                                            signal=obs.signal,
                                            fix=True, overwrite=True)
    except IndexError:
        return False
    obs.signal = jfix
    gfilled = gapfill.fill_glitches(obs, nbuf=10, use_pca=False, modes=1,
                                    signal=obs.signal,
                                    glitch_flags=obs.flags.jumps_2pi)
    obs.signal = gfilled
    jdets = has_any_cuts(jflags)
    return True

def get_glitches(obs):
    gflags = flags.get_glitch_flags(obs,
                                    signal_name='signal',
                                    t_glitch=1e-5, buffer=10,
                                    hp_fc=1, n_sig=10, overwrite=True)
    gstats = obs.flags.glitches.get_stats()
    obs.restrict('dets', obs.dets.vals[np.asarray(gstats['intervals']) < 10])
    
def preprocess_data(obs, dtype_tod=np.float32, site='so_sat3', remove_hwpss=True):

    # Wrap extra info
    wrap_info(obs, site)

    if obs.signal is not None:
        # Data cuts
        status = select_data(obs)
        if not(status): return False
    
        # Detrend, subtract hwpss
        detrend_tod(obs, method='median', signal_name='signal')
        if remove_hwpss:
            subtract_hwpss(obs)
        
        # Trending cuts
        get_trending_cuts(obs)

        # Jump detection
        status = get_jumps(obs)
        if not(status): return False
    
        # Glitches detection
        get_glitches(obs)

        # Detrend
        if remove_hwpss:
            detrend_tod(obs, method='median', signal_name='signal')
    
        # check if all detectors are cut before going into p2p cuts
        if obs.dets.count == 0:
            return False

        # P2P cuts
        ptp_flags = flags.get_ptp_flags(obs, signal_name='signal')
        bad_dets = has_all_cut(ptp_flags)
        obs.restrict('dets', obs.dets.vals[~bad_dets])
    return True

def cal_pW(obs):
    obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW).T

def cal_rel(obs):
    # LPF + PCA
    filt = filters.low_pass_sine2(1, width=0.1)
    sigfilt = filters.fourier_filter(obs, filt, signal_name='signal')
    obs.wrap('lpf_signal', sigfilt, [(0,'dets'),(1,'samps')])
    obs.restrict('samps',(10*200, -10*200))

    try:
        pca_out = pca.get_pca(obs, signal=obs['lpf_signal'])
    except np.linalg.LinAlgError:
        return False
    pca_signal = pca.get_pca_model(obs, pca_out, signal=obs['lpf_signal'])

    med = np.median(pca_signal.weights[:,0])
    
    # Relative calib
    obs.signal = np.divide(obs.signal.T, pca_signal.weights[:,0]/med).T
    return True

def calibrate_data(obs):
    # Get data in pW 
    cal_pW(obs)
    # Get relative calibrated data
    status = cal_rel(obs)
    if not(status):
        return False
    # Get absolute calibrated data
    obs.signal = np.multiply(obs.signal.T, obs.abscal.abscal_factor).T
    return True

def readout_filter(obs):
    iir_par = obs.iir_params[f'ufm_{obs.det_info.wafer.array[0]}']
    if iir_par['a'] is None or iir_par['b'] is None:
        return False
    filt = filters.iir_filter(iir_params=iir_par, invert=True)

    obs.signal = filters.fourier_filter(obs, filt, signal_name='signal')
    
    return True

def deconvolve_detector_tconst(obs):
    filt = filters.timeconst_filter(timeconst = obs.det_cal.tau_eff,
                                        invert=True)
    
    obs.signal = filters.fourier_filter(obs, filt, signal_name='signal')

def demodulate_hwp(obs):
    apodize.apodize_cosine(obs)
    hwp.demod_tod(obs)
    obs.restrict('samps',(30*200, -30*200))

    return obs    

def IP_correct(obs):
    filt = filters.low_pass_sine2(0.5, width=0.1)
    
    T_lpf = filters.fourier_filter(obs, filt, signal_name='dsT')
    Q_lpf = filters.fourier_filter(obs, filt, signal_name='demodQ')
    U_lpf = filters.fourier_filter(obs, filt, signal_name='demodU')
    
    obs.wrap('T_lpf', T_lpf, axis_map=[(0,'dets'), (1,'samps')])
    obs.wrap('Q_lpf', Q_lpf, axis_map=[(0,'dets'), (1,'samps')])
    obs.wrap('U_lpf', U_lpf, axis_map=[(0,'dets'), (1,'samps')])
    
    obs.restrict('samps', (obs.samps.offset+10*200,
                           obs.samps.offset + obs.samps.count-10*200))

    detrend_tod(obs, method='mean', signal_name='demodQ')
    detrend_tod(obs, method='mean', signal_name='demodU')
    detrend_tod(obs, method='mean', signal_name='Q_lpf')
    detrend_tod(obs, method='mean', signal_name='U_lpf')
    detrend_tod(obs, method='mean', signal_name='T_lpf')
    
    coeffsQ = np.zeros(obs.dets.count)
    coeffsU = np.zeros(obs.dets.count)

    for di in range(obs.dets.count):
        I = np.linalg.inv(np.tensordot(np.atleast_2d(obs.T_lpf[di]),
                                       np.atleast_2d(obs.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(obs.Q_lpf[di]),
                      np.atleast_2d(obs.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsQ[di] = c[0]
    
        I = np.linalg.inv(np.tensordot(np.atleast_2d(obs.T_lpf[di]),
                                       np.atleast_2d(obs.T_lpf[di]), (1, 1)))
        c = np.matmul(np.atleast_2d(obs.U_lpf[di]),
                      np.atleast_2d(obs.T_lpf[di]).T)
        c = np.dot(I, c.T).T
        coeffsU[di] = c[0]
        
    obs.demodQ -= np.multiply(obs.T_lpf.T, coeffsQ).T
    obs.demodU -= np.multiply(obs.T_lpf.T, coeffsU).T

    return 

def cut_outlier_detectors(obs):
    ivar = 1.0/np.var(obs.demodQ, axis=-1)
    sigma = (np.percentile(ivar,84) - np.percentile(ivar, 16))/2
    mask_det = ivar > np.median(ivar) + 2*sigma
    obs.restrict('dets', obs.dets.vals[~mask_det])

def pca_dsT(obs):
    n_modes = 2
    model = pca.get_pca_model(obs, signal=obs.dsT, n_modes=n_modes)
    obs.dsT = pca.add_model(obs, model, signal=obs.dsT, scale=-1.)
    return

def high_pass_correct_dsT(obs, get_params_from_data):
    speed = (np.sum(np.abs(np.diff(np.unwrap(obs.hwp_angle)))) /
            (obs.timestamps[-1] - obs.timestamps[0])) / (2 * np.pi)

    lpf_cutoff = speed * 0.85
    lpf_cfg = {'type': 'sine2',
               'cutoff': lpf_cutoff,
               'trans_width': 0.1}
    lpf = filters.get_lpf(lpf_cfg)

    if get_params_from_data:
        fknee = np.median(obs.fk)
        alpha = -1*np.median(obs.alpha)
    else:
        fknee = 0.1
        alpha = 2

    c1f_filter = filters.counter_1_over_f(fknee, alpha) 
    filt = lpf*c1f_filter
    obs.dsT = filters.fourier_filter(obs, filt, signal_name='dsT', detrend=None)
    return

def get_psd(obs):
    #freq, Pxx_demodQ = fft_ops.calc_psd(obs, signal=obs.demodQ, nperseg=nperseg, merge=True)
    #_, Pxx_demodU = fft_ops.calc_psd(obs, signal=obs.demodU, nperseg=nperseg, merge=True)
    # TODO: To use above, need to implement (to avoid bugs):
    # nperseg = 2**x, use power as it's faster in fft operations
    # set maxsample = greater than 2**x, so that it's the same as welch 
    dt = obs.timestamps - obs.timestamps[0]
    freq, Pxx_demodQ = welch(obs.demodQ, fs=1/np.mean(np.diff(dt)), nperseg=10*60*1/np.mean(np.diff(dt)))
    _, Pxx_demodU = welch(obs.demodQ, fs=1/np.mean(np.diff(dt)), nperseg=10*60*1/np.mean(np.diff(dt)))
    obs.merge( AxisManager(OffsetAxis("nusamps", len(freq))))
    obs.wrap("freqs", freq, [(0,"nusamps")])
    obs.wrap('Pxx_demodQ', Pxx_demodQ, [(0, 'dets'), (1, 'nusamps')])
    obs.wrap('Pxx_demodU', Pxx_demodU, [(0, 'dets'), (1, 'nusamps')])
    return

def high_pass_correct(obs, get_params_from_data):
    if get_params_from_data:
        # wrap psd
        get_psd(obs)

        noise_obs = fft_ops.fit_noise_model(obs, pxx=obs.Pxx_demodQ, f=obs.freqs,
                                    merge_fit=True,fwhite=[0.1, 1], f_max=1.5,
                                    lowf=0.1, merge_name='noise_fit_statsQ')
        # we wrap the fitted params because we will use them for the dsT 1/f
        obs.wrap('fk', obs.noise_fit_statsQ.fit[:,0] , [(0, 'dets')])
        obs.wrap('alpha', obs.noise_fit_statsQ.fit[:,0] , [(0, 'dets')])

        fknee = np.median(obs.noise_fit_statsQ.fit[:,0])
        alpha = np.median(obs.noise_fit_statsQ.fit[:,2])
    else:
        fknee = 0.1
        alpha = 2

    # High-pass filter
    hpf = filters.counter_1_over_f(fknee, alpha)
    hpf_Q = filters.fourier_filter(obs, hpf, signal_name='demodQ')
    hpf_U = filters.fourier_filter(obs, hpf, signal_name='demodU')

    obs.demodQ = hpf_Q
    obs.demodU = hpf_U

    return obs
    
def filter_data(obs, calc_hpf_params):
    """ All the post-demodulation operations """
    ### Polarization
    # Correct for IP leakage at TOD level
    IP_correct(obs)
    # Custom high-pass filter
    high_pass_correct(obs, calc_hpf_params)

    ### Temperature
    # PCA
    pca_dsT(obs)
    # Counter 1/f
    high_pass_correct_dsT(obs, calc_hpf_params)
    
    # Cut detectors with wigh variance
    cut_outlier_detectors(obs)

    # we have to make the glitch_flags for the mapmaker
    # TODO: rename to more generic
    obs.flags.move('glitch_flags', None) # this is because I added it at the beginning and I cannot overwrite
    obs.flags.reduce(flags=['turnarounds', 'jumps_2pi', 'glitches'],
                     method='union', wrap=True, new_flag='glitch_flags',
                     remove_reduced=True)
    
def calibrate_obs_otf(obs, dtype_tod=np.float32, site='so_sat3',
                      det_left_right=False, det_in_out=False,
                      det_upper_lower=False,
                      remove_hwpss=True, calc_hpf_params=False):
    status = preprocess_data(obs, dtype_tod, site, remove_hwpss)
    if not(status):
        return False
    if obs.dets.count<=1:
        return obs # this will happen when ptp_cuts cuts all detectors

    status = calibrate_data(obs)
    if not(status):
        return False
    
    status = readout_filter(obs)
    if not(status):
        return False # The readout_filter failed for not having parameters

    deconvolve_detector_tconst(obs)
    
    demodulate_hwp(obs)
    
    filter_data(obs, calc_hpf_params)

    splits.det_splits_relative(obs, det_left_right=det_left_right,
                               det_upper_lower=det_upper_lower,
                               det_in_out=det_in_out,
                               wrap=True)
    return obs

def get_pwv(obs, data_dir):
    try:
        pwv_info = hk_utils.get_detcosamp_hkaman(obs, alias=['pwv'],
                                        fields = ['site.env-radiometer-class.feeds.pwvs.pwv',],
                                        data_dir = data_dir)
        pwv_all = pwv_info['env-radiometer-class']['env-radiometer-class'][0]
        pwv = np.nanmedian(pwv_all)
    except (KeyError, ValueError):
        pwv = 0.0
    return pwv

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD,
              dtype_tod=np.float32, only_hits=False, site='so_sat3',
              l2_data='/global/cfs/cdirs/sobs/untracked/data/site/hk'):
    my_tods = []
    my_inds = []
    my_ra_ref = []
    pwvs = []
    if inds is None: inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            meta = context.get_meta(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band},)
            tod = context.get_obs(meta, no_signal=True)
            #tod = context.get_obs(obs_id, dets={"wafer_slot":detset,
            #                                    "wafer.bandpass":band},
            #                      no_signal=True)
            to_remove = []
            for field in tod._fields:
                if field!='obs_info' and field!='flags' and field!='signal' and field!='focal_plane' and field!='timestamps' and field!='boresight': to_remove.append(field)
            for field in to_remove:
                tod.move(field, None)
            preprocess_data(tod, dtype_tod=dtype_tod, site=site)
            if only_hits==False:
                ra_ref_start, ra_ref_stop = get_ra_ref(tod)
                my_ra_ref.append((ra_ref_start/utils.degree,
                                  ra_ref_stop/utils.degree))
            else:
                my_ra_ref.append(None)
            my_tods.append(tod)
            my_inds.append(ind)

            tod_temp = tod.restrict('dets', meta.dets.vals[:1], in_place=False)
            pwvs.append(get_pwv(tod_temp, data_dir=l2_data))
            del tod_temp
            
        except RuntimeError: continue
    return my_tods, my_inds, my_ra_ref, pwvs

def write_hits_map(context, obslist, shape, wcs, t0=0, comm=mpi.COMM_WORLD,
                   tag="", verbose=0, site='so_sat3'):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    
    hits = enmap.zeros(shape, wcs, dtype=np.float64)
    
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata
        # from scratch, which is pointless, but shouldn't cost that much time.
        obs = context.get_obs(obs_id, dets={"wafer_slot":detset,
                                            "wafer.bandpass":band},
                              no_signal=True)
        obs = calibrate_obs_otf(obs, site=site)
        if obs is False: 
            L.info('tod %s:%s:%s failed in the preprocessing'%(obs_id,detset,band))
            continue
        rot = None
        pmap_local = coords.pmat.P.for_tod(obs, comps='T', geom=hits.geometry,
                                           rot=rot, threads="domdir",
                                           weather=mapmaking.unarr(obs.weather),
                                           site=mapmaking.unarr(obs.site),
                                           hwp=True)
        obs_hits = pmap_local.to_weights(obs, comps='T', )
        hits = hits.insert(obs_hits[0,0], op=np.ndarray.__iadd__)
    return bunch.Bunch(hits=hits)

def make_depth1_map(context, obslist, shape, wcs, noise_model, comps="TQU",
                    t0=0, dtype_tod=np.float32, dtype_map=np.float64,
                    comm=mpi.COMM_WORLD, tag="", verbose=0,
                    preprocess_config=None, split_labels=None,
                    singlestream=False, det_in_out=False, det_left_right=False,
                    det_upper_lower=False, site='so_sat3', recenter=None,
                    calc_hpf_params=False):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    # Set up our mapmaking equation
    if split_labels==None:
        # this is the case where we did not request any splits at all
        signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps,
                                              dtype=dtype_map, tiled=False,
                                              ofmt="", singlestream=singlestream,
                                              recenter=recenter )
    else:
        # this is the case where we asked for at least 2 splits (1 split set).
        # We count how many split we'll make, we need to define the Nsplits
        # maps inside the DemodSignalMap
        Nsplits = len(split_labels)
        signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps,
                                              dtype=dtype_map, tiled=False,
                                              ofmt="", Nsplits=Nsplits,
                                              singlestream=singlestream,
                                              recenter=recenter)
    signals    = [signal_map]
    mapmaker   = mapmaking.DemodMapmaker(signals, noise_model=noise_model,
                                         dtype=dtype_tod,
                                         verbose=verbose>0,
                                         singlestream=singlestream)
    if comm.rank == 0: L.info(pre + "Building RHS")
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata
        # from scratch, which is pointless, but shouldn't cost that much time.
        if preprocess_config is None:
            start_time = time.time()
            obs = context.get_obs(obs_id, dets={"wafer_slot":detset,
                                                "wafer.bandpass":band}, )
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print("Elapsed time make_depth1_map get_obs:", elapsed_time,
            #      "seconds")
        else:
            obs = preprocess_tod.load_preprocess_tod(obs_id,
                                                     configs=preprocess_config,
                                                     dets={'wafer_slot':detset,
                                                           'wafer.bandpass':band},
            )
            # TODO: modify below according to processes included in database
            
        # Correct HWP and PID polarization angles
        try:
            obs = hwp_angle_model.apply_hwp_angle_model(obs)
        except ValueError:
            continue # this is to skip the "hwp rotation direction is ambiguous" error

        if obs.dets.count <= 1: continue
        obs = calibrate_obs_otf(obs, dtype_tod=dtype_tod, det_in_out=det_in_out,
                                det_left_right=det_left_right,
                                det_upper_lower=det_upper_lower,
                                site=site, calc_hpf_params=calc_hpf_params)
        if obs is False:
            L.info('tod %s:%s:%s failed in the preprocessing'%(obs_id,detset,band)
            )
            continue
        if obs.dets.count <= 1: continue

        # And add it to the mapmaker
        if split_labels==None:
            # this is the case of no splits
            mapmaker.add_obs(name, obs)
        else:
            # this is the case of having splits. We need to pass the split_labels
            # at least. If we have detector splits fixed in time, then we pass the
            # masks in det_split_masks. Otherwise, det_split_masks will be None.
            mapmaker.add_obs(name, obs, split_labels=split_labels)

        nobs_kept += 1
        L.info('Done with tod %s:%s:%s'%(obs_id,detset,band))
    
    nobs_kept = comm.allreduce(nobs_kept)
    if nobs_kept == 0: raise DataMissing("All data cut")
    for signal in signals:
        signal.prepare()
    if comm.rank == 0: L.info(pre + "Writing F+B outputs")
    wmap = []
    weights = []
    for n_split in range(signal_map.Nsplits):
        wmap.append( signal_map.rhs[n_split] )
        div = np.diagonal(signal_map.div[n_split], axis1=0, axis2=1)
        div = np.moveaxis(div, -1, 0) # this moves the last axis to the 0th position
        weights.append(div)
    return bunch.Bunch(wmap=wmap, weights=weights, signal=signal_map, t0=t0 )

def write_depth1_map(prefix, data, split_labels=None):
    if split_labels==None:
        # we have no splits, so we save index 0 of the lists
        data.signal.write(prefix, "full_wmap", data.wmap[0])
        data.signal.write(prefix, "full_weights", data.weights[0])
        data.signal.write(prefix, "full_hits", data.signal.hits)
    else:
        # we have splits
        Nsplits = len(split_labels)
        for n_split in range(Nsplits):
            data.signal.write(prefix, "%s_wmap"%split_labels[n_split],
                              data.wmap[n_split])
            data.signal.write(prefix, "%s_weights"%split_labels[n_split],
                              data.weights[n_split])
            data.signal.write(prefix, "%s_hits"%split_labels[n_split],
                              data.signal.hits[n_split])
            
def write_depth1_info(oname, info, split_labels=None):
    utils.mkdir(os.path.dirname(oname))
    if split_labels==None:
        bunch.write(oname+'_full_info.hdf', info[0])
    else:
        # we have splits
        Nsplits = len(split_labels)
        for n_split in range(Nsplits):
            bunch.write(oname+'_%s_info.hdf'%split_labels[n_split], info[n_split])

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, colors={'DEBUG':colors.reset,
                                    'INFO':colors.lgreen,
                                    'WARNING':colors.lbrown,
                                    'ERROR':colors.lred,
                                    'CRITICAL':colors.lpurple}):
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

def handle_empty(prefix, tag, comm, e, L):
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
    comm_intra = comm.Split(comm.rank // 1) # this is a dummy intra communicator we don't need since we will do atomic maps here

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
    ch.setFormatter(ColoredFormatter(
        "%(rank)3d " + "%3d %3d" % (comm.rank, comm.rank) +
        " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter(comm.rank))
    L.addHandler(ch)

    context = Context(args['context'])
    # obslists is a dict, obskeys is a list, periods is an array, only rank 0
    # will do this and broadcast to others.
    if comm.rank==0:
        obslists, obskeys, periods, \
            obs_infos = mapmaking.build_obslists(context,
                                                 args['query'],
                                                 mode=args['mode'],
                                                 nset=args['nset'],
                                                 wafer=args['wafer'],
                                                 freq=args['freq'],
                                                 ntod=args['ntod'],
                                                 tods=args['tods'],
                                                 fixed_time=args['fixed_time'],
                                                 mindur=args['mindur'])
        L.info('Done with build_obslists')
    else:
        obslists = None ; obskeys = None; periods=None ; obs_infos = None
    obslists = comm.bcast(obslists, root=0) ; obskeys = comm.bcast(obskeys, root=0) ; periods = comm.bcast(periods, root=0) ; obs_infos = comm.bcast(obs_infos, root=0)

    tags = []
    cwd = os.getcwd()
    
    split_labels = []
    if args['det_in_out']:
        split_labels.append('det_in');split_labels.append('det_out')
    if args['det_left_right']:
        split_labels.append('det_left');split_labels.append('det_right')
    if args['det_upper_lower']:
        split_labels.append('det_upper');split_labels.append('det_lower')
    if args['scan_left_right']:
        split_labels.append('scan_left');split_labels.append('scan_right')
    if not split_labels:
        split_labels = None
    
    # We open the data base for checking if we have maps already,
    # if we do we will not run them again.
    if os.path.isfile('./'+args['atomic_db']) and not args['only_hits']:
        conn = sqlite3.connect('./'+args['atomic_db']) # open the connector, in reading mode only
        cursor = conn.cursor()
        keys_to_remove = []
        # Now we have obslists and splits ready, we look through the database
        # to remove the maps we already have from it
        for key, value in  obslists.items():
            if split_labels == None:
                # we want to run only full maps
                query_ = 'SELECT * from atomic where obs_id="%s" and telescope="%s" and freq_channel="%s" and wafer="%s" and split_label="full"'%(value[0][0], obs_infos[value[0][3]].telescope, key[2], key[1] )
                res = cursor.execute(query_)
                matches = res.fetchall()
                if len(matches)>0:
                    # this means the map (key,value) is already in the data base,
                    # so we have to remove it to not run it again
                    # it seems that removing the maps from the obskeys is enough.
                    keys_to_remove.append(key)
            else:
                # we are asking for splits
                missing_split = False
                for split_label in split_labels:
                    query_ = 'SELECT * from atomic where obs_id="%s" and telescope="%s" and freq_channel="%s" and wafer="%s" and split_label="%s"'%(value[0][0], obs_infos[value[0][3]].telescope, key[2], key[1], split_label )
                    res = cursor.execute(query_)
                    matches = res.fetchall()
                    if len(matches)==0:
                        # this means one of the requested splits is missing
                        # in the data base
                        missing_split = True
                        break
                if missing_split == False:
                    # this means we have all the splits we requested for the
                    # particular obs_id/telescope/freq/wafer
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            obskeys.remove(key)
            del obslists[key]
        conn.close() # I close since I only wanted to read

    obslist = [item[0] for key, item in obslists.items()]
    my_tods, my_inds, my_ra_ref, pwvs = read_tods(context, obslist,
                                                  comm=comm,
                                                  dtype_tod=args['dtype_tod'],
                                                  only_hits=args['only_hits'],
                                                  l2_data=args['l2_data_path'])
    my_costs     = np.array([tod.samps.count*len(mapmaking.find_usable_detectors(tod)) for tod in my_tods])
    valid        = np.where(my_costs>0)[0]
    my_tods_2, my_inds_2, my_costs = [[a[vi] for vi in valid] for a in [my_tods, my_inds, my_costs]]
    # we will do the profile and footprint here, and then allgather the
    # subshapes and subwcs.This way we don't have to communicate the
    # massive arrays such as timestamps
    subshapes = [] ; subwcses = []
    for idx,oi in enumerate(my_inds_2):
        pid, detset, band = obskeys[oi]
        obslist = obslists[obskeys[oi]]
        my_tods_atomic = [my_tods_2[idx]] ; my_infos = [obs_infos[obslist[0][3]]]
        if recenter is None:
            subshape, subwcs = find_footprint(context, my_tods_atomic, wcs,
                                              comm=comm_intra)
            subshapes.append(subshape) ; subwcses.append(subwcs)
        else:
            subshape = shape; subwcs = wcs
            subshapes.append(subshape) ; subwcses.append(subwcs)
    all_inds              = utils.allgatherv(my_inds_2, comm)
    all_costs             = utils.allgatherv(my_costs, comm)
    all_ra_ref            = comm.allgather(my_ra_ref)
    all_pwvs              = comm.allgather(pwvs)
    all_subshapes         = comm.allgather(subshapes)
    all_subwcses          = comm.allgather(subwcses)
    all_ra_ref_flatten    = [x for xs in all_ra_ref for x in xs]
    all_pwvs_flatten    = [x for xs in all_pwvs for x in xs]
    all_subshapes_flatten = [x for xs in all_subshapes for x in xs]
    all_subwcses_flatten  = [x for xs in all_subwcses for x in xs]
    mask_weights = utils.equal_split(all_costs, comm.size)[comm.rank]
    my_inds_2    = all_inds[mask_weights]
    my_ra_ref    = [all_ra_ref_flatten[idx] for idx in mask_weights]
    pwvs         = [all_pwvs_flatten[idx] for idx in mask_weights]
    my_subshapes = [all_subshapes_flatten[idx] for idx in mask_weights]
    my_subwcses  = [all_subwcses_flatten[idx] for idx in mask_weights]
    del obslist, my_inds, my_tods, my_costs, valid, all_inds, all_costs, all_ra_ref, all_ra_ref_flatten, mask_weights, all_subshapes_flatten, all_subwcses_flatten, my_tods_2, all_pwvs, all_pwvs_flatten

    for idx,oi in enumerate(my_inds_2):
        pid, detset, band = obskeys[oi]
        obslist = obslists[obskeys[oi]]
        t       = utils.floor(periods[pid,0])
        t5      = ("%05d" % t)[:5]
        prefix  = "%s/%s/atomic_%010d_%s_%s" % (args['odir'], t5, t, detset, band)
        
        subshape = my_subshapes[idx]
        subwcs   = my_subwcses[idx]

        tag     = "%5d/%d" % (oi+1, len(obskeys))
        utils.mkdir(os.path.dirname(prefix))
        meta_done = os.path.isfile(prefix + "_full_info.hdf")
        maps_done = os.path.isfile(prefix + ".empty") or (
            os.path.isfile(prefix + "_full_map.fits") and
            os.path.isfile(prefix + "_full_ivar.fits") and 
            os.path.isfile(prefix + "_full_hits.fits")
        )
        L.info("%s Proc period %4d dset %s:%s @%.0f dur %5.2f h with %2d obs" % (tag, pid, detset, band, t, (periods[pid,1]-periods[pid,0])/3600, len(obslist)))

        my_ra_ref_atomic = [my_ra_ref[idx]]
        pwv_atomic = [pwvs[idx]]
        # Save file for data base of atomic maps. We will write an individual file,
        # another script will loop over those files and write into sqlite data base
        if not args['only_hits']:
            info = []
            if split_labels is None:
                # this means the mapmaker was run without any splits requested
                info.append(bunch.Bunch(pid=pid,
                                 obs_id=obslist[0][0].encode(),
                                 telescope=obs_infos[obslist[0][3]].telescope.encode(),
                                 freq_channel=band.encode(),
                                 wafer=detset.encode(),
                                 ctime=int(t),
                                 split_label='full'.encode(),
                                 split_detail='full'.encode(),
                                 prefix_path=str(cwd+'/'+prefix+'_full').encode(),
                                 elevation=obs_infos[obslist[0][3]].el_center,
                                 azimuth=obs_infos[obslist[0][3]].az_center,
                                 RA_ref_start=my_ra_ref_atomic[0][0],
                                 RA_ref_stop=my_ra_ref_atomic[0][1],
                                 pwv=pwv_atomic
                                ))
            else:
                # splits were requested and we loop over them
                for split_label in split_labels:
                    info.append(bunch.Bunch(pid=pid,
                                 obs_id=obslist[0][0].encode(),
                                 telescope=obs_infos[obslist[0][3]].telescope.encode(),
                                 freq_channel=band.encode(),
                                 wafer=detset.encode(),
                                 ctime=int(t),
                                 split_label=split_label.encode(),
                                 split_detail=''.encode(),
                                 prefix_path=str(cwd+'/'+prefix+'_%s'%split_label).encode(),
                                 elevation=obs_infos[obslist[0][3]].el_center,
                                 azimuth=obs_infos[obslist[0][3]].az_center,
                                 RA_ref_start=my_ra_ref_atomic[0][0],
                                 RA_ref_stop=my_ra_ref_atomic[0][1],
                                 pwv=pwv_atomic
                                ))

        if not args['only_hits']:
            try:
                # 5. make the maps
                mapdata = make_depth1_map(context, obslist, subshape, subwcs,
                                          noise_model, t0=t, comm=comm_intra,
                                          tag=tag,
                                          preprocess_config=args['preprocess_config'],
                                          recenter=recenter,
                                          dtype_map=args['dtype_map'],
                                          dtype_tod=args['dtype_tod'],
                                          comps=args['comps'],
                                          verbose=args['verbose'],
                                          split_labels=split_labels,
                                          singlestream=args['singlestream'],
                                          det_in_out=args['det_in_out'],
                                          det_left_right=args['det_left_right'],
                                          det_upper_lower=args['det_upper_lower'],
                                          site=args['site'],
                                          calc_hpf_params=args["calc_hpf_params"])
                # 6. write them
                write_depth1_map(prefix, mapdata, split_labels=split_labels, )
                write_depth1_info(prefix, info, split_labels=split_labels )
            except DataMissing as e:
                # This will happen if we decide to abort a map while we are doing
                # the preprocessing.
                #handle_empty(prefix, tag, comm_intra, e, L)
                continue
        else:
            mapdata = write_hits_map(context, obslist, subshape, subwcs,
                                     t0=t, comm=comm_intra, tag=tag,
                                     verbose=args['verbose'],)
            if comm_intra.rank == 0:
                oname = "%s_%s.%s" % (prefix, "full_hits", 'fits')
                enmap.write_map(oname, mapdata.hits)
    if comm.rank == 0:
        print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
