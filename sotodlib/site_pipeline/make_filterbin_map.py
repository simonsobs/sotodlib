import argparse
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, sqlite3, itertools
from sotodlib import coords, mapmaking
from sotodlib.core import Context,  metadata as metadata_core, FlagManager
from sotodlib.core.flagman import has_any_cuts, has_all_cut
from sotodlib.io import metadata
from sotodlib.tod_ops import flags, jumps, gapfill, filters, detrend_tod, apodize, pca, fft_ops, sub_polyf
from sotodlib.hwp import hwp
from sotodlib.obs_ops import splits
from sotodlib.site_pipeline import preprocess_tod
from pixell import enmap, utils, fft, bunch, wcsutils, tilemap, colors, memory, mpi
from scipy import ndimage, interpolate
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew
from types import SimpleNamespace
#from tqdm import tqdm
from . import util

defaults = {"query": "1",
            "area": None,
            "nside": None,
            "nside_tile": "auto",
            "odir": "./output",
            "preprocess_config":None,
            "comps": "TQU",
            "mode": "per_obs",
            "ntod": None,
            "tods": None,
            "nset": None,
            "wafer": None,
            "freq": None,
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
            "window":0.0, # not implemented yet
            "dtype_tod": np.float32,
            "dtype_map": np.float64,
            "atomic_db": "atomic_maps.db",
            "fixed_time": None,
            "mindur": None,
            "ext": "fits",
           }

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None,
                     help="Path to mapmaker config.yaml file")

    parser.add_argument("--context", help='context file')
    parser.add_argument("--query", help='query, can be a file (list of obs_id) or selection string')
    parser.add_argument("--area", help='wcs geometry')
    parser.add_argument("--nside", type=int, help='healpix nside')
    parser.add_argument("--nside_tile", help='nside for healpix tiles; can be None, "auto", or int')
    parser.add_argument("--odir", help='output directory')
    parser.add_argument("--ext", help='output file extension')
    parser.add_argument("--preprocess_config", type=str, help='file with the config file to run the preprocessing pipeline')
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
    parser.add_argument("--wafer",   type=str, help="Detector set to map with")
    parser.add_argument("--freq",    type=str, help="Frequency band to map with")
    parser.add_argument("--max-dets",type=int, )
    parser.add_argument("--fixed_ftime", type=int, )
    parser.add_argument("--mindur", type=int, )
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

# from Tomoki: https://docs.google.com/presentation/d/1_6b19UxTKsYiSnixXi6u2Y6B83Sv2eLxDNJhAnVNong/edit#slide=id.g26e92ae9d46_0_23 slide 3
def correct_hwp(obs, bandpass='f090'):
    telescope = obs.obs_info.telescope
    if int(obs.hwp_solution.pid_direction) == 0:
        sag_sign = int(np.sign(obs.hwp_solution.offcenter[0]))
        pid_sign = -1 * sag_sign
        obs.hwp_angle *= pid_sign
    if telescope == 'satp1':
        if obs.hwp_solution.primary_encoder == 1:
            if bandpass == 'f090':
                obs.hwp_angle = np.mod(obs.hwp_angle + np.deg2rad(-1.66+49.1-90), 2*np.pi)
            elif bandpass == 'f150':
                obs.hwp_angle = np.mod(obs.hwp_angle + np.deg2rad(-1.66+49.4-90), 2*np.pi)
        elif obs.hwp_solution.primary_encoder == 2:
            if bandpass == 'f090':
                obs.hwp_angle = np.mod(obs.hwp_angle + np.deg2rad(-1.66+49.1+90), 2*np.pi)
            elif bandpass == 'f150':
                obs.hwp_angle = np.mod(obs.hwp_angle + np.deg2rad(-1.66+49.4+90), 2*np.pi)
    elif telescope == 'satp3':
        if obs.hwp_solution.primary_encoder == 1:
            if bandpass == 'f090':
                obs.hwp_angle = np.mod(-1*obs.hwp_angle + np.deg2rad(-1.66-2.29+90), 2*np.pi)
            elif bandpass == 'f150':
                obs.hwp_angle = np.mod(-1*obs.hwp_angle + np.deg2rad(-1.66-1.99+90), 2*np.pi)
        elif obs.hwp_solution.primary_encoder == 2:
            if bandpass == 'f090':
                obs.hwp_angle = np.mod(-1*obs.hwp_angle + np.deg2rad(-1.66-2.29-90), 2*np.pi)
            elif bandpass == 'f150':
                obs.hwp_angle = np.mod(-1*obs.hwp_angle + np.deg2rad(-1.66-1.99-90), 2*np.pi)
    return;

def ptp_cuts(aman, signal_name='dsT', kurtosis_threshold=5):
    while True:
        if aman.dets.count > 0:
            ptps = np.ptp(aman[signal_name], axis=1)
        else:
            break
        kurtosis_ptp = kurtosis(ptps)
        if kurtosis_ptp < kurtosis_threshold:
            print(f'dets:{aman.dets.count}, ptp_kurt: {kurtosis_ptp:.1f}')
            break
        else:
            max_is_bad_factor = np.max(ptps)/np.median(ptps)
            min_is_bad_factor = np.median(ptps)/np.min(ptps)
            if max_is_bad_factor > min_is_bad_factor:
                aman.restrict('dets', aman.dets.vals[ptps < np.max(ptps)])
            else:
                aman.restrict('dets', aman.dets.vals[ptps > np.min(ptps)])
            print(f'dets:{aman.dets.count}, ptp_kurt: {kurtosis_ptp:.1f}')
    print(f'dets: {aman.dets.count}')

def calibrate_obs_with_preprocessing(obs, dtype_tod=np.float32, site='so_sat1', det_left_right=False, det_in_out=False, det_upper_lower=False):
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    # Since now I have flags already calculated, I can union them into a "glitch_flags" flags, which will be used to discard overcut detectors and calculate the cost for MPI.
    # Union of flags into glitch_flags.
    obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds + obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags, )
    good_dets = mapmaking.find_usable_detectors(obs)

    if obs.signal is not None and len(good_dets)>0:
        obs.restrict("dets", good_dets)
        # Adding detector splits if we asked for them
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
        # peak to peak
        ptp_cuts(obs, signal_name='signal')
        detrend_tod(obs, method='median', signal_name='hwpss_remove')
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW / obs.abscal.abscal_factor).T
        obs.hwpss_remove = np.multiply(obs.hwpss_remove.T, obs.det_cal.phase_to_pW / obs.abscal.abscal_factor).T
        # PCA relcal
        filt = filters.low_pass_sine2(1, width=0.1)
        sigfilt = filters.fourier_filter(obs, filt, signal_name='hwpss_remove')
        obs.wrap('lpf_hwpss_remove', sigfilt, [(0,'dets'),(1,'samps')])
        obs.restrict('samps',(10*200, -10*200))
        if obs.dets.count<=1: return obs # check if we have enough detectors
        pca_out = pca.get_pca(obs,signal=obs.lpf_hwpss_remove)
        pca_signal = pca.get_pca_model(obs, pca_out, signal=obs.lpf_hwpss_remove)
        median = np.median(pca_signal.weights[:,0])
        obs.signal = np.divide(obs.signal.T, pca_signal.weights[:,0]/median).T
        apodize.apodize_cosine(obs)
        hwp.demod_tod(obs)
        obs.restrict('samps',(30*200, -30*200))
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
        #obs.move('hwpss_model', None)
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

def calibrate_obs_otf(obs, dtype_tod=np.float32, site='so_sat1', det_left_right=False, det_in_out=False, det_upper_lower=False):
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))
    # Restrict non optical detectors, which have nans in their focal plane coordinates and will crash the mapmaking operation.
    # Union of flags into glitch_flags.
    #obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds + obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags, )
    obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0, 'dets'), (1, 'samps')]) # This is a glitch_flags full of 0s when we don't have the preprocess data base
    if obs.signal is not None:
        obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
        splits.det_splits_relative(obs, det_left_right=det_left_right, det_upper_lower=det_upper_lower, det_in_out=det_in_out, wrap=True)

        #flags.get_turnaround_flags(obs, t_buffer=0.1, truncate=True)
        flags.get_turnaround_flags(obs)
        flags.get_det_bias_flags(obs, rfrac_range=(0.05, 0.9), psat_range=(0, 20))
        bad_dets = has_all_cut(obs.flags.det_bias_flags)
        obs.restrict('dets', obs.dets.vals[~bad_dets])
        if obs.dets.count<=1: return obs # check if I cut all the detectors after the det bias flags
        detrend_tod(obs, method='median')
        hwp.get_hwpss(obs)
        hwp.subtract_hwpss(obs)
        flags.get_trending_flags(obs, max_trend=2.5, n_pieces=10)
        tdets = has_any_cuts(obs.flags.trends)
        obs.restrict('dets', obs.dets.vals[~tdets])
        if obs.dets.count<=1: return obs # check if I cut all the detectors after the trending flags
        jflags, _, jfix = jumps.twopi_jumps(obs, signal=obs.hwpss_remove, fix=True, overwrite=True)
        obs.hwpss_remove = jfix
        gfilled = gapfill.fill_glitches(obs, nbuf=10, use_pca=False, modes=1, signal=obs.hwpss_remove, glitch_flags=obs.flags.jumps_2pi)
        obs.hwpss_remove = gfilled
        gflags = flags.get_glitch_flags(obs, t_glitch=1e-5, buffer=10, signal_name='hwpss_remove', hp_fc=1, n_sig=10, overwrite=True)
        #gdets = has_any_cuts(obs.flags.glitches)
        gstats = obs.flags.glitches.get_stats()
        obs.restrict('dets', obs.dets.vals[np.asarray(gstats['intervals']) < 10])
        detrend_tod(obs, method='median', signal_name='hwpss_remove')
        # peak to peak
        ptp_cuts(obs, signal_name='signal')
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW / obs.abscal.abscal_factor ).T
        obs.hwpss_remove = np.multiply(obs.hwpss_remove.T, obs.det_cal.phase_to_pW / obs.abscal.abscal_factor).T
        #LPF and PCA
        filt = filters.low_pass_sine2(1, width=0.1)
        sigfilt = filters.fourier_filter(obs, filt, signal_name='hwpss_remove')
        obs.wrap('lpf_hwpss_remove', sigfilt, [(0,'dets'),(1,'samps')])
        obs.restrict('samps',(10*200, -10*200))
        # check if we have enough detectors
        if obs.dets.count<=1: return obs
        pca_out = pca.get_pca(obs,signal=obs.lpf_hwpss_remove)
        pca_signal = pca.get_pca_model(obs, pca_out, signal=obs.lpf_hwpss_remove)
        median = np.median(pca_signal.weights[:,0])
        obs.signal = np.divide(obs.signal.T, pca_signal.weights[:,0]/median).T
        filt = filters.iir_filter(iir_params=obs.iir_params[f'ufm_{obs.det_info.wafer.array[0]}'], invert=True)
        obs.signal = filters.fourier_filter(obs, filt)
        apodize.apodize_cosine(obs)
        hwp.demod_tod(obs)
        obs.restrict('samps',(30*200, -30*200))
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
        # cut detectors
        ivar = 1.0/np.var(obs.demodQ, axis=-1)
        sigma = (np.percentile(ivar,84) - np.percentile(ivar, 16))/2
        mask_det = ivar > np.median(ivar) + 5*sigma
        obs.restrict('dets', obs.dets.vals[~mask_det])
        # we have to make the glitch_flags for the mapmaker
        obs.flags.move('glitch_flags', None) # this is because I added it at the beginning and I cannot overwrite
        obs.flags.reduce(flags=['turnarounds', 'jumps_2pi', 'glitches'], method='union', wrap=True, new_flag='glitch_flags', remove_reduced=True)
    return obs

def model_func(x, sigma, fk, alpha):
    return sigma**2 * (1 + (x/fk)**alpha)

def log_fit_func(x, sigma, fk, alpha):
    return np.log(model_func(x, sigma, fk, alpha))

def calibrate_obs_tomoki(obs, dtype_tod=np.float32, site='so_sat1', det_left_right=False, det_in_out=False, det_upper_lower=False):
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))
    obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0, 'dets'), (1, 'samps')])
    # Restrict non optical detectors, which have nans in their focal plane coordinates and will crash the mapmaking operation.
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    obs.restrict('dets', obs.dets.vals[(0.2<obs.det_cal.r_frac)&(obs.det_cal.r_frac<0.8)])

    if obs.signal is not None:
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

        nperseg = 200*1000

        obs.focal_plane.gamma = np.arctan(np.tan(obs.focal_plane.gamma))
        flags.get_turnaround_flags(obs, t_buffer=0.1, truncate=True)
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW).T
        freq, Pxx = fft_ops.calc_psd(obs, nperseg=nperseg, merge=True)
        wn = fft_ops.calc_wn(obs)
        obs.wrap('wn', wn, [(0, 'dets')])

        obs.restrict('dets', obs.dets.vals[(20<obs.wn*1e6)&(obs.wn*1e6<40)])
        print(f'dets: {obs.dets.count}')

        if obs.dets.count<=1: return obs
        # peak to peak restrict
        obs.restrict('dets', obs.dets.vals[np.ptp(obs.signal, axis=1) < 0.5])
        print(f'dets: {obs.dets.count}')

        if obs.dets.count<=1: return obs

        hwp.get_hwpss(obs)
        hwp.subtract_hwpss(obs)
        obs.move('signal', None)
        obs.move('hwpss_remove', 'signal')
        freq, Pxx = fft_ops.calc_psd(obs, nperseg=nperseg, merge=False)
        obs.Pxx = Pxx

        detrend_tod(obs, method='median')
        apodize.apodize_cosine(obs, apodize_samps=2000)

        # the demodulation will happen here
        speed = (np.sum(np.abs(np.diff(np.unwrap(obs.hwp_angle)))) /
                (obs.timestamps[-1] - obs.timestamps[0])) / (2 * np.pi)
        bpf_center = 4 * speed
        bpf_width = speed * 2. * 0.9
        bpf_cfg = {'type': 'sine2',
                   'center': bpf_center,
                   'width': bpf_width,
                   'trans_width': 0.1}

        lpf_cutoff = speed * 0.9
        lpf_cfg = {'type': 'sine2',
                   'cutoff': lpf_cutoff,
                   'trans_width': 0.1}
        hwp.demod_tod(obs, bpf_cfg=bpf_cfg, lpf_cfg=lpf_cfg)

        obs.restrict('samps', (obs.samps.offset+2000, obs.samps.offset + obs.samps.count-2000))
        obs.move('signal', None)
        detrend_tod(obs, signal_name='dsT', method='linear')
        detrend_tod(obs, signal_name='demodQ', method='linear')
        detrend_tod(obs, signal_name='demodU', method='linear')
        freq, Pxx_demodQ = fft_ops.calc_psd(obs, signal=obs.demodQ, nperseg=nperseg, merge=True)
        freq, Pxx_demodU = fft_ops.calc_psd(obs, signal=obs.demodU, nperseg=nperseg, merge=True)
        obs.wrap('Pxx_demodQ', Pxx_demodQ, [(0, 'dets'), (1, 'nusamps')])
        obs.wrap('Pxx_demodU', Pxx_demodU, [(0, 'dets'), (1, 'nusamps')])

        mask = np.ones_like(obs.dsT, dtype='bool')

        lamQ, lamU = [], []
        AQ, AU = [], []

        for di, det in enumerate(obs.dets.vals[:]):
            x = obs.dsT[di][mask[di]]
            y1 = obs.demodQ[di][mask[di]]
            y2 = obs.demodU[di][mask[di]]

            z1 = np.polyfit(x, y1, 1)
            z2 = np.polyfit(x, y2, 1)
            _lamQ, _AQ = z1[0], z1[1]
            _lamU, _AU = z2[0], z2[1]

            lamQ.append(_lamQ)
            lamU.append(_lamU)
            AQ.append(_AQ)
            AU.append(_AU)

        lamQ, lamU = np.array(lamQ), np.array(lamU)
        obs.wrap('lamQ', lamQ, [(0, 'dets')])
        obs.wrap('lamU', lamU, [(0, 'dets')])

        AQ, AU = np.array(AQ), np.array(AU)
        obs.wrap('AQ', AQ, [(0, 'dets')])
        obs.wrap('AU', AU, [(0, 'dets')])

        obs.demodQ -= (obs.dsT * obs.lamQ[:, np.newaxis] + obs.AQ[:, np.newaxis])
        obs.demodU -= (obs.dsT * obs.lamU[:, np.newaxis] + obs.AU[:, np.newaxis])

        freq, Pxx_demodQ_new = fft_ops.calc_psd(obs, signal=obs.demodQ, nperseg=nperseg, merge=False)
        freq, Pxx_demodU_new = fft_ops.calc_psd(obs, signal=obs.demodU, nperseg=nperseg, merge=False)
        obs.Pxx_demodQ = Pxx_demodQ_new
        obs.Pxx_demodU = Pxx_demodU_new

        mask_valid_freqs = (1e-4<obs.freqs) & (obs.freqs < 1.9)
        x = obs.freqs[mask_valid_freqs]
        obs.wrap_new('sigma', ('dets', ))
        obs.wrap_new('fk', ('dets', ))
        obs.wrap_new('alpha', ('dets', ))

        for di, det in enumerate(obs.dets.vals):
            y = obs.Pxx_demodQ[di, mask_valid_freqs]
            popt, pcov = curve_fit(log_fit_func, x, np.log(y), p0=(np.sqrt(np.median(y[x>0.2])), 0.01, -2.), maxfev=100000)
            obs.sigma[di] = popt[0]
            obs.fk[di] = popt[1]
            obs.alpha[di] = popt[2]

        kurt_threshold=0.5
        skew_threshold=0.5

        valid_scan = np.logical_and(np.logical_or(obs.flags["left_scan"].mask(),
                                              obs.flags["right_scan"].mask()),
                                ~obs.flags["turnarounds"].mask())

        subscan_indices_l = sub_polyf._get_subscan_range_index(obs.flags["left_scan"].mask())
        subscan_indices_r = sub_polyf._get_subscan_range_index(obs.flags["right_scan"].mask())
        subscan_indices = np.vstack([subscan_indices_l, subscan_indices_r])
        subscan_indices= subscan_indices[np.argsort(subscan_indices[:, 0])]

        subscan_Qstds = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Ustds = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Qkurt = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Ukurt = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Qskew = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Uskew = np.zeros([obs.dets.count, len(subscan_indices)])

        for subscan_i, subscan in enumerate(subscan_indices):
            _Qsig= obs.demodQ[:,subscan[0]:subscan[1]+1]
            _Usig= obs.demodU[:,subscan[0]:subscan[1]+1]

            _Qmean = np.mean(_Qsig, axis=1)[:,np.newaxis]
            _Umean = np.mean(_Usig, axis=1)[:,np.newaxis]

            _Qstd = np.std(_Qsig, axis=1)
            _Ustd = np.std(_Usig, axis=1)

            _Qkurt = kurtosis(_Qsig, axis=1)
            _Ukurt = kurtosis(_Usig, axis=1)

            _Qskew = skew(_Qsig, axis=1)
            _Uskew = skew(_Usig, axis=1)

            obs.demodQ[:,subscan[0]:subscan[1]+1] -= _Qmean
            obs.demodU[:,subscan[0]:subscan[1]+1] -= _Umean

            subscan_Qstds[:, subscan_i] = _Qstd
            subscan_Ustds[:, subscan_i] = _Ustd
            subscan_Qkurt[:, subscan_i] = _Qkurt
            subscan_Ukurt[:, subscan_i] = _Ukurt
            subscan_Qskew[:, subscan_i] = _Qskew
            subscan_Uskew[:, subscan_i] = _Uskew

        badsubscan_indicator = (np.abs(subscan_Qkurt) > kurt_threshold) | (np.abs(subscan_Ukurt) > kurt_threshold) |\
                                (np.abs(subscan_Qskew) > skew_threshold) | (np.abs(subscan_Uskew) > skew_threshold)
        badsubscan_flags = np.zeros([obs.dets.count, obs.samps.count], dtype='bool')
        for subscan_i, subscan in enumerate(subscan_indices):
            badsubscan_flags[:, subscan[0]:subscan[1]+1] = badsubscan_indicator[:, subscan_i, np.newaxis]
        badsubscan_flags = so3g.proj.RangesMatrix.from_mask(badsubscan_flags)

        obs.flags.wrap('bad_subscan', badsubscan_flags)

        filt = filters.counter_1_over_f(np.median(obs.fk), -2*np.median(obs.alpha))
        obs.demodQ = filters.fourier_filter(obs, filt, signal_name='demodQ')
        obs.demodU = filters.fourier_filter(obs, filt, signal_name='demodU')

        freq, Pxx_demodQ = fft_ops.calc_psd(obs, signal=obs.demodQ, nperseg=nperseg, merge=False)
        freq, Pxx_demodU = fft_ops.calc_psd(obs, signal=obs.demodU, nperseg=nperseg, merge=False)

        obs.Pxx_demodQ = Pxx_demodQ
        obs.Pxx_demodU = Pxx_demodU

        wn = fft_ops.calc_wn(obs, obs.Pxx_demodQ, low_f=0.1, high_f=1.)
        obs.wrap('inv_var', wn**(-2), [(0, 'dets')])
        if True:
            lo, hi = np.percentile(obs.inv_var, [3, 97])
            obs.restrict('dets', obs.dets.vals[(lo < obs.inv_var) & (obs.inv_var < hi)])
        if obs.dets.count<=1: return obs

        glitches_T = flags.get_glitch_flags(obs, signal_name='dsT', merge=True, name='glitches_T')
        glitches_Q = flags.get_glitch_flags(obs, signal_name='demodQ', merge=True, name='glitches_Q')
        glitches_U = flags.get_glitch_flags(obs, signal_name='demodU', merge=True, name='glitches_U')
        obs.flags.reduce(flags=['glitches_T', 'glitches_Q', 'glitches_U'], method='union', wrap=True, new_flag='glitches', remove_reduced=True)
        obs.flags.move('glitch_flags', None)
        obs.flags.reduce(flags=['turnarounds', 'bad_subscan', 'glitches'], method='union', wrap=True, new_flag='glitch_flags', remove_reduced=True)
    return obs

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD, dtype_tod=np.float32, only_hits=False, site='so_sat1'):
    my_tods = []
    my_inds = []
    my_ra_ref = []
    if inds is None: inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            tod = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, no_signal=True)
            to_remove = []
            for field in tod._fields:
                if field!='obs_info' and field!='flags' and field!='signal' and field!='focal_plane' and field!='timestamps' and field!='boresight': to_remove.append(field)
            for field in to_remove:
                tod.move(field, None)
            #tod = calibrate_obs_with_preprocessing(tod, dtype_tod=dtype_tod, site=site)
            tod = calibrate_obs_otf(tod, dtype_tod=dtype_tod, site=site)
            #tod = calibrate_obs_tomoki(tod, dtype_tod=dtype_tod, site=site)
            if only_hits==False:
                ra_ref_start, ra_ref_stop = get_ra_ref(tod)
                my_ra_ref.append((ra_ref_start/utils.degree, ra_ref_stop/utils.degree))
            else:
                my_ra_ref.append(None)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError: continue
    return my_tods, my_inds, my_ra_ref

def write_hits_map(context, obslist, shape=None, wcs=None, nside=None, nside_tile='auto', t0=0, comm=mpi.COMM_WORLD, tag="", verbose=0, site='so_sat1'):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time.
        obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, no_signal=True)
        obs = calibrate_obs_otf(obs, site=site)
        rot = None
        if shape is not None:
            hits = enmap.zeros(shape, wcs, dtype=np.float64)
            pmap_local = coords.pmat.P.for_tod(obs, comps='T', geom=hits.geometry, rot=rot, threads="domdir", weather=mapmaking.unarr(obs.weather), site=mapmaking.unarr(obs.site), hwp=True)
            obs_hits = pmap_local.to_weights(obs, comps='T', )
            hits = hits.insert(obs_hits[0,0], op=np.ndarray.__iadd__)
        else:
            hp_geom = SimpleNamespace(nside=nside, nside_tile=nside_tile)
            threads = ["tiles", "simple"][hp_geom.nside_tile is None]
            pmap_local = coords.pmat.P.for_tod(obs, comps='T', geom=None, hp_geom=hp_geom, threads=threads, weather=mapmaking.unarr(obs.weather), site=mapmaking.unarr(obs.site), hwp=True)
            obs_hits = pmap_local.to_weights(obs, comps='T', )
            hits = mapmaking.untile_healpix(obs_hits)
    return bunch.Bunch(hits=hits)

def make_depth1_map(context, obslist, noise_model, shape=None, wcs=None, nside=None, nside_tile='auto', comps="TQU", t0=0, dtype_tod=np.float32, dtype_map=np.float64, comm=mpi.COMM_WORLD, tag="", verbose=0, preprocess_config=None, split_labels=None, singlestream=False, det_weights=None, det_in_out=False, det_left_right=False, det_upper_lower=False, site='so_sat1', recenter=None, ext='fits'):
    L = logging.getLogger(__name__)
    pre = "" if tag is None else tag + " "
    if comm.rank == 0: L.info(pre + "Initializing equation system")
    # Set up our mapmaking equation
    if split_labels==None:
        # this is the case where we did not request any splits at all
        Nsplits = 1
    else:
        Nsplits = len(split_labels)

    if nside is not None and shape is None:
        if recenter is not None:
            raise NotImplementedError("recenter not supported for Healpix")
        signal_map = mapmaking.DemodSignalMapHealpix(nside, nside_tile, comm, comps=comps, dtype=dtype_map, ofmt="", Nsplits=Nsplits, singlestream=singlestream, ext=ext)
    elif nside is None and shape is not None:
        signal_map = mapmaking.DemodSignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=False, ofmt="", Nsplits=Nsplits, singlestream=singlestream, recenter=recenter, ext=ext)
    else:
        raise ValueError("Exactly one of nside and shape should be None")
    signals    = [signal_map]
    mapmaker   = mapmaking.DemodMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0, singlestream=singlestream)
    if comm.rank == 0: L.info(pre + "Building RHS")
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = "%s:%s:%s" % (obs_id, detset, band)
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time.
        if preprocess_config is None:
            obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, )
        else:
            obs = preprocess_tod.load_preprocess_tod(obs_id, configs=preprocess_config, dets={'wafer_slot':detset, 'wafer.bandpass':band}, )
        correct_hwp(obs, bandpass=band)
        #obs = calibrate_obs_tomoki(obs, dtype_tod=dtype_tod, det_in_out=det_in_out, det_left_right=det_left_right, det_upper_lower=det_upper_lower, site=site)
        if obs.dets.count <= 1: continue
        #obs = calibrate_obs_with_preprocessing(obs, dtype_tod=dtype_tod, det_in_out=det_in_out, det_left_right=det_left_right, det_upper_lower=det_upper_lower, site=site)
        obs = calibrate_obs_otf(obs, dtype_tod=dtype_tod, det_in_out=det_in_out, det_left_right=det_left_right, det_upper_lower=det_upper_lower, site=site)
        if obs.dets.count <= 1: continue

        if obs.dets.count == 0: continue
        # And add it to the mapmaker
        if split_labels==None:
            # this is the case of no splits
            mapmaker.add_obs(name, obs)
        else:
            # this is the case of having splits. We need to pass the split_labels at least. If we have detector splits fixed in time, then we pass the masks in det_split_masks. Otherwise, det_split_masks will be None
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
        weights.append(signal_map.div[n_split])
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
    area = args['area']
    if area is not None:
        shape, wcs = enmap.read_map_geometry(args['area'])
        wcs        = wcsutils.WCS(wcs.to_header())
    else:
        shape, wcs = None, None

    nside = args['nside']
    if nside is not None: nside = int(nside)
    nside_tile = args['nside_tile']
    if nside_tile is not None and nside_tile != 'auto': nside_tile = int(nside_tile)

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
    ch.setFormatter(ColoredFormatter( "%(rank)3d " + "%3d %3d" % (comm.rank, comm.rank) + " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter(comm.rank))
    L.addHandler(ch)

    context = Context(args['context'])
    # obslists is a dict, obskeys is a list, periods is an array, only rank 0 will do this and broadcast to others.
    if comm.rank==0:
        obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(context, args['query'], mode=args['mode'], nset=args['nset'], wafer=args['wafer'], freq=args['freq'], ntod=args['ntod'], tods=args['tods'], fixed_time=args['fixed_time'], mindur=args['mindur'])
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

    # we open the data base for checking if we have maps already, if we do we will not run them again.
    if os.path.isfile('./'+args['atomic_db']) and not args['only_hits']:
        conn = sqlite3.connect('./'+args['atomic_db']) # open the connector, in reading mode only
        cursor = conn.cursor()
        keys_to_remove = []
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
                    keys_to_remove.append(key)
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
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            obskeys.remove(key)
            del obslists[key]
        conn.close() # I close since I only wanted to read

    if area is not None:
        obslist = [item[0] for key, item in obslists.items()]
        my_tods, my_inds, my_ra_ref = read_tods(context, obslist, comm=comm, dtype_tod=args['dtype_tod'], only_hits=args['only_hits'])
        my_costs     = np.array([tod.samps.count*len(mapmaking.find_usable_detectors(tod)) for tod in my_tods])
        valid        = np.where(my_costs>0)[0]
        my_tods_2, my_inds_2, my_costs = [[a[vi] for vi in valid] for a in [my_tods, my_inds, my_costs]]
        # we will do the profile and footprint here, and then allgather the subshapes and subwcs. This way we don't have to communicate the massive arrays such as timestamps
        subshapes = [] ; subwcses = []
        for idx,oi in enumerate(my_inds_2):
            pid, detset, band = obskeys[oi]
            obslist = obslists[obskeys[oi]]
            my_tods_atomic = [my_tods_2[idx]] ; my_infos = [obs_infos[obslist[0][3]]]
            if recenter is None and shape is not None:
                subshape, subwcs = find_footprint(context, my_tods_atomic, wcs, comm=comm_intra)
                subshapes.append(subshape) ; subwcses.append(subwcs)
            else:
                subshape = shape; subwcs = wcs
                subshapes.append(subshape) ; subwcses.append(subwcs)
        all_inds              = utils.allgatherv(my_inds_2, comm)
        all_costs             = utils.allgatherv(my_costs, comm)
        all_ra_ref            = comm.allgather(my_ra_ref)
        all_subshapes         = comm.allgather(subshapes)
        all_subwcses          = comm.allgather(subwcses)
        all_ra_ref_flatten    = [x for xs in all_ra_ref for x in xs]
        all_subshapes_flatten = [x for xs in all_subshapes for x in xs]
        all_subwcses_flatten  = [x for xs in all_subwcses for x in xs]
        mask_weights = utils.equal_split(all_costs, comm.size)[comm.rank]
        my_inds_2    = all_inds[mask_weights]
        my_ra_ref    = [all_ra_ref_flatten[idx] for idx in mask_weights]
        my_subshapes = [all_subshapes_flatten[idx] for idx in mask_weights]
        my_subwcses  = [all_subwcses_flatten[idx] for idx in mask_weights]
        del obslist, my_inds, my_tods, my_costs, valid, all_inds, all_costs, all_ra_ref, all_ra_ref_flatten, mask_weights, all_subshapes_flatten, all_subwcses_flatten, my_tods_2
    else:
        obslist = [item[0] for key, item in obslists.items()]
        my_inds_2 = list(range(comm.rank, len(obslist), comm.size)) # This misses out the mpi load balancing but probably ok (?)
        my_subshapes = [None] * len(my_inds_2)
        my_subwcses = [None] * len(my_inds_2)
        my_ra_ref = [[0, 0]] * len(my_inds_2)
        del obslist

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
        # this is for the tags
        if not args['only_hits']:
            if split_labels is None:
                # this means the mapmaker was run without any splits requested
                tags.append( (obslist[0][0], obs_infos[obslist[0][3]].telescope, band, detset, int(t), 'full', 'full', cwd+'/'+prefix+'_full', obs_infos[obslist[0][3]].el_center, obs_infos[obslist[0][3]].az_center, my_ra_ref_atomic[0][0], my_ra_ref_atomic[0][1], 0.0) )
            else:
                # splits were requested and we loop over them
                for split_label in split_labels:
                    tags.append( (obslist[0][0], obs_infos[obslist[0][3]].telescope, band, detset, int(t), split_label, '', cwd+'/'+prefix+'_%s'%split_label, obs_infos[obslist[0][3]].el_center, obs_infos[obslist[0][3]].az_center, my_ra_ref_atomic[0][0], my_ra_ref_atomic[0][1], 0.0) )

        if not args['only_hits']:
            try:
                # 5. make the maps
                mapdata = make_depth1_map(context, obslist, noise_model, subshape, subwcs, nside, nside_tile, t0=t, comm=comm_intra, tag=tag, preprocess_config=args['preprocess_config'], recenter=recenter, dtype_map=args['dtype_map'], dtype_tod=args['dtype_tod'], comps=args['comps'], verbose=args['verbose'], split_labels=split_labels, singlestream=args['singlestream'], det_in_out=args['det_in_out'], det_left_right=args['det_left_right'], det_upper_lower=args['det_upper_lower'], site=args['site'], ext=args['ext'])
                # 6. write them
                write_depth1_map(prefix, mapdata, split_labels=split_labels, )
            except DataMissing as e:
                # This will happen if we decide to abort a map while we are doing the preprocessing.
                handle_empty(prefix, tag, comm_intra, e, L)
                continue
        else:
            mapdata = write_hits_map(context, obslist, subshape, subwcs, nside, nside_tile, t0=t, comm=comm_intra, tag=tag, verbose=args['verbose'],)
            if comm_intra.rank == 0:
                oname = "%s_%s.%s" % (prefix, "full_hits", 'fits')
                if nside is None:
                    enmap.write_map(oname, mapdata.hits)
                else:
                    import healpy as hp
                    hp.write_map(oname, (mapdata.hits).view(args['dtype_map']), nest=True)
    comm.Barrier()
    # gather the tags for writing into the sqlite database
    tags_total = comm.gather(tags, root=0)
    if comm.rank == 0 and not args['only_hits']:
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