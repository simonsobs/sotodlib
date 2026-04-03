# Copyright (c) 2019-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""
    A maximum likelihood mapmaking script for demodulated SAT data.
    This script assumes a previous preprocessing step has been run that produces demodulated TODs, and takes these as input to produce maps.    
"""
import warnings
import argparse
import os
import sys
import time
import yaml

import numpy as np
import so3g

from pixell import utils, enmap, mpi, fft, bunch, wcsutils
from sotodlib.core import Context, LabelAxis, AxisManager # type: ignore
from sotodlib.core.metadata.loader import LoaderError # type: ignore
from sotodlib import mapmaking, preprocess # type: ignore

from scipy.signal import welch

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--query", type=str)
parser.add_argument("--area", type=str)
parser.add_argument("--odir", type=str)
parser.add_argument("prefix", nargs="?")
parser.add_argument(      "--comps",   type=str, default="QU")
parser.add_argument("-W", "--wafers",  type=str, default=None, help="Detector wafer subsets to map with. Can include list separated\
                    by commas, e.g. 'ws0,ws1,ws3'")
parser.add_argument("-B", "--band",    type=str, default=None, help="Bandpass to map")
parser.add_argument("-c", "--context", type=str, default="/so/metadata/satp3/contexts/use_this.yaml", help="Path to Context file")
parser.add_argument(    "--preproc-conf", type=str, default=None, help="Paths to preprocessing configuration file.\
    If providing 2 files, representing 2 layers of preprocessing, they should be separated by a comma.\
    Assumes preprocessing have already been run.")
parser.add_argument(      "--tods",    type=str, default=None, help="Pattern to restrict tod selection further. E.g. --tods [0], --tods [:1],\
    --tods [::100], --tods [[0,1,5,10]]")
parser.add_argument("-n", "--ntod",    type=int, help="Special case of `tods` above. Implemented as follows: [:ntod]")
parser.add_argument("-N", "--nmat",    type=str, default="corr", help="Noise model to use. Options:\
    'white' = white noise, \
    'uncorr' = Noise autocorrelation but no cross-detector correlations, \
    'corr' = Detector correlations, \
    'corr2' = Adds a pre-whitening operator before finding common-modes")
parser.add_argument(      "--max-dets",type=int, default=None, help="Restrict number of detectors to map. Mainly for debug runs.")
parser.add_argument("-S", "--site",    type=str, default="so_sat3")
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument("-q", "--quiet",   action="count", default=0)
parser.add_argument("-@", "--center-at", type=str, default=None)
parser.add_argument("-w", "--window",  type=float, default=0.0)
parser.add_argument("-i", "--inject",  type=str,   default=None)
parser.add_argument(      "--nmat-dir",  type=str, default="{odir}/nmats", help="Directory to save/load noise models")
parser.add_argument(      "--nmat-mode", type=str, default="build", help="How to build the noise matrix. Options:\
    'build': Always build from tod. \
    'cache': Use if available in nmat-dir, otherwise build and save. \
    'load': Load from nmat-dir, error if missing. \
    'save': Build from tod and save.")
parser.add_argument("-d", "--downsample", type=str, default="1", help="Downsample TOD by this factor")
parser.add_argument(      "--maxiter",    type=str, default="500", help="Maximum number of CG iterations")
parser.add_argument(      "--interpol",   type=str, default="nearest", help="Interpolation mode for the pointing matrix")
parser.add_argument("-T", "--tiled"  ,   type=int, default=1, help="Distribute the maps in tiles")
parser.add_argument("--extra_cuts", action="store_true",  default=False, help="Add extra quality cuts to the data after preprocessing to catch any anomalies that may have survived. "
    "These cuts include: "
    "1) Cutting detectors that are overly cut relative to the median cut fraction across detectors, "
    "2) Cutting detectors with unreasonable sensitivity values, "
    "3) Cutting detectors with extreme signal values, "
    "4) Cutting observations with less than 10 detectors after these cuts.")
args = parser.parse_args()

####### Multi-pass setup
def setup_passes(downsample="1", maxiter="500", interpol="nearest"):
    tmp            = bunch.Bunch()
    tmp.downsample = utils.parse_ints(downsample)
    tmp.maxiter    = utils.parse_ints(maxiter)
    tmp.interpol   = interpol.split(",")
    # The entries may have different lengths. We use the max
    # and then pad the others by repeating the last element.
    # The final output will be a list of bunches
    npass        = max([len(tmp[key]) for key in tmp])
    passes       = []
    for i in range(npass):
        entry = bunch.Bunch()
        for key in tmp:
            entry[key] = tmp[key][min(i,len(tmp[key])-1)]
        passes.append(entry)
    return passes

####### Utility functions for restructuring the demodulated tods
def axman_empty_like(iaxman, *newaxes):
    skip  = [ax.name for ax in newaxes]
    axes  = [iaxman[axname] for axname in iaxman._axes if axname not in skip]
    axes += newaxes
    return AxisManager(*axes)

def axman_with_modified_axis(axman, newaxis, op):
    oaxman = axman_empty_like(axman, newaxis)
    for key, axes in axman._assignments.items():
        val = getattr(axman, key)
        if newaxis.name not in axes:
            oaxman.wrap(*mapmaking.get_wrappable(axman, key))
        elif isinstance(val, AxisManager):
            oaxman.wrap(key, axman_with_modified_axis(axman[key], newaxis, op))
        else:
            newval = op(axman[key], name=key, axes=axes)
            oaxman.wrap(key, newval, index_list(axes))
    return oaxman

def index_list(vals): return list(enumerate(vals))

def cutdup(cuts, nmul):
    res = cuts
    for i in range(nmul-1):
        res = res+cuts
    return res

def resample_rfft(ftod, n):
    """Given the last-axis fourier transform ftod of some array tod,
    return the fourier transform of that tod resampled to length n"""
    return ftod[...,:n//2+1].copy()

def prepare_demod_data(obs, frel=1.9, comps="QU"):
    """Downsample demodulated TODs such that its Nyquist frequency is frel*fhwp.
    frel is set to 1.9 by default, which corresponds to the factor used in the ISO preprocessing
    low-pass filter cutoff.
    
    We also reformat the data into a new AxisManager with ncomps as many detectors,
    corresponding to each component sensitive timestream, depending on whether we are solving for T, QU or TQU.
    """
    # First we compute the downsampling factor
    ndet    = obs.dets.count
    insamp  = obs.samps.count
    duration= obs.timestamps[-1]-obs.timestamps[0]
    srate   = insamp/duration
    ifmax   = srate/2
    fhwp    = obs.obs_info.hwp_freq_mean
    ofmax   = frel*fhwp
    down    = utils.nint(ifmax/ofmax)
    
    # Make downsampled obs which we will modify to produce our output
    obs_down= mapmaking.downsample_obs(obs, down, skip_signal=True)
    onsamp  = obs_down.samps.count

    # Reformat the data
    assert comps == "T" or comps == "TQU" or comps == "QU"
    detnames = []
    ndup = 0
    if "T"  in comps:
        detnames.append(np.char.add(obs.dets.vals, "_one"))
        ndup += 1
    if "QU" in comps:
        detnames.append(np.char.add(obs.dets.vals, "_cos"))
        detnames.append(np.char.add(obs.dets.vals, "_sin"))
        ndup += 2
    detaxis = LabelAxis("dets", np.concatenate(detnames))
    def dupdets(val, name, axes):
        if isinstance(val, so3g.proj.ranges.RangesMatrix):
            return so3g.proj.ranges.RangesMatrix(cutdup(val.ranges,ndup))
        else:
            ntile  = np.full(len(axes),1,int)
            ntile[axes.index("dets")] = ndup
            return np.tile(val, ntile)
    obs_demod = axman_with_modified_axis(obs_down, detaxis, dupdets)
    # The demodulated signal
    osignal = np.zeros_like(obs.signal, shape=(detaxis.count, onsamp))
    # Downsample demodulated TODs
    i = 0
    if "T" in comps:
        ftod    = fft.rfft(obs.dsT)
        ftod    = resample_rfft(ftod, onsamp)
        # Fix fourier units to account for the resampling.
        ftod   *= 1/insamp
        fft.irfft(ftod, osignal[i*ndet:(i+1)*ndet])
        # osignal[i*ndet:(i+1)*ndet] = obs.dsT[:,::down] # decimate instead of fft resampling (this could be useful to match the pointing downsampling scheme)
        i      += 1

    if "QU" in comps:
        ftod    = fft.rfft(obs.demodQ)
        ftod    = resample_rfft(ftod,onsamp)
        ftod   *= 1/insamp
        fft.irfft(ftod, osignal[i*ndet:(i+1)*ndet])
        # osignal[i*ndet:(i+1)*ndet] = obs.demodQ[:,::down] # decimate instead of fft resampling (this could be useful to match the pointing downsampling scheme)
        i      += 1

        ftod    = fft.rfft(obs.demodU)
        ftod    = resample_rfft(ftod,onsamp)
        ftod   *= 1/insamp
        fft.irfft(ftod, osignal[i*ndet:(i+1)*ndet])
        # osignal[i*ndet:(i+1)*ndet] = obs.demodU[:,::down] # decimate instead of fft resampling (this could be useful to match the pointing downsampling scheme)
    
    obs_demod.wrap("signal", osignal, [(0,"dets"),(1,"samps")])

    # Construct pointing weights
    # iQ = np.cos(2*obs_demod.focal_plane.gamma[:ndet])
    # iU = np.sin(2*obs_demod.focal_plane.gamma[:ndet])
    obs_demod.focal_plane.move("gamma", None)
    if comps == "TQU":
        oT, oQ, oU = np.zeros((3,3,ndet),dtype_tod)
        # T-detectors have response [1,0,0]
        oT[0], oQ[0], oU[0] = 1,  0,   0
        # cos-detectors have response [0,+detQ,-detU]
        # oT[1], oQ[1], oU[1] = 0, iQ, -iU
        oT[1], oQ[1], oU[1] = 0, 1, 0 # rotation already applied in preprocessing
        # sin-detectors have response [0,+detU,+detQ]
        # oT[2], oQ[2], oU[2] = 0, iU,  iQ
        oT[2], oQ[2], oU[2] = 0, 0,  1 # rotation already applied in preprocessing
        obs_demod.focal_plane.wrap("T", oT.reshape(-1), [(0,"dets")])
    elif comps == "QU":
        oQ, oU = np.zeros((2,2,ndet),dtype_tod)
        # oQ[0], oU[0] = iQ, -iU
        # oQ[1], oU[1] = iU,  iQ
        oQ[0], oU[0] = 1, 0 # rotation already applied in preprocessing
        oQ[1], oU[1] = 0, 1 # rotation already applied in preprocessing
    obs_demod.focal_plane.wrap("Q", oQ.reshape(-1), [(0,"dets")])
    obs_demod.focal_plane.wrap("U", oU.reshape(-1), [(0,"dets")])

    return obs_demod

####### Extra quality cuts utilities
def cut_overly_cut_dets(obs, excess_cut_threshold=0.2, glitch_flags: str = "flags.glitch_flags"):
    """Cut detectors that are overly cut relative to the median cut fraction across detectors. 
    This is to catch any anomalous detectors that may have survived the preprocessing cuts."""
    cut_frac = mapmaking.rangemat_sum(obs[glitch_flags]) / obs.samps.count
    median_cut_frac = np.median(cut_frac)
    median_active_frac = 1 - median_cut_frac
    if median_active_frac < 0.3:
        raise ValueError(f"Median active fraction across detectors is {median_active_frac:.2f}, which is quite low.")
    excess_cut = (cut_frac - median_cut_frac) / median_active_frac
    valid_dets = excess_cut < excess_cut_threshold
    return obs.dets.vals[valid_dets]

def estimate_dsens(obs, wband=[0.5, 1.8]):
    """Estimate detector sensitivty in uK.rt(s) from PSD. Assuming input signal units is K.
    This is expensive in general, but for downsampled SAT data not so much."""
    nperseg = obs.samps.count // 16
    dt = np.median(np.diff(obs.timestamps))
    fs = 1/dt
    signal_muK = obs.signal * 1e6 # convert from K to uK
    freqs, psd = welch(signal_muK, fs=fs, nperseg=nperseg, axis=-1)
    # Select white noise band
    valid_bins = (freqs > wband[0]) & (freqs < wband[1])   
    # White noise level in uK * sqrt(s)
    dsens = np.sqrt(np.median(psd[:, valid_bins], axis=-1))
    return dsens

def cut_bad_dsens(obs, wband = [0.5, 1.8], dsens_low=150, dsens_high=4000):
    """Cut detectors with unreasonable sensitivity values."""
    dsens = estimate_dsens(obs, wband=wband)
    valid_dets = (dsens > dsens_low) & (dsens < dsens_high)
    return obs.dets.vals[valid_dets]

####### Main script
SITE    = args.site
verbose = args.verbose - args.quiet
comm    = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.area)

# Reconstruct that wcs in case default fields have changed; otherwise
# we risk adding information in MPI due to reconstruction, and that
# can cause is_compatible failures.
wcs = wcsutils.WCS(wcs.to_header())

comps = args.comps
ncomp = len(comps)
dtype_tod = np.float32
dtype_map = np.float64
nmat_dir  = args.nmat_dir.format(odir=args.odir)
prefix= args.odir + "/"
if args.prefix: prefix += args.prefix + "_"
utils.mkdir(args.odir)
L = mapmaking.init(level=mapmaking.DEBUG, rank=comm.rank)

recenter = None
if args.center_at:
    recenter = mapmaking.parse_recentering(args.center_at)

with mapmaking.mark('context'):
    context = Context(args.context)

if args.preproc_conf is not None:
    preprocess_config_str = [s.strip() for s in args.preproc_conf.split(",")]
    preprocess_config = []
    for preproc_cf in preprocess_config_str:
        preproc_local = yaml.safe_load(open(preproc_cf, 'r'))
        preprocess_config.append(preproc_local)
else:
    if comm.rank == 0:
        L.info("No preprocessing configuration file provided. Exiting...")
    mpi.Finalize()
    sys.exit(1)

if len(preprocess_config)==1:
    preproc_init = preprocess_config[0]
    preproc_proc = None
else:
    preproc_init = preprocess_config[0]
    preproc_proc = preprocess_config[1]

wafers  = args.wafers.split(",") if args.wafers else None
bands   = args.band  .split(",") if args.band   else None
sub_ids = mapmaking.get_subids(args.query, context=context)
sub_ids = mapmaking.filter_subids(sub_ids, wafers=wafers, bands=bands)

# Restrict tods further if requested
if args.tods:
    sub_ids = eval("sub_ids" + args.tods)
# This one is just a special case of the much more general one above
if args.ntod is not None:
    sub_ids = sub_ids[:args.ntod]

if len(sub_ids) == 0:
    if comm.rank == 0:
        print("No tods found!")
    mpi.Finalize()
    sys.exit(1)
L.info("Reading %d tods" % (len(sub_ids)))

if args.inject:
    map_to_inject = enmap.read_map(args.inject).astype(dtype_map)

# Initializing multipass configuration
passes = setup_passes(downsample=args.downsample, maxiter=args.maxiter, interpol=args.interpol)
mapmaker_prev = None
eval_prev = None
for ipass, passinfo in enumerate(passes):
    L.info("Starting pass %d/%d maxit %d down %d interp %s" % (ipass+1, len(passes), passinfo.maxiter, passinfo.downsample, passinfo.interpol))
    pass_prefix = prefix + "pass%d_" % (ipass+1)
    # Multipass mapmaking
    # Will do multiple passes over the data, using the previous pass both as
    # the initial condition and to temporarily subtract when estimating the noise
    # model. Typically the different passes will differ by sample rate. This means
    # that e.g. SignalCut will be different for each pass, and the degrees of freedom
    # will need to be translated between them.
    if   args.nmat == "white":  
        noise_model = mapmaking.NmatWhite()
    elif args.nmat == "uncorr": 
        noise_model = mapmaking.NmatUncorr(spacing="exp", nbin=100, window=args.window)
    elif args.nmat == "corr":
        noise_model = mapmaking.NmatDetvecs(verbose=verbose>1, window=args.window,
            mode_bins=[1e-4,1.8], bmin_eigvec=5, bin_edges="exp", single_lim=None,
            mp_significance=0.999, wnoise_band=[0.5,1.8], detrend_order=2,
        )
    elif args.nmat == "corr2":
        nmat_uncorr = mapmaking.NmatUncorr()
        nmat_corr   = mapmaking.NmatDetvecs(verbose=verbose>1, window=args.window,
            mode_bins=[1e-4,1e-1,4.0], bmin_eigvec=5, eig_lim=5, bin_edges="exp",
        )
        noise_model = mapmaking.NmatScaledvecs(nmat_uncorr, nmat_corr, window=args.window)
    else: 
        raise ValueError("Unrecognized noise model '%s'" % args.nmat)

    # Set up the pointing matrix and mapmaker
    signal_map = mapmaking.SignalMap(
        shape,
        wcs,
        comm=comm,
        comps=comps,
        dtype=dtype_map,
        recenter=recenter,
        tiled=args.tiled > 0,
        interpol=args.interpol,
    )
    signal_cut  = mapmaking.SignalCut(comm=comm, dtype=dtype_tod)
    signals     = [signal_cut, signal_map]
    mapmaker    = mapmaking.MLMapmaker(
        signals,
        noise_model=noise_model,
        dtype=dtype_tod,
        verbose=verbose>0,
    )

    # Feed our mapmaker with data
    nkept = 0 # this is to keep track of whether any rank has data to process
    # Round-robin assignment of the tods to the MPI ranks
    for ind in range(comm.rank, len(sub_ids), comm.size):
        sub_id = sub_ids[ind]
        obs_id, wafer, band = sub_id.split(":")
        name = sub_id.replace(":", "_")
        L.debug("Processing %s" % name)
        try:
            # meta = context.get_meta(obs_id=obs_id, dets={"dets:wafer_slot":wafer, "wafer.bandpass": band})
            # # Optionally restrict to maximum number of detectors. This is mainly
            # # useful for doing fast debug runs. Before doing this we make sure to
            # # sort the detector list so we chop off a deterministic subset of detectors.
            # meta.restrict("dets", np.sort(meta.dets.vals))
            # dets = meta['dets'].vals
            # if args.max_dets is not None:
            #     meta.restrict('dets', meta['dets'].vals[:args.max_dets])
            # if len(dets) == 0:
            #     L.debug("Skipped %s (no dets left)" % (name))
            #     continue
            # Read and preprocess the data
            with mapmaking.mark("read_and_preprocess_obs %s" % name):
                if len(preprocess_config)==1:
                    # NOTE: Passing with meta fails for now, so we are unable to do direct restrictions on meta hence the commented code above
                    # We instead restrict the obs after loading and preprocessing.
                    obs = preprocess.load_and_preprocess(obs_id, configs=preproc_init, dets={"dets:wafer_slot":wafer, "wafer.bandpass": band})
                else:
                    obs = preprocess.multilayer_load_and_preprocess(obs_id, configs_init=preproc_init, configs_proc=preproc_proc, dets={"dets:wafer_slot":wafer, "wafer.bandpass": band})
            
            # Optionally restrict to maximum number of detectors. This is mainly
            # useful for doing fast debug runs. Before doing this we make sure to
            # sort the detector list so we chop off a deterministic subset of detectors.
            # Note: This should ideally be done before loading the obs, see comments above.
            obs.restrict("dets", np.sort(obs.dets.vals))
            if args.max_dets is not None:
                obs.restrict('dets', obs['dets'].vals[:args.max_dets])
            
            # We add flags if missing.
            if "glitch_flags" not in obs.flags:
                if comm.rank == 0:
                    print("Adding glitch flags by hand")
                obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds + 
                               obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags +
                               obs.preprocess.jumps_slow.jump_flag + obs.preprocess.source_flags.moon +
                               obs.preprocess.noisy_subscan_flags.valid_subscans,)
            
            dets = obs['dets'].vals
            if len(dets) == 0:
                L.debug("Skipped %s (no dets left)" % (name))
                continue

            # We wrap site and weather info into the obs, needed by the mapmaker for pointing transformation
            obs.wrap("weather", np.full(1, "toco"))
            obs.wrap("site",    np.full(1, SITE))
            
            obs = prepare_demod_data(obs, comps=comps, frel=1.9)
            
            # Extra quality cuts to catch any anomalies that survived the preprocessing.
            if args.extra_cuts:
                # - Get rid of overly cut dets
                good_dets = cut_overly_cut_dets(obs, excess_cut_threshold=0.2)
                obs.restrict("dets", good_dets)
                # - Get rid of unreasonable sensitivity dets
                good_dets = cut_bad_dsens(obs, wband=[0.5, 1.8], dsens_low=150, dsens_high=4000)
                obs.restrict("dets", good_dets)
                if obs.dets.count == 0:
                    raise ValueError("Unrealistic sensitivity.")
                # - Get rid of dets with extreme signal values (assuming signal units are K)
                sig_max = np.max(np.abs(np.diff(obs.signal,axis=1)),1)
                good_dets = sig_max < 100
                obs.restrict("dets", obs.dets.vals[good_dets])
                if obs.dets.count == 0:
                    raise ValueError("Extreme signal values.")
                # - Get rid of observations with ndets<100 after these extra cuts
                if obs.dets.count < 100 * ncomp:
                    raise ValueError("ndet < 100 after cuts.")

            # Maybe load precomputed noise model.
            nmat_file = nmat_dir + "/nmat_%s.hdf" % name
            if args.nmat_mode == "load" or args.nmat_mode == "cache" and os.path.isfile(nmat_file):
                print("Reading noise model %s" % nmat_file)
                nmat = mapmaking.read_nmat(nmat_file)
            else: 
                nmat = None

            # And add it to the mapmaker
            with mapmaking.mark("add_obs %s" % name):
                if ipass > 0:
                    # Evaluate the final model of the previous pass' mapmaker
                    # for this observation.
                    signal_estimate = eval_prev.evaluate(mapmaker_prev.data[len(mapmaker.data)])
                    # Resample this to the current downsampling level
                    signal_estimate = mapmaking.resample.resample_fft_simple(signal_estimate, obs.samps.count)
                else: 
                    signal_estimate = None
                # signal_estimate = mapmaker.transeval(name, obs, mapmaker_prev, x_prev) if ipass > 0 else None
                mapmaker.add_obs(name, obs, noise_model=nmat, signal_estimate=signal_estimate, deslope=False)
                del signal_estimate
            del obs
            nkept += 1

            # Maybe save the noise model we built (only if we actually built one rather than
            # reading one in)
            if args.nmat_mode in ["save", "cache"] and nmat is None:
                print("Writing noise model %s" % nmat_file)
                utils.mkdir(nmat_dir)
                mapmaking.write_nmat(nmat_file, mapmaker.data[-1].nmat)
        except (LoaderError,IndexError,ValueError) as e:
            L.debug("Skipped %s (%s)" % (name, str(e)))
            continue

    # If one rank doesn't have any data, the resources allocated need to be reevaluated.
    # We kill the program
    should_exit = (nkept == 0)
    if should_exit:
        L.debug(f"rank {comm.rank} has no data. We shut down the program.")

    # Broadcast exit signal: If any rank has should_exit=True, all should exit
    exit_signal = comm.allreduce(should_exit, op=mpi.LOR)

    if exit_signal:
        mpi.Finalize()
        sys.exit(1)

    L.info("Done building")

    with mapmaking.mark("prepare"):
        mapmaker.prepare()

    L.info("Done preparing")

    signal_map.write(pass_prefix, "rhs", signal_map.rhs)
    if ipass == 0 and "T" in comps:
        # The internal hits map constructor in sotodlib assumes we always solve for T
        # TODO: This should be fixed
        signal_map.write(prefix, "hits", signal_map.hits)
    signal_map.write(pass_prefix, "div", signal_map.div)
    signal_map.write(pass_prefix, "bin", signal_map.precon(signal_map.rhs))
    if ipass == 0:
        L.info("Wrote rhs, hits, div, bin")
    else:
        L.info("Wrote rhs, div, bin")
    
    # Set up initial condition
    x0 = None if ipass == 0 else mapmaker.translate(mapmaker_prev, eval_prev.x_zip)

    t1 = time.time()
    for step in mapmaker.solve(maxiter=passinfo.maxiter, x0=x0, maxerr=1e-6):
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
    eval_prev = mapmaker.evaluator(step.x_zip)
