from argparse import ArgumentParser
import os
import sys
import time
import warnings
from typing import Union

import numpy as np
import yaml
from pixell import bunch, enmap, fft, mpi, utils, wcsutils
from sotodlib import mapmaking, tod_ops
from sotodlib.core import Context, FlagManager
from sotodlib.site_pipeline import util
import so3g

defaults = {"query": "1",
            "odir": "./outputs",
            "prefix": "map",
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
            "inject": None,
            "nocal": True,
            "nmat_dir": "/nmats",
            "nmat_mode": "build",
            "downsample": 1,
            "maxiter": 500,
            "tiled": 1,
            "wafer": None,
            "interpol": "nearest"
           }

def get_parser(parser: ArgumentParser=None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to mapmaker config.yaml file")
    parser.add_argument("--query", type=str)
    parser.add_argument("--freq", type=str, help="Frequency band. (f090, f150...)")
    parser.add_argument("--area", type=str, help="Path to FITS file describing the mapping geometry")
    parser.add_argument("--odir", type=str, help="Directory for saving output maps")
    parser.add_argument("--prefix", type=str, help="Filename prefix. ({prefix}_sky_map.fits)")
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
    parser.add_argument("-@", "--center-at", type=str)
    parser.add_argument("-w", "--window",  type=float)
    parser.add_argument("-i", "--inject",  type=str)
    parser.add_argument(      "--nocal",   action="store_true", default=True, help="No relcal or abscal")
    parser.add_argument(      "--nmat-dir", type=str, help="Directory to where nmats are loaded from/saved to")
    parser.add_argument(      "--nmat-mode", type=str, help="How to build the noise matrix. 'build': Always build from tod. 'cache': Use if available in nmat-dir, otherwise build and save. 'load': Load from nmat-dir, error if missing. 'save': Build from tod and save.")
    parser.add_argument("-d", "--downsample", type=int, help="Downsample TOD by this factor")
    parser.add_argument(      "--maxiter",    type=int, help="Maximum number of iterative steps")
    parser.add_argument("-T", "--tiled"  ,    type=int)
    parser.add_argument("-W", "--wafer"  ,   type=str, nargs='+', help="Detector wafer subset to map with")
    parser.add_argument("--interpol", type=str)
    return parser


def _get_config(config_file: str) -> dict:
    return yaml.safe_load(open(config_file,'r'))


def _setup_passes(downsample: Union[str, int]="1", maxiter: Union[str, int]="500", interpol: str="nearest") -> bunch.Bunch:
    tmp = bunch.Bunch()
    tmp.downsample = utils.parse_ints(downsample)
    tmp.maxiter = utils.parse_ints(maxiter)
    tmp.interpol = interpol.split(",")
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


def main(config_file: str=None, defaults: dict=defaults, **args) -> None:

    cfg = dict(defaults)

    logger = mapmaking.log.init(level=mapmaking.log.DEBUG, rank=comm.rank)
    # Update the default dict with values provided from a config.yaml file
    if config_file is not None:
        cfg_from_file = _get_config(config_file)
        cfg.update({k: v for k, v in cfg_from_file.items() if v is not None})
    else:
        logger.info("No config file provided, assuming default values")

    # Merge flags from config file and defaults with any passed through CLI
    cfg.update({k: v for k, v in args.items() if v is not None})
    # Certain fields are required. Check if they are all supplied here
    required_fields = ['freq','area','context']
    for req in required_fields:
        if req not in cfg.keys():
            raise KeyError(f"{req} is a required argument. Please supply it in a config file or via the command line")

    args = cfg
    warnings.simplefilter('ignore')
    SITE = args['site']
    verbose = args['verbose'] - args['quiet']
    comm = mpi.COMM_WORLD
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
    nmat_dir = os.path.join(args['odir'],args['nmat_dir'])
    prefix= args['odir'] + "/"
    if args['prefix']: prefix += args['prefix'] + "_"
    utils.mkdir(args['odir'])

    recenter = None
    if args['center_at']:
        recenter = mapmaking.parse_recentering(args['center_at'])

    with mapmaking.mark('context'):
        context = Context(args['context'])

    #ids = context.obsdb.query(args.query)['obs_id']
    ids = mapmaking.get_ids(args['query'], context = context)

    # restrict tod selection further. E.g. --tods [0], --tods[:1], --tods[::100], --tods[[0,1,5,10]], etc.
    if args['tods']:
        ids2 = eval("ids" + args['tods'])
        if type(ids2) != type(ids): 
            ids2 = np.array([ids2])
        ids = ids2
    # This one is just a special case of the much more general one above
    if args['ntod'] is not None:
        ids = ids[:args['ntod']]

    if len(ids) == 0:
        if comm.rank == 0:
            logger.info("No tods found!")
        sys.exit(1)
    logger.info(f"Reading {len(ids)} tods")

    if args['inject']:
        map_to_inject = enmap.read_map(args['inject']).astype(dtype_map)

    passes = _setup_passes(
        downsample=args["downsample"], maxiter=args["maxiter"], interpol=args["interpol"]
    )
    for pass_ind, pass_cfg in enumerate(passes):
        logger.info(
            f"Starting pass {pass_ind + 1}/{len(passes)} maxit {pass_cfg.maxiter} down {pass_cfg.downsample} interp {pass_cfg.interpol}"
        )
        pass_prefix = f"{prefix}pass{pass_ind:03d}_"
        if   args['nmat'] == "uncorr": noise_model = mapmaking.NmatUncorr()
        elif args['nmat'] == "corr":   noise_model = mapmaking.NmatDetvecs(verbose=verbose>1,
                downweight=[1e-4, 0.25, 0.50], window=args['window'])
        else: raise ValueError(f"Unrecognized noise model {args['nmat']}")

        signal_cut = mapmaking.SignalCut(comm, dtype=dtype_tod)
        signal_map = mapmaking.SignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, recenter=recenter, tiled=args['tiled'] > 0)
        signals    = [signal_cut, signal_map]
        mapmaker   = mapmaking.MLMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)

        # First feed our mapmaker with data
        nset_kept_tot = 0
        for ind in range(comm.rank, len(ids), comm.size):
            # Detsets correspond to separate files, so treat them as separate TODs.
            obs_id = ids[ind]
            detsets = context.obsfiledb.get_detsets(obs_id)
            nset_kept = 0

            for detset in detsets:
                if args['nset'] is not None and nset_kept >= args['nset']: continue
                name = f"{obs_id}_{detset}"
                logger.debug(f"Processing {name}")

                # Cut out detector wafers we're not interested in, if args.wafer is specified
                if args['wafer'] is not None:
                    wafer_list = args['wafer']
                    dets_dict = {'dets:wafer_slot':wafer_list}
                else: dets_dict ={}

                dets_dict['band'] = args['freq']
                # Get the resolved list of detectors, to keep it below args.max_dets.
                meta = context.get_meta(obs_id=obs_id, dets=dets_dict)
                dets = meta['dets'].vals
                if args['max_dets'] is not None:
                    meta.restrict('dets', meta['dets'].vals[:args['max_dets']])
                if len(dets) == 0:
                    logger.debug(f"Skipped {name} (no dets left)")
                    continue

                with mapmaking.mark(f"read_obs {name}"):
                    obs = context.get_obs(obs_id=obs_id, meta=meta)

                # Fix boresight
                mapmaking.fix_boresight_glitches(obs)
                # Get our sample rate. Would have been nice to have this available in the axisman
                srate = (obs.samps.count - 1) / (obs.timestamps[-1] - obs.timestamps[0])

                # Add site and weather, since they're not in obs yet
                obs.wrap("weather", np.full(1, "vacuum"))
                obs.wrap("site", np.full(1, "so"))

                # Prepare our data. FFT-truncate for faster fft ops
                obs.restrict("samps", [0, fft.fft_len(obs.samps.count)])

                # Desolope to make it periodic. This should be done *before*
                # dropping to single precision, to avoid unnecessary loss of precision due
                # to potential high offses in the raw tod.
                utils.deslope(obs.signal, w=5, inplace=True)
                obs.signal = obs.signal.astype(dtype_tod)
                if 'flags' not in obs._fields:
                    obs.wrap('flags', FlagManager.for_tod(obs))

                if "glitch_flags" not in obs.flags:
                    obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.signal.shape),
                            [(0,'dets'),(1,'samps')])

                # Optionally skip all the calibration. Useful for sims.
                if not args['nocal']:
                    # Disqualify overly cut detectors
                    good_dets = mapmaking.find_usable_detectors(obs, glitch_flags="glitch_flags")
                    obs.restrict("dets", good_dets)
                    if obs.dets.count == 0:
                        logger.debug(f"Skipped {name} (all dets cut)")
                        continue
                    # Gapfill glitches. This function name isn't the clearest
                    tod_ops.get_gap_fill(obs, flags=obs.flags.glitch_flags, swap=True)
                    # Gain calibration
                    gain  = 1
                    for gtype in ["relcal", "abscal"]:
                        gain *= obs[gtype][:, None]
                    obs.signal *= gain
                    # Fourier-space calibration
                    fsig = fft.rfft(obs.signal)
                    freq = fft.rfftfreq(obs.samps.count, 1/srate)
                    # iir filter
                    iir_filter = tod_ops.filters.iir_filter()(freq, obs)
                    fsig /= iir_filter
                    gain /= iir_filter[0].real # keep track of total gain for our record
                    fsig /= tod_ops.filters.timeconst_filter(None)(freq, obs)
                    fft.irfft(fsig, obs.signal, normalize=True)
                    del fsig

                    # Apply pointing correction.
                    #obs.focal_plane.xi    += obs.boresight_offset.xi
                    #obs.focal_plane.eta   += obs.boresight_offset.eta
                    #obs.focal_plane.gamma += obs.boresight_offset.gamma
                    obs.focal_plane.xi += obs.boresight_offset.dx
                    obs.focal_plane.eta += obs.boresight_offset.dy
                    obs.focal_plane.gamma += obs.boresight_offset.gamma

                # Injecting at this point makes us insensitive to any bias introduced
                # in the earlier steps (mainly from gapfilling). The alternative is
                # to inject it earlier, and then anti-calibrate it
                if args['inject']:
                    mapmaking.inject_map(obs, map_to_inject, recenter=recenter, interpol=args['interpol'])
                utils.deslope(obs.signal, w=5, inplace=True)

                if args['downsample'] != 1:
                    obs = mapmaking.downsample_obs(obs, pass_cfg['downsample'])

                # Maybe load precomputed noise model
                nmat_file = f"{nmat_dir}/nmat_{name}.hdf"
                if args['nmat_mode'] == "load" or args['nmat_mode'] == "cache" and os.path.isfile(nmat_file):
                    logger.info(f"Reading noise model {nmat_file}")
                    nmat = mapmaking.read_nmat(nmat_file)
                else: nmat = None

                # And add it to the mapmaker
                # FIXME: How to handle multipass here?
                with mapmaking.mark(f"add_obs {name}"):
                    signal_estimate = None if pass_ind == 0 else mapmaker.transeval(name, obs, mapmaker_prev, x_prev)
                    mapmaker.add_obs(name, obs, noise_model=nmat, signal_estimate=signal_estimate)

                del signal_estimate
                del obs
                nset_kept += 1

                # Maybe save the noise model we built (only if we actually built one rather than
                # reading one in)
                if args['nmat_mode'] in ["save", "cache"] and nmat is None:
                    logger.info(f"Writing noise model {nmat_file}")
                    utils.mkdir(nmat_dir)
                    mapmaking.write_nmat(nmat_file, mapmaker.data[-1].nmat)
            nset_kept_tot += nset_kept

        nset_kept_tot = comm.allreduce(nset_kept_tot)
        if nset_kept_tot == 0:
            if comm.rank == 0:
                logger.info("All tods failed. Giving up")
            sys.exit(1)

        logger.info("Done building")

        with mapmaking.mark("prepare"):
            mapmaker.prepare()

        logger.info("Done preparing")

        signal_map.write(pass_prefix, "rhs", signal_map.rhs)
        signal_map.write(pass_prefix, "div", signal_map.div)
        signal_map.write(pass_prefix, "bin", enmap.map_mul(signal_map.idiv, signal_map.rhs))

        logger.info("Wrote rhs, div, bin")
        x0 = None if pass_ind == 0 else mapmaker.translate(mapmaker_prev, x_prev)
        t1 = time.time()
        for step in mapmaker.solve(maxiter=pass_cfg['maxiter'], x0=x0):
            t2 = time.time()
            dump = step.i % 10 == 0
            logger.info(f"CG step {step.i:4d} {step.err:15.7e} {t2-t1:8.3f} {'(write)' if dump else ''}")
            if dump:
                for signal, val in zip(signals, step.x):
                    if signal.output:
                        signal.write(prefix, f"map{step.i:4d}", val)
            t1 = time.time()

        logger.info("Done")
        for signal, val in zip(signals, step.x):
            if signal.output:
                signal.write(prefix, "map", val)
        comm.Barrier()

        mapmaker_prev = mapmaker
        x_prev = step.x


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
