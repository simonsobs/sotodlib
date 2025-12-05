import os
import time
import warnings
from argparse import ArgumentParser

import numpy as np
from pixell import bunch, enmap, mpi, tilemap, utils, wcsutils

from sotodlib import mapmaking
from sotodlib.core import Context
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.site_pipeline.utils.depth1_maps_utils import (
    DEFAULTS, DataMissing, LoaderError, _get_config, calibrate_obs,
    commit_depth1_map, commit_depth1_tods, create_mapmaker_config,
    find_footprint, map_to_calculate, read_tods, write_depth1_map)


def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to mapmaker config.yaml file")

    parser.add_argument("--query", type=str)
    parser.add_argument("--area", type=str, help="Path to FITS file describing the mapping geometry")
    parser.add_argument("--odir", type=str, help="Directory for saving output maps")
    parser.add_argument("--preprocess_config", type=str, help="Preprocess configuration file")
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
    parser.add_argument(      "--nmat-dir", type=str, help="Directory to where nmats are loaded from/saved to")
    parser.add_argument(      "--nmat-mode", type=str, help="How to build the noise matrix. 'build': Always build from tod. 'cache': Use if available in nmat-dir, otherwise build and save. 'load': Load from nmat-dir, error if missing. 'save': Build from tod and save.")
    parser.add_argument("-d", "--downsample", type=int, help="Downsample TOD by this factor")
    parser.add_argument(      "--maxiter",    type=int, help="Maximum number of iterative steps")
    parser.add_argument("-T", "--tiled"  ,    type=int)
    parser.add_argument("-W", "--wafer"  ,   type=str, nargs='+', help="Detector wafer subset to map with")
    parser.add_argument(      "--freq" ,  type=str, nargs='+', help="Frequency band to map with")
    parser.add_argument("-g", "--tasks-per-group", type=int, help="number of tasks per group. By default it is 1, but can be higher if you want more than one MPI job working on a depth-1 map, e.g. if you don't have enough memory for so many MPI jobs")
    parser.add_argument("--rhs", action="store_true", default=False, help="Save the rhs maps")
    parser.add_argument("--bin", action="store_true", default=False, help="Save the bin maps")
    parser.add_argument("--srcsamp", type=str, help="Path to mask file where True regions indicate where bright object mitigation should be applied. Mask is in equatorial coordinates. Not tiled, so should be low-res to not waste memory.")
    parser.add_argument("--unit", type=str, help="Unit of the maps")
    parser.add_argument("--min-dets", type=int, help="Minimum number of detectors for an obs (per wafer per freq)")
    return parser

def make_depth1_map(context, obslist, shape, wcs, noise_model, L, preproc, comps="TQU", t0=0, dtype_tod=np.float32, dtype_map=np.float64, comm=mpi.COMM_WORLD, tag="", niter=100, site='so', tiled=0, verbose=0, downsample=1, interpol='nearest', srcsamp_mask=None, unit='K', min_dets=50):
    pre = "" if tag is None else tag + " "
    if comm.rank == 0:
        L.info(f"{pre} Initializing equation system")

    # Set up our mapmaking equation
    signal_cut = mapmaking.SignalCut(comm, dtype=dtype_tod)
    signal_map = mapmaking.SignalMap(shape, wcs, comm, comps=comps, dtype=dtype_map, tiled=tiled>0, interpol=interpol, ofmt="")
    signals = [signal_cut, signal_map]
    if srcsamp_mask is not None:
        signal_srcsamp = mapmaking.SignalSrcsamp(comm, srcsamp_mask, dtype=dtype_tod)
        signals.append(signal_srcsamp)
    mapmaker   = mapmaking.MLMapmaker(signals, noise_model=noise_model, dtype=dtype_tod, verbose=verbose>0)
    if comm.rank == 0:
        L.info(f"{pre} Building RHS")
    time_rhs   = signal_map.rhs*0
    # And feed it with our observations
    nobs_kept  = 0
    for oi in range(len(obslist)):
        obs_id, detset, band = obslist[oi][:3]
        name = f"{obs_id}:{detset}:{band}"
        # Read in the signal too. This seems to read in all the metadata from scratch,
        # which is pointless, but shouldn't cost that much time
        #obs = context.get_obs(obs_id, dets={"wafer_slot":detset, "band":band})
        try:
            obs = pp_util.load_and_preprocess(obs_id, preproc, dets={'wafer_slot':detset,'wafer.bandpass':band}, context=context,)
        except LoaderError:
            # this means the obs is not on the preprocessing db, so we skip it
            continue
        obs = calibrate_obs(obs, band, site=site, nocal=False, unit=unit, min_dets=min_dets)
        if obs is None:
            # this means we skip the full obs on calibrate_obs
            continue
        if downsample != 1:
            obs = mapmaking.downsample_obs(obs, downsample)
        if obs.dets.count == 0:
            print(f'Skipping {obs_id}, {detset}, {band}')
            continue
        # And add it to the mapmaker
        try:
            mapmaker.add_obs(name, obs)
        except KeyError:
            print(f'rank = {comm.rank} {obs_id}, {detset}, {band}')
        # Also build the RHS for the per-pixel timestamp. First
        # make a white noise weighted timestamp per sample timestream
        Nt = np.zeros_like(obs.signal, dtype=dtype_tod)
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
    if nobs_kept == 0:
        raise DataMissing("All data cut")
    mapmaker.prepare()
    # mapmaker doesn't know about time_rhs, so handle it manually
    if signal_map.tiled:
        time_rhs = tilemap.redistribute(time_rhs, comm)
    else:
        time_rhs = utils.allreduce(time_rhs, comm)
    if signal_map.tiled:
        bin_ = tilemap.map_mul(signal_map.idiv,signal_map.rhs)
    else:
        bin_ = enmap.map_mul(signal_map.idiv,signal_map.rhs)
    if comm.rank == 0:
        L.info(f"{pre} Solving")
    t1 = time.time()
    for step in mapmaker.solve(maxiter=niter):
        t2 = time.time()
        if comm.rank == 0:
            L.info(f"{tag} CG step {step.i:5d} {step.err:15.7e} {t2-t1:6.1f} {(t2-t1)/nobs_kept:6.3f}")
        t1 = time.time()
    map_  = step.x[1]
    ivar = signal_map.div[0,0]
    with utils.nowarn():
        tmap = utils.remove_nan(time_rhs / ivar)
    return bunch.Bunch(map=map_, ivar=ivar, tmap=tmap, signal=signal_map, t0=t0, bin=bin_)


def main(config_file, defaults=DEFAULTS, **args):

    dtype_tod = np.float32
    dtype_map = np.float64

    args = create_mapmaker_config(defaults=defaults, config_file=config_file, args=args)

    warnings.simplefilter('ignore')
    # Set up our communicators
    comm = mpi.COMM_WORLD
    comm_intra = comm.Split(comm.rank // args['tasks_per_group'])
    comm_inter = comm.Split(comm.rank  % args['tasks_per_group'])

    SITE = args['site']
    verbose = args['verbose'] - args['quiet']
    _, wcs = enmap.read_map_geometry(args['area'])
    mapcat_settings = {"database_type": args["mapcat_database_type"],
                       "database_name": args["mapcat_database_name"],
                       "depth_one_parent": args["mapcat_depth_one_parent"],}

    # Reconstruct that wcs in case default fields have changed; otherwise
    # we risk adding information in MPI due to reconstruction, and that
    # can cause is_compatible failures.
    wcs = wcsutils.WCS(wcs.to_header())

    comps = args['comps']
    utils.mkdir(args['odir'])

    # Set up logging.
    L = mapmaking.init(level=mapmaking.DEBUG, rank=comm.rank)

    # set up the preprocessing
    preproc = _get_config(args['preprocess_config'])

    with mapmaking.mark('context'):
        context = Context(args['context'])

    srcsamp_mask = enmap.read_map(args['srcsamp']) if args['srcsamp'] else None

    if args['nmat'] == "uncorr":
        noise_model = mapmaking.NmatUncorr()
    elif args['nmat'] == "corr":
        noise_model = mapmaking.NmatDetvecs(verbose=verbose>1, downweight=[1e-4, 0.25, 0.50], window=args['window'])
    else:
        raise ValueError(f"Unrecognized noise model '{args['nmat']}'")

    obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(context,
                                                                     args['query'],
                                                                     mode='depth_1',
                                                                     nset=args['nset'],
                                                                     ntod=args['ntod'],
                                                                     tods=args['tods'],
                                                                     freq=args['freq'],
                                                                     per_tube=True)

    for oi in range(comm_inter.rank, len(obskeys), comm_inter.size):
        pid, detset, band = obskeys[oi]
        obslist = obslists[obskeys[oi]]
        t = utils.floor(periods[pid,0])
        t5 = f"{t:05d}"[:5]
        prefix  = f"{args['odir']}/{t5}/depth1_{t:010d}_{detset}_{band}"
        tag = f"{oi+1:5d}/{len(obskeys)}"
        map_name = f"{t5}/depth1_{t:010d}_{detset}_{band}"

        utils.mkdir(os.path.dirname(prefix))

        if comm_intra.rank == 0:
            L.info(f"{tag} Proc period {pid:4d} dset {detset}:{band} @{t:.0f} dur {(periods[pid,1]-periods[pid,0])/3600:5.2f} h with {len(obslist):2d} obs")
        try:
            # Read in the metadata and use it to determine which tods are
            #    good and estimate how costly each is
            my_tods, my_inds = read_tods(context, obslist, comm=comm_intra, no_signal=True, site=SITE, L=L, min_dets=args['min_dets'])
            my_costs = np.array([tod.samps.count*len(mapmaking.find_usable_detectors(tod, maxcut=0.3)) for tod in my_tods])

            # Prune tods that have no valid detectors
            valid = np.where(my_costs>0)[0]
            my_tods, my_inds, my_costs = [[a[vi] for vi in valid] for a in [my_tods, my_inds, my_costs]]
            all_inds = utils.allgatherv(my_inds, comm_intra)
            all_costs = utils.allgatherv(my_costs, comm_intra)
            if len(all_inds)  == 0:
                raise DataMissing("No valid tods")
            if sum(all_costs) == 0:
                raise DataMissing("No valid detectors in any tods")

            # Estimate the scan profile and footprint. The scan profile can be done
            # with a single task, but that task might not be the first one, so just
            # make it mpi-aware like the footprint stuff
            subshape, subwcs = find_footprint(context, my_tods, wcs, comm=comm_intra)
        except DataMissing as e:
            # This happens if we ended up with no valid tods for some reason
            L.info(f"Skipping {map_name} with {e}")
            continue

        # Redistribute the valid tasks. Tasks with nothing to do don't continue
        # past here.
        my_inds = all_inds[utils.equal_split(all_costs, comm_intra.size)[comm_intra.rank]]
        comm_good = comm_intra.Split(len(my_inds) > 0)
        if len(my_inds) == 0:
            if comm_intra.rank == 0:
                L.info(f"Map {map_name} has not data.")
            continue

        map_calculate = map_to_calculate(map_name=map_name, inds_to_use=my_inds, mapcat_settings=mapcat_settings)
        if not map_calculate:
            if comm_intra.rank == 0:
                L.info(f"Map {map_name} already calculated.")
            continue

        # Write out the depth1 metadata
        if comm_intra.rank == 0:
            tods = commit_depth1_tods(map_name=map_name, obslist=obslist, obs_infos=obs_infos, band=band, inds=my_inds,
                                        mapcat_settings=mapcat_settings)
        try:
            # Make the maps
            mapdata = make_depth1_map(context, [obslist[ind] for ind in my_inds],
                    subshape, subwcs, noise_model, L, preproc, comps=comps, t0=t, comm=comm_good, tag=tag,
                    niter=args['maxiter'], dtype_map=dtype_map, dtype_tod=dtype_tod, site=SITE, tiled=args['tiled']>0,
                    verbose=verbose>0, downsample=args['downsample'], srcsamp_mask=srcsamp_mask, unit=args['unit'],
                    min_dets=args['min_dets'])

            # Write them
            write_depth1_map(prefix, mapdata, dtype=dtype_tod, binned=args['bin'], rhs=args['rhs'], unit=args['unit'])
            if comm_intra.rank == 0 :
                L.info(f"Finished map {map_name}")
                commit_depth1_map(map_name=map_name, prefix=prefix, detset=detset, band=band,
                                  ctime=periods[pid][0],
                                  start_time=periods[pid][0],
                                  stop_time=periods[pid][1],
                                  tods=tods,
                                  mapcat_settings=mapcat_settings)

        except DataMissing as e:
            L.info(f"Skipping {map_name} with {e}")
    return True

if __name__ == '__main__':
    from sotodlib.site_pipeline import util
    util.main_launcher(main, get_parser)
