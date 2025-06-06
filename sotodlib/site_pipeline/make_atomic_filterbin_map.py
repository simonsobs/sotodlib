from __future__ import annotations
from argparse import ArgumentParser
from typing import Optional, Union, Callable
from dataclasses import dataclass
import time
import datetime as dt
import warnings
import os
import logging
import yaml
import traceback
import ephem

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import numpy as np
import sotodlib.site_pipeline.util as util
from sotodlib import coords, mapmaking
from sotodlib.core import Context
from sotodlib.io import hk_utils
from sotodlib.preprocess import preprocess_util
from sotodlib.utils.procs_pool import get_exec_env
from so3g.proj import coords as so3g_coords
from pixell import enmap, wcsutils, colors, memory
from pixell import utils as putils
from pixell.mpiutils import FakeCommunicator


@dataclass
class Cfg:
    """
    Class to configure make-atomic-filterbin-map
    
    Args
    --------
    context: str
        Path to context file
    preprocess_config: str
        Path to config file(s) to run the preprocessing pipeline.
        If 2 files, representing 2 layers of preprocessing, they
        should be separated by a comma.
    area: str
        WCS kernel for rectangular pixels. Filename to map/geometry
        or valid string for coords.get_wcs_kernel.
    nside: int
        Nside for HEALPIX pixels
    query: str
        Query, can be a file (list of obs_id) or selection string (will select only CMB scans by default)
    odir: str
        Output directory
    update_delay: float
        Number of days in the past to start obs list
    site: str
    hk_data_path: str
        Path to housekeeping data
    nproc: int
        Number of procs for the multiprocessing pool
    atomic_db: str
        Path to the atomic map database
    comps: str
        Components to map, only TQU implemented
    singlestream: bool
        Map without demodulation (e.g. with a static HWP)
    only_hits: bool
        Only create a hits map
    all_splits: bool
        If True, map all implemented splits
    det_in_out: bool
        Make focal plane split: inner vs outer detector
    det_left_right: bool
        Make focal plane split: left vs right detector
    det_upper_lower: bool
        Make focal plane split: upper vs lower detector
    scan_left_right: bool
        Make samples split: left-going vs right going scans
    ntod: int
        Run the first ntod observations in your query
    tods: str
        Run a specific obs
    nset: int
        Run the first nset wafers
    wafer: str
        Run a specific wafer
    freq: str
        Run a specific frequency band
    unit: str
        Unit of data. Default is K
    use_psd: bool
        True by default. Use white noise measured by PSD
        as the weights for mapmaking. Must be provided by
        the preprocessing
    wn_label: str
        Path where to find the white noise per det by the
        preprocessing
    center_at: str
    max_dets: int
    fixed_time: int
    min_dur: int
    verbose: int
    quiet: int
    window: float
    dtype_tod: str
        Data type for timestreams
    dtype_map: str
        Data type for maps
    """
    def __init__(
        self,
        context: str,
        preprocess_config: str,
        area: Optional[str] = None,
        nside: Optional[int] = None,
        query: str = "type == 'obs' and subtype == 'cmb'",
        odir: str = "./output",
        update_delay: Optional[float] = None,
        site: str = 'so_sat3',
        hk_data_path: Optional[str] = None,
        nproc: int = 1,
        atomic_db: Optional[str] = None,
        comps: str = 'TQU',
        singlestream: bool = False,
        only_hits: bool = False,
        all_splits: bool = False,
        det_in_out: bool = False,
        det_left_right: bool = False,
        det_upper_lower: bool = False,
        scan_left_right: bool = False,
        ntod: Optional[int] = None,
        tods: Optional[str] = None,
        nset: Optional[int] = None,
        wafer: Optional[str] = None,
        freq: Optional[str] = None,
        center_at: Optional[str] = None,
        max_dets: Optional[int] = None,
        fixed_time: Optional[int] = None,
        min_dur: Optional[int] = None,
        verbose: int = 0,
        quiet: int = 0,
        window: Optional[float] = None,
        dtype_tod: str = 'float32',
        dtype_map: str = 'float64',
        unit: str = 'K',
        use_psd: bool = True,
        wn_label: str = 'preprocess.noiseQ_mapmaking.white_noise'
    ) -> None:
        self.context = context
        self.preprocess_config = preprocess_config
        self.area = area
        self.nside = nside
        self.query = query
        self.odir = odir
        self.update_delay = update_delay
        self.site = site
        self.hk_data_path = hk_data_path
        self.nproc = nproc
        self.atomic_db = atomic_db
        self.comps = comps
        self.singlestream = singlestream
        self.only_hits = only_hits
        self.all_splits = all_splits
        self.det_in_out = det_in_out
        self.det_left_right = det_left_right
        self.det_upper_lower = det_upper_lower
        self.scan_left_right = scan_left_right
        self.ntod = ntod
        self.tods = tods
        self.nset = nset
        self.wafer = wafer
        self.freq = freq
        self.center_at = center_at
        self.max_dets = max_dets
        self.fixed_time = fixed_time
        self.min_dur = min_dur
        self.verbose = verbose
        self.quiet = quiet
        self.window = window
        self.dtype_tod = dtype_tod
        self.dtype_map = dtype_map
        self.unit = unit
        self.use_psd = use_psd
        self.wn_label = wn_label
    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
            return cls(**d)

class DataMissing(Exception):
    pass

def get_pwv(start_time, stop_time, data_dir):
    try:
        pwv_info = hk_utils.get_hkaman(
            float(start_time), float(stop_time), alias=['pwv'],
            fields=['site.env-radiometer-class.feeds.pwvs.pwv'],
            data_dir=data_dir)
        pwv_all = pwv_info['env-radiometer-class']['env-radiometer-class'][0]
        pwv = np.nanmedian(pwv_all)
    except (KeyError, ValueError):
        pwv = 0.0
    return pwv

def get_sun_distance(site, ctime, az, el):
    site_ = so3g_coords.SITES[site].ephem_observer()
    dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
    site_.date = ephem.Date(dtime)
    sun = ephem.Sun(site_)
    return np.degrees(ephem.separation((sun.az, sun.alt), (np.radians(az), np.radians(el))))

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
            import psutil
            p = psutil.Process(os.getpid())
            self.t0 = p.create_time()
        except ImportError:
            # Otherwise measure from creation of this filter
            self.t0 = time.time()

    def filter(self, record):
        record.rank = self.rank
        record.wtime = time.time() - self.t0
        record.wmins = record.wtime/60.
        record.whours = record.wmins/60.
        record.mem = memory.current()/1024.**3
        record.resmem = memory.resident()/1024.**3
        record.memmax = memory.max()/1024.**3
        return record


def future_write_to_log(e, errlog):
    errmsg = f'{type(e)}: {e}'
    tb = ''.join(traceback.format_tb(e.__traceback__))
    f = open(errlog, 'a')
    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
    f.close()

def main(
    config_file: str,
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable) -> None:
    args = Cfg.from_yaml(config_file)

    # Set up logging.
    L = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(
        "%(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter())
    L.addHandler(ch)

    verbose = args.verbose - args.quiet

    recenter = None
    if args.center_at:
        recenter = mapmaking.parse_recentering(args.center_at)

    if args.area is not None: # Set shape, wcs for rectpix
        try:
            # Load shape, wcs from a map or geometry
            shape, wcs = enmap.read_map_geometry(args.area)
            wcs = wcsutils.WCS(wcs.to_header())
        except FileNotFoundError:
            # See if area is a wcs_kernel string
            try:
                shape = None
                wcs = coords.get_wcs_kernel(args.area)
            except ValueError:
                L.error("'area' not a valid filename or wcs_kernel string")
                exit(1)
        if recenter is not None:
            # wcs_kernel string not allowed for recenter=True
            if shape is None:
                L.error("'area' must be a map geometry file, not wcs_kernel string, to use recenter")
                exit(1)
        else:
            # If not recenter, we set shape=None so get_footprint will be called
            shape = None

    elif args.nside is not None:
        pass  # here I will map in healpix
    else:
        L.error('Neither rectangular area or nside specified, exiting.')
        exit(1)

    noise_model = mapmaking.NmatWhite()
    putils.mkdir(args.odir)

    preprocess_config_str = [s.strip() for s in args.preprocess_config.split(",")]
    preprocess_config = [] ; errlog = []
    for preproc_cf in preprocess_config_str:
        preproc_local = yaml.safe_load(open(preproc_cf, 'r'))
        preprocess_config.append( preproc_local )
        errlog.append( os.path.join(os.path.dirname(
            preproc_local['archive']['index']), 'errlog.txt') )


    if (args.update_delay is not None):
        min_ctime = int(time.time()) - args.update_delay*86400
        args.query += f" and timestamp>={min_ctime}"

    # Check for map data type
    if args.dtype_map == 'float32' or args.dtype_map == 'single':
        warnings.warn("You are using single precision for maps, we advice to use double precision")

    context_obj = Context(args.context)
    # obslists is a dict, obskeys is a list, periods is an array
    try:
        obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(
            context_obj, args.query, nset=args.nset, wafer=args.wafer,
            freq=args.freq, ntod=args.ntod, tods=args.tods,
            fixed_time=args.fixed_time, mindur=args.min_dur)
    except mapmaking.NoTODFound as err:
        L.exception(err)
        exit(1)
    L.info(f'Running {len(obslists)} maps after build_obslists')

    split_labels = []
    if args.all_splits:
        raise ValueError('all_splits not implemented yet')
    if args.det_in_out:
        split_labels.append('det_in')
        split_labels.append('det_out')
    if args.det_left_right:
        split_labels.append('det_left')
        split_labels.append('det_right')
    if args.det_upper_lower:
        split_labels.append('det_upper')
        split_labels.append('det_lower')
    if args.scan_left_right:
        split_labels.append('scan_left')
        split_labels.append('scan_right')
    if not split_labels:
        split_labels.append('full')

    # We open the data base for checking if we have maps already,
    # if we do we will not run them again.
    if isinstance(args.atomic_db, str):
        if os.path.isfile(args.atomic_db) and not args.only_hits:
            engine = create_engine("sqlite:///%s" % args.atomic_db, echo=False)
            Session = sessionmaker(bind=engine)
            session = Session()

            keys_to_remove = []
            # Now we have obslists and splits ready, we look through the database
            # to remove the maps we already have from it
            for key, value in obslists.items():
                missing_split = False
                for split_label in split_labels:
                    query_ = select(mapmaking.AtomicInfo).filter_by(
                        obs_id=value[0][0],
                        telescope=obs_infos[value[0][3]].telescope,
                        freq_channel=key[2],
                        wafer=key[1],
                        split_label=split_label)
                    matches = session.execute(query_).scalars().all()
                    if len(matches) == 0:
                        # this means one of the requested splits is missing
                        # in the data base
                        missing_split = True
                        break
                if missing_split is False:
                    # this means we have all the splits we requested for the
                    # particular obs_id/telescope/freq/wafer
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                obskeys.remove(key)
                del obslists[key]
            engine.dispose()

    obslists_arr = [item for key, item in obslists.items()]

    L.info(f'Running {len(obslists_arr)} maps after removing duplicate maps')

    # clean up lingering files from previous incomplete runs
    if len(preprocess_config)==1:
        policy_dir_init = os.path.join(os.path.dirname(preprocess_config[0]['archive']['policy']['filename']), 'temp')
    else:
        policy_dir_init = os.path.join(os.path.dirname(preprocess_config[0]['archive']['policy']['filename']), 'temp')
        policy_dir_proc = os.path.join(os.path.dirname(preprocess_config[1]['archive']['policy']['filename']), 'temp_proc')
    for obs in obslists_arr:
        obs_id = obs[0][0]
        if len(preprocess_config)==1:
            preprocess_util.cleanup_obs(obs_id, policy_dir_init, errlog[0], preprocess_config[0], subdir='temp', remove=False)
        else:
            preprocess_util.cleanup_obs(obs_id, policy_dir_init, errlog[0], preprocess_config[0], subdir='temp', remove=False)
            preprocess_util.cleanup_obs(obs_id, policy_dir_proc, errlog[1], preprocess_config[1], subdir='temp_proc', remove=False)
    run_list = []
    for oi, ol in enumerate(obslists_arr):
        pid = ol[0][3]
        detset = ol[0][1]
        band = ol[0][2]
        obslist = ol
        t = putils.floor(periods[pid, 0])
        t5 = ("%05d" % t)[:5]
        prefix = "%s/%s/atomic_%010d_%s_%s" % (
            args.odir, t5, t, detset, band)

        tag = "%5d/%d" % (oi+1, len(obskeys))
        putils.mkdir(os.path.dirname(prefix))
        #pwv_atomic = get_pwv(periods[pid, 0], periods[pid, 1], args.hk_data_path)

        # Save file for data base of atomic maps.
        # We will write an individual file,
        # another script will loop over those files
        # and write into sqlite data base
        if not args.only_hits:
            info_list = []
            for split_label in split_labels:
                info = mapmaking.AtomicInfo(
                    obs_id=obslist[0][0],
                    telescope=obs_infos[obslist[0][3]].telescope,
                    freq_channel=band,
                    wafer=detset,
                    ctime=int(t),
                    split_label=split_label
                )
                info.split_detail = ''
                info.prefix_path = str(prefix + '_%s' % split_label)
                info.elevation = obs_infos[obslist[0][3]].el_center
                info.azimuth = obs_infos[obslist[0][3]].az_center
                #info.pwv = float(pwv_atomic)
                info.roll_angle = obs_infos[obslist[0][3]].roll_center
                info.sun_distance = get_sun_distance(args.site, int(t), obs_infos[obslist[0][3]].az_center, obs_infos[obslist[0][3]].el_center)
                info_list.append(info)
        # inputs that are unique per atomic map go into run_list
        if args.area is not None:
            run_list.append([obslist, shape, wcs, info_list, prefix, t, tag])
        elif args.nside is not None:
            run_list.append([obslist, None, None, info_list, prefix, t, tag])

    futures = [executor.submit(
            mapmaking.make_demod_map, args.context, r[0],
            noise_model, r[3], preprocess_config, r[4],
            shape=r[1], wcs=r[2], nside=args.nside,
            comm=FakeCommunicator(), t0=r[5], tag=r[6],
            recenter=recenter,
            dtype_map=args.dtype_map,
            dtype_tod=args.dtype_tod,
            comps=args.comps,
            verbose=verbose,
            split_labels=split_labels,
            singlestream=args.singlestream,
            site=args.site, unit=args.unit,
            use_psd=args.use_psd,
            wn_label=args.wn_label,) for r in run_list]
    for future in as_completed_callable(futures):
        L.info('New future as_completed result')
        try:
            errors, outputs, d_ = future.result()
            for n_split in range(len(split_labels)):
                info3 = object.__new__(mapmaking.AtomicInfo)
                info3.__dict__ = d_[n_split]
                mapmaking.atomic_db_aux(args.atomic_db, info3)
        except Exception as e:
            future_write_to_log(e, errlog)
            continue
        futures.remove(future)
        for ii in range(len(errors)):
            for idx_prepoc in range(len(preprocess_config)):
                if isinstance(outputs[ii][idx_prepoc], dict):
                    preprocess_util.cleanup_mandb(errors[ii], outputs[ii][idx_prepoc], preprocess_config[idx_prepoc], L)
    L.info("Done")
    return True


def get_parser(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        p = ArgumentParser()
    else:
        p = parser
    p.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    p.add_argument(
        "--nprocs", type=int, help="Number of processors to use."
        )
    return p

if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nprocs)
    if rank == 0:
        main(args.config_file, executor, as_completed_callable)
