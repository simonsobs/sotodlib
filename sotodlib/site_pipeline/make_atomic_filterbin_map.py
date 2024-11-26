from argparse import ArgumentParser
import time
import warnings
import os
import logging
import yaml
import multiprocessing
import traceback
import sqlite3
import numpy as np
import so3g
from sotodlib import coords, mapmaking
from sotodlib.core import Context
from sotodlib.io import hk_utils
from sotodlib.preprocess import preprocess_util
from pixell import enmap, utils as putils, bunch
from pixell import wcsutils, colors, memory, mpi
from concurrent.futures import ProcessPoolExecutor, as_completed
import sotodlib.site_pipeline.util as util

defaults = {
    "area": None,
    "nside": None,
    "query": "type == 'obs' and subtype == 'cmb'",
    "odir": "./output",
    "update_delay": None,
    "comps": "TQU",
    "nproc": 1,
    "ntod": None,
    "tods": None,
    "nset": None,
    "wafer": None,
    "freq": None,
    "center_at": None,
    "dec_ref": -40.0,
    "site": 'so_sat3',
    "max_dets": None,  # not implemented yet
    "verbose": 0,
    "quiet": 0,
    "tiled": 0,  # not implemented yet
    "singlestream": False,
    "only_hits": False,
    "det_in_out": False,
    "det_left_right": False,
    "det_upper_lower": False,
    "scan_left_right": False,
    "window": 0.0,  # not implemented yet
    "dtype_tod": 'float32',
    "dtype_map": 'float64',
    "atomic_db": "atomic_maps.db",
    "fixed_time": None,
    "min_dur": None,
    "hk_data_path": None,
    }


def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to mapmaker config.yaml file")
    parser.add_argument("--context", type=str, help='Path to context file')
    parser.add_argument("--preprocess_config", type=str,
                        help='Path to config file to run the\
                        preprocessing pipeline')
    parser.add_argument("--area",
                        help='WCS kernel for rectangular pixels')
    parser.add_argument("--nside",
                        help='Nside for HEALPIX pixels')
    parser.add_argument("--query",
                        help='Query, can be a file (list of obs_id)\
                        or selection string (will select only CMB scans\
                        by default)')
    parser.add_argument("--odir",
                        help='Output directory')
    parser.add_argument('--update_delay', type=int,
                        help="Number of days (unit is days) in the past\
                        to start observation list")
    parser.add_argument("--nproc", type=int,
                        help='Number of procs in\
                        the multiprocessing pool')
    parser.add_argument("--comps", type=str,
                        help="Components to map (TQU by default)")
    parser.add_argument("--singlestream", action="store_true",
                        help="Map without demodulation (e.g. with\
                        a static HWP)")
    parser.add_argument("--only_hits", action="store_true",
                        help='Only create a hits map')

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
    parser.add_argument("--dec_ref", type=float,
                        help="Decl. at which we will calculate the\
                        reference R.A.")
    parser.add_argument("--center_at", type=str)
    parser.add_argument("--max_dets", type=int, )
    parser.add_argument("--fixed_ftime", type=int, )
    parser.add_argument("--min_dur", type=int, )
    parser.add_argument("--site", type=str, )
    parser.add_argument("--verbose", action="count", )
    parser.add_argument("--quiet", action="count", )
    parser.add_argument("--window", type=float, )
    parser.add_argument("--dtype_tod", type=str)
    parser.add_argument("--dtype_map", type=str)
    parser.add_argument("--atomic_db", type=str,
                        help='name of the atomic map database, will be\
                        saved where this script is being run')
    parser.add_argument("--hk_data_path",
                        help='Path to housekeeping data')
    return parser


def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def get_ra_ref(obs, site='so_sat3', dec_ref=-40):
    t_start = obs.obs_info.start_time
    t_stop = obs.obs_info.stop_time
    az = np.arange(
        (obs.obs_info.az_center-1.0*obs.obs_info.az_throw),
        (obs.obs_info.az_center+1.0*obs.obs_info.az_throw),
        1.0)*putils.degree
    el = obs.obs_info.el_center*putils.degree
    csl = so3g.proj.CelestialSightLine.az_el(
        t_start*np.ones(len(az)),
        az, el*np.ones(len(az)),
        site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_start = np.interp(dec_ref*putils.degree, dec_, ra_)
    csl = so3g.proj.CelestialSightLine.az_el(
        t_stop*np.ones(len(az)), az, el*np.ones(len(az)),
        site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_stop = np.interp(dec_ref*putils.degree, dec_, ra_)
    return ra_ref_start, ra_ref_stop


def find_footprint(context, tod, ref_wcs, return_pixboxes=False, pad=1):
    # Measure the pixel bounds of each observation relative to our
    # reference wcs
    pixboxes = []
    my_shape, my_wcs = coords.get_footprint(tod, ref_wcs)
    my_pixbox = enmap.pixbox_of(ref_wcs, my_shape, my_wcs)
    pixboxes.append(my_pixbox)
    if len(pixboxes) == 0:
        raise DataMissing("No usable obs to estimate footprint from")
    pixboxes = np.array(pixboxes)
    # Handle sky wrapping. This assumes cylindrical coordinates
    # with sky-wrapping in the x-direction, and that there's an
    # integer number of pixels around the sky. Could be done more
    # generally, but would be much more involved,
    # and this should be good enough.
    nphi = putils.nint(np.abs(360/ref_wcs.wcs.cdelt[0]))
    widths = pixboxes[:, 1, 0]-pixboxes[:, 0, 0]
    pixboxes[:, 0, 0] = putils.rewind(
        pixboxes[:, 0, 0], ref=pixboxes[0, 0, 0], period=nphi)
    pixboxes[:, 1, 0] = pixboxes[:, 0, 0] + widths
    # It's now safe to find the total pixel bounding box
    union_pixbox = np.array([np.min(pixboxes[:, 0], 0)-pad,
                             np.max(pixboxes[:, 1], 0)
                             + pad])
    # Use this to construct the output geometry
    shape = union_pixbox[1]-union_pixbox[0]
    wcs = ref_wcs.deepcopy()
    wcs.wcs.crpix -= union_pixbox[0, ::-1]
    if return_pixboxes:
        return shape, wcs, pixboxes
    else:
        return shape, wcs


class DataMissing(Exception):
    pass


def get_pwv(obs, data_dir):
    try:
        pwv_info = hk_utils.get_detcosamp_hkaman(
            obs, alias=['pwv'],
            fields=['site.env-radiometer-class.feeds.pwvs.pwv'],
            data_dir=data_dir)
        pwv_all = pwv_info['env-radiometer-class']['env-radiometer-class'][0]
        pwv = np.nanmedian(pwv_all)
    except (KeyError, ValueError):
        pwv = 0.0
    return pwv


def read_tods(context, obslist,
              dtype_tod=np.float32, only_hits=False, site='so_sat3',
              l2_data=None,
              dec_ref=None):
    context = Context(context)
    # this function will run on multiprocessing and can be returned in any
    # random order we will also return the obslist to keep track of the order
    my_tods = []
    my_ra_ref = []
    pwvs = []
    ind = 0
    obs_id, detset, band, obs_ind = obslist[ind]
    meta = context.get_meta(
        obs_id, dets={"wafer_slot": detset, "wafer.bandpass": band})
    tod = context.get_obs(meta, no_signal=True)
    to_remove = []
    for field in tod._fields:
        if field not in ['obs_info', 'flags', 'signal', 'focal_plane', 'timestamps', 'boresight']:
            to_remove.append(field)
    for field in to_remove:
        tod.move(field, None)
    if only_hits is False:
        ra_ref_start, ra_ref_stop = get_ra_ref(tod, site=site, dec_ref=dec_ref)
        my_ra_ref.append((ra_ref_start/putils.degree,
                          ra_ref_stop/putils.degree))
    else:
        my_ra_ref.append(None)
    tod.flags.wrap(
        'glitch_flags', so3g.proj.RangesMatrix.zeros(tod.shape[:2]),
        [(0, 'dets'), (1, 'samps')])
    my_tods.append(tod)

    tod_temp = tod.restrict('dets', meta.dets.vals[:1], in_place=False)
    if l2_data is not None:
        pwvs.append(get_pwv(tod_temp, data_dir=l2_data))
    else:
        pwvs.append(np.nan)
    del tod_temp
    return bunch.Bunch(obslist=obslist, my_tods=my_tods,
                       my_ra_ref=my_ra_ref, pwvs=pwvs)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, colors={'DEBUG': colors.reset,
                                    'INFO': colors.lgreen,
                                    'WARNING': colors.lbrown,
                                    'ERROR': colors.lred,
                                    'CRITICAL': colors.lpurple}):
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


def main(config_file=None, defaults=defaults, **args):
    # Set up logging.
    L = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(
        "%(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter())
    L.addHandler(ch)

    cfg = dict(defaults)
    # Update the default dict with values provided from a config.yaml file
    if config_file is not None:
        cfg_from_file = _get_config(config_file)
        cfg.update({k: v for k, v in cfg_from_file.items() if v is not None})
    else:
        L.error("No config file provided, assuming default values")
    # Merge flags from config file and defaults with any passed through CLI
    cfg.update({k: v for k, v in args.items() if v is not None})
    # Certain fields are required. Check if they are all supplied here
    required_fields = ['context', 'preprocess_config']
    for req in required_fields:
        if req not in cfg.keys():
            raise KeyError("{} is a required argument. Please supply it in a\
            config file or via the command line".format(req))
    args = cfg
    warnings.simplefilter('ignore')
    comm = mpi.FAKE_WORLD  # Fake communicator since we won't use MPI
    verbose = args['verbose'] - args['quiet']
    if args['area'] is not None:
        shape, wcs = enmap.read_map_geometry(args['area'])
        wcs = wcsutils.WCS(wcs.to_header())
    elif args['nside'] is not None:
        pass  # here I will map in healpix
    else:
        L.error('Neither rectangular area or nside specified, exiting.')
        exit(1)

    noise_model = mapmaking.NmatWhite()
    putils.mkdir(args['odir'])

    recenter = None
    if args['center_at']:
        recenter = mapmaking.parse_recentering(args['center_at'])
    preprocess_config = yaml.safe_load(open(args['preprocess_config'], 'r'))
    errlog = os.path.join(os.path.dirname(
        preprocess_config['archive']['index']), 'errlog.txt')

    multiprocessing.set_start_method('spawn')
    if (args['update_delay'] is not None):
        min_ctime = int(time.time()) - args['update_delay']*86400
        args['query'] += f" and timestamp>={min_ctime}"

    context = Context(args['context'])
    # obslists is a dict, obskeys is a list, periods is an array, only rank 0
    # will do this and broadcast to others.
    try:
        obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(
            context, args['query'], nset=args['nset'], wafer=args['wafer'],
            freq=args['freq'], ntod=args['ntod'], tods=args['tods'],
            fixed_time=args['fixed_time'], mindur=args['min_dur'])
    except mapmaking.NoTODFound as err:
        L.exception(err)
        exit(1)
    L.info(f'Done with build_obslists, running {len(obslists)} maps')
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
        split_labels.append('full')

    # We open the data base for checking if we have maps already,
    # if we do we will not run them again.
    if os.path.isfile(args['atomic_db']) and not args['only_hits']:
        # open the connector, in reading mode only
        conn = sqlite3.connect('./'+args['atomic_db'])
        cursor = conn.cursor()
        keys_to_remove = []
        # Now we have obslists and splits ready, we look through the database
        # to remove the maps we already have from it
        for key, value in obslists.items():
            missing_split = False
            for split_label in split_labels:
                query_ = 'SELECT * from atomic where obs_id="%s" and\
                telescope="%s" and freq_channel="%s" and wafer="%s" and\
                split_label="%s"' % (
                    value[0][0], obs_infos[value[0][3]].telescope, key[2],
                    key[1], split_label)
                res = cursor.execute(query_)
                matches = res.fetchall()
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
        conn.close()  # I close since I only wanted to read

    obslists_arr = [item for key, item in obslists.items()]
    tod_list = []  # this list will receive the outputs from read_tods
    L.info('Starting with read_tods')
    with ProcessPoolExecutor(args['nproc']) as exe:
        futures = [exe.submit(
            read_tods, args['context'], obslist, dtype_tod=args['dtype_tod'],
            only_hits=args['only_hits'], l2_data=args['hk_data_path'],
            site=args['site'], dec_ref=args['dec_ref'])
                   for obslist in obslists_arr]
        for future in as_completed(futures):
            try:
                tod_list.append(future.result())
            except Exception as e:
                future_write_to_log(e, errlog)
                continue
            futures.remove(future)
    # flatten the list of lists
    L.info('Done with read_tods')

    my_tods = [bb.my_tods for bb in tod_list]
    if args['area'] is not None:
        subgeoms = []
        for obs in my_tods:
            if recenter is None:
                subshape, subwcs = coords.get_footprint(obs[0], wcs)
                subgeoms.append((subshape, subwcs))
            else:
                subshape = shape
                subwcs = wcs
                subgeoms.append((subshape, subwcs))
    run_list = []
    for oi in range(len(my_tods)):
        # tod_list[oi].obslist[0] is the old obslist
        pid = tod_list[oi].obslist[0][3]
        detset = tod_list[oi].obslist[0][1]
        band = tod_list[oi].obslist[0][2]
        obslist = tod_list[oi].obslist
        t = putils.floor(periods[pid, 0])
        t5 = ("%05d" % t)[:5]
        prefix = "%s/%s/atomic_%010d_%s_%s" % (
            args['odir'], t5, t, detset, band)
        if args['area'] is not None:
            subshape, subwcs = subgeoms[oi]

        tag = "%5d/%d" % (oi+1, len(obskeys))
        putils.mkdir(os.path.dirname(prefix))
        my_ra_ref_atomic = tod_list[oi].my_ra_ref
        pwv_atomic = tod_list[oi].pwvs[0]
        # Save file for data base of atomic maps.
        # We will write an individual file,
        # another script will loop over those files
        # and write into sqlite data base
        if not args['only_hits']:
            info = []
            for split_label in split_labels:
                info.append(bunch.Bunch(
                    pid=pid,
                    obs_id=obslist[0][0].encode(),
                    telescope=obs_infos[obslist[0][3]].telescope.encode(),
                    freq_channel=band.encode(),
                    wafer=detset.encode(),
                    ctime=int(t),
                    split_label=split_label.encode(),
                    # DONT FORGET TO IMPLEMENT THIS
                    split_detail=''.encode(),
                    prefix_path=str(cwd + '/' + prefix + '_%s' %
                                    split_label).encode(),
                    elevation=obs_infos[obslist[0][3]].el_center,
                    azimuth=obs_infos[obslist[0][3]].az_center,
                    RA_ref_start=my_ra_ref_atomic[0][0],
                    RA_ref_stop=my_ra_ref_atomic[0][1],
                    pwv=float(pwv_atomic)))
        # inputs that are unique per atomic map go into run_list
        if args['area'] is not None:
            run_list.append([obslist, subshape, subwcs, info, prefix, t])
        elif args['nside'] is not None:
            run_list.append([obslist, None, None, info, prefix, t])
    # Done with creating run_list
    with ProcessPoolExecutor(args['nproc']) as exe:
        futures = [exe.submit(
            mapmaking.make_demod_map, args['context'], r[0],
            noise_model, r[3], preprocess_config, r[4],
            shape=r[1], wcs=r[2], nside=args['nside'],
            comm=comm, t0=r[5], tag=tag,
            recenter=recenter,
            dtype_map=args['dtype_map'],
            dtype_tod=args['dtype_tod'],
            comps=args['comps'],
            verbose=verbose,
            split_labels=split_labels,
            singlestream=args['singlestream'],
            site=args['site']) for r in run_list]
        for future in as_completed(futures):
            L.info('New future as_completed result')
            try:
                errors, outputs = future.result()
            except Exception as e:
                future_write_to_log(e, errlog)
                continue
            futures.remove(future)
            for ii in range(len(errors)):
                preprocess_util.cleanup_mandb(errors[ii], outputs[ii],
                                              preprocess_config, L)
    L.info("Done")
    return True


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
