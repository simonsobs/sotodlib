from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, itertools, multiprocessing, traceback
from sotodlib import coords, mapmaking
from sotodlib.core import Context,  metadata as metadata_core, FlagManager, AxisManager, OffsetAxis
from sotodlib.io import metadata, hk_utils
from sotodlib.site_pipeline import preprocess_tod as pt
from pixell import enmap, utils as putils, fft, bunch, wcsutils, tilemap, colors, memory, mpi
from concurrent.futures import ProcessPoolExecutor, as_completed
import sotodlib.site_pipeline.util as util

defaults = {"area": None,
            "nside": None,
            "query": "type == 'obs' and subtype == 'cmb'",
            "odir": "./output",
            "preprocess_config": None,
            "update-delay": None,
            "comps": "TQU",
            "mode": "per_obs",
            "nproc": 1,
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
            "l2_data_path": "/global/cfs/cdirs/sobs/untracked/data/site/hk",           
           }

def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to mapmaker config.yaml file")
    
    parser.add_argument("--context", type=str,
                        help='context file')
    parser.add_argument("--area",
                        help='wcs kernel')
    parser.add_argument("--nside",
                        help='Nside if you want map in HEALPIX')
    parser.add_argument("--query",
                        help='query, can be a file (list of obs_id) or selection string')
    parser.add_argument("--odir",
                        help='output directory')
    parser.add_argument("--preprocess_config", type=str,
                        help='file with the config file to run the preprocessing pipeline')
    parser.add_argument('--update-delay', type=int,
                        help="Number of days (unit is days) in the past to start observation list.")
    parser.add_argument("--mode", type=str, )
    parser.add_argument("--nproc", type=int, help='Number of procs in the multiprocessing pool')
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
    az = np.arange((obs.obs_info.az_center-0.5*obs.obs_info.az_throw)*putils.degree,
                   (obs.obs_info.az_center+0.5*obs.obs_info.az_throw)*putils.degree, 0.5*putils.degree)
    el = obs.obs_info.el_center*putils.degree
    
    csl = so3g.proj.CelestialSightLine.az_el(t_start*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_start = np.interp(-40*putils.degree, dec_, ra_)
    
    csl = so3g.proj.CelestialSightLine.az_el(t_stop*np.ones(len(az)), az, el*np.ones(len(az)), site=site, weather='toco')
    ra_, dec_ = csl.coords().transpose()[:2]
    ra_ref_stop = np.interp(-40*putils.degree, dec_, ra_)
    return ra_ref_start, ra_ref_stop

def find_footprint(context, tod, ref_wcs, return_pixboxes=False, pad=1):
    # Measure the pixel bounds of each observation relative to our
    # reference wcs
    pixboxes = []
    my_shape, my_wcs = coords.get_footprint(tod, ref_wcs)
    my_pixbox = enmap.pixbox_of(ref_wcs, my_shape, my_wcs)
    pixboxes.append(my_pixbox)
    if len(pixboxes) == 0: raise DataMissing("No usable obs to estimate footprint from")
    pixboxes = np.array(pixboxes)
    # Handle sky wrapping. This assumes cylindrical coordinates with sky-wrapping
    # in the x-direction, and that there's an integer number of pixels around
    # the sky. Could be done more generally, but would be much more involved,
    # and this should be good enough.
    nphi     = putils.nint(np.abs(360/ref_wcs.wcs.cdelt[0]))
    widths   = pixboxes[:,1,0]-pixboxes[:,0,0]
    pixboxes[:,0,0] = putils.rewind(pixboxes[:,0,0],
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

class DataMissing(Exception): pass

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

def read_tods(context, obslist,
              dtype_tod=np.float32, only_hits=False, site='so_sat3',
              l2_data='/global/cfs/cdirs/sobs/untracked/data/site/hk'):
    """
        context : str
        Path to context file
    """
    context = Context(context)
    # this function will run on multiprocessing and can be returned in any random order
    # we will also return the obslist to keep track of the order
    my_obslist = [] ; my_tods = [] ; my_ra_ref = [] ; pwvs = [] 
    inds = range(len(obslist))
    ind = 0
    obs_id, detset, band, obs_ind = obslist[ind]
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
    if only_hits==False:
        ra_ref_start, ra_ref_stop = get_ra_ref(tod)
        my_ra_ref.append((ra_ref_start/putils.degree,
                          ra_ref_stop/putils.degree))
    else:
        my_ra_ref.append(None)
    tod.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(tod.shape[:2]),
               [(0, 'dets'), (1, 'samps')])
    my_tods.append(tod)

    tod_temp = tod.restrict('dets', meta.dets.vals[:1], in_place=False)
    pwvs.append(get_pwv(tod_temp, data_dir=l2_data))
    del tod_temp
    return obslist, my_tods, my_ra_ref, pwvs

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

def handle_empty(prefix, tag, e, L):
    # This happens if we ended up with no valid tods for some reason
    L.info("%s Skipped: %s" % (tag, str(e)))
    putils.mkdir(os.path.dirname(prefix))
    with open(prefix + ".empty", "w") as ofile: ofile.write("\n")

def make_demod_map_dummy(context):
    return None

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

    comm       = mpi.FAKE_WORLD # Fake communicator since we won't use MPI
    
    verbose = args['verbose'] - args['quiet']
    if args['area'] is not None:
        shape, wcs = enmap.read_map_geometry(args['area'])
        wcs        = wcsutils.WCS(wcs.to_header())
    elif args['nside'] is not None:
        nside = int(args['nside'])
    else:
        print('Neither rectangular area or nside specified, exiting.')
        exit(1)

    noise_model = mapmaking.NmatWhite()
    ncomp      = len(args['comps'])
    meta_only  = False
    putils.mkdir(args['odir'])
    
    recenter = None
    if args['center_at']:
        recenter = mapmaking.parse_recentering(args['center_at'])
    
    # Set up logging.
    L   = logging.getLogger(__name__)
    L.setLevel(logging.INFO)
    ch  = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(" %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"))
    ch.addFilter(LogInfoFilter())
    L.addHandler(ch)
    
    if args['preprocess_config'] is not None:
        preprocess_config = yaml.safe_load(open(args['preprocess_config'],'r'))
        outs = []
        errlog = os.path.join(os.path.dirname(preprocess_config['archive']['index']),
                          'errlog.txt')
    else:
        preprocess_config = None
        outs = None
        errlog = None

    multiprocessing.set_start_method('spawn')

    if (args['update_delay'] is not None):
        min_ctime = int(time.time()) - args['update_delay']*86400
        args['query'] += f" and timestamp>={min_ctime}"

    context = Context(args['context'])
    # obslists is a dict, obskeys is a list, periods is an array, only rank 0
    # will do this and broadcast to others.
    try:
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
    except mapmaking.NoTODFound as err:
        print(err)
        exit(1)
    L.info(f'Done with build_obslists, running {len(obslists)} maps')

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

    obslists_arr = [item for key, item in obslists.items()]
    my_oblists=[]; my_tods = []; my_ra_ref=[]; pwvs=[]
    L.info('Starting with read_tods')
    with ProcessPoolExecutor(args['nproc']) as exe:
        futures = [exe.submit(read_tods, args['context'], obslist, 
                             dtype_tod=args['dtype_tod'],
                              only_hits=args['only_hits'],
                              l2_data=args['l2_data_path']) for obslist in obslists_arr]
        for future in as_completed(futures):
            #L.info('New future as_completed result')
            try:
                my_obslist_here, my_tods_here, my_ra_ref_here, pwvs_here = future.result()
                my_oblists.append(my_obslist_here)
                my_tods.append(my_tods_here)
                my_ra_ref.append(my_ra_ref_here)
                pwvs.append(pwvs_here)
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                L.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
    # flatten the list of lists
    my_oblists = list(itertools.chain.from_iterable(my_oblists))
    my_tods = list(itertools.chain.from_iterable(my_tods))
    my_ra_ref = list(itertools.chain.from_iterable(my_ra_ref))
    pwvs = list(itertools.chain.from_iterable(pwvs))
    L.info('Done with read_tods')

    if args['area'] is not None:
        # we will do the profile and footprint here, and then allgather the
        # subshapes and subwcs.This way we don't have to communicate the
        # massive arrays such as timestamps
        subshapes = [] ; subwcses = []
        for obs in my_tods:
            if recenter is None:
                subshape, subwcs = find_footprint(context, obs, wcs,)
                subshapes.append(subshape) ; subwcses.append(subwcs)
            else:
                subshape = shape; subwcs = wcs
                subshapes.append(subshape) ; subwcses.append(subwcs)
        # subshape and subwcs are in the order given by my_oblists

    run_list = []
    for oi in range(len(my_tods)):
        # we will need to build the obskey from my_oblists
        #obs_1722126466_satp1_1111111', 'ws3', 'f150', 0)
        pid = my_oblists[oi][3];
        detset = my_oblists[oi][1];
        band = my_oblists[oi][2];
        #obskey = (pid, detset, band)
        obslist = [my_oblists[oi]]
        t       = putils.floor(periods[pid,0])
        t5      = ("%05d" % t)[:5]
        prefix  = "%s/%s/atomic_%010d_%s_%s" % (args['odir'], t5, t, detset, band)
        
        if args['area'] is not None:
            subshape = subshapes[oi]
            subwcs   = subwcses[oi]

        tag     = "%5d/%d" % (oi+1, len(obskeys))
        putils.mkdir(os.path.dirname(prefix))
        meta_done = os.path.isfile(prefix + "_full_info.hdf")
        maps_done = os.path.isfile(prefix + ".empty") or (
            os.path.isfile(prefix + "_full_map.fits") and
            os.path.isfile(prefix + "_full_ivar.fits") and 
            os.path.isfile(prefix + "_full_hits.fits")
        )
        #L.info("%s Proc period %4d dset %s:%s @%.0f dur %5.2f h with %2d obs" % (tag, pid, detset, band, t, (periods[pid,1]-periods[pid,0])/3600, len(obslist)))

        my_ra_ref_atomic = [my_ra_ref[oi]]
        pwv_atomic = [pwvs[oi]]
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
        # inputs that are unique per atomic map go into run_list
        if args['area'] is not None:
            run_list.append([obslist, subshape, subwcs, info, prefix, t])
        elif args['nside'] is not None:
            run_list.append([obslist, info, prefix, t])
    # Done with creating run_list

    with ProcessPoolExecutor(args['nproc']) as exe:
        if args['area'] is not None:
            futures = [exe.submit(mapmaking.make_demod_map, args['context'], r[0],
                                noise_model, r[3], preprocess_config, r[4],
                                shape=r[1], wcs=r[2],
                                comm = comm, t0=r[5], tag=tag, recenter=recenter,
                                dtype_map=args['dtype_map'],
                                dtype_tod=args['dtype_tod'],
                                comps=args['comps'],
                                verbose=args['verbose'],
                                split_labels=split_labels,
                                singlestream=args['singlestream'],
                                site=args['site']) for r in run_list]
        elif args['nside'] is not None:
            futures = [exe.submit(mapmaking.make_demod_map, args['context'], r[0],
                                noise_model, r[1], preprocess_config, r[2],
                                nside = nside,
                                comm = comm, t0=r[3], tag=tag, recenter=recenter,
                                dtype_map=args['dtype_map'],
                                dtype_tod=args['dtype_tod'],
                                comps=args['comps'],
                                verbose=args['verbose'],
                                split_labels=split_labels,
                                singlestream=args['singlestream'],
                                site=args['site']) for r in run_list]
        for future in as_completed(futures):
            L.info('New future as_completed result')
            try:
                errors, outputs = future.result()
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                L.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            if preprocess_config is not None:
                for ii in range(len(errors)):
                    pt.cleanup_mandb(errors[ii], outputs[ii], preprocess_config, L)
    print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)