from argparse import ArgumentParser
from typing import Optional
from dataclasses import dataclass
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
import sotodlib.site_pipeline.util as util
from sotodlib.utils.mp_pool import get_exec_env, as_completed
from sotodlib import coords, mapmaking
from sotodlib.core import Context
from sotodlib.io import hk_utils
from sotodlib.preprocess import preprocess_util
from pixell import enmap, utils as putils, bunch
from pixell import wcsutils, colors, memory
from pixell.fake_communicator import FakeCommunicator


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
        WCS kernel for rectangular pixels
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
        site: str = "so_sat3",
        hk_data_path: Optional[str] = None,
        nproc: int = 1,
        atomic_db: Optional[str] = None,
        comps: str = "TQU",
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
        dtype_tod: str = "float32",
        dtype_map: str = "float64",
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

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
            return cls(**d)


class DataMissing(Exception):
    pass


def get_pwv(obs, data_dir):
    try:
        pwv_info = hk_utils.get_detcosamp_hkaman(
            obs,
            alias=["pwv"],
            fields=["site.env-radiometer-class.feeds.pwvs.pwv"],
            data_dir=data_dir,
        )
        pwv_all = pwv_info["env-radiometer-class"]["env-radiometer-class"][0]
        pwv = np.nanmedian(pwv_all)
    except (KeyError, ValueError):
        pwv = 0.0
    return pwv


def read_tods(
    context,
    obslist,
    dtype_tod=np.float32,
    only_hits=False,
    site="so_sat3",
    l2_data=None,
):
    context = Context(context)
    # this function will run on multiprocessing and can be returned in any
    # random order we will also return the obslist to keep track of the order
    my_tods = []
    pwvs = []
    ind = 0
    obs_id, detset, band, obs_ind = obslist[ind]
    meta = context.get_meta(obs_id, dets={"wafer_slot": detset, "wafer.bandpass": band})
    tod = context.get_obs(meta, no_signal=True)
    to_remove = []
    for field in tod._fields:
        if field not in [
            "obs_info",
            "flags",
            "signal",
            "focal_plane",
            "timestamps",
            "boresight",
        ]:
            to_remove.append(field)
    for field in to_remove:
        tod.move(field, None)
    tod.flags.wrap(
        "glitch_flags",
        so3g.proj.RangesMatrix.zeros(tod.shape[:2]),
        [(0, "dets"), (1, "samps")],
    )
    my_tods.append(tod)

    tod_temp = tod.restrict("dets", meta.dets.vals[:1], in_place=False)
    if l2_data is not None:
        pwvs.append(get_pwv(tod_temp, data_dir=l2_data))
    else:
        pwvs.append(np.nan)
    del tod_temp
    return bunch.Bunch(obslist=obslist, my_tods=my_tods, pwvs=pwvs)


class ColoredFormatter(logging.Formatter):
    def __init__(
        self,
        msg,
        colors={
            "DEBUG": colors.reset,
            "INFO": colors.lgreen,
            "WARNING": colors.lbrown,
            "ERROR": colors.lred,
            "CRITICAL": colors.lpurple,
        },
    ):
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
        record.wmins = record.wtime / 60.0
        record.whours = record.wmins / 60.0
        record.mem = memory.current() / 1024.0**3
        record.resmem = memory.resident() / 1024.0**3
        record.memmax = memory.max() / 1024.0**3
        return record


def future_write_to_log(e, errlog):
    errmsg = f"{type(e)}: {e}"
    tb = "".join(traceback.format_tb(e.__traceback__))
    f = open(errlog, "a")
    f.write(f"\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n")
    f.close()


def main(config_file: str) -> None:

    args = Cfg.from_yaml(config_file)

    rank, executor = get_exec_env(args)
    if rank == 0:
        # Set up logging.
        L = logging.getLogger(__name__)
        L.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            ColoredFormatter("%(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s")
        )
        ch.addFilter(LogInfoFilter())
        L.addHandler(ch)

        verbose = args.verbose - args.quiet
        if args.area is not None:
            shape, wcs = enmap.read_map_geometry(args.area)
            wcs = wcsutils.WCS(wcs.to_header())
        elif args.nside is not None:
            pass  # here I will map in healpix
        else:
            L.error("Neither rectangular area or nside specified, exiting.")
            exit(1)

        noise_model = mapmaking.NmatWhite()
        putils.mkdir(args.odir)

        recenter = None
        if args.center_at:
            recenter = mapmaking.parse_recentering(args.center_at)
        preprocess_config_str = [s.strip() for s in args.preprocess_config.split(",")]
        preprocess_config = []
        errlog = []
        for preproc_cf in preprocess_config_str:
            preproc_local = yaml.safe_load(open(preproc_cf, "r"))
            preprocess_config.append(preproc_local)
            errlog.append(
                os.path.join(
                    os.path.dirname(preproc_local["archive"]["index"]), "errlog.txt"
                )
            )

        if args.update_delay is not None:
            min_ctime = int(time.time()) - args.update_delay * 86400
            args.query += f" and timestamp>={min_ctime}"

        # Check for map data type
        if args.dtype_map == "float32" or args.dtype_map == "single":
            warnings.warn(
                "You are using single precision for maps, we advice to use double precision"
            )

        context_obj = Context(args.context)
        # obslists is a dict, obskeys is a list, periods is an array, only rank 0
        # will do this and broadcast to others.
        try:
            obslists, obskeys, periods, obs_infos = mapmaking.build_obslists(
                context_obj,
                args.query,
                nset=args.nset,
                wafer=args.wafer,
                freq=args.freq,
                ntod=args.ntod,
                tods=args.tods,
                fixed_time=args.fixed_time,
                mindur=args.min_dur,
            )
        except mapmaking.NoTODFound as err:
            L.exception(err)
            exit(1)
        L.info(f"Done with build_obslists, running {len(obslists)} maps")
        cwd = os.getcwd()

        split_labels = []
        if args.all_splits:
            raise ValueError("all_splits not implemented yet")
        if args.det_in_out:
            split_labels.append("det_in")
            split_labels.append("det_out")
        if args.det_left_right:
            split_labels.append("det_left")
            split_labels.append("det_right")
        if args.det_upper_lower:
            split_labels.append("det_upper")
            split_labels.append("det_lower")
        if args.scan_left_right:
            split_labels.append("scan_left")
            split_labels.append("scan_right")
        if not split_labels:
            split_labels.append("full")

        # We open the data base for checking if we have maps already,
        # if we do we will not run them again.
        if isinstance(args.atomic_db, str):
            if os.path.isfile(args.atomic_db) and not args.only_hits:
                # open the connector, in reading mode only
                conn = sqlite3.connect(args.atomic_db)
                cursor = conn.cursor()
                keys_to_remove = []
                # Now we have obslists and splits ready, we look through the database
                # to remove the maps we already have from it
                for key, value in obslists.items():
                    missing_split = False
                    for split_label in split_labels:
                        query_ = (
                            'SELECT * from atomic where obs_id="%s" and\
                        telescope="%s" and freq_channel="%s" and wafer="%s" and\
                        split_label="%s"'
                            % (
                                value[0][0],
                                obs_infos[value[0][3]].telescope,
                                key[2],
                                key[1],
                                split_label,
                            )
                        )
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
        L.info("Starting with read_tods")

        futures = [
            executor.submit(
                read_tods,
                args.context,
                obslist,
                dtype_tod=args.dtype_tod,
                only_hits=args.only_hits,
                l2_data=args.hk_data_path,
                site=args.site,
            )
            for obslist in obslists_arr
        ]
        for future in as_completed(futures):
            try:
                tod_list.append(future.result())
            except Exception as e:
                # if read_tods fails for some reason we log into the first preproc DB
                future_write_to_log(e, errlog[0])
                continue
            futures.remove(future)
        # flatten the list of lists
        L.info("Done with read_tods")

        my_tods = [bb.my_tods for bb in tod_list]
        if args.area is not None:
            subgeoms = []
            for obs in my_tods:
                if recenter is None:
                    subshape, subwcs = coords.get_footprint(obs[0], wcs)
                    subgeoms.append((subshape, subwcs))
                else:
                    subshape = shape
                    subwcs = wcs
                    subgeoms.append((subshape, subwcs))

        # clean up lingering files from previous incomplete runs
        for obs in obslists_arr:
            obs_id = obs[0][0]
            if len(preprocess_config) == 1:
                preprocess_util.save_group_and_cleanup(
                    obs_id, preprocess_config[0], subdir="temp", remove=False
                )
            else:
                preprocess_util.save_group_and_cleanup(
                    obs_id, preprocess_config[0], subdir="temp", remove=False
                )
                preprocess_util.save_group_and_cleanup(
                    obs_id, preprocess_config[1], subdir="temp_proc", remove=False
                )

        run_list = []
        for oi in range(len(my_tods)):
            # tod_list[oi].obslist[0] is the old obslist
            pid = tod_list[oi].obslist[0][3]
            detset = tod_list[oi].obslist[0][1]
            band = tod_list[oi].obslist[0][2]
            obslist = tod_list[oi].obslist
            t = putils.floor(periods[pid, 0])
            t5 = ("%05d" % t)[:5]
            prefix = "%s/%s/atomic_%010d_%s_%s" % (args.odir, t5, t, detset, band)
            if args.area is not None:
                subshape, subwcs = subgeoms[oi]

            tag = "%5d/%d" % (oi + 1, len(obskeys))
            putils.mkdir(os.path.dirname(prefix))
            pwv_atomic = tod_list[oi].pwvs[0]
            # Save file for data base of atomic maps.
            # We will write an individual file,
            # another script will loop over those files
            # and write into sqlite data base
            if not args.only_hits:
                info = []
                for split_label in split_labels:
                    info.append(
                        bunch.Bunch(
                            pid=pid,
                            obs_id=obslist[0][0].encode(),
                            telescope=obs_infos[obslist[0][3]].telescope.encode(),
                            freq_channel=band.encode(),
                            wafer=detset.encode(),
                            ctime=int(t),
                            split_label=split_label.encode(),
                            split_detail="".encode(),
                            prefix_path=str(
                                cwd + "/" + prefix + "_%s" % split_label
                            ).encode(),
                            elevation=obs_infos[obslist[0][3]].el_center,
                            azimuth=obs_infos[obslist[0][3]].az_center,
                            pwv=float(pwv_atomic),
                        )
                    )
            # inputs that are unique per atomic map go into run_list
            if args.area is not None:
                run_list.append([obslist, subshape, subwcs, info, prefix, t])
            elif args.nside is not None:
                run_list.append([obslist, None, None, info, prefix, t])
        # Done with creating run_list

        futures = [
            executor.submit(
                mapmaking.make_demod_map,
                args.context,
                r[0],
                noise_model,
                r[3],
                preprocess_config,
                r[4],
                shape=r[1],
                wcs=r[2],
                nside=args.nside,
                comm=FakeCommunicator(),
                t0=r[5],
                tag=tag,
                recenter=recenter,
                dtype_map=args.dtype_map,
                dtype_tod=args.dtype_tod,
                comps=args.comps,
                verbose=verbose,
                split_labels=split_labels,
                singlestream=args.singlestream,
                site=args.site,
            )
            for r in run_list
        ]
        for future in as_completed(futures):
            L.info("New future as_completed result")
            try:
                errors, outputs = future.result()
            except Exception as e:
                future_write_to_log(e, errlog)
                continue
            futures.remove(future)
            for ii in range(len(errors)):
                for idx_prepoc in range(len(preprocess_config)):
                    preprocess_util.cleanup_mandb(
                        errors[ii],
                        outputs[ii][idx_prepoc],
                        preprocess_config[idx_prepoc],
                        L,
                    )
        L.info("Done")
    return True


def get_parser(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        p = ArgumentParser()
    else:
        p = parser
    p.add_argument("--config_file", type=str, help="yaml file with configuration.")
    return p


if __name__ == "__main__":
    util.main_launcher(main, get_parser)
