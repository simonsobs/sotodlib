from dataclasses import dataclass, fields, asdict, field
from typing import Literal, Any, Dict, Union, Optional, List, Tuple
import datetime as dt
import yaml
import pandas as pd
import requests
from tqdm import tqdm
from io import StringIO
import json
import h5py
import logging
import numpy as np
import sys
from influxdb import InfluxDBClient
from collections import defaultdict
import traceback

from sotodlib.io import hkdb
from sotodlib.io.hkdb import HkConfig
from sotodlib.core import Context
from sotodlib.core.metadata import ManifestDb
from pixell import enmap, enplot

from sotodlib.io.ancil.pwv import (
    params_250701,
    _gaussian,
    defeature_toco_250701,
    apex_to_tocolin_250701,
)


logger = logging.getLogger(__name__)

for h in logger.handlers[:]:
    logger.removeHandler(h)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class ReportDataConfig:
    def __init__(
        self,
        platform: Literal["satp1", "satp2", "satp3", "lat"],
        site_url: str,
        ctx_path: str,
        start_time: Union[dt.datetime, float, str],
        stop_time: Union[dt.datetime, float, str],
        hk_cfg: Union[HkConfig, str, Dict[str, Any]],
        load_hkdb: bool = True,
        buffer_time: float = 3600,
        influx_client_kw: Optional[Dict[str, Any]] = None,
        longterm_obs_file: Optional[str] = None,
        preprocess_sourcedb_path: Optional[str] = None,
        load_source_footprints: bool = True,
        make_cov_map: bool = True,
        cal_targets: Optional[List[str]] = None,
        show_hk_pb: bool = False,
        noise_scale_factor: float = 1
    ) -> None:
        self.ctx_path: str = ctx_path
        self.platform: Literal["satp1", "satp2", "satp3", "lat"] = platform
        self.site_url: str = site_url
        self.buffer_time: float = buffer_time
        self.longterm_obs_file: Optional[str] = longterm_obs_file
        self.preprocess_sourcedb_path: Optional[str] = preprocess_sourcedb_path
        self.load_source_footprints: bool = load_source_footprints
        self.load_hkdb: bool = load_hkdb
        self.show_hk_pb: bool = show_hk_pb
        self.make_cov_map = make_cov_map
        self.noise_scale_factor = noise_scale_factor

        if cal_targets is None:
            self.cal_targets = ["jupiter", "saturn", "tau_A", "tauA", "cenA", "mars"]
        else:
            self.cal_targets = cal_targets

        if isinstance(start_time, float):
            self.start_time: dt.datetime = dt.datetime.fromtimestamp(start_time)
        elif isinstance(start_time, str):
            self.start_time = dt.datetime.fromisoformat(start_time)
        else:
            self.start_time = start_time

        if isinstance(stop_time, float):
            self.stop_time: dt.datetime = dt.datetime.fromtimestamp(stop_time)
        elif isinstance(stop_time, str):
            self.stop_time = dt.datetime.fromisoformat(stop_time)
        else:
            self.stop_time = stop_time

        self.hk_cfg = hk_cfg

        if influx_client_kw is None:
            self.influx_client_kw: Dict[str, Any] = {}
        else:
            self.influx_client_kw = influx_client_kw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportDataConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Dict]) -> "ReportDataConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass
class ObsInfo:
    obs_id: str
    start_time: float
    stop_time: float
    duration: float
    wafer_slots_list: str
    stream_ids_list: str
    obs_type: str
    obs_subtype: str
    obs_tube_slot: str
    obs_tags: str = ""
    pwv: float = np.nan
    temp: float = np.nan
    uv: float = np.nan
    wind_speed: float = np.nan
    wind_dir: float = np.nan
    el_center: float = np.nan
    boresight: float = np.nan
    hwp_freq_mean: float = np.nan
    num_valid_dets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    array_nep: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    det_nep: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @classmethod
    def from_obsdb_entry(cls, data) -> "ObsInfo":
        get = data.get

        return cls(
            obs_id=get("obs_id"),
            start_time=get("start_time"),
            stop_time=get("stop_time"),
            duration=get("duration"),
            wafer_slots_list=get("wafer_slots_list"),
            stream_ids_list=get("stream_ids_list"),
            obs_type=get("type"),
            obs_subtype=get("subtype"),
            obs_tube_slot=get("tube_slot"),

            boresight=(
                -get("roll_center")
                if get("roll_center") is not None
                else np.nan
            ),

            el_center=get("el_center") or np.nan,
            hwp_freq_mean=get("hwp_freq_mean") or np.nan,
        )

    def __repr__(self):
        inner = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        return f"ObsInfo({inner})"


def get_apex_data(cfg: ReportDataConfig):
    """
    Load APEX pwv data within the time range in the ReportDataConfig.
    """
    APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'

    request = requests.post(APEX_DATA_URL, data={
            'wdbo': 'csv/download',
            'max_rows_returned': 999999,
            'start_date': cfg.start_time.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
                + cfg.stop_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'tab_pwv': 'on',
            'shutter': 'SHUTTER_OPEN',
            #'tab_shutter': 'on',
        })

    # check if any data was found
    lines = request.text.splitlines()
    found_data = False
    for line in lines:
        l = line.strip()
        if l and not l.startswith("#"):
            found_data = True
            break

    if not found_data:
        return None

    def date_converter(d):
        if isinstance(d, bytes):
            naive_dt = dt.datetime.fromisoformat(d.decode('utf-8'))
        else:
            naive_dt = dt.datetime.fromisoformat(d)
        return naive_dt.replace(tzinfo=dt.timezone.utc)

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', dt.datetime), ('pwv', float)],
    )

    outdata = {'timestamps':[d.timestamp() for d in data['dates']],
               'pwv':data['pwv']}
    return outdata


def load_hkdb(cfg: ReportDataConfig) -> hkdb.HkResult:
    """
    Load hkdb data from the range specified in the ReportDataConfig.
    Uses hk_cfg file or dict specified in the config.
    """
    if isinstance(cfg.hk_cfg, str):
        hk_cfg: HkConfig = HkConfig.from_yaml(cfg.hk_cfg)
    elif isinstance(cfg.hk_cfg, dict):
        hk_cfg = HkConfig.from_dict(cfg.hk_cfg)
    else:
        hk_cfg = cfg.hk_cfg

    result = hkdb.load_hk(
        hkdb.LoadSpec(
            cfg=hk_cfg,
            start=cfg.start_time.timestamp() - cfg.buffer_time,
            end=cfg.stop_time.timestamp() + cfg.buffer_time,
            fields=["pwv", "temp", "uv", "wind_spd", "wind_dir"],
        ),
        show_pb=cfg.show_hk_pb,
    )

    return result.data


def load_pwv(cfg: ReportDataConfig) -> hkdb.HkResult:
    """
    Load PWV data from the range specified in the ReportDataConfig.
    Uses hk_cfg file or dict specified in the config.
    """
    if isinstance(cfg.hk_cfg, str):
        hk_cfg: HkConfig = HkConfig.from_yaml(cfg.hk_cfg)
    elif isinstance(cfg.hk_cfg, dict):
        hk_cfg = HkConfig.from_dict(cfg.hk_cfg)
    else:
        hk_cfg = cfg.hk_cfg

    result = hkdb.load_hk(
        hkdb.LoadSpec(
            cfg=hk_cfg,
            start=cfg.start_time.timestamp() - cfg.buffer_time,
            end=cfg.stop_time.timestamp() + cfg.buffer_time,
            fields=["pwv"],
        ),
        show_pb=cfg.show_hk_pb,
    )

    if "env-radiometer-class.pwvs.pwv" in result.data.keys():
        return result.data["env-radiometer-class.pwvs.pwv"]
    else:
        return None


def get_and_merge_apex_pwv(cfg: ReportDataConfig, toco_pwv=None):
    """
    Load APEX PWV measurements and merge with TOCO PWV when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        (timestamps, pwv) sorted by timestamp.
    """
    try:
        apex = get_apex_data(cfg)
    except Exception as e:
        errmsg = f"{type(e)}: {e}"
        tb = "".join(traceback.format_tb(e.__traceback__))
        logger.error(f"get_apex_data failed with {errmsg}\n{tb}")
        apex = None

    if apex is not None:
        times = np.asarray(apex["timestamps"])
        pwv = np.asarray(apex["pwv"])

        m = pwv < 999
        apex = (
            times[m],
            apex_to_tocolin_250701(pwv[m]),
        )

    if toco_pwv is not None:
        times = np.asarray(toco_pwv[0])
        pwv = np.asarray(toco_pwv[1])

        m = (pwv > 0) & (pwv <= 3)
        toco_pwv = (
            times[m],
            defeature_toco_250701(pwv[m]),
        )

    if apex is None and toco_pwv is None:
        return None

    if apex is None:
        return toco_pwv

    if toco_pwv is None:
        return apex

    times = np.concatenate([toco_pwv[0], apex[0]])
    pwv = np.concatenate([toco_pwv[1], apex[1]])

    order = np.argsort(times)

    return times[order], pwv[order]


def load_influx_data(cfg: ReportDataConfig) -> pd.DataFrame:
    """
    Loads InfluxDb data.

    """
    client = InfluxDBClient(**cfg.influx_client_kw)
    buff_time = dt.timedelta(seconds=cfg.buffer_time)

    t0_str = (cfg.start_time - buff_time).isoformat().replace("+00:00", "Z")
    t1_str = (cfg.stop_time + buff_time).isoformat().replace("+00:00", "Z")

    keys = ['time', 'num_valid_dets', 'bandpass', 'wafer_slot', 'tel_tube']
    if cfg.platform == "lat":
        keys += ['array_noise_T', 'det_noise_T']
    elif cfg.platform in ['satp1', 'satp2', 'satp3']:
        keys += [f"{prefix}_{suffix}" for prefix in ["array_noise", "det_noise"] for suffix in ["T", "Q", "U"]]

    query = f"""
        SELECT """ + ", ".join(keys) +  f""" from "autogen"."preprocesstod" WHERE (
            "telescope"::tag = '{cfg.platform}'
            AND time >= '{t0_str}'
            AND time <= '{t1_str}'
        )
    """

    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    missing_keys = [key for key in keys if key not in df]

    if missing_keys:
        logger.warn(f"missing keys from influxdb: {missing_keys}")
        return None

    df["time"] = pd.to_datetime(df["time"])
    df["timestamp"] = df["time"].apply(lambda x: x.timestamp())

    return df


def merge_influx_and_obs_list(df: pd.DataFrame, obs_list: List[ObsInfo], platform: str, noise_scale_factor: float) -> None:
    # Only CMB obs will have preprocessing data
    timestamps = np.array([o.start_time for o in obs_list if o.obs_subtype=='cmb'])
    tube_slots = np.array([f"lat{o.obs_tube_slot}" if platform=="lat" else platform for o in obs_list if o.obs_subtype=='cmb'])
    obsids = np.array([o.obs_id for o in obs_list if o.obs_subtype=='cmb'])

    def find_obsid(ts: float, tel_tube: str) -> Optional[str]:
        m = tube_slots == tel_tube
        if not np.any(m):
            return ""
        idx = np.argmin(np.abs(ts - timestamps[m]))
        if np.isclose(ts, timestamps[m][idx], atol=1e-6):
            return obsids[m][idx]
        else:
            return ""

    def det_nep_agg(nep_series, N_series):
        mask = np.isfinite(nep_series) & (nep_series > 0) & np.isfinite(N_series) & (N_series > 0)
        if not np.any(mask):
            return np.nan
        N = N_series[mask]
        nep = nep_series[mask]
        return np.sqrt(np.sum(N) / np.sum(N / nep**2))

    def array_nep_agg(nep_series):
        mask = np.isfinite(nep_series) & (nep_series > 0)
        if not np.any(mask):
            return np.nan
        nep = nep_series[mask]
        return 1 / np.sqrt(np.sum(1 / nep**2))

    df["obs_id"] = df.apply(
        lambda row: find_obsid(row["timestamp"], row["tel_tube"]),
        axis=1,
    )

    det_nep_cols = [c for c in df.columns if "det_noise_" in c]
    array_nep_cols = [c for c in df.columns if "array_noise_" in c]
    agg_dict = {"num_valid_dets": "sum"}

    for c in det_nep_cols:
        agg_dict[c] = lambda s, n= "num_valid_dets": det_nep_agg(s, df.loc[s.index, n])

    for c in array_nep_cols:
        agg_dict[c] = array_nep_agg

    totals = (
        df.groupby(["obs_id", "bandpass"], as_index=False)
          .agg(agg_dict)
    )

    totals_dict = {}

    for obs_id, g in totals.groupby("obs_id"):
        band_dict = {}

        for row in g.itertuples(index=False):
            band = getattr(row, "bandpass")
            band_dict[band] = {
                c: getattr(row, c)
                for c in agg_dict.keys()
            }

        totals_dict[obs_id] = band_dict

    obs_lookup = {o.obs_id: o for o in obs_list}

    for obs_id, band_totals in totals_dict.items():
        band_totals.pop("NC", None)

        if not obs_id:
            continue

        obs_entry = obs_lookup[obs_id]

        det_dtype = [(band, "i4") for band in band_totals]
        det_row = tuple(vals["num_valid_dets"] for vals in band_totals.values())
        obs_entry.num_valid_dets = np.array([det_row], dtype=det_dtype)

        for key, attr_name in zip(
            ["array_noise_", "det_noise_"],
            ["array_nep", "det_nep"],
        ):

            nep_dtype = []
            nep_row = []

            for band, vals in band_totals.items():

                band_keys = sorted(
                    k for k in vals
                    if k.startswith(key)
                )

                nep_dtype.append(
                    (band, [(k, "f8") for k in band_keys])
                )

                nep_row.append(
                    tuple(
                        noise_scale_factor * vals[k]
                        for k in band_keys
                    )
                )

            setattr(
                obs_entry,
                attr_name,
                np.array([tuple(nep_row)], dtype=nep_dtype),
            )


def populate_hkdb_fields(hkdb_data, pwv, obs_list):
    """
    Attach weather + PWV data to obs_list.
    """

    fields = {
        "temp": ("env-vantage.weather_data.temp_outside", None),
        "uv": ("env-vantage.weather_data.UV", None),
        "wind_speed": ("env-vantage.weather_data.wind_speed", None),
        "wind_dir": ("env-vantage.weather_data.wind_dir", None),
        "pwv": (None, lambda v: (-0.1 < v) & (v < 4.0)),
    }

    def compute_timeseries(times, values, obs):
        m = (times >= obs.start_time) & (times <= obs.stop_time)
        if not np.any(m):
            return np.nan
        return np.nanmean(values[m])

    for attr, (key, post_mask) in fields.items():

        if attr == "pwv":
            if pwv is None:
                logger.warning("\tPWV data not found")
                for o in obs_list:
                    o.pwv = np.nan
                continue

            t, v = pwv
            t = np.asarray(t)
            v = np.asarray(v)

            for o in obs_list:
                val = compute_timeseries(t, v, o)
                if np.isfinite(val) and post_mask(val):
                    o.pwv = val
                else:
                    o.pwv = np.nan

            continue

        if hkdb_data is None or key not in hkdb_data:
            logger.warning(f"\t{attr} data not found")
            for o in obs_list:
                setattr(o, attr, np.nan)
            continue

        t, v = hkdb_data[key]
        t = np.asarray(t)
        v = np.asarray(v)

        for o in obs_list:
            setattr(o, attr, compute_timeseries(t, v, o))


@dataclass
class Footprint:
    """
    Class to contain calibration footprint information.

    Attributes
    -------------
    obs_id: str
        Observation ID of the scan.
    target: str
        Calibration target of the footprint.
    xi_p: np.ndarray
        Xi of the source relative to the boresight over the course of the scan.
    eta_p: np.ndarray
        Eta of the source relative to the boresight over the course of the scan.
    bounds: np.ndarray
        Array of coordinates that define the boundary of the footprint.
        This is what will be plotted, as the full source coordinates is too much
        data. This is computed by taking the alpha-shape of the source
        coordinates.
    """

    wafer: str
    source: str
    count: int
    obsids: List[str]

    def to_h5(self, group: h5py.Group):
        """
        Store a footprint in an hdf5 group.
        """
        group.attrs["target"] = self.target
        group.attrs["obs_id"] = self.obs_id
        group.create_dataset("wafers", data=','.join(wafers.astype(str)))

    @classmethod
    def from_h5(cls, group: h5py.Group) -> "Footprint":
        """Convert from an hdf5 group into a footprint object"""
        fp = cls(
            obs_id=group.attrs["obs_id"],
            target=group.attrs["target"],
            wafers=np.array(group["wafers"]),
        )
        return fp


@dataclass
class ReportData:
    """
    High-level summary data of the region of time specified in the ReportConfig.

    Attributes
    ------------
    cfg: ReportConfig
        Configuration object used to generate the report
    obs_list: List[ObsInfo]
        List of observation info for each observation in the configured time range.
        This contains data from the ObsDb, HkDb, and InfluxDb database.
    pwv: np.ndarray
        PWV throughout the specified time range. This is a 2D array where
        the first element is an array of timestamps, and the second element is
        an array of PWV values. This is pulled from the HKDB.
    cal_footprint: List[Footprint]
        List of Footprint objects which describe source footprint found during

    """

    cfg: ReportDataConfig
    obs_list: List[ObsInfo]
    pwv: Optional[np.ndarray] = None
    source_footprints: Optional[List[Footprint]] = None
    w: Optional[enmap.ndmap] = None
    longterm_obs_df: Optional[pd.DataFrame] = None


    @classmethod
    def build(cls, cfg: ReportDataConfig) -> "ReportData":
        ctx = Context(cfg.ctx_path)

        logger.info("Building obs List")
        rows = ctx.obsdb.query(
            f"start_time >= {cfg.start_time.timestamp()} and "
            f"start_time <= {cfg.stop_time.timestamp()}"
        )

        # Get tags
        obs_tags = defaultdict(list)
        for obs_id, tag in ctx.obsdb.conn.execute("SELECT obs_id, tag FROM tags"):
            obs_tags[obs_id].append(tag)

        obs_list = [None] * len(rows)
        for i, o in enumerate(rows):
            obs_list[i] = ObsInfo.from_obsdb_entry(o)
            obs_list[i].obs_tags = ",".join(obs_tags.get(o["obs_id"], []))
            # obs_list[i].obs_tags = ",".join(ctx.obsdb.get(o['obs_id'], tags=True)['tags'])

        if cfg.longterm_obs_file is not None:
            logger.info("Getting longterm data")
            longterm_obs_df = cls.load(cfg.longterm_obs_file)
        else:
            longterm_obs_df = None

        # hkdb
        if cfg.load_hkdb:
            try:
                logger.info("Getting hkdb data")
                hkdb_data = load_hkdb(cfg)
            except Exception as e:
                logger.error(f"load_hkdb failed with {e}")
                hkdb_data = None
        else:
            hkdb_data = None

        # Toco PWV
        if (
            hkdb_data is not None and
            "env-radiometer-class.pwvs.pwv" in hkdb_data.keys()
        ):
            toco_pwv = hkdb_data["env-radiometer-class.pwvs.pwv"]
        else:
            toco_pwv = None

        logger.info("Merging Toco and APEX PWV")
        pwv = get_and_merge_apex_pwv(cfg, toco_pwv)

        logger.info("Loading Influx data")
        influx_df = load_influx_data(cfg)

        if influx_df is not None:
            logger.info("Merging InfluxDb data with obs list")
            merge_influx_and_obs_list(influx_df, obs_list, cfg.platform, cfg.noise_scale_factor)
        else:
            logger.warn("InfluxDb data not found")

        # Get hkdb params for each obs
        populate_hkdb_fields(hkdb_data, pwv, obs_list)

        data: "ReportData" = cls(
            cfg=cfg, obs_list=obs_list, pwv=pwv, longterm_obs_df=longterm_obs_df
        )

        if cfg.load_source_footprints:
            logger.info("Loading Source Footprints")
            source_footprints = get_source_footprints(data)
            data.source_footprints = source_footprints

        return data


    def save(self, data_path: str, overwrite: bool=True, update_footprints: bool=True) -> None:
        """
        Save compiled data to an H5 file.
        """
        with h5py.File(data_path, "w" if overwrite else "a") as hdf:
            d = self.cfg.__dict__
            for k, v in d.items():
                if isinstance(v, dt.datetime):
                    d[k] = v.isoformat()
                else:
                    try:
                        d[k] = v
                    except:
                        raise Exception(f"Key: {k} cannot be converted to h5.")

            if "cfg" not in hdf.attrs:
                hdf.attrs["cfg"] = json.dumps(d)

            for obs in self.obs_list:
                if obs.obs_id not in hdf:
                    g = hdf.create_group(obs.obs_id)
                else:
                    g = hdf[obs.obs_id]
                for k, v in asdict(obs).items():
                    if v is None:
                        v = ""

                    if isinstance(v, np.ndarray) and v.dtype.names is not None:
                        if k in g:
                            del g[k]
                        g.create_dataset(k, data=v)
                    elif isinstance(v, (list, np.ndarray)) and np.issubdtype(np.array(v).dtype, np.number):
                        if k in g:
                            del g[k]
                        g.create_dataset(k, data=np.array(v))
                    elif isinstance(v, (int, float, np.floating)):
                        g.attrs[k] = v
                    else:
                        g.attrs[k] = str(v)

            if update_footprints and self.source_footprints is not None:
                fp_grp = hdf.create_group("source_footprints")
                wafers = np.array([fp.wafer for fp in self.source_footprints], dtype='S')
                sources = np.array([fp.source for fp in self.source_footprints], dtype='S')
                counts = np.array([fp.count for fp in self.source_footprints], dtype='i4')
                obsids = np.array([','.join(fp.obsids).encode('utf-8') for fp in self.source_footprints], dtype=h5py.special_dtype(vlen=bytes))
                fp_grp.create_dataset('wafer', data=wafers)
                fp_grp.create_dataset('source', data=sources)
                fp_grp.create_dataset('count', data=counts)
                fp_grp.create_dataset('obsids', data=obsids)


    @classmethod
    def load(cls, data_path: str) -> "ReportData":
        obs_list = []
        with h5py.File(data_path, "r") as hdf:
            # Load config
            cfg = ReportDataConfig(**json.loads(hdf.attrs["cfg"]))

            if "pwv" in hdf:
                pwv = np.array(hdf["pwv"])
            else:
                pwv = None

            for obs_id, g in hdf.items():
                if obs_id in ["pwv", "cfg", "source_footprints"]:
                    continue

                kwargs = {"obs_id": obs_id}

                for k, v in g.attrs.items():
                    if isinstance(v, bytes):
                        kwargs[k] = v.decode()
                    else:
                        kwargs[k] = v

                for k, dset in g.items():
                    data = dset[()]

                    if isinstance(data, np.ndarray) and data.dtype.names is not None:
                        kwargs[k] = data
                    elif data.shape == ():
                        kwargs[k] = float(data)
                    else:
                        kwargs[k] = data

                obs_list.append(ObsInfo(**kwargs))

            if "source_footprints" in hdf:
                fps = []
                fp_grp = hdf["source_footprints"]
                wafers = [w.decode() for w in fp_grp['wafer'][()]]
                sources = [s.decode() for s in fp_grp['source'][()]]
                counts = fp_grp['count'][()]
                obsids_list = [o.decode().split(',') for o in fp_grp['obsids'][()]]

                for wafer, source, count, obsids in zip(wafers, sources, counts, obsids_list):
                    fps.append(Footprint(
                        wafer=wafer, source=source, count=count, obsids=obsids
                    ))
            else:
                fps = None

        return cls(
            cfg=cfg,
            obs_list=obs_list,
            pwv=pwv,
            source_footprints=fps,
        )


def get_source_footprints(d: ReportData) -> List[Footprint]:
    fps: List[Footprint] = []
    db = ManifestDb(d.cfg.preprocess_sourcedb_path)
    entries = db.inspect()
    source_obs_list = []
    for entry in db.inspect():
        if (float(entry['obs:obs_id'].split('_')[1]) >= d.cfg.start_time.timestamp()) & \
        (float(entry['obs:obs_id'].split('_')[1]) <= d.cfg.stop_time.timestamp()):
            source_obs_list.append(entry['obs:obs_id'])

    coverage_data = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'obsids': set()}))

    for obs in source_obs_list:
        entry = db.inspect({'obs:obs_id': obs})
        if entry[0]['coverage'] == '':
            continue
        for pair in entry[0]['coverage'].split(','):
            source, wafer = pair.split(':')
            source = source.casefold()
            if source in d.cfg.cal_targets:
                coverage_data[wafer][source]['count'] += 1
                coverage_data[wafer][source]['obsids'].add(obs)

    for wafer, sources in coverage_data.items():
        for source, info in sources.items():
            fps.append(Footprint(wafer=wafer,
                                 source=source,
                                 count=info['count'],
                                 obsids=sorted(info['obsids'])))

    return fps
