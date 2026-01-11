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
from influxdb import InfluxDBClient
from collections import defaultdict

from sotodlib.io import hkdb
from sotodlib.io.hkdb import HkConfig
from sotodlib.core import Context
from sotodlib.core.metadata import ManifestDb
from pixell import enmap, enplot


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReportDataConfig:
    def __init__(
        self,
        platform: Literal["satp1", "satp2", "satp3", "lat"],
        site_url: str,
        ctx_path: str,
        start_time: Union[dt.datetime, float, str],
        stop_time: Union[dt.datetime, float, str],
        hk_cfg: Union[HkConfig, str, Dict[str, Any]],
        buffer_time: float = 3600,
        influx_client_kw: Optional[Dict[str, Any]] = None,
        longterm_obs_file: Optional[str] = None,
        preprocess_sourcedb_path: Optional[str] = None,
        load_source_footprints: bool = True,
        make_cov_map: bool = True,
        cal_targets: Optional[List[str]] = None,
        show_hk_pb: bool = False,
    ) -> None:
        self.ctx_path: str = ctx_path
        self.platform: Literal["satp1", "satp2", "satp3", "lat"] = platform
        self.site_url: str = site_url
        self.buffer_time: float = buffer_time
        self.longterm_obs_file: Optional[str] = longterm_obs_file
        self.preprocess_sourcedb_path: Optional[str] = preprocess_sourcedb_path
        self.load_source_footprints: bool = load_source_footprints
        self.show_hk_pb: bool = show_hk_pb
        self.make_cov_map = make_cov_map

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
    def from_yaml(cls, path: str) -> "ReportDataConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass
class ObsInfo:
    obs_id: str
    start_time: float
    stop_time: float
    wafer_slots_list: str
    stream_ids_list: str
    obs_type: str
    obs_subtype: str
    obs_tube_slot: str
    obs_tags: str = ""
    pwv: float = np.nan
    el_center: float = np.nan
    boresight: float = np.nan
    hwp_freq_mean: float = np.nan
    num_valid_dets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    array_nep: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    det_nep: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @classmethod
    def from_obsdb_entry(cls, data) -> "ObsInfo":
        obs_info = cls(
            obs_id=data["obs_id"],
            start_time=data["start_time"],
            stop_time=data["stop_time"],
            wafer_slots_list=data["wafer_slots_list"],
            stream_ids_list=data["stream_ids_list"],
            obs_type=data["type"],
            obs_subtype=data["subtype"],
            obs_tube_slot=data["tube_slot"],
            boresight=-data["roll_center"] if data["roll_center"] is not None else np.nan,
            el_center=data["el_center"] if data["el_center"] is not None else np.nan,
            hwp_freq_mean=data["hwp_freq_mean"] if ("hwp_freq_mean" in data and data["hwp_freq_mean"] is not None) else np.nan
        )
        return obs_info

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
            'max_rows_returned': 79400,
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


def get_hk_and_pwv_data(cfg: ReportDataConfig):
    """
    Load the pwv from either the CLASS or APEX radiometer.
    Merge both datasets when available.
    """
    try:
        result = load_pwv(cfg)
    except Exception as e:
        logger.error(f"load_pwv failed with {e}")
        result = None
    try:
        result_apex = get_apex_data(cfg)
    except Exception as e:
        logger.error(f"get_apex_data failed with {e}")
        result_apex = None
    if result_apex is not None:
        result_apex = (np.array(result_apex['timestamps']), np.array(0.03+0.84 * result_apex['pwv']))

    if result is not None:
        if result_apex is not None:
            combined_times = np.concatenate((result[0], result_apex[0]))
            combined_data = np.concatenate((result[1], result_apex[1]))

            sorted_indices = np.argsort(combined_times)
            sorted_times = combined_times[sorted_indices]
            sorted_data = combined_data[sorted_indices]
            return (sorted_times, sorted_data)
        else:
            return result
    elif result_apex is not None:
        return result_apex
    else:
        return None


def load_qds_data(cfg: ReportDataConfig) -> pd.DataFrame:
    """
    Loads QDS data from influxdb.

    """
    client = InfluxDBClient(**cfg.influx_client_kw)
    buff_time = dt.timedelta(seconds=cfg.buffer_time)

    t0_str = (cfg.start_time - buff_time).isoformat().replace("+00:00", "Z")
    t1_str = (cfg.stop_time + buff_time).isoformat().replace("+00:00", "Z")

    keys = ['time', 'num_valid_dets', 'bandpass']
    if cfg.platform == "lat":
        keys += ['array_net_T', 'det_net_T']
    elif cfg.platform in ['satp1', 'satp2', 'satp3']:
        keys += [f"{prefix}_{suffix}" for prefix in ["array_net", "det_net"] for suffix in ["T", "Q", "U"]]

    query = f"""
        SELECT """ + ", ".join(keys) +  f""" from "autogen"."preprocesstod" WHERE (
            "tel_tube"::tag = '{cfg.platform}'
            AND time >= '{t0_str}'
            AND time <= '{t1_str}'
        )
    """

    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    missing_keys = [key for key in keys if key not in df]

    if missing_keys:
        logger.warn(f"missing keys from qds: {missing_keys}")
        return None

    df["time"] = pd.to_datetime(df["time"])
    df["timestamp"] = df["time"].apply(lambda x: x.timestamp())

    return df


def merge_qds_and_obs_list(df: pd.DataFrame, obs_list: List[ObsInfo]) -> None:
    timestamps = np.array([o.start_time for o in obs_list])
    obsids = [o.obs_id for o in obs_list]

    def find_obsid(ts: float) -> Optional[str]:
        idx = np.argmin(np.abs(ts - timestamps))
        if np.isclose(ts, timestamps[idx], atol=0.1):
            return obsids[idx]
        else:
            return ""

    df["obs_id"] = df["timestamp"].apply(find_obsid)

    nep_cols = [c for c in df.columns if "_net_" in c]
    agg_dict = {"num_valid_dets": "sum", **{c: "sum" for c in nep_cols}}

    totals = (
        df.groupby(["obs_id", "bandpass"], as_index=False)
          .agg(agg_dict)
    )

    totals_dict = (
    totals.groupby("obs_id")
          .apply(
              lambda g: {
                  row["bandpass"]: {c: row[c] for c in agg_dict.keys()}
                  for _, row in g.iterrows()
              }
          )
          .to_dict()
    )

    for obs_id, band_totals in totals_dict.items():
        band_totals.pop("NC", None)

        if not obs_id:
            continue

        obs_entry = obs_list[obsids.index(obs_id)]

        det_dtype = [(band, "i4") for band in band_totals.keys()]
        row = tuple(vals["num_valid_dets"] for _, vals in band_totals.items())
        obs_entry.num_valid_dets = np.array([row], dtype=det_dtype)

        for key, attr_name in zip(["array_net_", "det_net_"], ["array_nep", "det_nep"]):
            all_keys = sorted({
                k
                for vals in band_totals.values() if isinstance(vals, dict)
                for k in vals if k.startswith(key)
            })

            nep_dtype = [(band, [(k, "f8") for k in all_keys]) for band in band_totals.keys()]

            nep_row = tuple(
                tuple(vals.get(k, np.nan) for k in all_keys)
                for _, vals in band_totals.items()
            )

            setattr(obs_entry, attr_name, np.array([nep_row], dtype=nep_dtype))


def generate_coverage_map(ctx_path: str, obs_list: List[ObsInfo]):
    from sotodlib.mapmaking.utils import downsample_obs
    from sotodlib import coords

    ctx = Context(ctx_path)
    res = 10./60 * coords.DEG

    cmb_obs_list = [o for o in obs_list if o.obs_subtype == "cmb"]

    geom = enmap.fullsky_geometry(res=res, proj='car')
    w = None
    for o in tqdm(cmb_obs_list, total=len(cmb_obs_list)):
        aman = ctx.get_obs(o.obs_id, no_signal=True)
        aman.restrict("dets", np.isfinite(aman.focal_plane.gamma))
        aman = downsample_obs(aman, 100, skip_signal=True)
        aman.restrict("dets", aman.dets.vals[::10])
        p = coords.P.for_tod(aman, geom=geom, comps='T')
        w = p.to_weights(aman, dest=w)

    return w

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
        This contains data from the ObsDb, HkDb, and QDS database.
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

        logger.info("Building Obs List")
        obs_list = [
            ObsInfo.from_obsdb_entry(o)
            for o in ctx.obsdb.query(
                f"start_time >= {cfg.start_time.timestamp()} and "
                f"start_time <= {cfg.stop_time.timestamp()}"
            )
        ]

        for i, o in enumerate(obs_list):
            o.obs_tags = ",".join(ctx.obsdb.get(o.obs_id, tags=True)['tags'])

        if cfg.longterm_obs_file is not None:
            logger.info("Getting longterm data")
            longterm_obs_df = cls.load(cfg.longterm_obs_file, cov_map_path=None)
        else:
            longterm_obs_df = None

        logger.info("Loading PWV data")
        pwv = get_hk_and_pwv_data(cfg)

        logger.info("Loading QDS data")
        qds_df = load_qds_data(cfg)

        if qds_df is not None:
            logger.info("Merging PWV and QDS data with obs list")
            merge_qds_and_obs_list(qds_df, obs_list)
        else:
            logger.warn("QDS data not found")

        # Add PWV data to obs_list
        if pwv is not None:
            for o in obs_list:
                m = np.logical_and.reduce([pwv[0] >= o.start_time, pwv[0] <= o.stop_time])
                _pwv = np.nanmean(pwv[1][m])
                if -0.1 < _pwv < 3.5:
                    o.pwv = _pwv
        else:
            logger.warn("pwv data not found")
            for o in obs_list:
                o.pwv = -9999.

        data: "ReportData" = cls(
            cfg=cfg, obs_list=obs_list, pwv=None, longterm_obs_df=longterm_obs_df
        )

        if cfg.load_source_footprints:
            logger.info("Loading Source Footprints")
            source_footprints = get_source_footprints(data)
            data.source_footprints = source_footprints

        if cfg.make_cov_map:
            logger.info("Making Coverage Map")
            try:
                data.w = generate_coverage_map(cfg.ctx_path, obs_list)
            except Exception as e:
                logger.error(f"Coverage map failed with {e}")

        return data


    def save(self, data_path: str, cov_map_path: str, overwrite: bool=True, update_footprints: bool=True) -> None:
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
            if self.pwv is not None and "pwv" not in hdf:
                hdf.create_dataset("pwv", data=self.pwv)

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

            if cov_map_path is not None and self.w is not None:
                enmap.write_map(cov_map_path + ".fits", self.w)
                f = enplot.plot(self.w, grid=True, downgrade=1, mask=0, ticks=10)
                enplot.write(cov_map_path + ".png", f[0])


    @classmethod
    def load(cls, data_path: str, cov_map_path: str) -> "ReportData":
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

        if cov_map_path is not None:
            w = enmap.read_map(cov_map_path + ".fits")
        else:
            w = None

        return cls(
            cfg=cfg,
            obs_list=obs_list,
            pwv=pwv,
            source_footprints=fps,
            w=w,
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
