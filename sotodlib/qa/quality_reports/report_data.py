from dataclasses import dataclass, fields
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
# import alphashape
from influxdb import InfluxDBClient
from collections import defaultdict

from sotodlib.io import hkdb
from sotodlib.io.hkdb import HkConfig
from sotodlib.core import Context
from sotodlib.core.metadata import ManifestDb

logger = logging.getLogger(__name__)

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
    num_valid_dets: str = ""

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
        )
        return obs_info

    def __repr__(self):
        inner = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        return f"ObsInfo({inner})"


def arr_to_obs_list(obs_arr: np.ndarray) -> List[ObsInfo]:
    """
    Convert a npy structured array back to a list of ObsInfo objects.
    """
    obs_list: List[ObsInfo] = []
    for entry in obs_arr:
        kw = {}
        for k, v in obs_arr.dtype.fields.items():
            if v[0].type == np.bytes_:  # Convert back to string
                kw[k] = entry[k].decode()
            else:  # Is a numeric type
                kw[k] = entry[k]
        obs_list.append(ObsInfo(**kw))
    return obs_list


def obs_list_to_arr(obs_list: List[ObsInfo]) -> np.ndarray:
    """
    Convert a list of ObsInfo objects to a numpy structured array for processing
    and storage.
    """
    dtype = []
    data = []
    for field in fields(ObsInfo):
        if field.type is str:
            typ: Any = "|S100"
        else:
            typ = field.type
        dtype.append((field.name, typ))
    for obs in obs_list:
        data.append(tuple(getattr(obs, name) for name, _ in dtype))
    return np.array(data, dtype=dtype)


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

    # don't try and load pwv earlier than 90 days ago
    if cfg.start_time < dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90):
        return None

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

    return result


def get_hk_and_pwv_data(cfg: ReportDataConfig):
    """
    Load the pwv from either the CLASS or APEX radiometer.
    Merge both datasets when available.
    """
    result = load_pwv(cfg)
    result_apex = get_apex_data(cfg)
    if result_apex is not None:
        result_apex = (np.array(result_apex['timestamps']), np.array(0.03+0.84 * result_apex['pwv']))

    if hasattr(result, 'pwv'):
        if result_apex is not None:
            combined_times = np.concatenate((result.pwv[0], result_apex[0]))
            combined_data = np.concatenate((result.pwv[1], result_apex[1]))

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

    keys = ['time', 'num_valid_dets', 'wafer.bandpass']#, '"wafer_slot"::tag', '"wafer.bandpass"::tag']

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

    totals = (
        df[["num_valid_dets", "obs_id", "wafer.bandpass"]]
        .groupby(["obs_id", "wafer.bandpass"])["num_valid_dets"]
        .sum()
        .reset_index()
    )

    totals_dict = (
        totals.groupby("obs_id")
              .apply(lambda g: dict(zip(g["wafer.bandpass"], g["num_valid_dets"])))
              .to_dict()
    )

    for obs_id, band_totals in totals_dict.items():
        band_totals.pop("NC", None)
        if not obs_id:
            continue
        obs_entry = obs_list[obsids.index(obs_id)]
        obs_entry.num_valid_dets = str(band_totals)


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
    pwv: np.ndarray
    source_footprints: Optional[List[Footprint]] = None
    longterm_obs_df: Optional[pd.DataFrame] = None

    @classmethod
    def build(cls, cfg: ReportDataConfig) -> "ReportData":
        ctx = Context(cfg.ctx_path)

        obs_list = [
            ObsInfo.from_obsdb_entry(o)
            for o in ctx.obsdb.query(
                f"start_time >= {cfg.start_time.timestamp()} and "
                f"start_time <= {cfg.stop_time.timestamp()}"
            )
        ]

        #for i, o in tqdm(enumerate(obs_list), total=len(obs_list)):
        for i, o in enumerate(obs_list):
            o.obs_tags = ",".join(ctx.obsdb.get(o.obs_id, tags=True)['tags'])

        if cfg.longterm_obs_file is not None:
            longterm_obs_df = cls.load(cfg.longterm_obs_file)
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
            cfg=cfg, obs_list=obs_list, pwv=pwv, longterm_obs_df=longterm_obs_df
        )

        if cfg.load_source_footprints:
            logger.info("Loading Source Footprints")
            source_footprints = get_source_footprints(data)
            data.source_footprints = source_footprints
        return data

    def save(self, path: str) -> None:
        """
        Save compiled data to an H5 file.
        """
        with h5py.File(path, "w") as hdf:
            d = self.cfg.__dict__
            for k, v in d.items():
                if isinstance(v, dt.datetime):
                    d[k] = v.isoformat()
                else:
                    try:
                        d[k] = v
                    except:
                        raise Exception(f"Key: {k} cannot be converted to h5.")
            hdf.attrs["cfg"] = json.dumps(d)
            hdf.create_dataset("pwv", data=self.pwv)
            hdf.create_dataset("obs_list", data=obs_list_to_arr(self.obs_list))

            if self.source_footprints is not None:
                fp_grp = hdf.create_group("source_footprints")
                wafers = np.array([fp.wafer for fp in self.source_footprints], dtype='S')
                sources = np.array([fp.source for fp in self.source_footprints], dtype='S')
                counts = np.array([fp.count for fp in self.source_footprints], dtype='i4')
                obsids = np.array([','.join(fp.obsids).encode('utf-8') for fp in self.source_footprints], dtype=h5py.special_dtype(vlen=bytes))
                fp_grp.create_dataset('wafer', data=wafers)
                fp_grp.create_dataset('source', data=sources)
                fp_grp.create_dataset('count', data=counts)
                fp_grp.create_dataset('obsids', data=obsids)
                # for i, fp in enumerate(self.source_footprints):
                #     grp = fp_grp.create_group(f"fp_{i}")
                #     grp.attrs["target"] = fp.target
                #     grp.attrs["obs_id"] = fp.obs_id
                #     grp.create_dataset("wafers", data=','.join(wafers.astype(str)))

    @classmethod
    def load(cls, path: str) -> "ReportData":
        with h5py.File(path, "r") as hdf:
            # Load config
            cfg = ReportDataConfig(**json.loads(hdf.attrs["cfg"]))

            obs_list = arr_to_obs_list(hdf["obs_list"])
            pwv = np.array(hdf["pwv"])

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
                # fps = [Footprint.from_h5(fp) for fp in hdf["source_footprints"].values()]
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
        # coverage = np.array([hitpair.split(':') for hitpair in entry[0]['coverage'].split(',')])
        # for source in d.cfg.cal_targets:
        #     if source in coverage[:, 0]:
        #         fps.append(Footprint(obs_id=obs,
        #                              target=source,
        #                              wafers=coverage[coverage[:, 0] == source, 1]))
        for pair in entry[0]['coverage'].split(','):
            source, wafer = pair.split(':')
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
