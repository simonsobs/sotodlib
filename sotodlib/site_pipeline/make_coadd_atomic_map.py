from argparse import ArgumentParser
import os
import yaml
import traceback
import sqlite3
import datetime as dt
from dateutil.relativedelta import relativedelta
from typing import Literal, Union, Dict, Any, Optional, Tuple, List

from sotodlib import mapmaking
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from pixell import utils as putils


class CoaddAtomicConfig:
    """
    Class to configure make-coadd-atomic-map
    
    Arguments
    ---------
    platform : str
        Telescope platform. Choices: ["satp1", "satp2", "satp3", "lat"]
    interval : str
        Interval to group atomic maps for coadding.
        Choices: ["daily", "weekly", "monthly"]
    start_time : str
        Start time for querying atomic maps, in format "%Y-%m-%dT%H:%M:%S%z"
        Example: "2025-08-19T12:00:00+00:00"
    stop_time : str
        Stop time for querying atomic maps, in same format as start_time.
    start_weekday : int
        Starting weekday for "weekly" interval. Integers ranging from 0 to 6,
        where Monday = 0, Sunday = 6.
    atomic_db : str
        Path to input atomic maps database.
    output_db : str
        Path to output coadded maps database.
    bands : List[str]
        List of bands to coadd, where '+' combines both bands.
        Choices: ["f090", "f150", "f090+f150"]
    split_label : str
        Split label from atomic map files.
    geom_file_prefix : str
        Prefix path to geometry file, omitting the band.
    unit : str
        Temperature unit.
    overwrite : bool
        Set to True to re-run coadding and overwrite database if time interval
        found in database.
    output_root : str
        Path to directory for output map files.
    plot : bool
        Set to True to also output PNG plots when writing maps.
    """
    def __init__(
        self,
        platform: Literal["satp1", "satp2", "satp3", "lat"],
        interval: Literal["daily", "weekly", "monthly"],
        start_time: Union[dt.datetime, float, str],
        stop_time: Union[dt.datetime, float, str, None] = None,
        start_weekday: int = 6,
        atomic_db: str = None,
        output_db: str = None,
        bands: List[str] = ["f090", "f150", "f090+f150"],
        split_label: str = "full",
        geom_file_prefix: str = None,
        unit: str = 'K',
        overwrite: bool = False,
        output_root: str = None,
        plot: bool = False
    ) -> None:
        self.platform: Literal["satp1", "satp2", "satp3", "lat"] = platform
        self.interval = interval
        self.atomic_db = atomic_db
        self.output_db = output_db
        self.bands = bands
        self.split_label = split_label
        self.geom_file_prefix = geom_file_prefix
        self.unit = unit
        self.overwrite = overwrite
        self.output_root = output_root
        self.plot = plot
        
        def convert_to_datetime(
            time: Union[dt.datetime, float, str, None],
        ) -> dt.datetime:
            if isinstance(time, type(None)):
                return dt.datetime.now(tz=dt.timezone.utc)
            if isinstance(time, str):
                return dt.datetime.fromisoformat(time)
            elif isinstance(time, (int, float)):
                return dt.datetime.fromtimestamp(time)
            elif isinstance(time, dt.datetime):
                return time
            else:
                raise Exception(f"Could not convert type {type(time)} to datetime")
                
        self.start_time = convert_to_datetime(start_time)
        self.stop_time = convert_to_datetime(stop_time)
        self.time_intervals: List[Tuple[dt.datetime, dt.datetime]] = []
        time_of_day = dt.time(self.start_time.hour, self.start_time.minute, self.start_time.second, tzinfo=self.start_time.tzinfo)
        if self.interval == "daily":
            delta = dt.timedelta(days=1)
        elif self.interval == "weekly":
            current_weekday = self.start_time.weekday()
            days_offset = (current_weekday - start_weekday) % 7
            aligned_date = (self.start_time - dt.timedelta(days=days_offset)).date()
            self.start_time = dt.datetime.combine(aligned_date, time_of_day, tzinfo=self.start_time.tzinfo)
            delta = dt.timedelta(weeks=1)
        elif self.interval == "monthly":
            aligned_date = dt.date(self.start_time.year, self.start_time.month, 1)
            self.start_time = dt.datetime.combine(aligned_date, time_of_day, tzinfo=self.start_time.tzinfo)
            delta = relativedelta(months=1)
        start: dt.datetime = self.start_time
        self.time_intervals = []
        now = dt.datetime.now(tz=dt.timezone.utc)
        while start < self.stop_time:
            stop: dt.datetime = start + delta
            if stop > now - dt.timedelta(hours=1):
                # Give a buffer of 1 hour to compile report for previous interval
                break
            self.time_intervals.append((start, stop))
            start += delta
            
    @classmethod
    def from_yaml(cls, path) -> "CoaddAtomicConfig":
        with open(path, "r") as f:
            return CoaddAtomicConfig(**yaml.safe_load(f))


def main(config_file: str, verbosity: int) -> None:
    logger = init_logger("make_coadd_atomic_map", verbosity=verbosity)
    cfg = CoaddAtomicConfig.from_yaml(config_file)
    logger.info(f"Setup {cfg.interval} intervals")
    logger.debug(cfg.time_intervals)
    
    putils.mkdir(cfg.output_root)

    logger.info(f"Database initialized at {cfg.output_db}")
    init_output_db(cfg.output_db, cfg.interval)

    for start_time, stop_time in cfg.time_intervals:
        time_str = f"{start_time:%Y%m%d}_{stop_time:%Y%m%d}"
        logger.info(f"Coadding interval {time_str}")

        for band in cfg.bands:
            logger.info(f'Coadding band {band}')
            try:
                success, err = mapmaking.make_coadd_map(cfg.atomic_db, cfg.output_root,
                                                   cfg.output_db, band, cfg.platform, 
                                                   cfg.split_label, start_time, stop_time, 
                                                   cfg.interval, cfg.geom_file_prefix, 
                                                   overwrite=cfg.overwrite, unit=cfg.unit, 
                                                   logger=logger, plot=cfg.plot)
                if not success:
                    logger.warning(err)
            except Exception as e:
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.error(f"Failed to coadd map for {time_str}, {band}: {tb} {e}")
    return True

def init_output_db(output_db, interval):
    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    create_stmt = f'''
        CREATE TABLE IF NOT EXISTS {interval} (
            telescope TEXT,
            freq_channel TEXT,
            split_label TEXT,
            prefix_path TEXT,
            geom_file_path TEXT,
            obslist TEXT,
            start_time FLOAT,
            stop_time FLOAT,
            PRIMARY KEY (telescope, freq_channel, split_label, prefix_path, geom_file_path, obslist, start_time, stop_time)
        )
    '''
    cur.execute(create_stmt)
    conn.commit()
    conn.close()


def get_parser(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        p = ArgumentParser()
    else:
        p = parser
    p.add_argument(
        "--config-file", type=str, help="yaml file with configuration."
    )
    p.add_argument(
        '--verbosity',
        help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
        default=2,
        type=int
    )
    return p

if __name__ == '__main__':
    main_launcher(main, get_parser)
