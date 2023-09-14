import argparse
import datetime as dt
import time
from typing import Optional

from sotodlib.io.imprinter import Imprinter
from sotodlib.site_pipeline.monitor import Monitor

def main(
    config: str,
    min_ctime: Optional[float] = None,
    max_ctime: Optional[float] = None,
    stream_ids: Optional[str] = None,
    force_single_stream: bool = False,
    update_delay: float = 1,
    update_delay_timecodes: Optional[float] = 7,
    min_ctime_timecodes: Optional[float] = None,
    max_ctime_timecodes: Optional[float] = None,
    from_scratch: bool = False,
    logger = None
    ):
    """
    Update the book plan database with new data from the g3tsmurf database.

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    min_ctime : Optional[float], optional
        The minimum ctime to include in the book plan, by default None
    max_ctime : Optional[float], optional
        The maximum ctime to include in the book plan, by default None
    stream_ids : Optional[str], optional
        The stream ids to consider, list supplied as a comma separated string
        (e.g. "1,2,3"), by default None
    force_single_stream : bool, optional
        If True, tream multi-wafer data as if it were single wafer data, by default False
    update_delay : float, optional
        The range of time to search through g3tsmurf db for new data in units of
        days, by default 1
    update_delay_timecodes : float, optional
        The range of time to search through g3tsmurf db for new data in units of
        days for timecode books, by default 7
    min_ctime_timecodes : Optional[float], optional
        The minimum ctime to include in the book plan for timecode (hk, smurf, stray) books, by default None
    max_ctime_timecodes : Optional[float], optional
        The maximum ctime to include in the book planor timecode (hk, smurf, stray) books, by default None

    from_scratch : bool, optional
        If True, start to search from beginning of time, by default False
    """
    if stream_ids is not None:
        stream_ids = stream_ids.split(",")
    imprinter = Imprinter(
        config, db_args={'connect_args': {'check_same_thread': False}}
    )
    
    # leaving min_ctime and max_ctime as None will go through all available 
    # data, so preferreably set them to a reasonable range based on update_delay
    if not from_scratch and min_ctime is None:
        min_ctime = dt.datetime.now() - dt.timedelta(days=update_delay)
    if isinstance(min_ctime, dt.datetime):
        min_ctime = min_ctime.timestamp()
    if isinstance(max_ctime, dt.datetime):
        max_ctime = max_ctime.timestamp()

    # obs and oper books
    imprinter.update_bookdb_from_g3tsmurf(
        min_ctime=min_ctime, max_ctime=max_ctime,
        ignore_singles=False,
        stream_ids=stream_ids,
        force_single_stream=force_single_stream
    )

    ## over-ride timecode book making if specific values given
    if update_delay_timecodes is None and min_ctime_timecodes is None:
        min_ctime_timecodes = min_ctime
    elif min_ctime_timecodes is None:
        min_ctime_timecodes = (
            dt.datetime.now() - dt.timedelta(days=update_delay_timecodes)
        )
    if max_ctime_timecodes is None:
        max_ctime_timecodes = max_ctime

    if isinstance(min_ctime_timecodes, dt.datetime):
        min_ctime_timecodes = min_ctime_timecodes.timestamp()
    if isinstance(max_ctime, dt.datetime):
        max_ctime_timecodes = max_ctime_timecodes.timestamp()

    # hk books
    imprinter.register_hk_books(
        min_ctime=min_ctime_timecodes, 
        max_ctime=max_ctime_timecodes,
    )
    # smurf and stray books
    imprinter.register_timecode_books(
        min_ctime=min_ctime_timecodes, 
        max_ctime=max_ctime_timecodes,
    )

    monitor = None
    if "monitor" in imprinter.config:
        imprinter.logger.info("Will send monitor information to Influx")
        try:
            monitor = Monitor.from_configs(
                imprinter.config["monitor"]["connect_configs"]
            )
        except Exception as e:
            imprinter.logger.error(f"Monitor connectioned failed {e}")
            monitor = None

    if monitor is not None:
        record_book_counts(monitor, imprinter)
    

def record_book_counts(monitor, imprinter):
    """Send a record of the current book count status to the InfluxDb
    site-pipeline montir
    """
    tags = [{"telescope" : imprinter.config["monitor"]["telescope"]}]
    log_tags = {}
    script_run = time.time()

    monitor.record(
        "unbound", 
        [ len(imprinter.get_unbound_books()) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "bound", 
        [ len(imprinter.get_bound_books()) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "uploaded", 
        [ len(imprinter.get_uploaded_books()) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "failed", 
        [ len(imprinter.get_failed_books()) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.write()

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="g3tsmurf db configuration file")
    parser.add_argument('--min-ctime', type=float, help="Minimum creation time")
    parser.add_argument('--max-ctime', type=float, help="Maximum creation time")
    parser.add_argument('--stream-ids', type=str, help="Stream IDs")
    parser.add_argument(
        '--force-single-stream', help="Force single stream", action="store_true"
    )
    parser.add_argument(
        '--update-delay', type=float, 
        help="Days to subtract from now to set as minimum ctime",
        default=1
    )
    parser.add_argument(
        '--from-scratch', help="Builds or updates database from scratch",
        action="store_true"
    )
    parser.add_argument(
        '--min-ctime-timecodes', type=float, 
        help="Minimum creation time for timecode books"
    )
    parser.add_argument(
        '--max-ctime-timecodes', type=float, 
        help="Maximum creation time for timecode books"
    )
    parser.add_argument(
        '--update-delay-timecodes', type=float, 
        help= "Days to subtract from now to set as minimum ctime "
              "for timecode books",
        default=7
    )

    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
