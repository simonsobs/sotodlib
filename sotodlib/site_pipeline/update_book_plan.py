"""
Plan Books: read the level 2 G3tSmurf database and register the Books that
should be made from it in the Imprinter ("book") database. This script writes
no data; it only decides which level 2 observations belong together and
records the resulting Books with status UNBOUND, ready for ``make_book``.

Three registration passes run per invocation:

1. ``obs``/``oper`` Books, once per tel tube. For each level 2 observation in
   the window, look for observations on the *other* configured stream_ids that
   overlap it by at least ``min_overlap`` (30 s); mutually overlapping
   observations become one multi-wafer Book. Observations tagged as operations
   always get their own single-wafer ``oper`` Book, as do observations with
   low-precision timing (they cannot share a sample grid).
2. ``hk`` Books, one per timecode directory under ``<data_prefix>/hk``,
   excluding the most recent (still growing) one.
3. ``smurf`` and ``stray`` Books, one per timecode, driven by the suprsync
   TimeCodes rows in G3tSmurf. ``smurf`` Books need all metadata transfers
   complete; ``stray`` Books additionally need all file transfers complete and
   every obs/oper Book in the timecode successfully bound.

Nothing is ever planned past the finalization time (see
``G3tSmurf.get_final_time``), so ``update_g3thk_database`` and
``update_g3tsmurf_db`` must both have run recently. If the databases are stale
by more than ``--delay_warning`` hours the script warns; past
``--delay_error`` hours it raises, rather than plan Books over data that may
still be in transit.

The two search windows are set separately: ``--update-delay`` (default 1 day)
for obs/oper Books, and ``--update-delay-timecodes`` (default 7 days) for the
timecode Books, since a timecode can only be closed out well after the fact.

Errors are collected per tel tube so one broken tube does not stop the others,
then re-raised together at the end.

See the Data Packaging page of the sotodlib documentation for the Book model,
the imprinter configuration file format, and how to recover failed Books.
"""

import argparse
import datetime as dt
import time
from typing import Optional
from sqlalchemy import not_

from sotodlib.site_pipeline.monitor import Monitor
from sotodlib.site_pipeline.utils.logging import init_logger

from sotodlib.io.imprinter import (
    Imprinter,
    Books,
    BOUND,
    UNBOUND,
    UPLOADED,
    FAILED,
    DONE,
)

logger = init_logger(__name__, "update_book_plan: ")

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
    use_monitor: bool = False,
    delay_warning: float = 3, 
    delay_error: float = 6,
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
    use_monitor : bool
        if True, will send monitor information to influx, set to false by
        default so we can use identical config files for development
    delay_warning: float, optional
        if max_ctime - SMURF.final time > delay_warning: print warning about stale 
        databases. Additionally, for any incomplete observations (obs), if max_ctime - 
        obs.timestamp > delay_warning: look to see if a new stream has been started for 
        obs.stream_id and force completion of the earlier obs.
        delay_warning is specified in hours.
    delay_error: float, optional
        if max_ctime - SMURF.final time > delay_error: raise an error about stale 
        databases. Additionally, for any incomplete observations (obs), if max_ctime - 
        obs.timestamp > delay_error: raise error about incomplete obseravtions.
        delay_error is specified in hours.
    """
    if stream_ids is not None:
        stream_ids = stream_ids.split(",")

    imprinter = Imprinter(
        config, 
        db_args={'connect_args': {'check_same_thread': False}},
        logger=logger,
        make_db=from_scratch,
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
    logger.info("Registering obs/oper Books")
    _, update_errors = imprinter.update_bookdb_from_g3tsmurf(
        min_ctime=min_ctime, max_ctime=max_ctime,
        ignore_singles=False,
        stream_ids=stream_ids,
        force_single_stream=force_single_stream,
        delay_warning=delay_warning, 
        delay_error=delay_error,
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
    logger.info("Registering any HK Books")
    imprinter.register_hk_books(
        min_ctime=min_ctime_timecodes, 
        max_ctime=max_ctime_timecodes,
    )
    # smurf and stray books
    logger.info("Registering any timecode Books")
    imprinter.register_timecode_books(
        min_ctime=min_ctime_timecodes, 
        max_ctime=max_ctime_timecodes,
    )

    monitor = None
    if use_monitor and "monitor" in imprinter.config:
        logger.info("Will send monitor information to Influx")
        try:
            monitor = Monitor.from_configs(
                imprinter.config["monitor"]["connect_configs"]
            )
        except Exception as e:
            logger.error(f"Monitor connectioned failed {e}")
            monitor = None

    if monitor is not None:
        logger.info("Sending Updates to monitor")
        record_book_counts(monitor, imprinter)

    if update_errors is not None:
        for tube, error in update_errors:
            logger.error(f"Errors updating book database: error in tube {tube}: {error}")
        raise ValueError(f"Errors updating book database: {update_errors}")
    

def record_book_counts(monitor, imprinter):
    """Send a record of the current book count status to the InfluxDb
    site-pipeline monitor
    """
    tags = [{"telescope" : imprinter.config["monitor"]["telescope"]}]
    log_tags = {}
    script_run = time.time()

    session = imprinter.get_session()
    def get_count( q ):
        return session.query(Books).filter(q).count()
    
    monitor.record(
        "unbound", 
        [ get_count(Books.status == UNBOUND) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "bound", 
        [ get_count(Books.status == BOUND) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "uploaded", 
        [ get_count(Books.status == UPLOADED) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "failed", 
        [ get_count(Books.status == FAILED) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "done", 
        [ get_count(Books.status == DONE) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.record(
        "has_level2", 
        [ get_count(not_(Books.lvl2_deleted)) ], 
        [script_run], 
        tags, 
        imprinter.config["monitor"]["measurement"], 
        log_tags=log_tags
    )

    monitor.write()

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Register the Books that should be made from the "
            "level 2 G3tSmurf database into the imprinter database. Writes no "
            "data; run make_book afterwards to bind the registered Books."
        )
    parser.add_argument('--config', type=str,
                        help="Imprinter configuration file")
    parser.add_argument('--min-ctime', type=float,
                        help="Minimum observation ctime to consider. "
                        "Overrides --update-delay.")
    parser.add_argument('--max-ctime', type=float,
                        help="Maximum observation ctime to consider. Capped "
                        "at the finalization time regardless.")
    parser.add_argument('--stream-ids', type=str,
                        help="Comma separated list of stream_ids to consider, "
                        "e.g. 'ufm_mv6,ufm_mv9'. Defaults to the stream_ids "
                        "in the imprinter config.")
    parser.add_argument(
        '--force-single-stream',
        help="Treat multi-wafer data as single wafer data, i.e. register one "
        "Book per level 2 observation with no overlap grouping",
        action="store_true"
    )
    parser.add_argument(
        '--update-delay', type=float, 
        help="Days to subtract from now to set as minimum ctime",
        default=1
    )
    parser.add_argument(
        '--delay_warning', type=float, 
        help="Hours before incomplete observation fixes and database warnings are issued",
        default=3,
    )
    parser.add_argument(
        '--delay_error', type=float, 
        help="Hours before incomplete observation fixes and database errors are issued",
        default=6,
    )
    parser.add_argument(
        '--from-scratch',
        help="Create the imprinter database if needed and search from the "
        "beginning of time. Overrides --update-delay.",
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
    parser.add_argument('--use-monitor', help="Send updates to influx",
                        action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))