"""
Script for running updates on (or creating) a g3tsmurf database, the index of
level 2 SMuRF data. This setup is specifically designed to work when the data
is dynamically coming in, meaning it is designed to work from something like a
cronjob.

Each run does the following, over the ctime range being updated:

1. ``index_metadata``: walk the ``smurf/`` action folders and index tunes,
   channel assignments and other sodetlib metadata.
2. ``index_archive``: index the ``timestreams/`` .g3 files and their frames,
   grouping them into level 2 observations.
3. ``index_action_observations`` (only with ``--index-via-actions``):
   reconstruct observations from action folders. Needed for data older than
   Oct 2022, but it races with automatic level 2 deletion so it is off at the
   site.
4. ``index_timecodes``: index the suprsync rows that say a timecode's files or
   metadata have finished transferring.
5. ``update_finalization``: advance the per-agent finalization times, i.e. how
   far we trust the level 2 data to be complete. This reads the G3tHk
   database, so ``update_g3thk_database`` must have run first.
6. Revisit recent observations with no stop time or no tuneset and try to
   complete them.

The script then raises if it finds *completed* observations that have bad
timing or no tuneset (no readout IDs), since neither can be bound into a Book.
Observations that a human has already inspected and accepted can be listed
one per line in a file passed as ``--checked-file`` to suppress the error.

The time range defaults to the last ``--update-delay`` days. ``--min_ctime``
may not be later than the current finalization time, otherwise a gap would be
silently left in the index.

Configuration file, an expanded version of the one used to connect to a
G3tSmurf database::

    data_prefix : "/path/to/daq-node/"
    g3tsmurf_db: "/path/to/g3tsmurf.db"
    g3thk_db: "/path/to/g3hk.db"

    finalization:
        servers:
            - smurf-suprsync: "smurf-sync-so1"        ## instance-id
              timestream-suprsync: "timestream-sync-so1"
              pysmurf-monitor: "monitor-so1"

The user running this script must have read, write, and execute permissions
on the database file.

See the Data Packaging page of the sotodlib documentation for the full
pipeline description.
"""
import os
import datetime as dt
import argparse
import logging
from sqlalchemy import or_
from typing import Optional


from sotodlib.site_pipeline.utils.profiler import profile, add_profile_args

from sotodlib.io.load_smurf import G3tSmurf, Observations, logger
from sotodlib.io.datapkg_utils import load_configs

@profile("update_g3tsmurf_db")
def main(
    config: Optional[str] = None, update_delay: float = 2,
    from_scratch: bool = False, verbosity: int = 2,
    min_ctime: Optional[float]=None, max_ctime: Optional[float]=None,
    index_via_actions: bool=False, checked_file: Optional[str]=None,
):
    """
    Arguments
    ---------
    config: string
        configuration file for G3tSmurf
    update_delay: float
        number of days to 'look back' to update observation information
    from_scratch: bool
        if True, run database update with minimum ctime of 1.6e9 (all SO time).
        overrides update_delay
    verbosity: int
        0-3, higher numbers = more printouts
    min_ctime: float
        minimum ctime to start the search, overrides the time set by
        update_delay
    max_ctime: float
        maximum ctime to search, otherwise searches through 'now'
    index_via_actions: bool
        if True, will look through action folders to create observations, this
        will be necessary for data older than Oct 2022 but creates concurancy
        issues on systems (like the site) running automatic deletion of level 2
        data.
    checked_file: str
        a file name that contains a list of observations that would by default
        cause errors to be thrown during this script but have been manually
        checked and dealt with

    Notes
    -----
    The `profile` decorator adds three further keyword arguments, `profile`,
    `profile_type` and `profile_output`; see
    `sotodlib.site_pipeline.utils.profiler`. They are added to the command line
    by `add_profile_args` in `get_parser`.
    """
    show_pb = True if verbosity > 1 else False

    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    make_db = False
    if from_scratch:
        logger.info("Building Database from Scratch, May take awhile")
        min_ctime = 1.6e9
        make_db = True
    if min_ctime is None:
        min_ctime = (dt.datetime.now() - dt.timedelta(days=update_delay)).timestamp()

    cfgs = load_configs( config )
    SMURF = G3tSmurf.from_configs(cfgs, make_db=make_db)

    updates_start = dt.datetime.now().timestamp()
    if max_ctime is not None:
        assert max_ctime > min_ctime, "max_ctime is before min_ctime"
        ## if we're setting a maximum ctime then we need to be sure we don't 
        ## believe the database is more updated than that.
        updates_start = max_ctime
    
    ## make sure we don't have a gap between currently finalized time and when we're 
    ## starting updates now
    if not from_scratch:
        current_time = SMURF.last_update
        if min_ctime > current_time:
            raise ValueError(
                f"min_ctime {min_ctime} is higher than current database coverage"
                f" {current_time}"
            )
    else:
        current_time = None
    logger.info(
        f"G3tSmurf is updated through {current_time}. Beginning updates"
        f" from {min_ctime} to {max_ctime}"
    )

    session = SMURF.Session()
    SMURF.index_metadata(
        min_ctime=min_ctime, 
        max_ctime=max_ctime, 
        session=session
    )

    logger.info("Starting to index files")
    SMURF.index_archive(
        min_ctime=min_ctime,
        max_ctime=max_ctime,
        show_pb=show_pb,
        session=session
    )
    if index_via_actions:
        SMURF.index_action_observations(
            min_ctime=min_ctime,
            max_ctime=max_ctime,
            session=session
        )
    SMURF.index_timecodes(
        min_ctime=min_ctime,
        max_ctime=max_ctime,
        session=session
    )
    logger.info("Starting Finialization Update")
    SMURF.update_finalization(update_time=updates_start, session=session)
    SMURF.last_update = updates_start
    logger.info(f"G3tSmurf Finalization Time now {updates_start}")
    
    new_obs = session.query(Observations).filter(
        or_(
            ## the replace nonsense is because g3tsmurf is timezone naive. longer
            ## term thing to deal with.
            Observations.start >= dt.datetime.fromtimestamp(
                min_ctime, tz=dt.timezone.utc
            ).replace(tzinfo=None),
            Observations.start == None,
        )
    ).all()

    raise_list_timing = []
    raise_list_readout_ids = []
    
    obs_to_edit = [
        obs for obs in new_obs
        if obs.stop is None or len(obs.tunesets) == 0
    ]
    logger.info(f"Updating {len(obs_to_edit)} incomplete observations")

    ## still loop over new_obs to check others for timing
    for obs in new_obs:
        if obs.stop is None or len(obs.tunesets)==0:
            SMURF.update_observation_files(
                obs,
                session,
                force=True,
            )
        if (obs.stop is not None) and (not obs.timing):
            raise_list_timing.append(obs.obs_id)

        if (obs.stop is not None) and len(obs.tunesets)==0:
            raise_list_readout_ids.append(obs.obs_id)


    if len(raise_list_timing) > 0 or len(raise_list_readout_ids) > 0:
        logger.info(
            f"Found observations with bad timing or missing readout ids "
            "checking to see if they've been manually cleared"
        )
        if checked_file is None or not os.path.exists(checked_file):
            logger.warning(
                f"File {checked_file} does not exist so cannot check if "
                "problematic observations have been manually cleared"
            )
            cleared = []
        else:
            with open(checked_file) as f:
                cleared = [c.strip("\n").strip() for c in f.readlines()]
        raise_list_timing = [x for x in raise_list_timing if x not in cleared]
        raise_list_readout_ids = [
            x for x in raise_list_readout_ids if x not in cleared
        ]
    if len(raise_list_timing) > 0 or len(raise_list_readout_ids) > 0:
        raise ValueError(
            f"Found {len(raise_list_timing)} observations with bad timing"
            f" obs_ids are {raise_list_timing}.\nFound "
            f"{len(raise_list_readout_ids)} observations without Tunesets"
            f" obs_ids are {raise_list_readout_ids}."
        )

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Index level 2 SMuRF data into a G3tSmurf database "
            "and advance the data transfer finalization time. Run "
            "update_g3thk_database first."
        )

    parser.add_argument('config',
                        help="g3tsmurf db configuration file. Requires the "
                        "'data_prefix', 'g3tsmurf_db', 'g3thk_db' and "
                        "'finalization' keys.")
    parser.add_argument('--update-delay', help="Days to subtract from now to set as minimum ctime",
                        default=2, type=float)
    parser.add_argument('--from-scratch',
                        help="Index from ctime 1.6e9 (all SO time), creating "
                        "the database if needed. Overrides --update-delay.",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                        default=2, type=int)
    parser.add_argument("--min_ctime",
        help="minimum ctime to start search, overrides time set by update-delay",
        default=None, type=int
    )
    parser.add_argument("--max_ctime",
        help="maximum ctime to search, otherwise searches through 'now'",
        default=None, type=int
    )
    parser.add_argument('--index-via-actions',
                        help="Look through action folders to create "
                        "observations. Needed for data older than Oct 2022, "
                        "but causes concurrency issues on systems (like the "
                        "site) that automatically delete level 2 data.",
                        action="store_true")
    parser.add_argument("--checked-file",
        help="Path to a file listing, one obs_id per line, observations with "
        "bad timing or missing readout ids that have been manually checked. "
        "Listed observations do not cause this script to raise.")
    
    add_profile_args(parser)

    return parser

if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
