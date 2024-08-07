"""
Script for running updates on (or creating) a g3tsmurf database. This setup
is specifically designed to work when the data is dynamically coming in. Meaning is 
is designed to work from something like a cronjob. 
"""
import os
import yaml
import datetime as dt
import numpy as np
import argparse
import logging
from sqlalchemy import not_, or_, and_
from typing import Optional

from sotodlib.site_pipeline.monitor import Monitor
from sotodlib.io.load_smurf import G3tSmurf, Observations, logger
from sotodlib.io.datapkg_utils import load_configs


def main(config: Optional[str] = None, update_delay: float = 2, 
         from_scratch: bool = False, verbosity: int = 2,
         index_via_actions: bool=False, use_monitor:bool=False):
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
    index_via_actions: bool
        if True, will look through action folders to create observations, this
        will be necessary for data older than Oct 2022 but creates concurancy 
        issues on systems (like the site) running automatic deletion of level 2 
        data.
    use_monitor : bool
        if True, will send monitor information to influx, set to false by
        default so we can use identical config files for development
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

    if from_scratch:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = dt.datetime.utcfromtimestamp(int(1.6e9))
        make_db = True
    else:
        min_time = dt.datetime.now() - dt.timedelta(days=update_delay)
        make_db = False

    cfgs = load_configs( config )
    SMURF = G3tSmurf.from_configs(cfgs, make_db=make_db)

    updates_start = dt.datetime.now().timestamp()

    session = SMURF.Session()
    SMURF.index_metadata(min_ctime=min_time.timestamp(), session=session)
    SMURF.index_archive(
        min_ctime=min_time.timestamp(), 
        show_pb=show_pb, 
        session=session
    )
    if index_via_actions:
        SMURF.index_action_observations(
            min_ctime=min_time.timestamp(),
            session=session
        )    
    SMURF.index_timecodes(
        min_ctime=min_time.timestamp(),
        session=session
    )
    SMURF.update_finalization(update_time=updates_start, session=session)
    SMURF.last_update = updates_start

    new_obs = session.query(Observations).filter(
        or_(
            Observations.start >= min_time,
            Observations.start == None,
        )
    ).all()

    raise_list_timing = []
    raise_list_readout_ids = []
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
        raise ValueError(
            f"Found {len(raise_list_timing)} observations with bad timing"
            f" obs_ids are {raise_list_timing}.\nFound "
            f"{len(raise_list_readout_ids)} observations without Tunesets" 
            f" obs_ids are {raise_list_readout_ids}."
        )

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="g3tsmurf db configuration file")
    parser.add_argument('--update-delay', help="Days to subtract from now to set as minimum ctime",
                        default=2, type=float)
    parser.add_argument('--from-scratch', help="Builds or updates database from scratch",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                        default=2, type=int)
    parser.add_argument('--index-via-actions', help="Look through action folders to create observations",
                        action="store_true")
    parser.add_argument('--use-monitor', help="Send updates to influx",
                        action="store_true")
    return parser

if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
