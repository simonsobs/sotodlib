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

from sotodlib.site_pipeline.monitor import Monitor
from sotodlib.io.load_smurf import G3tSmurf, Observations, logger as default_logger


def update_g3tsmurf_db(config=None, update_delay=2, from_scratch=False,
                       verbosity=2, logger=None):

    show_pb = True if verbosity > 1 else False

    if logger is None:
        logger = default_logger
    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    cfgs = yaml.safe_load( open(config, "r"))
    SMURF = G3tSmurf.from_configs(cfgs)

    if from_scratch:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = dt.datetime.utcfromtimestamp(int(1.6e9))
    else:
        min_time = dt.datetime.now() - dt.timedelta(days=update_delay)

    monitor = None
    if "monitor" in cfgs:
        logger.info("Will send monitor information to Influx")
        monitor = Monitor.from_configs(cfgs["monitor"]["connect_configs"])
        
    SMURF.index_metadata(min_ctime=min_time.timestamp())
    SMURF.index_archive(min_ctime=min_time.timestamp(), show_pb=show_pb)
    SMURF.index_action_observations(min_ctime=min_time.timestamp())    

    session = SMURF.Session()

    new_obs = session.query(Observations).filter(Observations.start >= min_time).all()

    for obs in new_obs:
        if obs.stop is None:
            SMURF.update_observation_files(
                obs, 
                session, 
                force=True,
            )
        
        if monitor is not None:
            if obs.stop is not None:
                record_timing(monitor, obs, cfgs)
                record_tuning(monitor, obs, cfgs)

def _obs_tags(obs, cfgs):
    
    tags = [{
        "tel_tube" : cfgs["monitor"]["tel_tube"], 
        "stream_id" : obs.stream_id
    }]

    log_tags = {"observation": obs.obs_id, "stream_id": obs.stream_id}

    return tags, log_tags

def record_tuning(monitor, obs, cfgs):
    """Send a record of the Tune Status to the Influx QDS database.
    Will be used to alter if the database readout_ids are not working.
    """
    tags, log_tags = _obs_tags(obs, cfgs)
    if not monitor.check("timing_on", obs.obs_id, tags=tags[0]):
        monitor.record(
            "has_tuneset", 
            [ len(obs.tunesets)==1 ], 
            [obs.timestamp], 
            tags, 
            "data_pkg", 
            log_tags=log_tags
        )
        monitor.write()


def record_timing(monitor, obs, cfgs):
    """Send a record of the timing status to the Influx QDS database
    """
    tags, log_tags = _obs_tags(obs, cfgs)

    if not monitor.check("timing_on", obs.obs_id, tags=tags[0]):
        monitor.record(
            "timing_on", 
            [obs.timing], 
            [obs.timestamp], 
            tags, 
            "data_pkg", 
            log_tags=log_tags
        )
        monitor.write()

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
    return parser

main = update_g3tsmurf_db


if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    update_g3tsmurf_db(**vars(args))
