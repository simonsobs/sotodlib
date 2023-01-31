"""
Script for running updates on (or creating) a g3tsmurf database. This setup
is specifically designed to work when the data is dynamically coming in. Meaning is 
is designed to work from something like a cronjob. 
"""
import os

import datetime as dt
import numpy as np
import argparse
import logging
from sotodlib.io.load_smurf import G3tSmurf, Observations, dump_DetDb, logger as default_logger


def update_g3tsmurf_db(config=None, detdb_filename=None, update_delay=2, from_scratch=False,
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

    SMURF = G3tSmurf.from_configs(config)

    if from_scratch:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = dt.datetime.utcfromtimestamp(int(1.6e9))
    else:
        min_time = dt.datetime.now() - dt.timedelta(days=update_delay)
        
    SMURF.index_archive(min_ctime=min_time.timestamp(), show_pb=show_pb)
    SMURF.index_metadata(min_ctime=min_time.timestamp())

    session = SMURF.Session()

    new_obs = session.query(Observations).filter(Observations.start >= min_time,
                                                 Observations.stop == None).all()
    for obs in new_obs:
        SMURF.update_observation_files(
            obs, 
            session, 
            force=True,
        )

    if detdb_filename is not None:
        dump_DetDb(SMURF, detdb_filename)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="g3tsmurf db configuration file")
    parser.add_argument('--detdb-filename', help="File for dumping the context detector database")
    parser.add_argument('--update-delay', help="Days to subtract from now to set as minimum ctime",
                        default=2, type=float)
    parser.add_argument('--from-scratch', help="Builds or updates database from scratch",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    update_g3tsmurf_db(**args)
