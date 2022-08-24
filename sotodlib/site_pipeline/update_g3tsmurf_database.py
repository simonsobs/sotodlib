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
from sotodlib.io.load_smurf import G3tSmurf, Observations, dump_DetDb, logger

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_prefix', help="The prefix to data locations, use individual"
                                            "locations flags if data is stored in non-standard folders")
    parser.add_argument('db_path', help="Path to Database to update")
    
    parser.add_argument('--timestream-folder', help="Absolute path to folder with .g3 timestreams. Overrides data_prefix")
    parser.add_argument('--smurf-folder', help="Absolute path to folder with pysmurf archived data. Overrides data_prefix")
    
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
    
    if args.timestream_folder is None:
        args.timestream_folder = os.path.join( args.data_prefix, 'timestreams/')
       
    if args.smurf_folder is None:
        args.smurf_folder = os.path.join( args.data_prefix, 'smurf/')
    
    if args.verbosity > 1:
        show_pb=True
    else:
        show_pb=False
        
    if args.verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbosity == 2:
        logger.setLevel(logging.INFO)
    elif args.verbosity == 3:
        logger.setLevel(logging.DEBUG)
        
    SMURF = G3tSmurf(args.timestream_folder, 
                     db_path=args.db_path,
                     meta_path=args.smurf_folder)

    if args.from_scratch:
        print("Building Database from Scratch, May take awhile")
        min_time = dt.datetime.utcfromtimestamp(int(1.6e9))
    else:
        min_time = dt.datetime.now() - dt.timedelta(days=args.update_delay)
        
    SMURF.index_archive(min_ctime=min_time.timestamp(), show_pb=show_pb)
    SMURF.index_metadata(min_ctime=min_time.timestamp())

    session = SMURF.Session()

    new_obs = session.query(Observations).filter( Observations.start >= min_time,
                                                  Observations.stop == None).all()
    for obs in new_obs:
        SMURF.update_observation_files(
            obs, 
            session, 
            force=True,
        )

    if args.detdb_filename is not None:
        dump_DetDb(SMURF, args.detdb_filename)

