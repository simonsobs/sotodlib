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
from sqlalchemy import not_, or_, and_, desc

from sotodlib.io.g3thk_db import G3tHk, HKFiles, logger as default_logger


def main(config=None, from_scratch=False, verbosity=2, logger=None):

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
    HK = G3tHk.from_configs(cfgs)

    if from_scratch or HK.session.query(HKFiles).count()==0:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = int(1.6e9)
    else:
        ## start at the last file in the database
        last_file = HK.session.query(HKFiles)
        last_file = last_file.order_by(desc(HKFiles.global_start_time)).first()
        min_time = last_file.global_start_time - 10

    HK.add_hkfiles(min_ctime=min_time, show_pb=show_pb)

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="g3tsmurf db configuration file")
    parser.add_argument('--from-scratch', help="Builds or updates database from scratch",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    return parser



if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
