"""
Script for running updates on (or creating) a g3tsmurf database. This setup
is specifically designed to work when the data is dynamically coming in. Meaning is 
is designed to work from something like a cronjob. 
"""
import os
from typing import Optional
from pathlib import Path
import yaml
import datetime as dt
import numpy as np
import argparse
import logging
from sqlalchemy import not_, or_, and_, desc

from sotodlib.io.g3thk_db import G3tHk, HKFiles, logger


def core(config: Optional[str]=None, from_scratch: bool=False, verbosity: int=2):

    show_pb = True if verbosity > 1 else False

    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    HK = G3tHk.from_configs(config)

    if from_scratch or HK.session.query(HKFiles).count()==0:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = int(1.6e9)
    else:
        ## start at the last file in the database
        last_file = HK.session.query(HKFiles)
        last_file = last_file.order_by(desc(HKFiles.global_start_time)).first()

        logger.info(f"Starting from last file in database: {last_file.filename}")
        min_time = last_file.global_start_time - 10
        logger.debug(f"Setting minium time to {min_time}")

    HK.add_hkfiles(min_ctime=min_time, show_pb=show_pb)

def main(config: Optional[str]=None, from_scratch: bool=False, verbosity: int=2,
         profile: bool=False, profile_output: Optional[Path]=None):

    if profile:
        import pyinstrument
        timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"update_g3thk_database_{timestamp}.html"
        output_filename = profile_output / filename if profile_output is not None else filename
        profiler = pyinstrument.Profiler()
        profiler.start()
    
    try:
        core(
            config=config, from_scratch=from_scratch, verbosity=verbosity
        )
    finally:
        if profile:
            profiler.stop()
            if profile_output is not None:
                with open(output_filename, "w") as f:
                    f.write(profiler.output_html())


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="g3tsmurf db configuration file")
    parser.add_argument('--from-scratch', help="Builds or updates database from scratch",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    parser.add_argument("--profile", help="Run with pyinstrument profiling", action="store_true")
    parser.add_argument("--profile-output", help="Directory to output pyinstrument profiling results to, if --profile is set", 
                        type=Path, default=Path("/data/data-package/profiles"))
    
    return parser


if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
