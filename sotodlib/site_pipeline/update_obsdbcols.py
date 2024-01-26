"""update_obsdbcols.py

Add columns to entries of an obsdb for books that are also in the corresponding obsfiledb.
The config file could be of the form:

.. code-block:: yaml
    obsdb_extra_cols: List or single string with form {name}: {type}
    obsdb: dummyobsdb.sqlite
    obsfiledb: dummyobsfiledb.sqlite
"""

from sotodlib.core.metadata import ObsDb, ObsFileDb
from sotodlib.core import Context 
from sotodlib.site_pipeline.check_book import main as checkbook
from sotodlib.io import load_book
import os
import glob
import yaml
import numpy as np
import time
import argparse
import logging
from sotodlib.site_pipeline import util
from typing import Optional

logger = util.init_logger(__name__, 'update-obsdb: ')

def main(config: str,
        complete_obsdb: Optional[bool] = False,
        recency: float = None, 
        verbosity: Optional[int] = 2,
        logger=None):

    """
    Create or update an obsdb for observation or operations data.

    Arguments
    ----------
    config : str
        Path to config file
    complete_obsdb : bool
        Whether to add to the ObsDb books found in ObsFileDb but not 
        already in the ObsDb.
    recency : float
        How far back in time to look for databases, in days. If None, 
        goes back to the UNIX start date (default: None)
    verbosity : int
        Output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug
    logger : logging
        When to output the print statements

    """

    if logger is None:
        logger = globals()['logger']
    else:
        globals()['logger'] = logger
    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    logger.info("Updating existing obsdb with new columns")

    config_dict = yaml.safe_load(open(config, "r"))

    if "obsdb" in config_dict:
        obsdb = ObsDb(map_file=config_dict["obsdb"])
    else:
        logger.error("No obsdb named in the configuration file")
        return
    if complete_obsdb:
        obsdb_keys = list(obsdb.info()["fields"].keys())
        
    if "obsfiledb" in config_dict:
        obsfiledb = ObsFileDb(map_file=config_dict["obsfiledb"])
    else:
        logger.error("No obsfiledb specified in the configuration file")
        return
    
    if "obsdb_extra_cols" in config_dict:
        col_list = []
        for col, typ in config_dict["obsdb_extra_cols"].items():
            col_list.append(col+" "+typ)
        obsdb.add_obs_columns(col_list)
    else: 
        logger.error("No obsdb_extra_cols specified in the configuration file")
        return


    #How far back we should look
    tnow = time.time()
    if recency is not None:
        tback = tnow - recency*86400
    else:
        tback = 0 #Back to the UNIX Big Bang 

    obs_ids = obsfiledb.get_obs()
    for obs_id in obs_ids:
        obs = obsdb.get(obs_id)
        if (obs is None) and (not complete_obsdb):
            logger.info(f"{obs_id} in obsfiledb but not in obsdb, skipping")
            continue
        if (obs is not None) and (obs["start_time"]<tback):
            logger.info(f"{obs_id} starts too far in the past, skipping")
            continue

        obs_files = obsfiledb.get_files(obs_id)
        first_file_key = next(iter(obs_files))
        book_path = os.path.dirname(obs_files[first_file_key][0][0])
        if os.path.exists(os.path.join(book_path, "M_index.yaml")):
            index = yaml.safe_load(open(os.path.join(book_path, "M_index.yaml"), "rb"))
            if (obsdb.get(obs_id) is None) and complete_obsdb and (index["start_time"]>=tback):
                existing_cols_dict = {key:index[key] for key in iter(obsdb_keys) if key in index}
                obsdb.update_obs(obs_id, existing_cols_dict)
                logger.info(f"Added {obs_id} to obsdb")

            new_cols_dict = {col:index[col] for col in iter(config_dict["obsdb_extra_cols"]) if col in index}
            obsdb.update_obs(obs_id, new_cols_dict)
            logger.debug(f"Added {new_cols_dict.items()} to {obs_id}")
        else:
            logger.error(f"No index file found for book {obs_id}")

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="ObsDb, ObsfileDb configuration file", 
        type=str, required=True)
    parser.add_argument('--recency', default=None, type=float,
        help="Days to subtract from now to set as minimum ctime. If None, no minimum")
    parser.add_argument('--complete_obsdb', action="store_true",
        help="Whether to add to the ObsDb books found in ObsFileDb but not already in the ObsDb.")
    parser.add_argument("--verbosity", default=2, type=int,
        help="Increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug")
    return parser



if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
