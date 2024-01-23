"""update_obsdb.py

Create and/or update an obsdb and obsfiledb based on some books.
The config file could be of the form:

.. code-block:: yaml

    base_dir: path_to_base_directories. Can be a list or a single string.
    obsdb_cols:
      start_time: float
      stop_time: float
      n_samples: int
      telescope: str
      tube_slot: str
      type: str
      subtype: str

    obsdb: dummyobsdb.sqlite
    obsfiledb: dummyobsfiledb.sqlite
    lat_tube_list_dir: path to lat_tube_list.yaml,a dict matching tubes and bands
    tolerate_stray_files: True
    skip_bad_books: True
    extra_extra_files:
    - Z_bookbinder_log.txt
    extra_files:
    - M_index.yaml
    - M_book.yaml

"""

from sotodlib.core.metadata import ObsDb
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

def check_meta_type(bookpath: str):
    metapath = os.path.join(bookpath, "M_index.yaml")
    meta = yaml.safe_load(open(metapath, "rb"))
    if meta is None:
        return "empty"
    elif "type" not in meta:
        return "notype"
    else:
        return meta["type"]


def telescope_lookup(telescope: str):
    """
    Set a number of common queries given a telescope name

    Arguments
    ----------
    telescope : str
        Name of telescope in M_index

    """
    if telescope == "sat" or telescope == "satp1":
        return {"telescope": "satp1", "telescope_flavor": "sat",
                "tube_flavor": "mf", "detector_flavor": "tes"}
    elif telescope == "satp2":
        return {"telescope": "satp2","telescope_flavor": "sat",
                "tube_flavor": "uhf", "detector_flavor": "tes"}
    elif telescope == "satp3":
        return {"telescope": "satp3", "telescope_flavor": "sat",
                "tube_flavor": "mf", "detector_flavor": "tes"}
    elif telescope == "lat":
        return {"telescope": "lat", "telescope_flavor": "lat"}
    else:
        logger.error("unknown telescope type given by bookbinder")
        return {}


def main(config: str, 
        recency: float = None, 
        booktype: Optional[str] = "both",
        verbosity: Optional[int] = 2,
        overwrite: Optional[bool] = False,
        logger=None):

    """
    Create or update an obsdb for observation or operations data.

    Arguments
    ----------
    config : str
        Path to config file
    recency : float
        How far back in time to look for databases, in days. If None, 
        goes back to the UNIX start date (default: None)
    booktype : str
        Look for observations or operations data or both (default: both)
    verbosity : int
        Output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug
    overwrite : bool
        if False, do not re-check existing entries
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

    logger.info("Updating obsdb")
    bookcart = []

    if booktype not in ["obs", "oper", "both"]:
        logger.warning("Specified booktype inadapted to update_obsdb")
    
    if booktype == "both":
        accept_type = ["obs", "oper"]
    else:
        accept_type = [booktype]

    config_dict = yaml.safe_load(open(config, "r"))
    try:
        base_dir = config_dict["base_dir"]
    except KeyError:
        logger.error("No base directory base_dir specified in config file!")

    new_obsdb = True
    if "obsdb" in config_dict:
        if os.path.isfile(config_dict["obsdb"]):
            bookcartobsdb = ObsDb(map_file=config_dict["obsdb"])
            new_obsdb = False
        else:
            logger.error("No obsdb at the indicated location")
            bookcartobsdb = ObsDb()
    else:
        logger.warning("No obsdb named in the configuration file")
        bookcartobsdb = ObsDb()
        
    if "obsdb_cols" in config_dict:
        col_list = []
        for col, typ in config_dict["obsdb_cols"].items():
            col_list.append(col+" "+typ)
        bookcartobsdb.add_obs_columns(col_list)
    if "skip_bad_books" not in config_dict:
        config_dict["skip_bad_books"] = False
        
    #How far back we should look
    tnow = time.time()
    if recency is not None:
        tback = tnow - recency*86400
    else:
        tback = 0 #Back to the UNIX Big Bang 
    
    existing = bookcartobsdb.query()["obs_id"]
    #Check if there are one or multiple base_dir specified
    if isinstance(base_dir,str):
        base_dir = [base_dir]
    for bd in base_dir:
        #Find folders that are book-like and recent
        for dirpath, _, _ in os.walk(bd):
            last_mod = max(os.path.getmtime(root) for root, _, _ in os.walk(dirpath))
            if last_mod < tback:#Ignore older directories
                continue
            if os.path.exists(os.path.join(dirpath, "M_index.yaml")):
                _, book_id = os.path.split(dirpath)
                if book_id in existing and not overwrite:
                    continue
                #Looks like a book folder
                bookcart.append(dirpath)
    #Check the books for the observations we want


    for bookpath in sorted(bookcart):
        if check_meta_type(bookpath) in accept_type:

            try:
                #obsfiledb creation
                checkbook(
                    bookpath, config, add=True, 
                    overwrite=True, logger=logger
                )
            except Exception as e:
                if config_dict["skip_bad_books"]:
                    logger.warning(f"failed to add {bookpath}")
                    continue
                else:
                    raise e

            index = yaml.safe_load(open(os.path.join(bookpath, "M_index.yaml"), "rb"))
            obs_id = index.pop("book_id")
            tags = index.pop("tags")
            detsets = index.pop("detsets")

            if "obsdb_cols" in config_dict:
                very_clean = {col:index[col] for col in iter(config_dict["obsdb_cols"]) if col in index}
            else:
                col_list = []
                clean = {key:val for key, val in index.items() if val is not None}
                very_clean = {key:val for key, val in clean.items() if type(val) is not list}
                for key, val in very_clean.items():
                    col_list.append(key+" "+type(val).__name__)
                bookcartobsdb.add_obs_columns(col_list)
            if "skip_bad_books" not in config_dict:
                config_dict["skip_bad_books"] = False
            #Adding info that should be there for all observations
            #Descriptive string columns
            try:
                telescope = index["telescope"]
                flavors = telescope_lookup(telescope)
                for flav in flavors:
                    bookcartobsdb.add_obs_columns([flav+" str"])
                    very_clean[flav] = flavors[flav]
                if telescope == "lat":
                   lat_tube_list_dir = config_dict["lat_tube_list_dir"]
                   lat_tube_list = yaml.safe_load(open(os.path.join(lat_tube_list_dir, "lat_tube_list.yaml"), "rb"))
                   tube_flavor = lat_tube_list[index["tube_slot"]]
                   bookcartobsdb.add_obs_columns("tube_flavor str")
                   very_clean["tube_flavor"] = tube_flavor

            except KeyError:
                logger.error("No telescope key in index file or error with lat_tube_list")
                very_clean["telescope_flavor"] = "unknown"
            stream_ids = index.pop("stream_ids")
            if stream_ids is not None:
                bookcartobsdb.add_obs_columns(["wafer_count int"])
                very_clean["wafer_count"] = len(stream_ids)

            #Time
            try:
                start = index["start_time"]
                end = index["stop_time"] 
                bookcartobsdb.add_obs_columns(["timestamp float", "duration float"])
                very_clean["timestamp"] = start
                very_clean["duration"] = end - start
            except KeyError:
                logger.error("Incomplete timing information for obs_id {obs_id}")

            #Scanning motion
            stream_file = os.path.join(bookpath,"*{}*.g3".format(stream_ids[0]))
            stream = load_book.load_book_file(stream_file, no_signal=True)

            for coor in ["az", "el"]:
                try:
                    coor_enc = stream.ancil[coor+"_enc"]
                    bookcartobsdb.add_obs_columns([f"{coor}_center float", 
                                                   f"{coor}_throw float"])
                    very_clean[f"{coor}_center"] = .5 * (coor_enc.max() + coor_enc.min())
                    very_clean[f"{coor}_throw"] = .5 * (coor_enc.max() - coor_enc.min())
                except KeyError:
                    logger.error(f"No {coor} pointing in some streams for obs_id {obs_id}")

            try:
                if very_clean["telescope_flavor"] == "sat":
                    bore_enc = stream.ancil["boresight_enc"]
                    very_clean["roll_center"] = -.5 * (bore_enc.max() + bore_enc.min())
                    very_clean["roll_throw"] = .5 * (bore_enc.max() - bore_enc.min())
                if very_clean["telescope_flavor"] == "lat":
                    el_enc = stream.ancil["el_enc"]
                    corot_enc = stream.ancil["corotator_enc"]
                    roll = el_enc - 60. - corot_enc
                    very_clean["roll_center"] = .5 * (roll.max() + roll.min())
                    very_clean["roll_throw"] = .5 * (roll.max() - roll.min())

                bookcartobsdb.add_obs_columns(["roll_center float", "roll_throw float"])
            except KeyError:
                logger.error(f"Unable to compute roll for obs_id {obs_id}")
                

            # Make sure no invalid tags before update.
            tags = [t.strip() for t in tags if t.strip() != '']

            bookcartobsdb.update_obs(obs_id, very_clean, tags=tags)
           
        else:
            bookcart.remove(bookpath)
    if new_obsdb:
        if "obsdb" in config_dict:
            bookcartobsdb.to_file(config_dict["obsdb"])
        else:
            bookcartobsdb.to_file("obsdb_from_{}_to_{}.sqlite".format(int(tback), int(tnow)))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="ObsDb, ObsfileDb configuration file", 
        type=str, required=True)
    parser.add_argument('--recency', default=None, type=float,
        help="Days to subtract from now to set as minimum ctime. If None, no minimum")
    parser.add_argument("--verbosity", default=2, type=int,
        help="Increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug")
    parser.add_argument("--booktype", default="both", type=str,
        help="Select book type to look for: obs, oper, both(default)")
    parser.add_argument("--overwrite", action="store_true",
        help="If true, writes over existing entries")
    return parser



if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
