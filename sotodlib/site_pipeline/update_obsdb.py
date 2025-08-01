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
    lat_tube_list_file: path to yaml dict matching tubes and bands
    tolerate_stray_files: True
    skip_bad_books: True
    known_bad_books_file: path to \n-separated file listing bad books 
    extra_extra_files:
    - Z_bookbinder_log.txt
    extra_files:
    - M_index.yaml
    - M_book.yaml

"""

from sotodlib.core.metadata import ObsDb
from sotodlib.core import Context 
from sotodlib.site_pipeline import check_book
from sotodlib.io import load_book
import os
import glob
import yaml
import re
import numpy as np
import time
import argparse
import logging
from sotodlib.site_pipeline import util
from typing import Optional
from itertools import product

logger = util.init_logger('update_obsdb', 'update-obsdb: ')

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
        return {"telescope": "lat", "telescope_flavor": "lat",
                "detector_flavor": "tes"}
    else:
        logger.error("unknown telescope type given by bookbinder")
        return {}


def main(config: str, 
        recency: float = None, 
        booktype: Optional[str] = "both",
        verbosity: Optional[int] = 2,
        overwrite: Optional[bool] = False,
        fastwalk: Optional[bool] = False):

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
    fastwalk : bool
        if True, assume the directories have a structure /base_dir/obs|oper/\d{5}/...
        Then replace base_dir with only the directories where \d{5} is greater or 
        equal to recency.
    """
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

    if "obsdb" in config_dict:
        bookcartobsdb = ObsDb(map_file=config_dict["obsdb"])
    else:
        logger.warning("No obsdb named in the configuration file")
        bookcartobsdb = ObsDb("obsdb.sqlite")
        
    if "obsdb_cols" in config_dict:
        col_list = []
        for col, typ in config_dict["obsdb_cols"].items():
            col_list.append(col+" "+typ)
        bookcartobsdb.add_obs_columns(col_list)
    if "skip_bad_books" not in config_dict:
        config_dict["skip_bad_books"] = False
    
    config_dict["known_bad_books"] = []
    if "known_bad_books_file" in config_dict:
        try:
            with open(config_dict["known_bad_books_file"], "r") as bbf:
                config_dict["known_bad_books"] = bbf.read().split("\n")
        except:
            raise IOError("Bad books file couldn't be read in")
        
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
    if fastwalk:
        abv_tback = int(f"{int(tback):05}"[:5]) #Make sure we have at least five chars
        abv_tnow = int(f"{int(tnow):05}"[:5])
        abv_codes = np.arange(abv_tback, abv_tnow+1)
        #Build the combinations base_dir/booktype/\d{5}
        base_dir = [f"{os.path.join(x[0], x[1], str(x[2]))}" for x in product(base_dir, accept_type, abv_codes)]
        logger.info(f"Looking in the following directories only: {str(base_dir)}")

    for bd in base_dir:
        #Find folders that are book-like and recent
        for dirpath, _, _ in os.walk(bd):
            if os.path.exists(os.path.join(dirpath, "M_index.yaml")):
                _, book_id = os.path.split(dirpath)
                if book_id in existing and not overwrite:
                    continue
                if book_id in config_dict["known_bad_books"]:
                    logger.debug(f"{book_id} known to be bad, skipping it")
                    continue
                edit_time = os.path.getmtime(dirpath)
                if edit_time > tback:
                    #Looks like a book folder and edited recently enough
                    bookcart.append(dirpath)
    
    logger.info(f"Found {len(bookcart)} new books in {time.time()-tnow} s")
    #Check the books for the observations we want
    bad_book_counter = 0
    for bookpath in sorted(bookcart):
        if check_meta_type(bookpath) in accept_type:
            t1 = time.time()
            logger.info(f"Examining book at {bookpath}")
            try:
                #obsfiledb creation
                ok, obsfiledb_info = check_book.scan_book_dir(
                    bookpath, logger, config_dict, prep_obsfiledb=True)
                if not ok:
                    raise RuntimeError("check_book found fatal errors, not adding.")
                check_book.add_to_obsfiledb(
                    obsfiledb_info, logger, config_dict, overwrite=True)
                logger.info(f"Ran check_book in {time.time()-t1} s")
            except Exception as e:
                if config_dict["skip_bad_books"]:
                    config_dict["known_bad_books"].append(book_id)
                    logger.error(f"failed to add {bookpath}. There are now {len(config_dict['known_bad_books'])} known bad books.")
                    bad_book_counter +=1
                    if "known_bad_books_file" in config_dict:
                        with open(config_dict["known_bad_books_file"], "w") as bbf:
                            bbf.write("\n".join(config_dict["known_bad_books"]))
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

            #Adding info that should be there for all observations
            #Descriptive string columns
            try:
                telescope = index["telescope"]
                flavors = telescope_lookup(telescope)
                for flav in flavors:
                    bookcartobsdb.add_obs_columns([flav+" str"])
                    very_clean[flav] = flavors[flav]
                if telescope == "lat":
                   lat_tube_list = yaml.safe_load(
                       open(config_dict["lat_tube_list_file"], "rb")
                   )
                   tube_flavor = lat_tube_list[index["tube_slot"]]
                   bookcartobsdb.add_obs_columns("tube_flavor str")
                   very_clean["tube_flavor"] = tube_flavor

            except KeyError:
                logger.error("No telescope key in index file or error with lat_tube_list")
                very_clean["telescope_flavor"] = "unknown"
            
            #Stream_ids and wafers
            try:
                stream_ids = index["stream_ids"]
                bookcartobsdb.add_obs_columns(["wafer_count int"])
                very_clean["wafer_count"] = len(stream_ids)
                wafer_slots = index["wafer_slots"]
                if len(wafer_slots) < len(stream_ids):
                    logger.error("Missing info on some stream_ids")
                    continue
                bookcartobsdb.add_obs_columns(["wafer_slots_list str", "stream_ids_list str"])
                wafer_slots_list = ""
                stream_ids_list = ",".join(stream_ids)
                for slot in wafer_slots:
                    if slot["stream_id"] in stream_ids:
                        wafer_slots_list += slot["wafer_slot"]+","
                very_clean["wafer_slots_list"] = wafer_slots_list[:-1]#Eliminate last comma
                very_clean["stream_ids_list"] = stream_ids_list
            except KeyError:
                logger.error("Unable to find stream_ids or wafer slots")

            #Time
            try:
                start = index["start_time"]
                end = index["stop_time"] 
                bookcartobsdb.add_obs_columns(["timestamp float", "duration float"])
                very_clean["timestamp"] = start
                very_clean["duration"] = end - start
            except KeyError:
                logger.error("Incomplete timing information for obs_id {obs_id}")
            
            #SAT HWP
            if very_clean["telescope_flavor"] == "sat":
                try:
                    very_clean["hwp_freq_mean"] = index["hwp_freq_mean"]
                    very_clean["hwp_freq_stdev"] = index["hwp_freq_stdev"]
                    bookcartobsdb.add_obs_columns(["hwp_freq_mean float", 
                                                   "hwp_freq_stdev float"])
                except KeyError:
                    logger.error(f"No HWP frequency info for obs_id {obs_id}")

            #Scanning motion
            stream_file = os.path.join(bookpath,"*{}*.g3".format(stream_ids[0]))
            stream = load_book.load_book_file(stream_file, no_signal=True)

            for coor in ["az", "el"]:
                try:
                    coor_enc = stream.ancil[coor+"_enc"]
                    bookcartobsdb.add_obs_columns([f"{coor}_center float", 
                                                   f"{coor}_throw float"])
                    very_clean[f"{coor}_center"] = round(.5 * (coor_enc.max() + coor_enc.min()), 4)
                    very_clean[f"{coor}_throw"] = round(.5 * (coor_enc.max() - coor_enc.min()), 4)
                except KeyError:
                    logger.error(f"No {coor} pointing in some streams for obs_id {obs_id}")

            try:
                if very_clean["telescope_flavor"] == "sat":
                    bore_enc = stream.ancil["boresight_enc"]
                    very_clean["roll_center"] = round(-.5 * (bore_enc.max() + bore_enc.min()), 4)
                    very_clean["roll_throw"] = round(.5 * (bore_enc.max() - bore_enc.min()), 4)
                if very_clean["telescope_flavor"] == "lat":
                    el_enc = stream.ancil["el_enc"]
                    corot_enc = stream.ancil["corotator_enc"]
                    roll = el_enc - 60. - corot_enc
                    very_clean["roll_center"] = round(.5 * (roll.max() + roll.min()), 4)
                    very_clean["roll_throw"] = round(.5 * (roll.max() - roll.min()), 4)

                bookcartobsdb.add_obs_columns(["roll_center float", "roll_throw float"])
            except KeyError:
                logger.error(f"Unable to compute roll for obs_id {obs_id}")
                

            # Make sure no invalid tags before update.
            tags = [t.strip() for t in tags if t.strip() != '']

            bookcartobsdb.update_obs(obs_id, very_clean, tags=tags)
            logger.info(f"Finished {obs_id} in {time.time()-t1} s")
        else:
            bookcart.remove(bookpath)
    if bad_book_counter != 0:
        logger.error(f"Found {bad_book_counter} new bad books, There are now {len(config_dict['known_bad_books'])} known bad books.")
        raise(Exception)

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="ObsDb, ObsfileDb configuration file", 
        type=str, required=True)
    parser.add_argument("--recency", default=None, type=float,
        help="Days to subtract from now to set as minimum ctime. If None, no minimum")
    parser.add_argument("--verbosity", default=2, type=int,
        help="Increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug")
    parser.add_argument("--booktype", default="both", type=str,
        help="Select book type to look for: obs, oper, both(default)")
    parser.add_argument("--overwrite", action="store_true",
        help="If true, writes over existing entries")
    parser.add_argument("--fastwalk", action="store_true",
        help="Assume known directory tree shape and speed up walkthrough")
    return parser



if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
