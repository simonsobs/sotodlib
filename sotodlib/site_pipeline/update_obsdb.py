"""update_obsdb.py

Create and/or update an obsdb and obsfiledb based on some books.
The config file could be of the form:

.. code-block:: yaml

    base_dir: path_to_base_directory
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
    tolerate_stray_files: True 
    extra_extra_files:
    - Z_bookbinder_log.txt
    extra_files:
    - M_index.yaml
    - M_book.yaml

"""

from sotodlib.core.metadata import ObsDb
from sotodlib.core import Context 
from sotodlib.site_pipeline.check_book import main as checkbook
import os
import glob
import yaml
import numpy as np
import time
import argparse
from typing import Optional

def check_meta_type(bookpath):
    metapath = os.path.join(bookpath, "M_index.yaml")
    meta = yaml.safe_load(open(metapath, "rb"))
    if meta is None:
        return "empty"
    elif "type" not in meta:
        return "notype"
    else:
        return meta["type"]

def main(config:str, 
        recency:float=None, 
        booktype:Optional[str]="both",
        verbosity:Optional[int]=2,
        overwrite:Optional[bool]=False):
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
    """
    bookcart = []
    bookcartobsdb = ObsDb()

    if booktype=="both":
        accept_type = ["obs", "oper"]
    else:
        accept_type = [booktype]

    config_dict = yaml.safe_load(open(config, "r"))
    try:
        base_dir = config_dict["base_dir"]
    except KeyError:
        if verbosity==0:
            print("No base directory base_dir specified in config file!")
    if "obsdb" in config_dict:
        if os.path.isfile(config_dict["obsdb"]):
            bookcartobsdb = ObsDb.from_file(config_dict["obsdb"])
    if "obsdb_cols" in config_dict:
        col_list = []
        for col, typ in config_dict["obsdb_cols"].items():
            col_list.append(col+" "+typ)
        bookcartobsdb.add_obs_columns(col_list)

    #How far back we should look
    tnow = time.time()
    if recency is not None:
        tback = tnow - recency*86400
    else:
        tback = 0 #Back to the UNIX Big Bang 
    
    existing = bookcartobsdb.query()["obs_id"]

    #Find folders that are book-like and recent
    for dirpath,_, _ in os.walk(base_dir):
        last_mod = max(os.path.getmtime(root) for root,_,_ in os.walk(dirpath))
        if last_mod<tback:#Ignore older directories
            continue
        if os.path.exists(os.path.join(dirpath, "M_index.yaml")):
            _, book_id = os.path.split(dirpath)
            if book_id in existing and not overwrite:
                continue
            #Looks like a book folder
            bookcart.append(dirpath)
    #Check the books for the observations we want

    for bookpath in bookcart:
        if check_meta_type(bookpath) in accept_type:

            try:
                #obsfiledb creation
                checkbook(bookpath, config, add=True, overwrite=True)
            except Exception as e:
                if config_dict["skip_bad_books"]:
                    print(f"failed to add {bookpath}")
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

            start = index.get("start_time")
            end = index.get("end_time") 
            if None not in [start, end]:
                bookcartobsdb.add_obs_columns(["timestamp float", "duration float"])
                very_clean["timestamp"] = start
                very_clean["duration"] = end - start
            
            if tags != [] and tags != [""]:
                bookcartobsdb.update_obs(obs_id, very_clean, tags=tags)
            else:
                bookcartobsdb.update_obs(obs_id, very_clean)
           
        else:
            bookcart.remove(bookpath)
    if "obsdb" in config_dict:
        bookcartobsdb.to_file(config_dict["obsdb"])
    else:
        bookcartobsdb.to_file("obsdb_from{}_to{}".format(tback, tnow))


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
