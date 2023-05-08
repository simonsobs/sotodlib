#update_obsdb
"""
I give you a bunch of books, look through to build me an obsdb and an obsfilesb so I can then choose what I want

Use the metadata functions to manipulate the obsdb

We want to do it after the librarian has the data. If they can save the data somewhere permanent 

bkbind->imprinter make books
librarian updates its knowledge. We want to work off the librarian
/mnt/so1/shared/site-pipeline/data_pkg/books
UCSD books are the better ones for now but they are not fully complete
"""
from sotodlib.core.metadata import ObsDb, ObsFileDb
from sotodlib.core import Context
import os
import glob
import yaml
import numpy as np
import time
import argparse


def check_meta_type(bookpath):
    metapath = os.path.join(bookpath, "M_index.yaml")
    meta = yaml.safe_load(open(metapath, "rb"))
    if meta is None:
        return "empty"
    elif "type" not in meta:
        return "notype"
    else:
        return meta["type"]

def update_obsdb(base_dir, recency=2., verbosity=2, preexist_obsdb=None):
    bookcart = []
    bookcartobsdb = ObsDb()
    bookcartobsfiledb = ObsFileDb()
    if preexist_obsdb is not None:
        bookcartobsfiledb = ObsDb.from_file(os.path.join(base_dir, preexist_obsdb))
    #How far back we should look
    tnow = time.time()
    tback = tnow - recency*86400
    #Find folders that are book-like and recent
    for dirpath,_, _ in os.walk(base_dir):
        last_mod = max(os.path.getmtime(root) for root,_,_ in os.walk(dirpath))
        if last_mod<tback:#Ignore older directories
            continue
        if os.path.exists(os.path.join(dirpath, "M_index.yaml")):
            #Looks like a context file
            bookcart.append(dirpath)
    #Check the books for obsdb
    for bookpath in bookcart:
        if check_meta_type(bookpath)=="obsdb":
            context = Context(os.path.join(bookpath, "M_index.yaml"), "rb")
            single_obsdb = context.obsdb 
            singe_obsfiledb = context.obsfiledb
            bookcartobsdb.update_obs(single_obsdb)
            #bookcartobsfiledb.add_obsfile()
        else:
            bookcart.remove(bookpath)

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help="base directory from which to look for books", type=str)
    parser.add_argument('--recency', help="Days to subtract from now to set as minimum ctime",
                        default=2, type=float)
    parser.add_argument('--preexist_obsdb', help="Builds or updates database from scratch",
                        action="store", default=None)
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    return parser
def main():
    parser = get_parser(parser=None)
    args = parser.parse_args()
    update_obsdb(**vars(args))


if __name__ == '__main__':
    main()
