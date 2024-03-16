import os
import traceback
import argparse
import datetime as dt
from typing import Optional
from sotodlib.io.imprinter import Imprinter

def make_lock(fname):
    if os.path.exists(fname):
        raise ValueError(f"Tried to make lockfile {fname} which already"
                          " exists")
    with open(fname, 'w') as f:
        print(f"writing lock file {fname}")
        f.write(str(dt.datetime.now().timestamp()))

def check_lock(fname, timeout):
    if not os.path.exists(fname):
        return True
    with open(fname, 'r') as f:
        t = float(f.readline())
    if dt.datetime.now().timestamp() > t + timeout*3600:
        raise ValueError(f"lockfile {fname} is over {timeout} hours old")
    return False

def remove_lock(fname):
    if not os.path.exists(fname):
        raise ValueError(f"lockfile {fname} does not exist at removal?")
    print(f"removing lock file {fname}")
    os.remove(fname)

def main(config: str):
    """Make books based on imprinter db
    
    Parameters
    ----------
    config : str
        path to imprinter configuration file
    """
    imprinter = Imprinter(
        config, 
        db_args={'connect_args': {'check_same_thread': False}}
    )
    # lockname will be unique even if two imprinter configs are in 
    # the same folder
    b,f = os.path.split(config)
    l = ".make_book." + f.replace(".yaml", ".lock")
    lockname = os.path.join(b,l)

    if not check_lock(lockname, 6):
        imprinter.logger.warning("Not running make_book because of lockfile")
        return
    make_lock(lockname)


    # get unbound books
    unbound_books = imprinter.get_unbound_books()
    already_failed_books = imprinter.get_failed_books()
    
    print(f"Found {len(unbound_books)} unbound books and "
        f"{len(already_failed_books)} failed books")
    for book in unbound_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())

    print("Retrying failed books") 
    failed_books = imprinter.get_failed_books()
    for book in failed_books:
        if book in already_failed_books:
            print(f"Book {book.bid} has already failed twice, not re-trying")
            continue
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())
            # it has failed twice, ideally we want people to look at it now
            # do something here

    remove_lock(lockname)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', 
        type=str, 
        help="Path to imprinter configuration file"
    )
    parser.add_argument('output_root', type=str, help="Root path of the books")
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
