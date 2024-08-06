import datetime as dt
from typing import Optional
import argparse

from sotodlib.io.imprinter import Imprinter


def main(
    config: str,
    cleanup_delay: float = 7,
    max_ctime: Optional[float] = None,
    dry_run: Optional[bool] = False,
    ):
    """
    Use the imprinter database to clean up already bound level 2 files. 

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    cleanup_delay : float, optional
        The amount of time to delay book deletion in units of days, by default 1
    max_ctime : Optional[datetime], optional
        The maximum datetime to delete level 2 data. Overrides cleanup_delay.
    dry_run : Optional[bool], 
        If true, only prints deletion to logger
    """

    if max_ctime is not None:
        max_time = dt.datetime.utcfromtimestamp(max_ctime)
    else:
        max_time = None

    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}})
    book_list = imprinter.get_level2_deleteable_books(max_time=max_time, cleanup_delay=cleanup_delay)

    for book in book_list:
        imprinter.delete_level2_files(book, dry_run=dry_run)

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Config file for Imprinter")
    parser.add_argument('--cleanup-delay', type=float, default=7, 
        help="Days to keep level 2 data before cleaning")
    parser.add_argument('--max-ctime', type=float, 
        help="Maximum ctime to delete to, overrides cleanup_delay ONLY if its an earlier time")
    parser.add_argument('--dry-run', action="store_true", 
        help="if passed, only prints delete behavior")
    return parser

if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
