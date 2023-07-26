import datetime as dt
from typing import Optional
import typer

from sotodlib.io.imprinter import Imprinter


def main(
    config: str,
    cleanup_delay: float = 7,
    max_time: Optional[dt.datetime] = None,
    ):
    """
    Use the imprinter database to clean up already bound level 2 files. 

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    cleanup_delay : float, optional
        The amount of time to delay book deletion in units of days, by default 1
    max_time : Optional[datetime], optional
        The maximum datetime to delete level 2 data. Overrides cleanup_delay
    """

    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}})
    max_time = dt.datetime.now() - dt.timedelta(days=cleanup_delay)

    book_list = imprinter.get_level2_deleteable_books(max_time=max_time)

    for book in book_list:
        imprinter.delete_level2_files(book, dry_run=False)

if __name__ == "__main__":
    typer.run(main)
