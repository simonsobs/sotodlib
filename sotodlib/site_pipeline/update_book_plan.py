import datetime as dt
from typing import Optional
import typer

from sotodlib.io.imprinter import Imprinter


def main(
    config: str,
    min_ctime: Optional[float] = None,
    max_ctime: Optional[float] = None,
    stream_ids: Optional[str] = None,
    force_single_stream: bool = False,
    update_delay: float = 1,
    from_scratch: bool = False
    ):
    """
    Update the book plan database with new data from the g3tsmurf database.

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    min_ctime : Optional[float], optional
        The minimum ctime to include in the book plan, by default None
    max_ctime : Optional[float], optional
        The maximum ctime to include in the book plan, by default None
    stream_ids : Optional[str], optional
        The stream ids to consider, list supplied as a comma separated string
        (e.g. "1,2,3"), by default None
    force_single_stream : bool, optional
        If True, tream multi-wafer data as if it were single wafer data, by default False
    update_delay : float, optional
        The range of time to search through g3tsmurf db for new data in units of days, by default 1
    from_scratch : bool, optional
        If True, start to search from beginning of time, by default False

    """
    if stream_ids is not None:
        stream_ids = stream_ids.split(",")
    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}})
    # leaving min_ctime and max_ctime as None will go through all available data,
    # so preferreably set them to a reasonable range based on update_delay
    if not from_scratch:
        min_ctime = dt.datetime.now() - dt.timedelta(days=update_delay)
    if isinstance(min_ctime, dt.datetime):
        min_ctime = min_ctime.timestamp()
    if isinstance(max_ctime, dt.datetime):
        max_ctime = max_ctime.timestamp()
    # obs and oper books
    imprinter.update_bookdb_from_g3tsmurf(min_ctime=min_ctime, max_ctime=max_ctime,
                                          ignore_singles=False,
                                          stream_ids=stream_ids,
                                          force_single_stream=force_single_stream)
    # hk books
    imprinter.register_hk_books()
    # smurf and stray books
    imprinter.register_timecode_books()
if __name__ == "__main__":
    typer.run(main)
