import argparse
import datetime as dt
from typing import Optional

from sotodlib.io.imprinter import Imprinter, BOUND, UPLOADED
from sotodlib.site_pipeline.util import init_logger

logger = init_logger(__name__, "update_librarian: ")

def main(config: str):
    """
    Update the book plan database with new data from the g3tsmurf database.

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    """

    imprinter = Imprinter(
        config, 
        db_args={'connect_args': {'check_same_thread': False}},
    )

    session = imprinter.get_session()
    to_upload = imprinter.get_bound_books(session=session)

    failed_list = []
    for book in to_upload:
        success, err = imprinter.upload_book_to_librarian(
            book, session=session, raise_on_error=False
        )
        if not success:
            failed_list.append( (book.bid, err) )
        ## don't just continually fail
        if len(failed_list) > 5:
            break
    
    if len(failed_list) != 0:
        # raise the first error so we know something is wrong
        logger.error(f"Failed to upload books {[f[0] for f in failed_list]}")
        raise failed_list[0][1]




def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help="imprinter configuration file"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
