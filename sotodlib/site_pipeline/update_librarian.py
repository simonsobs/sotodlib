import argparse
import datetime as dt
from typing import Optional

from sotodlib.io.imprinter import Imprinter, BOUND, UPLOADED


def main( config: str):
    """
    Update the book plan database with new data from the g3tsmurf database.

    Parameters
    ----------
    config : str
        Path to config file for imprinter
    """

    imprinter = Imprinter(
        config, db_args={'connect_args': {'check_same_thread': False}}
    )

    session = imprinter.get_session()
    to_upload = imprinter.get_bound_books(session=session)

    for book in to_upload:
        try:
            imprinter.upload_book_to_librarian(book, session=session)
        except:
            pass


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
