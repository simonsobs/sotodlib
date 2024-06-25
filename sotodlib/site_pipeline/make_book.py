import os
import traceback
import argparse
import datetime as dt
from typing import Optional
from sotodlib.io.imprinter import Imprinter

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
