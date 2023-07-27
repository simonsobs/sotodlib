from typing import Optional
import traceback
import argparse

from ..io.imprinter import Imprinter


def main(config: str, output_root: str, source: Optional[str], logger=None):
    """Make books based on imprinter db
    
    Parameters
    ----------
    im_config : str
        path to imprinter configuration file
    output_root : str
        root path of the books 
    source: str, optional
        data source to use, e.g., sat1, latrt, tsat. If None, use all sources
    """
    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}})
    # get unbound books
    unbound_books = imprinter.get_unbound_books()
    failed_books = imprinter.get_failed_books()
    if source is not None:
        unbound_books = [book for book in unbound_books if book.tel_tube == source]
        failed_books = [book for book in failed_books if book.tel_tube == source]
    print(f"Found {len(unbound_books)} unbound books and {len(failed_books)} failed books")
    for book in unbound_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())

    print("Retrying previously failed books") 
    for book in failed_books:
        print(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            print(f"Error binding book {book.bid}: {e}")
            print(traceback.format_exc())
            # it has failed twice, ideally we want people to look at it now
            # do something here


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to imprinter configuration file")
    parser.add_argument('output_root', type=str, help="Root path of the books")
    parser.add_argument('--source', type=str, help="Data source to use, e.g., sat1, latrt, tsat. If None, use all sources")
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
