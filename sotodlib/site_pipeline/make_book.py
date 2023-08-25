from typing import Optional
import traceback
import argparse

from ..io.imprinter import Imprinter, BOUND


def main(config: str, output_root: str, source: Optional[str], book_id: Optional[str] = None, logger=None):
    """Make books based on imprinter db
    
    Parameters
    ----------
    im_config : str
        path to imprinter configuration file
    output_root : str
        root path of the books 
    source: str, optional
        data source to use, e.g., sat1, latrt, tsat. If None, use all sources
    book_id: str, optional
        optionally specify a particular book_id to bind
    """
    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}})
    # get list of books to bind: default to all unbound books
    if books_to_bind is None:
        books_to_bind = imprinter.get_unbound_books()
    else:  # optionally focus on a particular book
        book_ = imprinter.get_book(book_id)
        if book_ is None:
            raise ValueError(f"Book {book_id} not found in imprinter db")
        if book_.status == BOUND:
            raise ValueError(f"Book {book_id} is already bound")
        books_to_bind = [book_]
        
    if source is not None:
        books_to_bind = [book for book in books_to_bind if book.tel_tube == source]

    logger.info(f"Found {len(books_to_bind)} books to bind")
    for book in books_to_bind:
        logger.info(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book, output_root=output_root)
        except Exception as e:
            logger.error(f"Error binding book {book.bid}: {e}")
            logger.error(traceback.format_exc())

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
