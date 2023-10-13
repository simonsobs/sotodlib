from typing import Optional
import traceback
import argparse

from sotodlib.io.imprinter import Imprinter


def main(config: str, logger=None):
    """Make books based on imprinter db
    
    Parameters
    ----------
    im_config : str
        path to imprinter configuration file
    """
    imprinter = Imprinter(config, db_args={'connect_args': {'check_same_thread': False}}, logger=logger)
    # get unbound books
    unbound_books = imprinter.get_unbound_books()
    already_failed_books = imprinter.get_failed_books()
    
    logger.info(f"Found {len(unbound_books)} unbound books and "
                f"{len(already_failed_books)} failed books")
    for book in unbound_books:
        logger.info(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book)
        except Exception as e:
            logger.error(f"Error binding book {book.bid}: {e}")
            logger.error(traceback.format_exc())

    logger.info("Retrying failed books")
    failed_books = imprinter.get_failed_books()
    for book in failed_books:
        if book in already_failed_books:
            logger.info(f"Book {book.bid} has already failed twice, not re-trying")
            continue
        logger.info(f"Binding book {book.bid}")
        try:
            imprinter.bind_book(book)
        except Exception as e:
            logger.error(f"Error binding book {book.bid}: {e}")
            logger.error(traceback.format_exc())
            # it has failed twice, ideally we want people to look at it now
            # do something here


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to imprinter configuration file")
    parser.add_argument('output_root', type=str, help="Root path of the books")
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
