import os
import traceback
import argparse

import datetime as dt
from typing import Optional
from sotodlib.io.imprinter import Imprinter, Books, FAILED
import sotodlib.io.imprinter_utils as utils

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def bind_books_parallel(platform, book_list, n_proc ):
    imprint = Imprinter.for_platform(platform)
    session = imprint.get_session()
    bid_list = [book.bid for book in book_list]
    failed_list = []
    with ProcessPoolExecutor(n_proc) as exe:
        futures = [
            exe.submit(_bookbinding_helper, platform, bid) for bid in bid_list
        ]
        for future in as_completed(futures):
            bid, status, message, _  = future.result()
            imprint.logger.info(f"Just finished book {bid}")
            book = session.query(Books).filter(Books.bid == bid).one()
            book.status = status
            book.message = message
            session.commit()
            if status == FAILED:
                failed_list.append( book.bid)
    return failed_list

def _bookbinding_helper(platform, bid ):
    imprint = Imprinter.for_platform(platform)
    return imprint._run_book_binding(bid)


def main(config: str, parallel_oper:bool=False):
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
    
    if parallel_oper:
        parallel_list = [
            book for book in unbound_books if book.type == 'oper'
        ]
        bind_books_parallel(imprinter.daq_node, parallel_list, n_proc=4)

    unbound_books = imprinter.get_unbound_books()
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
        if 'NoHWPData' in book.message:
            print(
                f"Book {book.bid} does not HWP data reading out, binding "
                    "anyway"
            )
            require_hwp = False
        else:
            require_hwp = True    
        try:         
            utils.set_book_rebind(imprinter, book)
            imprinter.bind_book(book, require_hwp=require_hwp)
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
    #parser.add_argument('output_root', type=str, help="Root path of the books")
    parser.add_argument("--parallel-oper", action="store_true")
    return parser


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
