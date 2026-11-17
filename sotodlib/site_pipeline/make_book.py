"""
Bind Books: the only data packaging element that writes level 3 data. It takes
every Book registered as UNBOUND by ``update_book_plan`` and writes it out to
the imprinter's ``output_root`` staging area, then makes one retry pass over
FAILED Books.

Binding is delegated to ``sotodlib.io.bookbinder``:

* ``obs``/``oper`` Books use ``BookBinder``, which reads the level 2 frames,
  puts every wafer on a common sample grid, fills small gaps, co-samples the
  housekeeping fields named in the imprinter config's ``hk_fields``, and
  writes ``D_<stream_id>_*.g3`` plus ``A_ancil_*.g3``. ``oper`` Books also get
  the raw sodetlib outputs copied into ``Z_smurf/``.
* ``hk``, ``smurf`` and ``stray`` Books use ``TimeCodeBinder``, which copies
  level 2 files verbatim (into a zip for ``smurf`` Books at schema >= 1).

``M_book.yaml`` and ``M_index.yaml`` are written last, and obs/oper Books are
re-read by ``check_book.BookScanner`` to confirm internal consistency before
the status is set to BOUND. Anything that raises along the way leaves the Book
FAILED with the traceback stored in its ``message`` column.

The retry pass is deliberately shallow: Books that were already FAILED when
the script started are skipped, so nothing is retried twice in one run. Books
that fail their retry send an alert to ``--alert-webhook``. To triage the
remaining failures use ``python -m sotodlib.io.imprinter_cli <platform>
report|autofix|failed``.

With ``--n-proc`` > 1 the Books are bound in a process pool, and each worker
rebuilds its Imprinter with ``Imprinter.for_platform``, so ``DATAPKG_ENV``
must be set.

See the Data Packaging page of the sotodlib documentation for the Book model,
Book statuses, and the catalogue of known binding failures.
"""

import os
import traceback
import argparse

import datetime as dt
from typing import Optional
from sotodlib.io.imprinter import Imprinter, Books, FAILED
import sotodlib.io.imprinter_utils as utils
from sotodlib.site_pipeline.utils.alerts import send_alert

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
            bid, status, message, err  = future.result()
            imprint.logger.info(f"Just finished book {bid}")
            book = session.query(Books).filter(Books.bid == bid).one()
            book.status = status
            book.message = message
            session.commit()
            if status == FAILED:
                failed_list.append( book.bid)
                print(f"Error binding book {book.bid}: {err}")
            else:
                print(f"Finished binding {book.bid}")
    return failed_list

def _bookbinding_helper(platform, bid ):
    imprint = Imprinter.for_platform(platform)
    return imprint._run_book_binding(bid)


def main(config: str, n_proc:int=1, alert_webhook: list[str]=None):
    """Make books based on imprinter db
    
    Parameters
    ----------
    config : str
        path to imprinter configuration file
    n_proc : int
        Number of processes
    alert_webhook : list[str]
        Webhook URLs to send alerts
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

    if n_proc>1:
        multiprocessing.set_start_method('spawn')
        bind_books_parallel(imprinter.daq_node, unbound_books, n_proc=n_proc)
    else:
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
            alert = send_alert(
                alert_webhook,
                alertname=book.bid,
                tag='bookbinder',
                error=str(e),
                timestamp=book.start
            )
            print(alert)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Bind every UNBOUND Book in the imprinter database, "
            "writing level 3 data to the imprinter's output_root, then retry "
            "Books that failed on a previous run."
        )
    parser.add_argument(
        'config',
        type=str,
        help="Path to imprinter configuration file"
    )
    parser.add_argument(
        "--n-proc", type=int, default=1,
        help="Number of processes to bind Books with. Values > 1 require "
        "DATAPKG_ENV to be set, since workers rebuild the Imprinter from the "
        "platform name."
    )
    parser.add_argument(
        "--alert-webhook", type=str, default='', nargs="+",
        help="Webhook address(es) to send alerts to when a Book fails twice"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
