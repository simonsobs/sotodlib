""" Command-line interface for helping to clear out or clean up failed books. 

Make sure you have the DATAPKG_ENV environment variable set to fill in the tags present in the site-pipeline-configs config files

--action=failed: the script will cycle through failed books and
give the option to retry binding (useful if software changes mean we expect the
books to now be successfully bound), to skip binding those files (push
timestreams into the stray books), or do nothing.

--action=timecodes: not implemented yet

"""

import os
import argparse
from typing import Optional

from sotodlib.io.imprinter import Imprinter, Books
import sotodlib.io.imprinter_utils as utils

def main():

    parser = get_parser(parser=None)
    args = parser.parse_args()

    print("You must be running this from an account with write permissions to"
         " the database and the book write directory.")
    
    imprint = Imprinter.for_platform(args.platform)

    if args.action == 'failed':
        check_failed_books(imprint)
    elif args.action == 'timecodes':
        raise NotImplementedError
    elif args.action == 'autofix':
        autofix_failed_books(imprint, args.test_mode)
    else:
        raise ValueError(
            "Chosen action must be 'failed' 'autofix' or 'timecodes'"
        )

def fix_single_book(imprint:Imprinter, book:Books):
    print(
        f"Book ID: {book.bid}\n"
        f"Failed with message:\n"
        f"{book.message}"
    )
    resp = None
    while resp is None:
        resp = input(
            "Possible Actions to Take: "
            "\n\t1. Retry Binding"
            "\n\t2. Retry Binding with lvl2 updates"
            "\n\t3. Rebind with flag(s)"
            "\n\t4. Permanently Skip Binding"
            "\n\t5. Do Nothing"
            "\nInput Response: "
        )
        try: 
            resp = int(resp)
        except:
            print(f"Invalid Response {resp}")
            resp=None
        if resp is None or resp < 1 or resp > 5:
            print(f"Invalid Response {resp}")
            resp=None
    print(f"You selected {resp}")
    if resp == 1 or resp == 2:
        utils.set_book_rebind(imprint, book, update_level2=(resp==2) )
    elif resp == 3:
        utils.set_book_rebind(imprint, book)
        resp = input("Ignore Tags? (y/n)")
        ignore_tags = resp.lower() == 'y'
        resp = input("Drop Ancillary Duplicates? (y/n)")
        ancil_drop_duplicates = resp.lower() == 'y'
        resp = input("Allow Low Precision Timing? (y/n)")
        allow_bad_timing = resp.lower() == 'y'
        imprint.bind_book(
            book, ignore_tags=ignore_tags, ancil_drop_duplicates=ancil_drop_duplicates,
            allow_bad_timing=allow_bad_timing,
        )
    elif resp == 4:
        utils.set_book_wont_bind(imprint, book)
    elif resp == 5:
        pass
    else:
        raise ValueError("how did I get here?")


def check_failed_books(imprint:Imprinter):
    fail_list = imprint.get_failed_books()
    for book in fail_list:
        fix_single_book(imprint, book)

def _last_line(book):
    splits = book.message.split('\n')
    for s in splits[::-1]:
        if len(s) > 0:
            return s

def autofix_failed_books(imprint:Imprinter, test_mode=False):
    fail_list = imprint.get_failed_books()
    for book in fail_list:
        print("-----------------------------------------------------")
        print(f"On book {book.bid}. Has error:\n{_last_line(book)}")
        if 'SECOND-FAIL' in book.message:
            print(f"I already tried to fix {book.bid}")
            continue
        elif 'LEVEL2-FAIL' in book.message:
            print(f"Level 2 failure for {book.bid}")
            continue
        elif (
            "BookDirHasFiles" in book.message or
            "contains files. Delete" in book.message # old valueerror message
        ):
            print(f"Removing {book.bid} files to try again")
            try:
                if not test_mode:
                    utils.set_book_rebind(imprint, book)
                    imprint.bind_book(book)
            except Exception as e :
                print(f"Book {book.bid} failed again!")
        elif "DuplicateAncillaryData" in book.message:
            print(f"Binding {book.bid} while fixing Duplicate Ancil Data")
            try:
                if not test_mode:
                    utils.set_book_rebind(imprint, book)
                    imprint.bind_book(book, ancil_drop_duplicates=True,)
            except Exception as e :
                print(f"Book {book.bid} failed again!")
                book.message = book.message + \
                    ' SECOND-FAIL. Tried with `ancil_drop_duplicates=True`'
                imprint.get_session().commit()
        elif 'TimingSystemOff' in book.message:
            if "Timing counters not incrementing" in book.message:
                print(
                    f"{book.bid} Timing system errors weren't caught by level 2"
                )
                if book.type == 'obs':
                    if not test_mode:
                        book.message = book.message + \
                            " LEVEL2-FAIL! Book re-update needed at level 2" \
                            " and probably have to delete book"
                        imprint.get_session().commit()
                    continue
                elif book.type == 'oper':
                    if not test_mode:
                        utils.set_book_rebind(imprint, book, update_level2=True)
                else:
                    raise ValueError(f"What book got me here? {book.bid}")
            print(f"Binding {book.bid} with low precision timing")
            try:
                if not test_mode:
                    utils.set_book_rebind(imprint, book)
                    imprint.bind_book(book, allow_bad_timing=True,)
            except Exception as e:
                print(f"Book {book.bid} failed again!")
                book.message = book.message + \
                    ' SECOND-FAIL. Tried with `allow_bad_timing=True`'
                imprint.get_session().commit()
        elif 'MissingReadoutIDError' in book.message:
            print(f"Book {book.bid} does not have readout ids, not binding")
            if not test_mode:
                utils.set_book_wont_bind(imprint, book)
        else:
            print(f"I cannot catagorize book {book.bid}")


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Make sure you have the DATAPKG_ENV environment "
            "variable set to fill in the tags present in the "
            "site-pipeline-configs config files. This includes the 'configs' " 
            "tag pointing to your site-pipeline-configs directory"
        )
    parser.add_argument(
        "platform", 
        help="platform (lat, satp1, satp2, satp3)",
        type=str
    )
    parser.add_argument(
        "action", 
        help=" 'failed' or 'timecodes' or 'autofix'",
        type=str
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help="if true, do not try to rebind books",
    )
    return parser