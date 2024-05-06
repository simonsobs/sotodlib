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
    else:
        raise ValueError("Chosen action must be 'failed' or 'timecodes'")

def check_failed_books(imprint:Imprinter):
    fail_list = imprint.get_failed_books()
    for book in fail_list:
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
                "\n\t2. Rebind with flag(s)"
                "\n\t3. Permanently Skip Binding"
                "\n\t4. Do Nothing"
                "\nInput Response: "
            )
            try: 
                resp = int(resp)
            except:
                print(f"Invalid Response {resp}")
                resp=None
            if resp is None or resp < 1 or resp > 4:
                print(f"Invalid Response {resp}")
                resp=None
        print(f"You selected {resp}")
        if resp == 1:
            utils.set_book_rebind(imprint, book)
        elif resp == 2:
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
        elif resp == 3:
            utils.set_book_wont_bind(imprint, book)
        elif resp == 4:
            pass
        else:
            raise ValueError("how did I get here?")

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
        help=" 'failed' or 'timecodes' ",
        type=str
    )
    return parser