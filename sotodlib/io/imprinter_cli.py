""" Command-line interface for helping to clear out or clean up failed books. 

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

def main(
        imprint_config:str, 
        action:str, 
        g3_config:Optional[str]=None,
        output_root:Optional[str]=None
    ):
    print("You must be running this from an account with write permissions to"
         " the database and the book write directory.")
    
    imprint = Imprinter(imprint_config)
    ## overrides that often happen because running inside and outside dockers
    if output_root is not None:
        if not os.path.exists(output_root):
            raise ValueError(f"Output root {output_root} does not exist")
        imprint.output_root = output_root
    if g3_config is not None:
        if not os.path.exists(g3_config):
            raise ValueError(f"G3tSmurf config file {g3_config} does not exist")
        imprint.g3tsmurf_config = g3_config

    if action == 'failed':
        check_failed_books(imprint)
    elif action == 'timecodes':
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
            imprint.bind_book(
                book, ignore_tags=ignore_tags, ancil_drop_duplicates=ancil_drop_duplicates
            )
        elif resp == 3:
            utils.set_book_wont_bind(imprint, book)
        elif resp == 4:
            pass
        else:
            raise ValueError("how did I get here?")

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--imprint-config", help="Imprinter config file",
        type=str, required=True)
    parser.add_argument("--action", help=" 'failed' or 'timecodes' ",
        type=str, required=True)
    parser.add_argument("--g3-config", help="G3tSmurf config file",
        type=str)
    parser.add_argument("--output-root", help="Overide imprinter output root?",
        type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))