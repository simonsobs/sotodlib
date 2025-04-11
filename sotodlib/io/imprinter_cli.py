""" Command-line interface for helping to clear out or clean up failed books. 

Make sure you have the DATAPKG_ENV environment variable set to fill in the tags present in the site-pipeline-configs config files

--action=failed: the script will cycle through failed books and
give the option to retry binding (useful if software changes mean we expect the
books to now be successfully bound), to skip binding those files (push
timestreams into the stray books), or do nothing.

--action=autofix: run through all the book errors and fix ones that have well
understood issues

"""

import os
import argparse
import datetime as dt
from typing import Optional

from sotodlib.io.imprinter import Imprinter, Books, FAILED
import sotodlib.io.imprinter_utils as utils

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
            "\n\t5. Permanently Skip Binding. Delete lvl2."
            "\n\t6. Do Nothing"
            "\nInput Response: "
        )
        try: 
            resp = int(resp)
        except:
            print(f"Invalid Response {resp}")
            resp=None
        if resp is None or resp < 1 or resp > 6:
            print(f"Invalid Response {resp}")
            resp=None
    print(f"You selected {resp}")
    if resp == 1 or resp == 2:
        utils.set_book_rebind(imprint, book, update_level2=(resp==2) )
    elif resp == 3:
        utils.set_book_rebind(imprint, book)
        print("Set each flag [] represents defaults")
        ignore_tags = set_tag_and_validate( "Ignore Tags? (y/[n])" )
        ancil_drop_duplicates = set_tag_and_validate(
            "Drop Ancillary Duplicates? (y/[n])"
        )
        allow_bad_timing = set_tag_and_validate(
            "Allow Bad Timing? (Low Precision or Dropped Samples)? (y/[n])"
        )
        require_acu = set_tag_and_validate("Require ACU data? ([y]/n)")
        require_hwp = set_tag_and_validate("Require HWP data? ([y]/n)")
        require_monotonic_times = set_tag_and_validate(
            "Require Monotonic Housekeeping times? ([y]/n)"
        )
        imprint.bind_book(
            book, ignore_tags=ignore_tags, ancil_drop_duplicates=ancil_drop_duplicates,
            allow_bad_timing=allow_bad_timing,
            require_acu=require_acu,
            require_hwp=require_hwp,
            require_monotonic_times=require_monotonic_times,
        )
    elif resp == 4:
        utils.set_book_wont_bind(imprint, book)
    elif resp == 5:
        sure = set_tag_and_validate(
            "Are you sure you want to delete level 2?"
        )
        if sure:
            utils.delete_level2_obs_and_book(imprint, book)
        else:
            print("Skipping")
    elif resp == 6:
        pass
    else:
        raise ValueError("how did I get here?")

def set_tag_and_validate(msg):
    resp = input(msg)
    while (resp.lower() != 'n' and resp.lower() != 'y'):
        print(f"Response {resp} is invalid")
        resp = input(msg)
    return resp.lower() == 'y'

def check_failed_books(imprint:Imprinter):
    fail_list = imprint.get_failed_books()
    for book in fail_list:
        fix_single_book(imprint, book)

def _last_line(book):
    splits = book.message.split('\n')
    for s in splits[::-1]:
        if len(s) > 0:
            return s

class BookError:
    def __init__(self, imprint, book):
        self.imprint = imprint
        self.book = book
    
    @staticmethod
    def has_error(book):
        raise NotImplementedError
    def fix_book(self):
        raise NotImplementedError
    def report_error(self):
        raise NotImplementedError

class BookDirHasFiles(BookError):
    @staticmethod
    def has_error(book):
        return "BookDirHasFiles" in book.message
    def fix_book(self):
        utils.set_book_rebind(self.imprint, self.book)
        self.imprint.bind_book(self.book)
    def report_error(self):
        return f"{self.book.bid} has files already in staged" 

class SecondFail(BookError):
    @staticmethod
    def has_error(book):
        return "SECOND-FAIL" in book.message
    def fix_book(self):
        return 
    def report_error(self):
        msg = f"{self.book.bid} has already failed once during autofix\n\t" 
        msg += _last_line(self.book)
        return msg

class MissingReadoutIDs(BookError):
    @staticmethod
    def has_error(book):
        return 'MissingReadoutIDError' in book.message
    def fix_book(self):
        utils.set_book_wont_bind(self.imprint, self.book)
    def report_error(self):
        return f"{self.book.bid} does not have readout ids"

class NoScanFrames(BookError):
    @staticmethod
    def has_error(book):
        return 'NoScanFrames' in book.message
    def fix_book(self):
        utils.set_book_wont_bind(self.imprint, self.book)
    def report_error(self):
        return f"{self.book.bid} does not have detector data"

class NoHWPData(BookError):
    @staticmethod
    def has_error(book):
        return 'NoHWPData' in book.message
    def fix_book(self):
        utils.set_book_rebind(self.imprint, self.book)
        self.imprint.bind_book(self.book, require_hwp=False,)
    def report_error(self):
        return f"{self.book.bid} does not HWP data reading out"

class DuplicateAncillaryData(BookError):
    @staticmethod
    def has_error(book):
        return "DuplicateAncillaryData" in book.message
    def fix_book(self):
        utils.set_book_rebind(self.imprint, self.book)
        self.imprint.bind_book(self.book, ancil_drop_duplicates=True,)
    def report_error(self):
        return f"{self.book.bid} has duplicate ancillary data"

class NoMountData(BookError):
    @staticmethod
    def has_error(book):
        return "NoMountData" in book.message
    def fix_book(self):
        if book.type == 'obs':
            print("Cannot autofix obs books where the ACU was not reading out")
            return
        elif book.type == 'oper':
            utils.set_book_rebind(self.imprint, self.book)
            self.imprint.bind_book(self.book, require_acu=False,)
        else: 
            raise ValueError(f"What book got me here? {book.bid}")
    def report_error(self):
        return f"{self.book.bid} does not ACU data reading out"

class TimingSystemOff(BookError):
    """Two places this error is thrown. If we get to the timing counter
    incrementing error then we know the level 2 metadata isn't set correctly.
    This can mess up observation books because low precision timing means the
    different slots cannot be bound together.
    """
    @staticmethod
    def has_error(book):
        return 'TimingSystemOff' in book.message

    def fix_book(self):
        if "Timing counters not incrementing" in book.message:
            if book.type == 'obs':
                print(
                    "Cannot autofix obs books where timing counters aren't" 
                    " incrementing"
                )
            elif book.type == 'oper':
                utils.set_book_rebind(imprint, book, update_level2=True)
                imprint.bind_book(book)
            else:
                raise ValueError(f"What book got me here? {book.bid}")
        else:
            utils.set_book_rebind(self.imprint, self.book)
            self.imprint.bind_book(self.book, allow_bad_timing=True,)
    
    def report_error(self):
        if "Timing counters not incrementing" in book.message:
            msg = f"{book.bid} has timing system errors not caught at level 2"
            if book.type == 'obs':
                msg += "\n\t LEVEL2-FAIL: probably have to delete book, update"
                msg += "update level 2, and replan books"
            return msg
        else:
            return f"{book.bid} has low precision timing"

class FileTooLargeError(BookError):
    @staticmethod
    def has_error(book):
        return "FileTooLargeError" in book.message
    def fix_book(self):
        utils.set_book_rebind(self.imprint, self.book)       
        utils.delete_level2_obs_and_book(self.imprint, self.book)
    def report_error(self):
        msg = f"{self.book.bid} has too large level 2 files\n"
        l = _last_line(self.book)
        msg += f"\t {l.split('/')[-1]}"
        return msg

class BadTimeSamples(BookError):
    @staticmethod
    def has_error(book):
        return "BadTimeSamples" in book.message
    def fix_book(self):
        utils.set_book_rebind(self.imprint, self.book)        
        self.imprint.bind_book(self.book, allow_bad_timing=True,)
    def report_error(self):
        msg = f"{self.book.bid} has dropped time samples\n"
        for l in self.book.message.split('\n'):
            if len(l)>0 and l[0] == '\t':
                msg += l
        return msg

AUTOFIX_ERRORS = [
    SecondFail,
    BookDirHasFiles,
    MissingReadoutIDs,
    NoScanFrames,
    NoHWPData,
    DuplicateAncillaryData,
    NoMountData,
    TimingSystemOff,
    FileTooLargeError,
    BadTimeSamples,
]

def process_book_failure(
    imprint:Imprinter, 
    book:Books, 
    report=True, 
    fix=False,
    error_list = None,
):
    if error_list is None:
        error_list = AUTOFIX_ERRORS

    found = False
    for error in error_list:
        if error.has_error(book):
            found = True
            err = error(imprint, book)
            if report:
                print( err.report_error() )
            if fix:
                try:
                    err.fix_book()  
                except:
                    print(f"Failed on {book.bid}. Recording Failure")
                    book.status = FAILED #just in case we cleared the error
                    book.message = book.message + \
                    f'\nSECOND-FAIL. Tried with to fix with {error.__name__}'
                    imprint.get_session().commit()
        if found:
            break ## stop on first found error
    return found
            

def autofix_failed_books(
    imprint:Imprinter, min_ctime=None, max_ctime=None,
    report=True, fix=True, error_list=None,
):
    session = imprint.get_session()
    failed = session.query(Books).filter(Books.status == FAILED)
    if min_ctime is not None:
        failed = failed.filter(
            Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
        )
    if max_ctime is not None:
        failed = failed.filter(
            Books.start <= dt.datetime.utcfromtimestamp(max_ctime),
        )
    failed = failed.all()
    for book in failed:
        success = process_book_failure(
            imprint, book, 
            report=report, 
            fix=fix, 
            error_list=error_list
        )
        if not success:
            print(f"Cannot categorize {book.bid}")

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
        help=" 'failed' or 'report' or 'autofix'",
        type=str
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help="if true, do not try to rebind books",
    )
    return parser

def main():

    parser = get_parser(parser=None)
    args = parser.parse_args()

    print("You must be running this from an account with write permissions to"
         " the database and the book write directory.")
    
    imprint = Imprinter.for_platform(args.platform)

    if args.action == 'failed':
        check_failed_books(imprint)
    elif args.action == 'report':
        autofix_failed_books(imprint, report=True, fix=False)
    elif args.action == 'autofix':
        autofix_failed_books(imprint, report=True, fix=True)
    else:
        raise ValueError(
            "Chosen action must be 'failed' 'autofix' or 'report'"
        )