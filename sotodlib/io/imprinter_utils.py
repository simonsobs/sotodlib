"""Functions for working with an imprinter instance. These are generally
functions expected to be needed for humans making one-off changes to the
imprinter setup. Functions run as part of the automated pipeline are defined in imprinter.py
"""
from sqlalchemy import or_
import datetime as dt
import os
import os.path as op
import shutil
import time

from sotodlib.io.imprinter import ( 
    Books,
    WONT_BIND,
    FAILED,
    UNBOUND,
    BOUND,
    UPLOADED,
    DONE,    
)

from .load_smurf import (
    TimeCodes,
    SupRsyncType,
)

def set_book_wont_bind(imprint, book, message=None, session=None):
    """Change book status to WONT_BIND, meaning the files indicated into 
    this book will not be bound. .g3 files here will go into stray books.

    This function should not be integrated into automated data packaging.

    This book status is necessary because without it the automated data 
    packaging could continue to try and register and fail on the same book. 

    Parameters
    -----------
    imprint: Imprinter instance
    book: str or Book 
    message: str or None
        if not none, update book message to explain why the setting is changing
    session: BookDB session.
    
    """
    if session is None:
        session = imprint.get_session()

    if isinstance(book, str):
        book = imprint.get_book(book)

    if book.status != FAILED:
        imprint.logger.warning(
            f"Book {book} has not failed before being set not to WONT_BIND"
        )
    book.status = WONT_BIND 
    if message is not None:
        book.message = message
    session.commit()

def set_book_rebind(imprint, book, update_level2=False):
    """ Delete any existing staged files for a book as 
    set it's status to UNBOUND

    Parameters
    -----------
    imprint: Imprinter instance
    book: str or Book 
    update_level2: bool
        if true, update all the level 2 observation entries for the book. This is sometimes the reason books fail to bind
    """
    book_dir = imprint.get_book_abs_path(book)

    if op.exists(book_dir):
        print(f"Removing all files from {book_dir}")
        shutil.rmtree(book_dir)
    else: 
        print(f"Found no files in {book_dir} to remove")

    book.status = UNBOUND
    imprint.get_session().commit()

    if update_level2:
        g3session, SMURF = imprint.get_g3tsmurf_session(return_archive=True)
        obs_dict = imprint.get_g3tsmurf_obs_for_book(book)
        print(f"Updating level 2 observations")
        for _, obs in obs_dict.items():
            SMURF.update_observation_files(obs, g3session, force=True)

def find_overlaps(imprint, obs_id, min_ctime, max_ctime):
    """ helper function for when a level 2 observation could span multiple
    books. Creates a list of ObsSets with that obs_id, prints info to screen and
    returns the list. imprinter then has a function
    imprinter.register_book(obsset, commit=True) that can be used to register
    the desired observation. Example usage::

        rsets = utils.find_overlaps(
            imprint, 'obs_ufm_mv9_1714406208', <- obs id from error message
            min_ctime, max_ctime
        )
        imprint.register_book( rsets[0], commit=True)

    obs_id: level 2 obs_id that overlaps multiple observations
    """
    obsset = imprint.update_bookdb_from_g3tsmurf(
        min_ctime=min_ctime, max_ctime=max_ctime,
        return_obsset=True,
    )
    rsets = []
    for o in obsset:
        if obs_id in o.obs_ids:
            rsets.append(o)
    for i,r in enumerate(rsets):
        print(f"-----ObsSet {i}----------")
        for o in r:
            print("\t", o)

    return rsets

def block_fix_duplicate_timestamps(imprint):
    """Run through and fix all the books with duplicated ancillary timestamps"""

    failed_books = imprint.get_failed_books()
    fix_list = []
    for book in failed_books:
        if "duplicate timestamps" in book.message:
            fix_list.append(book)
    imprint.logger.info(
        f"Found {len(fix_list)} books with duplicate HK data to fix"
    )

    for book in fix_list:
        imprint.logger.info(f"Setting book {book.bid} for rebinding")
        set_book_rebind(imprint, book)
        imprint.logger.info(
            f"Binding book {book.bid} dropping duplicate HK data"
        )
        imprint.bind_book(book, ancil_drop_duplicates=True)

def block_set_rebind(imprint, update_level2=False):
    """Run through and set all books with files errors to be rebound"""

    failed_books = imprint.get_failed_books()
    
    fix_list = []
    for book in failed_books:
        if "Delete to retry bookbinding" in book.message:
            fix_list.append(book)
    imprint.logger.info(
        f"Found {len(fix_list)} books with files to be removed"
    )
    for book in fix_list:
        imprint.logger.info(f"Setting book {book.bid} for rebinding")
        set_book_rebind(imprint, book, update_level2=update_level2)    

def get_timecode_final(imprint, time_code, type='all'):
    """Check if all required entries in the g3tsmurf database are present for
    smurf or stray book regisitration.
    
    Parameters
    -----------
    imprint: Imprinter instance
    time_code: int
        5-digit ctime code to check for finalization
    type: str
        book type to check for
    
    Returns
    --------
    is_final, bool
    reason, int
        0 if the books are ready to be registered
        1 if we are missing metadata entries
        2 if we are missing file entries
        3 if unbound or failed books are preventing registration
    """
    assert type in ['all','stray','smurf']
    
    g3session, SMURF = imprint.get_g3tsmurf_session(return_archive=True)
    session = imprint.get_session()

    servers = SMURF.finalize["servers"]
    meta_agents = [s["smurf-suprsync"] for s in servers]
    files_agents = [s["timestream-suprsync"] for s in servers]

    meta_query = or_(*[TimeCodes.agent == a for a in meta_agents])
    files_query = or_(*[TimeCodes.agent == a for a in files_agents])

    tcm = g3session.query(TimeCodes.agent).filter(
        TimeCodes.timecode==time_code,
        meta_query,
        TimeCodes.suprsync_type == SupRsyncType.META.value,
    ).distinct().all()

    if type == 'smurf':
        if len(tcm) == len(meta_agents):
            return True, 0
        else:
            return False, 1
    
    if len(tcm) != len(meta_agents):
        return False, 1

    tcf = g3session.query(TimeCodes.agent).filter(
        TimeCodes.timecode==time_code,
        files_query,
        TimeCodes.suprsync_type == SupRsyncType.FILES.value,
    ).distinct().all()
    
    if len(tcf) != len(files_agents):
        return False, 2
    
    book_start = dt.datetime.utcfromtimestamp(time_code * 1e5)
    book_stop = dt.datetime.utcfromtimestamp((time_code + 1) * 1e5)

    q = session.query(Books).filter(
        Books.start >= book_start,
        Books.start < book_stop,
        or_(Books.type == 'obs', Books.type == 'oper'),
        or_(Books.status == UNBOUND, Books.status == FAILED), 
    )
    if q.count() > 0:
        return False, 3

    return True, 0

    
def set_timecode_final(imprint, time_code):
    """Add required entires to the g3tsmurf database in order to force the smurf
    and/or stray books to be created. Will be used if there are errors in
    suprsync data transfer.
    
    Parameters
    -----------
    imprint: Imprinter instance
    time_code: 5-digit ctime code to finalize
    """

    g3session, SMURF = imprint.get_g3tsmurf_session(return_archive=True)

    servers = SMURF.finalize["servers"]
    
    for server in servers:
        tcf = g3session.query(TimeCodes).filter(
            TimeCodes.timecode == time_code,
            TimeCodes.agent == server["timestream-suprsync"],
            TimeCodes.suprsync_type == SupRsyncType.FILES.value,
        ).first()
        if tcf is None:
            tcf = TimeCodes(
                stream_id="fake",
                suprsync_type=SupRsyncType.FILES.value,
                timecode=time_code,
                agent=server["timestream-suprsync"],
            )
        g3session.add(tcf)

        tcm = g3session.query(TimeCodes).filter(
            TimeCodes.timecode==time_code,
            TimeCodes.agent == server["smurf-suprsync"],
            TimeCodes.suprsync_type == SupRsyncType.META.value,
        ).first()
        if tcm is None:
            tcm = TimeCodes(
                stream_id="fake",
                suprsync_type=SupRsyncType.META.value,
                timecode=time_code,
                agent=server["smurf-suprsync"],
            )
        g3session.add(tcm)    
    g3session.commit()