"""Functions for working with an imprinter instance. These are generally
functions expected to be needed for humans making one-off changes to the
imprinter setup. Functions run as part of the automated pipeline are defined in imprinter.py
"""

from sotodlib.io.imprinter import ( 
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

def set_timecode_final(imprint, time_code):
    """Add required entires to the g3tsmurf database in order to force the smurf
    and/or stray books to be created. Will be used if there are errors in
    suprsync data transfer.
    
    Parameters
    -----------
    imprint: Imprinter instance
    time_code: 5-digit ctime code to finalize
    """

    g3session = imprint.get_g3tsmurf_session()

    streams = []
    for tube in imprint.tubes:
        streams.extend( imprint.tubes[tube].get("slots") )
    
    for stream in streams:
        tcf = g3session.query(TimeCodes).filter(
            TimeCodes.timecode==time_code,
            TimeCodes.stream_id == stream,
            TimeCodes.suprsync_type == SupRsyncType.FILES.value,
        ).one_or_none()
        if tcf is None:
            tcf = TimeCodes(
                stream_id=stream,
                suprsync_type=SupRsyncType.FILES.value,
                timecode=time_code,
                agent="fake",
            )
        g3session.add(tcf)

        tcm = g3session.query(TimeCodes).filter(
            TimeCodes.timecode==time_code,
            TimeCodes.stream_id == stream,
            TimeCodes.suprsync_type == SupRsyncType.META.value,
        ).one_or_none()
        if tcm is None:
            tcm = TimeCodes(
                stream_id=stream,
                suprsync_type=SupRsyncType.META.value,
                timecode=time_code,
                agent="fake",
            )
        g3session.add(tcm)    
    g3session.commit()