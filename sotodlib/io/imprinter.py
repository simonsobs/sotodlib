import datetime as dt, os.path as op, os
import numpy as np
from collections import OrderedDict
from typing import List
import yaml, traceback
import shutil
import logging
from glob import glob

import sqlalchemy as db
from sqlalchemy import or_, and_, not_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import so3g
from spt3g import core

import sotodlib
from .bookbinder import BookBinder
from .load_smurf import (
    G3tSmurf,
    Observations as G3tObservations,
    SmurfStatus,
    get_channel_info,
    TimeCodes,
    SupRsyncType,
    Files,
)
from .g3thk_db import G3tHk, HKFiles
from ..site_pipeline.util import init_logger


####################
# useful constants #
####################

# book status
WONT_BIND = -2 # books we have tagged by hand that we will not bind
FAILED = -1 # books where automated binding failed
UNBOUND = 0
REBIND = 1
BOUND = 2
UPLOADED = 3
DONE = 4

# tel tube, stream_id, slot mapping
VALID_OBSTYPES = ["obs", "oper", "smurf", "hk", "stray", "misc"]


class BookExistsError(Exception):
    """Exception raised when a book already exists in the database"""

    pass


class BookBoundError(Exception):
    """Exception raised when a book is already bound"""

    pass


class NoFilesError(Exception):
    """Exception raised when no files are found in the book"""

    pass


###################
# database schema #
###################

Base = declarative_base()


class Observations(Base):
    """
    Attributes
    ----------
    obs_id: observation id
    bid: book id

    """

    __tablename__ = "observations"
    obs_id = db.Column(db.String, primary_key=True)
    bid = db.Column(db.String, db.ForeignKey("books.bid"))
    book = relationship("Books", back_populates="obs")


class Books(Base):
    """
    Attributes
    ----------
    bid: book id
    start: start time of book
    stop: stop time of book
    max_channels: maximum number of channels in book
    obs: list of observations within the book
    type: type of book, e.g., "oper", "hk", "obs"
    status: integer, stage of processing, 0 is unbound, 1 is bound,
    message: error message if book failed
    tel_tube: telescope tube
    slots: slots in comma separated string
    created_at: time when book was created
    updated_at: time when book was updated
    timing: bool, whether timing system is on
    path: str, location of book directory
    lvl2_deleted: bool, if level 2 data has been purged yet
    """

    __tablename__ = "books"
    bid = db.Column(db.String, primary_key=True)
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)
    max_channels = db.Column(db.Integer)
    obs = relationship("Observations", back_populates="book")  # one to many
    type = db.Column(db.String)
    status = db.Column(db.Integer, default=UNBOUND)
    message = db.Column(db.String, default="")
    tel_tube = db.Column(db.String)
    slots = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )
    timing = db.Column(db.Boolean)
    path = db.Column(db.String)
    lvl2_deleted = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<Book: {self.bid}>"


##############
# main logic #
##############


# convenient decorator to repeat a method over all data sources
def loop_over_sources(method):
    def wrapper(self, *args, **kwargs):
        for source in self.sources:
            method(self, source, *args, **kwargs)

    return wrapper


class Imprinter:
    def __init__(self, im_config=None, db_args={}, logger=None):
        """Imprinter manages the book database.
        
        Imprinter at the site is set up to be one per level 2 daq node. Some pieces
        of this code have worked with one imprinter for multiple daq nodes but this 
        has not be end-to-end tested.

        Example configuration file::
        
          db_path: imprinter.db
          
          sources:
            tel-tube1:
              g3tsmurf: g3tsmurf config file
              slots:
                - stream_id1
                - stream_id2
                - stream_id3
            tel-tube2:
              g3tsmurf: g3tsmurf config file
              slots:
                - stream_id4
                - stream_id5
          
          hk_sources:
            daq-node:
              g3tsmurf: g3tsmurf config file

        The sources entry lists the different tel-tubes that will be bound into 
        separate books containing only the stream_ids listed under slots. TODO: 
        There is currently no option to have an empty slot. The tel-tube entries 
        define the folder books are written to ex: output_root/tel-tube/obs/

        The hk_sources entry determines if housekeeping books should be made, and 
        where. While it take a g3tsmurf config key, that is just for symmetry with
        sources. Those config files only need to include a data_prefix and g3thk_db 
        entry if only hk books are being created from that configuration. The 
        daq-node entry defines where the books are written to ex: output_root/daq-node/hk

        The g3tsmurf configurations are assumed to be one per daq node, meaning 
        this example could have the same config file repeated several times. The 
        repetition is based on the idea that one imprinter could manage multiple 
        daq-nodes, although that is not the current setup.

        Parameters
        ----------
        im_config: str
            path to imprinter configuration file
        db_args: dict
            arguments to pass to sqlalchemy.create_engine
        logger: logger
            logger object
        """
        # load config file and parameters
        with open(im_config, "r") as f:
            self.config = yaml.safe_load(f)

        self.db_path = self.config["db_path"]
        self.sources = self.config.get("sources", {})
        self.hk_sources = self.config.get("hk_sources", {})

        # check whether db_path directory exists
        if not op.exists(op.dirname(self.db_path)):
            # to create the database, we first make sure that the folder exists
            os.makedirs(op.abspath(op.dirname(self.db_path)), exist_ok=True)
        self.engine = db.create_engine(f"sqlite:///{self.db_path}", **db_args)

        # create all tables or (do nothing if tables exist)
        Base.metadata.create_all(self.engine)

        self.session = None
        self.g3tsmurf_sessions = {}
        self.archives = {}
        self.hk_archives = {}

        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger("imprinter")
            if not self.logger.hasHandlers():
                self.logger = init_logger("imprinter")

    def get_session(self):
        """Get a new session or return the existing one

        Returns
        -------
        session: BookDB session

        """
        if self.session is None:
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        return self.session

    def get_g3tsmurf_session(self, source, return_archive=False):
        """Get a new g3tsmurf session or return an existing one.

        Parameter
        ---------
        source: str
            data source, e.g., "sat1", "tsat", "latrt"
        return_archive: bool
            whether to return SMURF archive object

        Returns
        -------
        session: g3tsmurf session
        archive: if return_archive is true

        """
        if source not in self.g3tsmurf_sessions:
            (
                self.g3tsmurf_sessions[source],
                self.archives[source],
            ) = create_g3tsmurf_session(self.sources[source]["g3tsmurf"])
        if not return_archive:
            return self.g3tsmurf_sessions[source]
        return self.g3tsmurf_sessions[source], self.archives[source]

    def get_g3thk(self, source):
        """Get a G3tHk database using the g3tsmurf config file."""
        if source not in self.hk_archives:
            if source in self.config["hk_sources"]:
                s_type = "hk_sources"
            elif source in self.config["sources"]:
                s_type = "sources"
            else:
                raise ValueError(f"Cannot find {source} in configs")
            self.hk_archives[source] = G3tHk.from_configs(
                self.config[s_type][source]["g3tsmurf"]
            )
        return self.hk_archives[source]

    def register_book(self, obsset, bid=None, commit=True, session=None):
        """Register book to database

        Parameters
        ----------
        obsset: ObsSet object
            thin wrapper of a list of observations
        bid: str
            book id
        commit: boolean
            if True, commit the session
        session: SQLAlchemy session
            BookDB session

        Returns
        -------
        book: book object

        """
        if session is None:
            session = self.get_session()
        # create book id if not provided
        if bid is None:
            bid = obsset.get_id()
        assert obsset.mode is not None
        # check whether book exists in the database
        if self.book_exists(bid, session=session):
            raise BookExistsError(f"Book {bid} already exists in the database")
        self.logger.info(f"Registering book {obsset} (mode: {obsset.mode})")

        # fill in book attributes: start, end, max_channels
        # start and end are the earliest and latest observation time
        # max_channels is the maximum number of channels in the book
        if any([len(o.files) == 0 for o in obsset]):
            raise NoFilesError(f"No files found for observations in {bid}")
        # get start and end times (initial guess)
        start_t = np.max([np.min([f.start for f in o.files]) for o in obsset])
        stop_t = np.min([np.max([f.stop for f in o.files]) for o in obsset])
        max_channels = int(
            np.max([np.max([f.n_channels for f in o.files]) for o in obsset])
        )
        timing_on = np.all([o.timing for o in obsset])

        # create observation and book objects
        observations = [Observations(obs_id=obs_id) for obs_id in obsset.obs_ids]
        book = Books(
            bid=bid,
            obs=observations,
            type=obsset.mode,
            status=UNBOUND,
            start=start_t,
            stop=stop_t,
            max_channels=max_channels,
            tel_tube=obsset.tel_tube,
            slots=",".join(
                [s for s in obsset.slots if obsset.contains_stream(s)]
            ),  # not worth having a extra table
            timing=timing_on,
        )

        # add book to database
        session.add(book)
        if commit:
            session.commit()
        return book

    def register_books(self, bids, obs_lists, commit=True, session=None):
        """Register multiple books to database.

        Parameters
        ----------
        bids: list
            a list of book ids (str)
        obs_lists: list
            a list of list where the smaller list consists of
            different observations and the outer loop goes through
            different books
        commit: boolean
            if True, commit the session
        session: BookDB session

        """
        if session is None:
            session = self.get_session()
        for bid, obs_list in zip(bids, obs_lists):
            self.register_book(session, bid, obs_list, commit=False)
        if commit:
            session.commit()

    def register_hk_books(self, commit=True, session=None):
        """Register housekeeping books to database"""
        session = session or self.get_session()

        # fixme: tel_tube and daq_node may not be the same
        for daq_node in self.hk_sources:
            with open(self.hk_sources[daq_node]["g3tsmurf"], "r") as f:
                g3tsmurf_cfg = yaml.safe_load(f)
            data_root = g3tsmurf_cfg["data_prefix"]

            # all ctime dir except the last ctime dir will be considered complete
            ctime_dirs = sorted(glob(op.join(data_root, "hk", "*")))
            for ctime_dir in ctime_dirs[:-1]:
                ctime = op.basename(ctime_dir)
                book_id = f"hk_{ctime}_{daq_node}"
                # check whether the book exists
                if self.get_book(book_id) is not None:
                    continue
                book = Books(
                    bid=book_id,
                    type="hk",
                    status=UNBOUND,
                    start=dt.datetime.utcfromtimestamp(int(ctime) * 1e5),
                    stop=dt.datetime.utcfromtimestamp((int(ctime) + 1) * 1e5),
                    tel_tube=daq_node,  # fixme: tel_tube and daq_node may not be the same
                )
                session.add(book)
            if commit:
                session.commit()

    @loop_over_sources
    def register_timecode_books(self, source, commit=True, session=None):
        """Register smurf and stray books with database. We do not expect many
        stray books as nearly all .g3 files will be associated with some
        obs or oper book.
        """
        session = session or self.get_session()

        g3session = self.get_g3tsmurf_session(source)
        streams = self.sources[source].get("slots")

        tcs = g3session.query(TimeCodes.timecode).distinct().all()
        stream_filt = or_(*[TimeCodes.stream_id == s for s in streams])
        stream_filt_files = or_(*[Files.stream_id == s for s in streams])

        for (tc,) in tcs:
            q = g3session.query(TimeCodes).filter(TimeCodes.timecode == tc, stream_filt)
            files = q.filter(TimeCodes.suprsync_type == SupRsyncType.FILES.value)
            meta = q.filter(TimeCodes.suprsync_type == SupRsyncType.META.value)

            if files.count() == len(streams) and meta.count() == len(streams):
                ## register books
                book_start = dt.datetime.utcfromtimestamp(tc * 1e5)
                book_stop = dt.datetime.utcfromtimestamp((tc + 1) * 1e5)

                book_id = f"smurf_{tc}_{source}"
                if self.get_book(book_id) is None:
                    self.logger.debug(f"registering {book_id}")
                    book1 = Books(
                        bid=book_id,
                        type="smurf",
                        status=UNBOUND,
                        tel_tube=source,
                        start=book_start,
                        stop=book_stop,
                    )
                    session.add(book1)
                q = g3session.query(Files).filter(
                    Files.start >= book_start,
                    Files.start < book_stop,
                    stream_filt_files,
                    Files.obs_id == None,
                )
                book_id = f"stray_{tc}_{source}"

                if q.count() > 0 and self.get_book(book_id) is None:
                    book2 = Books(
                        bid=book_id,
                        type="stray",
                        status=UNBOUND,
                        tel_tube=source,
                        start=book_start,
                        stop=book_stop,
                    )
                    session.add(book2)
        if commit:
            session.commit()

    def _get_binder_for_book(self, 
        book, 
        output_root="out", 
        pbar=False, 
        ignore_tags=False
    ):
        """get the appropriate bookbinder for the book based on its type"""
        if book.type == 'hk':
            g3tsmurf_cfg_file = self.hk_sources[book.tel_tube]["g3tsmurf"]
        else:
            g3tsmurf_cfg_file = self.sources[book.tel_tube]["g3tsmurf"]
        with open(g3tsmurf_cfg_file, "r") as f:
            g3tsmurf_cfg = yaml.safe_load(f)
        data_root = g3tsmurf_cfg["data_prefix"]

        if book.type in ["obs", "oper"]:
            session_id = book.bid.split("_")[1]
            first5 = session_id[:5]
            odir = op.join(output_root, book.tel_tube, book.type, first5)
            if not op.exists(odir):
                os.makedirs(odir)
            book_path = os.path.join(odir, book.bid)

            # after sanity checks, now we proceed to bind the book.
            # get files associated with this book, in the form of
            # a dictionary of {stream_id: [file_paths]}
            filedb = self.get_files_for_book(book)
            obsdb = self.get_g3tsmurf_obs_for_book(book)
            readout_ids = self.get_readout_ids_for_book(book)

            # bind book using bookbinder library
            bookbinder = BookBinder(
                book, obsdb, filedb, data_root, readout_ids, book_path,
                ignore_tags=ignore_tags,
            )
            return bookbinder

        elif book.type in ["hk", "smurf"]:
            # get source directory for hk book
            root = op.join(data_root, book.type)
            first5 = book.bid.split("_")[1]
            assert first5.isdigit(), f"first5 of {book.bid} is not a digit"
            book_path_src = op.join(root, first5)

            # get target directory for hk book
            odir = op.join(output_root, book.tel_tube, book.type)
            if not op.exists(odir):
                os.makedirs(odir)
            book_path_tgt = os.path.join(odir, book.bid)

            class _FakeBinder:  # dummy class to mimic baseline bookbinder
                def __init__(self, indir, outdir):
                    self.indir = indir
                    self.outdir = outdir

                def get_metadata(self):
                    return {
                        "book_id": book.bid,
                        # dummy start and stop times
                        "start_time": float(first5) * 1e5,
                        "stop_time": (float(first5) + 1) * 1e5,
                        "telescope": book.tel_tube.lower(),
                        "type": book.type,
                    }

                def bind(self, pbar=False):
                    shutil.copytree(
                        self.indir,
                        self.outdir,
                        ignore=shutil.ignore_patterns(
                            "*.dat", "*_mask.txt", "*_freq.txt"
                        ),
                    )

            return _FakeBinder(book_path_src, book_path_tgt)

        elif book.type in ["stray"]:
            flist = self.get_files_for_book(book)

            # get source directory for stray book
            root = op.join(data_root, "timestreams")
            first5 = book.bid.split("_")[1]
            assert first5.isdigit(), f"first5 of {book.bid} is not a digit"
            book_path_src = op.join(root, first5)

            # get target directory for hk book
            odir = op.join(output_root, book.tel_tube, book.type)
            if not op.exists(odir):
                os.makedirs(odir)
            book_path_tgt = os.path.join(odir, book.bid)

            class _FakeBinder:  # dummy class to mimic baseline bookbinder
                def __init__(self, indir, outdir, file_list):
                    self.indir = indir
                    self.outdir = outdir
                    self.file_list = file_list

                def get_metadata(self):
                    return {
                        "book_id": book.bid,
                        # dummy start and stop times
                        "start_time": float(first5) * 1e5,
                        "stop_time": (float(first5) + 1) * 1e5,
                        "telescope": book.tel_tube.lower(),
                        "type": book.type,
                    }

                def bind(self, pbar=False):
                    if not os.path.exists(self.outdir):
                        os.makedirs(self.outdir)
                    for f in self.file_list:
                        relpath = os.path.relpath(f, self.indir)
                        path = os.path.join(self.outdir, relpath)
                        base, _ = os.path.split(path)
                        if not os.path.exists(base):
                            os.makedirs(base)
                        shutil.copy(f, os.path.join(self.outdir, relpath))

            return _FakeBinder(book_path_src, book_path_tgt, flist)
        else:
            raise NotImplementedError(
                f"binder for book type {book.type} not implemented"
            )

    def bind_book(
        self,
        book,
        session=None,
        output_root="out",
        message="",
        test_mode=False,
        pbar=False,
        ignore_tags=False
    ):
        """Bind book using bookbinder

        Parameters
        ----------
        bid: str
            book id
        session: BookDB session
        output_root: str
            output root directory
        message: string
            message to be added to the book
        test_mode : bool
            If in test_mode, this function will still run on already copied
            books, and will not update any db fields in the db. This is useful
            for testing purposes.
        ignore_tags : bool
            If true, book will be bound even if the tags between different level 2
            operations don't match. Only ever expected to be turned on by hand.
        """
        if session is None:
            session = self.get_session()
        # get book id and book object, depending on whether book id is given or not
        if isinstance(book, Books):
            bid = book.bid
        elif isinstance(book, str):
            bid = book
            book = self.get_book(bid, session=session)
        else:
            raise NotImplementedError
        # check whether book exists in the database
        if not self.book_exists(bid, session=session):
            raise BookExistsError(f"Book {bid} does not exist in the database")
        # check whether book is already bound
        if (book.status == BOUND) and (not test_mode):
            raise BookBoundError(f"Book {bid} is already bound")
        assert book.type in VALID_OBSTYPES

        try:
            # find appropriate binder for the book type
            binder = self._get_binder_for_book(
                book, 
                output_root=output_root,
                ignore_tags=ignore_tags,
            )
            binder.bind(pbar=pbar)

            # write M_book file
            m_book_file = os.path.join(binder.outdir, "M_book.yaml")
            book_meta = {}
            book_meta["book"] = {
                "type": book.type,
                "schema_version": 0,
                "book_id": book.bid,
                "finalized_at": dt.datetime.utcnow().isoformat(),
            }
            book_meta["bookbinder"] = {
                "codebase": sotodlib.__file__,
                "version": sotodlib.__version__,
                "context": self.config.get("context", "unknown"),
            }
            with open(m_book_file, "w") as f:
                yaml.dump(book_meta, f)

            # write M_index file
            mfile = os.path.join(binder.outdir, "M_index.yaml")
            with open(mfile, "w") as f:
                yaml.dump(binder.get_metadata(), f)

            # not sure if this is the best place to update
            book.status = BOUND
            book.path = op.abspath(binder.outdir)
            self.logger.info("Book {} bound".format(book.bid))
            book.message = message
            if not test_mode:
                session.commit()
            else:
                session.rollback()
        except Exception as e:
            session.rollback()
            book.status = FAILED
            self.logger.error("Book {} failed".format(book.bid))
            err_msg = traceback.format_exc()
            self.logger.error(err_msg)
            message = f"{message}\ntrace={err_msg}" if message else err_msg
            book.message = message

            if not test_mode:
                session.commit()
            else:
                session.rollback()

            raise e

    def get_book(self, bid, session=None):
        """Get book from database.

        Parameters
        ----------
        bid: str
            book id
        session: BookDB session

        Returns
        -------
        book: book object or None if not found

        """
        if session is None:
            session = self.get_session()
        return session.query(Books).filter(Books.bid == bid).one_or_none()

    def get_books(self, session=None):
        """Get all books from database.

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of book objects

        """
        if session is None:
            session = self.get_session()
        return session.query(Books).all()

    def get_books_by_status(self, status, session=None):
        """Get all books with a given status from database.

        Parameters
        ----------
        session: BookDB session
        status: int
        
        Returns
        -------
        books: list of book objects

        """
        if session is None: session = self.get_session()
        return session.query(Books).filter(Books.status == status).all()
    
    def get_level2_deleteable_books(self, session=None, cleanup_delay=None, max_time=None):
        """Get all bound books from database where we need to delete the level2
        data

        Parameters
        ----------
        session: BookDB session
        cleanup_delay: float
            amount of time to delay book deletation relative to g3tsmurf finalization
            time in units of days. 
        max_time: datetime
            maxmimum time of book start to search. Overrides cleanup_delay if
            earlier
        
        Returns
        -------
        books: list of book objects
        """

        if session is None:
            session = self.get_session()
        if cleanup_delay is None:
            cleanup_delay = 0

        base_filt = and_(
            Books.status == BOUND,
            Books.lvl2_deleted == False,
            or_(  ## not implementing smurf deletion just yet
                Books.type == "obs",
                Books.type == "oper",
                Books.type == "stray",
                Books.type == "hk",
            ),
        )
        sources = session.query(Books.tel_tube).filter(base_filt).distinct().all()
        source_filt = []
        for source, in sources:
            streams = self.sources[source].get("slots")
            _, SMURF = self.get_g3tsmurf_session(source, return_archive=True)
            limit = SMURF.get_final_time(streams, check_control=False)
            max_stop = dt.datetime.utcfromtimestamp(limit) - dt.timedelta(days=cleanup_delay)
            
            source_filt.append( and_(Books.tel_tube == source, Books.stop <= max_stop) )

        q = session.query(Books).filter(
            base_filt,
            or_(*source_filt),
        )
        
        if max_time is not None:
            q = q.filter(Books.stop <= max_time)

        return q.all()

    # some aliases for readability
    def get_unbound_books(self, session=None):
        """Get all unbound books from database.

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of book objects

        """
        return self.get_books_by_status(UNBOUND, session)

    def get_bound_books(self, session=None):
        """Get all bound books from database.

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of book objects

        """
        return self.get_books_by_status(BOUND, session)
    
    def get_failed_books(self, session=None):
        """Get all failed books from database

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of books

        """
        return self.get_books_by_status(FAILED, session)

    def get_rebind_books(self, session=None):
        """Get all books to be rebinded from database

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of books

        """
        return self.get_books_by_status(REBIND, session)

    def get_uploaded_books(self, session=None):
        """Get all books uploaded to librarian from database

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of books

        """
        return self.get_books_by_status(UPLOADED, session)

    def book_exists(self, bid, session=None):
        """Check if a book exists in the database.

        Parameters
        ----------
        session: BookDB session
        bid: str
            book id

        Returns
        -------
        boolean: True if book exists, False otherwise

        """
        if session is None:
            session = self.get_session()
        return self.get_book(bid, session=session) is not None

    def book_bound(self, bid, session=None):
        """Check if a book has been bound

        Parameters
        ----------
        bid: str
            book id
        session: BookDB session

        Returns
        -------
        boolean: True if book is bound, False otherwise

        """
        if session is None:
            session = self.get_session()
        book = self.get_book(bid, session=session)
        if book is None:
            return False
        return book.status == BOUND

    def set_book_wont_bind(self, book, message=None, session=None):
        """Change book status to WONT_BIND, meaning the files indicated into 
        this book will not be bound. .g3 files here will go into stray books.

        This function should not be integrated into automated data packaging.

        This book status is necessary because without it the automated data 
        packaging could continue to try and register and fail on the same book. 

        Parameters
        -----------
        book: str or Book 
        message: str or None
            if not none, update book message to explain why the setting is changing
        session: BookDB session.
        
        """
        if session is None:
            session = self.get_session()

        if isinstance(book, str):
            book = self.get_book(book)

        if book.status != FAILED:
            self.logger.warning(
                f"Book {book} has not failed before being set not to WONT_BIND"
            )
        book.status = WONT_BIND 
        if message is not None:
            book.message = message
        session.commit()

    def commit(self, session=None):
        """Commit the book db.

        Parameters
        ----------
        session: BookDB session

        """
        if session is None:
            session = self.get_session()
        session.commit()

    def rollback(self, session=None):
        """Similar to commit, rollback the session

        Parameters
        ----------
        session: BookDB session

        """
        if session is None:
            session = self.get_session()
        session.rollback()

    @loop_over_sources
    def update_bookdb_from_g3tsmurf(
        self,
        source,
        min_ctime=None,
        max_ctime=None,
        min_overlap=30,
        ignore_singles=False,
        stream_ids=None,
        force_single_stream=False,
    ):
        """Update bdb with new observations from g3tsmurf db.

        Parameters
        ----------
        source: str
            g3tsmurf db source. e.g., 'sat1', 'latrt', 'tsat'
        min_ctime: float
            minimum ctime to consider (in seconds since unix epoch)
        max_ctime: float
            maximum ctime to consider (in seconds since unix epoch)
        min_overlap: float
            minimum overlap between book and observation (in seconds)
        ignore_singles: boolean
            if True, ignore single observations
        stream_ids: list
            a list of stream ids (str) to consider. Defaults to list in
            imprinter configuration file.
        force_single_stream: boolean
            if True, treat observations from different streams separately
        """
        session, SMURF = self.get_g3tsmurf_session(source, return_archive=True)
        # set sensible ctime range is none is given
        if min_ctime is None:
            min_ctime = (
                session.query(G3tObservations.timestamp)
                .order_by(G3tObservations.timestamp)
                .first()[0]
            )
        if max_ctime is None:
            max_ctime = dt.datetime.now().timestamp()

        # get wafers
        if stream_ids is None:
            streams = self.sources[source].get("slots")
            if streams is None:
                raise ValueError(
                    f"Imprinter missing slot / stream_id" " information for {source}"
                )
        else:
            # if user input is present, format it like the query response
            streams = stream_ids
        self.logger.debug(f"Looking for observations from stream_ids {streams}")

        # restrict to given stream ids (wafers)
        stream_filt = or_(*[G3tObservations.stream_id == s for s in streams])

        # check data transfer finalization
        final_time = SMURF.get_final_time(
            streams, min_ctime, max_ctime, check_control=True
        )
        if final_time < max_ctime:
            max_ctime = final_time

        # check for incomplete observations in time range
        q_incomplete = session.query(G3tObservations).filter(
            G3tObservations.timestamp >= min_ctime,
            G3tObservations.timestamp <= max_ctime,
            stream_filt,
            G3tObservations.stop == None,
        )
        # if we have incomplete observations in our stream_id list we cannot
        # bookbind any observations overlapping the incomplete ones.
        if q_incomplete.count() > 0:
            max_ctime = min([obs.timestamp for obs in q_incomplete.all()])
        max_stop = dt.datetime.utcfromtimestamp(max_ctime)

        # find all complete observations that start within the time range
        obs_q = session.query(G3tObservations).filter(
            G3tObservations.timestamp >= min_ctime,
            G3tObservations.timestamp < max_ctime,
            stream_filt,
            G3tObservations.stop < max_stop,
            not_(G3tObservations.stop == None),
        )

        output = []
        for stream in streams:
            # loop through all observations for this particular stream_id
            for str_obs in obs_q.filter(G3tObservations.stream_id == stream).all():
                # distinguish different types of books
                # operation book
                if get_obs_type(str_obs) == "oper":
                    output.append(
                        ObsSet(
                            [str_obs],
                            mode="oper",
                            slots=self.sources[source]["slots"],
                            tel_tube=source,
                        )
                    )
                elif get_obs_type(str_obs) == "obs":
                    # force each observation to be its own book
                    if force_single_stream:
                        output.append(
                            ObsSet(
                                [str_obs],
                                mode="obs",
                                slots=self.slots,
                                tel_tube=self.tel_tube,
                            )
                        )
                    else:
                        # query for all possible types of overlapping observations from other streams
                        q = obs_q.filter(
                            G3tObservations.stream_id != str_obs.stream_id,
                            stream_filt,
                            or_(
                                and_(
                                    G3tObservations.start <= str_obs.start,
                                    G3tObservations.stop >= str_obs.stop,
                                ),
                                and_(
                                    G3tObservations.stop >= str_obs.start,
                                    G3tObservations.stop <= str_obs.stop,
                                ),
                                and_(
                                    G3tObservations.start >= str_obs.start,
                                    G3tObservations.start <= str_obs.stop,
                                ),
                            ),
                        )

                        # if we failed to find any overlapping observations
                        if q.count() == 0 and not ignore_singles:
                            # append only this one when there's no overlapping segments
                            output.append(
                                ObsSet(
                                    [str_obs],
                                    mode="obs",
                                    slots=self.sources[source]["slots"],
                                    tel_tube=source,
                                )
                            )

                        elif q.count() > 0:
                            # obtain overlapping observations (returned as a tuple of stream_id)
                            obs_list = q.all()
                            # add the current obs too
                            obs_list.append(str_obs)
                            # check to make sure ALL observations overlap all others
                            # and the overlap passes the minimum requirement
                            overlap_time = np.min([o.stop for o in obs_list]) - np.max(
                                [o.start for o in obs_list]
                            )
                            if overlap_time.total_seconds() < 0:
                                continue
                            if overlap_time.total_seconds() < min_overlap:
                                continue
                            # add all of the possible overlaps
                            output.append(
                                ObsSet(
                                    obs_list,
                                    mode="obs",
                                    slots=self.sources[source]["slots"],
                                    tel_tube=source,
                                )
                            )

        # remove exact duplicates in output
        output = drop_duplicates(output)

        # now our output is a list of ObsSet, where each ObsSet contains
        # all observations that overlap each other which should go
        # into the same book.

        # register books in the book database (bdb)
        for oset in output:
            try:
                self.register_book(oset, commit=False)
            except BookExistsError:
                self.logger.debug(f"Book already registered: {oset}, skipping")
                continue
            except NoFilesError:
                self.logger.warning(f"No files found for {oset}, skipping")
                continue
        self.commit()

    def get_files_for_book(self, book):
        """Get all files in a book

        Parameters
        ----------
        book: Book object

        Returns
        -------
        files: dict if book.type is 'obs' or 'oper', otherwise list
            {obs_id: [file_paths...]}

        """
        if book.type in ["obs", "oper", "stray"]:
            session = self.get_g3tsmurf_session(book.tel_tube)
        if book.type in ["obs", "oper"]:
            obs_ids = [o.obs_id for o in book.obs]
            obs = (
                session.query(G3tObservations)
                .filter(G3tObservations.obs_id.in_(obs_ids))
                .all()
            )

            res = OrderedDict()
            for o in obs:
                res[o.obs_id] = sorted([f.name for f in o.files])
            return res
        elif book.type in ["stray"]:
            ## TODO: errant stream ids
            ## TODO: wont_bind book files
            ## TODO: Files that have not been added to any book
            tcode = int(book.bid.split("_")[1])
            streams = self.sources[book.tel_tube].get("slots")
            stream_filt_files = or_(*[Files.stream_id == s for s in streams])
            flist = (
                session.query(Files)
                .filter(
                    Files.name.like(f"%/{tcode}/%"),
                    stream_filt_files,
                    Files.obs_id == None,
                )
                .all()
            )
            return [f.name for f in flist]
        elif book.type == "hk":
            HK = self.get_g3thk(book.tel_tube)
            flist = (
                HK.session.query(HKFiles)
                .filter(
                    HKFiles.global_start_time >= book.start.timestamp(),
                    HKFiles.global_start_time < book.stop.timestamp(),
                )
                .all()
            )
            return [f.path for f in flist]

        else:
            raise NotImplementedError(
                f"book type {book.type} not understood for" " file search"
            )

    def get_readout_ids_for_book(self, book):
        """
        Get all readout IDs for a book

        Parameters
        -----------
        book: Book object

        Returns
        -------
        readout_ids: dict
            {obs_id: [readout_ids...]}}

        """
        _, SMURF = self.get_g3tsmurf_session(book.tel_tube, return_archive=True)
        out = {}
        # load all obs and associated files
        for obs_id, files in self.get_files_for_book(book).items():
            self.logger.info(f"Retrieving readout_ids for {obs_id}...")
            status = SmurfStatus.from_file(files[0])
            ch_info = get_channel_info(status, archive=SMURF)
            if "readout_id" not in ch_info:
                raise ValueError(
                    f"Readout IDs not found for {obs_id}. Indicates issue with G3tSmurf Indexing"
                )
            checks = ["NONE" in rid for rid in ch_info.readout_id]
            if np.all(checks):
                raise ValueError(
                    f"Readout IDs not found for {obs_id}. Indicates issue with G3tSmurf Indexing"
                )
            if np.any(checks):
                self.logger.warning(
                    f"Found {sum(checks)} channels without readout_id. Were fixed tones running?"
                )
            # make sure all rchannel ids are sorted
            assert list(ch_info.rchannel) == sorted(ch_info.rchannel)
            out[obs_id] = ch_info.readout_id
        return out

    def copy_smurf_files_to_book(self, book, book_path):
        """
        Copies smurf ancillary files to an operation book.

        Parameters
        -----------
        book : Books
            book object
        book_path : path
            Output path of book

        Returns
        ---------
        files : List[path]
            list of all metadata files copied to the book
        meta : dict[path]
            Dictionary of destination paths of important metadata such as
            tunes, bgmaps, or IVs.
        """
        if book.type != "oper":
            raise TypeError("Book must have type 'oper'")

        session, arc = self.get_g3tsmurf_session(book.tel_tube, return_archive=True)

        obs_ids = [o.obs_id for o in book.obs]
        obs = (
            session.query(G3tObservations)
            .filter(G3tObservations.obs_id.in_(obs_ids))
            .all()
        )

        files = []
        for ob in obs:
            files.extend(get_smurf_files(ob, arc.meta_path))

        dirname = os.path.join(book_path, "smurf")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        meta_files = {}
        for f in files:
            basename = os.path.basename(f)
            dest = os.path.join(book_path, basename)
            shutil.copyfile(f, dest)

            if f.endswith("iv_analysis.npy"):
                meta_files["iv"] = basename
            elif f.endswith("bg_map.npy"):
                meta_files["bgmap"] = basename
            elif f.endswith("bias_step_analysis.npy"):
                meta_files["bias_steps"] = basename
            elif f.endswith("take_noise.npy"):
                meta_files["noise"] = basename

        return files, meta_files

    def get_g3tsmurf_obs_for_book(self, book):
        """
        Get all g3tsmurf observations for a book

        Parameters
        -----------
        book: Book object

        Returns
        -------
        obs: dict
            {obs_id: G3tObservations}

        """
        session = self.get_g3tsmurf_session(book.tel_tube)
        obs_ids = [o.obs_id for o in book.obs]
        obs = (
            session.query(G3tObservations)
            .filter(G3tObservations.obs_id.in_(obs_ids))
            .all()
        )
        return {o.obs_id: o for o in obs}

    def delete_level2_files(self, book, dry_run=True):
        """Delete level 2 data from already bound books

        Parameters
        ----------
        book: book object
        dry_run: bool
            if true, just prints plans to self.logger.info
        """
        if book.status != BOUND:
            raise ValueError(f"Book must be bound to delete level 2 files")

        self.logger.info(f"Removing level 2 files for {book.bid}")
        if book.type == "obs" or book.type == "oper":
            session, SMURF = self.get_g3tsmurf_session(
                book.tel_tube, return_archive=True
            )
            odic = self.get_g3tsmurf_obs_for_book(book)

            for oid, obs in odic.items():
                SMURF.delete_observation_files(
                    obs, session, dry_run=dry_run, my_logger=self.logger
                )
        elif book.type == "stray":
            session, SMURF = self.get_g3tsmurf_session(
                book.tel_tube, return_archive=True
            )
            flist = self.get_files_for_book(book)
            for f in flist:
                db_file = session.query(Files).filter(Files.name == f).one()
                SMURF.delete_file(
                    db_file, session, dry_run=dry_run, my_logger=self.logger
                )
        elif book.type == "hk":
            HK = self.get_g3thk(book.tel_tube)
            flist = self.get_files_for_book(book)
            for f in flist:
                hkfile = HK.session.query(HKFiles).filter(HKFiles.path == f).one()
                HK.delete_file(hkfile, dry_run=dry_run, my_logger=self.logger)
        else:
            raise NotImplementedError(
                f"Do not know how to delete level 2 files"
                f" for book of type {book.type}"
            )
        if not dry_run:
            book.lvl2_deleted = True
            self.session.commit()

    def delete_book_files(self, book):
        """Delete all files associated with a book

        Parameters
        ----------
        book: Book object

        """
        # remove all files within the book
        try:
            shutil.rmtree(book.path)
        except Exception as e:
            self.logger.warning(f"Failed to remove {book.path}: {e}")
            self.logger.error(traceback.format_exc())


    def all_bound_until(self):
        """report a datetime object to indicate that all books are bound
        by this datetime.
        """
        session = self.get_session()
        # sort by start time and find the start time by which
        # all books are bound
        books = session.query(Books).order_by(Books.start).all()
        for book in books:
            if book.status < BOUND:
                return book.start
        return book.start  # last book

#####################
# Utility functions #
#####################


_primary_idx_map = {}


def get_frame_times(frame):
    """
    Returns timestamps for a G3Frame of detector data.

    Parameters
    --------------
    frame : G3Frame
        Scan frame containing detector data

    Returns
    --------------
    high_precision : bool
        If true, timestamps are computed from timing counters. If not, they are
        software timestamps

    timestamps : np.ndarray
        Array of timestamps for samples in the frame, in G3Time units (1e-8 sec)

    """
    if len(_primary_idx_map) == 0:
        for i, name in enumerate(frame["primary"].names):
            _primary_idx_map[name] = i

    c0 = frame["primary"].data[_primary_idx_map["Counter0"]]
    c2 = frame["primary"].data[_primary_idx_map["Counter2"]]

    if np.any(c0):
        return True, counters_to_timestamps(c0, c2) * core.G3Units.s
    else:
        return False, np.array(frame["data"].times)


def get_start_and_end(files):
    """
    Gets start and end time for a list of L2 detector data files

    Parameters
    -------------
    files : np.ndarray
        List of L2 G3 files for a single detector set in an observation

    Returns
    --------
    t0 : float
        Start time of observation
    t1 : float
        End time or observation
    """
    _files = sorted(files)

    # Get start time
    found_scan = False
    for file in _files:
        for frame in core.G3File(file):
            if frame.type == core.G3FrameType.Scan:
                t0 = get_frame_times(frame)[1][0]
                found_scan = True
                break
        if found_scan:
            break

    # Get end time
    frame = None
    found_scan = False
    for file in _files[::-1]:
        for _frame in core.G3File(file):
            if _frame.type == core.G3FrameType.Scan:
                frame = _frame
                found_scan = True

        if found_scan:
            break

    t1 = get_frame_times(frame)[1][-1]

    return t0, t1


def create_g3tsmurf_session(config):
    """create a connection session to g3tsmurf database

    Parameters
    ----------
    config: str
        path to a yaml file that contains g3tsmurf db conn details
        or it could be a dictionary with keys, etc.

    Returns
    -------
    session: sqlalchemy session
        a session to the g3tsmurf database

    """
    # create database connection
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    config["db_args"] = {"connect_args": {"check_same_thread": False}}
    SMURF = G3tSmurf.from_configs(config)
    session = SMURF.Session()
    return session, SMURF


def stream_timestamp(obs_id):
    """parse observation id to obtain a (stream, timestamp) pair

    Parameters
    ----------
    obs_id: str
        observation id

    Returns
    -------
    stream: str
        stream id
    timestamp: str
        timestamp of the observation

    """
    return "_".join(obs_id.split("_")[1:-1]), obs_id.split("_")[-1]


def get_obs_type(obs: G3tObservations):
    """Get the type of observation based on the observation id

    Parameters
    ----------
    obs: G3tObservations object

    Returns
    -------
    obs_type: str

    """
    if obs.tag is None:
        return None
    tags = obs.tag.split(",")
    # assert tags[0] in VALID_OBSTYPES, f"Unknown tag {tags[0]}"  # this is too strict for now
    return tags[0]


class ObsSet(list):
    """A thin wrapper around a list of observations that are all
    part of the same book"""

    def __init__(self, *args, mode="obs", slots=None, tel_tube=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.slots = slots  # all possible slots (not those in the ObsSet)
        self.tel_tube = tel_tube

    @property
    def obs_ids(self):
        """list of observation ids"""
        return sorted([obs.obs_id for obs in self])

    def contains_stream(self, stream_id):
        """check whether an ObsSet has a given stream id

        Parameters
        ----------
        obsset: dict with obs ids as keys

        Returns
        -------
        boolean

        """
        for obs_id in self.obs_ids:
            if stream_id in stream_timestamp(obs_id)[0]:
                return True
        return False

    def __eq__(self, other):
        """check if two ObsSets are equal"""
        return self.obs_ids == other.obs_ids

    def get_id(self, mode=None):
        """get a unique id for this ObsSet. Following the specification
        in the TOD format document.

        Parameters
        ----------
        mode: str
            observation mode / type, e.g., obs, oper, etc.

        Returns
        -------
        id: str
            unique id for this ObsSet, used to identify the book in the book database

        """
        # if no mode is specified, use the one provided for the set
        if mode is None:
            mode = self.mode

        assert mode in VALID_OBSTYPES, f"Invalid mode {mode}"

        # common values across modes
        # parse observation time and name the book with the first timestamp
        _, timestamp = stream_timestamp(self.obs_ids[0])

        # the rest of the id depends on the mode
        if mode in ["obs", "oper"]:
            # get slot flags
            slot_flags = ""
            for slot in self.slots:
                slot_flags += "1" if self.contains_stream(slot) else "0"
            bid = f"{mode}_{timestamp}_{self.tel_tube}_{slot_flags}"
        else:
            raise NotImplementedError

        return bid

    def __str__(self):
        return f"{self.get_id()}: {self.obs_ids}"


def drop_duplicates(obsset_list: List[ObsSet]):
    """Drop duplicate obssets from a list of obssets

    Parameters
    ----------
    obsset_list: list of ObsSet objects

    Returns
    -------
    obsset_list: list of ObsSet objects
        with duplicates removed
    """
    ids_list = [oset.obs_ids for oset in obsset_list]
    new_obsset_list = list(
        OrderedDict(
            (tuple(ids), oset) for (ids, oset) in zip(ids_list, obsset_list)
        ).values()
    )
    return new_obsset_list


def require(dct, keys):
    """
    Check whether a dictionary contains all the required keys.

    Parameters
    ----------
    dct: dict
        dictionary to check
    keys: list
        list of required keys
    """
    if dict is None:
        raise ValueError("Missing required dictionary")
    for k in keys:
        if k not in dct:
            raise ValueError(f"Missing required key {k}")
