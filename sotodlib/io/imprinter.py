import datetime as dt, os.path as op, os
import numpy as np
from collections import OrderedDict
from typing import List
import yaml, traceback
import shutil
import logging
from pathlib import Path
from glob import glob
import time

import sqlalchemy as db
from sqlalchemy import or_, and_, not_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import so3g
from spt3g import core

from .bookbinder import BookBinder, TimeCodeBinder
from .load_smurf import (
    G3tSmurf,
    Observations as G3tObservations,
    SmurfStatus,
    get_channel_info,
    TimeCodes,
    SupRsyncType,
    Files,
)
from .datapkg_utils import load_configs, get_imprinter_config
from .check_book import BookScanner
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

# file patterns excluded from smurf books
SMURF_EXCLUDE_PATTERNS = ["*.dat", "*_mask.txt", "*_freq.txt"]

# file size limits
MAX_OBS_LVL2_SIZE = 10 # Gb per level 2 file
MAX_OPER_LVL2_SIZE = 5 # Gb per level 2 file

class BookExistsError(Exception):
    """Exception raised when a book already exists in the database"""
    pass

class BookBoundError(Exception):
    """Exception raised when a book is already bound"""
    pass

class NoFilesError(Exception):
    """Exception raised when no files are found in the book"""
    pass

class MissingReadoutIDError(Exception):
    """Exception raised when we find books with observations with missing readout IDs"""
    pass

class OverlapObsError(Exception):
    """Exception raised when we find observations that could be registered to
    multiple books"""
    pass

class FileTooLargeError(Exception):
    """Exception raised when we find level 2 files that are larger than our
    maximum allowable sizes"""
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
    path: str, location of book directory relative to output_root of imprinter
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
    schema = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<Book: {self.bid}>"


##############
# main logic #
##############


# convenient decorator to repeat a method over all data sources
def loop_over_tubes(method):
    def wrapper(self, *args, **kwargs):
        outs = []
        for tube in self.tubes:
            x = method(self, tube, *args, **kwargs)
            if x is not None:
                outs.extend(x)
        if len(outs)>0:
            return outs        
    return wrapper


class Imprinter:
    def __init__(self, im_config=None, db_args={}, logger=None, make_db=False):
        """Imprinter manages the book database.
        
        Imprinter at the site is set up to be one per level 2 daq node with a
        one-to-one matching between imprinter instance and g3tsmurf instance. 

        Example configuration file::
        
          db_path: imprinter.db
          daq_node: daq-node
          g3tsmurf: /path/to/config.yaml
          output_root: /abs/path/to/output
          librarian_conn: string (optional)
          build_hk: True 
          build_det: True
          hk_fields:
            az: acu.acu_udp_stream.Corrected_Azimuth
            el: acu.acu_udp_stream.Corrected_Elevation
            boresight: acu.acu_udp_stream.Corrected_Boresight
            az_mode:  acu.acu_status.Azimuth_mode
            hwp_freq: hwp-bbb-e1.HWPEncoder.approx_hwp_freq

          tel_tubes:
            tel_tube1:
              tube_slot: c0
              wafer_slots:
                - wafer_slot: ws0
                  stream_id: stream_id0
                - wafer_slot: ws1
                  stream_id: stream_id1
                - wafer_slot: ws2
                  stream_id: stream_id2
            tel_tube2:
              tube_slot: i1
              wafer_slots:
                - wafer_slot: ws0
                  stream_id: stream_id3
                - wafer_slot: ws1
                  stream_id: stream_id4
                - wafer_slot: ws2
                  stream_id: None
          
        Standard Book directory structure based off config file example:
        output_root/
            tel_tube1/
                obs/
                oper/
            tel_tube2/
                obs/
                oper/
            daq-node/
                smurf/
                stray/
                hk/
        
        The tel_tubes entry lists the different tel-tubes that will be bound
        into separate books. Each tel_tube entry is expected to have a tube_slot
        name and is required to have a wafer_slots list. Each wafer slot must
        have a wafer_slot and stream_id entry. If stream_id is None then it will
        be treated as an empty slot. The wafers/stream_ids listed in each
        tel_tube are bound together. For the SATs we expect one entry under 
        tel_tubes but the LAT will have many.
        
        The build_det_data and build_hk entries determines if detector books
        (obs,oper,smurf,stray) and housekeeping (hk) books should be made,
        respectively. 

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
        self.config = load_configs(im_config)

        self.db_path = self.config.get("db_path")
        self.daq_node = self.config.get("daq_node")        
        self.output_root = self.config.get("output_root")
        self.g3tsmurf_config = self.config.get("g3tsmurf")
        g3tsmurf_cfg = load_configs(self.g3tsmurf_config)
        self.lvl2_data_root = g3tsmurf_cfg["data_prefix"]

        self.build_hk = self.config.get("build_hk")
        self.build_det = self.config.get("build_det")

        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger("imprinter")
            if not self.logger.hasHandlers():
                self.logger = init_logger("imprinter")

        self.tube_configs = self.config.get("tel_tubes", {})
        self.tubes = dict()
        for tube, cfg in self.tube_configs.items():
            self.tubes[tube] = dict()
            self.tubes[tube]['slots'] = list()
            if 'wafer_slots' not in self.tube_configs[tube]:
                if 'slots' in self.tube_configs[tube]:
                    self.tubes[tube]['slots'] = self.tube_configs[tube]['slots']
                    self.logger.warning(f"Tube {tube} missing wafer_slots "
                    "entry. Are you using an old configuration format?")
                    continue
                
            for slot in self.tube_configs[tube]['wafer_slots']:
                if slot['stream_id'].lower() == 'none':
                    self.tubes[tube]['slots'].append(
                        f"NONE_{slot['wafer_slot']}"
                    )
                else:
                    self.tubes[tube]['slots'].append(slot['stream_id'])
                                        
        # raise error if database does not exist
        if not op.exists(self.db_path):
            if not make_db:
                raise ValueError(
                    f"Imprinter database path {self.db_path} does not exist "
                    "pass make_db=True to make database"
                )
            else:
                # check whether db_path directory exists
                if not op.exists(op.dirname(self.db_path)):
                    # we first make sure that the folder exists
                    os.makedirs(
                        op.abspath(op.dirname(self.db_path)), exist_ok=True
                    )
        self.engine = db.create_engine(f"sqlite:///{self.db_path}", **db_args)

        # create all tables or (do nothing if tables exist)
        Base.metadata.create_all(self.engine)

        self.session = None
        self.g3tsmurf_session = None
        self.archive = None
        self.hk_archive = None
        self.librarian = None

    @classmethod
    def for_platform(cls, platform, *args, **kwargs):
        config = get_imprinter_config(platform)
        return cls( config, *args, **kwargs)

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

    def get_g3tsmurf_session(self, return_archive=False):
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
        if self.g3tsmurf_session is None:
            (
                self.g3tsmurf_session,
                self.archive,
            ) = create_g3tsmurf_session(self.g3tsmurf_config)
        if not return_archive:
            return self.g3tsmurf_session
        return self.g3tsmurf_session, self.archive

    def get_g3thk(self):
        """Get a G3tHk database using the g3tsmurf config file."""
        if self.hk_archive is None:
            self.hk_archive = G3tHk.from_configs(
                self.g3tsmurf_config
            )
        return self.hk_archive

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
        assert obsset.mode in ['obs','oper']
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
        observations = [
            Observations(obs_id=obs_id) for obs_id in obsset.obs_ids
        ]
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
            schema=0,
        )
        book.path = self.get_book_path(book)

        self.logger.info(f"registering {bid}")
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

    def register_hk_books(self, min_ctime=None, max_ctime=None, session=None):
        """Register housekeeping books to database"""
        session = session or self.get_session()
        if not self.build_hk:
            return

        if min_ctime is None:
            min_ctime = 16000e5
        if max_ctime is None:
            max_ctime = 5e10

        # all ctime dir except the last ctime dir will be considered complete
        ctime_dirs = sorted(glob(op.join(self.lvl2_data_root, "hk", "*")))
        for ctime_dir in ctime_dirs[:-1]:
            ctime = op.basename(ctime_dir)
            if int(ctime) < int(min_ctime//1e5):
                continue
            if int(ctime) > int(max_ctime//1e5):
                continue
            book_id = f"hk_{ctime}_{self.daq_node}"
            # check whether the book exists
            if self.get_book(book_id) is not None:
                continue
            self.logger.info(f"registering {book_id}")
            book = Books(
                bid=book_id,
                type="hk",
                status=UNBOUND,
                start=dt.datetime.utcfromtimestamp(int(ctime) * 1e5),
                stop=dt.datetime.utcfromtimestamp((int(ctime) + 1) * 1e5),
                tel_tube=self.daq_node,  
                schema=0,
            )
            book.path = self.get_book_path(book)
            session.add(book)
            session.commit()

    def register_timecode_books(
        self, 
        min_ctime=None, 
        max_ctime=None,
        session=None
    ):
        """Register smurf and stray books with database. We do not expect many
        stray books as nearly all .g3 files will be associated with some obs or
        oper book.

        stray and smurf books are made per DAQ node, meaning there will be one
        per imprinter instance on the site setup. The making of these books
        depends on the imprinter configuration having the correct stream_ids
        indexed per tel-tube. All stream_ids not part of those lists
        will be moved to stray books but only the finalization information from
        the specified stream ids will be used to determine when to output these
        books.

        smurf books are registered whenever all the relevant metadata timecode
        entries have been found. stray books are registered when metadata and
        file timecode entries exist AND all obs/oper books in that time
        range have been bound successfully.
        """

        if not self.build_det:
            return
        session = session or self.get_session()
        g3session, SMURF = self.get_g3tsmurf_session(return_archive=True)

        final_time = SMURF.get_final_time(
            self.all_slots, min_ctime, max_ctime, check_control=False
        )
        final_tc = int(final_time//1e5)
        servers = SMURF.finalize["servers"]
        meta_agents = [s["smurf-suprsync"] for s in servers]
        files_agents = [s["timestream-suprsync"] for s in servers]

        meta_query = or_(*[TimeCodes.agent == a for a in meta_agents])
        files_query = or_(*[TimeCodes.agent == a for a in files_agents])

        tcs = g3session.query(TimeCodes.timecode)
        if min_ctime is not None:
            tcs = tcs.filter(TimeCodes.timecode >= int(min_ctime//1e5))
        if max_ctime is not None:
            tcs = tcs.filter(TimeCodes.timecode <= int(max_ctime//1e5)+1)
        tcs = tcs.distinct().all()

        for (tc,) in tcs:
            if tc >= final_tc:
                self.logger.info(
                    f"Not ready to make timecode books for {tc} because final"
                    f" timecode is {final_tc}"
                )
                continue
            q = g3session.query(TimeCodes).filter(
                TimeCodes.timecode == tc,
            )
            files = g3session.query(TimeCodes.agent).filter(
                TimeCodes.timecode==tc,
                files_query,
                TimeCodes.suprsync_type == SupRsyncType.FILES.value,
            ).distinct().all()
            meta = g3session.query(TimeCodes.agent).filter(
                TimeCodes.timecode==tc,
                meta_query,
                TimeCodes.suprsync_type == SupRsyncType.META.value,
            ).distinct().all()

            if not len(meta) == len(meta_agents):
                # not ready to make any books
                self.logger.debug(f"Not ready to make timecode books for {tc}")
                continue
            
            ## register smurf  books
            book_start = dt.datetime.utcfromtimestamp(tc * 1e5)
            book_stop = dt.datetime.utcfromtimestamp((tc + 1) * 1e5)

            book_id = f"smurf_{tc}_{self.daq_node}"
            if self.get_book(book_id) is None:
                self.logger.info(f"registering {book_id}")
                smurf_book = Books(
                    bid=book_id,
                    type="smurf",
                    status=UNBOUND,
                    tel_tube=self.daq_node,
                    start=book_start,
                    stop=book_stop,
                    schema=1,
                )
                smurf_book.path = self.get_book_path(smurf_book)
                session.add(smurf_book)
                session.commit()

            if not len(files) == len(files_agents):
                #not ready to make stray books
                self.logger.debug(f"Not ready to make stray books for {tc}")
                continue

            book_id = f"stray_{tc}_{self.daq_node}"
            if self.get_book(book_id) is not None:
                # book already registered
                continue

            # look for failed or unbound books
            q = session.query(Books).filter(
                Books.start >= book_start,
                Books.start < book_stop,
                or_(Books.type == 'obs', Books.type == 'oper'),
                or_(Books.status == UNBOUND, Books.status == FAILED), 
            )
            if q.count() > 0:
                self.logger.info(
                    f"Not ready to register {book_id} due to unbound or "
                    "failed obs/oper books."
                )
                continue

            flist = self.get_files_for_stray_book(
                min_ctime= tc * 1e5,
                max_ctime= (tc + 1) * 1e5
            )
            if len(flist) > 0:
                stray_book = Books(
                    bid=book_id,
                    type="stray",
                    status=UNBOUND,
                    tel_tube=self.daq_node,
                    start=book_start,
                    stop=book_stop,
                    schema=0,
                )
                stray_book.path = self.get_book_path(stray_book)            
                self.logger.info(f"registering {book_id}")
                session.add(stray_book)
                session.commit()

    def get_book_abs_path(self, book):
        if book.path is None:
            book_path = self.get_book_path(book)
        else:
            book_path = book.path
        return os.path.join(self.output_root, book_path)

    def get_book_path(self, book):

        if book.type in ["obs", "oper"]:
            session_id = book.bid.split("_")[1]
            first5 = session_id[:5]
            odir = op.join(book.tel_tube, book.type, first5)
            return os.path.join(odir, book.bid)
        elif book.type in ["hk", "smurf"]:
            # get source directory for hk book
            first5 = book.bid.split("_")[1]
            assert first5.isdigit(), f"first5 of {book.bid} is not a digit"
            odir = op.join(book.tel_tube, book.type, book.bid)
            if book.type == 'smurf' and book.schema > 0:
                return odir + '.zip'
            return odir
        elif book.type in ["stray"]:
            first5 = book.bid.split("_")[1]
            assert first5.isdigit(), f"first5 of {book.bid} is not a digit"
            odir = op.join(book.tel_tube, book.type)
            return os.path.join(odir, book.bid)
        else:
            raise NotImplementedError(
                f"book type {book.type} not implemented"
            )

    def _get_binder_for_book(self, 
        book, 
        ignore_tags=False,
        ancil_drop_duplicates=False,
        allow_bad_timing=False,
        require_hwp=True,
        require_acu=True,
        require_monotonic_times=True,
        min_ctime=None, 
        max_ctime=None,
    ):
        """get the appropriate bookbinder for the book based on its type"""

        if book.type in ["obs", "oper"]:
            book_path = self.get_book_abs_path(book)

            # after sanity checks, now we proceed to bind the book.
            # get files associated with this book, in the form of
            # a dictionary of {stream_id: [file_paths]}
            filedb = self.get_files_for_book(book)

            # check if any of the files are too large. This is before
            # readout id checking because that one usually also raises
            # but for these we want to delete level 2 files
            if book.type == "obs":
                file_limit = MAX_OBS_LVL2_SIZE
            else:
                file_limit = MAX_OPER_LVL2_SIZE
            for _, flist in filedb.items():
                sz = np.array([os.path.getsize(f)/1e9 for f in flist])
                if np.any(sz > file_limit):
                    msg = "Files in book are too large:\n"
                    msg += "\n".join( [
                        f"{f},{round(s,2)} GB" for f,s in zip(flist, sz)
                    ])
                    raise FileTooLargeError(msg)

            obsdb = self.get_g3tsmurf_obs_for_book(book)
            readout_ids = self.get_readout_ids_for_book(book)
            hk_fields = self.config.get('hk_fields')

            if hk_fields is None:
                raise ValueError("`hk_fields` entry required for bookbinding.")

            # bind book using bookbinder library
            bookbinder = BookBinder(
                book, obsdb, filedb, self.lvl2_data_root, readout_ids, book_path, hk_fields,
                ignore_tags=ignore_tags,
                ancil_drop_duplicates=ancil_drop_duplicates,
                allow_bad_timing=allow_bad_timing,
                require_hwp=require_hwp,
                require_acu=require_acu,
                require_monotonic_times=require_monotonic_times,
                min_ctime=min_ctime, max_ctime=max_ctime,
            )
            return bookbinder

        elif book.type in ["hk", "smurf"]:
            # get source directory for hk book
            root = op.join(self.lvl2_data_root, book.type)
            timecode = book.bid.split("_")[1]
            assert timecode.isdigit(), f"timecode of {book.bid} is not a digit"
            book_path_src = op.join(root, timecode)

            # get target directory for hk book
            book_path_tgt = self.get_book_abs_path(book)
            odir, _ = op.split(book_path_tgt)
            if not op.exists(odir):
                os.makedirs(odir)
            
            bookbinder = TimeCodeBinder(
                book, timecode, book_path_src, book_path_tgt,
                ignore_pattern=SMURF_EXCLUDE_PATTERNS,
            )
            return bookbinder

        elif book.type in ["stray"]:
            flist = self.get_files_for_book(book)

            # get source directory for stray book
            root = op.join(self.lvl2_data_root, "timestreams")
            timecode = book.bid.split("_")[1]
            assert timecode.isdigit(), f"timecode of {book.bid} is not a digit"
            book_path_src = op.join(root, timecode)

           # get target directory for book
            book_path_tgt = self.get_book_abs_path(book)
            odir, _ = op.split(book_path_tgt)
            if not op.exists(odir):
                os.makedirs(odir)
                    
            bookbinder = TimeCodeBinder(
                book, timecode, book_path_src, book_path_tgt, 
                file_list=flist,
            )
            return bookbinder
        else:
            raise NotImplementedError(
                f"binder for book type {book.type} not implemented"
            )

    def _run_book_binding(
        self,
        book,
        session=None,
        message="",
        pbar=False,
        ignore_tags=False,
        ancil_drop_duplicates=False,
        allow_bad_timing=False,
        require_hwp=True,
        require_acu=True,
        require_monotonic_times=True,
        min_ctime=None, max_ctime=None,
        check_configs={}
    ):
        """Function that calls book-binding but does not update the database 
        this is implemented so be able to run things in parallel
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
        if (book.status == BOUND) :
            raise BookBoundError(f"Book {bid} is already bound")
        assert book.type in VALID_OBSTYPES

        ## LATs don't have HWPs. We can change this if we ever make LAT HWPs :D
        ## or if we ever plan to run the SATs without HWPs
        if 'lat' in self.daq_node:
            require_hwp = False
        err = None
        try:
            # find appropriate binder for the book type
            binder = self._get_binder_for_book(
                book, 
                ignore_tags=ignore_tags,
                ancil_drop_duplicates=ancil_drop_duplicates,
                allow_bad_timing=allow_bad_timing,
                require_acu=require_acu,
                require_hwp=require_hwp,
                require_monotonic_times=require_monotonic_times,
                min_ctime=min_ctime, max_ctime=max_ctime
            )
            binder.bind(pbar=pbar)
            
            # write M_index file
            if book.type in ['obs', 'oper']:
                tc = self.tube_configs[book.tel_tube]
            else:
                tc = {}
            binder.write_M_files(self.daq_node, tc)

            if book.type in ['obs', 'oper']:
                # check that detectors books were written out correctly
                self.logger.info("Checking Book {}".format(book.bid))
                check = BookScanner(
                    self.get_book_abs_path(book), config=check_configs
                )
                check.go()

            self.logger.info("Book {} bound".format(book.bid))
            status = BOUND
            
        except Exception as e:
            self.logger.error("Book {} failed".format(book.bid))
            err_msg = traceback.format_exc()
            self.logger.error(err_msg)
            message = f"{message}\ntrace={err_msg}" if message else err_msg
            status = FAILED
            err = e
        
        return book.bid, status, message, err

    def bind_book(
        self,
        book,
        session=None,
        message="",
        pbar=False,
        ignore_tags=False,
        ancil_drop_duplicates=False,
        allow_bad_timing=False,
        require_hwp=True,
        require_acu=True,
        require_monotonic_times=True,
        min_ctime=None, max_ctime=None,
        check_configs={}
    ):
        """Bind book using bookbinder

        Parameters
        ----------
        bid: str
            book id
        session: BookDB session
        message: string
            message to be added to the book
        test_mode : bool
            If in test_mode, this function will still run on already copied
            books, and will not update any db fields in the db. This is useful
            for testing purposes.
        ignore_tags : bool
            If true, book will be bound even if the tags between different level 2
            operations don't match. Only ever expected to be turned on by hand.
        ancil_drop_duplicates: if true, will drop duplicate data from ancilary  
            files. added to deal with an ocs aggregator error. Only ever 
            expected to be turned on by hand.
        allow_bad_timing: if true, will bind books even if the timing is low 
            precision
        require_hwp: bool, optional
            if True, requires that we find HWP data before binding the book. 
            hard-coded to False if self.daq_node is lat
        require_acu: bool, optional
            if True, requires that we have ACU data and that it has no dropouts 
            longer than 10s
        require_monotonic_times: bool, optional
            if True, requires that all HK data is monotonically increasing, 
            should never be set to False for obs books but less important for 
            oper books if there were ACU aggregation issues
        min_ctime: float, optional
            if not None, cuts the book down to have this minimum ctime
        max_ctime: float, optional
            if not None, cuts the book down to have this maximum ctime
        check_configs: dict
            additional non-default configurations to send to check book
        """
        if session is None:
            session = self.get_session()

        bid, status, message, err = self._run_book_binding(
            book,
            session=session,
            message=message,
            pbar=pbar,
            ignore_tags=ignore_tags,
            ancil_drop_duplicates=ancil_drop_duplicates,
            allow_bad_timing=allow_bad_timing,
            require_hwp=require_hwp,
            require_acu=require_acu,
            require_monotonic_times=require_monotonic_times,
            min_ctime=min_ctime, max_ctime=max_ctime,
            check_configs=check_configs
        )
        book = session.query(Books).filter(Books.bid == bid).one()
        book.status = status
        book.message = message
        session.commit()
        if status == FAILED:
            raise err


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
        return session.query(Books).filter(
            Books.status == status
        ).order_by(Books.start).all()

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
    
    def get_done_books(self, session=None):
        """Get all "done" books from database. Done means staged files are deleted.

        Parameters
        ----------
        session: BookDB session

        Returns
        -------
        books: list of book objects

        """
        return self.get_books_by_status(DONE, session)

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

    @property
    def all_slots(self):
        return [x for xs in [
            t.get('slots') for (_,t) in self.tubes.items()
        ] for x in xs]

    def _find_incomplete(self, min_ctime, max_ctime, streams=None):
        """return G3tSmurf session query for incomplete observations
        """
        if streams is None:
            streams = self.all_slots

        session = self.get_g3tsmurf_session()
        q = session.query(G3tObservations).filter(
            G3tObservations.timestamp >= min_ctime,
            G3tObservations.timestamp <= max_ctime,
            G3tObservations.stream_id.in_(streams),
            or_(
                G3tObservations.stop == None,
                G3tObservations.stop >= dt.datetime.utcfromtimestamp(max_ctime),
            ),
        )
        return q

    @loop_over_tubes
    def update_bookdb_from_g3tsmurf(
        self,
        tube,
        min_ctime=None,
        max_ctime=None,
        min_overlap=30,
        ignore_singles=False,
        stream_ids=None,
        force_single_stream=False,
        return_obsset=False,
        incomplete_timeouts = (3,6),
    ):
        """Update bdb with new observations from g3tsmurf db.

        Parameters
        ----------
        tube: str
            which tube to look at. e.g., 'lati6', 'lati1', 'satp1'
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
        return_obsset: boolean
            if True, return the list of observation sets instead of registering
            books. Useful as a debugging tool
        incomplete_timeouts: tuple
            hours for raising a (warning, error) if incomplete observations are
            found in the g3tsmurf database
        """
        if not self.build_det:
            return
        session, SMURF = self.get_g3tsmurf_session(return_archive=True)
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
            streams = self.tubes[tube].get("slots")
            if streams is None:
                raise ValueError(
                    f"Imprinter missing slot / stream_id" " information for {source}"
                )
        else:
            # if user input is present, format it like the query response
            streams = stream_ids
        self.logger.debug(f"Looking for observations from stream_ids {streams}")

        # check data transfer finalization
        final_time = SMURF.get_final_time(
            streams, min_ctime, max_ctime, check_control=True
        )
        if final_time < max_ctime:
            max_ctime = final_time
        self.logger.debug(f"Searching between {min_ctime} and {max_ctime}")

        # check for incomplete observations in time range
        q_incomplete = self._find_incomplete(min_ctime, max_ctime, streams)

        # if we have incomplete observations in our stream_id list we cannot
        # bookbind any observations overlapping the incomplete ones.
        if q_incomplete.count() > 0:
            new_ctime = min([obs.timestamp for obs in q_incomplete.all()])
            if max_ctime - new_ctime > 3600*incomplete_timeouts[1]:
                raise ValueError(
                    f"Found {q_incomplete.count()} incomplete observations. "
                    f"New max ctime would be {new_ctime}. More than "
                    f"{incomplete_timeouts[1]} hours in the past."
                )
            elif max_ctime - new_ctime > 3600*incomplete_timeouts[0]:
                level = self.logger.warning
            else:
                level = self.logger.debug
            level(
                f"Found {q_incomplete.count()} incomplete observations. "
                f"updating max ctime to {new_ctime}"
            )
            max_ctime = new_ctime

        min_start = dt.datetime.utcfromtimestamp(min_ctime)
        max_stop = dt.datetime.utcfromtimestamp(max_ctime)

        # find observations in time range that are already in books
        already_registered = [ x[0] for x in
            self.get_session().query(Observations.obs_id).join(Books).filter(
            Books.start >= min_start-dt.timedelta(hours=3),
            Books.stop <= max_stop+dt.timedelta(hours=3),
        ).all()]

        # find all complete observations that start within the time range and
        # are not already in books
        obs_q = session.query(G3tObservations).filter(
            G3tObservations.timestamp >= min_ctime,
            G3tObservations.timestamp < max_ctime,
            G3tObservations.stream_id.in_(streams),
            G3tObservations.stop < max_stop,
            not_(G3tObservations.stop == None),
            G3tObservations.obs_id.not_in(already_registered),
        )
        self.logger.debug(
            f"Found {obs_q.count()} level 2 observations to consider"
        )

        output = []
        def add_to_output(obs_list, mode):
            output.append(
                ObsSet( 
                    obs_list,
                    mode=mode,
                    slots=self.tubes[tube]["slots"],
                    tel_tube=tube,
                )
            )

        for stream in streams:
            # loop through all observations for this particular stream_id
            for str_obs in obs_q.filter(G3tObservations.stream_id == stream).all():
                # distinguish different types of books
                # operation book
                if get_obs_type(str_obs) == "oper":
                    add_to_output([str_obs], "oper")
                elif get_obs_type(str_obs) == "obs":
                    # force each observation to be its own book
                    if force_single_stream:
                        add_to_output([str_obs], "obs")
                    else:
                        # query for all possible types of overlapping 
                        # observations from other streams
                        q = obs_q.filter(
                            G3tObservations.stream_id != str_obs.stream_id,
                            G3tObservations.stream_id.in_(streams),
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
                            # append only this one when there's no overlapping 
                            # segments
                            add_to_output([str_obs], "obs")

                        elif q.count() > 0:
                            # obtain overlapping observations (returned as a tuple of stream_id)
                            obs_list = q.all()
                            # add the current obs too
                            obs_list.append(str_obs)

                            # remove overlapping operations
                            obs_list = [obs for obs in obs_list 
                                if get_obs_type(obs) == "obs"
                            ]
                            # check to make sure ALL observations overlap all 
                            # others and the overlap passes the minimum 
                            # requirement
                            overlap_time = (
                                np.min([o.stop for o in obs_list]) - 
                                np.max([o.start for o in obs_list])
                            )

                            if overlap_time.total_seconds() < 0:
                                continue
                            if overlap_time.total_seconds() < min_overlap:
                                continue

                            t_list = [o for o in obs_list if o.timing]
                            nt_list = [o for o in obs_list if not o.timing]
                            
                            if len(t_list)+len(nt_list) != len(obs_list):
                                raise ValueError(f"Cannot safely split timing"  
                                    f"info for {obs_list}"
                                )
                                

                            if len(t_list) > 0:
                                # add all of the possible overlaps
                                add_to_output(t_list, "obs")

                            if len(nt_list) > 0:
                                self.logger.debug(
                                    "registering single wafer books"       
                                    f" for {nt_list} because of low "
                                    "precision timing"
                                )
                                for obs in nt_list:
                                    add_to_output([obs], "obs")

        # remove exact duplicates in output
        output = drop_duplicates(output)
        self.logger.debug(f"Found {len(output)} possible books to register")
        # now our output is a list of ObsSet, where each ObsSet contains
        # all observations that overlap each other which should go
        # into the same book.
        if return_obsset:
            return output

        # look for repeat obs_ids
        obs_ids = []
        [obs_ids.extend( o.obs_ids) for o in output]
        if np.any([obs_ids.count(o) != 1 for o in obs_ids]):
            repeats = np.where( [obs_ids.count(o) != 1 for o in obs_ids])[0]
            raise OverlapObsError(f"Found repeated level 2 obs_ids in book "
                    f"list. {np.unique([obs_ids[x] for x in repeats])} overlap "
                    "multiple observations. Need to choose which books to put " 
                    "them in.")

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
        if book.type in ["obs", "oper"]:
            session = self.get_g3tsmurf_session()
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
            return self.get_files_for_stray_book(book)
        
        elif book.type == "hk":
            HK = self.get_g3thk()
            flist = (
                HK.session.query(HKFiles)
                .filter(
                    HKFiles.global_start_time >= book.start.timestamp(),
                    HKFiles.global_start_time < book.stop.timestamp(),
                )
                .all()
            )
            return [f.path for f in flist]
        elif book.type == "smurf":
            tcode = int(book.bid.split("_")[1])
            basepath = os.path.join(
                self.lvl2_data_root, 'smurf', str(tcode)
            )
            ignore = shutil.ignore_patterns(*SMURF_EXCLUDE_PATTERNS)
            flist = []
            for root, _, files in os.walk(basepath):
                to_ignore = ignore('', files)
                flist.extend([
                    os.path.join(basepath, root, f) 
                    for f in files if f not in to_ignore
                ])
            return flist

        else:
            raise NotImplementedError(
                f"book type {book.type} not understood for file search"
            )

    def get_files_for_stray_book(
        self, book=None, min_ctime=None, max_ctime=None
    ):
        """generate list of files that are not in detector books and should
        going into stray books. if book is None then we expect both min and max
        ctime to be provided

        Arguments
        ----------
        book: optional, book instance
        min_ctime: optional, minimum ctime value to search
        max_ctime: optional, maximum ctime value to search

        Returns
        --------
        list of files that should go into a stray book
        """
        if book is None:
            assert min_ctime is not None and max_ctime is not None
            start = dt.datetime.utcfromtimestamp(min_ctime)
            stop = dt.datetime.utcfromtimestamp(max_ctime)

            tcode = int(min_ctime//1e5)
            if max_ctime > (tcode+1)*1e5:
                self.logger.error(
                    f"Max ctime {max_ctime} is higher than would be expected "
                    f"for a single stray book with min ctime {min_ctime}. only"
                    " checking the first timecode directory"
                )
        else:
            assert book.type == 'stray'
            start = book.start
            stop = book.stop
            tcode = int(book.bid.split("_")[1])
        
        session = self.get_session()
        g3session, SMURF = self.get_g3tsmurf_session(return_archive=True)
        path = os.path.join(SMURF.archive_path, str(tcode))
        registered_obs = [ 
            x[0] for x in session.query(Observations.obs_id).join(Books).filter(
            Books.start >= start, 
            Books.start < stop,
            Books.status != WONT_BIND,
        ).all()]
        db_files = g3session.query(Files).filter(
            Files.name.like(f"{path}%")
        ).all()

        stray_files = []
        for f in db_files:
            if f.obs_id is None or f.obs_id not in registered_obs:
                stray_files.append(f.name)

        return stray_files
            

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
        _, SMURF = self.get_g3tsmurf_session(return_archive=True)
        out = {}
        # load all obs and associated files
        for obs_id, files in self.get_files_for_book(book).items():
            self.logger.debug(f"Retrieving readout_ids for {obs_id}...")
            status = SmurfStatus.from_file(files[0])
            if not np.any(status.mask):
                raise MissingReadoutIDError(
                    f"Readout IDs not found for {obs_id}. SMuRF system was not "
                    "set up for science observations."
                )
            ch_info = get_channel_info(status, archive=SMURF)
            if "readout_id" not in ch_info:
                raise MissingReadoutIDError(
                    f"Readout IDs not found for {obs_id}. Indicates issue with G3tSmurf Indexing"
                )
            checks = ["NONE" in rid for rid in ch_info.readout_id]
            if np.all(checks):
                raise MissingReadoutIDError(
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

        session, arc = self.get_g3tsmurf_session(
            book.tel_tube, return_archive=True
        )

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
            elif f.endswith("bias_wave_analysis.npy"):
                meta_files["bias_wave"] = basename

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
        session = self.get_g3tsmurf_session()
        obs_ids = [o.obs_id for o in book.obs]
        obs = (
            session.query(G3tObservations)
            .filter(G3tObservations.obs_id.in_(obs_ids))
            .all()
        )
        return {o.obs_id: o for o in obs}

    def _librarian_connect(self):
        """
        start connection to librarian
        """
        from hera_librarian import LibrarianClient
        from hera_librarian.settings import client_settings
        conn = client_settings.connections.get(
            self.config.get("librarian_conn")
        )
        if conn is None:
            raise ValueError(f"'librarian_conn' not in imprinter config")
        self.librarian = LibrarianClient.from_info(conn)

    def upload_book_to_librarian(self, book, session=None, raise_on_error=True):
        """Upload bound book to the librarian

        Parameters
        ----------
        book: Book object
        session: imprinter sqlalchemy session
            session that made book
        raise_on_error: bool
            raise an error if the librarian throws an error on the upload
        """

        if session is None:
            session = self.get_session()
        if self.librarian is None:
            self._librarian_connect()
        
        assert book.status == BOUND, "cannot upload unbound books"

        self.logger.info(f"Uploading book {book.bid} to librarian")
        try:     
            self.librarian.upload(
                Path( self.get_book_abs_path(book) ), 
                Path( book.path ), 
            )
            book.status = UPLOADED
            session.commit()
        except Exception as e:
            self.logger.error(
                f"Failed to upload book {book.bid}."
            )
            if raise_on_error:
                raise e
            else:
                return False, e
        return True, None
            
    def check_book_in_librarian(
        self, 
        book, 
        n_copies=1, 
        n_tries=1,
        raise_on_error=True
    ):
        """have the librarian validate the books is stored offsite. returns true
        if at least n_copies are storied offsite.
        """
        if self.librarian is None:
            self._librarian_connect()
        try:
            resp = self.librarian.validate_file(book.path)
            in_lib = sum(
                [(x.computed_same_checksum) for x in resp]
            ) >= n_copies
            if not in_lib:
                self.logger.info(f"received response from librarian {resp}")
                if n_tries > 1:
                    if book.type == 'smurf':
                        wait=30
                    else: 
                        wait=5
                    self.logger.warning(
                        f"Waiting {wait} seconds and trying book {book.bid} "
                        "with the librarian again"
                    )
                    time.sleep(wait)
                    return self.check_book_in_librarian(
                        book, n_copies=n_copies, 
                        n_tries=n_tries-1, 
                        raise_on_error=raise_on_error
                    )
        except Exception as e:
            if raise_on_error:
                raise e
            else: 
                self.logger.warning(
                    f"Failed to check libraian status for {book.bid}: {e}"
                )
                self.logger.warning(traceback.format_exc())
                in_lib = False
        return in_lib
        
    def delete_level2_files(self, book, verify_with_librarian=True,        
        n_copies_in_lib=2, n_tries=1, dry_run=True):
        """Delete level 2 data from already bound books

        Parameters
        ----------
        book: book object
        dry_run: bool
            if true, just prints plans to self.logger.info
        """
        if book.lvl2_deleted:
            self.logger.debug(
                f"Level 2 for {book.bid} has already been deleted"
            )
            return 0
        if book.status < UPLOADED:
            self.logger.warning(
                f"Book {book.bid} is not uploaded, not deleting level 2"
            )
            return 1
        if verify_with_librarian:
            in_lib = self.check_book_in_librarian(
                book, n_copies=n_copies_in_lib, n_tries=n_tries,
                raise_on_error=False
            )
            if not in_lib:
                self.logger.warning(
                    f"Book {book.bid} does not have {n_copies_in_lib} copies"
                    " will not delete level 2"
                )
                return 2
        
        self.logger.info(f"Removing level 2 files for {book.bid}")
        if book.type == "obs" or book.type == "oper":
            session, SMURF = self.get_g3tsmurf_session(
                return_archive=True
            )
            odic = self.get_g3tsmurf_obs_for_book(book)

            for oid, obs in odic.items():
                SMURF.delete_observation_files(
                    obs, session, dry_run=dry_run, my_logger=self.logger
                )
        elif book.type == "stray":
            session, SMURF = self.get_g3tsmurf_session(
                return_archive=True
            )
            flist = self.get_files_for_book(book)
            for f in flist:
                db_file = session.query(Files).filter(Files.name == f).one()
                SMURF.delete_file(
                    db_file, session, dry_run=dry_run, my_logger=self.logger
                )
        elif book.type == "smurf":
            tcode = int(book.bid.split("_")[1])
            basepath = os.path.join(
                self.lvl2_data_root, 'smurf', str(tcode)
            )
            if not dry_run:
                shutil.rmtree(basepath)

        elif book.type == "hk":
            HK = self.get_g3thk()
            flist = self.get_files_for_book(book)
            hkf_list = [
                HK.session.query(HKFiles).filter(
                    HKFiles.path == f
                ).one() for f in flist
            ]
            HK.batch_delete_files(
                hkf_list, dry_run=dry_run, my_logger=self.logger
            )
        else:
            raise NotImplementedError(
                f"Do not know how to delete level 2 files"
                f" for book of type {book.type}"
            )
        if not dry_run:
            book.lvl2_deleted = True
            self.session.commit()
        return 0

    def delete_book_staged(self, book, check_level2=False, 
        verify_with_librarian=False, n_copies_in_lib=1, override=False):
        """Delete all files associated with a book

        Parameters
        ----------
        book: Book object

        """
        if book.status == DONE:
            self.logger.debug(
                f"Book {book.bid} has already had staged files deleted"
            )
            return 0
        if not override:
            if book.status < UPLOADED:
                self.logger.warning(
                    "Cannot delete non-uploaded books without override"
                )
                return 1
        if check_level2 and not book.lvl2_deleted:
            self.logger.warning(
                f"Level 2 data not deleted for {book.bid}, not deleting "
                "staged"
            )
            return 2
        if verify_with_librarian:
            in_lib = self.check_book_in_librarian(
                book, n_copies=n_copies_in_lib, raise_on_error=False
            )
            if not in_lib:
                self.logger.warning(
                    f"Book {book.bid} does not have {n_copies_in_lib} copies"
                    " will not delete staged"
                )
                return 3

        # remove all files within the book
        book_path = self.get_book_abs_path(book)
        try:
            self.logger.info(
                f"Removing {book.bid} from staged"
            )
            if book.type == 'smurf' and book.schema == 1:
                os.remove(book_path)
            else:
                shutil.rmtree( book_path )
        except Exception as e:
            self.logger.warning(f"Failed to remove {book_path}: {e}")
            self.logger.error(traceback.format_exc())
            return 4
        book.status = DONE
        self.session.commit()
        return 0

    def find_missing_lvl2_obs_from_books(
        self, min_ctime, max_ctime
    ):
        """create a list of level 2 observation IDs that are not registered in 
        the imprinter database
        
        Arguments
        ----------
        min_ctime: minimum ctime value to search
        max_ctime: maximum ctime value to search

        Returns
        --------
        list of level 2 observation ids not in books
        """
        session = self.get_session()
        g3session, SMURF = self.get_g3tsmurf_session(return_archive=True)
        registered_obs = [
            x[0] for x in session.query(Observations.obs_id).join(Books).filter(
            Books.start >= dt.datetime.utcfromtimestamp(min_ctime), 
            Books.start < dt.datetime.utcfromtimestamp(max_ctime),
        ).all()]
        missing_obs = g3session.query(G3tObservations).filter(
            G3tObservations.timestamp >= min_ctime,
            G3tObservations.timestamp < max_ctime,
            G3tObservations.stream_id.in_(self.all_slots),
            G3tObservations.obs_id.not_in(registered_obs)
        ).all()
        return missing_obs

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
    config = load_configs(config)
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
