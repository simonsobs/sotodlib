import os
import yaml
import logging
import shutil
import numpy as np
import datetime as dt
from sqlalchemy import or_, and_, not_
from collections import OrderedDict

from .load_smurf import (
    TimeCodes,
    SupRsyncType,
    Finalize,
    SmurfStatus,
    logger as smurf_log
)
from .imprinter import (
    Books,
    Imprinter,
    BOUND,
    UNBOUND,
    UPLOADED,
    FAILED,
    WONT_BIND,
    DONE,
    SMURF_EXCLUDE_PATTERNS,
)
import sotodlib.io.imprinter_utils as utils
from .imprinter_cli import autofix_failed_books
from .datapkg_utils import walk_files, just_suprsync

from .bookbinder import log as book_logger

def combine_loggers(imprint, fname=None):
    log_list = [imprint.logger, smurf_log, book_logger]
    logger = logging.getLogger("DataPackaging")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    # Create a file handler
    if fname is not None:
        handler = logging.FileHandler(fname)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        [l.addHandler(handler) for l in log_list]  

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

class DataPackaging:
    def __init__(self, platform, log_filename=None):
        self.platform = platform
        self.imprint = Imprinter.for_platform(platform)
        self.logger = combine_loggers(self.imprint, fname=log_filename)
        self.session = self.imprint.get_session()
        if self.imprint.build_det:
            self.g3session, self.SMURF = self.imprint.get_g3tsmurf_session(return_archive=True)
        else: 
            self.g3session = None
            self.SMURF = None
        self.HK = self.imprint.get_g3thk()

    def get_first_timecode_on_disk(self, include_hk=True):
        tc = 50000
        if self.imprint.build_det:
            tc = min([
                tc,
                int(sorted(os.listdir(self.SMURF.meta_path))[0]),
                int(sorted(os.listdir(self.SMURF.archive_path))[0]),
            ])
        if include_hk:
            tc = min([
                tc,
                int(sorted(os.listdir(self.HK.hkarchive_path))[0])
            ])
        if tc == 50000:
            raise ValueError(f"Found no timecode folders for {self.platform}")
        return tc   

    def get_first_timecode_in_staged(self, include_hk=True):
        q = self.session.query(Books).filter(
            Books.status == UPLOADED,
        )
        if not include_hk:
            q = q.filter(Books.type != 'hk')
        first = q.order_by(Books.start).first()
        tc = int( first.start.timestamp() // 1e5)
        return tc   

    def all_files_in_timecode(self, timecode, include_hk=True):
        flist = []
        if self.imprint.build_det:
            stc = os.path.join(self.SMURF.meta_path, str(timecode))
            flist.extend(walk_files(stc, include_suprsync=True))
            ttc = os.path.join(self.SMURF.archive_path, str(timecode))
            flist.extend(walk_files(ttc, include_suprsync=True))
        if include_hk:
            htc = os.path.join(self.HK.hkarchive_path, str(timecode))
            flist.extend(walk_files(htc, include_suprsync=True))
        return flist

    def get_suprsync_files(self, timecode):
        if not self.imprint.build_det:
            return []
        stc = os.path.join(self.SMURF.meta_path, str(timecode))
        ttc = os.path.join(self.SMURF.archive_path, str(timecode))
        flist = []

        if not os.path.exists(stc) and not os.path.exists(ttc):
            return flist
        if os.path.exists(ttc) and 'suprsync' in os.listdir(ttc):
            for root, _, files in os.walk(os.path.join(ttc, 'suprsync')):
                for name in files:
                    flist.append(os.path.join(ttc, root, name))
        if os.path.exists(stc) and 'suprsync' in os.listdir(stc):
            for root, _, files in os.walk(os.path.join(stc, 'suprsync')):
                for name in files:
                    flist.append(os.path.join(stc, root, name))
        return flist

    def check_hk_registered(self, timecode, complete):
        min_ctime = timecode*1e5
        max_ctime = (timecode+1)*1e5

        self.HK.add_hkfiles(
            min_ctime=min_ctime, max_ctime=max_ctime, 
            show_pb=False, update_last_file=False,
        )
        self.imprint.register_hk_books(
            min_ctime=min_ctime, 
            max_ctime=max_ctime, 
        )
        # check the hk book is registered
        book = self.session.query(Books).filter( 
            Books.bid == f"hk_{timecode}_{self.platform}"
        ).one_or_none()
        if book is None:
            complete[0] = False
            complete[1] += f"HK book hk_{timecode}_{self.platform} missing\n"
        elif book.status == UNBOUND:
            try:
                self.imprint.bind_book(book)
            except:
                self.logger.warning(f"Failed to bind {book.bid}")
            if book.status < BOUND:
                complete[0] = False
                complete[1] += f"Book hk_{timecode}_{self.platform} not bound"
        return complete

    def make_timecode_complete(
        self, timecode, try_binding_books=True, try_single_obs=True,
        include_hk=True,
    ):
        """
        Carefully go through an entire timecode and check that the data packaging as
        complete as it can be. The verification will also try and fix any errors
        found in the system. Updating databases, registering books, and binding
        books if try_binding_books is True

        Arguments
        ----------
        timecode: int
            5-digit ctime to check for completion
        try_binding_books: bool
            if true, go through and try to bind any newly registered books
        try_single_obs: bool
            if true, tries to register any missing observations as single wafer
            observations if registering as multi-wafer observations fails. This
            happens sometimes if the stream lengths are very close to the
            minimum overlap time
        include_hk: bool
            if true, also checkes everything related to hk 
        """
        
        complete = [True, ""]
        min_ctime = timecode*1e5
        max_ctime = (timecode+1)*1e5

        if not self.imprint.build_det:
            ## no detector data tracked by imprinter
            if include_hk:
                return self.check_hk_registered(timecode, complete)
            else:
                self.logger.warning(
                    f"No detector data built for platform "
                    f"{self.imprint.daq_node} and not checking HK. Nothing to "
                    "check for completion"
                )
                return complete

        has_smurf, has_timestreams = True, True
        stc = os.path.join(self.SMURF.meta_path, str(timecode))
        ttc = os.path.join(self.SMURF.archive_path, str(timecode))

        if not os.path.exists(stc):
            self.logger.debug(f"TC {timecode}: No level 2 smurf folder")
            has_smurf = False
        if not os.path.exists(ttc):
            self.logger.debug(f"TC {timecode}: No level 2 timestream folder")
            has_timestreams = False

        if os.path.exists(ttc) and just_suprsync(ttc):
            self.logger.info(
                f"TC {timecode}: Level 2 timestreams is only suprsync"
            )
            has_timestreams = False
        if os.path.exists(stc) and just_suprsync(stc):
            self.logger.info(f"TC {timecode}: Level 2 smurf is only suprsync")
            has_smurf = False
        
        if not has_smurf and not has_timestreams:
            return complete
        if not has_smurf and has_timestreams:
            self.logger.error(f"TC {timecode}: Has timestreams folder without smurf!")        
        
        overall_final_ctime = self.SMURF.get_final_time(
            self.imprint.all_slots, check_control=False
        )
        tcode_limit = int(overall_final_ctime//1e5)
        if timecode+1 > tcode_limit:
            raise ValueError(
                f"We cannot check files from {timecode} because finalization time "
                f"is {overall_final_ctime}"
            )

        self.logger.info(f"Checking Timecode {timecode} for completion")
        ## check for files on disk to be in database
        missing_files = self.SMURF.find_missing_files(
            timecode, session=self.g3session
        )
        if len(missing_files) > 0:
            self.logger.warning(
                f"{len(missing_files)} files not in G3tSmurf"
            )
            self.SMURF.index_metadata(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
                session=self.g3session
            )
            self.SMURF.index_archive(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
                show_pb=False, 
                session=self.g3session
            )
            self.SMURF.index_timecodes(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
                session=self.g3session
            )
            still_missing = len(
                self.SMURF.find_missing_files(timecode, session=self.g3session)
            )
            if still_missing>0:
                msg = f"{still_missing} file(s) were not able to be added to the " \
                    "G3tSmurf database."
                self.logger.error(msg)            
                complete[0] = False
                complete[1] += msg+"\n"
        else:
            self.logger.debug("All files on disk are in G3tSmurf database")

        ## check for level 2 files to be assigned to level 2 observations
        missing_obs = self.SMURF.find_missing_files_from_obs(
            timecode, session=self.g3session
        )
        if len(missing_obs) > 0:
            msg = f"{len(missing_obs)} files not assigned lvl2 obs"
            no_tags = 0
            for fpath in missing_obs:
                if fpath[-6:] != "000.g3":
                    msg += f"\n{fpath} was not added to a larger observation." \
                        " Will be fixed later if possible."
                else:
                    status = SmurfStatus.from_file(fpath)
                    if len(status.tags)==0:
                        no_tags += 1
                    else:
                        msg += f"\Trying to add {fpath} to database"
                        self.SMURF.add_file( 
                            fpath, self.g3session, overwrite=True
                        )
            if no_tags > 0:
                msg += f"\n{no_tags} of the files have no tags, so these should "\
                    "not be observations."
            self.logger.warning(msg)
        
        ## if the stray book has already been bound then we cannot add
        ## more detector books without causing problems
        add_new_detector_books = True
        stray_book = self.session.query(Books).filter( 
            Books.bid == f"stray_{timecode}_{self.platform}",
            Books.status >= BOUND, 
        ).one_or_none()
        if stray_book is not None:
            add_new_detector_books = False

        ## check for incomplete observations
        ## add time to max_ctime to account for observations on the edge
        incomplete = self.imprint._find_incomplete( 
            min_ctime, max_ctime+24*2*3600
        )
        if incomplete.count() > 0:
            ic_list = incomplete.all()
            """Check if these are actually incomplete, imprinter incomplete checker
            includes making sure the stop isn't beyond max ctime. 
            """
            obs_list = []
            for obs in ic_list:
                if obs.stop is None or obs.timestamp <= max_ctime:
                    obs_list.append(obs)
                    
            ## complete these no matter what for file tracking / deletion 
            self.logger.warning(
                f"Found {len(obs_list)} incomplete observations. Fixing"
            )
            for obs in obs_list:
                self.logger.debug(f"Updating {obs}")
                self.SMURF.update_observation_files(
                    obs, 
                    self.g3session, 
                    force=True,
                )

        ## make sure all obs / operation books from this period are registered
        ## looks like short but overlapping observations are sometimes missed, 
        ## use `try_single_obs` flag to say if we want to try and clean those up
        missing = self.imprint.find_missing_lvl2_obs_from_books(
            min_ctime,max_ctime
        ) 
        if add_new_detector_books and len(missing) > 0:
            self.logger.info(
                f"{len(missing)} lvl2 observations are not registered in books."
                " Trying to register them"
            )
            ## add time to max_ctime to account for observations on the edge
            self.imprint.update_bookdb_from_g3tsmurf(
                min_ctime=min_ctime, max_ctime=max_ctime+24*2*3600,
            )
            still_missing = self.imprint.find_missing_lvl2_obs_from_books(
                min_ctime,max_ctime
            ) 
            if len(still_missing) > 0 and try_single_obs:
                self.logger.warning("Trying single stream registration")
                self.imprint.update_bookdb_from_g3tsmurf(
                    min_ctime=min_ctime, max_ctime=max_ctime+24*2*3600,
                    force_single_stream=True,
                )   
            still_missing = self.imprint.find_missing_lvl2_obs_from_books(
                min_ctime,max_ctime
            ) 
            if len(still_missing) > 0:
                msg = f"Level 2 observations {still_missing} could not be " \
                    "registered in books"
                self.logger.error(msg)
                complete[0] = False
                complete[1] += msg+"\n"
        elif not add_new_detector_books and len(missing)>0:
            msg = f"Have level 2 observations missing but cannot add new " \
                f"detector books because {timecode} was already finalized " \
                " and stray exists. These files should be in stray"
            self.logger.warning(msg)
        
        ## at this point, if an obs or oper book is going to be registered it is
        if try_binding_books:
            books = self.session.query(Books).filter(
                Books.status == UNBOUND,
                Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
                Books.start <= dt.datetime.utcfromtimestamp(max_ctime),
            ).all()
            self.logger.info(f"{len(books)} new books to bind")
            for book in books:
                try:
                    self.imprint.bind_book(book)
                except:
                    self.logger.warning(f"Failed to bind {book.bid}")
        
            failed = self.session.query(Books).filter(
                Books.status == FAILED,
                Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
                Books.start <= dt.datetime.utcfromtimestamp(max_ctime),
            ).all()
            if len(failed) > 0:
                self.logger.info(
                    f"{len(failed)} books failed to bind. trying to autofix"
                )
                autofix_failed_books(
                    self.imprint,
                    min_ctime=min_ctime,
                    max_ctime=max_ctime,
                )
        
        is_final, reason = utils.get_timecode_final(self.imprint, timecode)
        if not is_final:
            self.logger.info(
                f"Timecode {timecode} not counted as final: reason {reason}"
            )
            meta_entries = self.g3session.query(TimeCodes).filter(
                TimeCodes.timecode == timecode,
                TimeCodes.suprsync_type == SupRsyncType.META.value,
            ).count()
            file_entries = self.g3session.query(TimeCodes).filter(
                TimeCodes.timecode == timecode,
                TimeCodes.suprsync_type == SupRsyncType.FILES.value,
            ).count()
            if (
                meta_entries == len(self.imprint.all_slots) and 
                file_entries == len(self.imprint.all_slots)
            ):
                self.logger.info(
                    f"{timecode} was part of the mixed up timecode agent entries"
                )
            elif timecode < tcode_limit:
                self.logger.info(
                    f"At least one server was likely off during timecode {timecode}"
                )
            self.logger.info(
                f"Setting timecode {timecode} to final in SMuRF database"
            )
            utils.set_timecode_final(self.imprint, timecode)

        self.imprint.register_timecode_books(
            min_ctime=min_ctime, 
            max_ctime=max_ctime, 
        )

        if try_binding_books:
            books = self.session.query(Books).filter(
                Books.status == UNBOUND,
                Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
                Books.start <= dt.datetime.utcfromtimestamp(max_ctime),
            ).all()
            self.logger.info(f"{len(books)} new to bind")
            for book in books:
                try:
                    self.imprint.bind_book(book)
                except:
                    self.logger.warning(f"Failed to bind {book.bid}")
        
        # check the smurf book is registered
        book = self.session.query(Books).filter( 
            Books.bid == f"smurf_{timecode}_{self.platform}"
        ).one_or_none()
        if book is None:
            complete[0] = False
            complete[1] += f"SMuRF book smurf_{timecode}_{self.platform} missing\n"
        if include_hk:
            complete = self.check_hk_registered(timecode, complete)
        
        # check if there's a stray book
        stray = self.session.query(Books).filter( 
            Books.bid == f"stray_{timecode}_{self.platform}"
        ).one_or_none()
        if stray is None and try_binding_books:
            # all files should be in obs/oper books
            flist = self.imprint.get_files_for_stray_book(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
            )
            if len(flist) > 0:
                complete[0] = False
                complete[1] += f"Stray book stray_{timecode}_{self.platform} missing\n"
        elif stray is None and not try_binding_books:
            my_list = self.imprint.get_files_for_stray_book(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
            )
            if len(my_list) > 0:
                self.logger.warning(
                    f"We expect {len(my_list)} books in a stray book but need "
                    "to bind books to verify"
                )
                complete[0] = False
                complete[1] += f"Stray book stray_{timecode}_{self.platform} missing\n"
        else:
            flist = self.imprint.get_files_for_book(stray)
            my_list = self.imprint.get_files_for_stray_book(
                min_ctime=min_ctime, 
                max_ctime=max_ctime, 
            )
            assert np.all(
                sorted(flist) == sorted(my_list)
            ), "logic error somewhere"
        ## check that all books are bound
        books = self.session.query(Books).filter(
            or_(Books.status == UNBOUND, Books.status == FAILED),
            Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
            Books.start <= dt.datetime.utcfromtimestamp(max_ctime),
        ).count() 
        if books != 0:
            complete[0] = False
            complete[1] += f"Have {books} unbound or failed books in timecode \n"
        return complete

    def books_in_timecode(
        self, timecode, include_wont_fix=False, include_hk=True
    ):
        min_ctime = timecode*1e5
        max_ctime = (timecode+1)*1e5

        q = self.session.query(Books).filter(
            Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
            Books.start < dt.datetime.utcfromtimestamp(max_ctime),
        )
        if not include_wont_fix:
            q = q.filter(Books.status != WONT_BIND)
        if not include_hk:
            q = q.filter(Books.type != 'hk')
        return q.all()

    def file_list_from_database(
        self, timecode, deletable, verify_with_librarian, include_hk=True,
    ):
        file_list = []
        min_ctime = timecode*1e5
        max_ctime = (timecode+1)*1e5

        q = self.session.query(Books).filter(
            Books.start >= dt.datetime.utcfromtimestamp(min_ctime),
            Books.start < dt.datetime.utcfromtimestamp(max_ctime),
        )
        not_ready = q.filter( not_(or_( 
            Books.status == WONT_BIND, Books.status >= UPLOADED)
        )).count()
        if not_ready > 0:
            self.logger.error(
                f"There are {not_ready} non-uploaded books in this timecode"
            )
            deletable[0] = False
            deletable[1] += f"There are {not_ready} non-uploaded books in " \
                "this timecode\n"
        if not include_hk:
            q = q.filter(Books.type != 'hk')
        book_list = q.filter(Books.status >= UPLOADED).all()
        self.logger.debug(
            f"Found {len(book_list)} books in time code {timecode}"
        )

        for book in book_list:
            if book.lvl2_deleted:
                continue
            if verify_with_librarian:
                in_lib = self.imprint.check_book_in_librarian(
                    book, n_copies=1, raise_on_error=False
                )
                if not in_lib:
                    deletable[0] = False
                    deletable[1] += f"{book.bid} has not been uploaded to librarain\n"

            flist = self.imprint.get_files_for_book(book)
            if isinstance(flist, OrderedDict):
                x = []
                for k in flist:
                    x.extend(flist[k])
                flist=x
            file_list.extend(flist)
        # add suprsync files 
        file_list.extend( self.get_suprsync_files(timecode) )
        return file_list, deletable

    def verify_timecode_deletable(
        self, timecode, verify_with_librarian=True, include_hk=True,
    ):
        """
        Checkes that all books in that timecode are uploaded to the librarian 
        and that there is a copy offsite (if verify_with_librarian=True)

        Steps for checking:

        1. Walk the file system and build up a list of all files there
        2. Go book by book within timecode and build up the list of level 2 
        files that went into it using the databases. Add any files in suprsync 
        folders into this list since they aren't book bound but we'd like them 
        to be deleted 
        3. Compare the two lists and make sure they're the same.
        """
        deletable = [True, ""]

        files_on_disk = self.all_files_in_timecode(
            timecode, include_hk=include_hk
        )
        if len(files_on_disk) == 0:
            return deletable
        # these are files that are in the smurf directory but we don't save in the 
        # smurf books. mostly watching out for .dat files
        ignore = shutil.ignore_patterns(*SMURF_EXCLUDE_PATTERNS)
        ignored_files = ignore("", files_on_disk)
        self.logger.debug(
            f"Timecode {timecode} has {len(ignored_files)} ignored files"
        )
        files_in_database, deletable = self.file_list_from_database(
            timecode, deletable, verify_with_librarian, include_hk=include_hk
        )

        missed_files = []
        extra_files = []
        for f in files_on_disk:
            if f not in files_in_database and f not in ignored_files:
                missed_files.append(f)
        for f in files_in_database:
            if f not in files_on_disk:
                extra_files.append(f)
        if len(missed_files) == 0 and len(extra_files) == 0:
            self.logger.info(f"Timecode {timecode} has complete coverage")
        if len(missed_files)>0:
            msg = f"Files on disk but not in database {len(missed_files)}:\n"
            for f in missed_files:
                msg += f"\t{f}\n"
            self.logger.warning(msg)
            deletable[0] = False
            deletable[1] += msg
        if len(extra_files)>0:
            msg = f"Files in database but not on disk: {extra_files}"
            for f in missed_files:
                msg += f"\t{f}\n"
            self.logger.error(msg)
            deletable[0] = False
            deletable[1] += msg
        return deletable

    def delete_timecode_level2(
        self, timecode, dry_run=True, include_hk=True, 
        verify_with_librarian=True,
    ):
        book_list = self.books_in_timecode(timecode, include_hk=include_hk)
        books_not_deleted = []

        for book in book_list:
            stat = self.imprint.delete_level2_files(
                book, verify_with_librarian=verify_with_librarian,
                n_copies_in_lib=2, dry_run=dry_run
            )
            if stat > 0:
                books_not_deleted.append(book)    
        
        if len(books_not_deleted) > 0:
            msg = "Could not delete level 2 for books:\n"
            for book in books_not_deleted:
                msg += f'\t{book.bid}\n'   
            self.logger.error(msg)
            return False, ""
        return True, ""

    
    def delete_timecode_staged(
        self, timecode, include_hk=True, verify_with_librarian=False,
        check_level2=False,
    ):
        book_list = self.books_in_timecode(timecode, include_hk=include_hk)
        books_not_deleted = []
        for book in book_list:
            stat = self.imprint.delete_book_staged(
                book, check_level2=check_level2, 
                verify_with_librarian=verify_with_librarian
            )
            if stat > 0:
                books_not_deleted.append(book)        
        # cleanup
        for tube in self.imprint.tubes:
            for btype in ['obs', 'oper']:
                path = os.path.join(
                    self.imprint.output_root, tube, btype, str(timecode)
                )
                if os.path.exists(path) and len(os.listdir(path))==0:
                    os.rmdir(path)
        
        if len(books_not_deleted) > 0:
            msg = "Could not delete stages for books:\n"
            for book in books_not_deleted:
                msg += f'\t{book.bid}\n'   
            self.logger.error(msg)
            return False, "msg"
        return True, ""
    
    def check_and_delete_timecode(
        self, timecode, include_hk=True, verify_with_librarian=True
    ):
        check = self.make_timecode_complete(timecode, include_hk=include_hk)
        if not check[0]:
            self.logger.error(f"Timecode {timecode} not complete")
            self.logger.error(check[1])
            return check
        check = self.verify_timecode_deletable(
            timecode, include_hk=include_hk, 
            verify_with_librarian=False,
        )
        if not check[0]:
            self.logger.error(f"Timecode {timecode} not ready to delete")
            self.logger.error(check[1])
            return check
        
        check = self.delete_timecode_level2(
            timecode, dry_run=False, include_hk=include_hk,
            verify_with_librarian=verify_with_librarian,
        )

        if not self.imprint.build_det:
            return check
        stc = os.path.join(self.SMURF.meta_path, str(timecode))
        ttc = os.path.join(self.SMURF.archive_path, str(timecode))

        if os.path.exists(stc): 
            if len(os.listdir(stc)) == 0 or just_suprsync(stc):
                shutil.rmtree(stc)
        if os.path.exists(ttc):
            if len(os.listdir(ttc)) == 0 or just_suprsync(ttc):
                shutil.rmtree(ttc)
        return check
