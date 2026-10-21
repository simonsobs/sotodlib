"""
Tools for setting up the JobDB.

TODO: Add some quick tools for inspecting the jobdb here.
"""

import logging
import os
import shutil
import sqlite3
import sys
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, cast

import sqlalchemy as sqy
from sotodlib.site_pipeline import jobdb
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

LoggerLike = logging.Logger | logging.LoggerAdapter[logging.Logger]

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

# try:
#     from mpi4py import MPI
#
#     comm = MPI.COMM_WORLD
# except ImportError:
#     comm = None


def set_tag(job: jobdb.Job, key: str, new_val: Any):
    """
    Set the value of an existing tag on a job.

    Parameters
    ----------
    job : jobdb.Job
        The job whose tag should be updated.
    key : str
        The key identifying the tag to update.
    new_val : Any
        The new value to assign to the tag.

    Raises
    ------
    ValueError
        If the job does not contain a tag with the specified key.
    """
    for tag in job._tags:
        if tag.key == key:
            tag.value = new_val
            return

    raise ValueError(f'No tag called "{key}"')


def fail(job: jobdb.Job, errcode: Enum, msg: str, logger: Optional[LoggerLike] = None):
    """
    Mark a job as failed and record its error information.

    Parameters
    ----------
    job : jobdb.Job
        Job to mark as failed.
    errcode : Enum
        Enum member representing the error code.
    msg : str
        Detailed error message.
    logger : Optional[LoggerLike]
        Optional logger for recording the error.
    """
    if logger is not None:
        logger.error("%s (Err %s: %s)", msg, errcode.value, errcode.name)
    set_tag(job, "message", msg)
    if "errcode" in job.tags:
        set_tag(job, "errcode", errcode.value)
    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)


def sync_jobs(db_a_path: str, db_b_path: str, classes: Optional[Sequence[str]] = None):
    """
    Copy selected jobs and their tags from one SQLite database to another.

    Jobs are selected from the source database, optionally filtered by
    job class. A job is copied if it does not exist in the destination or
    if its creation_time matches the existing destination record.
    This check on creation_time is just a course check that its actually
    the oritinal job.

    Use this function with care! Its mostly for some specific edge cases
    and should not be used unsupervised on databases that are not backed up!

    Parameters
    ----------
    db_a_path : str
        Path to the source database.
    db_b_path : str
        Path to the destination database.
    classes : Optional[Sequence[str]]
        Job classes to select. If None, all classes are selected.
        An empty sequence selects no jobs.

    Raises
    ------
    sqlite3.Error
        If a database operation fails. Destination changes are rolled back.
    """
    conn_a = sqlite3.connect(db_a_path)
    conn_b = sqlite3.connect(db_b_path)

    try:
        if classes is not None and len(classes) == 0:
            return
        where_clause = ""
        params: Sequence[str] = ()
        if classes is not None:
            placeholders = ", ".join("?" for _ in classes)
            where_clause = f"WHERE jclass IN ({placeholders})"
            params = classes
        jobs = conn_a.execute(
            f"""
            SELECT
                id,
                jclass,
                jstate,
                lock,
                lock_owner,
                creation_time,
                visit_time,
                visit_count
            FROM jobs
            {where_clause}
            """,
            params,
        ).fetchall()

        for job in jobs:
            job_id, _, _, _, _, creation_time, _, _ = job
            existing = conn_b.execute(
                "SELECT creation_time FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

            should_copy = existing is None or creation_time == existing[0]
            if not should_copy:
                continue

            conn_b.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    id, jclass, jstate, lock, lock_owner,
                    creation_time, visit_time, visit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                job,
            )

            conn_b.execute(
                "DELETE FROM tags WHERE job_id = ?",
                (job_id,),
            )

            tags = conn_a.execute(
                'SELECT "key", value FROM tags WHERE job_id = ?',
                (job_id,),
            ).fetchall()

            conn_b.executemany(
                'INSERT INTO tags (job_id, "key", value) VALUES (?, ?, ?)',
                [(job_id, key, value) for key, value in tags],
            )

        conn_b.commit()
    except Exception:
        conn_b.rollback()
        raise
    finally:
        conn_a.close()
        conn_b.close()


def make_jobdb(
    comm: Optional["Comm"],
    path: str,
    merge_from: str = "",
    merge_classes: Optional[Sequence[str]] = None,
) -> jobdb.JobManager:
    """
    Open or initialize a job database.

    Rank 0 initializes the database and clears existing job locks.
    When merge_from identifies an existing database, its jobs are
    copied into the target database if the target exists, or the
    source database is copied if the target does not exist.

    Other MPI ranks wait for initialization to finish before opening
    the database.

    Parameters
    ----------
    comm : Optional[Comm]
        MPI communicator, or None when MPI is not being used.
    path : str
        Full path to the target SQLite database.
    merge_from : str
        Optional path to the source database to merge or copy.
        If you pass an empty string or a path that doensn't exist
        no merge is done.
    merge_classes : Optional[Sequence[str]]
        Optional job classes to select when merging. None selects all
        job classes.

    Returns
    -------
    jobdb.JobManager
        Database manager using a ten-second SQLite timeout and NullPool.
    """
    myrank = 0
    if comm is not None:
        myrank = comm.Get_rank()
    # Let rank 0 make jobdb first to avoid race conditions
    jdb = None
    if myrank == 0:
        if merge_from != "" and os.path.isfile(merge_from):
            if os.path.isfile(path):
                sync_jobs(merge_from, path, merge_classes)
            else:
                shutil.copyfile(merge_from, path)
        engine = sqy.create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
        jdb.clear_locks(jobs="all")

        if comm is None:
            return jdb
    if comm is not None:
        comm.barrier()
    if myrank != 0:
        engine = sqy.create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
    if jdb is None:
        raise ValueError("Jobdb is none somehow!")
    return jdb


def setup_jobs(
    comm: Optional["Comm"],
    jdb: str | jobdb.JobManager,
    jclass: str,
    get_jobdict: Callable[[jobdb.JobManager], dict[str, jobdb.Job]],
    get_jobit: Callable[[jobdb.JobManager], Iterable[Any]],
    get_jobstr: Callable[[Any], Optional[str]],
    get_tags: Callable[[Any], dict[str, str]],
    tag_check: list[tuple[str, Sequence[Any]]],
    overwrite: bool,
    retry_failed: bool,
    job_memory: Optional[float],
    job_memory_buffer: float,
    include_done: bool,
    logger: LoggerLike,
) -> tuple[jobdb.JobManager, list[jobdb.Job]]:
    """
    Discover, create, and select jobs for execution.

    Existing jobs are loaded from the job database and matched against a
    collection of candidate jobs generated by `get_jobit`. Missing jobs are
    created, eligible jobs are reopened when requested, and a consolidated
    list of jobs to process is returned.

    Jobs may be filtered by lock status, tags, recent visit time, and
    job state. Database writes are serialized across MPI ranks to avoid
    contention.

    Parameters
    ----------
    comm : Optional[Comm]
        MPI communicator used to prevent deadlock between processes.
    jdb : str | jobdb.JobManager
        Either the path to the db or an already loaded db.
    jclass : str
        Job class name passed to `JobManager.create_job` when creating
        missing jobs.
    get_jobdict : Callable[[jobdb.JobManager], dict[str, jobdb.Job]]
        Function that returns a mapping from job identifier strings to
        existing jobs in the database.
    get_jobit : Callable[[jobdb.JobManager], Iterable[Any]]
        Function that returns an iterable of candidate job descriptions.
    get_jobstr : Callable[[Any], Optional[str]]
        Function that converts a candidate job description into its unique
        job identifier string. Returning `None` causes the candidate to be
        skipped.
    get_tags : Callable[[Any], dict[str, str]]
        Function that generates the tag dictionary used when creating a new
        job from a candidate job description.
    tag_check : Sequence[tuple[str, Sequence[Any]]]
        Tag filters. Each pair contains a tag name and the allowed values
        for that tag. Jobs with a present tag whose value is not allowed
        are skipped. Jobs missing the tag are not excluded by this filter.
    overwrite : bool
        If True, reopen non-open jobs and include them for processing.
    retry_failed : bool
        If True, include failed jobs for processing.
    job_memory : Optional[float]
        Recent-visit exclusion window in hours. None disables this filter.
    job_memory_buffer : float
        Buffer in minutes: jobs visited more recently than this are not
        excluded by the job-memory filter.
    include_done : bool
        If True, include jobs in the done state.
    logger : LoggerLike
        Logger to log to.

    Returns
    -------
    jdb : jobdb.JobManager
        Job database manager instance.
    jobs : list[jobdb.Job]
        Complete list of jobs selected for processing across all MPI ranks.
    """
    if isinstance(jdb, str):
        jdb = make_jobdb(comm, jdb)
    myrank, nproc = 0, 1
    if comm is not None:
        myrank = comm.Get_rank()
        nproc = comm.Get_size()
    # Get the jobs, make them if we need to
    now = time.time()
    logger.info("Setting up jobdb")
    joblist = []
    jobs_to_make = []
    jobs_to_open = []
    logger.info("Getting jobdict")
    jobdict = None
    if myrank == 0:
        jobdict = get_jobdict(jdb)
    if comm is not None:
        jobdict = comm.bcast(jobdict)
    if jobdict is None:
        raise ValueError("jobdict is None!")
    logger.info("Getting potential jobs")
    it = get_jobit(jdb)
    logger.info("Processing possible jobs")
    for info in it:
        sys.stdout.flush()
        jobstr = get_jobstr(info)
        ignore_lock = False
        if jobstr is None:
            continue
        if jobstr in jobdict:
            job = jobdict[jobstr]
        else:
            tags = get_tags(info)
            job = jdb.create_job(
                jclass=jclass, tags=tags, check_existing=False, commit=False
            )
            jobs_to_make += [job]
            ignore_lock = True
        if cast(bool, job.lock) and not ignore_lock:
            continue
        tag_filt = False
        for tag, tag_list in tag_check:
            if tag not in job.tags:
                continue
            if job.tags[tag] not in tag_list:
                tag_filt = True
                break
        if tag_filt:
            continue
        visit_time = cast(Optional[float], job.visit_time)
        if (
            visit_time is not None
            and job_memory is not None
            and now - visit_time < 60 * 60 * job_memory
            and now - visit_time > 60 * job_memory_buffer
        ):
            continue

        if (
            overwrite
            or job.jstate.name == "open"
            or (job.jstate.name == "failed" and retry_failed)
        ):
            if job.jstate.name != "open":
                job.jstate = cast(sqy.Column[str], jobdb.JState.open)
                jobs_to_open += [job]
            else:
                joblist += [job]
        elif include_done and job.jstate.name == "done":
            joblist += [job]
    if comm is not None:
        comm.barrier()

    # Make the missing jobs
    # Doing this serially so that we don't lock up the db
    tot_missing = len(jobs_to_make)
    if comm is not None:
        tot_missing = comm.reduce(len(jobs_to_make), root=0)
    logger.info("Adding %s new jobs", tot_missing)
    tot_opening = len(jobs_to_open)
    if comm is not None:
        tot_opening = comm.reduce(len(jobs_to_open), root=0)
    logger.info("Opening %s old jobs", tot_opening)
    t0 = time.time()
    for i in range(nproc):
        if myrank == i:
            logger.debug("\tRank %s writing", i)
            jdb.commit_jobs(jobs_to_make)
            jdb.clear_locks(jobs=joblist)
            if len(jobs_to_open) > 0:
                jdb.update_jobs(jobs_to_open)
                with jdb.session_scope() as session:
                    for job in jobs_to_open:  # updated_jobs:
                        jid = job.id
                        refreshed_job = session.get(jobdb.Job, jid)
                        session.expunge(refreshed_job)
                        joblist.append(refreshed_job)
        if comm is not None:
            comm.barrier()
    t1 = time.time()
    logger.info("Took %s seconds to add", t1 - t0)

    # Get the final job list
    if comm is not None:
        all_jobs = comm.allgather(joblist)
        all_jobs = [job for jobs in all_jobs for job in jobs]
    else:
        all_jobs = joblist
    logger.info("%s jobs to run!", len(all_jobs))

    return jdb, all_jobs


def update_jobs_retry(
    jdb: jobdb.JobManager,
    jobs: Sequence[jobdb.Job],
    max_retries: int,
    logger: Optional[LoggerLike] = None,
):
    """
    Update jobs, retrying when SQLite reports a database lock.

    Parameters
    ----------
    jdb : jobdb.JobManager
        Database manager used to persist the jobs.
    jobs : Sequence[jobdb.Job]
        Jobs to update.
    max_retries : int
        Maximum number of attempts, with at least one attempt made.
    logger : Optional[LoggerLike]
        Optional logger for reporting success or failure.

    Raises
    ------
    OperationalError
        If a database operational error occurs for a reason other than
        a database lock.
    """
    t0 = time.time()
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            jdb.update_jobs(jobs)
            elapsed = time.time() - t0
            msg = f"Took {elapsed:.2f} seconds to write in {attempt} attempt(s)"
            if logger is None:
                print(msg)
            else:
                logger.debug(msg)
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            if attempt == attempts:
                msg = f"Failed to write after {attempt} attempt(s): {exc}"
                if logger is None:
                    print(msg)
                else:
                    logger.error(msg)
                raise
            time.sleep(1)
