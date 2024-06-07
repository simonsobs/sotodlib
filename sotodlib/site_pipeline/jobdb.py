"""
jobdb
=====

Connect to a database
---------------------

With sqlite::

  jdb = jobdb.JobManager(sqlite_file='jobsdb-test.db')

With postgres::

  db_url = sqy.engine.URL.create(
        "postgresql",
        username="abc",
        password="def",
        host="ghi",
        database="jkl")
  jdb = jobdb.JobManager(url=db_url)


Create new jobs
---------------

Check if job exists; create::

  jclass = 'my_analysis'
  tags = {'obs_id': '1234', 'wafer_slot': 10}
  if not len(jdb.get_jobs(jclass=jclass, tags=tags)):
    jdb.create_job(jclass, tags)

Identifying work
----------------

Find jobs to do::

  all_jobs = jdb.get_jobs(jclass=jclass, jstate='open')
  done_everything = len(all_jobs) == 0

Filter jobs to ones we might want to work on now::

  recent_memory = time.time() - 3600  # Don't retry until 1 hour has passed.
  to_do = [j for j in all_jobs \
           if j.jstate == 'open' and j.visit_time <= recent_memory]

Lock a job and work on it::

  with jdb.locked(to_do) as job:
    if job is not None:
      job.mark_visited()
      ok = do_a_job(job.tags)
      if ok:
        job.jstate = 'done'
      else:
        if job.visit_count > 5:
          job.state = 'failed'

Fixing things
-------------

Forcibly unlock all jobs (though feel free to be more targeted)::

  for j in jdb.get_jobs(jclass='my_analysis'):
    jdb.unlock(j.id)

Delete some jobs::

  for j in jdb.get_jobs(jclass='my_analysis'):
    jdb.remove_job(j.id)


"""

import sqlalchemy as sqy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, aliased
from contextlib import contextmanager
import argparse
import enum
import time
import os
import sys


__all__ = ['JobLockedError',
           'JobNotLockedError',
           'JobNotOwnedError',
           'JobNotUniqueError',
           'JState',
           'Job',
           'Tag',
           'JobManager']


class JobLockedError(Exception):
    pass


class JobNotLockedError(Exception):
    pass


class JobNotOwnedError(Exception):
    pass


class JobNotUniqueError(Exception):
    pass


class JState(enum.Enum):
    open = "open"
    done = "done"
    failed = "failed"
    ignored = "ignored"

    @classmethod
    def all(cls):
        return list(cls.__members__.keys())


Base = declarative_base()


class Job(Base):
    __tablename__ = 'jobs'
    id = sqy.Column(sqy.Integer, primary_key=True)
    jclass = sqy.Column(sqy.String)
    jstate = sqy.Column(sqy.Enum(JState))
    lock = sqy.Column(sqy.Float)
    lock_owner = sqy.Column(sqy.String)

    creation_time = sqy.Column(sqy.Float, default=0.)
    visit_time = sqy.Column(sqy.Float, default=0.)
    visit_count = sqy.Column(sqy.Integer, default=0)

    _tags = relationship("Tag", lazy='selectin', cascade='all')

    @property
    def tags(self):
        return Tag.items(self._tags)

    def __repr__(self):
        try:
            return (f'Job(id={self.id},jclass={self.jclass},'
                    f'jstate={self.jstate.value})')
        except Exception:
            return 'Job@%x' % id(self)

    def mark_visited(self, visit_count=None, now=None):
        if visit_count is None:
            visit_count = self.visit_count + 1
        if now is None:
            now = time.time()
        self.visit_count = visit_count
        self.visit_time = now


class Tag(Base):
    __tablename__ = 'tags'
    id = sqy.Column(sqy.Integer, primary_key=True)
    job_id = sqy.Column(sqy.Integer, sqy.ForeignKey('jobs.id'))
    key = sqy.Column(sqy.String)
    value = sqy.Column(sqy.String)

    def __repr__(self):
        return f'Tag(@job_id={self.id}:{self.key}={self.value})'

    def item(self):
        return self.key, self.value

    @staticmethod
    def items(tag_list):
        td = {}
        for t in tag_list:
            k, v = t.item()
            td[k] = v
        return td


class JobManager:
    def __init__(self, engine=None, url=None, sqlite_file=None):
        if engine is None:
            if url is None:
                if sqlite_file is None:
                    sqlite_file = 'my_jobs.db'
                engine = sqy.create_engine(
                    f'sqlite:///{sqlite_file}',
                    connect_args={'timeout': 3})
            else:
                engine = sqy.create_engine(url)
        self.engine = engine
        Base.metadata.create_all(self.engine)
        self.get_session = sessionmaker(bind=self.engine)

    def _lockstr(self):
        return 'me%i' % os.getpid()

    def create_job(self,
                   jclass=None,
                   tags={},
                   jstate=None,
                   creation_time=None,
                   visit_count=None,
                   visit_time=None):
        """Create a new job in the jobs table.

        Return the job.

        """
        existing = self.get_jobs(
            jclass, tags=tags,
            jstate=JState.all(), locked=None)
        if len(existing):
            raise JobNotUniqueError(
                'Found other records with same tags: '
                f'{[x.id for x in existing]}')

        if creation_time is None:
            creation_time = time.time()
        if jstate is None:
            jstate = JState.open
        tags = [Tag(key=k, value=str(v)) for k, v in tags.items()]
        job = Job(jclass=jclass,
                  jstate=jstate,
                  creation_time=creation_time,
                  visit_count=visit_count,
                  visit_time=visit_time,
                  _tags=tags)
        with self.session_scope() as session:
            session.add(job)
            session.commit()
            session.expunge(job)
        return job

    def get_jobs(self,
                 jclass=None,
                 tags=None,
                 jstate=None,
                 locked=None,
                 job_id=None):
        """Get a list of jobs meeting particular criteria.

        Note jclass and jstate can be string, a list (match any in
        list) or None (matches all job classes).

        The returned objects are detached from any database session,
        and should not be modified.  To operate on one or more of the
        jobs returned here, pass them first to the locked() context
        manager, which will check the records out from the database
        (if possible) for use in your process.

        """
        with self.get_session() as session:
            q = session.query(Job)
            if isinstance(jstate, str) and jstate == 'all':
                jstate = JState.all()
            if jclass is not None:
                if isinstance(jclass, str):
                    jclass = [jclass]
                q = q.filter(sqy.or_(*[Job.jclass == c for c in jclass]))
            if jstate is not None:
                if isinstance(jstate, str) or isinstance(jstate, JState):
                    jstate = [jstate]
                q = q.filter(sqy.or_(*[Job.jstate == s for s in jstate]))
            if tags is not None:
                for k, v in tags.items():
                    tags_alias = aliased(Tag)
                    q = q.join(tags_alias, Job.id == tags_alias.job_id).\
                        filter(sqy.and_(tags_alias.key == k,
                                        tags_alias.value == v))
            if job_id is not None:
                q = q.filter(Job.id.in_(job_id))
            if locked is True:
                q = q.filter(Job.lock != None)  # noqa: E711
            elif locked is False:
                q = q.filter(Job.lock == None)  # noqa: E711
            else:
                assert locked is None
            q = q.order_by(Job.jclass, Job.id)
            jobs = q.all()
            [session.expunge(j) for j in jobs]
        return jobs

    def lock(self, job_id, owner=None, force=False):
        """Lock a Job record by id.  If the Job is already locked, a
        JobLockedError is raised.

        Returns a Job object that has been expunged from the database
        session.  The object attributes can be modified, but won't be
        written back to the database unless the object is merged into
        a new session.

        """
        if owner is None:
            owner = self._lockstr()
        with self.session_scope() as session:
            q = session.query(Job)
            if force:
                q = q.filter(sqy.and_(Job.id == job_id))
            else:
                q = q.filter(sqy.and_(Job.id == job_id,
                                      Job.lock == None))  # noqa: E711
            n = q.update({Job.lock: time.time(), Job.lock_owner: owner})
            session.commit()

        with self.session_scope() as session:
            job = session.get(Job, job_id)
            session.expunge(job)

        if n == 0 or job.lock_owner != owner:
            raise JobLockedError()

        return job

    def unlock(self, job, merge=True):
        if not merge or isinstance(job, int):
            if isinstance(job, Job):
                job = job.id
            with self.session_scope() as session:
                session.query(Job).filter(Job.id == job).update(
                    {Job.lock: None, Job.lock_owner: None})
                session.commit()
        else:
            with self.session_scope() as session:
                j1 = session.query(Job).filter(Job.id == job.id).one()
                if j1.lock_owner is None:
                    raise JobNotLockedError()
                if j1.lock_owner != job.lock_owner:
                    raise JobNotOwnedError()
                job.lock = None
                job.lock_owner = None
                session.merge(job)
                session.commit()

    def clear_locks(self, jobs=None):
        if jobs is None:
            raise ValueError('Pass jobs="all" to clear all locks.')
        if jobs == 'all':
            jobs = [None]
        jobs = [(j.id if isinstance(j, Job) else j) for j in jobs]
        with self.session_scope() as s:
            for j in jobs:
                q = s.query(Job)
                if j is not None:
                    q = q.filter(Job.id == j)
                q.update({Job.lock: None, Job.lock_owner: None})

    def remove_job(self, job_id, check_locked=False):
        with self.session_scope() as session:
            if check_locked:
                q = session.query(Job).filter(
                    sqy.and_(Job.id == job_id,
                             Job.lock == None))  # noqa: E711
            else:
                q = session.query(Job).filter(Job.id == job_id)

            n = q.delete()
            session.commit()
        if n == 0:
            raise JobLockedError()

    @contextmanager
    def locked(self, jobs, count=None, owner=None):
        """Context Manager to grant exclusive access to one or more
        Job.  Job record is marked as locked, and this process may
        freely work on the job and alter the job data.  When execution
        leaves the context, the Job will be marked as unlocked.  Note
        the _database_ is only explicitly locked while this lock is
        being acquired and released.  In between, other entities can
        do other database stuff.

        Args:
          job (int, Job, or list): The Job to lock, or list of Jobs
            from which to try to draw lockable ones.
          count (int, None): The number of jobs to lock.  If specified
            as an integer, a list of up to that many jobs will be
            yielded.  If None, then a single job will be locked and
            yielded directly (if possible), otherwise None is yielded.
          owner (str): Override lock_owner string.

        Notes:
          If the job argument is a list, the function will try to
          yield one of the jobs from the list, skipping any that are
          locked by another session.  If no unlocked jobs are
          available, the usual exception will be raised or else a None
          yielded, as per none_if_locked argument.

        """
        if owner is None:
            owner = self._lockstr()
        if isinstance(jobs, (int, Job)):
            jobs = [jobs]
        locked = []
        for job in jobs:
            if len(locked) >= (1 if count is None else count):
                break
            if isinstance(job, Job):
                job = job.id
            try:
                j = self.lock(job)
            except JobLockedError:
                continue
            locked.append(j)
        try:
            if count is None:
                if len(locked):
                    yield locked[0]
                else:
                    yield None
            else:
                yield locked
        finally:
            for j in locked:
                self.unlock(j)

    def get_resource(self, jclass, n=None, jstate='open', tags={}):
        jobs = self.get_jobs(jclass, jstate=jstate, tags=tags)
        resources = []
        for job in jobs:
            if len(resources) >= (1 if n is None else n):
                break
            try:
                resources.append(ResourceHandle(self, job))
            except JobLockedError:
                pass
        if n is None:
            return resources[0] if len(resources) else None
        return resources

    @contextmanager
    def session_scope(self):
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class ResourceHandle:
    job_id = None

    def __init__(self, db, job):
        self.db = db
        self.db.lock(job.id)
        self.job_id = job.id
        self.tags = dict(job.tags)

    def __del__(self):
        if self.job_id is not None:
            self.db.unlock(self.job_id)


# CLI support ...

def get_db(args):
    if args.sqlite_file:
        return JobManager(sqlite_file=args.sqlite_file)
    else:
        raise

def get_query_kwargs(args):
    kw = {}
    if args.jstate is not None:
        kw['jstate'] = args.jstate.split(',')
    if args.jclass is not None:
        kw['jclass'] = args.jclass.split(',')
    if args.job_id is not None:
        kw['job_id'] = []
        for _id in args.job_id:
            kw['job_id'].extend([_x.strip() for _x in _id.split(',')])
    return kw

def select(jdb, kw={}):
    jobs = jdb.get_jobs(**kw)
    for j in jobs:
        locked = f'L@{j.lock:.1f}' if j.lock else 'unlocked'
        visited = f'V%i@%.1f' % (j.visit_count, j.visit_time)
        print(f'{j.id:6} {j.jclass:20} {j.jstate.value:8} {locked} {visited} '
              f'{j.tags}')
    return jobs

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file')
    parser.add_argument('-d', '--sqlite-file')

    query_parser = argparse.ArgumentParser(add_help=False)
    query_parser.add_argument('--jclass')
    query_parser.add_argument('--jstate')
    query_parser.add_argument('--job-id', action='append', default=None)
    modep = parser.add_subparsers(
        dest='mode')

    p = modep.add_parser(
        'select',
        parents=[query_parser], help=
        "Print rows from the database."
        )
    p.add_argument('--action', default='none', choices=
                   ['none', 'list', 'clear-locks', 'delete',
                    'set-open', 'set-done', 'set-failed', 'set-ignored'])

    return parser

def cli(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = get_parser()
    args = parser.parse_args(args=args)

    if args.mode == 'select':
        jdb = get_db(args)
        kw = get_query_kwargs(args)
        jobs = select(jdb, kw=kw)

        if args.action == 'none':
            print('Pass --action to do something to these records.')
        elif args.action == 'list':
            pass
        elif args.action == 'clear-locks':
            print('Clearing locks ...')
            jdb.clear_locks(jobs)
        elif args.action == 'delete':
            print('Removing jobs ...')
            for j in jobs:
                jdb.remove_job(j)
        elif args.action.startswith('set-'):
            for k in ['open', 'done', 'failed', 'ignored']:
                if args.action == f'set-{k}':
                    for j in jobs:
                        with jdb.locked(j) as _j:
                            if _j is None:
                                print(f'Failed to lock job_id={j.id}')
                            else:
                                print(f'Updating state for job_id={j.id}')
                                _j.jstate = k
        else:
            parser.error('Provide valid --action=... arg.')
    else:
        parser.error('Provide a mode')
