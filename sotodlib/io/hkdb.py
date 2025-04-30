"""
Module for loading L3 housekeeping data using a database that
indexes an archive of HK files.

"""
from dataclasses import dataclass, field
import yaml
import os
import logging
import time
import numpy as np
import pathlib
import fnmatch
import hashlib

import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import Union, List, Optional, Dict, Any
import so3g
from spt3g import core as spt3g_core
from tqdm.auto import tqdm

Base = declarative_base()
Session = sessionmaker()

log = logging.getLogger(__name__)

@dataclass
class HkConfig:
    """
    Configuration object for indexing and loading from an HK archive.

    If instantiating from a nested dictionary, the ``from_dict`` class method
    can be used to convert fields to their proper data types.

    Args
    ------------
    hk_root: str
        Root directory for the HK archive
    db_file: Optional[str]
        Path to the hk index database if the database is an sqlite file. Either
        this or db_url must be set
    db_url: Optional[Union[str, db.URL]]
        URL used for db engine. Either this or db_file must be set
    echo_db: bool
        Whether database operations should be echoed
    file_idx_lookback_time: Optional[float]
        Time [sec] to look back when scanning for new files to index
    show_index_pb: bool
        If true, shows progress bar when indexing
    aliases: Dict[str, str]
        Aliases for hk fields. In this dict, the key is the alias name, and the
        value is the field descriptor, in the format of ``agent.feed.field``.
        For example::

            {
                'fp_temp': 'cryo-ls372-lsa21yc.temperatures.Channel_02_T',
            }

        These aliases are only used on load, and do not affect how data is stored in the hkdb.
        Aliases should be valid python identifiers, since they will be set as attributes in
        the HkResult object.
    """
    hk_root: str
    db_file: Optional[str] = None
    db_url: Optional[Union[str, db.URL]] = None
    echo_db: bool = False
    file_idx_lookback_time: Optional[float] = None
    show_index_pb: bool = True
    aliases: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.db_file is None and self.db_url is None:
            raise ValueError("Either db_file or db_url must be set")
        if self.db_file is not None and self.db_url is not None:
            raise ValueError("Only one of db_file or db_url must be set")
        if self.db_file is not None:
            self.db_url = f"sqlite:///{self.db_file}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HkConfig":
        """
        Generates an HkConfig object from a dictionary whose keys are fields of
        the HkConfig dataclass.

        If the ``db_url`` is specified, it can be set as a string, a
        dictionary, or a sqlalchemy URL object. If it is of type dict, it
        will be converted to a URL object by passing it through to the keyword
        arguments of sqlalchemy's URL.create function. Environment variables
        will be expanded in both the string and dict representations.
        """
        _db_url = data.get('db_url')
        if isinstance(_db_url, dict):
            url_dict = data['db_url']
            for k, v in url_dict.items():
                url_dict[k] = os.path.expandvars(v)
            data['db_url'] = db.URL.create(**url_dict)
        elif isinstance(_db_url, str):
            data['db_url'] = os.path.expandvars(data['db_url'])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))


class HkFile(Base):
    """
    Database entry for a hk file.

    Args
    ------
    path: str
        Path to the hk file.
    start_time: float
        Starting ctime of the file
    end_time: float
        Ending ctime of the file
    size: int
        Size of file in bytes
    mod_time: float
        Modification time of file
    index_status: float
        Index status of file. Can be "unindexed", "indexed", or "failed"
    """
    __tablename__ = 'hk_files'
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String, nullable=False, unique=True)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)
    size = db.Column(db.Float)
    mod_time = db.Column(db.Float)
    index_status = db.Column(db.String)


class HkFrame(Base):
    """
    Database entry for an hk frame.

    Args
    --------
    file_id: int
        ID of the file containing the frame
    agent: str
        instance-id of the OCS agent for the frame
    feed: str
        Name of the OCS feed for the frame
    fields_hash: str
        Hash to identify the combination of fields included in this frame.
    byte_offset: int
        Offset of the frame in the G3 file in bytes
    start_time: float
        Starting ctime of the frame
    end_time: float
        Ending ctime of the frame
    """
    __tablename__ = 'hk_frames'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('hk_files.id'),
                        index=True)  # index required on postgres, or
                                     # else delete is slow
    file = relationship('HkFile')
    agent = db.Column(db.String)
    feed = db.Column(db.String)
    fields_hash = db.Column(db.String)
    byte_offset = db.Column(db.Integer)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)


class HkDb:
    """
    Helper class for createing database sessions

    Args
    ------
    cfg : Union[HKConfig, str]
        Configuration object or path to configuration file
    """
    def __init__(self, cfg: Union[HkConfig, str]):
        if isinstance(cfg, str):
            cfg = HkConfig.from_yaml(cfg)
        self.cfg = cfg

        self.engine = db.create_engine(cfg.db_url, echo=cfg.echo_db)
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)


def update_file_index(hkcfg: HkConfig, session=None, subdirs=None):
    """Updates HkFiles database with new files on disk.

    If subdirs is specified, it must be a list of (full path)
    sub-directories of hk_root; those will be scanned and any new
    files added to database.

    Otherwise, all subdirs of hk_root will be scanned, subject to any
    restriction from hkcfg.file_idx_lookback_time.

    """
    if session is None:
        hkdb = HkDb(hkcfg)
        session = hkdb.Session()

    if hkcfg.file_idx_lookback_time is not None and subdirs is None:
        min_ctime = time.time() - hkcfg.file_idx_lookback_time
    else:
        min_ctime = 0

    all_files = []
    if subdirs is None:
        subdirs = [os.path.join(hkcfg.hk_root, subdir)
                   for subdir in os.listdir(hkcfg.hk_root)]
    else:
        _root = pathlib.Path(hkcfg.hk_root)
        assert all([_root in pathlib.Path(sdir).parents for sdir in subdirs])

    for sdir in subdirs:
        if min_ctime > 0 and min_ctime > os.path.getmtime(sdir):
            continue
        all_files.extend([
            os.path.join(sdir, f)
            for f in os.listdir(sdir)
            if f.endswith('.g3')
        ])

    existing_files = [
        os.path.join(hkcfg.hk_root, path)
        for path, in session.query(HkFile.path).all()
    ]
    new_files = sorted(list(set(all_files) - set(existing_files)))

    files = []
    log.info(f"Adding {len(new_files)} new files to index...")
    for path in new_files:
        relpath = os.path.relpath(path, hkcfg.hk_root)
        files.append(HkFile(
            path=relpath,
            size=os.path.getsize(path),
            mod_time=os.path.getmtime(path),
            index_status='unindexed'
        ))

    session.add_all(files)
    session.commit()


def get_frames_from_file(
    hkcfg: HkConfig,
    file: HkFile,
    return_on_fail=True
) -> List[HkFrame]:
    """
    Returns HkFile and HkFrame objects corresponding to a given hk file.

    Args
    --------
    file : HkFile
        HkFile object corresponding to the file
    return_on_fail : bool
        If True, if there is a runtime error while reading the g3 file (usually
        caused by a forced shutdown), the function will still return parsed
        frames.

    Returns
    ---------
    frames : List[HkFrame]
        List of all HkFrames in the file
    """
    frames = []
    if os.path.isabs(file.path):
        path = str(file.path)
    else:
        path = os.path.join(hkcfg.hk_root, str(file.path))
    reader = so3g.G3IndexedReader(path)

    while True:
        byte_offset = reader.Tell()
        try:
            frame = reader.Process(None)
        except RuntimeError:
            log.error(f"Error processing file {file.path} byte offset: {byte_offset}")
            if return_on_fail:
                break
            else:
                raise
        if not frame:
            break
        else:
            frame = frame[0]

        if frame['hkagg_type'] != so3g.HKFrameType.data:
            continue

        # Process frame
        addr = frame['address']
        _, agent, _, feed = addr.split('.')
        start_time, stop_time = 1<<32, 0
        fields = []
        for block in frame['blocks']:
            ts = np.array(block.times) / spt3g_core.G3Units.s
            fields.extend(block.keys())
            start_time = min(start_time, ts[0])
            stop_time = max(stop_time, ts[-1])

        fields_hash = hashlib.sha256(','.join(sorted(fields)).encode('ascii')).hexdigest()[:16]
        frames.append(HkFrame(
            agent=agent, feed=feed, fields_hash=fields_hash, byte_offset=byte_offset,
            start_time=start_time, end_time=stop_time, file=file
        ))

    return frames


def update_frame_index(hkcfg: HkConfig, session=None):
    """Updates HkFrames database with frames from unindexed files"""
    if session is None:
        hkdb = HkDb(hkcfg)
        session = hkdb.Session()

    files = session.query(HkFile).filter(HkFile.index_status == 'unindexed').all()
    log.info(f"Indexing {len(files)} files")
    for file in tqdm(files, disable=(not hkcfg.show_index_pb), ascii=True):
        frames = get_frames_from_file(hkcfg, file)
        file_start, file_end = 1<<32, 0
        for f in frames:
            file_start = min(file_start, f.start_time)
            file_end = max(file_end, f.end_time)
        file.start_time = file_start
        file.end_time = file_end
        file.index_status ='indexed'
        try:
            session.add_all(frames)
            session.commit()
        except Exception as e:
            session.rollback()
            file.index_status = 'failed'
            session.commit()


def update_index_all(cfg: Union[HkConfig, str], subdirs=None):
    """Updates all HK index databases"""
    if isinstance(cfg, str):
        cfg = HkConfig.from_yaml(cfg)
    hkdb = HkDb(cfg)
    session = hkdb.Session()
    update_file_index(cfg, session=session, subdirs=subdirs)
    update_frame_index(cfg, session=session)


def purge_unindexed_files(hkcfg: HkConfig):
    """Remove any 'unindexed' files from the database.  This can be used
    to ignore files that disappeared from filesystem before being indexed.

    """
    hkdb = HkDb(hkcfg)
    session = hkdb.Session()
    session.query(HkFile).filter(HkFile.index_status == 'unindexed').delete(
        synchronize_session=False)
    session.commit()


#####################
# HK Loading stuff
#####################
@dataclass
class Field:
    agent: str
    feed: str
    field: str

    def __str__(self):
        return f"{self.agent}.{self.feed}.{self.field}"

    @staticmethod
    def _wcmatch(a, b):
        if '*' in a:
            return fnmatch.fnmatch(b, a)
        if '*' in b:
            return fnmatch.fnmatch(a, b)
        return a == b

    def matches(self, other):
        return (self.agent == other.agent
                and Field._wcmatch(self.feed, other.feed)
                and Field._wcmatch(self.field, other.field))

    @classmethod
    def from_str(cls, s):
        try:
            agent, feed, field = s.split('.')
        except Exception:
            raise ValueError(f"Could not parse field: {s}")

        return cls(agent, feed, field)

@dataclass
class LoadSpec:
    """
    HK loading specification

    Args
    -----
    cfg: HkConfig
        Configuration object
    fields: List[str]
        List of field specifications to load. This can either be a field
        descriptor, of the format ``agent.feed.field``, or an alias defined in
        the config. Field descriptors can contain wildcards in the feed and
        field portion., for instance ``agent.*.*`` will load all fields
        belonging to the specified agent.  ``agent.feed.*`` and
        ``agent.*.*word*`` will also work as expected.
    start: float
        Start time to load
    end: float
        End time to load
    downsample_factor: int
        Downsample factor for data
    hkdb: Optional[HkDb]
        HkDb instance to use. If not specified, will create a new one from the
        cfg. This should be set manually if you are calling ``load_hk`` in a loop
        to prevent connection build-up.
    """
    cfg: HkConfig
    fields: List[str]
    start: float
    end: float
    downsample_factor: int = 1
    hkdb: Optional[HkDb] = None

    def __post_init__(self):
        fs = []
        for f in self.fields:
            if f in self.cfg.aliases:
                fs.append(Field.from_str(self.cfg.aliases[f]))
            else:
                fs.append(Field.from_str(f))
        self.fields = fs


class HkResult:
    """
    Helper class for storing results of LoadHk. If aliases are set for any
    of the keys, they will be set as attributes of the object.

    Attributes
    ------------
    data: dict
        Dict where the key is the field descriptor, and the value is a list
        where val[0] are timestamps, and val[1] is data.
    """
    def __init__(self, data, aliases=None):
        if aliases is None:
            aliases = {}
        self._aliases = aliases
        self.data = data
        for alias, key in aliases.items():
            if key in self.data:
                setattr(self, alias, self.data[key])

    def save(self, path):
        np.savez(path, aliases=np.array(self._aliases), **self.data)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        aliases = d['aliases'].item()
        data = {k: d[k] for k in d.files if k != 'aliases'}
        return cls(data, aliases=aliases)


def load_hk(load_spec: Union[LoadSpec, dict], show_pb=False,
            fields=None, start=None, end=None,
            _field_list_scan=False):
    """
    Loads hk data

    Args
    ------
    load_spec: LoadSpec
        Load specification. See docstrings of the LoadSpec class.
    show_pb: bool
        If true, will show a progressbar :)
    fields: List[str]
        Fields to load (overrides load_spec.fields).
    start: float
        Starting timestamp (overrides load_spec.start).
    end: float
        Ending timestamp (overrides load_spec.end).
    _field_list_scan: bool
        Run in special mode to support get_field_list.
    """
    if isinstance(load_spec, dict):
        load_spec = LoadSpec(**load_spec)

    if load_spec.hkdb is not None:
        hkdb: HkDb = load_spec.hkdb
    else:
        hkdb = HkDb(load_spec.cfg)

    if fields is None:
        fields = load_spec.fields
    fields = [Field.from_str(f) if isinstance(f, str) else f
              for f in fields]

    if start is None:
        start = load_spec.start
    if end is None:
        end = load_spec.end

    agent_set = list(set(f.agent for f in fields))

    file_spec = {}  # {path: [offsets]}
    feeds = []
    with hkdb.Session.begin() as sess:
        query = sess.query(HkFrame).filter(
            HkFrame.start_time <= end,
            HkFrame.end_time >= start,
            HkFrame.agent.in_(agent_set)
        ).order_by(HkFrame.start_time)
        for frame in query:
            if _field_list_scan:
                feed_key = (frame.agent, frame.feed, frame.fields_hash)
                if feed_key in feeds:
                    continue
                feeds.append(feed_key)
            if frame.file.path not in file_spec:
                file_spec[frame.file.path] = []
            file_spec[frame.file.path].append(frame.byte_offset)

    # Convert all paths to absolute paths based on cfg.hk_root
    def create_abs_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(load_spec.cfg.hk_root, path)
    file_spec = {create_abs_path(k): v for k, v in file_spec.items()}

    result = {}  # {field: [timestamps, data]}
    field_misses = set()
    def get_result_field(agent, feed, field_name):
        f = Field(agent, feed, field_name)
        key = str(f)

        if key in result:
            return result[key]

        if key in field_misses:
            return None

        for field in fields:
            if field.matches(f):
                result[key] = [[], []]
                return result[key]
        # Cache field on miss
        field_misses.add(key)
        return None

    ds_factor = load_spec.downsample_factor
    nframes = np.sum([len(offsets) for offsets in file_spec.values()])

    pb = tqdm(total=nframes, disable=(not show_pb))
    for path in sorted(list(file_spec.keys())):
        offsets = file_spec[path]
        reader = so3g.G3IndexedReader(path)
        for offset in sorted(offsets):
            reader.Seek(offset)
            frame = reader.Process(None)[0]
            addr = frame['address']
            _, agent, _, feed = addr.split('.')
            for block in frame['blocks']:
                ts = np.array(block.times)[::ds_factor] / spt3g_core.G3Units.s
                for field_name, data in block.items():
                    field = get_result_field(agent, feed, field_name)
                    if field is None or _field_list_scan:
                        continue
                    field[0].append(ts)
                    field[1].append(np.array(data)[::ds_factor])
            pb.update()
    pb.close()

    if _field_list_scan:
        return list(result.keys())

    for k, d in result.items():
        if len(d[0]) == 0:
            result[k] = (np.array([]), np.array([]))
        else:
            result[k] = (np.hstack(d[0]), np.hstack(d[1]))

    return HkResult(result, aliases=load_spec.cfg.aliases)


def get_feed_list(load_spec: Union[LoadSpec, dict]) -> List[str]:
    """Return the list of feeds present in the db and for the time range
    specified by a LoadSpec.

    Args
    ----
    load_spec: LoadSpec
        Load specification. See docstrings of the LoadSpec class.

    Returns
    -------
    List[str]
      The list of feeds, as field spec strings, with wildcard for the
      field e.g. "an_agent.a_feed.*".

    Notes
    -----
    The .start_time and .end_time are respected in the query, but the
    .fields entry is ignored in the search.

    """
    if isinstance(load_spec, dict):
        load_spec = LoadSpec(**load_spec)

    if load_spec.hkdb is not None:
        hkdb: HkDb = load_spec.hkdb
    else:
        hkdb = HkDb(load_spec.cfg)

    pairs = set()
    with hkdb.Session.begin() as sess:
        query = sess.query(HkFrame.agent, HkFrame.feed).filter(
            HkFrame.start_time <= load_spec.end,
            HkFrame.end_time >= load_spec.start,
        )
        for row in query:
            pairs.add(tuple(row))

    return [f'{a}.{b}.*' for a, b in sorted(list(pairs))]


def get_field_list(load_spec: Union[LoadSpec, dict],
                   fields: List[Field]=None) -> List[str]:
    """Inspect the HK files to get the field names associated with
    each feed covered by the load_spec (or by the fields argument).
    This is shallow search in that only a single frame from every
    ``agent.feed`` combination matching the fields list is inspected.

    Args
    ----
    load_spec: LoadSpec
        Load specification. See docstrings of the LoadSpec class.
    fields:
        List of fields (which may include wildcards) to match against.

    Returns
    -------
    List[str]
      The list of fields, as field spec strings.

    Notes
    -----

    If fields is not specified, then it is taken from load_spec.  But
    normally you'd want it from the get_feeds_list.  The .start_time
    and .end_time are respected in the query, especially in the sense
    that the shallow data search will begin at .start_time.

    """
    return load_hk(load_spec, fields=fields, _field_list_scan=True)
