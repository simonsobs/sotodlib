"""
Module for loading L3 housekeeping data.
"""
from dataclasses import dataclass, field
import yaml
import os
import numpy as np

import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from typing import Union, Tuple, List, Optional, Dict
import so3g
from spt3g import core as spt3g_core
from tqdm.auto import tqdm, trange
from pprint import pprint

Base = declarative_base()
Session = sessionmaker()

@dataclass
class HkConfig:
    """
    Configuration object for loading and indexing HK.

    Args
    -----
    hk_root : str
        Root directory of hk files
    hk_db : str
        Path to hk index database
    echo_db : bool
        Whether to echo database operations
    aliases : Dict[str, str]
        Aliases for hk fields. In this dict, the key is the alias name, and the
        value is the field descriptor, in the format of ``agent.feed.field``.
        For example::

            {
                'fp_temp': 'cryo-ls372-lsa21yc.temperatures.Channel_02_T',
            }
    """
    hk_root: str
    hk_db: str
    echo_db: bool = False
    aliases: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))


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
        Ending ctime of the fil
    """
    __tablename__ = 'hk_files'
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String, nullable=False, unique=True)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)


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
    byte_offset: int
        Offset of the frame in the G3 file in bytes
    start_time: float
        Starting ctime of the frame
    end_time: float
        Ending ctime of the frame
    """
    __tablename__ = 'hk_frames'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('hk_files.id'))
    file = relationship('HkFile')
    agent = db.Column(db.String)
    feed = db.Column(db.String)
    byte_offset = db.Column(db.Integer)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)


def get_items_from_file(hk_path) -> Tuple[HkFile, List[HkFrame]]:
    """
    Returns HkFile and HkFrame objects corresponding to a given hk file.

    Returns
    ---------
    hk_file : HkFile
        HkFile object corresponding to the file
    frames : List[HkFrame]
        List of all HkFrames in the file
    """
    hkfile = HkFile(path=hk_path)
    frames = []
    reader = so3g.G3IndexedReader(hk_path)

    file_start, file_end = 1<<32, 0
    while True:
        byte_offset = reader.Tell()
        frame = reader.Process(None)
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
        for block in frame['blocks']:
            ts = np.array(block.times) / spt3g_core.G3Units.s
            start_time = min(start_time, ts[0])
            stop_time = max(stop_time, ts[-1])
        file_start = min(file_start, start_time)
        file_end = max(file_end, stop_time)
        frames.append(HkFrame(
            agent=agent, feed=feed, byte_offset=byte_offset,
            start_time=start_time, end_time=stop_time, file=hkfile
        ))

    hkfile.start_time = file_start
    hkfile.end_time = file_end
    return hkfile, frames

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

        self.engine = db.create_engine(f"sqlite:///{cfg.hk_db}", echo=cfg.echo_db)
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)


def index_files(paths, db: HkDb):
    """
    Indexes a group of files, adding items to the database.

    Args
    -----
    paths : Union[str, List[str]]
        Path or paths to files to index
    db : HkDb
        Database object to use
    """
    paths = np.atleast_1d(paths)
    items =[]
    for p in paths:
        hk_file, hk_frames = get_items_from_file(p)
        items.append(hk_file)
        items.extend(hk_frames)

    with db.Session.begin() as sess:
        sess.add_all(items)


def get_all_hk_files(cfg: HkConfig):
    """
    Gets list of all hk files in specified root directory
    """
    hk_files = []
    for root, _, files in os.walk(cfg.hk_root):
        for file in files:
            if file.endswith('.g3'):
                hk_files.append(os.path.join(root, file))
    return hk_files


def index_all(cfg: Union[HkConfig, str], show_pb=True, files_per_batch=1):
    """
    Indexes all files in the specified root directory.
    """
    hkdb = HkDb(cfg)
    all_files = get_all_hk_files()

    with hkdb.Session.begin() as sess:
        archived_paths = [path for path, in sess.query(HkFile.path).all()]
        remaining_files = sorted(list(set(all_files) - set(archived_paths)))
    
    for i in trange(0, len(remaining_files), files_per_batch, disable=(not show_pb)):
        index_files(remaining_files[i:i+files_per_batch], hkdb)


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
    
    def matches(self, other):
        if self.agent != other.agent:
            return False
        if self.feed != other.feed and self.feed != '*' and other.feed != '*':
            return False
        if self.field != other.field and self.field != '*' and other.field != '*':
            return False
        return True
    
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
    cfg : HkConfig
        HkConfig object
    fields : List[str]
        List of field specifications to load. This can either be a field
        descriptor, of the format ``agent.feed.field``, or an alias defined in
        the config. Field descriptors can contain wildcards, for instance
        ``agent.*.*`` will load all fields belonging to the specified agent.
        ``agent.feed.*`` and ``agent.*.field`` will also work as expected.
    start : float 
        Starting time of data to load
    end : float
        Ending time of data to load
    """
    cfg: HkConfig
    fields: List[str]
    start: float
    end: float

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


def load_hk(load_spec: LoadSpec, show_pb=False):
    """
    Loads hk data

    Args
    ------
    load_spec: LoadSpec
        Load specification. See docstrings of the LoadSpec class.
    show_pb: bool
        If true, will show a progressbar :)
    """
    hkdb = HkDb(load_spec.cfg)
    agent_set = list(set(f.agent for f in load_spec.fields))

    file_spec = {}  # {path: [offsets]}
    with hkdb.Session.begin() as sess:
        query = sess.query(HkFrame).filter(
            HkFrame.start_time <= load_spec.end,
            HkFrame.end_time >= load_spec.start,
            HkFrame.agent.in_(agent_set)
        )
        for frame in query:
            if frame.file.path not in file_spec:
                file_spec[frame.file.path] = []
            file_spec[frame.file.path].append(frame.byte_offset)
        
    result = {}  # {field: [timestamps, data]}
    def get_result_field(agent, feed, field_name):
        f = Field(agent, feed, field_name)
        key = str(f)
        if key in result:
            return result[key]
        for field in load_spec.fields:
            if field.matches(f):
                result[key] = [[], []]
        return None

    nframes = np.sum([len(offsets) for offsets in file_spec.values()])
    pb = tqdm(total=nframes, disable=(not show_pb))
    for path, offsets in file_spec.items():
        reader = so3g.G3IndexedReader(path)
        for offset in sorted(offsets):
            reader.Seek(offset)
            frame = reader.Process(None)[0]
            addr = frame['address']
            _, agent, _, feed = addr.split('.')
            for block in frame['blocks']:
                ts = np.array(block.times) / spt3g_core.G3Units.s
                for field_name, data in block.items():
                    field = get_result_field(agent, feed, field_name)
                    if field is None:
                        continue
                    field[0].append(ts)
                    field[1].append(np.array(data))
            pb.update()
    pb.close()
    for k in result:
        result[k][0] = np.hstack(result[k][0])
        result[k][1] = np.hstack(result[k][1])

    return HkResult(result, aliases=load_spec.cfg.aliases)
