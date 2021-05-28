import sqlalchemy as db
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

from spt3g import core as spt3g_core
import so3g
import datetime as dt
import os
import re
from tqdm import tqdm
import numpy as np
import yaml
import ast
from collections import namedtuple
from enum import Enum

import logging
logger = logging.getLogger(__name__)

from .. import core
from . import load as io_load


Base = declarative_base()
Session = sessionmaker()
num_bias_lines = 16


association_table = db.Table('association_chan_assign', Base.metadata,
    db.Column('tunesets', db.Integer, db.ForeignKey('tunesets.id')),
    db.Column('chan_assignments', db.Integer, db.ForeignKey('chan_assignments.id'))
)

association_table_obs = db.Table('association_obs', Base.metadata,
    db.Column('tunesets', db.Integer, db.ForeignKey('tunesets.id')),
    db.Column('observations', db.Integer, db.ForeignKey('obs.obs_id'))
)

association_table_dets = db.Table('detsets', Base.metadata,
    db.Column('name', db.Integer, db.ForeignKey('tunesets.name')),
    db.Column('det', db.Integer, db.ForeignKey('channels.name'))
)


SMURF_ACTIONS = {
    'observations':[
        'take_stream_data',
        'stream_data_on',
        'take_noise_psd',
        'take_g3_data',
        'stream_g3_on',         
    ],
    'channel_assignments':[
        'setup_notches',
    ],
    'tuning':[
        'setup_notches',
        'save_tune',
    ]
}


class Observations(Base):
    """Times on continuous detector readout. This table is named obs and serves
    as the ObsDb table when loading via Context.

    Attributes 
    -----------
    obs_id : string
        The "session_id" of the observation which is usually the integer ctime
        of the pysmurf action called to start the readout. (The first part of
        the .g3 file name).
    timestamp : integer
        Generally int(obs_id). Here to make relative querying easier
    duration : float
        The total observation time in seconds
    start : datetime.datetime
        The start of the observation as a datetime object
    stop : datetime.datetime
        The end of the observation as a datetime object
    tag : string
        Tags for this observation. These are planned to be populated through
        tags set while running sodetlib's stream data functions. Not implemented
        yet.
    files : list of SQLAlchemy instances of Files
        The list of .g3 files in this observation built through a relationship
        to the Files table. [f.path for f in Observation.files] will return
        paths to all the files.
    tunesets : list of SQLAlchemy instances of TuneSets 
        The TuneSets used in this observation. There is expected to be
        one per stream_id (SMuRF crate slot). 
    """
    __tablename__ = 'obs'
    ## ctime of beginning of the observation
        
    obs_id = db.Column(db.String, primary_key=True)
    timestamp = db.Column(db.Integer)
    # in seconds
    duration = db.Column(db.Float)

    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)
    
    tag = db.Column(db.String)
    ## one to many
    files = relationship("Files", back_populates='observation') 
    
    ## many to many
    tunesets = relationship("TuneSets", 
                           secondary=association_table_obs,
                           back_populates='observations')

    def __repr__(self):
        try:
            return f"{self.obs_id}: {self.start} -> {self.stop} [{self.stop-self.start}] ({self.tag})"
        except:
            return f"{self.obs_id}: {self.start} -> {self.stop} ({self.tag})"

    

class Files(Base):
    """Table to store file indexing info. This table is named files in sql and
    serves as the ObsFileDb when loading via Context.

    Attributes
    ------------
    id : integer
        auto-incremented primary key
    path : string
        complete absolute path to file
    name : string
        the file name
    start : datetime.datetime
        the start time for the file
    stop : datetime.datetime
        the stop time for the file
    sample_start : integer
        Not Implemented Yet
    sample_stop : integer
        Not Implemented Yet
    obs_id : String 
        observation id linking Files table to the Observation table
    observation : SQLAlchemy Observation Instance
    stream_id : The stream_id for the file. Generally of the form crateXslotY.
        These are expected to map one per UXM.
    n_frames : Integer
        Number of frames in the .g3 file
    frames : list of SQLALchemy Frame Instances
        List of database entries for the frames in this file
    n_channels : Integer
        The number of channels read out in this file
    detset : string
        TuneSet.name for this file. Used to map the TuneSet table to the Files
        table. Called detset to serve duel-purpose and map files to detsets
        while loading through Context. 
    tuneset : SQLAlchemy TuneSet Instance
    tune_id : integer 
        id of the tune file used in this file. Used to map Tune table to the
        Files table.
    tune : SQLAlchemy Tune Instance
    """
    __tablename__ = 'files'
    id = db.Column(db.Integer, primary_key=True)

    path = db.Column(db.String, nullable=False, unique=True)
    ## name is just the end file name
    name = db.Column(db.String, unique=True)
    
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)
    
    ## this is sample in an observation (I think?)
    sample_start = db.Column(db.Integer)
    sample_stop = db.Column(db.Integer)
    
    ## this is a string for compatibility with sotodlib, not because it makes sense here
    obs_id = db.Column(db.String, db.ForeignKey('obs.obs_id'))
    observation = relationship("Observations", back_populates='files')
    
    stream_id = db.Column(db.String)
    
    n_frames = db.Column(db.Integer)
    frames = relationship("Frames", back_populates='file')
    
    ## n_channels is a renaming of channels
    n_channels = db.Column(db.Integer)
    
    # breaking from linked table convention to match with obsfiledb requirements
    ## many to one
    detset = db.Column(db.String, db.ForeignKey('tunesets.name'))
    tuneset = relationship("TuneSets", back_populates='files')
    
    tune_id = db.Column(db.Integer, db.ForeignKey('tunes.id'))
    tune = relationship("Tunes", back_populates='files')
    
class Tunes(Base):
    """Indexing of 'tunes' available during observations. Should
    correspond to all tune files. 

    Attributes 
    -----------
    id : integer 
        primary key
    name : string
        name of tune file
    path : string
        absolute path of tune file
    stream_id : string
        stream_id for file
    start : datetime.datetime
        The time the tune file is made
    files : list of SQLAlchemy File instances
        All file using this tune file
    tuneset_id : integer
        id of tuneset this tune belongs to. Used to link Tunes table to TuneSets
        table
    tuneset : SQLAlchemy TuneSet instance
    """
    __tablename__ = 'tunes'
    __table_args__ = (
        db.UniqueConstraint('name', 'stream_id'),
    )
    id = db.Column( db.Integer, primary_key=True)
    
    name = db.Column(db.String)
    path = db.Column(db.String)
    stream_id = db.Column(db.String)
    
    ## should stop exist? tune file use does not need to be continguous
    start = db.Column(db.DateTime)
    
    ## files that use this tune file
    ## one to many
    files = relationship("Files", back_populates='tune')
    
    ## one to many
    tuneset_id = db.Column(db.Integer, db.ForeignKey('tunesets.id'))
    tuneset = relationship("TuneSets", back_populates='tunes')

class TuneSets(Base):
    """Indexing of 'tunes sets' available during observations. Should
    correspond to the tune files where new_master_assignment=True. TuneSets
    exist to combine sets of <=8 Channel Assignments since SMuRF tuning and
    setup is run "per smurf band" while channel readout reads all tuned bands.
    Every TuneSet is a Tune file but not all Tunes are a Tuneset. This is
    because new Tunes can be made for the same set of channel assignments as the
    cryostat / readout environments evolve.
    
    Attributes
    ----------
    id : integer 
        primary key
    name : string
        name of tune file
    path : string
        absolute path of tune file
    stream_id : string
        stream_id for file
    start : datetime.datetime
        The time the tune file is made
    stop : datetime.datetine
        Not Implemented Yet
    files : list of SQLAlchemy File instances.
    tunes : list of Tunes that are part of the TuneSet
    observations : list of observations that have this TuneSet
    chan_assignments : list of Channel Assignments that are part of the this
        tuneset 
    channels : list of Channels that are part of this TuneSet
    """
    __tablename__ = 'tunesets'
    __table_args__ = (
        db.UniqueConstraint('name', 'stream_id'),
    )
    id = db.Column( db.Integer, primary_key=True)
    
    name = db.Column(db.String)
    path = db.Column(db.String)
    stream_id = db.Column(db.String)
    
    ## should stop exist? tune file use does not need to be continguous
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)
    
    ## files that use this detset
    ## one to many
    files = relationship("Files", back_populates='tuneset')
    tunes = relationship("Tunes", back_populates='tuneset')
    ## many to many
    observations = relationship("Observations", 
                                secondary=association_table_obs,
                                back_populates='tunesets')
    
    ## many to many
    chan_assignments = relationship('ChanAssignments', 
                                    secondary=association_table,
                                    back_populates='tunesets')
    
    ## many to many
    dets = relationship('Channels', 
                        secondary=association_table_dets,
                        back_populates='detsets')
    
    @property
    def channels(self):
        return self.dets
    
class Bands(Base):
    """
    Indexing Table (may be removed). One row per SMuRF band per slot.
    """
    __tablename__ = 'bands'
    __table_args__ = (
        db.UniqueConstraint('number', 'stream_id'),
    )
    id = db.Column( db.Integer, primary_key=True )
    number = db.Column( db.Integer )
    stream_id = db.Column( db.String)
    
    
class ChanAssignments(Base):
    """The available channel assignments. TuneSets are made of up to eight of
    these assignments. 

    Attributes
    ----------
    id : integer 
        primary key
    ctime : integer
        ctime where the channel assignment was made
    name : string
        name of the channel assignment file
    path : string
        absolute path of channel assignment file
    stream_id : string
        stream_id for file
    band : integer
        Smurf band for this channel assignment
    tunesets : list of SQLAlchemy tunesets
        The Tunesets the channel assignment is in
    channels : list of SQLAlchemy channels
        The channels in the channel assignment
    """
    __tablename__ = 'chan_assignments'
    id = db.Column(db.Integer, primary_key=True)
    
    ctime = db.Column(db.Integer)
    path = db.Column(db.String, unique=True)
    name = db.Column(db.String)
    stream_id = db.Column(db.String)
    band = db.Column(db.Integer)
    
    ## Channel Assignments are put into detector sets
    ## many to many bidirectional 
    tunesets = relationship('TuneSets', 
                           secondary=association_table,
                           back_populates='chan_assignments')

    ## Each channel assignment is made of many channels
    ## one to many
    channels = relationship("Channels", back_populates='chan_assignment')
    
class Channels(Base):
    """All the channels tracked by SMuRF indexed by the ctime of the channel
    assignment file, SMuRF band and channel number. Many channels will map to
    one detector on a UFM.

    
    Attributes
    ----------
    id : integer 
        primary key
    name : string
        name of of channel. In the form of sch_<ctime>_<band>_<channel>
    stream_id : string
        stream_id for file
    subband : integer
        The subband of the channel
    channel : integer
        The assigned smurf channel
    frequency : float
        The frequency of the resonator when the channel assigments were made
    band : integer
        The smurf band for the channel
    ca_id : integer
        The id of the channel assignment for the channel. Used for SQL mapping
    chan_assignment : SQLAlchemy ChanAssignments Instance
    detsets : List of SQLAlchemy Tunesets
        The tunesets the channel can be found in
    """
    __tablename__ = 'channels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    stream_id = db.Column(db.String)
    
    ## smurf channels
    subband = db.Column(db.Integer)
    channel = db.Column(db.Integer)
    frequency = db.Column(db.Float)

    band = db.Column(db.Integer)

    ## many to one
    ca_id = db.Column(db.Integer, db.ForeignKey('chan_assignments.id'))
    chan_assignment = relationship("ChanAssignments", back_populates='channels')

    
    ## many to many
    detsets = relationship('TuneSets',
                         secondary=association_table_dets,
                         back_populates='dets')
    
    
type_key = ['Observation', 'Wiring', 'Scan']


class FrameType(Base):
    """Enum table for storing frame types"""
    __tablename__ = "frame_type"
    id = db.Column(db.Integer, primary_key=True)
    type_name = db.Column(db.String, unique=True, nullable=True)

    
class Frames(Base):
    """Table to store frame indexing info
    Attributes
    ----------
    id : Integer
        Primary Key
    file_id : integer
        id of the file the frame is part of, used for SQL mapping
    file : SQLAlchemy instance
    frame_idx : integer
        frame index in the file
    offset : integer
        frame offset used by so3g indexed reader
    type_name : string
        frame types, use Observation, Wiring, and Scan frames
    time : datetime.datetime
        The start of the frame
    n_samples : integer
        the number of samples in the frame
    n_channels : integer
        the number of channels in the frame
    start : datetime.datetime
        the start of the frame
    stop : datetime.datetime
        the end of the frame
    """
    __tablename__ = 'frame_offsets'
    __table_args__ = (
        db.UniqueConstraint('file_id', 'frame_idx', name='_frame_loc'),
    )

    id = db.Column(db.Integer, primary_key=True)

    file_id = db.Column(db.Integer, db.ForeignKey('files.id'))
    file = relationship("Files", back_populates='frames')

    frame_idx = db.Column(db.Integer, nullable=False)
    offset = db.Column(db.Integer, nullable=False)

    type_name = db.Column(db.String, db.ForeignKey('frame_type.type_name'))
    frame_type = relationship('FrameType')

    time = db.Column(db.DateTime, nullable=False)

    # Specific to data frames
    n_samples = db.Column(db.Integer)
    n_channels = db.Column(db.Integer)
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)

    def __repr__(self):
        return f"Frame({self.type_name})<{self.frame_idx}>"


class TimingParadigm(Enum):
    G3Timestream = 1
    SmurfUnixTime = 2
    TimingSystem = 3
    Mixed = 4

def get_sample_timestamps(frame):
    """
    Gets timestamps of samples in a G3Frame. This will try to get highest
    precision first and move to lower precision methods if that fails.

    Args
    ------
        frame (spt3g_core.G3Frame):
            A G3Frame(Scan) containing streamed detector data.

    Returns
    ---------
        times (np.ndarray):
            numpy array containing timestamps in seconds
        paradigm (TimingParadigm):
            Paradigm used to calculate timestamps.
    """
    logger.warning("get_sample_timestamps is deprecated, how did you get here?")
    if 'primary' in frame.keys():
        if False:
            # Do high precision timing calculation here when we have real data
            pass
        else:
            # Try to calculate the timestamp based on the SmurfProcessor's
            # "UnixTime" and the G3Timestream start time.  "UnixTime" is a
            # 32-bit nanosecond clock that steadily increases mod 2**32.
            unix_times = np.array(frame['primary']['UnixTime'])
            for i in np.where(np.diff(unix_times) < 0)[0]:
                # This corrects for any wrap around
                unix_times[i+1:] += 2**32
            times = frame['data'].start.time / spt3g_core.G3Units.s \
                + (unix_times - unix_times[0]) / 1e9

            return times, TimingParadigm.SmurfUnixTime
    else:
        # Calculate timestamp based on G3Timestream.times(). Note that this
        # only uses the timestream start and end time, and assumes samples are
        # equispaced.
        times = np.array([t.time / spt3g_core.G3Units.s
                          for t in frame['data'].times()])
        return times, TimingParadigm.G3Timestream

class G3tSmurf:
    def __init__(self, archive_path, db_path=None, meta_path=None, 
                 echo=False):
        """
        Class to manage a smurf data archive.

        Args
        -----
            archive_path: path
                Path to the data directory
            db_path: path, optional
                Path to the sqlite file. Defaults to ``<archive_path>/frames.db``
            meta_path: path, optional
                Path of directory containing smurf related metadata (ie. channel
                assignments). Required for full functionality.
            echo: bool, optional
                If true, all sql statements will print to stdout.
        """
        if db_path is None:
            db_path = os.path.join(archive_path, 'frames.db')
        self.archive_path = archive_path
        self.meta_path = meta_path
        self.db_path = db_path
        self.engine = db.create_engine(f"sqlite:///{db_path}", echo=echo)
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        # Defines frame_types
        self._create_frame_types()

    def _create_frame_types(self):
        session = self.Session()
        if not session.query(FrameType).all():
            print("Creating FrameType table...")
            for k in type_key:
                ft = FrameType(type_name=k)
                session.add(ft)
            session.commit()

    @staticmethod
    def _make_datetime(x):
        if isinstance(x,float) or isinstance(x,int):
            return dt.datetime.fromtimestamp(x)
        return x


    def add_file(self, path, session):
        """
        Indexes a single file and adds it to the sqlite database. Creates a
        single entry in Files and as many Frame entries as there are frames in
        the file.

        Args
        ----
            path: path 
                Path of the file to index
            session : SQLAlchemy session
                Current, active sqlalchemy session
        """

        frame_types = {
            ft.type_name: ft for ft in session.query(FrameType).all()
        }

        db_file = Files(path=path)
        session.add(db_file)
        try:
            splits = path.split('/')
            db_file.stream_id = splits[-2]
            db_file.name = splits[-1]
        except:
            ## should this fail silently?
            pass
        
        reader = so3g.G3IndexedReader(path)

        total_channels = 0
        file_start, file_stop = None, None
        frame_idx = 0
        while True:
            
            try: 
                db_frame_offset = reader.Tell()
                frames = reader.Process(None)
                if not frames:
                    break
            except RuntimeError as e:
                logger.warning(f"Failed to add {path}: file likely corrupted")
                session.rollback()
                return
                
            frame = frames[0]
            frame_idx += 1

            if str(frame.type) not in type_key:
                continue

            db_frame_frame_type = frame_types[str(frame.type)]

            timestamp = frame['time'].time / spt3g_core.G3Units.s
            db_frame_time = dt.datetime.fromtimestamp(timestamp)
            
            ## only make Frame once the non-nullable fields are known
            db_frame = Frames(frame_idx=frame_idx, file=db_file,
                             offset = db_frame_offset,
                             frame_type = db_frame_frame_type,
                             time = db_frame_time 
                             )
        
            data = frame.get('data')
            if data is not None:
                db_frame.n_samples = data.n_samples
                db_frame.n_channels = len(data)
                db_frame.start = dt.datetime.fromtimestamp(data.start.time / spt3g_core.G3Units.s)
                db_frame.stop = dt.datetime.fromtimestamp(data.stop.time / spt3g_core.G3Units.s)

                if file_start is None:
                    file_start = db_frame.start
                file_stop = db_frame.stop
                total_channels = max(total_channels, db_frame.n_channels)

            session.add(db_frame)

        db_file.start = file_start
        db_file.stop = file_stop
        db_file.n_channels = total_channels
        db_file.n_frames = frame_idx


    def index_archive(self, verbose=False, stop_at_error=False,
                     skip_old_format=True):
        """
        Adds all files from an archive to the File and Frame sqlite tables.
        Files must be indexed before the metadata entries can be made.

        Args
        ----
        verbose: bool
            Verbose mode
        stop_at_error: bool
            If True, will stop if there is an error indexing a file.
        skip_old_format: bool
            If True, will skip over indexing files before the name convention
            was changed to be ctime_###.g3. 
        """
        session = self.Session()
        indexed_files = [f[0] for f in session.query(Files.path).all()]

        files = []
        for root, _, fs in os.walk(self.archive_path):
            for f in fs:
                path = os.path.join(root, f)
                if path.endswith('.g3') and path not in indexed_files:
                    if skip_old_format and '2020-' in path:
                        continue
                    files.append(path)

        if verbose:
            print(f"Indexing {len(files)} files...")

        for f in tqdm(sorted(files)[::-1]):
            try:
                self.add_file(os.path.join(root, f), session)
                session.commit()
            except IntegrityError as e:
                # Database Integrity Errors, such as duplicate entries
                session.rollback()
                print(e)
            except RuntimeError as e:
                # End of stream errors, for G3Files that were not fully flushed
                session.rollback()
                print(f"Failed on file {f} due to end of stream error!")
            except Exception as e:
                # This will catch generic errors such as attempting to load
                # out-of-date files that do not have the required frame
                # structure specified in the TOD2MAPS docs.
                session.rollback()
                if stop_at_error:
                    raise e
                elif verbose:
                    print(f"Failed on file {f}:\n{e}") 
        session.close()

    def add_new_channel_assignment(self, stream_id, ctime, cha, cha_path, session):   
        """Add new entry to the Channel Assignments table. Called by the
        index_metadata function.

        Args
        -------
        stream_id : string
            The stream id for the particular SMuRF slot
        ctime : int
            The ctime of the SMuRF action called to create the channel
            assignemnt
        cha : string
            The file name of the channel assignment
        cha_path : path
            The absolute path to the channel assignment
        session : SQLAlchemy Session
            The active session
        """
        band = int(re.findall('b\d.txt', cha)[0][1])   

        ch_assign = session.query(ChanAssignments).filter(ChanAssignments.ctime == ctime,
                                                          ChanAssignments.stream_id == stream_id,
                                                          ChanAssignments.band == band)
        ch_assign = ch_assign.one_or_none()
        if ch_assign is None:
            ch_assign = ChanAssignments(ctime=ctime,
                                        path=cha_path,
                                        name=cha,
                                        stream_id=stream_id,
                                        band=band)
            session.add(ch_assign)

        notches = np.atleast_2d(np.genfromtxt(ch_assign.path, delimiter=','))
        if np.sum([notches[:,2]!=-1]) != len(ch_assign.channels):
            ch_made = [c.channel for c in ch_assign.channels]
            for notch in notches:
                ## smurf did not assign a channel
                if notch[2] == -1:
                    continue
                if int(notch[2]) in ch_made:
                    continue
                    
                ch_name = 'sch_{:10d}_{:01d}_{:03d}'.format(ctime, band, int(notch[2]))
                ch = Channels(subband=int(notch[1]),
                              channel=int(notch[2]),
                              frequency=notch[0],
                              name=ch_name,
                              chan_assignment=ch_assign,
                              band=band)
                
                if ch.channel == -1:
                    logger.warning(f"Un-assigned channel made in Channel Assignment {ch_assign.name}")
                    continue
                check = session.query(Channels).filter( Channels.ca_id == ch_assign.id,
                                           Channels.channel == ch.channel).one_or_none()
                if check is None:
                    session.add(ch)
        session.commit()   
        
    def _assign_set_from_file(self, tune_path, ctime=None, stream_id=None, session=None):
        """Build set of Channel Assignments that are (or should be) in the 
        tune file.

        Args
        -------
        tune_path : path
            The absolute path to the tune file
        ctime : int
            ctime of SMuRF Action
        stream_id : string
            The stream id for the particular SMuRF slot
        session : SQLAlchemy Session
            The active session
        """

        if session is None:
            session = self.Session()
        if stream_id is None:
            stream_id = tune_path.split('/')[-4]
        if ctime is None:
            ctime = int( tune_path.split('/')[-1].split('_')[0] )
            
        data = np.load(tune_path, allow_pickle=True).item()
        assign_set = []

        ### Determine the TuneSet
        for band in data.keys():
            if 'resonances' not in data[band]:
                ### tune file doesn't have info for this band
                continue
            ## try to use tune file to find channel assignment before just assuming "most recent"
            if 'channel_assignment' in data[band]:
                cha_name = data[band]['channel_assignment'].split('/')[-1]
                cha = session.query(ChanAssignments).filter(ChanAssignments.stream_id==stream_id,
                                                      ChanAssignments.name==cha_name).one_or_none()
            else:
                cha = session.query(ChanAssignments).filter(ChanAssignments.stream_id==stream_id,
                                                           ChanAssignments.ctime <= ctime,
                                                           ChanAssignments.band==band)
                cha = cha.order_by(db.desc(ChanAssignments.ctime)).first()
            if cha is None:
                logger.error(f"Missing Channel Assignment for tune file {tune_path}")
            assign_set.append(cha)
        return assign_set


    def add_new_tuning(self, stream_id, ctime, tune_path, session):    
        """Add new entry to the TunesSet table. Called by the
        index_metadata function.

        Args
        -------
        stream_id : string
            The stream id for the particular SMuRF slot
        ctime : int
            The ctime of the SMuRF action called to create the tuning file.
        tune_path : path
            The absolute path to the tune file
        session : SQLAlchemy Session
            The active session
        """
        name = tune_path.split('/')[-1]
        tune = session.query(Tunes).filter(Tunes.name == name,
                                          Tunes.stream_id == stream_id).one_or_none()
        if tune is None:
            tune = Tunes(name=name, start=dt.datetime.fromtimestamp(ctime),
                           path=tune_path, stream_id=stream_id)
            session.add(tune)
            session.commit()

        ## assign set is the set of channel assignments that make up this tune file
        assign_set = self._assign_set_from_file(tune_path, ctime=ctime,
                                           stream_id=stream_id, session=session)

        tuneset = None
        tunesets = session.query(TuneSets)
        ## tunesets with any channel assignments matching list
        tunesets = tunesets.filter( TuneSets.chan_assignments.any(
                    ChanAssignments.id.in_([a.id for a in assign_set]))).all()

        if len(tunesets)>0:
            for ts in tunesets:
                if np.all( sorted([ca.id for ca in ts.chan_assignments]) == sorted([a.id for a in assign_set]) ):
                    tuneset = ts

        if tuneset is None:
            logger.debug(f"New Tuneset Detected {stream_id}, {ctime}, {[[a.name for a in assign_set]]}")
            tuneset = TuneSets(name=name, path=tune_path, stream_id=stream_id,
                               start=dt.datetime.fromtimestamp(ctime))
            session.add(tuneset)
            session.commit()

            ## add a the assignments and channels to the detector set
            for db_cha in assign_set:
                tuneset.chan_assignments.append(db_cha)
                for ch in db_cha.channels:
                    tuneset.channels.append(ch)
            session.commit()

        tune.tuneset = tuneset
        session.commit()
    
    def add_new_observation(self, stream_id, ctime, obs_data, session, max_early=5,max_wait=100):
        """Add new entry to the observation table. Called by the
        index_metadata function.
        
        Args
        -------
        stream_id : string
            The stream id for the particular SMuRF slot
        ctime : int
            The ctime of the SMuRF action called to create the tuning file.
        obs_data: list
            List of files that come with the observation. Currently un-used
        session : SQLAlchemy Session
            The active session
        max_early : int     
            Buffer time to allow the g3 file to be earlier than the smurf action
        max_wait : int  
            Maximum amount of time between the streaming start action and the 
            making of .g3 files that belong to an observation
        
        TODO: how will simultaneous streaming with two stream_ids work?
        """
        obs = session.query(Observations).filter(Observations.obs_id == str(ctime)).one_or_none()
        if obs is None:
            obs = Observations(obs_id = str(ctime),
                               timestamp = ctime,
                               start = dt.datetime.fromtimestamp(ctime))
            session.add(obs)
            session.commit()

        self.update_observation_files(obs, session, max_early=max_early,
                    max_wait=max_wait)

    def update_observation_files(self, obs, session, max_early=5, max_wait=100, force=False):
        """ Update existing observation. A separate function to make it easier
        to deal with partial data transfers. See add_new_observation for args
        
        Args
        -----
        max_early : int     
            Buffer time to allow the g3 file to be earlier than the smurf action
        max_wait : int  
            Maximum amount of time between the streaming start action and the 
            making of .g3 files that belong to an observation
        session : SQLAlchemy Session
            The active session
        force : bool
            If true, will recalculate file/tune information even if observation 
            appears complete
        """
        
        if not force and obs.duration is not None and len(obs.tunesets) >= 1:
            return

        x=session.query(Files.name).filter(Files.start >= obs.start-dt.timedelta(seconds=max_early))
        x = x.order_by(Files.start).first()
        if x is None:
            ## no files to add at this point
            session.commit()
            return
        x = x[0]
        f_start, f_num = (x[:-3]).split('_')
        if int(f_start)-obs.start.timestamp() > max_wait:
            ## we don't have .g3 files for some reason
            pass
        else:
            flist = session.query(Files).filter(Files.name.like(f_start+'%'))
            flist = flist.order_by(Files.start).all()
            
            ## Use Status information to set Tuneset
            status = SmurfStatus.from_file(flist[0].path)
            if status.action is not None:
                if status.action not in SMURF_ACTIONS['observations']:
                    logger.warning(f"Status Action {status.action} from file does not \
                                    match accepted observation types")
                if status.action_timestamp != obs.timestamp:
                    logger.error(f"Status timestamp {status.action_timestamp} from file does \
                                not match observation timestamp {obs.timestamp}")
                    return
                                   
            
            if status.tune is not None:
                tune = session.query(Tunes).filter( Tunes.name == status.tune).one_or_none()
                if tune is None:
                    logger.warning(f"Tune {status.tune} not found in database, update error?")
                    tuneset = None
                else:
                    tuneset = tune.tuneset
            else:
                tuneset = session.query(TuneSets).filter(TuneSets.start <= obs.start)
                tuneset = tuneset.order_by(db.desc(TuneSets.start)).first()
                
            already_have = [ts.id for ts in obs.tunesets]
            if tuneset is not None:
                if not tuneset.id in already_have:
                    obs.tunesets.append(tuneset)

            ## Update file entries
            for f in flist:
                f.obs_id = obs.obs_id
                if tuneset is not None:
                    f.detset = tuneset.name
            obs.duration = flist[-1].stop.timestamp() - obs.timestamp
            obs.stop = flist[-1].stop
        session.commit()

    def index_metadata(self, min_ctime=16000*1e5, stop_at_error=False):
        """
            Adds all channel assignments, tunefiles, and observations in archive to database. 
            Adding relevant entries to Bands and Files as well.

            Args
            ----
            min_ctime : int
                Lowest ctime to start looking for new metadata
            stop_at_error: bool
                If True, will stop if there is an error indexing a file.
        """

        if self.meta_path is None:
            raise ValueError('Archiver needs meta_path attribute to index channel assignments')

        session = self.Session()

        logger.info(f"Ignoring ctime folders below {int(min_ctime//1e5)}")
        for ct_dir in sorted(os.listdir(self.meta_path)):        
            if int(ct_dir) < int(min_ctime//1e5):
                continue

            for stream_id in sorted(os.listdir( os.path.join(self.meta_path, ct_dir))):
                action_path = os.path.join(self.meta_path, ct_dir, stream_id)
                actions = sorted(os.listdir( action_path ))

                for action in actions:
                    try:
                        ctime = int( action.split('_')[0] )
                        astring = '_'.join(action.split('_')[1:])
                        logger.debug(f"Found action {astring} at ctime {ctime}")

                        ### Look for channel assignments before tuning files
                        if astring in SMURF_ACTIONS['channel_assignments']:

                            cha = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(cha) == 0:
                                raise ValueError("found action {} with no output".format(action))

                            ## find last channel assignment in directory
                            cha= [f for f in cha if 'channel_assignment' in f]
                            if len(cha) == 0:
                                logger.debug(f"{action} run with no new channel assignment")
                                continue

                            cha = sorted(cha)[-1]
                            cha_path = os.path.join(action_path, action, 'outputs', cha)
                            cha_ctime = int(cha.split('_')[0])

                            logger.debug(f"Add new channel assignment: {stream_id}, {cha_ctime}, {cha_path}")
                            self.add_new_channel_assignment(stream_id, cha_ctime, cha, cha_path, session)     


                        ### Look for tuning files before observations
                        if astring in SMURF_ACTIONS['tuning']:
                            tune = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(tune) == 0:
                                raise ValueError("found action {} with no output".format(action))

                            ## find last tune in directory
                            tune = [f for f in tune if 'tune' in f]
                            if len(tune) > 1:
                                logger.warning(f"found multiple tune files in {stream_id}, {ctime}, {action}")
                            if len(tune) == 0:
                                logger.warning(f"found no tune files in {stream_id}, {ctime}, {action}")
                                continue

                            tune = sorted(tune)[-1]
                            tune_ctime = int(tune.split('_')[0])
                            tune_path = os.path.join(action_path, action, 'outputs', tune)

                            logger.debug(f"Add new Tune: {stream_id}, {ctime}, {tune_path}")
                            self.add_new_tuning(stream_id, tune_ctime, tune_path, session)

                        ### Add Observations
                        if astring in SMURF_ACTIONS['observations']:
                            obs_path = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(obs_path) == 0:
                                raise ValueError("found action {} with no output".format(action))
                            logger.debug(f"Add new Observation: {stream_id}, {ctime}, {obs_path}")
                            self.add_new_observation(stream_id, ctime, obs_path, session)

                    except ValueError as e:
                        logger.info(e, stream_id, ctime)
                        if stop_at_error:
                            raise(e)
                    except IntegrityError as e:
                        # Database Integrity Errors, such as duplicate entries
                        session.rollback()
                        logger.info(f"Integrity Error at {stream_id}, {ctime}")
                    except Exception as e:
                        logger.info(stream_id, ctime)
                        raise(e)
        session.close()


    def _stream_ids_in_range(self, start, end):
        """
        Returns a list of all stream-id's present in a given time range.
        Skips 'None' because those only contain G3PipelineInfo frames.

        Args
        -----
            start : timestamp or DateTime
                start time for data
            end :  timestamp or DateTime
                end time for data
        Returns
        --------
            stream_ids: List of stream ids.
        """
        session = self.Session()
        start = self._make_datetime(start)
        end = self._make_datetime(end)
        all_ids = session.query(Files.stream_id).filter(
            Files.start < end,
            Files.stop >= start
        ).all()
        sids = []
        for sid, in all_ids:
            if sid not in sids and sid != 'None':
                sids.append(sid)
        return sids

    def load_data(self, start, end, stream_id=None, channels=None,
                  show_pb=True, load_biases=True):
        """
        Loads smurf G3 data for a given time range. For the specified time range
        this will return a chunk of data that includes that time range.

        This function returns an AxisManager with the following properties    
            * Axes:
                * samps : samples 
                * channels : resonator channels reading out
                * bias_lines (optional) : bias lines

            * Fields:
                * timestamps : (samps,) 
                    unix timestamps for loaded data
                * signal : (channels, samps) 
                    Array of the squid phase in units of radians for each channel
                * primary : AxisManager (samps,)
                    "primary" data included in the packet headers
                    'AveragingResetBits', 'Counter0', 'Counter1', 'Counter2', 
                    'FluxRampIncrement', 'FluxRampOffset', 'FrameCounter', 
                    'TESRelaySetting', 'UnixTime'
                * biases (optional): (bias_lines, samps)
                    Bias values during the data
                * ch_info : AxisManager (channels,)
                    Information about channels, including SMuRF band, channel, 
                    frequency.
        
        Args
        -----
            start : timestamp or DateTime
                start time for data
            end :  timestamp or DateTime
                end time for data
            stream_id : String 
                stream_id to load, in case there are multiple
            channels : list or None
                If not None, it should be a list that can be sent to get_channel_mask.
            detset : string
                the name of the detector set (tuning file) to load
            show_pb : bool, optional: 
                If True, will show progress bar.
            load_biases : bool, optional 
                If True, will return biases.

        Returns
        --------
            aman : AxisManager
                AxisManager for the data


        TODO: What to do with:
                1) status (SmurfStatus):
                    SmurfStatus object containing metadata info at the time of
                    the first Scan frame in the requested interval. If there
                    are no Scan frames in the interval, this will be None.
                2) timing_paradigm(TimingParadigm):
                    Tells you the method used to extract timestamps from the
                    frame data.
                    
        TODO: Track down differences between detector sets (tune files) and 
                information in status object
        """
        session = self.Session()
        start = self._make_datetime(start)
        end = self._make_datetime(end)
        
        if stream_id is None:
            sids = self._stream_ids_in_range(start, end)
            if len(sids) > 1:
                raise ValueError(
                    "Multiple stream_ids exist in the given range! "
                    "Must choose one.\n"
                    f"stream_ids: {sids}"
                )

        q = session.query(Files.path).join(Frames).filter(Frames.stop >= start,
                                                  Frames.start < end,
                                                  Frames.type_name=='Scan')
        if stream_id is not None:
            q.filter(Files.stream_id == stream_id)

        q = q.order_by(Files.start).distinct()
        flist = [x[0] for x in q.all()]
        
        scan_start = session.query(Frames.start).filter(Frames.start > start,
                                                        Frames.type_name=='Scan')
        scan_start = scan_start.order_by(Frames.start).first()
        
        try:
            status = self.load_status(scan_start[0])
        except:
            logger.info("Status load from database failed, using file load")
            status = None
        
        aman = load_file( flist, status=status, channels=channels,
                         archive=self, show_pb=show_pb)
        
        msk = np.all([aman.timestamps >= start.timestamp(),
                      aman.timestamps < end.timestamp()], axis=0)
        idx = np.where(msk)[0]
        if len(idx) == 0:
            logger.warning("No samples returned in time range")
            aman.restrict('samps', (0, 0))
        else:
            aman.restrict('samps', (idx[0], idx[-1]))
        session.close()
        
        return aman

    def load_status(self, time, show_pb=False):
        """
        Returns the status dict at specified unix timestamp.
        Loads all status frames between session start frame and specified time.

        Args:
            time (timestamp): Time at which you want the rogue status

        Returns:
            status (SmurfStatus instance): object indexing of rogue variables 
            at specified time.
        """
        return SmurfStatus.from_time(time, self, show_pb=show_pb)

def dump_DetDb(archive, detdb_file):
    """
    Take a G3tSmurf archive and create a a DetDb of the type used with Context
    
    Args
    -----
        archive : G3tSmurf instance
        detdb_file : filename
    """
    my_db = core.metadata.DetDb(map_file=detdb_file)
    my_db.create_table('base', column_defs=[])
    column_defs = [
        "'band' int",
        "'channel' int",
        "'frequency' float",
        "'chan_assignment' int",
    ]
    my_db.create_table('smurf', column_defs=column_defs)
    
    ddb_list = my_db.dets()['name']
    session = archive.Session()
    channels = session.query(Channels).all()
    msk = np.where([ch.name not in ddb_list for ch in channels])[0].astype(int)
    for ch in tqdm(np.array(channels)[msk]):
        my_db.get_id( name=ch.name )
        my_db.add_props('smurf', ch.name, band=ch.band, 
                        channel=ch.channel, frequency=ch.frequency,
                        chan_assignment=ch.chan_assignment.ctime)
    session.close()
    return my_db

class SmurfStatus:
    """
    This is a class that attempts to extract essential information from the
    SMuRF status dictionary so it is more easily accessible. If the necessary
    information for an attribute is not present in the dictionary, the
    attribute will be set to None.

    Args
    -----
        status  : dict
            A SMuRF status dictionary

    Attributes
    ------------
        status : dict
            Full smurf status dictionary
        num_chans: int
            Number of channels that are streaming
        mask : Optional[np.ndarray]
            Array with length ``num_chans`` that describes the mapping
            of readout channel to absolute smurf channel.
        mask_inv : np.ndarray
            Array with dimensions (NUM_BANDS, CHANS_PER_BAND) where
            ``mask_inv[band, chan]`` tells you the readout channel for a given
            band, channel combination.
        freq_map : Optional[np.ndarray]
            An array of size (NUM_BANDS, CHANS_PER_BAND) that has the mapping
            from (band, channel) to resonator frequency. If the mapping is not
            present in the status dict, the array will full of np.nan.
        filter_a : Optional[np.ndarray]
            The A parameter of the readout filter.
        filter_b : Optional[np.ndarray]
            The B parameter of the readout filter.
        filter_gain : Optional[float]
            The gain of the readout filter.
        filter_order : Optional[int]
            The order of the readout filter.
        filter_enabled : Optional[bool]
            True if the readout filter is enabled.
        downsample_factor : Optional[int]
            Downsampling factor
        downsample_enabled : Optional[bool]
            Whether downsampler is enabled
        flux_ramp_rate_hz : float
            Flux Ramp Rate calculated from the RampMaxCnt and the digitizer
            frequency.
    """
    NUM_BANDS = 8
    CHANS_PER_BAND = 512

    def __init__(self, status):
        self.status = status
        self.start = self.status.get('start')
        self.stop = self.status.get('stop')
        
        # Reads in useful status values as attributes
        mapper_root = 'AMCc.SmurfProcessor.ChannelMapper'
        self.num_chans = self.status.get(f'{mapper_root}.NumChannels')
        
        # Tries to set values based on expected rogue tree
        self.mask = self.status.get(f'{mapper_root}.Mask')
        self.mask_inv = np.full((self.NUM_BANDS, self.CHANS_PER_BAND), -1)
        if self.mask is not None:
            self.mask = np.array(ast.literal_eval(self.mask))

            # Creates inverse mapping
            for i, chan in enumerate(self.mask):
                b = chan // self.CHANS_PER_BAND
                c = chan % self.CHANS_PER_BAND
                self.mask_inv[b, c] = i

        tune_root = 'AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.tuneFilePath'        
        self.tune = self.status.get(tune_root)
        if self.tune is not None and len(self.tune)>0:
            self.tune = self.tune.split('/')[-1]
        
        pysmurf_root = 'AMCc.SmurfProcessor.SOStream'
        self.action = self.status.get(f'{pysmurf_root}.pysmurf_action')
        if self.action == '':
            self.action = None
        self.action_timestamp = self.status.get(f'{pysmurf_root}.pysmurf_action_timestamp')
        if self.action_timestamp == 0:
            self.action_timestamp = None
        
        filter_root = 'AMCc.SmurfProcessor.Filter'
        self.filter_a = self.status.get(f'{filter_root}.A')
        if self.filter_a is not None:
            self.filter_a = np.array(ast.literal_eval(self.filter_a))
        self.filter_b = self.status.get(f'{filter_root}.B')
        if self.filter_b is not None:
            self.filter_b = np.array(ast.literal_eval(self.filter_b))
        self.filter_gain = self.status.get(f'{filter_root}.Gain')
        self.filter_order = self.status.get(f'{filter_root}.Order')
        self.filter_enabled = not self.status.get('{filter_root}.Disabled')

        ds_root = 'AMCc.SmurfProcessor.Downsampler'
        self.downsample_factor = self.status.get(f'{ds_root}.Factor')
        self.downsample_enabled = not self.status.get(f'{ds_root}.Disabled')

        # Tries to make resonator frequency map
        self.freq_map = np.full((self.NUM_BANDS, self.CHANS_PER_BAND), np.nan)
        band_roots = [
            f'AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[{band}]'
            for band in range(self.NUM_BANDS)]
        for band in range(self.NUM_BANDS):
            band_root = band_roots[band]
            band_center = self.status.get(f'{band_root}.bandCenterMHz')
            subband_offset = self.status.get(f'{band_root}.toneFrequencyOffsetMHz')
            channel_offset = self.status.get(f'{band_root}.CryoChannels.centerFrequencyArray')

            # Skip band if one of these fields is None
            if None in [band_center, subband_offset, channel_offset]:
                continue

            subband_offset = np.array(ast.literal_eval(subband_offset))
            channel_offset = np.array(ast.literal_eval(channel_offset))
            self.freq_map[band] = band_center + subband_offset + channel_offset

        # Calculates flux ramp reset rate (Pulled from psmurf's code)
        rtm_root = 'AMCc.FpgaTopLevel.AppTop.AppCore.RtmCryoDet'
        ramp_max_cnt = self.status.get(f'{rtm_root}.RampMaxCnt')
        if ramp_max_cnt is None:
            self.flux_ramp_rate_hz = None
        else:
            digitizer_freq_mhz = float(self.status.get(
                f'{band_roots[0]}.digitizerFrequencyMHz', 614.4))
            ramp_max_cnt_rate_hz = 1.e6*digitizer_freq_mhz / 2.
            self.flux_ramp_rate_hz = ramp_max_cnt_rate_hz / (ramp_max_cnt + 1)

    @classmethod
    def from_file(cls, filename):
        """Generates a Smurf Status from a .g3 file.
    
        Args
        ----
            filename : str or list
        """
        if isinstance(filename, str):
            filenames = [filename]
        else:
            filenames = filename
        status = {}
        for file in filenames:
            reader = so3g.G3IndexedReader(file)
            while True:
                frames = reader.Process(None)
                if len(frames) == 0:
                    break
                frame = frames[0]
                if str(frame.type) == 'Wiring':
                    if status.get('start') is None:
                        status['start'] = frame['time'].time/spt3g_core.G3Units.s
                        status['stop'] = frame['time'].time/spt3g_core.G3Units.s
                    else:
                        status['stop'] = frame['time'].time/spt3g_core.G3Units.s
                    status.update(yaml.safe_load(frame['status']))
                    if frame['dump']:
                        break
        return cls(status)
    
    @classmethod
    def from_time(cls, time, archive, show_pb=False):
        """Generates a Smurf Status at specified unix timestamp.
        Loads all status frames between session start frame and specified time.

        Args
        -------
            time : (timestamp) 
                Time at which you want the rogue status
            archive : (G3tSmurf instance)
                The G3tSmurf archive to use to find the status
            show_pb : (bool)
                Turn on or off loading progress bar

        Returns
        --------
            status : (SmurfStatus instance)
                object indexing of rogue variables at specified time.
        """
        time = archive._make_datetime(time)
        session = archive.Session()
        session_start,  = session.query(Frames.time).filter(
            Frames.type_name == 'Observation',
            Frames.time <= time
        ).order_by(Frames.time.desc()).first()

        status_frames = session.query(Frames).filter(
            Frames.type_name == 'Wiring',
            Frames.time >= session_start,
            Frames.time <= time
        ).order_by(Frames.time)

        status = {
            'start':status_frames[0].time.timestamp(),
            'stop':status_frames[-1].time.timestamp(),
        }
        cur_file = None
        for frame_info in tqdm(status_frames.all(), disable=(not show_pb)):
            file = frame_info.file.path
            if file != cur_file:
                reader = so3g.G3IndexedReader(file)
                cur_file = file
            reader.Seek(frame_info.offset)
            frame = reader.Process(None)[0]
            status.update(yaml.safe_load(frame['status']))

        return cls(status)
        
    def readout_to_smurf(self, rchan):
        """
        Converts from a readout channel number to (band, channel).

        Args
        -----
            rchans : int or List[int]
                Readout channel to convert. If a list or array is passed,
                this will return an array of bands and array of smurf channels.

        Returns
        --------
            band, channel : (int, int) or (List[int], List[int])
                The band, channel combination that is has readout channel
                ``rchan``.
        """
        abs_smurf_chan = self.mask[rchan]
        return (abs_smurf_chan // self.CHANS_PER_BAND,
                abs_smurf_chan % self.CHANS_PER_BAND)

    def smurf_to_readout(self, band, chan):
        """
        Converts from (band, channel) to a readout channel number.
        If the channel is not streaming, returns -1.

        Args:
            band : int, List[int]
                The band number, or list of band numbers corresopnding to
                channel input array.
            chan : int, List[int]
                Channel number or list of channel numbers.
        """
        return self.mask_inv[band, chan]

def get_channel_mask(ch_list, status, archive=None, obsfiledb=None,
                     ignore_missing=True):
    """Take a list of desired channels and parse them so the different
    data loading functions can load them.
    
    Args
    ------
    ch_list : list
        List of desired channels the type of each list element is used
        to determine what it is:

        * int : absolute readout channel
        * (int, int) : band, channel
        * string : channel name (archive can not be None)
        * float : frequency in the smurf status (or should we use channel assignment?)

    status : SmurfStatus instance
        Status to use to generate channel loading mask
    archive : G3tSmurf instance
        Archive used to search for channel names / frequencies
    obsfiledb : ObsFileDb instance
        ObsFileDb used to search for channel names if archive is None
    ignore_missing : bool
        If true, will not raise errors if a requested channel is not found
    
    Returns
    -------
    mask : bool array
        Mask for the channels in the SmurfStatus
        
    TODO: When loading from name, need to check tune file in use during file. 
    """
    if status.mask is None:
        raise ValueError("Status Mask not set")
    
    session = None
    if archive is not None:
        session = archive.Session()
        
    msk = np.zeros( (status.num_chans,), dtype='bool')
    for ch in ch_list:
        if np.isscalar(ch):
            if np.issubdtype( type(ch), np.integer):
                #### this is an absolute readout channel
                if not ignore_missing and ~np.any(status.mask == ch):
                    raise ValueError(f"channel {ch} not found")
                msk[ status.mask == ch] = True
            
            elif np.issubdtype( type(ch), np.floating):
                #### this is a resonator frequency
                b,c = np.where( np.isclose(status.freq_map, ch, rtol=1e-7) )
                if len(b)==0:
                    if not ignore_missing:
                        raise ValueError(f"channel {ch} not found")
                    continue
                elif status.mask_inv[b,c][0]==-1:
                    if not ignore_missing:
                        raise ValueError(f"channel {ch} not streaming")
                    continue
                msk[status.mask_inv[b,c][0]] = True
                
            elif np.issubdtype( type(ch), np.str_):
                #### this is a channel name
                if session is not None:
                    channel = session.query(Channels).filter(Channels.name==ch).one_or_none()
                    if channel is None:
                        if not ignore_mission:
                            raise ValueError(f"channel {ch} not found in G3tSmurf Archive")
                        continue
                    b,c = channel.band, channel.channel
                elif obsfiledb is not None:
                    c = obsfiledb.conn.execute('select band,channel from channels where name=?',(ch,))
                    c = [(r[0],r[1]) for r in c]
                    if len(c) == 0:
                        if not ignore_mission:
                            raise ValueError(f"channel {ch} not found in obsfiledb")
                        continue
                    b,c = c[0]
                else:
                    raise ValueError("Need G3tSmurf Archive or Obsfiledb to pass channel names")
                
                idx = status.mask_inv[b,c]
                if idx == -1:
                    if not ignore_missing:
                        raise ValueError(f"channel {ch} not streaming")
                    continue
                msk[idx] = True
                
            else:
                raise TypeError(f"type {type(ch)} for channel {ch} not understood")
        else:
            if len(ch) == 2:
                ### this is a band, channel pair
                idx = status.mask_inv[ch[0], ch[1]]
                if idx == -1:
                    if not ignore_missing:
                        raise ValueError(f"channel {ch} not streaming")
                    continue
                msk[idx] = True
            else:
                raise TypeError(f"type for channel {ch} not understood")
    if session is not None:
        session.close()
    return msk

def _get_tuneset_channel_names(status, ch_map, archive):
    """Update channel maps with name from Tuneset
    """
    session = archive.Session()
    
    ## tune file in status
    if status.tune is not None and len(status.tune) > 0:
        tune_file = status.tune.split('/')[-1]
        tune = session.query(Tunes).filter(Tunes.name == tune_file).one_or_none()
        if tune is None :
            logger.info(f"Tune file {tune_file} not found in G3tSmurf archive")
            return ch_map
        if tune.tuneset is None:
            logger.info(f"Tune file {tune_file} has no TuneSet in G3tSmurf archive")
            return ch_map
    else:
        logger.info("Tune information not in SmurfStatus, using most recent Tune")
        tune = session.query(Tunes).filter(Tunes.start <= dt.datetime.utcfromtimestamp(status.start))
        tune = tune.order_by(db.desc(Tunes.start)).first()
        if tune.tuneset is None:
            logger.info(f"Tune file {tune.name} has no TuneSet in G3tSmurf archive")
            return ch_map
    
    for ch in tune.tuneset.channels:
        try:
            band, channel = ch.band, ch.channel
            msk = np.all( [ch_map['band']== band, ch_map['channel']==channel], axis=0)
            i = np.where(msk)[0][0]
            ch_map[i]['name'] = ch.name
        except:
            logger.info(f"Information retrival error for Detector {ch.name}")
            continue    
    session.close()
    return ch_map

def _get_detset_channel_names(status, ch_map, obsfiledb):
    """Update channel maps with name from obsfiledb
    """
    ## tune file in status
    if status.tune is not None and len(status.tune) > 0:
        c =  obsfiledb.conn.execute('select det from detsets '
                    'where name=?', (status.tune,))
        detsets = [r[0] for r in c]
    else:
        logger.info("Tune information not in SmurfStatus, using most recent Tune")
        c = obsfiledb.conn.execute('select tuneset_id from tunes '
                            'where start<=? '
                            'order by start desc', (dt.datetime.utcfromtimestamp(status.start),))
        tuneset_id = [r[0] for r in c][0]

        c = obsfiledb.conn.execute('select name from tunesets '
                                   'where id=?', (tuneset_id,))
        tuneset = [r[0] for r in c][0]

        c = obsfiledb.conn.execute('select det from detsets '
                            'where name=?', (tuneset,))
        detsets = [r[0] for r in c]
        
    for det in detsets:
        try:
            c = obsfiledb.conn.execute('select band,channel from channels '
                                      'where name=?', (det,))
            band, channel = [(r[0], r[1]) for r in c][0]
            msk = np.all( [ch_map['band']== band, ch_map['channel']==channel], axis=0)
            i = np.where(msk)[0][0]
            ch_map[i]['name'] = det
        except:
            logger.info(f"Information retrival error for Detector {det}")
            continue        
    return ch_map

def _get_channel_mapping(status, ch_map):
    """Generate baseline channel map from status object
    """
    for i, ch in enumerate(ch_map['idx']):
        try:
            sch = status.readout_to_smurf( ch )
            ch_map[i]['rchannel'] = 'r{:04d}'.format(ch)
            ch_map[i]['name'] = 'sch_NONE_{}_{:03d}'.format(sch[0],sch[1])
            ch_map[i]['freqs']= status.freq_map[sch[0], sch[1]]
            ch_map[i]['band'] = sch[0]
            ch_map[i]['channel'] = sch[1]
        except:
            ch_map[i]['rchannel'] = 'r{:04d}'.format(ch)
            ch_map[i]['name'] = 'rch_{:04d}'.format(ch)
            ch_map[i]['freqs']= -1
            ch_map[i]['band'] = -1
            ch_map[i]['channel'] = -1        
    return ch_map

def get_channel_info(status, mask=None, archive=None, obsfiledb=None):
    """Create the Channel Info Section of a G3tSmurf AxisManager
    
    This function returns an AxisManager with the following properties    
        * Axes:
            * channels : resonator channels reading out

        * Fields:
            * band : Smurf Band
            * channel : Smurf Channel
            * frequency : resonator frequency
            * rchannel : readout channel
    
    Args
    -----
    status : SmurfStatus instance
    mask : bool array
        mask of which channels to use
    archive : G3tSmurf instance (optionl)
        G3tSmurf instance for looking for tunes/tunesets
    obsfiledb : ObsfileDb instance (optional)
        ObsfileDb instance for det names / band / channel
        
    Returns
    --------
    ch_info : AxisManager
    
    """
    ch_list = np.arange( status.num_chans )
    if mask is not None:
        ch_list = ch_list[mask]
    
    ch_map = np.zeros( len(ch_list), dtype = [('idx', int), ('name', np.unicode_,30), 
                                              ('rchannel', np.unicode_,30), ('band', int),
                                             ('channel', int), ('freqs', float)])
    ch_map['idx'] = ch_list
    
    ch_map = _get_channel_mapping(status, ch_map)
    
    if archive is not None:
        ch_map = _get_tuneset_channel_names(status, ch_map, archive)
    elif obsfiledb is not None:
        ch_map = _get_detset_channel_names(status, ch_map, obsfiledb)
        
    ch_info = core.AxisManager( core.LabelAxis('channels', ch_map['name']),)
    ch_info.wrap('band', ch_map['band'], ([(0,'channels')]) )
    ch_info.wrap('channel', ch_map['channel'], ([(0,'channels')]) )
    ch_info.wrap('frequency', ch_map['freqs'], ([(0,'channels')]) )
    ch_info.wrap('rchannel', ch_map['rchannel'], ([(0,'channels')]) )
    
    return ch_info

def _get_timestamps(streams, load_type=None):
    """Calculate the timestamp field for loaded data
    
    Args
    -----
        streams : dictionary
            result from unpacking the desired data frames
        load_type : None or int
            if None, uses highest precision version possible. integer values will
            use the TimingParadigm class for indexing
    """
    if load_type is None:
        ## determine the desired loading type. Expand as logic as
        ## data fields develop
        if 'primary' in streams:
            if 'UnixTime' in streams['primary']:
                load_type = TimingParadigm.SmurfUnixTime
            else:
                load_type = TimingParadigm.G3Timestream
        else:
            load_type = TimingParadigm.G3Timestream
    
    if load_type == TimingParadigm.SmurfUnixTime:
        return io_load.hstack_into(None, streams['primary']['UnixTime'])/1e9
    if load_type == TimingParadigm.G3Timestream:
        return io_load.hstack_into(None, streams['time'])
    logger.error("Timing System could not be determined")
            
        
def load_file(filename, channels=None, ignore_missing=True, 
             load_biases=True, load_primary=True, status=None,
             archive=None, obsfiledb=None, show_pb=True):
    """Load data from file where there may not be a connected archive.

    Args
    ----
      filename : str or list 
          A filename or list of filenames (to be loaded in order).
          Note that SmurfStatus is only loaded from the first file
      channels: list or None
          If not None, it should be a list that can be sent to get_channel_mask.
      ignore_missing : bool
          If true, will not raise errors if a requested channel is not found
      load_biases : bool
          If true, will load the bias lines for each detector
      load_primary : bool
          If true, loads the primary data fields, old .g3 files may not have 
          these fields. 
      archive : a G3tSmurf instance (optional)
      obsfiledb : a ObsFileDb instance (optional, used when loading from context)
      status : a SmurfStatus Instance we don't want to use the one from the 
          first file
      det_axis : name of the axis used for channels / detectors

    Returns
    ---------
      aman : AxisManager
        AxisManager with the data with axes for `channels` and `samps`. It will
        always have fields `timestamps`, `signal`, `flags`(FlagManager),
        `ch_info` (AxisManager with `bands`, `channels`, `frequency`, etc).
    """   
    
    if isinstance(filename, str):
        filenames = [filename]
    else:
        filenames = filename
    
    if status is None:
        status = SmurfStatus.from_file(filenames[0])

    if channels is not None:
        ch_mask = get_channel_mask(channels, status, archive=archive, 
                                   obsfiledb=obsfiledb,
                                   ignore_missing=ignore_missing)
    else:
        ch_mask = None

    ch_info = get_channel_info(status, ch_mask, archive=archive, 
                                   obsfiledb=obsfiledb)

    subreq = [
        io_load.FieldGroup('data', ch_info.rchannel, timestamp_field='time'),
    ]
    if load_primary:
        subreq.extend( [io_load.FieldGroup('primary', [io_load.Field('*', wildcard=True)])] )
    if load_biases:
        subreq.extend( [io_load.FieldGroup('tes_biases', [io_load.Field('*', wildcard=True)]),])

    request = io_load.FieldGroup('root', subreq)
    streams = None
    try:
        for filename in tqdm( filenames , total=len(filenames), disable=(not show_pb)):
            streams = io_load.unpack_frames(filename, request, streams=streams)
    except KeyError:
        logger.error("Frames do not contain expected fields. Did Channel Mask change during the file?")
        raise
        
    count = sum(map(len,streams['time']))

    ## Build AxisManager
    aman = core.AxisManager(
        ch_info.channels.copy(),
        core.OffsetAxis('samps', count, 0)
    )
    aman.wrap( 'timestamps', _get_timestamps(streams), ([(0,'samps')]))

    # Conversion from DAC counts to squid phase
    aman.wrap( 'signal', np.zeros(aman.shape, 'float32'),
                 [(0, 'channels'), (1, 'samps')])
    for idx in range(aman.channels.count):
        io_load.hstack_into(aman.signal[idx], streams['data'][ch_info.rchannel[idx]])

    rad_per_count = np.pi / 2**15
    aman.signal *= rad_per_count

    aman.wrap('ch_info', ch_info)

    temp = core.AxisManager( aman.samps.copy() )
    for k in streams['primary'].keys():
        temp.wrap( k, io_load.hstack_into(None, streams['primary'][k]), ([(0,'samps')]) )
    aman.wrap('primary', temp)

    if load_biases:
        bias_axis = core.LabelAxis('bias_lines', np.arange(len(streams['tes_biases'].keys())))
        aman.wrap('biases', np.zeros((bias_axis.count, aman.samps.count)), 
                          [ (0,bias_axis), 
                            (1,'samps')])
        for k in streams['tes_biases'].keys():
            i = int(k[4:])
            io_load.hstack_into(aman.biases[i], streams['tes_biases'][k])
    aman.wrap('flags', core.FlagManager.for_tod(aman, 'channels', 'samps'))

    return aman

def load_g3tsmurf_obs(db, obs_id, dets=None):
    c = db.conn.execute('select path from files '
                    'where obs_id=?' +
                    'order by start', (obs_id,))
    flist = [row[0] for row in c]
    return load_file(flist, dets, obsfiledb=db)


io_load.OBSLOADER_REGISTRY['g3tsmurf'] = load_g3tsmurf_obs
