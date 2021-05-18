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
    db.Column('tunes', db.Integer, db.ForeignKey('tunes.id')),
    db.Column('chan_assignments', db.Integer, db.ForeignKey('chan_assignments.id'))
)

association_table_obs = db.Table('association_obs', Base.metadata,
    db.Column('tunes', db.Integer, db.ForeignKey('tunes.id')),
    db.Column('observations', db.Integer, db.ForeignKey('obs.obs_id'))
)

association_table_dets = db.Table('detsets', Base.metadata,
    db.Column('name', db.Integer, db.ForeignKey('tunes.name')),
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
        'setup_notches'
    ]
}


class Observations(Base):
    """Times on continuous detector readout
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
    tunes = relationship("Tunes", 
                           secondary=association_table_obs,
                           back_populates='observations')

    def __repr__(self):
        try:
            return f"{self.obs_id}: {self.start} -> {self.stop} [{self.stop-self.start}] ({self.tag})"
        except:
            return f"{self.obs_id}: {self.start} -> {self.stop} ({self.tag})"

    

class Files(Base):
    """Table to store file indexing info"""
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
    detset = db.Column(db.String, db.ForeignKey('tunes.name'))
    tune = relationship("Tunes", back_populates='files')
    

class Tunes(Base):
    """Indexing of 'detector sets' available during observations. Should
    correspond to the tune files. 
    """
    __tablename__ = 'tunes'
    id = db.Column( db.Integer, primary_key=True)
    
    name = db.Column(db.String, unique=True)
    path = db.Column(db.String)
    stream_id = db.Column(db.String)
    
    ## should stop exist? tune file use does not need to be continguous
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)
    
    ## files that use this detset
    ## one to many
    files = relationship("Files", back_populates='tune')
    
    ## many to many
    observations = relationship("Observations", 
                                secondary=association_table_obs,
                                back_populates='tunes')
    
    ## many to many
    chan_assignments = relationship('ChanAssignments', 
                                    secondary=association_table,
                                    back_populates='tunes')
    
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
    """The available channel assignments. Tune files are made of up to eight of
    these assignments. 
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
    tunes = relationship('Tunes', 
                           secondary=association_table,
                           back_populates='chan_assignments')

    ## Each channel assignment is made of many channels
    ## one to many
    channels = relationship("Channels", back_populates='chan_assignment')
    
class Channels(Base):
    """All the channels tracked by SMuRF indexed by the ctime of the channel
    assignment file, SMuRF band and channel number. Many channels will map to
    one detector on a UFM.
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
    detsets = relationship('Tunes',
                         secondary=association_table_dets,
                         back_populates='dets')
    
    
type_key = ['Observation', 'Wiring', 'Scan']


class FrameType(Base):
    """Enum table for storing frame types"""
    __tablename__ = "frame_type"
    id = db.Column(db.Integer, primary_key=True)
    type_name = db.Column(db.String, unique=True, nullable=True)

    
class Frames(Base):
    """Table to store frame indexing info"""
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
            
            db_frame_offset = reader.Tell()
            frames = reader.Process(None)
            if not frames:
                break
                
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
        band_number = int(re.findall('b\d.txt', cha)[0][1])   
        band = session.query(Bands).filter(Bands.number == band_number,
                                           Bands.stream_id == stream_id).one_or_none()
        if band is None:
            band = Bands(number = band_number,
                         stream_id = stream_id)
            session.add(band)

        ch_assign = session.query(ChanAssignments).filter(ChanAssignments.ctime == ctime,
                                                      ChanAssignments.band_id == band.id)
        ch_assign = ch_assign.one_or_none()
        if ch_assign is None:
            ch_assign = ChanAssignments(ctime=ctime,
                                        path=cha_path,
                                        band=band)
            session.add(ch_assign)

        notches = np.atleast_2d(np.genfromtxt(ch_assign.path, delimiter=','))
        if len(notches) != len(ch_assign.channels):
            for notch in notches:
                ch_name = 'sch_{:10d}_{:01d}_{:03d}'.format(ctime, band.number, int(notch[2]))
                ch = Channels(subband=notch[1],
                              channel=notch[2],
                              frequency=notch[0],
                              name=ch_name,
                              chan_assignment=ch_assign,
                              band=band)
                ## smurf did not assign a channel
                if ch.channel == -1:
                    continue
                check = session.query(Channels).filter( Channels.ca_id == ch_assign.id,
                                           Channels.channel == ch.channel).one_or_none()
                if check is None:
                    session.add(ch)
        session.commit()   
        
    def add_new_tuning(self, stream_id, ctime, tune_path, session):    
        """Add new entry to the Tuness table. Called by the
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
        tune = session.query(Tunes).filter(Tunes.name == name).one_or_none()
        if tune is None:
            tune = Tunes(name=name, start=dt.datetime.fromtimestamp(ctime),
                           path=tune_path, stream_id=stream_id)
            session.add(tune)
            session.commit()

        data = np.load(tune_path, allow_pickle=True).item()
        if len(tune.chan_assignments) != len(data):
            ## we are missing channel assignments for at least one band
            have_already = [cha.band.number for cha in tune.chan_assignments]

            for band in data.keys():
                if band in have_already:
                    continue
                if 'resonances' not in data[band]:
                    ### tune file doesn't have info for this band
                    continue

                db_Band = session.query(Bands).filter(Bands.stream_id == stream_id, 
                                  Bands.number==band).one_or_none()
                if db_Band is None:
                    raise ValueError("Unable to find Band that should exist")

                ##################################################################################
                ## Section waiting on Pysmurf Upgrade tracking channel assignments in tuning files
                ##################################################################################
                ## For now we assume the most recent channel assignment for the band

                db_cha = session.query(ChanAssignments).filter(ChanAssignments.band_id == db_Band.id,
                                                               ChanAssignments.ctime <= ctime )
                db_cha = db_cha.order_by(db.desc(ChanAssignments.ctime)).first()
                if db_cha is None:
                    raise ValueError("Unable to find Channel Assignment that should exist")
                in_cha_db = [sorted( [ch.channel for ch in db_cha.channels])]              
                in_tune_file =[sorted( [data[band]['resonances'][x]['channel'] for x in  data[band]['resonances'].keys()])]

                ## Check to make sure the most recent channel assignment matches the tuning file
                if not np.all( in_cha_db == in_tune_file ):
                    ### logic left in just in case. If the first channel assignment doesn't match, try the 
                    ### later ones
                    db_cha = session.query(ChanAssignments).filter(ChanAssignments.band_id == db_Band.id,
                                                                   ChanAssignments.ctime <= ctime )
                    db_chas = db_cha.order_by(db.desc(ChanAssignments.ctime)).all()[1:]
                    for db_cha in db_chas:
                        in_cha_db = sorted( [ch.channel for ch in db_cha.channels])
                        in_tune_file = sorted( [data[band]['resonances'][x]['channel'] for x in data[band]['resonances'].keys()])
                        if np.all( in_cha_db == in_tune_file):
                            break

                ## add a the assignments and channels to the detector set
                tune.chan_assignments.append(db_cha)
                for ch in db_cha.channels:
                    tune.channels.append(ch)
        session.commit()  
    
    def add_new_observation(self, stream_id, ctime, obs_data, session, max_early=5,max_wait=10):
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
            
        ################################################################################################
        ## Section waiting on tuning files being added to the data stream at the start of observations
        ## For now assume we just use the most recent tuning file.
        ################################################################################################

        self.update_observation_files(obs, session, max_early=max_early,
                    max_wait=max_wait)

    def update_observation_files(self, obs, session, max_early=5, max_wait=10):
        """ Update existing observation. A separate function to make it easier
        to deal with partial data transfers. See add_new_observation for args"""

        # TODO: Update Tune File Assignment to Use Status Information
        
        tune = session.query(Tunes).filter(Tunes.start <= obs.start)
        tune = tune.order_by(db.desc(Tunes.start)).first()
        already_have = [ds.id for ds in obs.tunes]

        if tune is not None:
            if not tune.id in already_have:
                obs.tunes.append(tune)
        else:
            # print('no tuning file found. should I build one?')
            ## Here is where we will put logic if we need to go backwards 
            pass

        x=session.query(Files.name).filter(Files.start >= obs.start-dt.timedelta(seconds=max_early)).order_by(Files.start).first()
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
            flist = session.query(Files).filter(Files.name.like(f_start+'%')).order_by(Files.start).all()
            for f in flist:
                f.obs_id = obs.obs_id
                if tune is not None:
                    f.detset = tune.name
            obs.duration = flist[-1].stop.timestamp() - obs.timestamp
            obs.stop = flist[-1].stop
        session.commit()

    def index_metadata(self, verbose = False, stop_at_error=False):
        """
            Adds all channel assignments, tunefiles, and observations in archive to database. 
            Adding relevant entries to Bands and Files as well.

            Args
            ----
            verbose: bool
                Verbose mode
            stop_at_error: bool
                If True, will stop if there is an error indexing a file.
        """
        if self.meta_path is None:
            raise ValueError('Archiver needs meta_path attribute to index channel assignments')

        session = self.Session()


        for ct_dir in sorted(os.listdir(self.meta_path)):
            ### ignore old things
            if int(ct_dir) < 16000:
                continue

            for stream_id in sorted(os.listdir( os.path.join(self.meta_path, ct_dir))):
                action_path = os.path.join(self.meta_path, ct_dir, stream_id)
                actions = sorted(os.listdir( action_path ))

                for action in actions:
                    try:
                        ctime = int( action.split('_')[0] )
                        astring = '_'.join(action.split('_')[1:])

                        ### Look for channel assignments before tuning files
                        if astring in SMURF_ACTIONS['channel_assignments']:

                            cha_path = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(cha_path) == 0:
                                raise ValueError("found action {} with no output".format(action))
                            cha = None; 
                            for f in cha_path:
                                ##################################################################################
                                ## Needs to account for multiple channel assignments per setup notches.
                                ## Error in pysmurf 4.2 pre-release
                                ## Currently just tracking all of them
                                ##################################################################################
                                match = re.findall('channel_assignment_b\d.txt', f)
                                if len(match)==1:
                                    cha = f
                                    cha_ctime = int(cha.split('_')[0])
                                    cha_path = os.path.join(action_path, action, 'outputs', cha)
                                    self.add_new_channel_assignment(stream_id, cha_ctime, cha, cha_path, session)

                            if cha is None:
                                raise ValueError("found action {}. unable to find channel assignment".format(action))

                        ### Look for tuning files before observations
                        ### Assumes only one tuning file per setup_notches
                        if astring in SMURF_ACTIONS['tuning']:
                            tune_path = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(tune_path) == 0:
                                raise ValueError("found action {} with no output".format(action))
                            tune = None
                            for f in tune_path:
                                match = re.findall('\d_tune.npy', f)
                                if len(match)==1:
                                    tune=f
                                    tune_ctime = int(tune.split('_')[0])
                                    tune_path = os.path.join(action_path, action, 'outputs', tune)
                                    break
                            if tune is None:
                                raise ValueError("found action {}. unable to find tune file".format(action))

                            self.add_new_tuning(stream_id, tune_ctime, tune_path, session)

                        if astring in SMURF_ACTIONS['observations']:
                            ### Sometimes there are just mask files. Sometimes there are mask and freq files.
                            obs_path = os.listdir(os.path.join(action_path, action, 'outputs'))
                            if len(obs_path) == 0:
                                raise ValueError("found action {} with no output".format(action))

                            self.add_new_observation(stream_id, ctime, obs_path, session)

                    except ValueError as e:
                        if verbose:
                            print(e, stream_id, ctime)
                        if stop_at_error:
                            raise(e)
                    #except KeyError as e:
                    #    if verbose:
                    #        print(e, stream_id, ctime)
                    #    if stop_at_error:
                    #        raise(e)
                    except Exception as e:
                        if verbose:
                            print(stream_id, ctime)
                        raise(e)


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

    def load_data(self, start, end, stream_id=None, dets=None, detset=None, 
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
            dets : list or None
                If not None, it should be a list that can be sent to get_channel_mask.
                Called dets for sotodlib compatibility. This function only looks at 
                SMuRF channels.
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
                
        TODO: Track down "Lazy-loaded attribute for Channels.band"
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

        if detset is None:
            detset = session.query(Tunes).filter(Tunes.start <= start)
            detset = detset.order_by( db.desc(Tunes.start) ).first() 
        else:
            detset = session.query(Tunes).filter(Tunes.name==detset).one()
        
        scan_start = session.query(Frames.start).filter(Frames.start > start,
                                                        Frames.type_name=='Scan')
        scan_start = scan_start.order_by(Frames.start).first()
        
        try:
            status = self.load_status(scan_start[0])
        except:
            logger.info("Status load from database failed, using file load")
            status = None
        
        aman = load_file( flist, status=status, dets=dets,
                         archive=self, detset=detset, show_pb=show_pb)
        
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
        my_db.add_props('smurf', ch.name, band=ch.band_number, 
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
                    b,c = channel.band_number, channel.channel
                elif obsfiledb is not None:
                    c = obsfiledb.conn.execute('select band_number,channel from channels where name=?',(ch,))
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

def get_channel_info(status, mask=None, detset=None):
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
    detset : Tune instance or string (optionl)
        detector set for the data
        
    Returns
    --------
    ch_info : AxisManager
    
    """
    #### Get Info for channels in this observation
    if detset is None:
        bnd_assign=[]
        ch_assign=[]
    else:
        bnd_assign = [ ch.band.number for ch in detset.channels ]
        ch_assign =  [ ch.channel for ch in detset.channels     ]
    
    channel_names = []
    channel_bands = []
    channel_channels = []
    channel_freqs = []
    channel_rch = []
   
    ch_list = np.arange( status.num_chans )
    if mask is not None:
        ch_list = ch_list[mask]
        
    for ch in ch_list:
        try:
            sch = status.readout_to_smurf(ch)

            x = np.where(np.all([bnd_assign == sch[0],
                                 ch_assign  == sch[1]], axis=0))[0]

            if len(x) == 0:
                ########################################
                ## There appear to be a non-trivial number of these when
                ## using Tunes. SMuRF Error or Indexing Error?
                ########################################
                name = 'sch_NONE_{}_{:03d}'.format(sch[0],sch[1])
                freq = status.freq_map[sch[0], sch[1]]
            else:
                channel = detset.channels[x[0]]
                name = channel.name
                ########################################
                ## Related to above, freq != status.freq for many of these
                ########################################
                freq = channel.frequency

        except:
            ## load 'readout channel' as a backup
            name='rch_{:04d}'.format(ch)
            sch = (-1,-1)
            freq=-1

        channel_names.append(name)
        channel_bands.append(sch[0])
        channel_channels.append(sch[1])
        channel_freqs.append(freq)
        channel_rch.append('r{:04d}'.format(ch)) 
    
    
    ch_info = core.AxisManager( core.LabelAxis('channels', channel_names),)
    ch_info.wrap('band', np.array(channel_bands), ([(0,'channels')]) )
    ch_info.wrap('channel', np.array(channel_channels), ([(0,'channels')]) )
    ch_info.wrap('frequency', np.array(channel_freqs), ([(0,'channels')]) )
    ch_info.wrap('rchannel', np.array(channel_rch), ([(0,'channels')]) )
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
        if 'primary' in streams and 'UnixTime' in 'primary':
            load_type = TimingParadigm.SmurfUnixTime
        else:
            load_type = TimingParadigm.G3Timestream
    
    if load_type == TimingParadigm.SmurfUnixTime:
        return io_load.hstack_into(None, streams['primary']['UnixTime'])/1e9
    if load_type == TimingParadigm.G3Timestream:
        return io_load.hstack_into(None, streams['time'])
    logger.error("Timing System could not be determined")
            
        
def load_file(filename, channels=None, ignore_missing=True, 
             load_biases=True, load_primary=True, status=None,
             archive=None, obsfiledb=None, detset=None, show_pb=True):
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
      detset : Detset database entry, used for channel names in channel info
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

    ch_info = get_channel_info(status, ch_mask, detset=detset)

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
    print(dets)
    flist = [row[0] for row in c]
    return load_file(flist, dets, obsfiledb=db)


io_load.OBSLOADER_REGISTRY['g3tsmurf'] = load_g3tsmurf_obs
