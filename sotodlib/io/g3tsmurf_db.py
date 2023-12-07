""" This module includes all the database definitions used with G3tSmurf, broken out 
here to reduce the overall number of lines in each module. 
"""
from enum import Enum
import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

Base = declarative_base()

association_table = db.Table(
    "association_chan_assign",
    Base.metadata,
    db.Column("tunesets", db.Integer, db.ForeignKey("tunesets.id")),
    db.Column("chan_assignments", db.Integer, db.ForeignKey("chan_assignments.id")),
)

association_table_obs = db.Table(
    "association_obs",
    Base.metadata,
    db.Column("tunesets", db.Integer, db.ForeignKey("tunesets.id")),
    db.Column("observations", db.Integer, db.ForeignKey("obs.obs_id")),
)

association_table_dets = db.Table(
    "detsets",
    Base.metadata,
    db.Column("name", db.Integer, db.ForeignKey("tunesets.name")),
    db.Column("det", db.Integer, db.ForeignKey("channels.name")),
)


class TimeCodes(Base):
    """Table for tracking if files and metadata are ready for bookbinding. Built
    to work off of the suprsync finalization files.

    Attributes
    ----------
    stream_id: string
        stream_id found in the timecode files
    suprsync_type: integer
        entry in SupRsyncType class. For tracking if suprsync is managing the
        timestreams (files) or smurf (meta) files
    timecode: integer
        the 5 digit ctime folder number that has been finalized
    agent: string
        instance id of the suprsync agent we got the finalization information
        from.
    """

    __tablename__ = "time_codes"
    __table_args__ = (db.UniqueConstraint("stream_id", "suprsync_type", "timecode"),)
    id = db.Column(db.Integer, primary_key=True)
    stream_id = db.Column(db.String)
    suprsync_type = db.Column(db.Integer)
    timecode = db.Column(db.Integer)
    agent = db.Column(db.String)


class SupRsyncType(Enum):
    """Files match with the timestream folders and meta match with the smurf
    folders.
    """

    FILES = 0
    META = 1

    @classmethod
    def from_string(cls, string):
        if string == "smurf":
            return cls.META
        elif string == "timestreams":
            return cls.FILES
        else:
            raise ValueError("SupRsync strings are 'smurf' or 'timestreams'")


class Finalize(Base):
    """Table for tracking what the finalization times for different agents were
    at the last g3tsmurf update. Will have a special row for the archive update
    time that will have the name G3tSMURF.

    Attributes
    -----------
    agent: string
        instance id of the suprsync agents
    time: float
        the finalization timestamp of the agent at the last update
    """

    __tablename__ = "finalize"
    id = db.Column(db.Integer, primary_key=True)
    agent = db.Column(db.String, unique=True)
    time = db.Column(db.Float)


class Observations(Base):
    """Times of continuous detector readout. This table is named obs and serves
    as the ObsDb table when loading via Context. This table is meant to by built
    off of Level 2 data, before the data from different stream_id/smurf slots have
    been bookbound and perfectly co-sampled.

    Observations are not created if the action folder has no associated .g3 files.

    Dec. 2021 -- The definitions of obs_id and timestamp changed to better
    match the operation of the smurf-streamer / sodetlib / pysmurf.
    Oct. 2022 -- The definition of obs_id changed again to include obs or oper
    tags based on if the observation is an sodetlib operation or not.

    Attributes
    -----------
    obs_id : string
        <obs|oper>_<stream_id>_<session_id>.
    timestamp : integer
        The .g3 session_id, which is also the ctime the .g3 streaming started
        and the first part .g3 file name.
    action_ctime : integer
        The ctime of the pysmurf action, generally slightly different than
        the .g3 session_id
    action_name : stream
        The name of the action used to create the observation.
    stream_id : string
        The stream_id of this observation. Generally corresponds to UFM or Smurf
        slot. Column is implemented since level 2 data is not perfectly co-sampled
        across stream_ids.
    timing : bool
        If true, the files of the entry observation were made with times
        referenced to the external timing system and high precision timestamps.
    duration : float
        The total observation time in seconds
    n_samples : integer
        The total number of samples in the observation
    start : datetime.datetime
        The start of the observation as a datetime object
    stop : datetime.datetime
        The end of the observation as a datetime object
    tag : string
        Tags for this observation in a single comma delimited string. These are populated
        through tags set while running sodetlib's stream data functions.
    calibration : bool
        Boolean that stores whether or not the observation is a calibration-type observation
        i.e. an IV curve, a bias step, etc.
    files : list of SQLAlchemy instances of Files
        The list of .g3 files in this observation built through a relationship
        to the Files table. [f.name for f in Observation.files] will return
        absolute paths to all the files.
    tunesets : list of SQLAlchemy instances of TuneSets
        The TuneSets used in this observation. There is expected to be
        one per stream_id (SMuRF crate slot).
    """

    __tablename__ = "obs"

    obs_id = db.Column(db.String, primary_key=True)
    timestamp = db.Column(db.Integer)
    action_ctime = db.Column(db.Integer)
    action_name = db.Column(db.String)

    stream_id = db.Column(db.String)
    timing = db.Column(db.Boolean)

    # in seconds
    duration = db.Column(db.Float)
    n_samples = db.Column(db.Integer)

    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)

    tag = db.Column(db.String)

    calibration = db.Column(db.Boolean)
    ## one to many
    files = relationship(
        "Files",
        back_populates="observation",
        order_by="Files.start",
    )

    ## many to many
    tunesets = relationship(
        "TuneSets", secondary=association_table_obs, back_populates="observations"
    )

    def __repr__(self):
        try:
            return f"{self.obs_id}: {self.start} -> {self.stop} [{self.stop-self.start}] ({self.tag})"
        except:
            return f"{self.obs_id}: {self.start} -> {self.stop} ({self.tag})"


class Tags(Base):
    """Tags used to mark Observations.

    To have these automatically added, use
    sodetlib.smurf_funct.smurf_ops.stream_g3_on( S, tag= 'tag1,tag2')
    or
    sodetlib.smurf_funct.smurf_ops.take_g3_data(S, dur, tag='tag1,tag2')

    Note, this table is not relationally mapped to the Observation.tag column.
        It is set up to match the Context design
    """

    __tablename__ = "tags"
    id = db.Column(db.Integer, primary_key=True)

    obs_id = db.Column(db.String)
    tag = db.Column(db.String)


class Files(Base):
    """Table to store file indexing info. This table is named files in sql and
    serves as the ObsFileDb when loading via Context.

    Attributes
    ------------
    id : integer
        auto-incremented primary key
    name : string
        complete absolute path to file
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
    timing : bool
        If true, every frame in the file has high precision timestamps (slightly
        different definition than obs.timing)
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

    __tablename__ = "files"
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String, nullable=False, unique=True)

    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)

    ## This is sample in an Observation
    sample_start = db.Column(db.Integer)
    sample_stop = db.Column(db.Integer)

    ## this is a string for compatibility with sotodlib, not because it makes
    ## sense here
    obs_id = db.Column(db.String, db.ForeignKey("obs.obs_id"))
    observation = relationship("Observations", back_populates="files")

    stream_id = db.Column(db.String)
    timing = db.Column(db.Boolean)

    n_frames = db.Column(db.Integer)
    frames = relationship("Frames", back_populates="file")

    ## n_channels is a renaming of channels
    n_channels = db.Column(db.Integer)

    # breaking from linked table convention to match with obsfiledb requirements
    ## many to one
    detset = db.Column(db.String, db.ForeignKey("tunesets.name"))
    tuneset = relationship("TuneSets", back_populates="files")

    tune_id = db.Column(db.Integer, db.ForeignKey("tunes.id"))
    tune = relationship("Tunes", back_populates="files")


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

    __tablename__ = "tunes"
    __table_args__ = (db.UniqueConstraint("name", "stream_id"),)
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String)
    path = db.Column(db.String)
    stream_id = db.Column(db.String)

    ## should stop exist? tune file use does not need to be continguous
    start = db.Column(db.DateTime)

    ## files that use this tune file
    ## one to many
    files = relationship("Files", back_populates="tune")

    ## one to many
    tuneset_id = db.Column(db.Integer, db.ForeignKey("tunesets.id"))
    tuneset = relationship("TuneSets", back_populates="tunes")

    @staticmethod
    def get_name_from_status(status):
        """Return the name format expected from SmurfStatus instance"""
        return status.stream_id + "_" + status.tune.strip(".npy")


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

    __tablename__ = "tunesets"
    __table_args__ = (db.UniqueConstraint("name", "stream_id"),)
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String)
    path = db.Column(db.String)
    stream_id = db.Column(db.String)

    ## should stop exist? tune file use does not need to be continguous
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)

    ## files that use this detset
    ## one to many
    files = relationship("Files", back_populates="tuneset")
    tunes = relationship("Tunes", back_populates="tuneset")
    ## many to many
    observations = relationship(
        "Observations", secondary=association_table_obs, back_populates="tunesets"
    )

    ## many to many
    chan_assignments = relationship(
        "ChanAssignments", secondary=association_table, back_populates="tunesets"
    )

    ## many to many
    dets = relationship(
        "Channels", secondary=association_table_dets, back_populates="detsets"
    )

    @property
    def channels(self):
        return self.dets


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

    __tablename__ = "chan_assignments"
    id = db.Column(db.Integer, primary_key=True)

    ctime = db.Column(db.Integer)
    path = db.Column(db.String, unique=True)
    name = db.Column(db.String)
    stream_id = db.Column(db.String)
    band = db.Column(db.Integer)

    ## Channel Assignments are put into detector sets
    ## many to many bidirectional
    tunesets = relationship(
        "TuneSets", secondary=association_table, back_populates="chan_assignments"
    )

    ## Each channel assignment is made of many channels
    ## one to many
    channels = relationship("Channels", back_populates="chan_assignment")


class Channels(Base):
    """All the channels tracked by SMuRF indexed by the ctime of the channel
    assignment file, SMuRF band and channel number. Many channels will map to
    one detector on a UFM.

    Dec. 2021 -- Updated channel names to include stream_id to ensure uniqueness

    Attributes
    ----------
    id : integer
        primary key
    name : string
        name of of channel. This is the unique readout id that will be matched with
        the unique detector id. Has the form of sch_<stream_id>_<ctime>_<band>_<channel>
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

    __tablename__ = "channels"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True)
    stream_id = db.Column(db.String)

    ## smurf channels
    subband = db.Column(db.Integer)
    channel = db.Column(db.Integer)
    frequency = db.Column(db.Float)

    band = db.Column(db.Integer)

    ## many to one
    ca_id = db.Column(db.Integer, db.ForeignKey("chan_assignments.id"))
    chan_assignment = relationship("ChanAssignments", back_populates="channels")

    ## many to many
    detsets = relationship(
        "TuneSets", secondary=association_table_dets, back_populates="dets"
    )


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

    __tablename__ = "frame_offsets"
    __table_args__ = (db.UniqueConstraint("file_id", "frame_idx", name="_frame_loc"),)

    id = db.Column(db.Integer, primary_key=True)

    file_id = db.Column(db.Integer, db.ForeignKey("files.id"))
    file = relationship("Files", back_populates="frames")

    frame_idx = db.Column(db.Integer, nullable=False)
    offset = db.Column(db.Integer, nullable=False)

    type_name = db.Column(db.String, db.ForeignKey("frame_type.type_name"))
    frame_type = relationship("FrameType")

    status_dump = db.Column(db.Boolean, nullable=False, default=False)
    time = db.Column(db.DateTime, nullable=False)

    # Specific to data frames
    n_samples = db.Column(db.Integer)
    n_channels = db.Column(db.Integer)
    start = db.Column(db.DateTime)
    stop = db.Column(db.DateTime)

    def __repr__(self):
        return f"Frame({self.type_name})<{self.frame_idx}>"
