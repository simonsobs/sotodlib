import sqlalchemy as db

from sqlalchemy import and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import os
import numpy as np
from so3g import hk

Base = declarative_base()
Session = sessionmaker()


# all database definitions for G3tHK
class HKFiles(Base):
    """This table is named hkfiles and serves as a db for holding
    brief information about each HK'ing file.

    Attributes
    ----------
    filename : string
        name of HK file
    global_start_time : integer
        timestamp that begins a new HK file
    path : string
        path to the HK file
    aggregator:
    fields:
    agents
    """
    __tablename__ = 'hkfiles'
    __table_args__ = (
            db.UniqueConstraint('filename'),
            db.UniqueConstraint('global_start_time'),
    )

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(db.String, nullable=False, unique=True)
    global_start_time = db.Column(db.Integer)
    path = db.Column(db.String)
    aggregator = db.Column(db.String)

    fields = relationship("HKFields", back_populates='hkfile')
    agents = relationship("HKAgents", back_populates='hkfile')


class HKAgents(Base):
    """This table is named HKAgents; serves as a db for holding
    critical information about each Agent in an HK file.

    Attributes
    ----------
    file_id : integer
        id that points a field back to its corresponding
        HK file in the hkfeeds table
    instance_id : string
        TODO
    alias : string
        TODO
    start : integer
        TODO
    stop : integer
        TODO
    hkfile:
    fields
    """
    __tablename__ = 'hkagents'
    # unique constraint for instance-id? needs more thought?
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('hkfiles.id'))
    hkfile = relationship("HKFiles", back_populates='agents')

    start = db.Column(db.Integer)
    stop = db.Column(db.Integer)

    fields = relationship("HKFields", back_populates='hkagent')


class HKFields(Base):
    """This table is named hkfields and serves as a db for all
    fields inside each HK'ing file in the hkfeeds table.

    Attributes
    ----------
    file_id : integer
        id that points a field back to its corresponding
        HK file in the hkfeeds table
    field : string
        name of HK field in corresponding HK file
    start : integer
        start time for each HK field in ctime
    end : integer
        end time for each HK field in ctime
    """
    __tablename__ = 'hkfields'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('hkfiles.id'))
    hkfile = relationship("HKFiles", back_populates='fields')

    agent_id = db.Column(db.Integer, db.ForeignKey('hkagents.id'))
    hkagent = relationship("HKAgents", back_populates='fields')

    field = db.Column(db.String)
    start = db.Column(db.Integer)
    end = db.Column(db.Integer)


class G3thk:
    def __init__(self, hkarchive_path, db_path=None, echo=False):
        """
        Class to manage a housekeeping data archive

        Args
        ____
        hkarchive_path : path
            Path to the data directory
        db_path : path, optional
            Path to the sqlite file
        echo : bool, optional
            If true, all sql statements will print to stdout
        """
        if db_path is None:
            db_path = os.path.join(hkarchive_path, '_HK.db')

        self.hkarchive_path = hkarchive_path
        self.db_path = db_path
        self.engine = db.create_engine(f"sqlite:///{db_path}")
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def load_fields(self, hkarchive_path, hkfile):
        """
        Load fields from .g3 file and start and end time for each field.

        Args
        ____

        hkarchive_path : string
            Path to data directory
        hkfile: string
            Name of .g3 HK file

        returns: list of field names and corresponding start and
                 end times from HK .g3 files
        """
        f = os.path.join(hkarchive_path, hkfile)

        # enact HKArchiveScanner
        hkas = hk.HKArchiveScanner()
        hkas.process_file(f)

        arc = hkas.finalize()

        # get fields from .g3 file
        fields, timelines = arc.get_fields()
        hkfs = []
        for key in fields.keys():
            hkfs.append(key)

        starts = []
        ends = []
        for i in range(len(hkfs)):
            data = arc.simple(hkfs[i])
            time = data[0]
            starts.append(time[0])
            ends.append(time[-1])

        return hkfs, starts, ends

    def get_agg(self, hkfile):  # only use hkarchive_path as the FULL path
        """
        Load aggregator information from .g3 file by splitting the field
        names

        Args
        ____

        hkarchivepath: str
        hkfile: str

        returns: aggregator
        """
        file = os.path.join(self.hkarchive_path, hkfile)

        # enact HKArchiveScanner
        hkas = hk.HKArchiveScanner()
        hkas.process_file(file)

        arc = hkas.finalize()

        # get fields from .g3 file
        fields, timelines = arc.get_fields()
        hkfs = []
        for key in fields.keys():
            hkfs.append(key)

        agg_name = hkfs[0].split('.')[0]
        for agg in hkfs:
            assert agg.split('.')[0] == agg_name, 'Aggregator name is not consistent'

        return agg_name

    def populate_hkfiles(self):
        """
        """
        for root, _, files in os.walk(self.hkarchive_path):
            for f in files:
                path = os.path.join(root, f)
                filename = path.split("/")[-1]
                global_start_time = int(filename.split(".")[0])
                aggregator = G3thk.get_agg(self.hkarchivepath, f)

                db_file = HKFiles(filename=filename,
                                  path=self.hkarchive_path,
                                  global_start_time=global_start_time,
                                  aggregator=aggregator)

                self.session.add(db_file)
                self.session.commit()

    def add_hkfiles(self):
        """
        """
        file_list = self.session.query(HKFiles).all()
        if len(file_list) == 0:
            G3thk.populate_hkfiles()
        else:
            last_file_id = file_list[-1].id
            # TODO: if statement for populating files when .g3 file written

    def populate_hkagents(self):
        """
        """
        file_list = self.session.query(HKFiles).all()

        for file in file_list:
            fields, starts, ends = G3thk.load_fields(file.path, file.filename)
            agents = []
            for field in fields:
                agent = field.split('.')[1]
                agents.append(agent)

            #  remove duplicate agent names to avoid multiple agent_ids for
            #  same instance_id
            agents = [*set(agents)]

            # get the start and stop for each agent
            for agent in agents:
                starts_agent = []
                ends_agent = []
                for i in range(len(fields)):
                    #  extract agent instance id from each field name
                    if fields[i].split('.')[1] == agent:
                        #  gather all the starts and stops for each field
                        #  in the agent
                        starts_agent.append(starts[i])
                        ends_agent.append(ends[i])

                    #  from the starts_agent and ends_agent, extract start
                    #  and stop time for the agent
                    agent_start = np.min(starts_agent)
                    agent_stop = np.max(ends_agent)

                    #  populate the HKAgents table
                    db_file = HKAgents(instance_id=agent,
                                       start=agent_start,
                                       stop=agent_stop,
                                       hkfile=file)
                    self.session.add(db_file)

                self.session.commit()

    def add_hkagents(self):
        """
        """
        file_list = self.session.query(HKFiles).all()
        last_file_id = file_list[-1].id

        hkagents_list = self.session.query(HKAgents).all()

        #  if hkagents table empty, populate table for the first time
        if len(hkagents_list) == 0:
            G3thk.populate_hkagents()

        #  when new.g3 file is written, update the table
        else:
            last_agent_file_id = hkagents_list[-1].file_id
            if last_file_id > last_agent_file_id:
                G3thk.populate_hkagents()

    def populate_hkfields(self):
        """
        """
        file_list = self.session.query(HKFiles).all()

        for file in file_list:
            fields, starts, ends = G3thk.load_fields(file.path, file.filename)

            for i in range(len(fields)):
                agentname = fields[i].split('.')[1]
                agent = self.session.query(HKAgents).filter(
                        and_(HKAgents.instance_id == agentname,
                             HKAgents.file_id == file.id)).all()

                db_file = HKFields(field=fields[i],
                                   start=starts[i],
                                   end=ends[i],
                                   hkfile=file,
                                   hkagent=agent[0])
                self.session.add(db_file)

            self.session.commit()

    def add_hkfields(self):
        """
        """
        file_list = self.session.query(HKFiles).all()
        last_file_id = file_list[-1].id

        hkfields_list = self.session.query(HKFields).all()
        # if hkfields table empy, populate table for 1st time
        if len(hkfields_list) == 0:
            G3thk.populate_hkfields()

        # when new .g3 file is written, update table
        else:
            last_field_file_id = hkfields_list[-1].file_id
            if last_file_id > last_field_file_id:
                G3thk.populate_hkfields()
