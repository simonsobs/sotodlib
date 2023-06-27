import sqlalchemy as db
from sqlalchemy.exc import IntegrityError

from sqlalchemy import and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import os
import yaml
import numpy as np
from tqdm import tqdm
from so3g import hk


import logging
logger = logging.getLogger(__name__)

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
    start : integer
        TODO
    end : integer
        TODO
    hkfile:
    fields

    TODO: add uniqueness constraint
    """
    __tablename__ = 'hkagents'
    # unique constraint for instance-id? needs more thought?
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('hkfiles.id'))
    hkfile = relationship("HKFiles", back_populates='agents')

    instance_id = db.Column(db.String)

    start = db.Column(db.Integer)
    end = db.Column(db.Integer)

    fields = relationship("HKFields", back_populates='hkagent')


class HKFields(Base):
    """This table is named hkfields and serves as a db for all
    fields inside each HK'ing file in the hkfeeds table.

    Attributes
    ----------
    file_id : integer
        id that points a field back to its corresponding
        HK file in the hkfeeds table
    agent_id : integer
        TODO
    field : string
        name of HK field in corresponding HK file
    start : integer
        start time for each HK field in ctime
    end : integer
        end time for each HK field in ctime
    
    TODO: add uniqueness constraint
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
    median = db.Column(db.Integer)
    mean = db.Column(db.Integer)
    min_val = db.Column(db.Integer)
    max_val = db.Column(db.Integer)
    stand_dev = db.Column(db.Integer)
    special_math = db.Column(db.Integer)


class G3tHk:
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
        self.engine = db.create_engine(f"sqlite:///{db_path}", echo=echo)
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def load_fields(self, hk_path):
        """
        Load fields from .g3 file and start and end time for each field.

        Args
        ____

        hk_path : string
            file path    

        returns: list of field names and corresponding start and
                 end times from HK .g3 files
        """

        # enact HKArchiveScanner
        hkas = hk.HKArchiveScanner()
        hkas.process_file(hk_path)

        arc = hkas.finalize()

        # get fields from .g3 file
        fields, timelines = arc.get_fields()
        hkfs = []
        for key in fields.keys():
            hkfs.append(key)

        starts = []
        ends = []
        medians = []
        means = []
        min_vals = []
        max_vals = []
        stds = []

        for i in range(len(hkfs)):
            data = arc.simple(hkfs[i])
            time = data[0]
            starts.append(time[0])
            ends.append(time[-1])
            try:
                medians.append(np.median(data[1]))
                means.append(np.mean(data[1]))
                min_vals.append(np.min(data[1]))
                max_vals.append(np.max(data[1]))
                stds.append(np.std(data[1]))
            except:
                medians.append(np.nan)
                means.append(np.nan)
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                stds.append(np.nan)
            

        return hkfs, starts, ends, medians, means, min_vals, max_vals, stds

    def _get_agg(self, path_to_file):  # only use hkarchive_path as the FULL path
        """
        Load aggregator information from .g3 file by splitting the field
        names

        Args
        ____

        path_to_file: the full path to a .g3 file which is used in 
                      populate_hkfiels() to get the aggregator name

        returns: aggregator name
        
        """
        # enact HKArchiveScanner
        hkas = hk.HKArchiveScanner()
        hkas.process_file(path_to_file)

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


    def _files_to_add(self, min_ctime=None, max_ctime=None):

        db_files = self.session.query(HKFiles).all()
        db_files = [f.path for f in db_files]

        dirs = []
        dir_list = os.listdir(self.hkarchive_path)
        for i in range(len(dir_list)):
            base = os.path.join(self.hkarchive_path, dir_list[i])
            dirs.append(base)
        dirs = sorted(dirs)

        path_list = []

        for i in range(len(dirs)):
            for root, _, files in sorted(os.walk(dirs[i])):
                for f in sorted(files):
                    path = os.path.join(root, f)
                    if f[-3:] != ".g3":
                        continue
                    if path in db_files:
                        continue
                    ftime = int(f.split(".")[0])
                    if min_ctime is not None and ftime < min_ctime:
                        continue
                    if max_ctime is not None and ftime > max_ctime:
                        continue 
                    path_list.append(path)

        logger.debug(f"{len(path_list)} files to add to database.")
        return path_list

    def add_hkfiles(self, min_ctime=None, max_ctime=None, show_pb=True):
        """Gather and add column information for hkfiles tables
        """

        file_list = self._files_to_add(
            min_ctime = min_ctime, 
            max_ctime = max_ctime,
        )        


        for path in tqdm( sorted(file_list), disable=(not show_pb)):
            root, filename = os.path.split(path)

            global_start_time = int(filename.split(".")[0])

            try: 
                aggregator = self._get_agg(path)

                db_file = HKFiles(filename=filename,
                                  path=path,
                                  global_start_time=global_start_time,
                                  aggregator=aggregator)

                self.session.add(db_file)
                self.session.commit()
            except IntegrityError as e:
                # Database Integrity Errors, such as duplicate entries
                self.session.rollback()
                logger.warning(f"Integrity error with error {e}")
            except RuntimeError as e:
                # End of stream errors, for G3Files that were not fully flushed
                self.session.rollback()
                logger.warning(f"Failed on file {filename} due to end of stream error!")
            except Exception as e:
                # This will catch generic errors such as attempting to load
                # out-of-date files that do not have the required frame
                # structure specified in the TOD2MAPS docs.
                self.session.rollback()
                if stop_at_error:
                    raise e
                logger.warning(f"Failed on file {filename}:\n{e}")

    
    def add_agents_and_fields(self, db_file):
        fields, starts, ends, medians, means, min_vals, max_vals, stds = self.load_fields(db_file.path)
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

            db_agent = self.session.query(HKAgents).filter(
                HKAgents.instance_id == agent,
                HKAgents.file_id == db_file.id,
            ).one_or_none()
            if db_agent is None:
                #  populate the HKAgents table
                db_agent = HKAgents(instance_id=agent,
                                    start=agent_start,
                                    end=agent_stop,
                                    hkfile=db_file)
                self.session.add(db_agent)
            else:
                # stop may have changed for an incomplete file?
                db_agent.end=agent_stop
            self.session.commit()
        
        for i in range(len(fields)):
            agentname = fields[i].split('.')[1]
            db_agent = self.session.query(HKAgents).filter(
                    and_(HKAgents.instance_id == agentname,
                         HKAgents.file_id == db_file.id)).one()
            db_field = self.session.query(HKFields).filter(
                HKFields.agent_id == db_agent.id,
                HKFields.file_id == db_file.id,
                HKFields.field == fields[i],
            ).one_or_none()

            if db_field is None:
                db_field = HKFields(field=fields[i],
                                   start=starts[i],
                                   hkfile=db_file,
                                   hkagent=db_agent)
                self.session.add(db_field)
            
            # update the math incase previous incomplete file?    
            db_field.end = ends[i]
            db_field.median = medians[i]
            db_field.mean = means[i]
            db_field.min_val = min_vals[i]
            db_field.max_val = max_vals[i]
            db_field.stand_dev = stds[i]

            self.session.commit()
        
    def populate_hkagents(self):
        """
        """
        file_list = self.session.query(HKFiles).all()

        for file in file_list:
            fields, starts, ends, medians, means, min_vals, max_vals, stds = self.load_fields(file.path)
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
                                    end=agent_stop,
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
            self.populate_hkagents()

        #  when new.g3 file is written, update the table
        else:
            last_agent_file_id = hkagents_list[-1].file_id
            if last_file_id > last_agent_file_id:
                self.populate_hkagents()

    def populate_hkfields(self):
        """
        """
        file_list = self.session.query(HKFiles).all()

        for file in file_list:
            fields, starts, ends, medians, means, min_vals, max_vals, stds = self.load_fields(file.path)

            for i in range(len(fields)):
                agentname = fields[i].split('.')[1]
                agent = self.session.query(HKAgents).filter(
                        and_(HKAgents.instance_id == agentname,
                             HKAgents.file_id == file.id)).all()

                db_file = HKFields(field=fields[i],
                                   start=starts[i],
                                   end=ends[i],
                                   median=medians[i],
                                   mean=means[i],
                                   min_val=min_vals[i],
                                   max_val=max_vals[i],
                                   stand_dev=stds[i],
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
            self.populate_hkfields()

        # when new .g3 file is written, update table
        else:
            last_field_file_id = hkfields_list[-1].file_id
            if last_file_id > last_field_file_id:
                self.populate_hkfields()

    @classmethod
    def from_configs(cls, configs):
        """
        Create a G3tHK instance from a configs dictionary

        Args
        ----
        configs - dictionary containing `data_prefix` and `g3thk_db` keys
        """
        if type(configs)==str:
            configs = yaml.safe_load(open(configs, "r"))
        
        return cls(os.path.join(configs["data_prefix"], "hk"), configs['g3thk_db'])

