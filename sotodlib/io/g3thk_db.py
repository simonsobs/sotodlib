import sqlalchemy as db
from sqlalchemy.exc import IntegrityError

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, selectinload

import os
import yaml
import numpy as np
from tqdm import tqdm
from so3g import hk

import logging
from .datapkg_utils import load_configs


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
    aggregator : string
        the aggregator corresponding to the data; ex: satp1, satp2, site

    """
    __tablename__ = "hkfiles"
    __table_args__ = (
        db.UniqueConstraint("filename"),
        db.UniqueConstraint("global_start_time"),
    )

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(db.String, nullable=False, unique=True)
    global_start_time = db.Column(db.Integer)
    path = db.Column(db.String)
    aggregator = db.Column(db.String)

    fields = relationship("HKFields", back_populates="hkfile")
    agents = relationship("HKAgents", back_populates="hkfile")


class HKAgents(Base):
    """This table is named hkagents; serves as a db for holding
    critical information about each agent in an HK file.

    Attributes
    ----------
    file_id : integer
        id that points an agent back to its corresponding
        HK file in the hkfiles table
    instance_id : string
        the identifier for the instrument/agent; ex: LSA22YG
        for a Lakeshore 372 with the serial number identified as
        an ocs instance-id
    start : integer
        start timestamp corresponding to the agent for corresponding file
    end : integer
        start timestamp corresponding to the agent for corresponding file

    """
    __tablename__ = "hkagents"
    # unique constraint for instance-id? needs more thought?
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("hkfiles.id"))
    hkfile = relationship("HKFiles", back_populates="agents")

    instance_id = db.Column(db.String)

    start = db.Column(db.Integer)
    stop = db.Column(db.Integer)

    fields = relationship("HKFields", back_populates="hkagent")


class HKFields(Base):
    """This table is named hkfields and serves as a db for all
    fields inside each HK file in the hkfeeds table.

    Attributes
    ----------
    file_id : integer
        id that points a field back to its corresponding
        HK file in the hkfiles table
    agent_id : integer
        id that points a field back to its corresponding HK
        agent in the hkagents table
    field : string
        name of HK field in corresponding HK file
    start : integer
        start time for each HK field in ctime for .g3 file
    end : integer
        end time for each HK field in ctime for .g3 file

    """
    __tablename__ = "hkfields"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("hkfiles.id"))
    hkfile = relationship("HKFiles", back_populates="fields")

    agent_id = db.Column(db.Integer, db.ForeignKey("hkagents.id"))
    hkagent = relationship("HKAgents", back_populates="fields")

    field = db.Column(db.String)
    start = db.Column(db.Integer)
    stop = db.Column(db.Integer)
    median = db.Column(db.Float)
    mean = db.Column(db.Float)
    min_val = db.Column(db.Float)
    max_val = db.Column(db.Float)
    stand_dev = db.Column(db.Float)
    special_math = db.Column(db.Float)


class G3tHk:
    def __init__(self, hkarchive_path, iids, db_path=None, echo=False):
        """
        Class to manage a housekeeping data archive

        Args
        ____
        hkarchive_path : path
            Path to the data directory
        iids : list
            List of agent instance ids
        db_path : path, optional
            Path to the sqlite file
        echo : bool, optional
            If true, all sql statements will print to stdout
        """
        if db_path is None:
            db_path = os.path.join(hkarchive_path, "_HK.db")

        self.hkarchive_path = hkarchive_path
        self.db_path = db_path
        self.iids = iids
        self.engine = db.create_engine(f"sqlite:///{db_path}", echo=echo)
        Session.configure(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)


    def load_fields(self, hk_path, iids):
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
            if any(iid in key for iid in iids):
                hkfs.append(key)

        starts = []
        stops = []
        medians = []
        means = []
        min_vals = []
        max_vals = []
        stds = []

        for i in range(len(hkfs)):
            data = arc.simple(hkfs[i])
            time = data[0]
            starts.append(time[0])
            stops.append(time[-1])
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

        return hkfs, starts, stops, medians, means, min_vals, max_vals, stds

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

        agg_name = hkfs[0].split(".")[0]
        for agg in hkfs:
            assert agg.split(".")[0] == agg_name, "Aggregator name is not consistent"

        return agg_name

    def _files_to_add(self, min_ctime=None, max_ctime=None, update_last_file=True):
        db_files = self.session.query(HKFiles)
        if db_files.count() == 0:
            db_files = []
        else:
            if update_last_file:
                last = db_files.order_by(db.desc(HKFiles.global_start_time)).first()
                db_files = db_files.filter(
                    HKFiles.global_start_time != last.global_start_time
                )
            db_files = db_files.all()
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

    def add_hkfiles(
        self,
        min_ctime=None,
        max_ctime=None,
        show_pb=True,
        stop_at_error=False,
        update_last_file=True,
    ):
        """Gather and add column information for hkfiles tables"""

        file_list = self._files_to_add(
            min_ctime=min_ctime,
            max_ctime=max_ctime,
            update_last_file=update_last_file,
        )

        for path in tqdm(sorted(file_list), disable=(not show_pb)):
            try:
                root, filename = os.path.split(path)
                self.add_file(path, overwrite=True)
                self.add_agents_and_fields(path, overwrite=True)

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

    def add_file(self, path, overwrite=False):
        db_file = (
            self.session.query(HKFiles)
            .filter(
                HKFiles.path == path,
            )
            .one_or_none()
        )

        if db_file is not None and not overwrite:
            return
        if db_file is None:
            db_file = HKFiles(path=path)
            self.session.add(db_file)

        root, filename = os.path.split(path)
        db_file.filename = filename
        db_file.global_start_time = int(filename.split(".")[0])
        db_file.aggregator = self._get_agg(path)

        self.session.commit()

    def add_agents_and_fields(self, path, overwrite=False):
        db_file = (
            self.session.query(HKFiles)
            .filter(
                HKFiles.path == path,
            )
            .one()
        )

        # line below may not be needed; is redundant
        db_agents = [a for a in db_file.agents if a.instance_id in self.iids]
        db_fields = db_file.fields

        out = self.load_fields(db_file.path, self.iids)
        fields, starts, stops, medians, means, min_vals, max_vals, stds = out

        agents = []
        for field in fields:
            agent = field.split(".")[1]
            agents.append(agent)

        #  remove duplicate agent names to avoid multiple agent_ids for
        #  same instance_id
        agents = [*set(agents)]

        # get the start and stop for each agent
        for agent in agents:
            starts_agent = []
            stops_agent = []
            for i in range(len(fields)):
                #  extract agent instance id from each field name
                if fields[i].split(".")[1] == agent:
                    #  gather all the starts and stops for each field
                    #  in the agent
                    starts_agent.append(starts[i])
                    stops_agent.append(stops[i])

            #  from the starts_agent and ends_agent, extract start
            #  and stop time for the agent
            agent_start = np.min(starts_agent)
            agent_stop = np.max(stops_agent)

            x = np.where([agent == a.instance_id for a in db_agents])[0]
            if len(x) == 0:
                # if we don't have an agent
                db_agent = HKAgents(
                    instance_id=agent,
                    start=agent_start,
                    stop=agent_stop,
                    hkfile=db_file,
                )
                self.session.add(db_agent)
            elif overwrite:
                # stop may have changed for an incomplete file?
                db_agent = db_agents[x[0]]
                db_agent.start = agent_start
                db_agent.stop = agent_stop

        self.session.commit()
        db_agents = db_file.agents

        for i in range(len(fields)):
            agentname = fields[i].split(".")[1]
            x = np.where([agentname == a.instance_id for a in db_agents])[0]
            if len(x) == 0:
                logger.error(
                    f"Somehow did not add agent {agentname} for {db_file.path}"
                )
                continue
            db_agent = db_agents[x[0]]
            x = np.where([fields[i] == f.field for f in db_fields])[0]
            if len(x) > 0 and not overwrite:
                continue
            elif len(x) == 0:
                db_field = HKFields(field=fields[i], hkfile=db_file, hkagent=db_agent)
                self.session.add(db_field)
            else:
                db_field = db_fields[x[0]]

            db_field.start = starts[i]
            db_field.stop = stops[i]
            db_field.median = medians[i]
            db_field.mean = means[i]
            db_field.min_val = min_vals[i]
            db_field.max_val = max_vals[i]
            db_field.stand_dev = stds[i]
            # update the math incase previous incomplete file?

        self.session.commit()

    def load_data(self, db_instance):
        """Load data from database objects

        Arguments
        ---------
        db_instance: HKFiles, HKAgents, HKFields, or list of one of these objects

        Returns
        -------
        load_range style dictionary. If a list of instances are passed then it
        concatenates matching keys.

        Note: the extra half second on these load_range calls is because
        load_range uses [start,stop) intervals but db_instance.stop is the last
        timestamp in the file for that field (which Katie thinks is correct for
        this specific implementation). Really, we should replace load_range with
        something that doesn't require going through time when we are really
        just asking for "all data from X in file"
        """
        if isinstance(db_instance, HKFields):
            x = hk.load_range(
                db_instance.start,
                db_instance.stop + 0.5,
                fields=[db_instance.field],
                data_dir=self.hkarchive_path,
            )
            return x
        elif isinstance(db_instance, HKAgents):
            fields = [f.field for f in db_instance.fields]
            x = hk.load_range(
                db_instance.start,
                db_instance.stop + 0.5,
                fields=fields,
                data_dir=self.hkarchive_path,
            )
            return x
        elif isinstance(db_instance, HKFiles):
            fields = [f.field for f in db_instance.fields]
            x = hk.load_range(
                min([f.start for f in db_instance.fields]),
                max([f.stop for f in db_instance.fields]) + 0.5,
                fields=fields,
                data_dir=self.hkarchive_path,
            )
            return x
        elif isinstance(db_instance, list):
            temp = [self.load_data(x) for x in db_instance]
            data = {}
            for x in temp:
                for k, item in x.items():
                    if k not in data:
                        data[k] = item
                    else:
                        data[k] = (
                            np.concatenate((data[k][0], item[0])),
                            np.concatenate((data[k][1], item[1])),
                        )
            return data
        else:
            raise ValueError(
                "db_instance must be HKFields, HKFiles, "
                "HKAgents instances or a list of these"
            )

    def get_db_agents(self, instance_id, start, stop):
        """Return list of HKAgents with given instance_id that are active
        for any subset of time between start and stop

        Arguments
        ---------
        instance_id : instance_id of agent
        start : ctime to begin search
        stop : ctime to end search

        Returns
        -------
        list of HKAgent instances
        """
        last = self.get_last_update()
        if stop > last:
            raise ValueError(
                f"stop time {stop} is beyond the last database " f"update of {last}"
            )

        agents = self.session.query(HKAgents).filter(
            db.or_(
                db.and_(HKAgents.start <= start, HKAgents.stop >= stop),
                db.and_(HKAgents.start >= start, HKAgents.start <= stop),
                db.and_(HKAgents.stop >= start, HKAgents.stop <= stop),
            )
        )
        if agents.count() == 0:
            logger.warning(f"no agents found between {start} and {stop}")

        agents = agents.filter(HKAgents.instance_id == instance_id).options(selectinload(HKAgents.fields)).all()
        return agents

    def get_last_update(self):
        """Return the last timestamp present in the database"""
        last_file = (
            self.session.query(HKFiles)
            .order_by(db.desc(HKFiles.global_start_time))
            .first()
        )
        return max([a.stop for a in last_file.agents])

    @classmethod
    def from_configs(cls, configs):
        """
        Create a G3tHK instance from a configs dictionary

        Args
        ----
        configs - dictionary containing `data_prefix` and `g3thk_db` keys
        """
        if type(configs) == str:
            configs = load_configs(configs)

        keys = configs["finalization"]["servers"][0].keys()
        
        iids = []
        for key in keys:
            agent = configs["finalization"]["servers"][0][key]
            iids.append(agent)

        return cls(
            hkarchive_path = os.path.join(configs["data_prefix"], "hk"), 
            db_path = configs["g3thk_db"],
            iids = iids
        )

    def delete_file(self, hkfile, dry_run=False, my_logger=None):
        """WARNING: Removes actual files from file system.

        Delete an hkfile instance, its on-disk file, and all associated agents
        and field entries in the database

        Arguments
        ---------
        hkfile: HKFile instance. Assumes file was queried uses self.session
        """

        if my_logger is None:
            my_logger = logger

        # remove field info
        my_logger.info(f"removing field entries for {hkfile.path} from database")
        if not dry_run:
            for f in hkfile.fields:
                self.session.delete(f)

        # remove agent info
        my_logger.info(f"removing agent entries for {hkfile.path} from database")
        if not dry_run:
            for a in hkfile.agents:
                self.session.delete(a)

        if not os.path.exists(hkfile.path):
            my_logger.warning(f"{hkfile.path} appears already deleted")
        else:
            my_logger.info(f"deleting {hkfile.path}")
            if not dry_run:
                os.remove(hkfile.path)

                base, _ = os.path.split(hkfile.path)
                if len(os.listdir(base)) == 0:
                    os.rmdir(base)

        my_logger.info(f"remove {hkfile.path} from database")
        if not dry_run:
            self.session.delete(hkfile)
            self.session.commit()
