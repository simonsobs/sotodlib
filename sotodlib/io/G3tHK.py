import sqlalchemy as db

from sqlalchemy import ForeignKey
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

import os
from so3g import hk
import datetime

Base = declarative_base()
Session = sessionmaker()

# all database definitions for G3tHK
class HKFiles(Base):
    """This table is named hkfeeds and serves as a db for holding
    brief information about each HK'ing file.

    Attributes
    _________
    filename : string
        name of HK file
    global_start_time : integer
        timestamp that begins a new HK file
    path : string
        path to the HK file
    """
    __tablename__ = 'hkfeeds'
    __table_args__ = (
            db.UniqueConstraint('filename'),
            db.UniqueConstraint('global_start_time'),
    )

    id = db.Column(db.Integer, primary_key=True)
    fields = relationship("HKFields", back_populates='hkfile')
    filename = db.Column(db.String, nullable=False, unique=True)
    global_start_time = db.Column(db.Integer)
    path = db.Column(db.String)
    # scanned
    # provider id
    # description


class HKFields(Base):
    """This table is named hkfields and serves as a db for all
    fields inside each HK'ing file in the hkfeeds table.

    Attributes
    _________
    feed_id : integer
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
    feed_id = db.Column(db.Integer, db.ForeignKey('hkfiles.id'))
    hkfeed = relationship("HKFiles", back_populates='fields')
    field = db.Column(db.String)
    start = db.Column(db.Integer)
    end = db.Column(db.Integer)


class G3tHK:
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

    def add_hkfiles(self, hkarchive_path, hkfile):
        """
        """
        global_start_time = hkfile.split(".")[0]
        global_start_time = int(global_start_time)

        db_file = HKFeeds(filename=hkfile,
                          path=hkarchive_path,
                          global_start_time=global_start_time)

        self.session.add(db_file)
        self.session.commit()

    def add_hkfields(self):
        """
        """
        feed_list = self.session.query(HKFeeds).all()

        for feed in feed_list:
            fields, starts, ends = self.load_fields(feed.path, feed.filename)

            for i in range(len(fields)):
                db_file = HKFields(field=fields[i],
                                   start=starts[i],
                                   end=ends[i],
                                   hkfeed=feed)
                self.session.add(db_file)

        self.session.commit()
        # TODO: can be more efficient; will repeat fields as it loops through every HK file every time you run
