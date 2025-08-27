"""
Core database tables storing information about depth one maps.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sqlmodel import JSON, Column, Field, SQLModel, Relationship

from .settings import settings


class DepthOneMap(BaseModel):
    """
    A Depth 1 map.

    Attributes
    ----------
    id : int
        Unique map identifier. Internal to SO
    map_name : str
        Name of depth 1 map
    map_path : str
        Non-localized path to str
    tube_slot : str
        OT for map
    wafers : str
        Standardized names of wafers used in this map
    frequency : str
        Frequency channel of map
    ctime : float
        Central unix time of map
    """

    id: int
    map_name: str
    map_path: str
    tube_slot: str
    wafers: str
    frequency: str
    ctime: float

    def __repr__(self):
        return f"DepthOneMap(id={self.id}, map_name={self.map_name}, map_path={self.map_path}, tube_slot={self.tube_slot}, wafers={self.wafers}, frequency={self.frequency}, ctime={self.ctime})"  # pragma: no cover


class DepthOneMapTable(DepthOneMap, SQLModel, table=True):
    """
    A depth-1 map. This is the table model
    providing SQLModel functionality. You can export a base model, for example
    for responding to a query with using the `to_model` method. Note some attributes
    are inherited from DepthOneMap.

    Attributes
    ----------
    id : int
        Unique map identifiers. Internal to SO
    map_name : str
        Name of depth 1 map
    map_path : str
        Non-localized path to str
    tube_slot : str
        OT for map
    wafers : str
        Standardized names of wafers used in this map
    frequency : str
        Frequency channel of map
    ctime : float
        Central unix time of map
    processing_status : list[ProcessingStatusTable]
        List of processing status tables associated with d1 map
    pointing_residual : list[PointingResidualTable]
        List of pointing residual table associated with d1 map
    """

    __tablename__ = "depth_one_maps"

    id: int = Field(primary_key=True)
    map_name: str = Field(index=True, unique=True, nullable=False)
    tube_slot: str = Field(index=True, nullable=False)
    frequency: str = Field(index=True, nullable=False)
    ctime: float = Field(index=True, nullable=False)

    processing_status: list["ProcessingStatusTable"] = Relationship(
        back_populates="dmap", cascade_delete=True
    )
    pointing_residual: list["PointingResidualTable"] = Relationship(
        back_populates="dmap", cascade_delete=True
    )
    tods: list["TODDepthOneTable"] = Relationship(
        back_populates="dmap", cascade_delete=True
    )

    def to_model(self) -> DepthOneMap:
        """
        Return an depth one map from table.

        Returns
        -------
        DepthOneMap : DepthOneMap
            Map corresponding to this id.
        """
        return DepthOneMap(
            id=self.id,
            map_name=self.map_name,
            map_path=self.map_path,
            tube_slot=self.tube_slot,
            wafers=self.wafers,
            frequency=self.frequency,
            ctime=self.ctime,
        )


class ProcessingStatus(BaseModel):
    """
    Processing status of a depth-1 map. Note a map
    can have multiple processing statuses

    Attributes
    ----------
    id : str
        Internal ID of the processing status
    map_name : str
        Name of depth 1 map being tracked. Primary Key, foreign into DepthOneMap
    processing_start : float
        Time processing started.
    processing_end : float
        Time processing Eneded
    processing_status : str
        Status of processing
    """

    id: int
    map_name: str
    processing_start: float | None
    processing_end: float | None
    processing_status: str

    def __repr__(self):
        # TODO: This should probably change based on process status once we nail that down
        return f"ProcessingStatus(id={self.id}, map_name={self.map_name}, processing_start={self.processing_start}, processing_end={self.processing_end}, processing_status={self.processing_status})"  # pragma: no cover


class ProcessingStatusTable(ProcessingStatus, SQLModel, table=True):
    """
    Table for tracking processing status of depth-1 maps
    providing SQLModel functionality. You can export a base model, for example
    for responding to a query with using the `to_model` method. Note some attributes
    are inherited from ProcessingStatus.

    Attributes
    ----------
    id : int
        Internal ID of the processing status
    map_name : str
        Name of depth 1 map being tracked. Foreign into DepthOneMap
    processing_start : float | None
        Time processing started. None if not started.
    processing_end : float | None
        Time processing ended. None if not ended.
    processing_status : str
        Status of processing
    """

    __tablename__ = "time_domain_processing"
    id: int = Field(primary_key=True)
    map_name: str = Field(
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_name",
        ondelete="CASCADE",
    )
    processing_start: float = Field(nullable=True)
    processing_end: float = Field(nullable=True)
    processing_status: str = Field(index=True, nullable=False)
    dmap: DepthOneMapTable = Relationship(back_populates="processing_status")

    def to_model(self) -> ProcessingStatus:
        """
        Return an processing from table.

        Returns
        -------
        ProcessingStatus : ProcessingStatus
            Processing status corresponding to this depth-1 map
        """
        return ProcessingStatus(
            id=self.id,
            map_name=self.map_name,
            processing_start=self.processing_start,
            processing_end=self.processing_end,
            processing_status=self.processing_status,
        )


class PointingResidual(BaseModel):
    """
    Pointing error for a depth one map, computed by
    comparing positions of PSes in that map to
    their known possitions

    Attributes
    ----------
    id : str
        Internal ID of the pointing error
    map_name : str
        Name of depth 1 map being tracked. Foreign into DepthOneMap
    ra_offset : float
        Calculated ra offset of PSes
    dec_offset : float
        Calculated dec offset of PSes
    """

    id: int
    map_name: str
    ra_offset: float | None
    dec_offset: float | None

    def __repr__(self):
        # TODO: This should probably change based on process status once we nail that down
        return f"PointingError(id={self.id}, map_name={self.map_name} has offset ra={self.ra_offset}, dec={self.dec_offset})"  # pragma: no cover


class PointingResidualTable(PointingResidual, SQLModel, table=True):
    """
    Table for tracking Pointing error for a depth one map,
    computed by comparing positions of PSes in that map to
    their known possitions

    Attributes
    ----------
    id : str
        Internal ID of the pointing error
    map_name : str
        Name of depth 1 map being tracked. Foreign into DepthOneMap
    ra_offset : float
        Calculated ra offset of PSes
    dec_offset : float
        Calculated dec offset of PSes
    """

    __tablename__ = "depth_1_pointing_residuals"
    id: int = Field(primary_key=True)
    map_name: str = Field(
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_name",
        ondelete="CASCADE",
    )
    ra_offset: float = Field(nullable=True)
    dec_offset: float = Field(nullable=True)
    dmap: DepthOneMapTable = Relationship(back_populates="pointing_residual")

    def to_model(self) -> PointingResidual:
        """
        Return an pointing residual from table.

        Returns
        -------
        PointingResidual : PointingResidual
            pointing residual corresponding to this depth-1 map
        """
        return PointingResidual(
            id=self.id,
            map_name=self.map_name,
            ra_offset=self.ra_offset,
            dec_offset=self.dec_offset,
        )


class TODDepthOne(BaseModel):
    """
    A TOD.

    Attributes
    ----------
    id : int
        Unique TOD identifier. Internal to SO
    map_name : str
        Name of map this TOD went into. Foreign key
    obs_id : str
        SO ID of TOD
    pwv : float
        Precipitable  water vapor at time of obs
    ctime : float
        Mean unix time of obs
    start_time : float
        Start time of obs
    stop_time : float
        End time of obs
    nsamples : int
        Number of samps in obs
    telescope : str
        Telescope making obs
    telescope_flavor : str
        Telescope LF/MF/UHF. Only for SATs
    tube_slot : str
        Tube of obs. Only for LAT
    tube_flavor : str
        LF/MF/UHF of tube. Only for LAT
    frequency : str
        Frequency of obs
    scan_type : str
        Type of scan.
    subtype : str
        Subtype of scan
    wafer_count : int
        Number of working wafers for scan
    duration : float
        Duration of scan in seconds
    az_center : float
        Az center of scan
    az_throw : float
        Az throw of scan
    el_center : float
        El center of scan
    el_throw : float
        El throw of scan
    roll_center : float
        Roll center of scan
    roll_throw : float
        Roll throw of scan
    wafer_slots_list : str
        List of live wafers for scan
    stream_ids_list : str
        Stream IDs live for scan
    """

    id: int
    map_name: str
    obs_id: str
    pwv: float
    ctime: float
    start_time: float
    stop_time: float
    nsamples: int
    telescope: str
    telescope_flavor: str
    tube_slot: str
    tube_flavor: str
    frequency: str
    scan_type: str
    subtype: str
    wafer_count: int
    duration: float
    az_center: float
    az_throw: float
    el_center: float
    el_throw: float
    roll_center: float
    roll_throw: float
    wafer_slots_list: str
    stream_ids_list: str

    def __iter__(self):
        """
        Iterable method for returning all attributes of class
        """
        for key in vars(self).keys():
            yield key, vars(self).keys()

    def __repr__(self):
        msg = f"TODDepthOne(id={self.id}, "
        for attr, value in self.__iter__():
            msg += f"{attr}={value}"
        msg += ")"
        return msg  # pragma: no cover


class TODDepthOneTable(TODDepthOne, SQLModel, table=True):
    """
    Table of TODs used in making depth 1 maps.

    Attributes
    ----------
    id : int
        Unique TOD identifier. Internal to SO
    map_name : str
        Name of map this TOD went into. Foreign key
    obs_id : str
        SO ID of TOD
    pwv : float
        Precipitable  water vapor at time of obs
    ctime : float
        Mean unix time of obs
    start_time : float
        Start time of obs
    stop_time : float
        End time of obs
    nsamples : int
        Number of samps in obs
    telescope : str
        Telescope making obs
    telescope_flavor : str
        Telescope LF/MF/UHF. Only for SATs
    tube_slot : str
        Tube of obs. Only for LAT
    tube_flavor : str
        LF/MF/UHF of tube. Only for LAT
    frequency : str
        Frequency of obs
    scan_type : str
        Type of scan.
    subtype : str
        Subtype of scan
    wafer_count : int
        Number of working wafers for scan
    duration : float
        Duration of scan in seconds
    az_center : float
        Az center of scan
    az_throw : float
        Az throw of scan
    el_center : float
        El center of scan
    el_throw : float
        El throw of scan
    roll_center : float
        Roll center of scan
    roll_throw : float
        Roll throw of scan
    wafer_slots_list : str
        List of live wafers for scan
    stream_ids_list : str
        Stream IDs live for scan
    """

    __tablename__ = "tod_depth_one"
    id: int = Field(primary_key=True)
    map_name: str = Field(
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_name",
        ondelete="CASCADE",
    )
    obs_id: str = Field(nullable=False)
    pwv: float = Field(index=True, nullable=True)
    ctime: float = Field(index=True, nullable=False)
    start_time: float = Field(index=True, nullable=False)
    stop_time: float = Field(index=True, nullable=False)
    nsamples: int = Field()
    telescope: str = Field(index=True, nullable=False)
    telescope_flavor: str = Field()
    tube_slot: str = Field()
    tube_flavor: str = Field()
    frequency: str = Field(index=True, nullable=False)
    scan_type: str = Field()
    subtype: str = Field()
    wafer_count: int = Field(index=True, nullable=False)
    duration: float = Field()
    az_center: float = Field()
    az_throw: float = Field()
    el_center: float = Field()
    el_throw: float = Field()
    roll_center: float = Field()
    roll_throw: float = Field()
    wafer_slots_list: str = Field(nullable=False)
    stream_ids_list: str = Field(nullable=False)
    dmap: DepthOneMapTable = Relationship(back_populates="tods")

    def to_model(self) -> TODDepthOne:
        """
        Return an tod from table.

        Returns
        -------
        TODDepthOne : TODDepthOne
            TOD
        """
        return TODDepthOne(
            id=self.id,
            map_name=self.map_name,
            obs_id=self.obs_id,
            pwv=self.pwv,
            ctime=self.ctime,
            start_time=self.start_time,
            stop_time=self.stop_time,
            nsamples=self.nsamples,
            telescope=self.telescope,
            telescope_flavor=self.telescope_flavor,
            tube_slot=self.tube_slot,
            tube_flavor=self.tube_flavor,
            frequency=self.frequency,
            scan_type=self.scan_type,
            subtype=self.subtype,
            wafer_count=self.wafer_count,
            duration=self.duration,
            az_center=self.az_center,
            az_throw=self.az_throw,
            el_center=self.el_center,
            el_throw=self.el_throw,
            roll_center=self.roll_center,
            roll_throw=self.roll_throw,
            wafer_slots_list=self.wafer_slots_list,
            stream_ids_list=self.stream_ids_list,
        )


ALL_TABLES = [
    DepthOneMapTable,
    ProcessingStatusTable,
    PointingResidualTable,
    TODDepthOneTable,
]


from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlite3 import Connection as SQLite3Connection


# This snippet turns on foreign keys if using SQLite
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


engine = create_engine(
    settings.sync_database_url,
    echo=True,
    future=True,
)


def get_session() -> Session:  # pragma: no cover
    session_maker = sessionmaker(bind=engine, expire_on_commit=False)
    with session_maker() as session:
        yield session
