"""
Core database tables storing information about sources.
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
        Unique source identifier. Internal to SO
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
        Unique source identifiers. Internal to SO
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

    def to_model(self) -> DepthOneMap:
        """
        Return an Extragalactic source from table.

        Returns
        -------
        ExtragalaticSource : DepthOneMap
            Source corresponding to this id.
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
        Return an Extragalactic source from table.

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
        Return an Extragalactic source from table.

        Returns
        -------
        ProcessingStatus : ProcessingStatus
            Processing status corresponding to this depth-1 map
        """
        return PointingResidual(
            id=self.id,
            map_name=self.map_name,
            ra_offset=self.ra_offset,
            dec_offset=self.dec_offset,
        )


ALL_TABLES = [DepthOneMapTable, ProcessingStatusTable, PointingResidualTable]


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


def get_session() -> Session:
    session_maker = sessionmaker(bind=engine, expire_on_commit=False)
    with session_maker() as session:
        yield session
