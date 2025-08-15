"""
Core database tables storing information about sources.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import JSON, Column, Field, SQLModel

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
    """

    __tablename__ = "depth_one_maps"

    id: int = Field(primary_key=True)
    map_name: str = Field(index=True, nullable=False)
    tube_slot: str = Field(index=True, nullable=False)
    frequency: str = Field(index=True, nullable=False)
    ctime: float = Field(index=True, nullable=False)

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
    Processing status of a depth-1 map

    Attributes
    ----------
    map_name : str
        Name of depth 1 map being tracked. Primary Key, foreign into DepthOneMap
    processing_start : float
        Time processing started.
    processing_end : float
        Time processing Eneded
    processing_status : str
        Status of processing
    """

    map_name: str
    processing_start: float | None
    processing_end: float | None
    processing_status: str

    def __repr__(self):
        # TODO: This should probably change based on process status once we nail that down
        return f"ProcessingStatus(map_name={self.map_name}, processing_start={self.processing_start}, processing_end={self.processing_end}, processing_status={self.processing_status})"  # pragma: no cover


class ProcessingStatusTable(ProcessingStatus, SQLModel, table=True):
    """
    Table for tracking processing status of depth-1 maps
    providing SQLModel functionality. You can export a base model, for example
    for responding to a query with using the `to_model` method. Note some attributes
    are inherited from ProcessingStatus.

    Attributes
    ----------
    map_name : str
        Name of depth 1 map being tracked. Primary Key, foreign into DepthOneMap
    processing_start : float | None
        Time processing started. None if not started.
    processing_end : float | None
        Time processing ended. None if not ended.
    processing_status : str
        Status of processing
    """

    __tablename__ = "processing_status"

    map_name: str = Field(
        primary_key=True,
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_name",
    )
    processing_start: float = Field(nullable=True)
    processing_end: float = Field(nullable=True)
    processing_status: str = Field(index=True, nullable=False)

    def to_model(self) -> ProcessingStatus:
        """
        Return an Extragalactic source from table.

        Returns
        -------
        ProcessingStatus : ProcessingStatus
            Processing status corresponding to this depth-1 map
        """
        return ProcessingStatus(
            map_name=self.map_name,
            processing_start=self.processing_start,
            processing_end=self.processing_end,
            processing_status=self.cProcessingStatus,
        )


ALL_TABLES = [DepthOneMapTable, ProcessingStatusTable]

async_engine = create_async_engine(settings.database_url, echo=True, future=True)


async def get_async_session() -> AsyncSession:
    async_session = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
