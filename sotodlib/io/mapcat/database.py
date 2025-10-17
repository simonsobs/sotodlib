"""
Core database tables storing information about depth one maps.
"""

from typing import Any
from sqlmodel import  Field, SQLModel, Relationship, JSON


class TODToMapTable(SQLModel, table=True):
    """
    Link table for many-to-many relationship between TODs and depth-1 maps.
    """

    __tablename__ = "link_tod_to_depth_one_map"

    tod_id: int = Field(
        foreign_key="tod_depth_one.tod_id",
        primary_key=True,
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )
    map_id: int = Field(
        foreign_key="depth_one_maps.map_id",
        primary_key=True,
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )


class DepthOneToCoaddTable(SQLModel, table=True):
    """
    Link table for many-to-many relationship between depth-1 maps and coadds.
    """

    __tablename__ = "link_depth_one_map_to_coadd"

    map_id: int = Field(
        foreign_key="depth_one_maps.map_id",
        primary_key=True,
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )
    coadd_id: int = Field(
        foreign_key="depth_one_coadds.coadd_id",
        primary_key=True,
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )


class DepthOneCoaddTable(SQLModel, table=True):
    """
    A co-add of multiple depth-1 maps. This is the table model,
    but this many-to-many relationship relies on the join table.
    """

    __tablename__ = "depth_one_coadds"

    coadd_id: int = Field(primary_key=True)
    coadd_name: str = Field(nullable=False)
    coadd_type: str = Field(nullable=False)
    coadd_path: str
    frequency: str = Field(nullable=False)
    ctime: float = Field(nullable=False)
    start_time: float = Field(nullable=False)
    stop_time: float = Field(nullable=False)

    maps: list["DepthOneMapTable"] = Relationship(
        back_populates="coadds",
        link_model=DepthOneToCoaddTable,
    )


class DepthOneMapTable(SQLModel, table=True):
    """
    A depth-1 map.

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
        Mean unix time of map
    start_time : float
        Start unix time of map
    end_time : float
        End unix time of map
    processing_status : list[ProcessingStatusTable]
        List of processing status tables associated with d1 map
    pointing_residual : list[PointingResidualTable]
        List of pointing residual table associated with d1 map
    tods: list[TODDepthOneTable]
        List of tods associated with d1 map
    pipeline_information: list[PipelineInformationTable]
        List of pipeline info associed with d1 map
    depth_one_sky_coverage : list[SkyCoverageTable]
        List of sky coverage patches for d1 map.
    """

    __tablename__ = "depth_one_maps"

    map_id: int = Field(primary_key=True)
    map_name: str = Field(index=True, unique=True, nullable=False)
    map_path: str
    tube_slot: str = Field(index=True, nullable=False)
    frequency: str = Field(index=True, nullable=False)
    ctime: float = Field(index=True, nullable=False)
    start_time: float = Field(index=True, nullable=False)
    stop_time: float = Field(index=True, nullable=False)

    processing_status: list["ProcessingStatusTable"] = Relationship(
        back_populates="map",
        cascade_delete=True,
    )
    pointing_residual: list["PointingResidualTable"] = Relationship(
        back_populates="map",
        cascade_delete=True,
    )
    tods: list["TODDepthOneTable"] = Relationship(
        back_populates="maps",
        link_model=TODToMapTable,
    )
    pipeline_information: list["PipelineInformationTable"] = Relationship(
        back_populates="map",
        cascade_delete=True,
    )
    depth_one_sky_coverage: list["SkyCoverageTable"] = Relationship(
        back_populates="map",
        cascade_delete=True,
    )
    coadds: list["DepthOneCoaddTable"] = Relationship(
        back_populates="maps",
        link_model=DepthOneToCoaddTable,
    )


class ProcessingStatusTable(SQLModel, table=True):
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

    processing_status_id: int = Field(primary_key=True)

    map_id: int = Field(
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_id",
        ondelete="CASCADE",
    )
    map: DepthOneMapTable = Relationship(back_populates="processing_status")

    processing_start: float = Field(nullable=True)
    processing_end: float = Field(nullable=True)
    processing_status: str = Field(index=True, nullable=False)


class PointingResidualTable(SQLModel, table=True):
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

    __tablename__ = "depth_one_pointing_residuals"
    pointing_residual_id: int = Field(primary_key=True)

    map_id: int = Field(
        index=True,
        nullable=False,
        foreign_key="depth_one_maps.map_id",
        ondelete="CASCADE",
    )

    ra_offset: float = Field(nullable=True)
    dec_offset: float = Field(nullable=True)
    map: DepthOneMapTable = Relationship(back_populates="pointing_residual")


class TODDepthOneTable(SQLModel, table=True):
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
    tod_id: int = Field(primary_key=True)
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
    maps: list[DepthOneMapTable] = Relationship(
        back_populates="tods", link_model=TODToMapTable
    )


class PipelineInformationTable(SQLModel, table=True):
    """
    Table for tracking processing information for a depth one map.

    Attributes
    ----------
    id : str
        Internal ID of the pipeline info
    map_name : str
        Name of depth 1 map being tracked. Foreign into DepthOneMap
    sotodlib_version : str
        Version of sotodlib used to make the map
    map_maker : str
        Mapmaker used to make the map
    preprocess_info : dict[str, Any]
        JSON of any additional preprocessing info
    """

    __tablename__ = "pipeline_information"

    pipeline_information_id: int = Field(primary_key=True)
    map_id: int = Field(foreign_key="depth_one_maps.map_id", nullable=False)
    map: DepthOneMapTable = Relationship(back_populates="pipeline_information")

    sotodlib_version: str
    map_maker: str
    preprocess_info: dict[str, Any] = Field(sa_type=JSON)

    map: DepthOneMapTable = Relationship(back_populates="pipeline_information")


class SkyCoverageTable(SQLModel, table=True):
    """
    Table for tracking sky coverage for a depth one map. x and y are 0->36 and
    0-18 respectively for CAR patches with 10x10 degrees each.

    Attributes
    ----------
    id : str
        Internal ID of the sky coverage
    map_name : str
        Name of depth 1 map being tracked. Foreign into DepthOneMap
    patch_coverage : str
        String which represents the sky coverage of the d1map
    """

    __tablename__ = "depth_one_sky_coverage"

    patch_id: int = Field(primary_key=True)

    x: int = Field(index=True)
    y: int = Field(index=True)

    map_id: int = Field(
        foreign_key="depth_one_maps.map_id", nullable=False, ondelete="CASCADE"
    )
    map: DepthOneMapTable = Relationship(back_populates="depth_one_sky_coverage")


ALL_TABLES = [
    TODToMapTable,
    DepthOneMapTable,
    ProcessingStatusTable,
    PointingResidualTable,
    TODDepthOneTable,
    PipelineInformationTable,
    SkyCoverageTable,
]
