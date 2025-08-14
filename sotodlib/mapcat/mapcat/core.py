"""
Core functionality providing access to the database.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mapcat.database import (
    DepthOneMap,
    DepthOneMapTable,
    ProcessingStatus,
    ProcessingStatusTable,
)


async def create_depth_one(
    map_name: str,
    map_path: str,
    tube_slot: str,
    wafers: str,
    frequency: str,
    ctime: float,
    session: AsyncSession,
) -> DepthOneMap:
    """
    Create a new depth 1 map in the depth 1 map table

    Parameters
    ----------
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
    session : AsyncSession
        Session to use

    Returns
    -------
    dmap.to_model() : DepthOneMap
        DepthOneMap instance initialized with parameters above
    """
    dmap = DepthOneMapTable(
        map_name=map_name,
        map_path=map_path,
        tube_slot=tube_slot,
        wafers=wafers,
        frequency=frequency,
        ctime=ctime,
    )

    async with session.begin():
        session.add(dmap)
        await session.commit()

    return dmap.to_model()


async def get_depth_one(map_id: int, session: AsyncSession) -> DepthOneMap:
    """
    Get a depth 1 map by id.

    Parameters
    ----------
    map_id : int
        Interal map ID number
    session : AsyncSession
        Session to use

    Returns
    -------
    dmap.to_model() : DepthOneMap
        Requested entry for depth 1 map

    Raises
    ------
    ValueError
        If map is not found
    """
    dmap = await session.get(DepthOneMapTable, map_id)

    if dmap is None:
        raise ValueError(f"Depth-1 map with ID {map_id} not found.")

    return dmap.to_map()


async def update_depth_one(
    map_id: int,
    map_name: str | None,
    map_path: str | None,
    tube_slot: str | None,
    wafers: str | None,
    frequency: str | None,
    ctime: str | None,
    session: AsyncSession,
) -> DepthOneMap:
    """
    Update a depth one map.

    Parameters
    ----------
    map_id : int
        ID of map to update
    map_name : str | None
        Name of depth 1 map
    map_path : str | None
        Non-localized path to str
    tube_slot : str | None
        OT for map
    wafers : str | None
        Standardized names of wafers used in this map
    frequency : str | None
        Frequency channel of map
    ctime : float | None
        Central unix time of map
    session : AsyncSession
        Session to use

    Returns
    -------
    dmap.to_model() : DepthOneMap
        Updated depth 1 map

    Raises
    ------
    ValueError
        If the map is not found
    """
    async with session.begin():
        dmap = await session.get(DepthOneMapTable, map_id)

        if dmap is None:
            raise ValueError(f"Depth 1 map with ID {map_id} not found.")
        dmap.map_name = map_name if map_name is not None else dmap.map_name
        dmap.map_path = map_path if map_path is not None else dmap.map_path
        dmap.tube_slot = tube_slot if tube_slot is not None else dmap.tube_slot
        dmap.wafers = wafers if wafers is not None else dmap.wafers
        dmap.frequency = frequency if frequency is not None else dmap.frequency
        dmap.ctime = ctime if ctime is not None else dmap.ctime

        await session.commit()

    return dmap.to_model()


async def delete_depth_one(map_id: int, session: AsyncSession) -> None:
    """
    Delete a depth one map from the database

    Parameters
    ----------
    map_id : int
        ID of depth one map
    session : AsyncSession
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the source is not found
    """
    async with session.begin():
        dmap = await session.get(DepthOneMapTable, map_id)

        if dmap is None:
            raise ValueError(f"Depth 1 map with ID {map_id} not found.")

        await session.delete(dmap)
        await session.commit()

    return


async def create_proccessing_status(
    map_name: str,
    processing_start: float | None,
    processing_end: float | None,
    processing_status: str,
    session: AsyncSession,
) -> ProcessingStatus:
    """
    Create a entry tracking depth one map processing status.

    Parameters
    ----------
    map_name : str
        Name of depth 1 map to track
    processing_start : float | None
        Time of processing start
    processing_end : float | None
        Time of processing end
    processing_status : str
        Status of processing
    session : AsyncSession
        Session to use

    Returns
    -------
    proc_stat.to_model() : ProcessingStatus
        ProcessingStatus instance initialized with parameters above
    """
    proc_stat = ProcessingStatusTable(
        map_name=map_name,
        processing_start=processing_start,
        processing_end=processing_end,
        processing_status=processing_status,
    )

    async with session.begin():
        session.add(proc_stat)
        await session.commit()

    return proc_stat.to_model()


async def get_proccessing_status(
    map_name: int, session: AsyncSession
) -> ProcessingStatus:
    """
    Get a depth 1 map by id.

    Parameters
    ----------
    map_name : int
        Map name to get status of
    session : AsyncSession
        Session to use

    Returns
    -------
    proc_stat.to_service() : ProcessingStatus
        Requested entry for depth 1 map

    Raises
    ------
    ValueError
        If map is not found
    """
    proc_stat = await session.get(ProcessingStatusTable, map_name)

    if proc_stat is None:
        raise ValueError(f"Depth-1 map with name {map_name} not found.")

    return proc_stat.to_map()


async def update_processing_status(
    map_name: str,
    processing_start: float | None,
    processing_end: float | None,
    processing_status: str | None,
    session: AsyncSession,
) -> ProcessingStatus:
    """
    Update a depth one map.

    Parameters
    ----------
    map_name : str
        Name of depth 1 map to track
    processing_start : float | None
        Time of processing start
    processing_end : float | None
        Time of processing end
    processing_status : str
        Status of processing
    session : AsyncSession
        Session to use

    Returns
    -------
    proc_stat.to_model() : ProcessingStatus
        ProcessingStatus instance updated with parameters above

    Raises
    ------
    ValueError
        If the processing status is not found
    """
    async with session.begin():
        proc_stat = await session.get(ProcessingStatusTable, map_name)

        if proc_stat is None:
            raise ValueError(f"Depth 1 map with name {map_name} not found.")
        proc_stat.processing_start = (
            processing_start
            if processing_start is not None
            else proc_stat.processing_start
        )
        proc_stat.processing_end = (
            processing_end if processing_end is not None else proc_stat.processing_end
        )
        proc_stat.processing_status = (
            processing_status
            if processing_status is not None
            else proc_stat.processing_status
        )

        await session.commit()

    return proc_stat.to_model()


async def delete_processing_status(map_name: str, session: AsyncSession) -> None:
    """
    Delete a processing status of a map from database

    Parameters
    ----------
    map_name : str
        Name of depth one map
    session : AsyncSession
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the processing status is not found
    """
    async with session.begin():
        proc_stat = await session.get(ProcessingStatusTable, map_name)

        if proc_stat is None:
            raise ValueError(f"Depth 1 map with name {map_name} not found.")

        await session.delete(proc_stat)
        await session.commit()

    return
