"""
The web API to access the mapcat database.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

import mapcat.core as core

from .database import (
    ALL_TABLES,
    DepthOneMap,
    ProcessingStatus,
    async_engine,
    get_async_session,
)


async def lifespan(f: FastAPI):
    # Use SQLModel to create the tables.
    print("Creating tables")
    for table in ALL_TABLES:
        print("Creating table", table)
        async with async_engine.begin() as conn:
            await conn.run_sync(table.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)

router = APIRouter(prefix="/api/v1")

SessionDependency = Annotated[AsyncSession, Depends(get_async_session)]


class DepthOneMapModificationRequest(BaseModel):
    """
    Class which defines which Depth 1 map attributes are available to modify.

    Attributes
    ----------
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
    """

    map_name: str | None
    map_path: str | None
    tube_slot: str | None
    wafers: str | None
    frequency: str | None
    ctime: float | None


class ProcessingStatusModificationRequest(BaseModel):
    """
    Class which defines with processing status attributes are avilable to modify.

    Attributes
    ----------
    map_name : float | None
        Name of map being tracked
    processing_start : float | None
        Time processing started.
    processing_end : float | None
        Time processing Eneded
    processing_status : str | None
        Status of processing
    """

    map_name: str | None
    processing_start: float | None
    processing_end: float | None
    processing_status: str | None


@router.put("/depthone/new")  # TODO : path?
async def create_depth_one(
    model: DepthOneMapModificationRequest,
    session: SessionDependency,
) -> DepthOneMap:
    """
    Create a new depth one map in the catalog.

    Parameters
    ----------
    model : DepthOneMapModificationRequest
        Class containing modifyable attributes of the map
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : DepthOneMap
        DepthOneMap which was specified by model and added to the database

    Raise
    -----
    HTTPException
        If the model does not contain required info or api response is malformed
    """

    try:
        response = await core.create_depth_one(
            map_name=model.map_name,
            map_path=model.map_path,
            tube_slot=model.tube_slot,
            wafers=model.wafers,
            frequency=model.frequency,
            ctime=model.ctime,
        )
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())

    return response


@router.get("/depthone/{map_id}")
async def get_depth_one(map_id: int, session: SessionDependency) -> DepthOneMap:
    """
    Get a depth 1 map by id

    Parameters
    ----------
    map_id : int
        ID of map to get
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : DepthOneMap
        DepthOneMap corresponding to map_id

    Raises
    ------
    HTTPException
        If map_id does not correspond to any depth 1 map
    """
    try:
        response = await core.get_depth_one(map_id=map_id, session=session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.post("/depthone/{map_id}")
async def update_depth_one(
    map_id: int,
    model: DepthOneMapModificationRequest,
    session: SessionDependency,
) -> DepthOneMap:
    """
    Update depth 1 map parameters by id

    Parameters
    ----------
    map_id : int
        ID of depth one map to update
    model : DepthOneMapModificationRequest
        Parameters of map to modify
    session : SessionDependency
        Session to use

    Returns
    -------
    response : DepthOneMap
        DepthOneMap that has been modified

    Raises
    ------
    HTTPException
        If ID does not correspond to any source
    """

    try:
        response = await core.update_depth_one(
            map_id=map_id,
            map_name=model.map_name,
            map_path=model.map_path,
            tube_slot=model.tube_slot,
            wafers=model.wafers,
            frequency=model.frequency,
            ctime=model.ctime,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.delete("/depthone/{map_id}")
async def delete_depth_one(map_id: int, session: SessionDependency) -> None:
    """
    Delete a depth one map by ID

    Parameters
    ----------
    map_id : int
        ID of depth one map to delete
    session : SessionDependency
        Session to use

    Returns
    -------
    None

    Raises
    ------
    HTTPException
        If map_id does not correspond to any depth one map
    """
    try:
        await core.delete_depth_one(map_id=map_id, session=session)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return


@router.put("/procstat/new")  # TODO : path?
async def create_processing_status(
    model: ProcessingStatusModificationRequest,
    session: SessionDependency,
) -> ProcessingStatus:
    """
    Create a new processing status for a depth one map in the catalog.

    Parameters
    ----------
    model : ProcessingStatusModificationRequest
        Class containing modifyable attributes of the map
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : ProcessingStatus
        ProcessingStatus which was specified by model and added to the database

    Raise
    -----
    HTTPException
        If the model does not contain required info or api response is malformed
    """

    try:
        response = await core.create_proccessing_status(
            map_name=model.map_name,
            processing_start=model.processing_start,
            processing_end=model.processing_end,
            processing_status=model.processing_status,
            session=session,
        )
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())

    return response


@router.get("/procstat/{map_name}")
async def get_depth_one(map_name: str, session: SessionDependency) -> ProcessingStatus:
    """
    Get a depth 1 map by id

    Parameters
    ----------
    map_name : str
        Name of map whos processing status to get
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : ProcessingStatus
        ProcessingStatus corresponding to map_name

    Raises
    ------
    HTTPException
        If map_name does not correspond to any depth 1 map processing status
    """
    try:
        response = await core.get_proccessing_status(map_name=map_name, session=session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.post("/procstat/{map_name}")
async def update_processing_status(
    model: ProcessingStatusModificationRequest,
    session: SessionDependency,
) -> ProcessingStatus:
    """
    Update depth 1 map parameters by id

    Parameters
    ----------
    model : ProcessingStatusModificationRequest
        Parameters of processing status to modify
    session : SessionDependency
        Session to use

    Returns
    -------
    response : ProcessingStatus
        ProcessingStatus that has been modified

    Raises
    ------
    HTTPException
        If ID does not correspond to any source
    """

    try:
        response = await core.update_processing_status(
            map_name=model.map_name,
            processing_start=model.processing_start,
            processing_end=model.processing_end,
            processing_status=model.processing_status,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.delete("/procstat/{map_name}")
async def delete_processing_status(map_name: str, session: SessionDependency) -> None:
    """
    Delete a depth one map by ID

    Parameters
    ----------
    map_name : str
        Name of depth one map whos processing status to delete
    session : SessionDependency
        Session to use

    Returns
    -------
    None

    Raises
    ------
    HTTPException
        If map_name does not correspond to any depth one map
    """
    try:
        await core.delete_processing_status(map_name=map_name, session=session)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return


app.include_router(router)
