"""
The web API to access the mapcat database.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

import sotodlib.mapcat.mapcat.core as core

from .database import (
    ALL_TABLES,
    DepthOneMap,
    ProcessingStatus,
    PointingResidual,
    engine,
    get_session,
)


def lifespan(f: FastAPI):
    # Use SQLModel to create the tables.
    print("Creating tables")
    for table in ALL_TABLES:
        print("Creating table", table)
        with engine.begin() as conn:
            conn.run_sync(table.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)

router = APIRouter(prefix="/api/v1")

SessionDependency = Annotated[AsyncSession, Depends(get_session)]


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


class PointingResidualModificationRequest(BaseModel):
    """
    Class which defines with pointing residual attributes are avilable to modify.

    Attributes
    ----------
    map_name : float | None
        Name of map whos pointing residual is being tracked
    ra_offset : float
        Calculated ra offset of PSes
    dec_offset : float
        Calculated dec offset of PSes
    """

    map_name: str | None
    ra_offset: float | None
    dec_offset: float | None


@router.put("/depthone/new")  # TODO : path?
def create_depth_one(
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
        response = core.create_depth_one(
            map_name=model.map_name,
            map_path=model.map_path,
            tube_slot=model.tube_slot,
            wafers=model.wafers,
            frequency=model.frequency,
            ctime=model.ctime,
            session=session,
        )
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())

    return response


@router.get("/depthone/{map_id}")
def get_depth_one(map_id: int, session: SessionDependency) -> DepthOneMap:
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
        response =  core.get_depth_one(map_id=map_id, session=session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.post("/depthone/{map_id}")
def update_depth_one(
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
        response = core.update_depth_one(
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
def delete_depth_one(map_id: int, session: SessionDependency) -> None:
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
        core.delete_depth_one(map_id=map_id, session=session)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return


@router.put("/procstat/new")  # TODO : path?
def create_processing_status(
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
        response = core.create_proccessing_status(
            map_name=model.map_name,
            processing_start=model.processing_start,
            processing_end=model.processing_end,
            processing_status=model.processing_status,
            session=session,
        )
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())

    return response


@router.get("/procstat/{proc_id}")
def get_processing_status(
    proc_id: int, session: SessionDependency
) -> ProcessingStatus:
    """
    Get a depth 1 map processing status by id

    Parameters
    ----------
    proc_id : int
        ID of processing status
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : ProcessingStatus
        ProcessingStatus corresponding to proc_id

    Raises
    ------
    HTTPException
        If proc_id does not correspond to any depth 1 map processing status
    """
    try:
        response = core.get_proccessing_status(proc_id=proc_id, session=session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.post("/procstat/{proc_id}")
def update_processing_status(
    proc_id: int,
    model: ProcessingStatusModificationRequest,
    session: SessionDependency,
) -> ProcessingStatus:
    """
    Update depth 1 map processing status by id

    Parameters
    ----------
    proc_id : int
        ID of processing status
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
        If ID does not correspond to any processing status
    """

    try:
        response = core.update_processing_status(
            proc_id=proc_id,
            map_name=model.map_name,
            processing_start=model.processing_start,
            processing_end=model.processing_end,
            processing_status=model.processing_status,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.delete("/procstat/{proc_id}")
def delete_processing_status(proc_id: int, session: SessionDependency) -> None:
    """
    Delete a processing by ID

    Parameters
    ----------
    proc_id : int
        ID of processing status
    session : SessionDependency
        Session to use

    Returns
    -------
    None

    Raises
    ------
    HTTPException
        If proc_id does not correspond to any processing status
    """
    try:
        core.delete_processing_status(proc_id=proc_id, session=session)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return


@router.put("/pointresid/new")  # TODO : path?
def create_pointing_residual(
    model: PointingResidualModificationRequest,
    session: SessionDependency,
) -> PointingResidual:
    """
    Create a new pointing residual for a depth one map in the catalog.

    Parameters
    ----------
    model : PointingResidualModificationRequest
        Class containing modifyable attributes of the pointing residual
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : PointingResidual
        PointingResidual which was specified by model and added to the database

    Raise
    -----
    HTTPException
        If the model does not contain required info or api response is malformed
    """
    try:
        response = core.create_pointing_residual(
            map_name=model.map_name,
            ra_offset=model.ra_offset,
            dec_offset=model.dec_offset,
            session=session,
        )
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())

    return response


@router.get("/pointresid/{point_id}")
def get_pointing_residual(
    point_id: int, session: SessionDependency
) -> PointingResidual:
    """
    Get a pointing residual by id

    Parameters
    ----------
    point_id : int
        ID of pointing residual
    session : SessionDependancy
        Session to use

    Returns
    -------
    response : PointingResidual
        PointingResidual corresponding to point_id

    Raises
    ------
    HTTPException
        If point_id does not correspond to any depth 1 map pointing residuals
    """
    try:
        response = core.get_pointing_residual(point_id=point_id, session=session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.post("/pointresid/{point_id}")
def update_pointing_residual(
    point_id: int,
    model: PointingResidualModificationRequest,
    session: SessionDependency,
) -> PointingResidual:
    """
    Update pointing residual by ID.

    Parameters
    ----------
    point_id : int
        ID of pointing residual
    model : PointingResidualModificationRequest
        Parameters of pointing residual to modify
    session : SessionDependency
        Session to use

    Returns
    -------
    response : PointingResidual
        Modified PointingResidual

    Raises
    ------
    HTTPException
        If ID does not correspond to any pointing residual
    """

    try:
        response = core.update_pointing_residual(
            point_id=point_id,
            map_name=model.map_name,
            ra_offset=model.ra_offset,
            dec_offset=model.dec_offset,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return response


@router.delete("/pointresid/{point_id}")
def delete_pointing_residual(point_id: int, session: SessionDependency) -> None:
    """
    Delete a pointing residual by ID

    Parameters
    ----------
    point_id : int
        ID of pointing residual
    session : SessionDependency
        Session to use

    Returns
    -------
    None

    Raises
    ------
    HTTPException
        If point_id does not correspond to any pointing residual
    """
    try:
        core.delete_pointing_residual(point_id=point_id, session=session)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return


app.include_router(router)
