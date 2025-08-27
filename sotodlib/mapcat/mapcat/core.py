"""
Core functionality providing access to the database.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from sotodlib.mapcat.mapcat.database import (
    DepthOneMap,
    DepthOneMapTable,
    ProcessingStatus,
    ProcessingStatusTable,
    PointingResidual,
    PointingResidualTable,
    TODDepthOne,
    TODDepthOneTable,
)

# TODO: Should update functions have none defaults? This would also entail reodering all funcs to put session immediately after id


def create_depth_one(
    map_name: str,
    map_path: str,
    tube_slot: str,
    wafers: str,
    frequency: str,
    ctime: float,
    session: Session,
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
    session : Session
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

    with session.begin():
        session.add(dmap)
        session.commit()

    return dmap.to_model()


def get_depth_one(map_id: int, session: Session) -> DepthOneMap:
    """
    Get a depth 1 map by id.

    Parameters
    ----------
    map_id : int
        Interal map ID number
    session : Session
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
    dmap = session.get(DepthOneMapTable, map_id)

    if dmap is None:
        raise ValueError(f"Depth-1 map with ID {map_id} not found.")

    return dmap.to_model()


def update_depth_one(
    map_id: int,
    map_name: str | None,
    map_path: str | None,
    tube_slot: str | None,
    wafers: str | None,
    frequency: str | None,
    ctime: str | None,
    session: Session,
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
    session : Session
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
    with session.begin():
        dmap = session.get(DepthOneMapTable, map_id)

        if dmap is None:
            raise ValueError(f"Depth 1 map with ID {map_id} not found.")
        dmap.map_name = map_name if map_name is not None else dmap.map_name
        dmap.map_path = map_path if map_path is not None else dmap.map_path
        dmap.tube_slot = tube_slot if tube_slot is not None else dmap.tube_slot
        dmap.wafers = wafers if wafers is not None else dmap.wafers
        dmap.frequency = frequency if frequency is not None else dmap.frequency
        dmap.ctime = ctime if ctime is not None else dmap.ctime

        session.commit()

    return dmap.to_model()


def delete_depth_one(map_id: int, session: Session) -> None:
    """
    Delete a depth one map from the database

    Parameters
    ----------
    map_id : int
        ID of depth one map
    session : Session
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the source is not found
    """
    with session.begin():
        dmap = session.get(DepthOneMapTable, map_id)

        if dmap is None:
            raise ValueError(f"Depth 1 map with ID {map_id} not found.")

        session.delete(dmap)
        session.commit()

    return


def create_processing_status(
    map_name: str,
    processing_start: float | None,
    processing_end: float | None,
    processing_status: str,
    session: Session,
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
    session : Session
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

    with session.begin():
        session.add(proc_stat)
        session.commit()

    return proc_stat.to_model()


def get_processing_status(proc_id: int, session: Session) -> ProcessingStatus:
    """
    Get the processing status of a depth 1 map by (processing status) id.

    Parameters
    ----------
    proc_id : int
        ID of processing status
    session : Session
        Session to use

    Returns
    -------
    proc_stat.to_service() : ProcessingStatus
        Requested entry for depth 1 map

    Raises
    ------
    ValueError
        If processing status is not found
    """
    proc_stat = session.get(ProcessingStatusTable, proc_id)

    if proc_stat is None:
        raise ValueError(f"Depth-1 map with ID {proc_id} not found.")

    return proc_stat.to_model()


def update_processing_status(
    proc_id: int,
    map_name: str | None,
    processing_start: float | None,
    processing_end: float | None,
    processing_status: str | None,
    session: Session,
) -> ProcessingStatus:
    """
    Update a depth one map processing status.

    Parameters
    ----------
    proc_id : int
        Internal ID of the processing status
    map_name : str
        Name of depth 1 map to track
    processing_start : float | None
        Time of processing start
    processing_end : float | None
        Time of processing end
    processing_status : str
        Status of processing
    session : Session
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
    with session.begin():
        proc_stat = session.get(ProcessingStatusTable, proc_id)

        if proc_stat is None:
            raise ValueError(f"Depth 1 map with ID {proc_id} not found.")
        proc_stat.map_name = map_name if map_name is not None else proc_stat.map_name
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

        session.commit()

    return proc_stat.to_model()


def delete_processing_status(proc_id: int, session: Session) -> None:
    """
    Delete a processing status of a map from database

    Parameters
    ----------
    proc_id : int
        Internal ID of the processing status
    session : Session
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the processing status is not found
    """
    with session.begin():
        proc_stat = session.get(ProcessingStatusTable, proc_id)

        if proc_stat is None:
            raise ValueError(f"Depth 1 map with name {proc_id} not found.")

        session.delete(proc_stat)
        session.commit()

    return


def create_pointing_residual(
    map_name: str,
    ra_offset: float | None,
    dec_offset: float | None,
    session: Session,
) -> PointingResidual:
    """
    Create a entry tracking depth one map processing status.

    Parameters
    ----------
    map_name : str
        Name of depth 1 map to track
    ra_offset : float
        Calculated ra offset of PSes
    dec_offset : float
        Calculated dec offset of PSes
    session : Session
        Session to use

    Returns
    -------
    point_resid.to_model() : PointingResidual
        PointingResidual instance initialized with parameters above
    """
    point_resid = PointingResidualTable(
        map_name=map_name,
        ra_offset=ra_offset,
        dec_offset=dec_offset,
    )
    with session.begin():
        session.add(point_resid)
        session.commit()

    return point_resid.to_model()


def get_pointing_residual(point_id: int, session: Session) -> PointingResidual:
    """
    Get a depth 1 map pointing residual by (pointing residual) id.

    Parameters
    ----------
    point_id : int
        ID of pointing residual
    session : Session
        Session to use

    Returns
    -------
    point_resid.to_service() : PointingResidual
        Requested pointing residual for depth 1 map

    Raises
    ------
    ValueError
        If pointing residual is not found
    """
    point_resid = session.get(PointingResidualTable, point_id)

    if point_resid is None:
        raise ValueError(f"Depth-1 map with ID {point_id} not found.")

    return point_resid.to_model()


def update_pointing_residual(
    point_id: int,
    map_name: str | None,
    ra_offset: float | None,
    dec_offset: float | None,
    session: Session,
) -> PointingResidual:
    """
    Update a depth one map pointing residual.

    Parameters
    ----------
    point_ID : int
        Internal ID of the pointing residual
    map_name : str
        Name of depth 1 map to track
    ra_offset : float
        Calculated ra offset of PSes
    dec_offset : float
        Calculated dec offset of PSes
    session : Session
        Session to use

    Returns
    -------
    point_resid.to_model() : PointingResidual
        PointingResidual instance updated with parameters above

    Raises
    ------
    ValueError
        If the pointing residual is not found
    """
    with session.begin():
        point_resid = session.get(PointingResidualTable, point_id)

        if point_resid is None:
            raise ValueError(f"Depth 1 map with ID {point_id} not found.")
        point_resid.map_name = (
            map_name if map_name is not None else point_resid.map_name
        )
        point_resid.ra_offset = (
            ra_offset if ra_offset is not None else point_resid.ra_offset
        )
        point_resid.dec_offset = (
            dec_offset if dec_offset is not None else point_resid.dec_offset
        )

        session.commit()

    return point_resid.to_model()


def delete_pointing_residual(point_id: int, session: Session) -> None:
    """
    Delete a pointing residual of a map from database

    Parameters
    ----------
    proc_id : int
        Internal ID of the pointing residual
    session : Session
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the pointing residual is not found
    """
    with session.begin():
        point_resid = session.get(PointingResidualTable, point_id)

        if point_resid is None:
            raise ValueError(f"Depth 1 map with name {point_id} not found.")

        session.delete(point_resid)
        session.commit()

    return


def create_tod(
    map_name: str,
    obs_id: float,
    pwv: float | None,
    ctime: float,
    start_time: float,
    stop_time: float,
    nsamples: int | None,
    telescope: str,
    telescope_flavor: str | None,
    tube_slot: str | None,
    tube_flavor: str | None,
    frequency: str,
    scan_type: str | None,
    subtype: str | None,
    wafer_count: int,
    duration: float | None,
    az_center: float | None,
    az_throw: float | None,
    el_center: float | None,
    el_throw: float | None,
    roll_center: float | None,
    roll_throw: float | None,
    wafer_slots_list: str,
    stream_ids_list: str,
    session: Session,
) -> TODDepthOne:
    """
    Create a entry tracking depth one map processing status.

    Parameters
    ----------
    map_name : str
        Name of map this TOD went into. Foreign key
    obs_id : str
        SO ID of TOD
    pwv : float | None
        Precipitable  water vapor at time of obs
    ctime : float
        Mean unix time of obs
    start_time : float
        Start time of obs
    stop_time : float
        End time of obs
    nsamples : int | None
        Number of samps in obs
    telescope : str
        Telescope making obs
    telescope_flavor : str | None
        Telescope LF/MF/UHF. Only for SATs
    tube_slot : str | None
        Tube of obs. Only for LAT
    tube_flavor : str | None
        LF/MF/UHF of tube. Only for LAT
    frequency : str
        Frequency of obs
    scan_type : str | None
        Type of scan.
    subtype : str | None
        Subtype of scan
    wafer_count : int
        Number of working wafers for scan
    duration : float | None
        Duration of scan in seconds
    az_center : float | None
        Az center of scan
    az_throw : float | None
        Az throw of scan
    el_center : float | None
        El center of scan
    el_throw : float | None
        El throw of scan
    roll_center : float | None
        Roll center of scan
    roll_throw : float | None
        Roll throw of scan
    wafer_slots_list : str
        List of live wafers for scan
    stream_ids_list : str
        Stream IDs live for scan
    session : Session
        Session to use

    Returns
    -------
    tod.to_model() : TODDepthOne
        TODDepthOne instance initialized with parameters above
    """
    tod = TODDepthOneTable(
        map_name=map_name,
        obs_id=obs_id,
        pwv=pwv,
        ctime=ctime,
        start_time=start_time,
        stop_time=stop_time,
        nsamples=nsamples,
        telescope=telescope,
        telescope_flavor=telescope_flavor,
        tube_slot=tube_slot,
        tube_flavor=tube_flavor,
        frequency=frequency,
        scan_type=scan_type,
        subtype=subtype,
        wafer_count=wafer_count,
        duration=duration,
        az_center=az_center,
        az_throw=az_throw,
        el_center=el_center,
        el_throw=el_throw,
        roll_center=roll_center,
        roll_throw=roll_throw,
        wafer_slots_list=wafer_slots_list,
        stream_ids_list=stream_ids_list,
    )
    with session.begin():
        session.add(tod)
        session.commit()

    return tod.to_model()


def get_tod(tod_id: int, session: Session) -> TODDepthOne:
    """
    Get a tod by id

    Parameters
    ----------
    tod_id : int
        ID of TOD
    session : Session
        Session to use

    Returns
    -------
    tod.to_service() : TODDepthOne
        Requested tod

    Raises
    ------
    ValueError
        If tod is not found
    """
    tod = session.get(TODDepthOneTable, tod_id)

    if tod is None:
        raise ValueError(f"Depth-1 map with ID {tod_id} not found.")

    return tod.to_model()


def update_tod(
    tod_id: int,
    map_name: str | None,
    obs_id: float | None,
    pwv: float | None,
    ctime: float | None,
    start_time: float | None,
    stop_time: float | None,
    nsamples: int | None,
    telescope: str | None,
    telescope_flavor: str | None,
    tube_slot: str | None,
    tube_flavor: str | None,
    frequency: str | None,
    scan_type: str | None,
    subtype: str | None,
    wafer_count: int | None,
    duration: float | None,
    az_center: float | None,
    az_throw: float | None,
    el_center: float | None,
    el_throw: float | None,
    roll_center: float | None,
    roll_throw: float | None,
    wafer_slots_list: str | None,
    stream_ids_list: str | None,
    session: Session,
) -> TODDepthOne:
    """
    Update a TOD

    Parameters
    ----------
    tod_id : int
        Internal ID of the TOD
    obs_id : float
        SO ID for TOD
    pwv : float | None
        Precipitable  water vapor at time of obs
    ctime : float
        Mean unix time of obs
    start_time : float
        Start time of obs
    stop_time : float
        End time of obs
    nsamples : int | None
        Number of samps in obs
    telescope : str
        Telescope making obs
    telescope_flavor : str | None
        Telescope LF/MF/UHF. Only for SATs
    tube_slot : str | None
        Tube of obs. Only for LAT
    tube_flavor : str | None
        LF/MF/UHF of tube. Only for LAT
    frequency : str
        Frequency of obs
    scan_type : str | None
        Type of scan.
    subtype : str | None
        Subtype of scan
    wafer_count : int
        Number of working wafers for scan
    duration : float | None
        Duration of scan in seconds
    az_center : float | None
        Az center of scan
    az_throw : float | None
        Az throw of scan
    el_center : float | None
        El center of scan
    el_throw : float | None
        El throw of scan
    roll_center : float | None
        Roll center of scan
    roll_throw : float | None
        Roll throw of scan
    wafer_slots_list : str
        List of live wafers for scan
    stream_ids_list : str
        Stream IDs live for scan
    session : Session
        Session to use

    Returns
    -------
    tod.to_model() : TODDepthOne
        TODDepthOne instance updated with parameters above

    Raises
    ------
    ValueError
        If the TOD is not found
    """
    with session.begin():
        tod = session.get(TODDepthOneTable, tod_id)

        if tod is None:
            raise ValueError(f"Depth 1 map with ID {tod_id} not found.")
        tod.map_name = map_name if map_name is not None else tod.map_name
        tod.obs_id = obs_id if obs_id is not None else tod.obs_id
        tod.pwv = pwv if pwv is not None else tod.pwv
        tod.ctime = ctime if ctime is not None else tod.dec_offset
        tod.start_time = start_time if start_time is not None else tod.start_time
        tod.stop_time = stop_time if stop_time is not None else tod.stop_time
        tod.nsamples = nsamples if nsamples is not None else tod.nsamples
        tod.telescope = telescope if telescope is not None else tod.telescope
        tod.telescope_flavor = (
            telescope_flavor if telescope_flavor is not None else tod.telescope_flavor
        )
        tod.tube_slot = tube_slot if tube_slot is not None else tod.tube_slot
        tod.tube_flavor = tube_flavor if tube_flavor is not None else tod.tube_flavor
        tod.frequency = frequency if frequency is not None else tod.frequency
        tod.scan_type = scan_type if scan_type is not None else tod.scan_type
        tod.subtype = subtype if subtype is not None else tod.subtype
        tod.wafer_count = wafer_count if wafer_count is not None else tod.wafer_count
        tod.duration = duration if duration is not None else tod.duration
        tod.az_center = az_center if az_center is not None else tod.az_center
        tod.az_throw = az_throw if az_throw is not None else tod.az_throw
        tod.el_center = el_center if el_center is not None else tod.el_center
        tod.el_throw = el_throw if el_throw is not None else tod.el_throw
        tod.roll_center = roll_center if roll_center is not None else tod.roll_center
        tod.roll_throw = roll_throw if roll_throw is not None else tod.roll_throw
        tod.wafer_slots_list = (
            wafer_slots_list if wafer_slots_list is not None else tod.wafer_slots_list
        )
        tod.stream_ids_list = (
            stream_ids_list if stream_ids_list is not None else tod.stream_ids_list
        )
        session.commit()

    return tod.to_model()


def delete_tod(tod_id: int, session: Session) -> None:
    """
    Delete a tod from database

    Parameters
    ----------
    tod_id : int
        Internal ID of the tod
    session : Session
        Session to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the tod is not found
    """
    with session.begin():
        tod = session.get(TODDepthOneTable, tod_id)

        if tod is None:
            raise ValueError(f"Depth 1 map with name {tod_id} not found.")

        session.delete(tod)
        session.commit()

    return
