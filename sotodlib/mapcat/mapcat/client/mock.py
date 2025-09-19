"""
Uses a local dictionary to implement the core.
"""

from importlib import import_module
from typing import Any

from mapcat.database import DepthOneMap, ProcessingStatus, PointingResidual

from .core import DepthOneBase, ProcessingStatusBase, PointingResidualBase


class DepthOneClient(DepthOneBase):
    """
    Class implementing methods which duplicated
    depth one map table functionality.

    Attributes
    ----------
    d1maptable : dict[int, DepthOneMap]
        Dictionary of DepthOneMap replicating a catalog
    n : int
        Number of entries in d1maptable

    Methods
    -------
    create_depth_one(self,*,map_name: str,map_path: str,tube_slot: str,wafers: str,frequency: str,ctime: float,)
        Create a depth one map and add it to the catalog
    get_depth_one(self, *, map_id: int)
        Get a depth one map by ID
    update_depth_one(self,*,map_id,map_name: str,map_path: str,tube_slot: str,wafers: str,frequency: str,ctime: float,)
        Update depth one map by ID
    delete_depth_one(self, *, map_id: int)
        Delete depth one map by ID
    """

    d1maptable: dict[int, DepthOneMap]
    n: int

    def __init__(self):
        super().__init__()
        """
        Initialize an empty table
        """
        self.d1maptable = {}
        self.n = 0

    def create_depth_one(
        self,
        *,
        map_name: str,
        map_path: str,
        tube_slot: str,
        wafers: str,
        frequency: str,
        ctime: float,
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
        d1map : DepthOneMap
            DepthOneMap instance initialized with parameters above
        """
        d1map = DepthOneMap(
            id=self.n,
            map_name=map_name,
            map_path=map_path,
            tube_slot=tube_slot,
            wafers=wafers,
            frequency=frequency,
            ctime=ctime,
        )
        self.d1maptable[self.n] = d1map
        self.n += 1

        return d1map

    def get_depth_one(self, *, map_id: int) -> DepthOneMap:
        """
        Get a depth 1 map by id.

        Parameters
        ----------
        map_id : int
            Interal map ID number

        Returns
        -------
        d1map : DepthOneMap
            Requested entry for depth 1 map

        Raises
        ------
        ValueError
            If map is not found
        """
        d1map = self.d1maptable.get(map_id, None)
        if d1map is None:
            raise ValueError(f"Depth-1 map with ID {map_id} not found.")
        return d1map

    def update_depth_one(
        self,
        *,
        map_id: int,
        map_name: str | None,
        map_path: str | None,
        tube_slot: str | None,
        wafers: str | None,
        frequency: str | None,
        ctime: str | None,
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
        new : DepthOneMap
            Updated depth 1 map

        Raises
        ------
        ValueError
            If the map is not found
        """
        current = self.get_depth_one(map_id=map_id)

        new = DepthOneMap(
            map_id=current.id,
            map_name=current.map_name if map_name is None else map_name,
            map_path=current.map_path if map_path is None else map_path,
            tube_slot=current.tube_slot if tube_slot is None else tube_slot,
            wafers=current.wafers if wafers is None else wafers,
            frequency=current.frequency if frequency is None else frequency,
            ctime=current.ctime if ctime is None else ctime,
        )

        self.d1maptable[map_id] = new

        return new

    def delete_depth_one(self, *, map_id: int) -> None:
        """
        Delete a depth one map from the database

        Parameters
        ----------
        map_id : int
            ID of depth one map

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the source is not found
        """
        d1map = self.d1maptable.pop(map_id, None)
        if d1map is None:
            raise ValueError(f"Depth 1 map with ID {map_id} not found.")

        self.n -= 1


class ProcessingStatusClient(ProcessingStatusBase):
    """
    Class implementing methods which duplicated
    processing statustable functionality.

    Attributes
    ----------
    procstattable : dict[int, ProcessingStatus]
        Dictionary of ProcessingStatus replicating a catalog
    n : int
        Number of entries in procstattable

    Methods
    -------
    create_processing_status(self,*,map_name: str,processing_start: float,processing_end: float,processing_status: str,)
        Create a processing status and add it to the catalog
    get_processing_status(self, *, map_id: int)
        Get a processing stats by ID
    update_processing_status(self,*,proc_id: int,map_name: str | None = None,processing_start: float | None = None,processing_end: float | None = None,processing_status: str | None = None,)
        Update processing status by ID
    delete_processing_status(self, *, proc_id: int)
        Delete processing status by ID
    """

    procstattable: dict[int, ProcessingStatus]
    n: int

    def __init__(self):
        super().__init__()
        """
        Initialize an empty table
        """
        self.procstattable = {}
        self.n = 0

    def create_processing_status(
        self,
        *,
        map_name: str,
        processing_start: float,
        processing_end: float,
        processing_status: str,
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

        Returns
        -------
        proc_stat : ProcessingStatus
            ProcessingStatus instance initialized with parameters above
        """
        proc_stat = ProcessingStatus(
            id=self.n,
            map_name=map_name,
            processing_start=processing_start,
            processing_end=processing_end,
            processing_status=processing_status,
        )
        self.procstattable[self.n] = proc_stat
        self.n += 1

        return proc_stat

    def get_processing_status(self, *, proc_id: int) -> ProcessingStatus:
        """
        Get a depth 1 map by id.

        Parameters
        ----------
        proc_id : int
            ID of processing status

        Returns
        -------
        proc_stat : ProcessingStatus
            Requested entry for depth 1 map

        Raises
        ------
        ValueError
            If map is not found
        """

        proc_stat = self.procstattable.get(proc_id, None)
        if proc_stat is None:
            raise ValueError(f"Depth-1 map with ID {proc_id} not found.")
        return proc_stat

    def update_processing_status(
        self,
        *,
        proc_id: int,
        map_name: str | None = None,
        processing_start: float | None = None,
        processing_end: float | None = None,
        processing_status: str | None = None,
    ) -> ProcessingStatus:
        """
        Update a depth one map.

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

        Returns
        -------
        new : ProcessingStatus
            ProcessingStatus instance updated with parameters above

        Raises
        ------
        ValueError
            If the processing status is not found
        """
        current = self.get_processing_status(proc_id=proc_id)

        new = ProcessingStatus(
            proc_id=current.id,
            map_name=current.map_name if map_name is None else map_name,
            processing_start=current.processing_start
            if processing_start is None
            else processing_start,
            processing_end=current.processing_end
            if processing_end is None
            else processing_end,
            processing_status=current.processing_status
            if processing_status is None
            else processing_status,
        )

        self.procstattable[proc_id] = new

        return new

    def delete_processing_status(self, *, proc_id: int) -> None:
        """
        Delete a processing status of a map from database

        Parameters
        ----------
        proc_id : int
            Internal ID of the processing status

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the processing status is not found
        """
        proc_stat = self.procstattable.pop(proc_id, None)
        if proc_stat is None:
            raise ValueError(f"Depth 1 map with ID {proc_id} not found.")

        self.n -= 1


class PointingResidualClient(PointingResidualBase):
    """
    Class implementing methods which duplicates
    pointing residual table functionality.

    Attributes
    ----------
    pointrestable : dict[int, PointingResidual]
        Dictionary of PointingResidual replicating a catalog
    n : int
        Number of entries in pointrestable

    Methods
    -------
    create_pointing_residual(self,*,map_name: str,ra_offset: float,dec_offset: float,)
        Create a pointing residual and add it to the catalog
    get_pointing_residual(self, *, resid_id: int)
        Get a pointing residual by ID
    update_pointing_residual(self,*,resid_id: int,map_name: str | None = None,ra_offset: float | None = None,dec_offset: float | None = None,)
        Update pointing residual by ID
    delete_pointing_residual(self, *, resid_id: int)
        Delete pointing residual by ID
    """

    pointrestable: dict[int, PointingResidual]
    n: int

    def __init__(self):
        super().__init__()
        """
        Initialize an empty table
        """
        self.pointrestable = {}
        self.n = 0

    def create_pointing_residual(
        self,
        *,
        map_name: str,
        ra_offset: float,
        dec_offset: float,
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

        Returns
        -------
        point_resid : PointingResidual
            PointingResidual instance initialized with parameters above
        """
        point_resid = ProcessingStatus(
            id=self.n,
            map_name=map_name,
            ra_offset=processing_start,
            dec_offset=dec_offset,
        )
        self.pointrestable[self.n] = point_resid
        self.n += 1

        return point_resid

    def get_pointing_residual(self, *, resid_id: int) -> PointingResidual:
        """
        Get a depth 1 map pointing residual by (pointing residual) id.

        Parameters
        ----------
        point_id : int
            ID of pointing residual

        Returns
        -------
        point_resid : PointingResidual
            Requested pointing residual for depth 1 map

        Raises
        ------
        ValueError
            If pointing residual is not found
        """
        point_resid = self.pointrestable.get(resid_id, None)
        if point_resid is None:
            raise ValueError(f"Depth-1 map with ID {resid_id} not found.")
        return point_resid

    def update_pointing_residual(
        self,
        *,
        resid_id: int,
        map_name: str | None = None,
        ra_offset: float | None = None,
        dec_offset: float | None = None,
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

        Returns
        -------
        new : PointingResidual
            PointingResidual instance updated with parameters above

        Raises
        ------
        ValueError
            If the pointing residual is not found
        """
        current = self.get_pointing_residual(resid_id=resid_id)

        new = PointingResidual(
            resid_id=current.id,
            map_name=current.map_name if map_name is None else map_name,
            ra_offset=current.ra_offset if ra_offset is None else ra_offset,
            dec_offset=current.dec_offset if dec_offset is None else dec_offset,
        )

        self.pointrestable[resid_id] = new

        return new

    def delete_pointing_residual(self, *, resid_id: int) -> None:
        """
        Delete a pointing residual of a map from database

        Parameters
        ----------
        proc_id : int
            Internal ID of the pointing residual

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the pointing residual is not found
        """
        point_resid = self.pointrestable.pop(resid_id, None)
        if point_resid is None:
            raise ValueError(f"Depth 1 map with ID {resid_id} not found.")

        self.n -= 1
