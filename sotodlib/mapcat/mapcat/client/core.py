"""
Abstract client class that must be implemented by both the mock and the 'real'
Note none of the returns should ever be actually returned so no-covered
client.
"""

from abc import ABC, abstractmethod
from typing import Any

from mapcat.database import DepthOneMap, ProcessingStatus, PointingResidual


class DepthOneBase(ABC):
    """
    Abstract methods for the DepthOneMap.
    Note none of these do anything, they
    just define the func signatures.
    """

    @abstractmethod
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
        Create a new depth one map
        """

        return  # pragma: no cover

    @abstractmethod
    def get_depth_one(self, *, map_id: int) -> DepthOneMap | None:
        """
        Get depth one map. If not found, return None
        """

        return None  # pragma: no cover

    @abstractmethod
    def update_depth_one(
        self,
        *,
        map_id: int,
        map_name: str,
        map_path: str,
        tube_slot: str,
        wafers: str,
        frequency: str,
        ctime: float,
    ) -> DepthOneMap:
        """
        Update new depth one map
        """

        return  # pragma: no cover

    @abstractmethod
    def delete_depth_one(self, *, map_id: int) -> None:
        """
        Delete a depth one map
        """

        return None  # pragma: no cover


class ProcessingStatusBase(ABC):
    """
    Abstract methods for the ProcessingStatus.
    Note none of these do anything, they
    just define the func signatures.
    """

    @abstractmethod
    def create_processing_status(
        self,
        *,
        map_name: str,
        processing_start: float,
        processing_end: float,
        processing_status: str,
    ) -> ProcessingStatus:
        """
        Create a new processing status
        """

        return None  # pragma: no cover

    @abstractmethod
    def get_processing_status(self, *, map_id: int) -> ProcessingStatus | None:
        """
        Get processing status. If not found, return None
        """

        return None  # pragma: no cover

    @abstractmethod
    def create_processing_status(
        self,
        *,
        proc_id: int,
        map_name: str,
        processing_start: float,
        processing_end: float,
        processing_status: str,
    ) -> ProcessingStatus:
        """
        Update a processing status
        """

        return None  # pragma: no cover

    @abstractmethod
    def delete_processing_status(self, *, proc_id: int) -> None:
        """
        Delete a processing status
        """

        return None  # pragma: no cover


class PointingResidualBase(ABC):
    """
    Abstract methods for PointingResidual.
    Note none of these do anything, they
    just define the func signatures.
    """

    @abstractmethod
    def create_pointing_residual(
        self,
        *,
        map_name: str,
        ra_offset: float,
        dec_offset: float,
    ) -> PointingResidual:
        """
        Create a new pointing residual
        """

        return None  # pragma: no cover

    @abstractmethod
    def get_pointing_residual(self, *, resid_id: int) -> PointingResidual | None:
        """
        Get pointing residual. If not found, return None
        """

        return None  # pragma: no cover

    @abstractmethod
    def create_pointing_residual(
        self,
        *,
        resid_id: int,
        map_name: str,
        ra_offset: float,
        dec_offset: float,
    ) -> PointingResidual:
        """
        Update a pointing residual
        """

        return None  # pragma: no cover

    @abstractmethod
    def delete_pointing_residual(self, *, resid_id: int) -> None:
        """
        Delete a processing status
        """

        return None  # pragma: no cover
