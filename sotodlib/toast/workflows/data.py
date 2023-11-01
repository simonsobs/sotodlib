# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Data I/O operations.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_load_data_context(operators):
    """Add commandline args and operators for loading from a context.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    raise NotImplementedError("LoadContext operator not yet merged.")
    # operators.append(
    #     so_ops.LoadContext(
    #         name="load_context",
    #     )
    # )
    return


@workflow_timer
def load_data_context(job, otherargs, runargs, data):
    """Load data from a Context.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    raise NotImplementedError("LoadContext operator not yet merged.")
    # Configured operators for this job
    job_ops = job.operators

    # Load it
    if job_ops.load_context.enabled:
        job_ops.load_context.apply(data)


def setup_load_data_hdf5(operators):
    """Add commandline args and operators for loading from HDF5 files.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.LoadHDF5(name="load_hdf5", enabled=False))


@workflow_timer
def load_data_hdf5(job, otherargs, runargs, data):
    """Load data from one or more HDF5 files.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    # Configured operators for this job
    job_ops = job.operators

    # Load it
    if job_ops.load_hdf5.enabled:
        job_ops.load_hdf5.apply(data)


def setup_load_data_books(operators):
    """Add commandline args and operators for directly loading L3 books.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.LoadBooks(name="load_books", enabled=False))


@workflow_timer
def load_data_books(job, otherargs, runargs, data):
    """Load data from one or more books.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    # Configured operators for this job
    job_ops = job.operators

    # Load it
    if job_ops.load_books.enabled:
        job_ops.load_books.apply(data)


def setup_save_data_hdf5(operators):
    """Add commandline args and operators for saving to HDF5 files.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.SaveHDF5(name="save_hdf5", enabled=False))


@workflow_timer
def save_data_hdf5(job, otherargs, runargs, data):
    """Save data to HDF5 files.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    # Configured operators for this job
    job_ops = job.operators

    # Load it
    if job_ops.save_hdf5.enabled:
        job_ops.save_hdf5.apply(data)


def setup_save_data_books(operators):
    """Add commandline args and operators for saving to L3 books.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SaveBooks(name="save_books", enabled=False))


@workflow_timer
def save_data_books(job, otherargs, runargs, data):
    """Save data to books.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    # Configured operators for this job
    job_ops = job.operators

    # Load it
    if job_ops.save_books.enabled:
        job_ops.save_books.apply(data)
