# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated detector response to point-like sources.

This includes artificial calibration sources (e.g. drone), solar
system objects, and catalogs of point sources.

"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_simulate_source_signal(operators):
    """Add commandline args and operators for artifical sources.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimSource(name="sim_source", enabled=False))


@workflow_timer
def simulate_source_signal(job, otherargs, runargs, data):
    """Simulate detector response from artificial sources.

    This uses Az/El detector pointing to observe a terrestrial calibration
    source (e.g. a drone).

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

    if job_ops.sim_source.enabled:
        job_ops.sim_source.detector_pointing = job_ops.det_pointing_azel
        # FIXME: this is not right- the field of view calculation in the
        # SimSource operator should loop over all observations and get the
        # maximum extent.
        # job_ops.sim_source.focalplane = telescope.focalplane
        raise NotImplementedError("SimSource disabled until FoV fixed.")
        if job_ops.sim_source.polarization_fraction != 0:
            job_ops.sim_source.detector_weights = job_ops.weights_azel
        job_ops.sim_source.apply(data)


def setup_simulate_sso_signal(operators):
    """Add commandline args and operators for solar system objects.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimSSO(name="sim_sso", enabled=False))


@workflow_timer
def simulate_sso_signal(job, otherargs, runargs, data):
    """Simulate detector response from solar system objects.

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

    if job_ops.sim_sso.enabled:
        job_ops.sim_sso.detector_pointing = job_ops.det_pointing_azel
        job_ops.sim_sso.detector_weights = job_ops.weights_azel
        job_ops.sim_sso.apply(data)


def setup_simulate_catalog_signal(operators):
    """Add commandline args and operators for point source catalogs.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimCatalog(name="sim_catalog", enabled=False))


@workflow_timer
def simulate_catalog_signal(job, otherargs, runargs, data):
    """Simulate detector response from a point source catalog.

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

    if job_ops.sim_catalog.enabled:
        job_ops.sim_catalog.detector_pointing = job_ops.det_pointing_radec
        job_ops.sim_catalog.apply(data)
