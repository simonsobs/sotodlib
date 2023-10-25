# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated optical pickup from various sources.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops


def setup_simulate_scan_synchronous_signal(operators):
    """Add commandline args and operators for scan-synchronous signal.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(toast.ops.SimScanSynchronousSignal(name="sim_sss", enabled=False))


def simulate_scan_synchronous_signal(job, otherargs, runargs, data):
    """Simulate scan-synchronous signal.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if otherargs.realization is not None:
        job_ops.sim_sss.realization = otherargs.realization

    if job_ops.sim_sss.enabled:
        job_ops.sim_sss.detector_pointing = job_ops.det_pointing_azel
        log.info_rank(
            "Running scan-synchronous signal simulation...", comm=data.comm.comm_world
        )
        job_ops.sim_sss.apply(data)
        log.info_rank(
            "Simulated Scan-synchronous signal", comm=data.comm.comm_world, timer=timer
        )
        job_ops.mem_count.prefix = "After simulating scan-synchronous signal"
        job_ops.mem_count.apply(data)


def setup_simulate_hwpss_signal(operators):
    """Add commandline args and operators for HWP synchronous signal.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.SimHWPSS(name="sim_hwpss", enabled=False))


def simulate_hwpss_signal(job, otherargs, runargs, data):
    """Simulate HWP synchronous signal.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.sim_hwpss.enabled and job_ops.sim_ground.hwp_angle is not None:
        job_ops.sim_hwpss.hwp_angle = job_ops.sim_ground.hwp_angle
        job_ops.sim_hwpss.detector_pointing = job_ops.det_pointing_azel
        job_ops.sim_hwpss.stokes_weights = job_ops.weights_azel
        log.info_rank("Running HWPSS simulation...", comm=data.comm.comm_world)
        job_ops.sim_hwpss.apply(data)
        log.info_rank(
            "Simulated HWP-synchronous signal",
            comm=data.comm.comm_world,
            timer=timer,
        )
