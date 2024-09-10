# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated detector response to the atmosphere.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from .job import workflow_timer


def setup_weather_model(operators):
    """Add commandline args and operators for appending weather models.
    Args:
        operators (list):  The list of operators to extend.
    Returns:
        None
    """
    operators.append(
        toast.ops.WeatherModel(
            name="weather_model",
            weather="atacama",
            enabled=False,
        )
    )


@workflow_timer
def append_weather_model(job, otherargs, runargs, data):
    """Append a weather model to the data.
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
    wm = job.operators.weather_model
    if wm.enabled:
        if otherargs.realization is not None:
            wm.realization = otherargs.realization
        log.info_rank(f"  Running {wm.name}...", comm=data.comm.comm_world)
        wm.apply(data)
        log.info_rank(
            f"  Applied {wm.name} in", comm=data.comm.comm_world, timer=timer
        )


def setup_simulate_atmosphere_signal(operators):
    """Add commandline args and operators for simulating atmosphere.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(
        toast.ops.SimAtmosphere(
            name="sim_atmosphere_coarse",
            add_loading=False,
            lmin_center=300 * u.m,
            lmin_sigma=30 * u.m,
            lmax_center=10000 * u.m,
            lmax_sigma=1000 * u.m,
            xstep=50 * u.m,
            ystep=50 * u.m,
            zstep=50 * u.m,
            zmax=2000 * u.m,
            nelem_sim_max=30000,
            gain=6e-4,
            realization=1000000,
            wind_dist=10000 * u.m,
            enabled=False,
        )
    )
    operators.append(
        toast.ops.SimAtmosphere(
            name="sim_atmosphere",
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=4 * u.m,
            ystep=4 * u.m,
            zstep=4 * u.m,
            zmax=200 * u.m,
            gain=4e-5,
            wind_dist=1000 * u.m,
            enabled=False,
        )
    )


@workflow_timer
def simulate_atmosphere_signal(job, otherargs, runargs, data):
    """Simulate atmosphere signal.

    This creates or loads a realization of the atmosphere for each
    observing session and uses detector pointing to integrate the power
    seen by each detector at each sample.

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

    # Check if we need an extra copy of the atmospheric signal
    final_signal = job_ops.sim_atmosphere.det_data
    if (
            hasattr(job_ops, "sim_hwpss")
            and job_ops.sim_hwpss.enabled
            and job_ops.sim_hwpss.atmo_signal is not None
        ):
        temp_signal = job_ops.sim_hwpss.atmo_signal
    else:
        temp_signal = None

    if otherargs.realization is not None:
        job_ops.sim_atmosphere_coarse.realization = 1000000 + otherargs.realization
        job_ops.sim_atmosphere.realization = otherargs.realization

    for sim_atm in job_ops.sim_atmosphere_coarse, job_ops.sim_atmosphere:
        if not sim_atm.enabled:
            continue
        sim_atm.detector_pointing = job_ops.det_pointing_azel_sim
        if sim_atm.polarization_fraction != 0:
            sim_atm.detector_weights = job_ops.weights_azel
        if temp_signal is not None:
            # Write the simulated atmosphere to a temporary array
            sim_atm.det_data = temp_signal
        log.info_rank(f"  Running {sim_atm.name}...", comm=data.comm.comm_world)
        sim_atm.apply(data)
        log.info_rank(
            f"  Applied {sim_atm.name} in", comm=data.comm.comm_world, timer=timer
        )
        if temp_signal is not None:
            # Restore original configuration
            sim_atm.det_data = final_signal

    if temp_signal is not None:
        # Add the atmospheric signal to the final target but also keep the
        # separate copy
        combine = toast.ops.Combine(
            op="add",
            first=final_signal,
            second=temp_signal,
            result=final_signal,
        ).apply(data)
        log.info_rank(
            f"  Added {temp_signal} to {final_signal} in",
            comm=data.comm.comm_world,
            timer=timer,
        )
