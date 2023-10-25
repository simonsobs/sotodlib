# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simulated observing / scanning motion of the telescope.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops

from .. import ops as so_ops
from ..instrument import simulated_telescope


def setup_simulate_observing(parser, operators):
    """Add commandline args and operators for simulated observing.

    Args:
        parser (ArgumentParser):  The parser to update.
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    parser.add_argument(
        "--hardware", required=False, default=None, help="Input hardware file"
    )
    parser.add_argument(
        "--det_info_file",
        required=False,
        default=None,
        help="Input detector info file for real hardware maps",
    )
    parser.add_argument(
        "--det_info_version",
        required=False,
        default=None,
        help="Detector info file version such as 'Cv4'",
    )
    parser.add_argument(
        "--thinfp",
        required=False,
        type=int,
        help="Thin the focalplane by this factor",
    )
    parser.add_argument(
        "--bands",
        required=True,
        help="Comma-separated list of bands: LAT_f030 (27GHz), LAT_f040 (39GHz), "
        "LAT_f090 (93GHz), LAT_f150 (145GHz), "
        "LAT_f230 (225GHz), LAT_f290 (285GHz), "
        "SAT_f030 (27GHz), SAT_f040 (39GHz), "
        "SAT_f090 (93GHz), SAT_f150 (145GHz), "
        "SAT_f230 (225GHz), SAT_f290 (285GHz). ",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--telescope",
        default=None,
        help="Telescope to simulate: LAT, SAT1, SAT2, SAT3, SAT4.",
    )
    group.add_argument(
        "--tube_slots",
        default=None,
        help="Comma-separated list of optics tube slots: c1 (LAT_UHF), i5 (LAT_UHF), "
        " i6 (LAT_MF), i1 (LAT_MF), i3 (LAT_MF), i4 (LAT_MF), o6 (LAT_LF),"
        " ST1 (SAT_MF), ST2 (SAT_MF), ST3 (SAT_UHF), ST4 (SAT_LF).",
    )
    group.add_argument(
        "--wafer_slots", default=None, help="Comma-separated list of wafer slots. "
    )
    parser.add_argument(
        "--sample_rate",
        required=False,
        default=10,
        help="Sampling rate",
        type=float,
    )
    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    operators.append(
        toast.ops.SimGround(
            name="sim_ground",
            weather="atacama",
            detset_key="pixel",
            session_split_key="wafer_slot",
        )
    )
    operators.append(so_ops.CoRotator(name="corotate_lat"))
    operators.append(toast.ops.PerturbHWP(name="perturb_hwp", enabled=False))


def simulate_observing(job, otherargs, runargs, comm):
    """Simulate the observing motion of the selected detectors with the schedule.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        comm (MPI.Comm):  The MPI world communicator (or None).

    Returns:
        None

    """
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    # Configured operators for this job
    job_ops = job.operators

    # Simulated telescope
    telescope = simulated_telescope(
        hwfile=otherargs.hardware,
        det_info_file=otherargs.det_info_file,
        det_info_version=otherargs.det_info_version,
        telescope_name=otherargs.telescope,
        sample_rate=otherargs.sample_rate * u.Hz,
        bands=otherargs.bands,
        wafer_slots=otherargs.wafer_slots,
        tube_slots=otherargs.tube_slots,
        thinfp=otherargs.thinfp,
        comm=comm,
    )

    # Load the schedule file
    schedule = toast.schedule.GroundSchedule()
    schedule.read(otherargs.schedule, comm=comm)
    log.info_rank("Loaded schedule in", comm=comm, timer=timer)
    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After loading schedule:  {mem}", comm)

    # Get the process group size
    group_size = toast.job_group_size(
        comm,
        runargs,
        schedule=schedule,
        focalplane=telescope.focalplane,
        full_pointing=otherargs.full_pointing,
    )

    # Create the toast communicator.
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # The data container
    data = toast.Data(comm=toast_comm)

    timer.clear()
    timer.start()

    job_ops.mem_count.prefix = "Before Simulation"
    job_ops.mem_count.apply(data)

    # Simulate the telescope pointing

    job_ops.sim_ground.telescope = telescope
    job_ops.sim_ground.schedule = schedule
    if job_ops.sim_ground.weather is None:
        job_ops.sim_ground.weather = telescope.site.name
    if otherargs.realization is not None:
        job_ops.sim_ground.realization = otherargs.realization
    log.info_rank("Running simulated observing...", comm=data.comm.comm_world)
    job_ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=comm, timer=timer)

    job_ops.mem_count.prefix = "After Scan Simulation"
    job_ops.mem_count.apply(data)

    # Apply LAT co-rotation
    if job_ops.corotate_lat.enabled:
        log.info_rank("Running simulated LAT corotation...", comm=data.comm.comm_world)
        job_ops.corotate_lat.apply(data)
        log.info_rank("Apply LAT co-rotation in", comm=comm, timer=timer)

    # Perturb HWP spin
    if job_ops.perturb_hwp.enabled:
        log.info_rank(
            "Running simulated HWP perturbation...", comm=data.comm.comm_world
        )
        job_ops.perturb_hwp.apply(data)
        log.info_rank("Perturbed HWP rotation in", comm=comm, timer=timer)
    return data
