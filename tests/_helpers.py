# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Internal helper functions for unit tests.
"""

import os
import datetime
import shutil
import tempfile

import numpy as np
import astropy.units as u

try:
    import toast
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast.schedule_sim_ground
    from toast.observation import default_values as defaults
    toast_available = True
except ImportError as e:
    toast_available = False

from sotodlib.sim_hardware import sim_nominal


def mpi_world():
    """Return the MPI world communicator or None.
    """
    comm = None
    procs = 1
    rank = 0
    if "MPI_DISABLE" not in os.environ:
        try:
            import mpi4py.MPI as MPI

            comm = MPI.COMM_WORLD
            procs = comm.size
            rank = comm.rank
        except Exception:
            pass
    return comm, procs, rank


def mpi_multi():
    """Return True if we have more than one MPI process in our environment.
    """
    comm, procs, rank = mpi_world()
    if procs > 1:
        return True
    else:
        return False


def create_outdir(subdir=None, mpicomm=None):
    """Create the top level output directory and per-test subdir.

    Args:
        subdir (str): the sub directory for this test.
        mpicomm (MPI.Comm):  Optional communicator.

    Returns:
        str: full path to the test subdir if specified, else the top dir.

    """
    rank = 0
    if mpicomm is not None:
        rank = mpicomm.rank
    retdir = None
    if rank == 0:
        pwd = os.path.abspath(".")
        testdir = os.path.join(pwd, "sotodlib_test_output")
        retdir = testdir
        if subdir is not None:
            retdir = os.path.join(testdir, subdir)
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        if not os.path.isdir(retdir):
            os.mkdir(retdir)
    if mpicomm is not None:
        retdir = mpicomm.bcast(retdir, root=0)
    return retdir


def close_data_and_comm(data):
    """Delete a toast data object AND the input Comm.

    Multiple toast.Data objects can be created from a single toast.Comm,
    and so the communicators are not freed when deleting the data object.
    However for unit tests we frequently use a helper function to produce
    a simulated dataset and then want to fully clean up that along with
    the communicators that were used.  This is especially true on CI
    services where repeatedly creating communicators without cleanup seem
    to cause sporadic deadlocks.  This is a convenience function which
    does that cleanup.

    Args:
        data (toast.Data):  The input data object

    Returns:
        None

    """
    cm = data.comm
    if cm.comm_world is not None:
        cm.comm_world.barrier()
    data.clear()
    del data
    cm.close()
    del cm


def observing_schedule(telescope, mpicomm=None, temp_dir=None):
    if not toast_available:
        print("toast unavailable- cannot build observing schedule")
        return
    schedule = None
    if mpicomm is None or mpicomm.rank == 0:
        tdir = temp_dir
        if tdir is None:
            tdir = tempfile.mkdtemp()

        sch_file = os.path.join(tdir, "schedule.txt")
        toast.schedule_sim_ground.run_scheduler(
            opts=[
                "--site-name",
                telescope.site.name,
                "--telescope",
                telescope.name,
                "--site-lon",
                "{}".format(telescope.site.earthloc.lon.to_value(u.degree)),
                "--site-lat",
                "{}".format(telescope.site.earthloc.lat.to_value(u.degree)),
                "--site-alt",
                "{}".format(telescope.site.earthloc.height.to_value(u.meter)),
                "--patch",
                "small_patch,1,40,-40,44,-44",
                "--el-min",
                "40",
                "--el-max",
                "60",
                "--start",
                "2025-01-01 00:00:00",
                "--stop",
                "2025-01-01 03:00:00",
                "--out",
                sch_file,
            ]
        )
        schedule = toast.schedule.GroundSchedule()
        schedule.read(sch_file)
        if temp_dir is None:
            shutil.rmtree(tdir)
    if mpicomm is not None:
        schedule = mpicomm.bcast(schedule, root=0)
    return schedule


def calibration_schedule(telescope):
    """Make a small observing schedule for calibration tasks.

    This is a fake schedule with no scanning.

    """
    if not toast_available:
        print("toast unavailable- cannot build calibration schedule")
        return

    scans = [
        toast.schedule.GroundScan(
            name="CAL",
            start=datetime.datetime(
                2025, 1, 1, hour=0, minute=0, second=0
            ),
            stop=datetime.datetime(
                2025, 1, 1, hour=0, minute=10, second=0
            ),
            boresight_angle=0 * u.degree,
            az_min=-1.0 * u.degree,
            az_max=1.0 * u.degree,
            el=60.0 * u.degree,
            scan_indx=0,
            subscan_indx=0,
        )
    ]

    schedule = toast.schedule.GroundSchedule(
        scans=scans,
        site_name=telescope.site.name,
        telescope_name=telescope.name,
        site_lat=telescope.site.earthloc.lat,
        site_lon=telescope.site.earthloc.lon,
        site_alt=telescope.site.earthloc.height,
    )

    return schedule


def create_comm(mpicomm):
    """Create a toast communicator.

    Use the specified MPI communicator to attempt to create 2 process groups.
    If less than 2 processes are used, create a single process group.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).

    Returns:
        toast.Comm: the 2-level toast communicator.

    """
    if not toast_available:
        raise RuntimeError("TOAST is not importable, cannot create a toast.Comm")
    toastcomm = None
    if mpicomm is None:
        toastcomm = toast.Comm(world=mpicomm)
    else:
        worldsize = mpicomm.size
        groupsize = 1
        if worldsize >= 2:
            groupsize = worldsize // 2
        toastcomm = toast.Comm(world=mpicomm, groupsize=groupsize)
    return toastcomm


def simulation_test_data(
    mpicomm,
    telescope_name="SAT4",
    wafer_slot="w42",
    bands=["SAT_f030",],
    sample_rate=10.0 * u.Hz,
    detset_key="pixel",
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    thin_fp=4,
    cal_schedule=False,
):
    """Create a data object with a simple ground sim.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the ground sim to make some observations for each
    group.  This is useful for testing many operators that need some pre-existing
    observations with boresight pointing.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        sample_rate (Quantity): the sample rate.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    if not toast_available:
        raise RuntimeError("TOAST is not importable, cannot simulate test data")

    # Create the communicator
    toastcomm = create_comm(mpicomm)

    hwp_rpm = 120.0
    hwp_name = defaults.hwp_angle
    if telescope_name == "LAT":
        hwp_name = None
        hwp_rpm = None

    # Simulated telescope
    telescope = sotoast.simulated_telescope(
        hw=None,
        hwfile=None,
        telescope_name=telescope_name,
        sample_rate=sample_rate,
        bands=bands,
        wafer_slots=wafer_slot,
        thinfp=thin_fp,
        comm=mpicomm,
    )

    data = toast.Data(toastcomm)

    # Create a schedule.

    if cal_schedule:
        schedule = calibration_schedule(telescope)
    else:
        schedule = observing_schedule(telescope, mpicomm=toastcomm.comm_world)

    sim_ground = toast.ops.SimGround(
        name="sim_ground",
        telescope=telescope,
        session_split_key="tele_wf_band",
        schedule=schedule,
        hwp_angle=hwp_name,
        hwp_rpm=hwp_rpm,
        weather="atacama",
        median_weather=True,
        detset_key=detset_key,
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second ** 2,
        use_ephem=False,
        use_qpoint=True,
    )
    sim_ground.apply(data)

    # corotator = so_ops.CoRotator(name="corotate_lat")
    # corotator.apply(data)

    return data


def simulation_test_multitube(
    mpicomm,
    telescope_name="LAT",
    tubes=None,
    sample_rate=1.0 * u.Hz,
    detset_key="pixel",
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    thin_fp=30,
):
    """Create a data object with a simple ground sim.

    This function generates data for a complete telescope at low sample rate, in order
    to test operations that need this.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        telescope_name (str):  the telescope to simulate.
        sample_rate (Quantity): the sample rate.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    if not toast_available:
        raise RuntimeError("TOAST is not importable, cannot simulate test data")

    # Create the communicator with a single group
    toastcomm = toast.Comm(world=mpicomm)

    # Simulated telescope
    telescope = sotoast.simulated_telescope(
        hw=None,
        hwfile=None,
        telescope_name=telescope_name,
        tube_slots=tubes,
        sample_rate=sample_rate,
        thinfp=thin_fp,
        comm=mpicomm,
    )

    data = toast.Data(toastcomm)

    # Create a schedule.

    schedule = observing_schedule(telescope, mpicomm=toastcomm.comm_world)

    hwp_name = None
    if telescope_name != "LAT":
        hwp_name = defaults.hwp_angle

    sim_ground = toast.ops.SimGround(
        name="sim_ground",
        telescope=telescope,
        session_split_key="tele_wf_band",
        schedule=schedule,
        weather="atacama",
        median_weather=True,
        detset_key=detset_key,
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second ** 2,
        hwp_angle=hwp_name,
    )
    sim_ground.apply(data)

    return data
