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
    toast_available = True
except ImportError as e:
    toast_available = False


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


def create_outdir(subdir=None, comm=None):
    """Create the top level output directory and per-test subdir.

    Args:
        subdir (str): the sub directory for this test.
        comm (MPI.Comm):  Optional communicator.

    Returns:
        str: full path to the test subdir if specified, else the top dir.

    """
    rank = 0
    if comm is not None:
        rank = comm.rank
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
    if comm is not None:
        retdir = comm.bcast(retdir, root=0)
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


# FIXME:  PR #183 has additional helper functions for instrument classes.
# Just use those once that is merged and delete this.
def toast_site():
    if not toast_available:
        print("toast unavailable- cannot create site")
        return
    return toast.instrument.GroundSite(
        "ATACAMA",
        -22.958 * u.degree,
        -67.786 * u.degree,
        5200.0 * u.meter,
    )


def observing_schedule(telescope, temp_dir=None):
    if not toast_available:
        print("toast unavailable- cannot build observing schedule")
        return
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
            "--start",
            "2025-01-01 00:00:00",
            "--stop",
            "2025-01-01 00:10:00",
            "--out",
            sch_file,
        ]
    )
    schedule = toast.schedule.GroundSchedule()
    schedule.read(sch_file)
    if temp_dir is None:
        shutil.rmtree(tdir)
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
            sun_az_begin=180 * u.degree,
            sun_az_end=180 * u.degree,
            sun_el_begin=20 * u.degree,
            sun_el_end=20 * u.degree,
            moon_az_begin=180 * u.degree,
            moon_az_end=180 * u.degree,
            moon_el_begin=20 * u.degree,
            moon_el_end=20 * u.degree,
            moon_phase=0.0,
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
