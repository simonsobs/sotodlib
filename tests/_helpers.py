# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Internal helper functions for unit tests.
"""

import os


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
