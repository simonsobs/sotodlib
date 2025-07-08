#!/usr/bin/env python3

import os
import sys
import unittest

from setuptools import find_packages, setup, Command
import versioneer

# The setup options
setup_opts = dict()

setup_opts["version"] = versioneer.get_version()
setup_opts["packages"] = find_packages(where=".", exclude="tests")

setup_opts["package_data"] = {
    "sotodlib": [
        "toast/ops/data/*"
    ]
}
setup_opts["include_package_data"] = True


# Command Class dictionary.
# Begin with the versioneer command class dictionary.
cmdcls = versioneer.get_cmdclass()

# Class to run unit tests

class SOTestCommand(Command):

    def __init__(self, *args, **kwargs):
        super(SOTestCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        Command.initialize_options(self)

    def finalize_options(self):
        Command.finalize_options(self)
        self.test_suite = True

    def mpi_world(self):
        # We could avoid putting this here by defining a custom TextTestRunner
        # that checked the per-process value "under-the-hood" and returned a
        # False value from wasSuccessful() if any process failed.
        comm = None
        if "MPI_DISABLE" not in os.environ:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
            except Exception:
                pass
        return comm

    def run(self):
        loader = unittest.TestLoader()
        runner = unittest.TextTestRunner(verbosity=2)
        suite = loader.discover("tests", pattern="test_*.py",
                                top_level_dir=".")
        ret = 0
        local_ret = runner.run(suite)
        if not local_ret.wasSuccessful():
            ret = 1
        comm = self.mpi_world()
        if comm is not None:
            ret = comm.allreduce(ret)
        sys.exit(ret)

# Add our custom test runner
cmdcls["test"] = SOTestCommand

# Add command class to setup options
setup_opts["cmdclass"] = cmdcls

# Do the setup.
setup(**setup_opts)
