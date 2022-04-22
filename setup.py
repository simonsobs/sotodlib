#!/usr/bin/env python3

import os
import sys
import re

import unittest

from setuptools import find_packages, setup, Extension
from setuptools.command.test import test as TestCommand

import versioneer

# The setup options
setup_opts = dict()

# Entry points / scripts.  Add scripts here and define the main() of each
# script in sotodlib.scripts.<foo>.main()
setup_opts["entry_points"] = {
    "console_scripts": [
        "so_hardware_sim = sotodlib.scripts.hardware_sim:main",
        "so_hardware_plot = sotodlib.scripts.hardware_plot:main",
        "so_hardware_trim = sotodlib.scripts.hardware_trim:main",
        "so_hardware_info = sotodlib.scripts.hardware_info:main",
        "so-metadata = sotodlib.core.metadata.manifest:main",
    ]
}

pipes = [
    "pipelines/toast_so_sim.py",
    "pipelines/toast_so_example.py",
    "pipelines/toast_bin_so3g.py",
    "pipelines/toast_so_tf.py",
    "pipelines/get_wafer_offset.py",
]

setup_opts["name"] = "sotodlib"
setup_opts["provides"] = "sotodlib"
setup_opts["version"] = versioneer.get_version()
setup_opts["description"] = "Simons Observatory TOD Simulation and Processing"
setup_opts["author"] = "Simons Observatory Collaboration"
setup_opts["author_email"] = "so_software@simonsobservatory.org"
setup_opts["url"] = "https://github.com/simonsobs/sotodlib"
setup_opts["packages"] = find_packages(where=".", exclude="tests")
setup_opts["license"] = "MIT"
setup_opts["requires"] = ["Python (>3.7.0)", ]
setup_opts["scripts"] = pipes
setup_opts["include_package_data"] = True
setup_opts["install_requires"] = [
    'numpy',
    'scipy',
    'matplotlib',
    'quaternionarray',
    'PyYAML',
    'toml',
    'skyfield',
    'pixell',
    'scikit-image',
    'pyfftw',
]

# Command Class dictionary.
# Begin with the versioneer command class dictionary.
cmdcls = versioneer.get_cmdclass()

# Class to run unit tests

class SOTestCommand(TestCommand):

    def __init__(self, *args, **kwargs):
        super(SOTestCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
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
