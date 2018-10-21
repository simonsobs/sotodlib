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
        "sotod_create_detdb = sotodlib.scripts.create_detdb.main",
    ]
}

setup_opts["name"] = "sotodlib"
setup_opts["provides"] = "sotodlib"
setup_opts["version"] = versioneer.get_version()
setup_opts["description"] = "Simons Observatory TOD Simulation and Processing"
setup_opts["author"] = "Simons Observatory Collaboration"
setup_opts["author_email"] = "so_software@simonsobservatory.org"
setup_opts["url"] = "https://github.com/simonsobs/sotodlib"
setup_opts["packages"] = ["sotodlib"]
setup_opts["license"] = "MIT"
setup_opts["requires"] = ["Python (>3.4.0)", ]

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

    def run(self):
        loader = unittest.TestLoader()
        runner = unittest.TextTestRunner(verbosity=2)
        suite = loader.discover("tests", pattern="test_*.py", top_level_dir=".")
        runner.run(suite)

# Add our custom test runner
cmdcls["test"] = SOTestCommand

# Add command class to setup options
setup_opts["cmdclass"] = cmdcls

# Do the setup.
setup(**setup_opts)
