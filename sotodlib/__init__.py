# Copyright (c) 2018-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Simons Observatory TOD Simulation and Processing.

This package contains tools for simulating and analyzing data from the
Simons Observatory.  It uses multiple external dependencies (some optional).
See documentation of the various sub-modules for more details.

Contents:

    * core:  Container classes for data, metadata, etc.
    * io:  Code for reading / writing data.
    * utils:  General utility functions and data helper functions.
    * scripts:  Commandline entry points.
    * Top level packages for actual data processing.

"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import logging
logger = logging.getLogger(__name__)
_log_fmt = '%(levelname)s: %(name)s: %(message)s'
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(_log_fmt))
logger.addHandler(_ch)
