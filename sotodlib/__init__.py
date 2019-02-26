# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory TOD Simulation and Processing.

This package contains tools for simulating and analyzing data from the
Simons Observatory.  It uses multiple external dependencies (some optional).
See documentation of the various sub-modules for more details.

Contents:

    * db: database access for detector properties, observations, etc.
    * toast: tools specifically for working with the external TOAST package.
    * scripts: commandline entry points.

"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
