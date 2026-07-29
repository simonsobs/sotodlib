# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""This module supports the retrieval and repackaging of various
ancillary data pertinent to processing of Simons Observatory survey
and calibration data.  These ancillary data include such things as
radiometer readings (from off-site or on-site devices), weather
information, details about the behavior of the platform (such as scan
speed) and other instrumentation (half-wave plate; wire grid).

The two main roles of the submodule are:

  1. The assembly of datasets for faster and more convenient access to
     information that is not already made available in a convenient
     way.

  2. Reduction of ancillary data to the level of per-observation
     statistitics that can be stored in an observation database, and
     used to classify / select telescope data.

The main user-facing interface in this module is a set of classes,
each responsible for assembling and reducing a specific dataset. They
expose common interfaces for updating the base data archive (if such
is needed) and for computing per-observation statistics from the data
for obsdb purposes.

"""

import logging
logger = logging.getLogger(__name__)

#: Map from engine identifier to engine class.  This is constructed
#: dynamically on module load.
ANCIL_ENGINES = {}

from . import configcls
from . import utils

from . import apex, so_hk, pwv
