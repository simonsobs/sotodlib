# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory core routines.

This module has containers and in-memory structures for data and metadata.

"""
from .context import Context, OBSLOADER_REGISTRY

from .axisman import AxisManager
from .axisman import IndexAxis, OffsetAxis, LabelAxis

from .flagman import FlagManager

from .hardware import Hardware

from . import util

from .resources import get_local_file, RESOURCE_DEFAULTS
