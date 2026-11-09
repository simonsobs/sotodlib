# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools to enable loading toast data from a context.
"""

import numpy as np
import yaml

from toast.utils import Logger
from toast.io import (
    VolumeIndex,
    load_hdf5,
)

from .. import core


def update_databases():
    # Query existing observations

    # Query HDF5 volume index to find new observations

    # Get context-specific rows from upstream DBs for our new observations

    # Override some columns of incoming upstream rows

    # Write new rows to our DBs
    pass


def create_context(volume_path, context_path, obsdb_path=None, obsfiledb_path=None, starting_context=None):
    update_databases()


def load_obs_toast(
    db,
    obs_id,
    dets=None,
    prefix=None,
    samples=None,
    no_signal=None,
    no_headers=None,
    special_channels=None,
    **kwargs
):
    pass



core.OBSLOADER_REGISTRY["toast"] = load_obs_toast
