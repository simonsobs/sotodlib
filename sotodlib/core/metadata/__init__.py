# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Metadata containers.
"""

from .resultset import ResultSet
from .detdb import DetDb
from .obsdb import ObsDb
from .obsfiledb import ObsFileDb
from .manifest import ManifestDb, ManifestScheme
from .loader import SuperLoader, LoaderInterface, Unpacker, merge_det_info
from . import cli

def get_example(db_type, *args, **kwargs):
    if db_type == 'DetDb':
        from .detdb import get_example
        return get_example(*args, **kwargs)
    else:
        raise ValueError('Unknown db_type: %s' % db_type)
