"""
Various AxisManager utility functions
"""

import numpy as np
from sotodlib.core import AxisManager

class RestrictionException(Exception):
    """Exception for when cannot restrict AxisManager properly"""

def restrict_to_times(am, t0, t1, in_place=False):
    """
    Restricts axis manager to a time range (t0, t1)
    """
    m = (t0 <= am.timestamps) & (am.timestamps < t1)
    if not m.any():
        raise RestrictionException()
    i0, i1 = np.where(m)[0][[0, -1]] + am.samps.offset
    return am.restrict('samps', (i0, i1+1), in_place=in_place)

def dict_to_am(d, skip_bad_types=False):
    """
    Attempts to convert a dictionary into an AxisManager. This can be used on
    dicts containing basic types such as (str, int, float) along with numpy
    arrays. The AxisManager will not have any structure such as "axes", but
    this is useful if you want to nest semi-arbitrary data such as the "meta"
    dict into an AxisManager.

    Args
    -----
    d : dict
        Dict to convert ot axismanager
    skip_bad_types : bool
        If True, will skip any value that is not a str, int, float, or
        np.ndarray. If False, this will raise an error for invalid types.
    """
    allowed_types = (str, int, float, np.ndarray)
    am = AxisManager()
    for k, v in d.items():
        if isinstance(v, dict):
            am.wrap(k, dict_to_am(v))
        elif isinstance(v, allowed_types):
            am.wrap(k, v)
        elif not skip_bad_types:
            raise ValueError(
                f"Key {k} is of type {type(v)} which cannot be wrapped by an "
                 "axismanager")
    return am
