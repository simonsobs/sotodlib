import numpy as np
from .. import core
from .helpers import _valid_arg

DEG = np.pi / 180


def apply_pointing_model(tod):
    """
    Returns an updated boresight AxisManager after applying pointing model parameters.

    Args:
        tod: AxisManager (must contain 'ancil' field)

    Returns:
        _boresight : AxisManager containing new boresight data.
    """

    _ancil = _valid_arg("ancil", src=tod)
    telescope = _valid_arg("obs_info", src=tod)["telescope"]
    # Start with a fresh axismanager
    _boresight = core.AxisManager(core.OffsetAxis("samps", len(tod.timestamps)))

    if "lat" in telescope:
        # Returns in Radians
        _boresight.wrap("az", _ancil.az_enc * DEG, [(0, "samps")])
        _boresight.wrap("el", _ancil.el_enc * DEG, [(0, "samps")])
        _boresight.wrap("roll", (_ancil.el_enc - 60 - _ancil.corotator_enc) * DEG, [(0, "samps")])

    elif "sat" in telescope:
        # Returns in radians
        _boresight.wrap("az", _ancil.az_enc * DEG, [(0, "samps")])
        _boresight.wrap("el", _ancil.el_enc * DEG, [(0, "samps")])
        _boresight.wrap("roll", -1 * _ancil.boresight_enc * DEG, [(0, "samps")])

    return _boresight
