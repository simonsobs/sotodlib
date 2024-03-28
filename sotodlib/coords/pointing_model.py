import numpy as np
from .. import core
from .helpers import _valid_arg

DEG = np.pi / 180


def apply_pointing_model(tod):
    """Redefines tod.boresight using pointing model parameters.

    Args:
        tod: AxisManager

    Returns:
        tod.boresight will be updated.
    """

    _a = _valid_arg("ancil", src=tod)
    telescope = _valid_arg("obs_info", src=tod)["telescope"]
    # Start with a fresh axismanager
    _b = core.AxisManager(core.OffsetAxis("samps", len(tod.timestamps)))

    if "lat" in telescope:
        # Returns in Radians
        _b.wrap("az", _a.az_enc * DEG, [(0, "samps")])
        _b.wrap("el", _a.el_enc * DEG, [(0, "samps")])
        _b.wrap("roll", (_a.el_enc - 60 - _a.corotator_enc) * DEG, [(0, "samps")])

    elif "sat" in telescope:
        # Returns in radians
        _b.wrap("az", _a.az_enc * DEG, [(0, "samps")])
        _b.wrap("el", _a.el_enc * DEG, [(0, "samps")])
        _b.wrap("roll", -1 * _a.boresight_enc * DEG, [(0, "samps")])

    tod["boresight"] = _b
