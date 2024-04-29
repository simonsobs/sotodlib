import numpy as np
from sotodlib.core import FlagManager

def det_splits_relative(aman, det_left_right=False, det_upper_lower=False, det_in_out=False, wrap=None):
    """
    Function for adding relative detector splits to aman. A new FlagManager called det_flags will be created and the flags put there.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    det_left_right: Bool
        Perform a detector left/right split
    det_upper_lower: Bool
        Perform a detector upper/lower split
    det_in_out: Bool
        Perform a detector in/out split
    wrap: Bool or str
        If True, the flags with the det splits will be wrapped to aman.det_flags. If a string, the flags with the det splits will be wrapped to aman.string

    Returns
    -------
    aman : AxisManager
        Updated aman with the new flags
    fm: FlagManager with the requested flags
    """

    fm = FlagManager.for_tod(aman)

    if det_left_right or det_in_out:
        xi = aman.focal_plane.xi
        xi_median = np.median(xi)    
    if det_upper_lower or det_in_out:
        eta = aman.focal_plane.eta
        eta_median = np.median(eta)
    if det_left_right:
        mask = xi <= xi_median
        fm.wrap_dets('det_left', np.logical_not(mask))
        mask = xi > xi_median
        fm.wrap_dets('det_right', np.logical_not(mask))
    if det_upper_lower:
        mask = eta <= eta_median
        fm.wrap_dets('det_lower', np.logical_not(mask))
        mask = eta > eta_median
        fm.wrap_dets('det_upper', np.logical_not(mask))
    if det_in_out:
        xi_center = np.min(xi) + 0.5 * (np.max(xi) - np.min(xi))
        eta_center = np.min(eta) + 0.5 * (np.max(eta) - np.min(eta))
        radii = np.sqrt((xi_center-xi)**2 + (eta_center-eta)**2)
        radius_median = np.median(radii)
        mask = radii <= radius_median
        fm.wrap_dets('det_in', np.logical_not(mask))
        mask = radii > radius_median
        fm.wrap_dets('det_out', np.logical_not(mask))

    if wrap is not None and wrap is not False:
        if wrap == True:
            aman.wrap('det_flags', fm)
        else:
            aman.wrap(wrap, fm)
    return aman, fm
