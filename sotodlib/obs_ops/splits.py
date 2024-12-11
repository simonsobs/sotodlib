import numpy as np
from sotodlib.core import FlagManager, AxisManager

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

    
    if wrap == True:
        if 'det_flags' in aman._fields:
            aman.move('det_flags', None)
        aman.wrap('det_flags', fm)
    elif isinstance(wrap, str):
        if wrap in aman._fields:
            aman.move(wrap, None)
        aman.wrap(wrap, fm)
    return fm

def get_split_flags(aman, proc_aman=None, split_cfg=None):
    '''
    Function returning flags used for null splits consumed by the mapmaking
    and bundling codes. Fields labeled ``field_name_flag`` contain boolean
    masks and ``_avg`` are the mean of the numerical based split flags to 
    be used for observation level splits.

    Arguments
    ---------
    aman: AxisManager
        Main axis manager containing signal.
    proc_aman: AxisManager
        Preprocess axis manager, usually loaded in ``aman.preprocess``.
    split_cfg: dict
        Dictionary containing the thresholds used for cutting

    Returns
    -------
    split_aman: AxisManager
        Axis manager containing splitting flags.
        ``cuts`` field is a FlagManager containing the detector and subscan based splits used in the mapmaker.
        ``<split_name>_threshold`` fields contain the threshold used for the split.
        Other fields conatain info for obs-level splits.
    '''
    if proc_aman is None:
        try:
            proc_aman = aman.preprocess
        except:
            raise ValueError('proc_aman is None and no preprocess field in aman provide valid preprocess metadata')

    if (not 't2p' in proc_aman) | (not 'hwpss_stats' in proc_aman):
        raise ValueError('t2p or hwpss_stats not in proc_aman must run after those steps in the pipeline.')

    # Set default set of splits
    default_cfg = {'high_gain': 0.115, 'high_noise': 3.5e-5, 'high_tau': 1.5e-3,
                   'det_A': 'A', 'pol_angle': 35, 'det_top': 'B', 'high_leakage': 1e-3,
                   'high_2f': 1.5e-3, 'right_focal_plane': 0, 'top_focal_plane': 0,
                   'central_pixels': 0.071 }
    if split_cfg is None:
        split_cfg = default_cfg

    split_aman = AxisManager(aman.dets)
    fm = FlagManager.for_tod(aman)
    # If provided split config doesn't include all of the splits in default
    for k in default_cfg.keys():
        if not k in split_cfg:
            split_cfg[k] = default_cfg[k]
        split_aman.wrap(f'{k}_threshold', split_cfg[k])

    # Gain split
    fm.wrap_dets('high_gain', aman.det_cal.phase_to_pW > split_cfg['high_gain'])
    fm.wrap_dets('low_gain', aman.det_cal.phase_to_pW <= split_cfg['high_gain'])
    split_aman.wrap('gain_avg', np.nanmean(aman.det_cal.phase_to_pW))
    # Noise split
    fm.wrap_dets('high_noise', proc_aman.noiseQ_fit.fit[:,1] > split_cfg['high_noise'])
    fm.wrap_dets('low_noise', proc_aman.noiseQ_fit.fit[:,1] <= split_cfg['high_noise'])
    split_aman.wrap('noise_avg', np.nanmean(proc_aman.noiseQ_fit.fit[:,1]))
    # Time constant split
    fm.wrap_dets('high_tau', aman.det_cal.tau_eff > split_cfg['high_tau'])
    fm.wrap_dets('low_tau', aman.det_cal.tau_eff <= split_cfg['high_tau'])
    split_aman.wrap('tau_avg', np.nanmean(aman.det_cal.tau_eff))
    # detAB split
    fm.wrap_dets('det_A', aman.det_info.wafer.pol <= split_cfg['det_A'])
    fm.wrap_dets('det_B', aman.det_info.wafer.pol > split_cfg['det_A'])
    # def pol split
    fm.wrap_dets('high_pol_angle', aman.det_info.wafer.angle > split_cfg['pol_angle'])
    fm.wrap_dets('low_pol_angle', aman.det_info.wafer.angle <= split_cfg['pol_angle'])
    # det top/bottom split
    fm.wrap_dets('det_type_top', aman.det_info.wafer.crossover > split_cfg['det_top'])
    fm.wrap_dets('det_type_bottom', aman.det_info.wafer.crossover <= split_cfg['det_top'])
    # T2P Leakage split
    fm.wrap_dets('high_leakage', np.sqrt(proc_aman.t2p.lamQ**2 + proc_aman.t2p.lamU**2) > split_cfg['high_leakage'])
    fm.wrap_dets('low_leakage', np.sqrt(proc_aman.t2p.lamQ**2 + proc_aman.t2p.lamU**2) <= split_cfg['high_leakage'])
    split_aman.wrap('leakage_avg', np.nanmean(np.sqrt(proc_aman.t2p.lamQ**2 + proc_aman.t2p.lamU**2)),
                    [(0, 'dets')])
    # High 2f amplitude split
    a2 = aman.det_cal.phase_to_pW*np.sqrt(proc_aman.hwpss_stats.coeffs[:,2]**2 + proc_aman.hwpss_stats.coeffs[:,3]**2)
    fm.wrap_dets('high_2f', a2 > split_cfg['high_2f'])
    fm.wrap_dets('low_2f', a2 <= split_cfg['high_2f'])
    split_aman.wrap('2f_avg', np.nanmean(a2), [(0, 'dets')])
    # Right/left focal plane split
    fm.wrap_dets('det_right', aman.focal_plane.xi > split_cfg['right_focal_plane'])
    fm.wrap_dets('det_left', aman.focal_plane.xi <= split_cfg['right_focal_plane'])
    # Top/bottom focal plane split
    fm.wrap_dets('det_upper', aman.focal_plane.eta > split_cfg['top_focal_plane'])
    fm.wrap_dets('det_lower', aman.focal_plane.eta <= split_cfg['top_focal_plane'])
    # Inner/outter pixel split
    r =  np.sqrt(aman.focal_plane.xi**2 + aman.focal_plane.eta**2)
    fm.wrap_dets('det_in', r < split_cfg['central_pixels'])
    fm.wrap_dets('det_out', r >= split_cfg['central_pixels'])
    # Left/right subscans
    if 'turnaround_flags' in proc_aman:
        fm.wrap('scan_left', proc_aman.turnaround_flags.left_scan)
        fm.wrap('scan_right', proc_aman.turnaround_flags.right_scan)

    split_aman.wrap('cuts', fm)

    return split_aman
