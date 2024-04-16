import numpy as np


def apply_hwp_angle_model(tod, band='f090', hwp_solution_attr='hwp_solution', hwp_angle_model_attr='hwp_angle_model'):
    """
    Updates and returns the tod AxisManager after applying hwp angle model parameters.
    hwp_angle_model correct the sign and offset of hwp_angle.

    hwp_angle_corrected = sign * hwp_angle + offset

    Args:
        tod: AxisManager

    Returns:
        tod: AxisManager containing new hwp_angle data
    """

    telescope = tod.obs_info.telescope
    if "lat" in telescope:
        return tod

    hwp = getattr(tod, hwp_solution_attr)

    # sign
    # Relative sign between observations will be estimated from hwp_metadata by multiple methods
    # sign matrix contains absolute sign correction factor for each relative sign estimation method.
    # sign matrix will be loaded from hwp_angle_model
    # We will be able to add "scan" and "template"
    sign_matrix = {
        'satp1': {'pid':  1, 'offcenter': -1, },
        'satp3': {'pid': -1, 'offcenter':  1, },
    }

    # check available sign estimation methods
    available_signs = []

    methods = ['pid', 'offcenter']
    for method in methods:
        _sign = getattr(hwp, method + '_direction')
        if _sign != 0:
            available_signs.append(_sign*sign_matrix[telescope][method])

    # check agreements of available estimation methods
    if len(available_signs) == 0:
        print('WARNING: relative hwp rotation direction is ambiguous')
        sign = 1.
    if np.all(np.array(available_signs) == available_signs[0]):
        sign = available_signs[0]
    else:
        print('WARNING: relative hwp rotation direction is ambiguous')
        sign = 1.

    # offset (degree). Convert to radian when we apply correction
    # Mechanically determined offsets
    encoder_offset = -1.66
    encoder_assembly_offset = {
        'satp1': {1: -90, 2: 90},
        'satp3': {1: 90, 2: -90},
    }
    # offset of the optical axis of the achromatic hwp
    # this needs to be calibrated optically, this will be loaded from hwp_angle_model
    ahwp_offset = {
        'satp1': {'f090': 49.1, 'f150': 49.4},
        'satp3': {'f090': -2.29, 'f150': -1.99},
    }

    offset = encoder_offset + \
        encoder_assembly_offset[telescope][hwp.primary_encoder] + \
        ahwp_offset[telescope][band]

    # apply correction
    tod.hwp_angle = np.mod(
        sign*tod.hwp_angle + np.deg2rad(offset), 2*np.pi)

    return tod
