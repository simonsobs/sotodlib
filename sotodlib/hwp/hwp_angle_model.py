import numpy as np


def apply_hwp_angle_model(tod, on_sign_ambiguous='fail'):
    """
    Returns the tod AxisManager after applying hwp angle model parameters.
    hwp_angle_model correct the sign and offset of hwp_angle.

    hwp_angle_corrected = sign * hwp_angle + offset

    Args:
        tod: AxisManager
        on_sign_ambiguous: Tolerance options for sign ambiguous
            fail: raise an error if there is any sign ambiguous
            pid: raise an error if pid sign does not exist
            offcenter: raise an error if offcenter sign does not exist

    Returns:
        tod: AxisManager with hwp_angle attribute
    """

    telescope = tod.obs_info.telescope
    if "lat" in telescope:
        return tod

    hwp = tod['hwp_solution']

    # sign matrix contains absolute sign correction factor for each
    # relative sign estimators.
    sign_matrix = {
        'satp1': {'pid':  1, 'offcenter': -1, },
        'satp3': {'pid': -1, 'offcenter':  1, },
    }

    methods = ['pid', 'offcenter']
    if on_sign_ambiguous in methods:
        sign = hwp[on_sign_ambiguous + '_direction']
        sign *= sign_matrix[telescope][on_sign_ambiguous]
        if sign == 0:
            raise ValueError('hwp rotation direction is ambiguous')
    elif on_sign_ambiguous == 'fail':
        available_signs = []
        for method in methods:
            _sign = hwp[method + '_direction']
            if _sign != 0:
                available_signs.append(_sign*sign_matrix[telescope][method])

        # check agreements of available estimation methods
        if len(available_signs) == 0:
            raise ValueError('hwp rotation direction is ambiguous')
        if np.all(np.array(available_signs) == available_signs[0]):
            sign = available_signs[0]
        else:
            raise ValueError('hwp rotation direction is ambiguous')
    else:
        raise ValueError('Invalid on_sign_ambiguous')

    # angle offset (degree).
    # Mechanically determined offsets
    encoder_offset = -1.66
    encoder_assembly_offset = {
        'satp1': {1: -90, 2: 90},
        'satp3': {1: 90, 2: -90},
    }
    # Added small origin offset correction
    # (center of reference slot to next edge)
    # June 2024; value confirmed by WG measurememnts CW vs. CCW.
    encoder_origin_offset = -1 * sign * 360 / 1140 * 3 / 2

    # angle offset of the optical axis of the achromatic hwp
    # This will be loaded from hwp_angle_model metadata
    ahwp_offset = {
        'satp1': 49.1,
        'satp3': -2.29,
    }

    offset = encoder_offset + \
        encoder_assembly_offset[telescope][hwp.primary_encoder] + \
        encoder_origin_offset + ahwp_offset[telescope]

    # apply correction
    hwp_angle = np.mod(sign*hwp.hwp_angle + np.deg2rad(offset), 2*np.pi)
    if 'hwp_angle' in tod._assignments.keys():
        tod.hwp_angle = hwp_angle
    else:
        tod.wrap('hwp_angle', hwp_angle, [(0, "samps")])

    return tod
