import numpy as np
from sotodlib import core
import logging

logger = logging.getLogger(__name__)


def default_model(telescope):
    """Returns default hwp_angle_model. This will be deleted in future.
    """

    model = core.AxisManager()
    if telescope == 'satp1':
        sign = core.AxisManager()
        sign.wrap('pid', 1)
        sign.wrap('offcenter', -1)
        model.wrap('mechanical_offset_1', np.deg2rad(-1.66 - 90 + 49.1))
        model.wrap('mechanical_offset_2', np.deg2rad(-1.66 + 90 + 49.1))

    elif telescope == 'satp2':
        sign = core.AxisManager()
        sign.wrap('pid', 1)
        sign.wrap('offcenter', -1)
        model.wrap('mechanical_offset_1', np.deg2rad(-1.66 - 90 + 7.7))
        model.wrap('mechanical_offset_2', np.deg2rad(-1.66 + 90 + 7.7))

    elif telescope == 'satp3':
        sign = core.AxisManager()
        sign.wrap('pid', -1)
        sign.wrap('offcenter', 1)
        model.wrap('mechanical_offset_1', np.deg2rad(-1.66 + 90 - 2.29))
        model.wrap('mechanical_offset_2', np.deg2rad(-1.66 - 90 - 2.29))

    else:
        raise ValueError('Not supported yet')

    model.wrap('sign_matrix', sign)
    model.wrap('sign_dependent_offset', np.deg2rad(-1. * 360 / 1140 * 3 / 2))
    model.wrap('model_version', 'v2')
    return model


def correct_sign(tod, telescope):
    """
    correct hwp angle sign which is known to be wrong.

    Args:
        tod: hwp_solution AxisManager
        telescope: name of telescope
    """
    if telescope == 'satp1':
        # start time, stop time, sign to correct
        correction = [
            # pid direction was wrong due to the bug of hwp-pid agent
            [1717446106, 1717706400, 'pid'],
            # satp1 SCR6 has unusual offcentering
            # stop time needs to be updated later
            [1751000000, 2000000000, 'offcenter'],
        ]
        for t0, t1, method in correction:
            if (t0 <= tod.timestamps[0]) and (tod.timestamps[-1] <= t1):
                tod[method + '_direction'] *= -1
                logger.info('This observation has known to have wrong sign'
                            f' of {method}. Apply correction.')


def apply_hwp_angle_model(tod, on_sign_ambiguous='fail'):
    """
    Applies `hwp_angle_model` to the `hwp_solution` and construct hwp angle
    calibrated in the telescope frame.
    This will populate the calibrated hwp angle as `hwp_angle` attribute.
    `hwp_solution` is the hwp angle measured in the encoder frame.
    `hwp_angle_model` corrects the sign and offset of `hwp_solution`.

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

    hwp = tod.get('hwp_solution', None)
    if hwp is None:
        raise ValueError('hwp_solution is missing')
    model = tod.get('hwp_angle_model', None)
    if model is None:
        logger.warn('hwp_angle_model metadata is missing. '
                    'Apply default model. This may be old.')
        model = default_model(telescope)

    assert model.model_version == 'v2'
    correct_sign(hwp, telescope)
    # construct sign
    methods = model.sign_matrix.keys()
    if on_sign_ambiguous in methods:
        sign = hwp[on_sign_ambiguous + '_direction']
        sign *= model.sign_matrix[on_sign_ambiguous]
        if sign == 0:
            raise ValueError('hwp rotation direction is ambiguous')
    elif on_sign_ambiguous == 'fail':
        available_signs = []
        for method in methods:
            _sign = hwp[method + '_direction']
            if _sign != 0:
                available_signs.append(_sign*model.sign_matrix[method])

        # check agreements of available estimation methods
        if len(available_signs) == 0:
            raise ValueError('hwp rotation direction is ambiguous')
        if np.all(np.array(available_signs) == available_signs[0]):
            sign = available_signs[0]
        else:
            raise ValueError('hwp rotation direction is ambiguous')
    else:
        raise ValueError('Invalid on_sign_ambiguous')

    # apply correction
    hwp_angle = np.mod(sign * (hwp.hwp_angle + model.sign_dependent_offset)
                + model[f'mechanical_offset_{hwp.primary_encoder}'], 2*np.pi)
    if 'hwp_angle' in tod._assignments.keys():
        tod.hwp_angle = hwp_angle
    else:
        tod.wrap('hwp_angle', hwp_angle, [(0, "samps")])

    return tod
