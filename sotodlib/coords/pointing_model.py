# Copyright (c) 2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np
from .. import core
from .helpers import _valid_arg
from so3g.proj import quat

import logging
logger = logging.getLogger(__name__)

DEG = np.pi / 180


def apply_basic_pointing_model(tod):
    """
    Updates and returns the tod.boresight AxisManager after applying pointing model parameters.

    Args:
        tod: AxisManager (must contain 'ancil' field)

    Returns:
        _boresight : AxisManager containing new boresight data.
    """

    _ancil = _valid_arg("ancil", src=tod)
    telescope = _valid_arg("obs_info", src=tod)["telescope"]
    # Start with a fresh axismanager
    _boresight = core.AxisManager(tod.samps)

    if "lat" in telescope:
        # Returns in Radians
        _boresight.wrap("az", _ancil.az_enc * DEG, [(0, "samps")])
        _boresight.wrap("el", _ancil.el_enc * DEG, [(0, "samps")])
        _boresight.wrap(
            "roll", (_ancil.el_enc - 60 - _ancil.corotator_enc) * DEG, [(0, "samps")]
        )

    elif "sat" in telescope:
        # Returns in radians
        _boresight.wrap("az", _ancil.az_enc * DEG, [(0, "samps")])
        _boresight.wrap("el", _ancil.el_enc * DEG, [(0, "samps")])
        _boresight.wrap("roll", -1 * _ancil.boresight_enc * DEG, [(0, "samps")])

    if 'boresight' in tod:
        del tod['boresight']

    tod.wrap('boresight', _boresight)

    return _boresight


def apply_pointing_model(tod, pointing_model=None, ancil=None,
                         wrap=None):
    """Applies a static pointing model to compute corrected boresight
    position and orientation in horizon coordinates.  The encoder
    values in tod.ancil are consumed as raw data, and the computed
    values are stored in tod.boresight.

    Args:
      tod (AxisManager): the observation data.
      pointing_model (AxisManager): if None, the pointing_model
        parameters are read from tod.pointing_model.
      ancil (AxisManager): if None, the encoders are read from
        tod.ancil.
      wrap (str): If specified, the name in tod where corrected
        boresight should be stored.  If None, the default of
        'boresight' is used.  Pass wrap=False to not store the
        result in tod.

    Returns:
      AxisManager: the corrected boresight.

    """
    if pointing_model is None and 'pointing_model' not in tod:
        logger.warning('No pointing_model found -- applying basic model.')
        assert wrap in (None, 'boresight'), \
            'When using naive pointing model, wrap=... not supported'
        return apply_basic_pointing_model(tod)

    pointing_model = _valid_arg(pointing_model, 'pointing_model',
                                src=tod)
    ancil = _valid_arg(ancil, 'ancil', src=tod)

    # Encoder values, to radians.
    vers = pointing_model['version']
    tel_type = vers.split('_')[0]
    if tel_type == 'sat':
        boresight = apply_pointing_model_sat(vers, pointing_model, tod, ancil)
    elif tel_type == 'lat':
        boresight = apply_pointing_model_lat(vers, pointing_model, tod, ancil)
    else:
        raise ValueError(f'Unimplemented pointing model "{vers}"')

    if wrap is None:
        wrap = 'boresight'
    if wrap is not False:
        if wrap in tod._fields:
            del tod[wrap]
        tod.wrap(wrap, boresight)
    return boresight


def apply_pointing_model_sat(vers, params, tod, ancil):
    az, el, roll = _get_sat_enc_radians(ancil)

    if vers == 'sat_naive':
        return _new_boresight(ancil.samps, az=az, el=el, roll=roll)

    elif vers == 'sat_v1':
        az1, el1, roll1 = model_sat_v1(params, az, el, roll)
        return _new_boresight(ancil.samps, az=az1, el=el1, roll=roll1)

    else:
        raise ValueError(f'Unimplemented pointing model "{vers}"')


def apply_pointing_model_lat(vers, tod, pointing_model, ancil):
    raise ValueError(f'Unimplemented pointing model "{vers}"')


#
# SAT model(s)
#

# sat_v1: you can expand v1, as long as new params don't do anything
# if their value is zero (and that should be the registered default).

defaults_sat_v1 = {
    'enc_offset_az': 0.,
    'enc_offset_el': 0.,
    'enc_offset_boresight': 0.,
    'base_tilt_cos': 0.,
    'base_tilt_sin': 0.,
    'bs_xi0': 0.,
    'bs_eta0': 0.,
}

def model_sat_v1(params, az, el, roll):
    """Applies pointing model to (az, el, roll).

    Args:
      params: AxisManager (or dict) of pointing parameters.
      az, el, roll: naive horizon coordinates, in radians, of the
        boresight.

    The implemented model parameters are:

      - bs_{xi,eta}0: within the focal plane (i.e. relative to the
        corrected boresight), the center of rotation of the boresight.
        Radians.
      - enc_offset_{az,el,boresight}: encoder offsets, in degrees.
      - base_tilt_{cos,sin}: base tilt coefficients, in radians.

    """
    _p = dict(defaults_sat_v1)
    if isinstance(params, dict):
        _p.update(params)
    else:
        _p.update({k: params[k] for k in params._fields.keys()})
    params, _p = _p, None

    for k, v in params.items():
        if k == 'version':
            continue
        if k not in defaults_sat_v1 and v != 0.:
            raise ValueError(f'Handling of model param "{k}" is not implemented.')

    # Construct offsetted encoders.
    az = az + params['enc_offset_az'] * DEG
    el = el + params['enc_offset_el'] * DEG
    roll = roll - params['enc_offset_boresight'] * DEG

    # Rotation that tilts the base (referred to vals after enc correction).
    base_tilt = get_base_tilt_q(params['base_tilt_cos'], params['base_tilt_sin'])

    # Rotation that takes a vector in array-centered focal plane coords
    # to a vector in boresight-rotation-centered focal plane coords.
    q_fp_bs = ~quat.rotation_xieta(params['bs_xi0'], params['bs_eta0'])

    # Horizon coordinates.
    q_hs = (base_tilt * quat.rotation_lonlat(-az, el)
            * ~q_fp_bs * quat.euler(2, roll) * q_fp_bs)

    neg_az, el, roll = quat.decompose_lonlat(q_hs)
    return -neg_az, el, roll


# Support functions

def _new_boresight(samps, az=None, el=None, roll=None):
    boresight = core.AxisManager(samps)
    for k, v in zip(['az', 'el', 'roll'], [az, el, roll]):
        boresight.wrap_new(k, shape=('samps',), dtype='float64')
        if v is not None:
            boresight[k][:] = v
    return boresight

def _get_sat_enc_radians(ancil):
    return (ancil.az_enc * DEG,
            ancil.el_enc * DEG,
            -ancil.boresight_enc * DEG)

def get_base_tilt_q(c, s):
    """Returns the quaternion rotation that applies base tilt, taking
    vectors in the platforms horizon coordinates to vectors in the
    site's local horizon coordinates.  The c and s parameters together
    define a direction and amplitude of the base tilt.

    In this implementation, c and s have the same meaning and sign
    convention as TPOINT parameters AN and AW, respectively.

    """
    # Imagine az=-phi
    phi = np.arctan2(s, c)
    # And that base tilt causes the true el to lie below the expected
    # (encoder) el, at that position.
    amp = (c**2 + s**2)**.5
    return quat.euler(2, phi) * quat.euler(1, amp) * quat.euler(2, -phi)
