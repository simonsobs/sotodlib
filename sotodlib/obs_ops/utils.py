from sotodlib import core
import logging

logger = logging.getLogger(__name__)


def correct_iir_params(aman):
    """
    Add iir_params if missing

    See `sotodlib.tod_ops.filters.iir_filter` for more detail

    Parameters
    ----------
    aman: AxisManager of observation

    Returns
    -------
    List of field names that have no iir_params

    """
    a = [1., -3.74145562, 5.25726624, -3.28776591,
         0.77203984, 0., 0., 0.,
         0., 0., 0., 0.,
         0., 0., 0., 0.]
    b = [5.28396689e-06, 2.11358676e-05, 3.17038014e-05, 2.11358676e-05,
         5.28396689e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

    iir_missing = []
    for _field, _sub_iir_params in aman.iir_params._fields.items():
        if isinstance(_sub_iir_params, core.AxisManager):
            if 'a' in _sub_iir_params._fields:
                if _sub_iir_params['a'] is None:
                    iir_missing.append(_field)
                    logger.warning(f'iir_params are missing on {_field}. '
                                   'Fill default params.')
                    _sub_iir_params['a'] = a
                    _sub_iir_params['b'] = b

    return iir_missing
