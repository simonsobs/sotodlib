import datetime
from sotodlib import core
import logging

logger = logging.getLogger(__name__)


def correct_iir_params(aman):
    """
    Correct missing iir_params by default values.
    This corrects iir_params only when the observation is within the time_range
    that is known to have problem.

    See `sotodlib.tod_ops.filters.iir_filter` for more details of iir_params

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

    time_range = {
        'satp1': None,
        'satp2': None,
        'satp3': ['2023-01-01', '2024-10-24'],
        'lat': None,
    }[aman.obs_info.telescope]

    iir_missing = []
    for _field, _sub_iir_params in aman.iir_params._fields.items():
        if isinstance(_sub_iir_params, core.AxisManager):
            if 'a' in _sub_iir_params._fields:
                if _sub_iir_params['a'] is None:
                    iir_missing.append(_field)

    within_range = False
    if time_range is not None:
        t0 = datetime.datetime.strptime(time_range[0], '%Y-%m-%d')
        t1 = datetime.datetime.strptime(time_range[1], '%Y-%m-%d')
        t0 = t0.replace(tzinfo=datetime.timezone.utc).timestamp()
        t1 = t1.replace(tzinfo=datetime.timezone.utc).timestamp()
        within_range = aman.timestamps[0] >= t0 and aman.timestamps[-1] <= t1

    if within_range:
        for field in iir_missing:
            logger.warning(f'iir_params are missing on {field}. '
                           'Fill default params.')
            aman[f'iir_params.{field}.a'] = a
            aman[f'iir_params.{field}.b'] = b
    else:
        if len(iir_missing) > 0:
            raise ValueError('iir_params are missing but the observaiton is '
                             'not in a time range that is known to have a '
                             'problem. ' + ' '.join(iir_missing))
    return iir_missing
