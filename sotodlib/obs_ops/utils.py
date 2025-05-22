import datetime
from sotodlib import core
import logging
import numpy as np

logger = logging.getLogger(__name__)


def correct_iir_params(aman, ignore_time=False, check_srate=-1):
    """
    Correct missing iir_params by default values.
    This corrects iir_params only when the observation is within the time_range
    that is known to have problem.

    See `sotodlib.tod_ops.filters.iir_filter` for more details of iir_params

    Parameters
    ----------
    aman: AxisManager of observation

    ignore_time: Boolean. True if we don't want to check if the observation is within
                 a known bad time range.

    check_srate: If greater than 0 will check that the observations sample rate is within
                 check_srate Hz of 200 Hz. If less than 0 the check is skipped.

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

    time_ranges = {
        'satp1': None,
        'satp2': None,
        'satp3': [('2023-01-01', '2024-10-24')],
        'lat': [(1745970733, 1746624934)],
    }[aman.obs_info.telescope]

    iir_missing = []
    for _field, _sub_iir_params in aman.iir_params._fields.items():
        if isinstance(_sub_iir_params, core.AxisManager):
            if 'a' in _sub_iir_params._fields:
                if _sub_iir_params['a'] is None:
                    iir_missing.append(_field)

    within_range = False
    if time_ranges is not None and not ignore_time:
        for t0, t1 in time_ranges:
            if isinstance(t0, str):
                t0 = datetime.datetime.strptime(t0, '%Y-%m-%d')
                t0 = t0.replace(tzinfo=datetime.timezone.utc).timestamp()
            if isinstance(t1, str):
                t1 = datetime.datetime.strptime(t1, '%Y-%m-%d')
                t1 = t1.replace(tzinfo=datetime.timezone.utc).timestamp()
            within_range = aman.timestamps[0] >= t0 and aman.timestamps[-1] <= t1
            if within_range:
                break

    if check_srate >= 0:
        srate = 1./np.mean(np.diff(aman.timestamps))
        if abs(srate - 200) > check_srate:
            raise ValueError(f"Sample rate is {srate}, too far from 200 Hz to use default params.")

    if within_range or ignore_time:
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
