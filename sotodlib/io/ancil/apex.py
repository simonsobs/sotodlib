"""This ancil submodule specializes in radiometer data from the APEX
telescope.

"""

import logging
import math
import requests

import datetime as dt
import numpy as np

from . import utils
from . import configcls as cc


logger = logging.getLogger(__name__)

APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'


def _to_timestamp(targets):
    """Convert one or more time-like targets to a unix timestamp.  Returns
    scalar (if targets is scalar) or array.  String targets can be
    YYYY-MM-DD or YYYY-MM-DDT:HH:MM:SS format.

    """
    if not hasattr(targets, '__getitem__'):
        return _to_timestamp([targets])[0]
    out = []
    for a in targets:
        if isinstance(a, (float, int)):
            out.append(float(a))
        elif isinstance(a, str):
            a = a.strip().replace(' ', 'T')
            if 'T' not in a:
                a = a + 'T00:00:00'
            out.append(_str_to_timestamp(a))
        else:
            raise ValueError(f"Cannot interpret as timestamp: '{a}'")
    return np.array(out)


def _str_to_timestamp(d):
    t = dt.datetime.fromisoformat(d)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.timestamp()

def _timestamp_to_str(t):
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) \
                      .strftime("%Y-%m-%dT%H:%M:%S")

def _pwv_to_float(p):
    if p == '':
        return np.nan
    return float(p)


def _parse_apex_csv(text):
    schema = {
        'Date time': ('timestamp', _str_to_timestamp),
        'Precipitable Water Vapor [mm]': ('pwv', _pwv_to_float),
        'Shutter Mode': ('shutter', str),
    }
    lines = [line for line in text.split('\n')
             if line.strip() != '' and line.strip()[0] != '#']
    headers, casts = zip(*[schema.get(h, (h, str)) for h in lines.pop(0).split(',')])
    rows = [[c(v) for c, v in zip(casts, row.split(','))] for row in lines]
    return utils.ResultSet(keys=headers, src=rows)


def get_apex(t0, t1, url=None, raw=False, max_rows=75000):
    if url is None:
        url = APEX_DATA_URL
    t0, t1 = _to_timestamp([t0, t1])
    # We expect about 1 point per minute.
    if (t1 - t0) > max_rows * 60 * .95:
        raise ValueError(f"Time range requested ({t1-t0} s) might "
                         f"require more than {max_rows} rows.")
    t0, t1 = map(_timestamp_to_str, [t0, t1])
    data = {
        'wdbo': 'csv/download',
        'max_rows_returned': max_rows,
        'start_date': f'{t0}..{t1}',
        'tab_pwv': 'on',
        ## Don't ask for shutter column, but also don't insist that
        ## shutter be open; process the nans.
        # 'shutter': 'SHUTTER_OPEN',
        # 'tab_shutter': 'on',
    }
    r = requests.post(url, data=data)
    if raw:
        return r
    return _parse_apex_csv(r.text)


@cc.register_engine('apex-pwv', cc.ApexPwvConfig)
class ApexPwv(utils.LowResTable):
    _fields = [
        ('mean', 'float'),
        ('start', 'float'),
        ('end', 'float'),
        ('span', 'float'),
    ]

    def _get_raw(self, time_range):
        return get_apex(time_range[0], time_range[1])

    def getter(self, targets=None, results=None, **kwargs):
        """Compute reduced APEX PWV stats for a bunch of time ranges.  Each
        entry in targets is a time range.

        """
        time_ranges = self._target_time_ranges(targets)
        for time_range in time_ranges:
            buf_range = (time_range[0] - 3600, time_range[1] + 3600)
            rs = self._load(buf_range)
            s = np.isfinite(rs['pwv'])
            s1 = (time_range[0] <= rs['timestamp']) * (rs['timestamp'] < time_range[1])
            if not np.any(s):
                data = {
                    'mean': math.nan,
                    'start': math.nan,
                    'end': math.nan,
                    'span': math.nan,
                }
            else:
                if (s1 * s).any():
                    s = s1 * s
                p = rs['pwv'][s]
                data = {
                    'mean': np.median(p).round(3),
                    'start': p[0].round(3),
                    'end': p[-1].round(3),
                    'span': (max(p) - min(p)).round(3),
                }
            yield utils.denumpy(data)


class ApexDataMocker:
    def __init__(self, t_max=None):
        self.t_max = t_max

    def get_raw(self, time_range):
        t0, t1 = time_range
        keys = ['timestamp', 'pwv', 'shutter']
        T = 120
        t0 = (t0 + T - 1) - t0 % T
        t1 = (t1 - t1 % T) + T / 2
        if self.t_max is not None:
            t1 = min(self.t_max, t1)
        tt = np.arange(t0, t1, T)
        pwv = tt * 0 + 0.77
        shutter = np.array(['OPEN'] * len(pwv))
        return utils.ResultSet(keys=keys, src=zip(tt, pwv, shutter))
