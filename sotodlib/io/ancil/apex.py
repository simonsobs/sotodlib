import logging
import math
import requests

import datetime as dt
import numpy as np


from . import utils


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


def get_apex(t0, t1, url=None, raw=False):
    if url is None:
        url = APEX_DATA_URL
    t0, t1 = map(_timestamp_to_str, _to_timestamp([t0, t1]))
    data = {
        'wdbo': 'csv/download',
        'max_rows_returned': 79400,
        'start_date': f'{t0}..{t1}',
        #'shutter': 'SHUTTER_OPEN',
        'tab_pwv': 'on',
        #'tab_shutter': 'on',
    }
    r = requests.post(url, data=data)
    if raw:
        return r
    return _parse_apex_csv(r.text)


class ApexPwv(utils.LowResTable):
    DEFAULTS = {
        'dataset_name': 'apex_pwv',
        'gap_size': 300,
        'dtypes': {'timestamp': 'float32',
                   'pwv': 'float32'},
    }

    def _get_raw(self, time_range):
        return get_apex(time_range[0], time_range[1])

    def getter(self, targets=None, results=None, **kwargs):
        """Compute reduced APEX PWV stats for a bunch of time ranges.  Each
        entry in targets is a time range.

        """
        time_ranges = self._target_time_ranges(targets)
        for time_range in time_ranges:
            rs = self._load(time_range)
            s = np.isfinite(rs['pwv'])
            if not np.any(s):
                data = {
                    'pwv_apex': math.nan,
                    'pwv_apex_start': math.nan,
                    'pwv_apex_end': math.nan,
                    'pwv_apex_span': math.nan,
                }
            else:
                p = rs['pwv'][s]
                data = {
                    'pwv_apex': np.median(p).round(3),
                    'pwv_apex_start': p[0].round(3),
                    'pwv_apex_end': p[-1].round(3),
                    'pwv_apex_span': (max(p) - min(p)).round(3),
                }
            yield utils.denumpy(data)
