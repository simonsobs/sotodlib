#
# Combine PWV data sources to get PWV stats for each observation.
#
import numpy as np
from scipy.interpolate import interp1d

import so3g

from . import utils

def _times_inside(t, ints):
    # This is not fast.
    mask = np.zeros(t.shape, bool)
    for lo, hi in ints.array():
        mask += (lo <= t) * (t < hi)
    return mask


def _buffered_times(t, buffer_t):
    return so3g.IntervalsDouble.from_array(
        np.array([t - buffer_t, t + buffer_t], dtype='float64').T)
    

def combine_pwv(rs_toco, rs_apex, time_range):
    """
    """
    # The idea is to use toco where you can, but fill it in with apex.

    # Create intervals around all toco points.
    buffer_t = 5 * 60
    
    # Start with toco.
    t, p = rs_toco['timestamp'], rs_toco['pwv']
    s = np.isfinite(p) * (p > .3) * (p < 3.)
    toco_i = _buffered_times(t[s], buffer_t)
    
    # Keep APEX points in ranges not covered by toco.
    keep_apex = (_times_inside(rs_apex['timestamp'], ~toco_i)
                 * np.isfinite(rs_apex['pwv']))

    rs0 = rs_toco.subset(rows=s)
    rs1 = rs_apex.subset(rows=keep_apex)

    utils.merge_rs(rs0, rs1)

    # How much of time_range is actually covered by measurements?
    coverage_i = _buffered_times(rs0['timestamp'], buffer_t)
    tr_i = so3g.IntervalsDouble.from_array(np.array([time_range]))
    tr_i = tr_i * coverage_i
    t_total = np.sum(tr_i.array() * [-1, 1])
    q = t_total / (time_range[1] - time_range[0])
    
    # Make an interpolator from those...
    pwv_i = interp1d(rs0['timestamp'], rs0['pwv'])
    
    # Probe times...
    y = pwv_i(np.arange(time_range[0], time_range[1], 60))
    return {'pwv': y.mean(),
            'pwv_std': y.std(),
            'pwv_qual': q,
            'rs': rs0}


class PwvCombo(utils.AncilEngine):
    DEFAULTS = {
        'dataset_name': 'pwv',
        'toco-dataset': 'toco-pwv',
        'apex-dataset': 'apex-pwv',
    }

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.friends = {
            self.cfg['toco-dataset']: None,
            self.cfg['apex-dataset']: None,
        }

    def getter(self, targets=None, results=None, raw=False, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        toco_src = self._get_friend(self.cfg['toco-dataset'])
        apex_src = self._get_friend(self.cfg['apex-dataset'])

        for time_range in time_ranges:
            buffered = (time_range[0] - 4 * 3600,
                        time_range[1] + 4 * 3600)
            rs_toco = toco_src._load(buffered)
            rs_apex = apex_src._load(buffered)
            x = combine_pwv(rs_toco, rs_apex, time_range)
            yield x
