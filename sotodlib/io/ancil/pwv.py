#
# Combine PWV data sources to get PWV stats for each observation.
#
import numpy as np
from scipy.interpolate import interp1d

import so3g

from . import utils
from . import configcls as cc


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
    if len(rs0.rows) == 0:
        return {'mean': np.nan,
                'std': np.nan,
                'qual': 0.}

    t, p = np.array(rs0.rows).T

    # How much of time_range is actually covered by measurements?
    coverage_i = _buffered_times(t, buffer_t)
    tr_i = so3g.IntervalsDouble.from_array(np.array([time_range]))
    tr_i = tr_i * coverage_i
    t_total = np.sum(tr_i.array() * [-1, 1])
    q = t_total / (time_range[1] - time_range[0])
    
    # Make an interpolator from those...
    pwv_i = interp1d(t, p)
    
    # Probe times...
    ptimes = np.arange(time_range[0], time_range[1], 60)
    s = (t[0] <= ptimes) * (ptimes <= t[-1])
    y = pwv_i(ptimes[s])
    return {'mean': y.mean().round(3),
            'std': y.std().round(3),
            'qual': q.round(3)}


@cc.register_engine('pwv-combo', cc.PwvComboConfig)
class PwvCombo(utils.AncilEngine):
    result_fields = ['mean', 'std', 'qual']

    def __init__(self, cfg):
        super().__init__(cfg)
        self.friends = {
            self.cfg.toco_dataset: None,
            self.cfg.apex_dataset: None,
        }

    def getter(self, targets=None, results=None, raw=False, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        toco_src = self._get_friend(self.cfg.toco_dataset)
        apex_src = self._get_friend(self.cfg.apex_dataset)

        for time_range in time_ranges:
            buffered = (time_range[0] - 4 * 3600,
                        time_range[1] + 4 * 3600)
            rs_toco = toco_src._load(buffered)
            rs_apex = apex_src._load(buffered)
            x = combine_pwv(rs_toco, rs_apex, time_range)
            yield utils.denumpy(x)
