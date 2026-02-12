"""This ancil submodule specializes in combining PWV readings from the
APEX and Toco radiometers.

"""

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


# The PWV correction model version 250701 includes corrections to both
# APEX and Toco readings.  Toco readings are debiased using a
# non-linear correction that amounts to a removing a gaussian-shaped
# feature centered around 1.43 mm.  APEX readings are adjusted with a
# linear correction to make them consistent on average with corrected
# Toco readings.  For more details see wiki:
#
#   https://simonsobs.atlassian.net/wiki/x/FgBBRQ
#

params_250701 = [
    # Linear model for rescaling apex to toco scale
    [0.78, 0.04],
    # Gaussian params for removing bias bump from toco radiometer
    [-0.20, 1.43, 0.33],
]

def _gaussian(v, amp, mu, sigma):
    return amp * np.exp(-0.5 * (v - mu)**2 / sigma**2)

def defeature_toco_250701(v):
    return v + _gaussian(v, *params_250701[1])

def apex_to_tocolin_250701(v):
    m, b = params_250701[0]
    return m * v + b


def combine_pwv(rs_toco, rs_apex, time_range):
    """Combine Toco and Apex PWV datasets (provided as ResultSet with
    timestamp and pwv columns) and produce statistics covering the
    stated time_range (timestamp, timestamp).

    The provided datasets should extend at least a few minutes beyond
    the time_range boundaries, so as to facilitate interpolation.

    PWV correction model is applied to the data before combination.
    Then, Toco points are used as much as possible but filled in with
    APEX whenever there's a gap of more than 10 minutes in the Toco
    stream.

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

    # Apply models to the data.
    toco_cor = rs_toco.subset(rows=s).asarray()
    apex_cor = rs_apex.subset(rows=keep_apex).asarray()

    del rs_toco, rs_apex

    toco_cor['pwv'] = defeature_toco_250701(toco_cor['pwv'])
    apex_cor['pwv'] = apex_to_tocolin_250701(apex_cor['pwv'])

    times = np.hstack((toco_cor['timestamp'], apex_cor['timestamp']))
    pwv = np.hstack((toco_cor['pwv'], apex_cor['pwv']))
    dataset_idx = np.array([0] * len(toco_cor) +
                           [1] * len(apex_cor))

    if len(times) == 0:
        return {'mean': np.nan,
                'std': np.nan,
                'qual': 0.}

    ii = np.argsort(times)
    times = times[ii]
    pwv = pwv[ii]
    dataset_idx = dataset_idx[ii]

    # How much of time_range is actually covered by measurements?
    coverage_i = _buffered_times(times, buffer_t)
    tr_i = so3g.IntervalsDouble.from_array(np.array([time_range]))
    tr_i = tr_i * coverage_i
    t_total = np.sum(tr_i.array() * [-1, 1])
    q = t_total / (time_range[1] - time_range[0])

    # Make an interpolator from those...
    pwv_i = interp1d(times, pwv)

    # Probe times...
    npts = max(1, int(round((time_range[1] - time_range[0]) / 60.))) + 1
    ptimes = np.linspace(time_range[0], time_range[1], npts)
    s = (times[0] <= ptimes) * (ptimes <= times[-1])
    if np.any(s):
        y = pwv_i(ptimes[s])
    else:
        # This likely corresponds to q = 0 case... just average the nearby meas.
        y = pwv
    return {'mean': y.mean().round(3),
            'std': y.std().round(3),
            'qual': q.round(3)}


@cc.register_engine('pwv-combo', cc.PwvComboConfig)
class PwvCombo(utils.AncilEngine):
    """Combine and reduce PWV measurements from two data sources (APEX and
    Toco).  Note the values ultimately recorded are a combination of
    all available readings (from both sources) in the relevant time
    range.  Each source is corrected, using a model, to make them
    intercompatible and descriptive of the site.

    There is no explicitly managed "base data" for this module, as it
    relies on base data from Apex and Toco modules.

    The fields computed for obsdb are:

    - "mean": average corrected radiometer readings in mm.
    - "std": stdev of such readings.
    - "qual": a quality indicator (0 to 1) indicating roughly what
      fraction of the requested time interval is covered by the
      datasets.

    """

    _fields = [
        ('mean', 'float'),
        ('std', 'float'),
        ('qual', 'float'),
    ]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.friends = {
            self.cfg.toco_dataset: None,
            self.cfg.apex_dataset: None,
        }

    def check_base(self):
        info = {'constituents': []}
        for k, v in self.friends.items():
            info['constituents'].append(k)
        return info

    def getter(self, targets=None, results=None, raw=False, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        toco_src = self._get_friend(self.cfg.toco_dataset)
        apex_src = self._get_friend(self.cfg.apex_dataset)

        for time_range in time_ranges:
            buffered = (time_range[0] - 1 * 3600,
                        time_range[1] + 1 * 3600)
            rs_toco = toco_src._load(buffered)
            rs_apex = apex_src._load(buffered)
            x = combine_pwv(rs_toco, rs_apex, time_range)
            yield utils.denumpy(x)
