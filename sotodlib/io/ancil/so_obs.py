import functools
import logging
import math
import os
import numpy as np

from sotodlib import core
from sotodlib.hwp import hwp_angle_model as ham

from . import utils
from . import configcls as cc

logger = logging.getLogger(__name__)


@cc.register_engine('hwp-stats', cc.HwpStatsConfig)
class HwpStats(utils.AncilEngine):
    _fields = [
        ('char', 'str'),
        ('vel', 'float'),
        ('stdev', 'float'),
        ('stability', 'float'),
    ]

    def getter(self, targets=None, results=None, **kwargs):
        model = _AngleModel(self.cfg.hwp_angle_model)
        hwpdb = core.metadata.ManifestDb(self.cfg.hwp_angles)
        hwp_root = os.path.split(self.cfg.hwp_angles)[0]

        no_sol = {
            'char': None,
            'vel': math.nan,
            'stdev': math.nan,
            'stability': math.nan,
        }
        ambig = {
            'char': 'ambig',
            'vel': math.nan,
            'stdev': math.nan,
            'stability': math.nan,
        }

        for obs in targets:
            m = model.get_model(obs)
            rec = hwpdb.match({'obs:obs_id': obs['obs_id']})
            if rec is None:
                # Probably just new...
                yield no_angles
                continue
            fn = os.path.join(hwp_root, rec['filename'])
            d = core.AxisManager.load(fn, rec['dataset'])
            t, a = _apply_model(obs, m, d)
            if a is None:
                yield ambig
                continue
            hwp = get_hwp_params(t, a)
            yield hwp


def _apply_model(obs, model, angles):
    aman = core.AxisManager()
    aman.wrap('obs_info', core.AxisManager())
    aman['obs_info'].wrap('telescope', obs['telescope'])
    aman.wrap('hwp_angle_model', model)
    aman.wrap('hwp_solution', angles)
    try:
        ham.apply_hwp_angle_model(aman)
    except ValueError:
        # Likely "direction ambiguous" error.
        return None, None
    return aman.hwp_solution.timestamps, aman.hwp_angle


def get_hwp_params(timestamps, hwp_angles):
    """Analyze hwp_angle timestream and characterize it with a few
    statistics.

    Args:
      timestamps: array of timestamps, in seconds.
      hwp_angles: array of hwp_angles, in radians.

    Returns a dict with entires:

    - ``char``: A string describing the overall behavior of the HWP
      during this time.
    - ``vel``: The typical rotation *velocity*, in rev/s, if that
      could be determined.
    - ``stdev``: The stdev of the rotation velocity relative to
      ``vel``.
    - ``stability``: Fraction of the time where spin seems to be
      constant rate.

    """

    dt = np.diff(timestamps)
    med_step = int(max(1, len(dt) / 10000))
    dt_typ = np.median(dt[::med_step])
    s = abs(dt/dt_typ - 1) < .1
    v = np.diff(hwp_angles)[s]
    v = (v + np.pi) % (2 * np.pi) - np.pi
    v /= (2 * np.pi) * dt[s]
    del dt, s

    v_typ = np.median(v[::med_step])
    stable = abs(v - v_typ) < .1

    v_mean = v[stable].mean()
    v_std = ((v - v_mean)**2).mean()**.5
    q = stable.sum() / len(stable)

    if q < .9:
        # Smooth the vel.
        t_smooth = 10.
        n_smooth = int(t_smooth / dt_typ)
        vds = np.convolve(v, np.ones(n_smooth)/n_smooth, 'valid')[::n_smooth]

        # First and last minute.
        vds0 = vds[:6].mean()
        vds1 = vds[-6:].mean()
        fast0, fast1 = [(abs(_v) > .4) for _v in [vds0, vds1]]
        slow0, slow1 = [(abs(_v) < .1) for _v in [vds0, vds1]]

        if fast0 and slow1:
            action = ('spindown')
        elif fast1 and slow0:
            action = ('spinup')
        else:
            action = ('unstable')
    else:
        action = 'stable'

    return utils.denumpy({
        'char': action,
        'vel': v_mean.round(3),
        'stdev': v_std.round(3),
        'stability': q.round(3),
    })


class _AngleModel:
    def __init__(self, hmod_file):
        self.db = core.metadata.ManifestDb(hmod_file)
        self.root = os.path.split(hmod_file)[0]

    @functools.lru_cache(maxsize=100)
    def _get_model(self, filename, dataset):
        return core.AxisManager.load(
            os.path.join(self.root, filename), dataset)

    def get_model(self, obs_info):
        res = self.db.match({f'obs:{k}': obs_info[k] for k in ['timestamp']})
        return self._get_model(res['filename'], res['dataset'])
