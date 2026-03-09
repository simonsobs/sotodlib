"""This ancil submodule specializes in datastreams saved to SO HK data feeds, including:

- the Toco radiometer data

"""

import h5py
import logging
import math
import time
import os

import numpy as np

from sotodlib.io import hkdb
from . import utils
from . import configcls as cc


logger = logging.getLogger(__name__)


@cc.register_engine('toco-pwv', cc.TocoPwvConfig)
class TocoPwv(utils.LowResTable):
    _fields = [
        ('mean', 'float'),
        ('start', 'float'),
        ('end', 'float'),
        ('span', 'float'),
    ]

    def _get_raw(self, time_range):
        cfg = hkdb.HkConfig.from_yaml(self.cfg.hkdb_config)
        cfg.aliases['pwv'] = 'env-radiometer-class.pwvs.pwv'

        db = hkdb.HkDb(cfg)
        lspec = hkdb.LoadSpec(
            cfg=cfg, start=0, end=0,
            fields=['pwv'],
            hkdb=db,
        )

        lspec.start, lspec.end = map(float, time_range)  # de-numpy
        result = hkdb.load_hk(lspec, show_pb=False)
        if not hasattr(result, 'pwv'):
            result.pwv = [[], []]
        return utils.ResultSet(keys=['timestamp', 'pwv'],
                              src=zip(*result.pwv))

    def getter(self, targets=None, results=None, raw=False, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        for time_range in time_ranges:
            buf_range = (time_range[0] - 3600, time_range[1] + 3600)
            rs = self._load(buf_range)
            p = rs['pwv']
            if len(p) == 0:
                yield {
                    'mean': math.nan,
                    'start': math.nan,
                    'end': math.nan,
                    'span': math.nan,
                }
                continue

            # Replace out-of-bounds negative with 0.15, because that in
            # the middle of the dead zone.
            p[p<0.3] = 0.15
            # Note any out-of-bounds positive points
            high = (p > 3.)
            # Reject sections that have too many high points (and so
            # patch with APEX).
            if high.sum() / len(high) > .3:
                p[:] = np.nan
            # But if there aren't too many, just replace them with pwv=3.
            p[high] = 3.

            # Prefer values in time range.
            s1 = (time_range[0] <= rs['timestamp']) * (rs['timestamp'] < time_range[1])
            if s1.any():
                p = p[s1]

            yield utils.denumpy({
                'mean': np.median(p).round(3),
                'start': p[0].round(3),
                'end': p[-1].round(3),
                'span': (max(p) - min(p)).round(3),
            })


class TocoPwvMocker:
    def __init__(self, t_max=None):
        self.t_max = t_max

    def get_raw(self, time_range):
        keys = ['timestamp', 'pwv']
        T = 60
        t0, t1 = time_range
        t0 = t0 - t0 % T
        t1 = (t1 - t1 % T) + T / 2
        if self.t_max is not None:
            t1 = min(self.t_max, t1)
        tt = np.arange(t0, t1, T)
        pwv = tt * 0 + 0.77
        return utils.ResultSet(keys=keys, src=zip(tt, pwv))
