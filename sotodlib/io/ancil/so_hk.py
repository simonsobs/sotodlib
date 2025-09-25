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


@cc.register_engine('weather-station', cc.WeatherStationConfig)
class WeatherStation(utils.AncilEngine):
    result_fields = ['wind_speed', 'wind_dir', 'uv', 'ambient_temp']

    def getter(self, targets=None, results=None, **kwargs):
        time_ranges = self._target_time_ranges(targets)

        cfg = hkdb.HkConfig.from_yaml(self.cfg.hkdb_config)
        cfg.aliases = {k: f'env-vantage.weather_data.{k}'
                       for k in ['temp_outside', 'UV', 'wind_speed', 'wind_dir']}

        db = hkdb.HkDb(cfg)
        lspec = hkdb.LoadSpec(
            cfg=cfg, start=0, end=0,
            fields=cfg.aliases,
            hkdb=db,
        )
        for time_range in time_ranges:
            lspec.start, lspec.end = time_range
            result = hkdb.load_hk(lspec, show_pb=False)

            s = (np.isfinite(result.wind_speed[1])
                 * np.isfinite(result.wind_dir[1]))
            wspd = result.wind_speed[1][s] * 1.609 # miles to km.
            wphi = np.radians(result.wind_dir[1][s])
            vx, vy = wspd * np.cos(wphi), wspd * np.sin(wphi)
            speed = ((np.mean(vx)**2 + np.mean(vy)**2)**.5).round(1)
            direc = (np.rad2deg(np.arctan2(np.mean(vy),
                                          np.mean(vx))) % 360.).round(1)
            yield utils.denumpy({
                'wind_speed': speed,
                'wind_dir': direc,
                'uv': np.nanmedian(result.UV[1]).round(1),
                'ambient_temp': np.nanmedian(result.temp_outside[1]).round(2),
            })


# Model for converting APEX PWV to Toco PWV.
_A2T_LIN = np.array([0.78775417, 0.04700976])
_A2T_GAUSS = (0.18090913, 1.60488581, 0.38610678)

def apex_pwv_to_toco_pwv(v):
    a, mu, sig = _A2T_GAUSS
    gau = a*np.exp(-0.5*(x - mu)**2 / sig**2)
    lin = np.polyval(_A2T_LIN, v)
    return lin + gau


@cc.register_engine('toco-pwv', cc.TocoPwvConfig)
class TocoPwv(utils.LowResTable):
    result_fields = ['mean', 'start', 'end', 'span']
    
    def _get_raw(self, time_range):
        cfg = hkdb.HkConfig.from_yaml(self.cfg.hkdb_config)
        cfg.aliases['pwv'] = 'env-radiometer-class.pwvs.pwv'

        db = hkdb.HkDb(cfg)
        lspec = hkdb.LoadSpec(
            cfg=cfg, start=0, end=0,
            fields=['pwv'],
            hkdb=db,
        )

        lspec.start, lspec.end = time_range
        result = hkdb.load_hk(lspec, show_pb=False)
        if not hasattr(result, 'pwv'):
            result.pwv = [[], []]
        return utils.ResultSet(keys=['timestamp', 'pwv'],
                              src=zip(*result.pwv))

    def getter(self, targets=None, results=None, raw=False, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        for time_range in time_ranges:
            rs = self._load(time_range)
            p = rs['pwv']
            if len(p) == 0:
                yield {
                    'mean': math.nan,
                    'start': math.nan,
                    'end': math.nan,
                    'span': math.nan,
                }
                continue
                
            # Replace out-of-bounds negative with 0.15
            p[p<0.3] = 0.15
            # Note any out-of-bounds positive points
            high = (p > 3.)
            # Reject sections that have too many high points (and so
            # patch with APEX).
            if high.sum() / len(high) > .3:
                p[:] = np.nan
            # But if there aren't too many, just replace them with pwv=3.
            p[high] = 3.

            print(np.median(p).round(3))
            yield utils.denumpy({
                'mean': np.median(p).round(3),
                'start': p[0].round(3),
                'end': p[-1].round(3),
                'span': (max(p) - min(p)).round(3),
            })
