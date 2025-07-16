import logging
import math
import numpy as np

from sotodlib import core
from sotodlib.hwp import hwp_angle_model as ham

from . import utils


logger = logging.getLogger(__name__)


class HwpStats(utils.AncilEngine):
    DEFAULTS = {}

    def update_base(self, time_range=None):
        pass

    def getter(self, targets=None, results=None, **kwargs):
        obs_ids = self._target_obs_ids(targets)

        ctx = core.Context(self.cfg['context_file'])

        for obs_id in obs_ids:
            try:
                meta = ctx.get_meta(obs_id)
            except core.metadata.loader.LoaderError as e:
                logger.debug(f'No result for obs_id={obs_id}; {e}')
                yield {
                    'hwp_stable': 0,
                    'hwp_rate': math.nan,
                    'hwp_vel': math.nan,
                    'hwp_dir': 0,
                }
                continue
            ham.apply_hwp_angle_model(meta)
            dt = np.diff(meta.hwp_solution.timestamps)
            dang = (np.diff(meta.hwp_angle) + np.pi) % (2*np.pi) - np.pi
            vel = (dang / dt) / (2 * np.pi) # rev / s
            
            # Do some median based screening
            mask = abs(dang - np.median(dang)) < .2
            stable = (mask.sum() / len(mask) > .95)
            typical_vel = vel[mask].mean().round(3)
            spin_dir = int(np.sign(typical_vel))
            yield utils.denumpy({
                'hwp_stable': int(stable),
                'hwp_rate': abs(typical_vel),
                'hwp_vel': typical_vel,
                'hwp_dir': spin_dir,
            })
