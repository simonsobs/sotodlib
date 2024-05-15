import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sotodlib
from sotodlib import hwp
from sotodlib.tod_ops.detrend import detrend_tod
from sotodlib.tod_ops import flags, pca
from sotodlib.core.flagman import has_any_cuts, has_all_cut, count_cuts


import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# PREPROCESSING
# NOTE: This is all preprocessing done for pca analysis but is just for personal use;
# the pca box method will take advantage of preprocessing that's already setup as part
# of the site pipeline architecture

def load_metas(ctx, oids, ufm):
    """ Given a set of observation ids and a context .yaml file,
    load up a list of metas restricted for a specific ufm

    Parameters:
        ctx (?): Context instance
        oids (list): list of observation ids

    Returns:
        metas (list): list of metadata Axismanagers
        ctx (??): a Context object that's required for preprocessing

    Note
    ----
        Currently assumes specific rfrac conditions.

    """
    metas = []
    for id in tqdm(oids, desc="loading metadata amans"):
        try:
            meta = ctx.get_meta(id)
            detsets = np.unique(meta.det_info.detset)
            detset = next((ds for ds in detsets if f'{ufm}' in ds), None)

            if detset is not None:
                meta.restrict(
                    'dets', meta.dets.vals[meta.det_info.detset == detset])
                m = (meta.det_cal.r_tes > 0) & (meta.det_cal.r_frac < 0.9)
                meta.restrict('dets', meta.dets.vals[m])

                metas.append(meta)

        except sotodlib.core.metadata.loader.IncompleteMetadataError:
            print(f'IncompleteMetadataError for {id}')
            continue

    # check if metas is empty
    if len(metas) == 0:
        logger.info("No valid metadata found")

    return metas


def load_amans(metas, ctx, ghz, good_dets=None):
    """ Loads and preprocesses an observation axismanager for each metadata axismanager

    Parameters:
