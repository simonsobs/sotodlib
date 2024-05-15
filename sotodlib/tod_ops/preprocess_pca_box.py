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
# NOTE: Docstrings not added to all functions cus i'm lazy and i know it won't be integrated
# into sotodlib

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
        print('oid', id)
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
        meta (list): list of metadata amans from `load_metas`
        ctx (??): Context instance (?)
        ghz (str): defines the bandpass of interest; i.e., 'f90s', 'f150s'
        good_dets (dict or None): dictionary of detector ids for a set of observations
            that were identified as good detectors in the first round of pca cuts. if None,
            no specific detectors are removed.

    Returns:
        amans (list): list of preprocessed observation amans

    """
    amans = []
    with tqdm(total=len(metas) * 3, desc='loading obs amans') as pbar:
        for idx, meta in enumerate(metas):
            aman = ctx.get_obs(meta)

            # detrend signal
            aman.signal = detrend_tod(aman, method='median', in_place=True)

            pbar.set_description(f'detrend signals - {idx+1}/{len(metas)}')
            pbar.update(1)

            flags.get_det_bias_flags(
                aman, rfrac_range=(0.05, 0.9), psat_range=(0, 20))
            bad_dets = has_all_cut(aman.flags.det_bias_flags)
            cut_dets = np.sum(bad_dets)

            if cut_dets > 700:
                print(
                    f'{aman.obs_info.obs_id} not a viable obs, it cut {cut_dets} dets')
                continue

            if good_dets:
                good = good_dets[aman.obs_info.obs_id]
                aman.restrict(
                    "dets", aman.dets.vals[np.isin(aman.det_info.det_id, good)])
            
            try:
                if all(angle == 0 for angle in aman.hwp_angle):
                    print(f'no HWP data in {aman.obs_info.obs_id}')
                    continue
                else:
                    hwp.hwp.get_hwpss(aman)
                    hwp.hwp.subtract_hwpss(aman)
                pbar.set_description(f'remove hwpss - {idx+1}/{len(metas)}')
                pbar.update(1)
            except UnboundLocalError as e:
                print(f'Unbound local error for the wafer for {aman.obs_info.obs_id}', e)

            # bandpass
            try:
                if ghz == 'f90s':
                    aman.restrict('dets', aman.dets.vals[np.isin(
                        aman.det_cal.bg, [0, 1, 4, 5, 8, 9])])
                    # print('Number of 90 GHz detectors', aman.dets.count)
                elif ghz == 'f150s':
                    aman.restrict('dets', aman.dets.vals[np.isin(
                        aman.det_cal.bg, [2, 3, 6, 7, 10, 11])])
                    # print('Number of 150 GHz detectors', aman.dets.count)

                # trending cuts
                tf = flags.get_trending_flags(
                    aman, n_pieces=10, max_trend=2.5)  # good for hour long obs
                trend_dets = has_any_cuts(tf)
                aman.restrict('dets', aman.dets.vals[~trend_dets])
                pbar.set_description(f'trending cuts - {idx+1}/{len(metas)}')
                pbar.update(1)
            except RuntimeWarning:
                print(
                    f"trending cuts cut all the detectors for {aman.obs_info.obs_id}")
                continue

            amans.append(aman)

    # check if there are too few detectors after bias flags and trending cuts
    amansf = [arr for arr in amans if arr.dets.count >= 40]

    return amansf

# min_sigma = 4


def jump_amans(amans):
    """
    Set a default value for min_sigma but should consider making that more fluid
    """
    with tqdm(total=len(amans)*3, desc='handling jumps') as pbar:
        for idx, aman in enumerate(amans):
            # 2pi jumps
            jranges1, heights1, fixed1 = sotodlib.tod_ops.jumps.twopi_jumps(
                aman, overwrite=True)
            n_cut1 = count_cuts(jranges1)
            keep1 = n_cut1 <= 5
            aman.restrict("dets", aman.dets.vals[keep1])
            sotodlib.tod_ops.gapfill.fill_glitches(
                aman, glitch_flags=aman.flags.jumps_2pi, nbuf=10, modes=1, use_pca=False, wrap='signal')

            pbar.set_description(f'2Pi Jumps - {idx+1}/{len(amans)}')
            pbar.update(1)

            # Slow jumps
            jranges2, heights2, fixed2 = sotodlib.tod_ops.jumps.slow_jumps(
                aman, overwrite=True)
            n_cut2 = count_cuts(jranges2)
            keep2 = n_cut2 <= 5
            aman.restrict("dets", aman.dets.vals[keep2])
            sotodlib.tod_ops.gapfill.fill_glitches(
                aman, glitch_flags=aman.flags.jumps_slow, nbuf=10, modes=1, use_pca=False, wrap='signal')

            pbar.set_description(f'Slow Jumps - {idx+1}/{len(amans)}')
            pbar.update(1)

            # find jumps with min_sigma = 4
            jranges3, heights3, fixed3 = sotodlib.tod_ops.jumps.find_jumps(
                aman, min_sigma=4, fix=True, overwrite=True)
            n_cut3 = count_cuts(jranges3)
            keep3 = n_cut3 <= 5
            aman.restrict("dets", aman.dets.vals[keep3])
            sotodlib.tod_ops.gapfill.fill_glitches(
                aman, glitch_flags=aman.flags.jumps, nbuf=10, modes=1, use_pca=False, wrap='signal')
            pbar.set_description(f'Find Jumps- {idx+1}/{len(amans)}')
            pbar.update(1)

    # checking for dets counts in amans to remove any faulty amans
    amans_filt = [arr for arr in amans if arr.dets.count >= 40]

    return amans_filt


def lpf_amans(amans):
    """
    """
    for aman in tqdm(amans, desc='Low pass filtering'):
        sine2lpf = sotodlib.tod_ops.filters.fourier_filter(aman,
                                                           sotodlib.tod_ops.filters.low_pass_sine2(
                                                               10, width=0.5),
                                                           detrend=None)

        aman.signal = sine2lpf
        # cutting first and last 30s (may be too aggressive?)
        aman.restrict("samps", (200*30, -200*30))