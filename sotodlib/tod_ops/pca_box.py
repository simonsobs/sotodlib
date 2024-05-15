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


def multi_run_pca(ctx, oids, ufm, ghz, num_runs=2):
    pca_run = 1  # Start with the first run
    good_dets = None

    for _ in range(num_runs):
        good_dets = results.get('good_dets', {}).get(
            'det_ids') if pca_run != 1 else None

        amans, pca_signals_cal = compute_pca(
            ctx, oids, ufm, ghz, pca_run, good_dets=good_dets)
        results = find_pcabounds(amans=amans, pca_signals_cal=pca_signals_cal)
        plot_pcabounds(
            amans=amans, pca_signals_cal=pca_signals_cal, results=results)

        print(f'Plotted some plots for run {pca_run}')

        # Increment to the next run
        pca_run += 1


def compute_pca(ctx, oids, ufm, ghz, pca_run, good_dets=None):
    """
    good_dets = dictionary of detids
    """
    metas = load_metas(ctx, oids, ufm)
    if good_dets and pca_run >= 2:
        amans = load_amans(metas, ctx, ghz, good_dets=good_dets)
    else:
        amans = load_amans(metas, ctx, ghz)

    amansf = jump_amans(amans)

    # low pass filter
    lpf_amans(amansf)

    # cal'd pca
    pca_signals_cal = pca_cal(amansf)

    pca_signals_cal['pca_run_iteration'] = pca_run

    return amansf, pca_signals_cal


def pca_cal(amans):
    """
    """
    pca_signals_cal = {}
    for aman in tqdm(amans, desc='Post-cal PCA'):
        aman.signal = np.multiply(aman.signal.T, aman.det_cal.phase_to_pW).T
        pca_out = pca.get_pca(aman)
        pca_signal = pca.get_pca_model(aman, pca_out)

        pca_signals_cal[aman.obs_info.obs_id] = pca_signal

    return pca_signals_cal


def find_pcabounds(amans, pca_signals_cal, **kwargs):
    """
    """
    xfac = kwargs.get('xfac', 2)
    yfac = kwargs.get('yfac', 1.5)

    results = {'good_dets': {'det_ids': {}},
               'baddets': {'det_ids': {}},
               'bounds': {}}
    for aman in amans:
        pca_signal = pca_signals_cal[aman.obs_info.obs_id]
        print('obs', aman.obs_info.obs_id)

        x = aman.det_cal.s_i
        y = np.abs(pca_signal.weights[:, 0])

        # remove positive Si values
        filt = np.where(x < 0)[0]
        xfilt = x[filt]
        yfilt = y[filt]

        # normalize weights
        ynorm = yfilt / np.median(yfilt)
        median_ynorm = np.median(ynorm)
        medianx = np.median(xfilt)

        # IQR of normalized weights
        q20 = np.percentile(ynorm, 20)
        q80 = np.percentile(ynorm, 80)
        iqry_norm = q80 - q20

        # IQR of Si's
        q20x = np.percentile(xfilt, 20)
        q80x = np.percentile(xfilt, 80)
        iqrx = q80x - q20x

        # Find box height using norm'd weights
        ylb_norm = median_ynorm - yfac * iqry_norm
        yub_norm = median_ynorm + yfac * iqry_norm

        # Convert y bounds back to the scale of the raw weights
        ylb = ylb_norm * np.median(yfilt)
        yub = yub_norm * np.median(yfilt)

        # Calculate box width
        xlb = medianx - xfac * iqrx
        xub = medianx + xfac * iqrx
        if xub > 0:
            mad = np.median(np.abs(xfilt - medianx))
            xub = medianx + xfac * mad

        xbounds = [xlb, xub]
        ybounds = [ylb, yub]

        # Get indices of the values in the box (indices are wrt `x` array)
        box_xfilt_inds = np.where((xfilt >= xlb) & (
            xfilt <= xub) & (yfilt >= ylb) & (yfilt <= yub))[0]
        box = filt[box_xfilt_inds]
        notbox = np.setdiff1d(np.arange(len(x)), box)

        goodids = aman.det_info.det_id[box]
        cutids = [detid for detid in goodids if detid != 'NO_MATCH']
        badids = aman.det_info.det_id[notbox]
        badcutids = [detid for detid in badids if detid != 'NO_MATCH']

        bad_removed = len(badids) - len(badcutids)
        good_removed = len(goodids) - len(cutids)

        if bad_removed > 0 or good_removed > 0:
            if bad_removed > 0:
                print(
                    f'NO_MATCH detectors removed from the bad detectors: {bad_removed}')
            if good_removed > 0:
                print(
                    f'NO_MATCH detectors removed from the good detectors: {good_removed}')

        # populate results dictionary
        results['bounds'].setdefault(aman.obs_info.obs_id, {}).update({
            'x': xbounds, 'y': ybounds})
        results['good_dets']['det_ids'].setdefault(
            aman.obs_info.obs_id, cutids)
        results['baddets']['det_ids'].setdefault(
            aman.obs_info.obs_id, badcutids)

        # saves with pca run this is (nominally, 1 or 2 but some might do > 2 runs)
        results['pca_run_iteration'] = pca_signals_cal['pca_run_iteration']
    return results


def plot_pcabounds(amans, pca_signals_cal, results):
    """
    """
    for aman in amans:
        pca_run = pca_signals_cal['pca_run_iteration']

        obs = aman.obs_info.obs_id
        pca_signal = pca_signals_cal[aman.obs_info.obs_id]
        print(f'plotting {obs}')

        goodids = results['good_dets']['det_ids'][obs]
        badids = results['baddets']['det_ids'][obs]

#        pdb.set_trace()
        ids = list(aman.det_info.det_id)
        good_indices = [ids.index(det_id) for det_id in goodids]
        bad_indices = [ids.index(det_id) for det_id in badids]

        xbounds = results['bounds'][obs]['x']
        ybounds = results['bounds'][obs]['y']

        ufm, ghz, _ = goodids[0].split('_')

        timestamps = aman.timestamps[::20]  # because of the lpf
        modes = pca_signal.modes[0][::20]

        fig = plt.figure(figsize=(10, 6))

        # Define axes
        ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
        ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
        ax3 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

        # ax1: good signals
        ax1.plot(timestamps, modes, color='black', linewidth=3,
                 label='0th mode', zorder=2, alpha=0.4)
        for ind in good_indices:
            weight = pca_signal.weights[ind, 0]
            signals = aman.signal[ind, ::20]
            ax1.plot(timestamps, signals / weight,
                     zorder=1, color='#D8BFD8', alpha=0.3)

        ax1.set_title(f'Good Detector Batch: ({len(goodids)} dets)')
        ax1.legend(loc='upper left')
        ax1.grid()

        # ax2: bad signals
        ax2.plot(timestamps, modes, color='black', linewidth=3,
                 label='0th mode', zorder=2, alpha=0.4)
        for ind in bad_indices:
            weight = pca_signal.weights[ind, 0]
            signals = aman.signal[ind, ::20]
            ax2.plot(timestamps, signals / weight,
                     zorder=1, color='#FFA07A', alpha=0.3)
        ax2.set_title(f'Bad Detector Batch: ({len(badids)} dets)')
        ax2.legend(loc='upper left')
        ax2.grid()

        # ax3: box
        weight = np.abs(pca_signal.weights[:, 0])
        Si = aman.det_cal.s_i
        ax3.plot(Si[good_indices], weight[good_indices], '.', color='#D8BFD8', markersize=10,
                 label=f'Good dets ({len(goodids)} dets)', alpha=0.3)

        ax3.plot(Si[bad_indices], weight[bad_indices], '.', color='#FFA07A', markersize=10,
                 label=f'Bad dets ({len(badids)} dets)', alpha=0.3)

        ax3.plot([xbounds[0], xbounds[1], xbounds[1], xbounds[0], xbounds[0]],
                 [ybounds[0], ybounds[0], ybounds[1], ybounds[1], ybounds[0]],
                 color='navy', linestyle='-.', linewidth=1.5, label='Boundary',
                 alpha=1)

        unique_bias_groups = sorted(set(aman.det_cal.bg))
        bias_groups = aman.det_cal.bg
        markers = ['o', '^', '^', 'D', 'x', '+']
        bias_group_markers = {
            bg: markers[i % len(markers)] for i, bg in enumerate(unique_bias_groups)}
        # legend_handles = []
        # for bg in unique_bias_groups:
        #  marker = bias_group_markers[bg]
        #  if marker == 'o':
        #      legend_handles.append(plt.Line2D([0], [0], marker=marker, color='gray', markersize=7, label=f'Bias Group {bg}', markerfacecolor='none', markeredgewidth=1))
        #   else:
        #       legend_handles.append(plt.Line2D([0], [0], marker=marker, color='gray', markersize=7, label=f'Bias Group {bg}', markerfacecolor='none', markeredgewidth=1))
        # for i in range(len(bias_groups)):
        #   bg = bias_groups[i]
        #    marker = bias_group_markers[bg]  # Get marker for the bias group
        #    # Choose color based on whether the index is in good_indices or not
        #    color = '#D8BFD8' if i in good_indices else '#FFA07A'
        #    ax3.plot(Si[i], weight[i], marker, color=color, alpha=0.3)
        # ax3.legend(handles=legend_handles, loc='upper right')

        ax3.set_xlabel('Si')
        ax3.set_ylabel('0th Mode Weights')

        if any(value > 0 for value in Si):
            ax3.set_xlim(np.min(Si), 0)

        ax3.legend()
        ax3.grid()

        fig.suptitle(f'{ufm} {ghz} {aman.obs_info.obs_id[0:20]}')

        # Automatically adjust padding horizontally and vertically
        plt.tight_layout()
        plt.show()

        plt.savefig(
            f'NEW_{ufm}_{ghz}_{aman.obs_info.obs_id[0:20]}_pca{pca_run}.png')
        plt.figure()


# PREPROCESSING


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

            if all(angle == 0 for angle in aman.hwp_angle):
                print(f'no HWP data in {aman.obs_info.obs_id}')
                continue
            else:
                hwp.hwp.get_hwpss(aman)
                hwp.hwp.subtract_hwpss(aman)
            pbar.set_description(f'remove hwpss - {idx+1}/{len(metas)}')
            pbar.update(1)

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
