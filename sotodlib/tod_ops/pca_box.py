import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sotodlib
from sotodlib import hwp
from sotodlib.tod_ops.detrend import detrend_tod
from sotodlib.tod_ops import flags, pca
from sotodlib.core.flagman import has_any_cuts, has_all_cut, count_cuts
from preprocess_pca_box import load_amans, load_metas, jump_amans, lpf_amans

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def find_pcabounds(amans, pca_signals_cal, **kwargs):
    """Finds the bounds of the pca box using IQR 
    statistics

    Parameters
    ----------
    amans : list
        list of observation axismanagers
    pca_signals_cal : dict
        dictionary of pca axismanagers per observation

    Returns
    -------
    dict
        `results` dictionary with the x and y bounds, and det id's of 
        good and bad detectors for each axismanager/observation

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


def plot_pcabounds(amans, pca_signals_cal, results, ufm, ghz):
    """Subplot of pca bounds as well as the good and bad detector
    timestreams with 0th mode weight overplotted

    Parameters
    ----------
    amans : list
        list of preprocessed observation axismanagers
    pca_signals_cal : dict
        dictionary of pca axismanagers for each observation
    results : dict
        dict of bounds and good and bad detector id's
    ufm : str
        UFM name (for plotting purposes)
    ghz : str
        Bandpass (for plotting purposes)

    """
    for aman in amans:
        pca_run = pca_signals_cal['pca_run_iteration']

        obs = aman.obs_info.obs_id
        pca_signal = pca_signals_cal[aman.obs_info.obs_id]
        print(f'plotting {obs}')

        goodids = results['good_dets']['det_ids'][obs]
        badids = results['baddets']['det_ids'][obs]

        ids = list(aman.det_info.det_id)
        good_indices = [ids.index(det_id) for det_id in goodids]
        bad_indices = [ids.index(det_id) for det_id in badids]

        xbounds = results['bounds'][obs]['x']
        ybounds = results['bounds'][obs]['y']

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

        # Todo: incorporate bias groups into Si vs weight plot
        # unique_bias_groups = sorted(set(aman.det_cal.bg))
        # bias_groups = aman.det_cal.bg
        # markers = ['o', '^', '^', 'D', 'x', '+']
        # bias_group_markers = {
        #    bg: markers[i % len(markers)] for i, bg in enumerate(unique_bias_groups)}
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
        plt.tight_layout()
        plt.savefig(
            f'{ufm}_{ghz}_{aman.obs_info.obs_id[0:20]}_pca{pca_run}.png')

        plt.figure()
