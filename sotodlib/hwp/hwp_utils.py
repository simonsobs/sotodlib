import time
import numpy as np
import matplotlib.pyplot as plt
from sotodlib import core, tod_ops

def plot_hwpss_fit_status(tod, hwpss_stats, plot_dets=None, plot_num_dets=3,
                         save_plot=False, save_path='./', save_name='hwpss_stats.png'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    if plot_dets is None:
        plot_step = hwpss_stats.dets.count/(plot_num_dets)
        plot_dets_idx = np.arange(0, hwpss_stats.dets.count, plot_step).astype(int)
        plot_dets = hwpss_stats.dets.vals[plot_dets_idx]
    else:
        plot_dets_idx = np.where(np.in1d(hwpss_stats.dets.vals, plot_dets))[0]

    for i, det_idx in enumerate(plot_dets_idx):
        ax[0].plot(hwpss_stats.binned_angle, hwpss_stats.binned_signal[det_idx], 
                alpha=0.8, color='tab:blue', label='binned signal' if i ==0 else None)

        modes = [int(mode_name[1:]) for mode_name in list(hwpss_stats.modes.vals[::2])]
        ax[0].plot(hwpss_stats.binned_angle, hwpss_stats.binned_model[det_idx], 
                alpha=0.8, color='tab:orange', label=f'binned model \n(modes = {modes})' if i ==0 else None)

    ax[0].legend()
    ax[0].set_xlabel('HWP angle [rad]')
    ax[0].set_title(f'random {plot_num_dets} detectors')

    ax[1].hist(hwpss_stats.redchi2s, bins=np.logspace(start=-1, stop=2, num=50))
    ax[1].axvline(x=np.nanmedian(hwpss_stats.redchi2s), linestyle='dashed', color='black',
                 label=f'median: {np.nanmedian(hwpss_stats.redchi2s):.2f}')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title(f'reduced chi2s distribution (Ndets={hwpss_stats.dets.count})')
    ax[1].legend()

    plt.suptitle(f'HWPSS Stats for Obs Timestamp: {tod.obs_info.timestamp:.0f}, dT = {np.ptp(tod.timestamps)/60:.1f} min', 
                     fontsize = 15)
    save_ts = str(int(time.time()))
    plt.subplots_adjust(top=0.85, bottom=0.2)
    if save_plot:
        plt.savefig(os.path.join(save_path, save_ts+'_'+save_name))
    return fig, ax, frac_samp_glitches, interval_glitches
    
    
    