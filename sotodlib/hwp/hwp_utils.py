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
    return fig, ax

def plot_preprocess_PSDs(aman, det=None, psd_before=None, psd_after=None, psd_dsT=None, psd_demodQ=None, psd_demodU=None,
                        take_square_root=True, amplitude_unit='pA',
                        save_plot=False, save_path='./', save_name='preprocess_PSDs.png'):
    if psd_before is None: psd_before = aman.psd
    if psd_after is None: psd_after = aman.psd_hwpss_remove
    if psd_dsT is None: psd_dsT = aman.psd_dsT
    if psd_demodQ is None: psd_demodQ = aman.psd_demodQ
    if psd_demodU is None: psd_demodU = aman.psd_demodU
    
    psd_dict = {
    'before': psd_before,
    'after': psd_after,
    'dsT': psd_dsT,
    'demodQ': psd_demodQ,
    'demodU': psd_demodU,
           }

    if take_square_root:
        power = 0.5
        ylabel = f'PSD [{amplitude_unit}/sqrt(Hz)]'
    else:
        power = 1
        ylabel = f'PSD [{amplitude_unit}^2/Hz]'

    if det is None:
        det = aman.dets.vals[0]
        
    det_idx = np.where(aman.dets.vals == det)[0][0]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, (psd_name, psd) in enumerate(psd_dict.items()):
        ax.loglog(psd.freqs, psd.Pxx[det_idx]**power, label=psd_name, alpha=0.3)
        if i == 0:
            ax.set_ylim(np.nanmin(psd.Pxx[det_idx]**power), np.nanmax(psd.Pxx[det_idx]**power))

    ax.legend()
    ax.set_xlabel('freq [Hz]')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Obs_timestamp:{aman.timestamps[0]:.0f}\ndet:{det}')
    fig.tight_layout()
    
    save_ts = str(int(time.time()))
    if save_plot:
        plt.savefig(os.path.join(save_path, save_ts+'_'+save_name))
    
    return fig, ax
