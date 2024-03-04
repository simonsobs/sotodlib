import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venn import venn

from sotodlib import hwp
import sotodlib.core as core
from sotodlib.core.flagman import has_all_cut

def plot_det_bias_flags(aman, msk, msks, rfrac_range=(0.1, 0.7),
                        psat_range=(0, 15), save_path="./", save_name="bias_cuts_venn.png"):
    """
    Function for plotting bias cuts.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    msk : RangesMatrix
        Result of flags.get_det_bias_flags
    msks : list of RangesMatrix
        Result of flags.get_det_bias_flags when full_output=True
    rfrac_range : Tuple
        Tuple (lower_bound, upper_bound) for rfrac det selection.
    psat_range : Tuple
        Tuple (lower_bound, upper_bound) for P_SAT from IV analysis.
        P_SAT in the IV analysis is the bias power at 90% Rn in pW.
    save_path : str
        Path to plot output directory.
    save_name : str
        Filename of plot.
    """
    all_bad_dets = has_all_cut(msk)
    msk_ids = []
    for msk in msks[:2]:
        bad_dets = has_all_cut(msk)
        msk_ids.append(np.where(bad_dets == True)[0])

    for i in range(0, 3, 2):
        bad_dets1 = has_all_cut(msks[i+2])
        bad_dets2 = has_all_cut(msks[i+3])
        booleans = [bad_dets1, bad_dets2]
        union = np.asarray(list(map(any, zip(*booleans))))
        msk_ids.append(np.where(union == True)[0])

    msk_dict = {'bg < 0': set(msk_ids[0]),
                'r_tes <= 0': set(msk_ids[1]),
                f'r_frac < {rfrac_range[0]} or > {rfrac_range[1]}': set(msk_ids[2]),
                f'p_sat*1e12 < {psat_range[0]} or > {psat_range[1]}': set(msk_ids[3])}

    venn(msk_dict)

    obs_ts = aman.timestamps[0]
    det = aman.dets.vals[0]
    plt.title(f"Obs_timestamp:{obs_ts:.0f}\ndet:{det}\nDetectors Cut per Range (Total cut: {len(np.where(all_bad_dets == True)[0])}/{len(aman.dets.vals)})")
    plt.tight_layout()
    plot_dir = os.path.join(save_path, f'{str(aman.timestamps[0])[:5]}', aman.obs_info.obs_id)
    os.makedirs(plot_dir, exist_ok=True)
    ufm = det.split('_')[2]
    plt.savefig(os.path.join(plot_dir, ufm+'_'+save_name))
    plt.close()

def plot_4f_2f_counts(aman, modes=np.arange(1,49), save_path='./', save_name='4f_2f_counts.png'):
    """
    Function for plotting 4f/2f counts for each bandpass.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    modes : list of int
        The HWPSS harmonic modes to extract.
    save_path : str
        Path to plot output directory.
    save_name : str
        Filename of plot.
    """
    hwpss_ratsatp1 = {}
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    for i, band in enumerate(['f090', 'f150']):
        hwpss_ratsatp1[band] = {}
        m90s = ((aman.det_cal.bg == 0) | (aman.det_cal.bg == 1) | (aman.det_cal.bg == 4) | (aman.det_cal.bg == 5) | (aman.det_cal.bg == 8) | (aman.det_cal.bg == 9))
        m150s = ((aman.det_cal.bg == 2) | (aman.det_cal.bg == 3) | (aman.det_cal.bg == 6) | (aman.det_cal.bg == 7) | (aman.det_cal.bg == 10) | (aman.det_cal.bg == 11))
        m = [m90s, m150s]
        hwpss_aman = aman.restrict('dets', aman.dets.vals[m[i]], in_place=False)
        hwpss_aman.restrict('samps',(20*60*200, -100))
        hwp.hwp.get_hwpss(hwpss_aman, modes=modes)
        a_4f = np.sqrt(hwpss_aman.hwpss_stats.coeffs[:,6]**2 + hwpss_aman.hwpss_stats.coeffs[:,7]**2)
        a_2f = np.sqrt(hwpss_aman.hwpss_stats.coeffs[:,2]**2 + hwpss_aman.hwpss_stats.coeffs[:,3]**2)
        hwpss_ratsatp1[band] = a_4f/a_2f

        hist, bins = np.histogram(hwpss_ratsatp1[band], bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        _ = axs[0, i].hist(hwpss_ratsatp1[band], bins = logbins,
                           color='C0', alpha=0.75)
        axs[0, i].axvline(np.median(hwpss_ratsatp1[band]), color = 'C0', ls = ':',
                          label = f"median: {np.median(hwpss_ratsatp1[band]):.2e}")
        axs[0, i].legend()
        axs[0, i].set_xscale('log')
        axs[0, i].set_title(f'Band {band}')
        axs[0, i].set_xlabel('A_4f/A_2f')
        
        hist, bins = np.histogram(a_4f, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        axs[1, i].hist(a_4f, bins = logbins,
                       color='C0', alpha=0.75)
        axs[1, i].axvline(np.median(a_4f), color = 'C0', ls = ':',
                          label = f"median: {np.median(a_4f):.2e}")
        axs[1, i].legend()
        axs[1, i].set_xscale('log')
        axs[1, i].set_xlabel('A_4f')
        
        hist, bins = np.histogram(a_2f, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        axs[2, i].hist(a_2f, bins = logbins,
                       color='C0', alpha=0.75)
        axs[2, i].axvline(np.median(a_2f), color = 'C0', ls = ':',
                          label = f"median: {np.median(a_2f):.2e}")
        axs[2, i].legend()
        axs[2, i].set_xscale('log')
        axs[2, i].set_xlabel('A_2f')
        
    for i in range(3):
        axs[i, 0].set_ylabel('Counts')

    obs_ts = aman.timestamps[0]
    det = aman.dets.vals[0]
    plt.suptitle(f'Obs_timestamp:{obs_ts:.0f}\ndet:{det}\n4f/2f Counts')
    plt.tight_layout()
    plot_dir = os.path.join(save_path, f'{str(aman.timestamps[0])[:5]}', aman.obs_info.obs_id)
    os.makedirs(plot_dir, exist_ok=True)
    ufm = det.split('_')[2]
    plt.savefig(os.path.join(plot_dir, ufm+'_'+save_name))

def plot_hwpss_fit_status(aman, hwpss_stats, plot_dets=None, plot_num_dets=3,
                          save_path='./', save_name='hwpss_stats.png'):
    """
    Function for plotting HWPSS fit status.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    hwpss_stats : AxisManager.hwpss_stats
        The HWPSS stats output.
    plot_dets : list
        List of dets to plot
    plot_num_dets : list
        Number of dets to plot.
    save_path : str
        Path to plot output directory.
    save_name : str
        Filename of plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    if plot_dets is None:
        plot_step = hwpss_stats.dets.count/(plot_num_dets)
        plot_dets_idx = np.arange(0, hwpss_stats.dets.count, plot_step).astype(int)
        plot_dets = hwpss_stats.dets.vals[plot_dets_idx]
    else:
        plot_dets_idx = np.where(np.in1d(hwpss_stats.dets.vals, plot_dets))[0]

    for i, det_idx in enumerate(plot_dets_idx):
        ax[0].errorbar(hwpss_stats.binned_angle, hwpss_stats.binned_signal[det_idx], yerr=hwpss_stats.sigma_bin[det_idx],
                       alpha=0.5, color='tab:blue', fmt='.-', capsize=2, zorder=2, label='binned signal' if i ==0 else None)
        ax[0].plot(aman.hwp_angle[:2000], aman.signal[det_idx][:2000]-np.median(aman.signal[det_idx][:2000]),
                   alpha=0.5, color='tab:red', marker='o', markersize=0.5, linestyle='None', zorder=1,
                   label='unbinned signal (2000 samps)' if i ==0 else None)

        modes = [int(mode_name[1:]) for mode_name in list(hwpss_stats.modes.vals[::2])]
        ax[0].plot(hwpss_stats.binned_angle, hwpss_stats.binned_model[det_idx], 
                   alpha=0.9, color='tab:orange', zorder=3, label=f'binned model \n(modes = {modes})' if i ==0 else None)

    ax[0].legend()
    ax[0].set_xlabel('HWP angle [rad]')
    ax[0].set_ylabel('Signal [Readout Radians]')
    ax[0].set_title(f'random {plot_num_dets} detectors')

    ax[1].hist(hwpss_stats.redchi2s, bins=np.logspace(start=-1, stop=2, num=50))
    ax[1].axvline(x=np.nanmedian(hwpss_stats.redchi2s), linestyle='dashed', color='black',
                  label=f'median: {np.nanmedian(hwpss_stats.redchi2s):.2f}')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title(f'reduced chi2s distribution (Ndets={hwpss_stats.dets.count})')
    ax[1].legend()

    obs_ts = aman.timestamps[0]
    det = aman.dets.vals[0]
    plt.suptitle(f'HWPSS Stats for Obs_timestamp:{obs_ts:.0f}, dT = {np.ptp(aman.timestamps)/60:.1f} min\ndet:{det}\n')
    plt.subplots_adjust(top=0.85, bottom=0.2)
    plot_dir = os.path.join(save_path, f'{str(aman.timestamps[0])[:5]}', aman.obs_info.obs_id)
    os.makedirs(plot_dir, exist_ok=True)
    ufm = det.split('_')[2]
    plt.savefig(os.path.join(plot_dir, ufm+'_'+save_name))

def plot_sso(aman, sso, xi_p, eta_p, wafer_pointing=None, save_path='./', save_name='sso_footprint.png'):
    
    if wafer_pointing is None:
        wafer_pointing = {'ws0': (-2.5, -0.5),
                          'ws1': (-2.5, -13),
                          'ws2': (-13, -7),
                          'ws3': (-13, 5),
                          'ws4': (-2.5, 11.5),
                          'ws5': (8.5, 5),
                          'ws6': (8.5, -7)}
        
    # Get default focal plane from sotodlib
    hw = np.load('/so/home/msilvafe/shared_files/sat_hw_positions.npz')
    xi_hw, eta_hw, dets_hw = hw['xi_hw'], hw['eta_hw'], hw['dets_hw']
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    ax.scatter(np.degrees(xi_hw), np.degrees(eta_hw), c='gray')
    ax.set_xlabel('$\\xi$ [degrees]')
    ax.set_ylabel('$\\eta$ [degrees]')
    ax.plot(np.degrees(xi_p), np.degrees(eta_p), color='C0', alpha=0.8)
    for k in wafer_pointing.keys():
        ax.text(wafer_pointing[k][0], wafer_pointing[k][1], k, fontsize=16)

    plt.suptitle(f'{sso} {aman.obs_info.obs_id}')
    plt.tight_layout()
    plot_dir = os.path.join(save_path, f'{str(aman.timestamps[0])[:5]}', aman.obs_info.obs_id)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, sso+'_'+save_name))