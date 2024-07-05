import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venn import venn

from sotodlib import hwp
import sotodlib.core as core
from sotodlib.core.flagman import has_all_cut

def plot_det_bias_flags(aman, det_bias_flags, rfrac_range=(0.1, 0.7),
                        psat_range=(0, 15), filename="./bias_cuts_venn.png"):
    """
    Function for plotting bias cuts.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    det_bias_flags : AxisManager
        Output msk_aman of tod_ops.flags.get_det_bias_flags with full_output=True.
    rfrac_range : Tuple
        Tuple (lower_bound, upper_bound) for rfrac det selection.
    psat_range : Tuple
        Tuple (lower_bound, upper_bound) for P_SAT from IV analysis.
        P_SAT in the IV analysis is the bias power at 90% Rn in pW.
    filename : str
        Full filename with direct path to plot output directory.
    """
    msk_names = ['bg', 'r_tes', 'r_frac_gt', 'r_frac_lt', 'p_sat_gt', 'p_sat_lt']
    msk = det_bias_flags['det_bias_flags']
    msks = []
    for name in msk_names:
        msks.append(det_bias_flags[f'{name}_flags'])
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

    plt.title(f"{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\nDetectors Cut per Range (Total cut: {len(np.where(all_bad_dets == True)[0])}/{len(aman.dets.vals)})")
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_4f_2f_counts(aman, modes=np.arange(1,49), filename='./4f_2f_counts.png'):
    """
    Function for plotting 4f/2f counts for each bandpass.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    modes : list of int
        The HWPSS harmonic modes to extract.
    filename : str
        Full filename with direct path to plot output directory.
    """
    hwpss_ratsatp1 = {}
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    for i, band in enumerate(['f090', 'f150']):
        hwpss_ratsatp1[band] = {}
        m90s = (aman.det_info.wafer.bandpass == 'f090')
        m150s = (aman.det_info.wafer.bandpass == 'f150')
        m = [m90s, m150s]
        hwpss_aman = aman.restrict('dets', aman.dets.vals[m[i]], in_place=False)
        if hwpss_aman.dets.count == 0:
            print(f"No dets in {band} band")
            continue
        stats = hwp.hwp.get_hwpss(hwpss_aman, modes=modes, merge_stats=False, merge_model=False)
        a_4f = np.sqrt(stats.coeffs[:,6]**2 + stats.coeffs[:,7]**2)
        a_2f = np.sqrt(stats.coeffs[:,2]**2 + stats.coeffs[:,3]**2)
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

    plt.suptitle(f'{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n4f/2f Counts')
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_hwpss_fit_status(aman, hwpss_stats, plot_dets=None, plot_num_dets=3,
                          filename='./hwpss_stats.png'):
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
    filename : str
        Full filename with direct path to plot output directory.
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

    plt.suptitle(f'{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\nHWPSS Stats\n')
    plt.subplots_adjust(top=0.85, bottom=0.2)
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_sso_footprint(aman, planet_aman, sso, wafer_offsets=None, focal_plane=None, filename='./sso_footprint.png'):
    """
    Function for plotting SSO footprint.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    planet_aman : AxisManager
        Axis manager of results from obs_ops.sources.get_sso for a single planet.
    sso : str
        Name of planet.
    wafer_offsets : dict
        Dictionary of wafer offsets.
        Ex: wafer_offsets = {'ws0': (-2.5, -0.5),
                             'ws1': (-2.5, -13),
                             'ws2': (-13, -7),
                             'ws3': (-13, 5),
                             'ws4': (-2.5, 11.5),
                             'ws5': (8.5, 5),
                             'ws6': (8.5, -7)}
    focal_plane : str
        Path to focal plane file.
    filename : str
        Full filename with direct path to plot output directory.
    """
    xi_p = planet_aman['xi_p']
    eta_p = planet_aman['eta_p']

    if wafer_offsets is None:
        # Default wafer offsets
        wafer_offsets = {'ws0': (-2.5, -0.5),
                         'ws1': (-2.5, -13),
                         'ws2': (-13, -7),
                         'ws3': (-13, 5),
                         'ws4': (-2.5, 11.5),
                         'ws5': (8.5, 5),
                         'ws6': (8.5, -7)}

    if focal_plane is None:
        print('No focal plane file given.')
        return False
    else:
        hw = np.load(focal_plane)
    xi_hw, eta_hw, dets_hw = hw['xi_hw'], hw['eta_hw'], hw['dets_hw']
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    ax.scatter(np.degrees(xi_hw), np.degrees(eta_hw), c='gray')
    ax.set_xlabel('$\\xi$ [degrees]')
    ax.set_ylabel('$\\eta$ [degrees]')
    ax.plot(np.degrees(xi_p), np.degrees(eta_p), color='C0', alpha=0.8)
    for k in wafer_offsets.keys():
        ax.text(wafer_offsets[k][0], wafer_offsets[k][1], k, fontsize=16)

    plt.suptitle(f'{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n{sso}')
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)


def plot_pcabounds(aman, pca_aman, ghz, filename):
    """Subplot of pca bounds as well as the good and bad detector
    timestreams with 0th mode weight overplotted

    Parameters
    ----------
    amans : AxisManager
        input AxisManager
    pca_aman : AxisManager
        Relcal output AxisManager
    ghz : str
        Bandpass (for plotting purposes)
    filename : str
        Full filename with direct path to plot output directory.
    
    """
    pca_dets = pca_aman.pca_det_mask
    good_indices = np.where(~pca_dets)[0]
    bad_indices = np.where(pca_dets)[0]

    xbounds = pca_aman.xbounds
    ybounds = pca_aman.ybounds

    timestamps = aman.timestamps  # should assume lpf signal though
    modes = pca_aman.pca_mode0

    fig = plt.figure(figsize=(10, 6))

    # Define axes
    ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    
    #print('len of times', len(timestamps), 'len modes', len(modes))
    # ax1: good signals
    ax1.plot(timestamps, modes, color='black', linewidth=3,
             label='0th mode', zorder=2, alpha=0.4)

    for ind in good_indices:
        weight = pca_aman.pca_weight0[ind] #, 0]
        signals = aman.signal[ind]
        ax1.plot(timestamps, signals / weight,
                 zorder=1, color='#D8BFD8', alpha=0.3)

    ax1.set_title(f'Good Detector Batch: ({len(good_indices)} dets)')
    ax1.legend(loc='upper left')
    ax1.grid()

    # ax2: bad signals
    ax2.plot(timestamps, modes, color='black', linewidth=3,
             label='0th mode', zorder=2, alpha=0.4)
    for ind in bad_indices:
        weight = pca_aman.pca_weight0[ind]
        signals = aman.signal[ind]
        ax2.plot(timestamps, signals / weight,
                 zorder=1, color='#FFA07A', alpha=0.3)
    ax2.set_title(f'Bad Detector Batch: ({len(bad_indices)} dets)')
    ax2.legend(loc='upper left')
    ax2.grid()

    # ax3: box
    weight = np.abs(pca_aman.pca_weight0)
    Si = aman.det_cal.s_i
    ax3.plot(Si[good_indices], weight[good_indices], '.', color='#D8BFD8', markersize=10,
             label=f'Good dets ({len(good_indices)} dets)', alpha=0.3)

    ax3.plot(Si[bad_indices], weight[bad_indices], '.', color='#FFA07A', markersize=10,
             label=f'Bad dets ({len(bad_indices)} dets)', alpha=0.3)

    ax3.plot([xbounds[0], xbounds[1], xbounds[1], xbounds[0], xbounds[0]],
             [ybounds[0], ybounds[0], ybounds[1], ybounds[1], ybounds[0]],
             color='navy', linestyle='-.', linewidth=1.5, label='Boundary',
             alpha=1)

    ax3.set_xlabel('Si')
    ax3.set_ylabel('0th Mode Weights')

    if any(value > 0 for value in Si):
        ax3.set_xlim(np.min(Si), 0)

    ax3.legend()
    ax3.grid()

    plt.suptitle(f'{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n{ghz}')
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)
