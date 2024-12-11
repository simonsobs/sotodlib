import numpy as np
import scipy.stats as stats
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venn import venn

from sotodlib import hwp
import sotodlib.core as core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts, sparse_to_ranges_matrix

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

def plot_pcabounds(aman, pca_aman, filename='./pca.png', signal=None, band=None, plot_ds_factor=20):
    """Subplot of pca bounds as well as the good and bad detector
    timestreams with 0th mode weight overplotted

    Parameters
    ----------
    aman : AxisManager
        input AxisManager
    pca_aman : AxisManager
        Relcal output AxisManager
    filename : str
        Full filename with direct path to plot output directory.
    signal : str or ndarray
        Signal name or signal array in aman. ``aman.signal`` is used if not provided.
    band : str
        Bandpass name. Assumes no bandpass separation if not provided.
    plot_ds_factor : int
        Factor to downsample signal plots. Default is 20.
    
    """
    if signal is None:
        signal = aman['signal']
    elif isinstance(signal, str):
        signal = aman[signal]
    elif isinstance(signal, np.ndarray):
        pass
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if band is None:
        xbounds = pca_aman.xbounds
        ybounds = pca_aman.ybounds
        modes = pca_aman.pca_mode0
    else:
        xbounds = pca_aman[f'{band}_xbounds']
        ybounds = pca_aman[f'{band}_ybounds']
        modes = pca_aman[f'{band}_pca_mode0']

    pca_dets = pca_aman.pca_det_mask
    good_indices = np.where(~pca_dets)[0]
    bad_indices = np.where(pca_dets)[0]

    fig = plt.figure(figsize=(10, 6))

    # Define axes
    ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    
    # ax1: good signals
    ax1.plot(aman.timestamps, modes, color='black', linewidth=3,
             label='0th mode', zorder=2, alpha=0.4)

    ax1.plot(aman.timestamps[::plot_ds_factor], np.divide(aman.signal[good_indices][:,::plot_ds_factor].T, pca_aman.pca_weight0[good_indices]), zorder=1, color='#D8BFD8', alpha=0.3)

    ax1.set_title(f'Good Detector Batch: ({len(good_indices)} dets)')
    ax1.legend(loc='upper left')
    ax1.grid()

    # ax2: bad signals
    ax2.plot(aman.timestamps, modes, color='black', linewidth=3,
             label='0th mode', zorder=2, alpha=0.4)

    ax2.plot(aman.timestamps[::plot_ds_factor], np.divide(aman.signal[bad_indices][:,::plot_ds_factor].T, pca_aman.pca_weight0[bad_indices]), zorder=1, color='#FFA07A', alpha=0.3)
        
    ax2.set_title(f'Bad Detector Batch: ({len(bad_indices)} dets)')
    ax2.legend(loc='upper left')
    ax2.grid()

    # ax3: box
    weight = np.abs(pca_aman.pca_weight0)
    ax3.plot(aman.det_cal.s_i[good_indices], weight[good_indices], '.', color='#D8BFD8', markersize=10,
             label=f'Good dets ({len(good_indices)} dets)', alpha=0.3)

    ax3.plot(aman.det_cal.s_i[bad_indices], weight[bad_indices], '.', color='#FFA07A', markersize=10,
             label=f'Bad dets ({len(bad_indices)} dets)', alpha=0.3)

    vertices = [(xbounds[0], ybounds[0]), (xbounds[1], ybounds[0]), (xbounds[1], ybounds[1]), (xbounds[0], ybounds[1])]
    box = matplotlib.patches.Polygon(vertices, closed=True, edgecolor='navy', linestyle='-.', fill=False, alpha=1, label='Boundary')
    ax3.add_patch(box)

    ax3.set_xlabel('Si')
    ax3.set_ylabel('0th Mode Weights')

    ax3.legend()
    ax3.grid()

    plt.suptitle(f'{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n{band}')
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_trending_flags(aman, trend_aman, filename='./trending_flags.png'):
    """
    Function for plotting trending flags.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    trend_aman : AxisManager
        Output trend_aman of tod_ops.flags.get_trending_flags with full_output=True.
    filename : str
        Full filename with direct path to plot output directory.
    """
    fig, axs = plt.subplots(1, 2, figsize=(5*2, 5*1))

    tdets = has_any_cuts(trend_aman.trend_flags)
    
    keep = np.abs(trend_aman.trends[~tdets, :]).T
    for i in range(keep.shape[1]):
        if i == 0:
            axs[0].plot(aman.timestamps[trend_aman.samp_start], keep[:, i], color = 'C0', 
                        alpha=0.75, marker='o', markersize=0.75, linestyle='None', label='Keep')
        else:
            axs[0].plot(aman.timestamps[trend_aman.samp_start], keep[:, i], color = 'C0', 
                        alpha=0.75, marker='o', markersize=0.75, linestyle='None')
    flagged = np.abs(trend_aman.trends[tdets, :]).T
    for i in range(flagged.shape[1]):
        if i == 0:
            axs[0].plot(aman.timestamps[trend_aman.samp_start], flagged[:, i], color = 'C1', 
                        marker='o', markersize=1.5, linestyle='None', label='Flagged')
        else:
            axs[0].plot(aman.timestamps[trend_aman.samp_start], flagged[:, i], color = 'C1', 
                        marker='o', markersize=1.5, linestyle='None')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Timestamp')
    axs[0].set_ylabel('Trend Slope')
    axs[0].set_title('Trending Slopes of Dets')
    axs[0].legend()

    axs[1].plot(aman.timestamps[::100], aman.signal[tdets][:,::100].T, color = 'C0', alpha = 0.5)
    axs[1].set_xlabel('Timestamp')
    axs[1].set_ylabel('Signal [Readout Radians]')
    axs[1].set_title(f'Trending Channels ({len(np.where(tdets == True)[0])} dets)')

    plt.suptitle(f"{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\nTrending Flags (Total cut: {len(np.where(tdets == True)[0])}/{len(aman.dets.vals)})")
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_signal(aman, signal=None, xx=None, signal_name="signal", x_name="timestamps", plot_ds_factor=50, plot_ds_factor_dets=None, xlim=None, alpha=0.2, yscale='linear', y_unit=None, filename="./signal.png"):
    from operator import attrgetter
    if plot_ds_factor_dets is None:
        plot_ds_factor_dets = plot_ds_factor
    if signal is None:
        signal = attrgetter(signal_name)(aman)
    if xx is None:
        xx = attrgetter(x_name)(aman)
    yy = signal[::plot_ds_factor_dets, 1::plot_ds_factor].copy() # (dets, samps); (dets, nusamps); (dets, nusamps, subscans)
    xx = xx[1::plot_ds_factor].copy() # (samps); (nusamps)
    if x_name == "timestamps":
        xx -= xx[0]
    if yy.ndim > 2: # Flatten subscan axis into dets
        yy = yy.swapaxes(1,2).reshape(-1, yy.shape[1])

    if xlim is not None:
        xinds = np.logical_and(xx >= xlim[0], xx <= xlim[1])
        xx = xx[xinds]
        yy = yy[:,xinds]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.plot(xx, yy.T, color='k', alpha=0.2)
    ax.set_yscale(yscale)
    if "freqs" in x_name:
        ax.set_xlabel("freq [Hz]")
    else:
        ax.set_xlabel(f"{x_name} [s]")
    y_unit = "" if y_unit is None else f" [{y_unit}]"
    ax.set_ylabel(f"{signal_name.replace('.Pxx', '')}{y_unit}")
    plt.suptitle(f"{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min")
    plt.tight_layout()
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_psd(aman, signal=None, xx=None, signal_name="psd.Pxx", x_name="psd.freqs", plot_ds_factor=4, plot_ds_factor_dets=20, xlim=None, alpha=0.2, yscale='log', y_unit=None, filename="./psd.png"):
    return plot_signal(aman, signal, xx, signal_name, x_name, plot_ds_factor, plot_ds_factor_dets, xlim, alpha, yscale, y_unit, filename)

def plot_signal_diff(aman, flag_aman, flag_type="glitches", flag_threshold=10, plot_ds_factor=50, filename="./glitch_signal_diff.png"):
    """
    Function for plotting the difference in signal before and after cuts from either glitches or jumps.
    
    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    flag_aman : AxisManager
        Output jump_aman of tod_ops.jumps.jumps_aman or
        glitch_aman of tod_ops.flags.get_glitch_flags with full_output=True.
    flag_type : str
        Flag type to plot. Options: ["glitches, "jumps"]. Default is "glitches"
    flag_threshold : int
        Threshold to cut dets. Equivalent to "max_n_glitch" and "max_n_jumps".
    plot_ds_factor : int
        Factor to downsample signal plots. Default is 50.
    filename : str
        Full filename with direct path to plot output directory.
    """
    if flag_type == "glitches":
        flags = flag_aman.glitch_flags
        plot_name = f"Glitch Flags Signal Diff (Every {plot_ds_factor} dets)"
    elif flag_type == "jumps":
        flags = flag_aman.jump_flag
        plot_name = f"Jump Flags Signal Diff (Every {plot_ds_factor} dets)"
    else:
        raise ValueError("Flag type not recognized. Must be 'glitches' or 'jumps'")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes.flatten()
    
    ax[0].plot(aman.timestamps[::plot_ds_factor], aman.signal[:,::plot_ds_factor].T, color='tab:blue', alpha=0.3)
    ax[0].set_xlabel("Timestamp")
    ax[0].set_ylabel("Signal [Readout Radians]")
    ax[0].set_title("Before")
    
    if flag_type == "glitches":
        flags = sparse_to_ranges_matrix(flag_aman.glitch_detection > flag_threshold)
    n_cut = count_cuts(flags)
    keep = n_cut <= flag_threshold
    flags_removed = aman.restrict("dets", aman.dets.vals[keep], in_place=False)

    ax[1].plot(flags_removed.timestamps[::plot_ds_factor], flags_removed.signal[:,::plot_ds_factor].T, color='tab:blue', alpha=0.3)
    ax[1].set_xlabel("Timestamp")
    ax[1].set_ylabel("Signal [Readout Radians]")
    ax[1].set_title("After")
    
    plt.suptitle(f"{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n{plot_name} (Total cut: {len(np.where(~keep)[0])}/{len(aman.dets.vals)})")
    plt.subplots_adjust(top=0.70, bottom=0.15)
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)

def plot_flag_stats(aman, flag_aman, flag_type="glitches", N_bins=30, filename="./glitch_stats.png"):
    """
    Function for plotting the glitches or jumps flags/cut statistics using the built in stats functions
    in the RangesMatrices class.
    Args:
    -----
    aman : AxisManager
        Input axis manager.
    flag_aman : AxisManager
        Output jump_aman of tod_ops.jumps.jumps_aman or
        glitch_aman of tod_ops.flags.get_glitch_flags with full_output=True.
    flag_type : str
        Flag type to plot. Options: ["glitches, "jumps"]. Default is "glitches"
    N_bins (int): Number of bins in the histogram.
    filename : str
        Full filename with direct path to plot output directory.
    """
    if flag_type == "glitches":
        flags = flag_aman.glitch_flags
        plot_name = "Glitch Stats"
    elif flag_type == "jumps":
        flags = flag_aman.jump_flag
        plot_name = "Jumps Stats"
    else:
        raise ValueError("Flag type not recognized. Must be 'glitches' or 'jumps'")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes.flatten()
    frac_samp_glitches = (
        100 * np.asarray(flags.get_stats()["samples"]) / aman.samps.count
    )
    glitchlog = np.log10(frac_samp_glitches[frac_samp_glitches > 0])
    if glitchlog.size == 0:
        raise ValueError("No flags found (empty array).")

    binmin = int(np.floor(np.min(glitchlog)))
    binmax = int(np.ceil(np.max(glitchlog)))
    _ = ax[0].hist(
        frac_samp_glitches, bins=np.logspace(binmin, binmax, N_bins), label="_nolegend_"
    )
    medsamps = np.median(frac_samp_glitches)
    ax[0].axvline(medsamps, color="C1", ls=":", lw=2, label=f"Median: {medsamps:.2e}%")
    meansamps = np.mean(frac_samp_glitches)
    ax[0].axvline(meansamps, color="C2", ls=":", lw=2, label=f"Mean: {meansamps:.2e}%")
    modesamps = stats.mode(frac_samp_glitches, keepdims=True)
    ax[0].axvline(
        modesamps[0][0],
        color="C3",
        ls=":",
        lw=2,
        label=f"Mode: {modesamps[0][0]:.2e}%, Counts: {modesamps[1][0]}",
    )
    stdsamps = np.std(frac_samp_glitches)
    ax[0].axvspan(
        meansamps - stdsamps,
        meansamps + stdsamps,
        color="wheat",
        alpha=0.2,
        label=f"$\sigma$: {stdsamps:.2e}%",
    )
    ax[0].legend()
    ax[0].set_xlim(10**binmin, 10**binmax)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Fraction of Samples Flagged\nper Detector [%]", fontsize=16)
    ax[0].set_ylabel("Counts", fontsize=16)
    ax[0].set_title(
        "Samples Flagged Stats\n$N_{\mathrm{dets}}$ = "
        + f"{aman.dets.count}"
        + " and $N_{\mathrm{samps}}$ = "
        + f"{aman.samps.count}"
    )

    interval_glitches = np.asarray(flags.get_stats()["intervals"])
    binlinmax = np.quantile(interval_glitches, 0.98)
    _ = ax[1].hist(np.clip(interval_glitches, 0, binlinmax), bins=np.linspace(0, binlinmax, N_bins))
    medints = np.median(interval_glitches)
    ax[1].axvline(
        medints, color="C1", ls=":", lw=2, label=f"Median: {medints:.2e} intervals"
    )
    meanints = np.mean(interval_glitches)
    ax[1].axvline(
        meanints, color="C2", ls=":", lw=2, label=f"Mean: {meanints:.2e} intervals"
    )
    modeints = stats.mode(interval_glitches, keepdims=True)
    ax[1].axvline(
        modeints[0][0],
        color="C3",
        ls=":",
        lw=2,
        label=f"Mode: {modeints[0][0]:.2e} intervals, Counts: {modeints[1][0]}",
    )
    stdints = np.std(interval_glitches)
    ax[1].axvspan(
        meanints - stdints,
        meanints + stdints,
        color="wheat",
        alpha=0.2,
        label=f"$\sigma$: {stdints:.2e} intervals",
    )

    ax[1].legend()
    ax[1].set_xlim(-1, binlinmax)
    ax[1].set_xlabel("Number of Flag Intervals\nper Detector", fontsize=16)
    ax[1].set_ylabel("Counts", fontsize=16)
    ax[1].set_title(
        "Ranges Flag Manager Stats\n$N_{\mathrm{dets}}$ with $\geq$ 1 interval = "
        + f"{len(interval_glitches[interval_glitches > 0])}/{aman.dets.count}\n(98th quantile bin max)"
    )
    
    plt.suptitle(f"{aman.obs_info.obs_id}, dT = {np.ptp(aman.timestamps)/60:.1f} min\n{plot_name}")
    plt.subplots_adjust(top=0.70, bottom=0.15)
    head_tail = os.path.split(filename)
    os.makedirs(head_tail[0], exist_ok=True)
    plt.savefig(filename)
