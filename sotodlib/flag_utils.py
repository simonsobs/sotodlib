import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy import stats

def plot_glitch_stats(tod, glitches=None, N_bins=30, save_path='./',
                      save_name='glitch_flag_stats.png', save_plot=True):
    """
    Function for plotting the glitch flags/cut statistics using the built in stats functions
    in the RangesMatrices class.
    Args:
    -----
    tod (AxisManager): Axis manager with glitch flags in it.
    flags (FlagManager): Name of the FlagManager that holds the glitch flags in tod.
    glitches (RangesMatrix): Name of the axis inside flags that holds the RangesMatrix with
                             the glitches in it.
    N_bins (int): Number of bins in the histogram.
    save_path (path): Path to directory where plot will be saved
    save_name (str): Name of file to save plot to in save_path
    save_plot (bool): Determines if plot is saved.

    Returns:
    --------
    fig (matplotlib.figure.Figure): Figure class
    ax (numpy.ndarray): Numpy array of matplotlib.axes._subplots.AxesSubplot class plot axes.
    frac_samp_glitches (ndarray, float): Array size 1 x N_dets with number of samples flagged
                                         per detector divided by the total number of samples 
                                         converted to percentage. 
    interval_glitches (ndarray, int): Array size 1 x N_dets with number flagged intervals per
                                      detectors.
    """
    if glitches == None:
        glitches = tod.flags.glitches
    fig, axes = plt.subplots(1,2,figsize = (15,5))
    ax = axes.flatten()
    frac_samp_glitches = 100*np.asarray(glitches.get_stats()['samples'])/\
                         tod.samps.count
    glitchlog = np.log10(frac_samp_glitches[frac_samp_glitches > 0])
    binmin = int(np.floor(np.min(glitchlog)))
    binmax = int(np.ceil(np.max(glitchlog)))
    _ = ax[0].hist(frac_samp_glitches, bins = np.logspace(binmin, binmax, N_bins),
                   label = '_nolegend_')
    medsamps = np.median(frac_samp_glitches)
    ax[0].axvline(medsamps,color = 'C1', ls = ':', lw = 2, label=f'Median: {medsamps:.2f}%')
    meansamps = np.mean(frac_samp_glitches)
    ax[0].axvline(meansamps,color = 'C2', ls = ':', lw = 2, label=f'Mean: {meansamps:.2f}%')
    modesamps = stats.mode(frac_samp_glitches)
    ax[0].axvline(modesamps[0][0],color = 'C3', ls = ':', lw = 2, 
                  label=f'Mode: {modesamps[0][0]:.2f}%, Counts: {modesamps[1][0]}')
    stdsamps = np.std(frac_samp_glitches)
    ax[0].axvspan(meansamps-stdsamps, meansamps+stdsamps, color = 'wheat', alpha = 0.2, 
                  label=f'$\sigma$: {stdsamps:.2f}%')
    ax[0].legend()
    ax[0].set_xlim(10**binmin, 10**binmax)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Fraction of Samples Flagged\nper Detector [%]', fontsize = 16)
    ax[0].set_ylabel('Counts', fontsize = 16)
    ax[0].set_title('Samples Flagged Stats\n$N_{\mathrm{dets}}$ = '+f'{tod.dets.count}'+
                    ' and $N_{\mathrm{samps}}$ = '+f'{tod.samps.count}', fontsize = 18)

    interval_glitches = np.asarray(glitches.get_stats()['intervals'])
    binlinmax = np.nanmax(interval_glitches)
    _ = ax[1].hist(interval_glitches, bins = np.linspace(0, binlinmax, N_bins))
    medints = np.median(interval_glitches)
    ax[1].axvline(medints,color = 'C1', ls = ':', lw = 2, 
                  label=f'Median: {medints:.2f} intervals')
    meanints = np.mean(interval_glitches)
    ax[1].axvline(meanints,color = 'C2', ls = ':', lw = 2, 
                  label=f'Mean: {meanints:.2f} intervals')
    modeints = stats.mode(interval_glitches)
    ax[1].axvline(modeints[0][0],color = 'C3', ls = ':', lw = 2,
                  label=f'Mode: {modeints[0][0]:.2f} intervals, Counts: {modeints[1][0]}')
    stdints = np.std(interval_glitches)
    ax[1].axvspan(meanints-stdints, meanints+stdints, color = 'wheat', alpha = 0.2,
                  label=f'$\sigma$: {stdints:.2f} intervals')

    ax[1].legend()
    ax[1].set_xlabel('Number of Flag Intervals\nper Detector', fontsize = 16)
    ax[1].set_ylabel('Counts', fontsize = 16)
    ax[1].set_title('Ranges Flag Manager Stats\n$N_{\mathrm{dets}}$ with $\geq$ 1 interval = '+
                    f'{len(interval_glitches[interval_glitches > 0])}/{tod.dets.count}', 
                    fontsize = 18)
    plt.suptitle(f'Glitch Stats for Obs Timestamp: {tod.obs_info.timestamp}, dT = {np.ptp(tod.timestamps)/60:.2f} min', 
                 fontsize = 18)
    save_ts = str(int(time.time()))
    plt.subplots_adjust(top=0.75, bottom=0.2)
    if save_plot:
        plt.savefig(os.path.join(save_path, save_ts+'_'+save_name))
    return fig, ax, frac_samp_glitches, interval_glitches
