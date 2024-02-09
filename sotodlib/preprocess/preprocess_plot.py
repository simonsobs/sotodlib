import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import sotodlib.core as core

def plot_det_bias_flags(aman, msk, save_path="./", save_name="bias_cuts.png"):
    """
    Function for plotting bias cuts.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    msk : RangesMatrix
        Result of flags.get_det_bias_flags
    save_path : str
        Path to plot output directory.
    save_name : str
        Filename of plot.
    """
    save_ts = str(int(time.time()))
    bad_dets = core.flagman.has_all_cut(msk)
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    obs_ts = aman.timestamps[0]
    det = aman.dets.vals[0]
    if len(np.where(bad_dets == True)[0]) >= 40:
        ax.plot(aman.timestamps[::100], aman.signal[~bad_dets][::20,::100].T, color = 'C0', alpha = 0.5)
        ax.set_title(f'Obs_timestamp:{obs_ts:.0f}\ndet:{det}\nEvery 20th Detector and 100th Sample\nAfter Detector Bias Cuts')
    else:
        ax.plot(aman.timestamps[::100], aman.signal[~bad_dets][:,::100].T, color = 'C0', alpha = 0.5)
        ax.set_title(f'Obs_timestamp:{obs_ts:.0f}\ndet:{det}\nEvery 100th Sample After Detector Bias Cuts')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Signal [Readout Radians]')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_ts + '_' + save_name))
    plt.close(fig)