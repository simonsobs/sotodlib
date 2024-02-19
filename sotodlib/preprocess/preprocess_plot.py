import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import sotodlib.core as core

def plot_det_bias_flags(aman, msks, rfrac_range=(0.1, 0.7),
                      psat_range=(0, 15), save_path="./", save_name="bias_cuts.png"):
    """
    Function for plotting bias cuts.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
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
    save_ts = str(int(time.time()))
    fig, axs = plt.subplots(1, 6, figsize=(5*6, 5*1))
    obs_ts = aman.timestamps[0]
    det = aman.dets.vals[0]
    ranges = ['bg >= 0',
              'r_tes > 0',
              f'r_frac >= {rfrac_range[0]}',
              f'r_frac <= {rfrac_range[1]}',
              f'p_sat*1e12 >= {psat_range[0]}',
              f'p_sat*1e12 <= {psat_range[1]}']
    for i, msk in enumerate(msks):
        bad_dets = has_all_cut(msk)
        if len(np.where(bad_dets == True)[0]) >= 40:
            axs[i].plot(aman.timestamps[::100], aman.signal[bad_dets][::20,::100].T, color = 'C0', alpha = 0.5)
        else:
            axs[i].plot(aman.timestamps[::100], aman.signal[bad_dets][:,::100].T, color = 'C0', alpha = 0.5)
        axs[i].set_title(f'{ranges[i]}')
        axs[i].set_xlabel('Timestamp')
    axs[0].set_ylabel('Signal [Readout Radians]')
    plt.suptitle(f'Obs_timestamp:{obs_ts:.0f}\ndet:{det}\nEvery 100th Sample After Detector Bias Cuts\n')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_ts + '_' + save_name))
    plt.close(fig)
