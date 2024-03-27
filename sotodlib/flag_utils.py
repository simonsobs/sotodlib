import inspect
import os
import time
from typing import Any, List, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from so3g.proj import RangesMatrix

from sotodlib.core import AxisManager


def plot_glitch_stats(
    tod,
    flags="flags",
    glitches="glitches",
    N_bins=30,
    save_path="./",
    save_name="glitch_flag_stats.png",
    save_plot=True,
):
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes.flatten()
    frac_samp_glitches = (
        100 * np.asarray(tod[flags][glitches].get_stats()["samples"]) / tod.samps.count
    )
    glitchlog = np.log10(frac_samp_glitches[frac_samp_glitches > 0])
    binmin = int(np.floor(np.min(glitchlog)))
    binmax = int(np.ceil(np.max(glitchlog)))
    _ = ax[0].hist(
        frac_samp_glitches, bins=np.logspace(binmin, binmax, N_bins), label="_nolegend_"
    )
    medsamps = np.median(frac_samp_glitches)
    ax[0].axvline(medsamps, color="C1", ls=":", lw=2, label=f"Median: {medsamps:.2f}%")
    meansamps = np.mean(frac_samp_glitches)
    ax[0].axvline(meansamps, color="C2", ls=":", lw=2, label=f"Mean: {meansamps:.2f}%")
    modesamps = stats.mode(frac_samp_glitches, keepdims=True)
    ax[0].axvline(
        modesamps[0][0],
        color="C3",
        ls=":",
        lw=2,
        label=f"Mode: {modesamps[0][0]:.2f}%, Counts: {modesamps[1][0]}",
    )
    stdsamps = np.std(frac_samp_glitches)
    ax[0].axvspan(
        meansamps - stdsamps,
        meansamps + stdsamps,
        color="wheat",
        alpha=0.2,
        label=f"$\sigma$: {stdsamps:.2f}%",
    )
    ax[0].legend()
    ax[0].set_xlim(10**binmin, 10**binmax)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Fraction of Samples Flagged\nper Detector [%]", fontsize=16)
    ax[0].set_ylabel("Counts", fontsize=16)
    ax[0].set_title(
        "Samples Flagged Stats\n$N_{\mathrm{dets}}$ = "
        + f"{tod.dets.count}"
        + " and $N_{\mathrm{samps}}$ = "
        + f"{tod.samps.count}",
        fontsize=18,
    )

    interval_glitches = np.asarray(tod[flags][glitches].get_stats()["intervals"])
    binlinmax = np.nanmax(interval_glitches)
    _ = ax[1].hist(interval_glitches, bins=np.linspace(0, binlinmax, N_bins))
    medints = np.median(interval_glitches)
    ax[1].axvline(
        medints, color="C1", ls=":", lw=2, label=f"Median: {medints:.2f} intervals"
    )
    meanints = np.mean(interval_glitches)
    ax[1].axvline(
        meanints, color="C2", ls=":", lw=2, label=f"Mean: {meanints:.2f} intervals"
    )
    modeints = stats.mode(interval_glitches, keepdims=True)
    ax[1].axvline(
        modeints[0][0],
        color="C3",
        ls=":",
        lw=2,
        label=f"Mode: {modeints[0][0]:.2f} intervals, Counts: {modeints[1][0]}",
    )
    stdints = np.std(interval_glitches)
    ax[1].axvspan(
        meanints - stdints,
        meanints + stdints,
        color="wheat",
        alpha=0.2,
        label=f"$\sigma$: {stdints:.2f} intervals",
    )

    ax[1].legend()
    ax[1].set_xlabel("Number of Flag Intervals\nper Detector", fontsize=16)
    ax[1].set_ylabel("Counts", fontsize=16)
    ax[1].set_title(
        "Ranges Flag Manager Stats\n$N_{\mathrm{dets}}$ with $\geq$ 1 interval = "
        + f"{len(interval_glitches[interval_glitches > 0])}/{tod.dets.count}",
        fontsize=18,
    )
    plt.suptitle(
        f"Glitch Stats for Obs Timestamp: {tod.obs_info.timestamp}, dT = {np.ptp(tod.timestamps)/60:.2f} min",
        fontsize=18,
    )
    save_ts = str(int(time.time()))
    plt.subplots_adjust(top=0.75, bottom=0.2)
    if save_plot:
        plt.savefig(os.path.join(save_path, save_ts + "_" + save_name))
    return fig, ax, frac_samp_glitches, interval_glitches


class FlagFunc(Protocol):
    def __call__(
        self,
        aman: AxisManager,
        *,
        signal: Optional[NDArray[np.floating]] = ...,
        merge: bool = ...,
        name=...,
        overwrite: bool = ...,
    ):
        ...


def flag_sliding_window(
    aman: AxisManager,
    flag_func: FlagFunc,
    signal: Optional[NDArray[np.floating]] = None,
    window_size: int = 10000,
    overlap: int = 1000,
    **kwargs,
) -> Tuple[RangesMatrix, List[Any]]:
    """
    Run a flag through a sliding window.
    Currently written only for flags that run on signal data.

    Arguments:

        aman: The AxisManager to pass to flagging function.

        flag_func: The flagging function.
                   First argument should be 'aman' and should also contain kwargs:
                   'signal', 'merge', 'name', 'overwrite'.

        signal: Data to flag, if None aman.signal is used.

        window_size: Size of window to use.

        overlap: Overlap between adjacent windows.

        **kwargs: kwargs to pass to flag_func.

    Returns:

        flag_ranges: RangesMatrix of the flag generated by flag_func.

        other: Any other outputs generated by flag_func,
               This is a list where each element is the outputs for an iteration of a window.
    """
    if signal is None:
        signal = aman.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal is not an array")

    merge_arg_names = ["merge", "overwrite", "name"]
    sig = inspect.signature(flag_func)
    merge_args = {ma: sig.parameters[ma].default for ma in merge_arg_names}
    merge_args.update(kwargs)
    merge_args = {ma: merge_args[ma] for ma in merge_arg_names}
    kwargs["merge"] = False

    other = []
    flag = np.zeros(signal.shape, dtype=bool)
    tot = flag.shape[-1]
    # Could easily be parallelized here
    # numba, jax, concurrent.futures, or joblib all plug in well
    for i in range(tot // (window_size - overlap)):
        start = i * (window_size - overlap)
        end = np.min((start + window_size, tot))
        out = flag_func(
            aman,
            signal=signal[..., start:end],
            **kwargs,
        )
        if isinstance(out, tuple):
            _flag = out[0]
            other.append(out[1:])
        else:
            _flag = out
            other.append(None)
        flag[..., start:end] += _flag.mask()
    flag_ranges = RangesMatrix.from_bitmask(flag)

    if merge_args["merge"]:
        _merge(aman, flag_ranges, merge_args["name"], merge_args["overwrite"])

    return flag_ranges, other


def _merge(tod: AxisManager, flag_ranges: RangesMatrix, name: str, overwrite: bool):
    if not isinstance(tod, AxisManager):
        print("TOD is not an AxisManager, not merging")
        return
    elif "dets" not in tod or "samps" not in tod:
        print("dets or samps axis not in TOD, not merging")
        return
    elif flag_ranges.shape != (tod.dets.count, tod.samps.count):
        print("Shape of flag does not match that of TOD, not merging")
        return
    if name in tod.flags._fields:
        if overwrite:
            tod.flags.move(name, None)
        else:
            print("Flag already exists and overwrite is False")
    if "flags" not in tod._fields:
        flags = AxisManager(tod.dets, tod.samps)
        tod.wrap("flags", flags)
    tod.flags.wrap(name, flag_ranges, [(0, "dets"), (1, "samps")])
