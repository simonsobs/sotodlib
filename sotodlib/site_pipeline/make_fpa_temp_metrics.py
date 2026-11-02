"""Compute and visualize focal-plane-array (FPA) temperature stability metrics.

Pipeline (see the ``__main__`` block / :func:`main`):

1. :func:`pkl` queries an observation database over a date range and writes
   azimuth-binned FPA temperature data to pickle files (via ``exe_azbin``).
2. :func:`get_fpa_metrics` loads those pickle files and computes RMS,
   standard deviation, deviation from setpoint, peak-to-peak turnaround, and
   Gaussian-fit metrics for each observation.
3. :func:`metric_analyzer` flags observations whose metrics exceed
   platform-specific thresholds, binned by azimuth range.
4. :func:`az_circle_plotter` renders a polar summary plot of the azimuth
   scan range with flag counts overlaid.
"""

import os
import argparse
import traceback
import datetime as dtime
from typing import Optional

import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from tqdm import tqdm

from sotodlib import core, tod_ops
from sotodlib.io import hkdb
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.site_pipeline.utils.logging import init_logger
logger = init_logger('make_fpa_temp_metrics', 'make_fpa_temp_metrics: ')
date = dtime.datetime.now()
DEG = np.pi / 180

def exe_p2p_ana_load(ctx, iobs):
    ws = "ws0"
    if "satp2" in iobs:
        band = "f220"
    else:
        band = "f090"
    for i, iws in zip(iobs.split("_")[-1], ws):
        if i == 0:
            pass
        else:
            # print(i, iws)
            ws = iws
            break
    meta = ctx.get_meta(
        iobs,
        dets={
            "wafer.type": "OPTC",
            "wafer_slot": "ws0",
            "wafer.bandpass": band,
        },
    )
    meta.restrict("dets", meta.det_info.det_id == meta.det_info.det_id[0])
    aman = ctx.get_obs(meta)

    _ = tod_ops.flags.get_turnaround_flags(aman, t_buffer=2, truncate=True)
    if len(aman.timestamps) > 4000 * 2000:
        print("Skip this data due to the data length is > 4000*200.")
        return np.nan, np.nan, np.nan, np.nan
    return aman


def load_az_temp(aman, hkcfg_file, platform="satp1"):
    cfg = hkdb.HkConfig.from_yaml(hkcfg_file)
    if platform == "satp1":
        spec = hkdb.LoadSpec(
            cfg=cfg,
            fields=[
                "acu.acu_udp_stream.Az_Pos_Err",
                "acu.acu_udp_stream.Corrected_Azimuth",
                "acu.acu_udp_stream.Az_Pos_Des",
                "cryo-ls372-lsa21yc.temperatures.Channel_02_T",
            ],
            start=float(aman.timestamps[0]),
            end=float(aman.timestamps[-1]),
        )
        res = hkdb.load_hk(spec, show_pb=True)
        keys = [
            "acu.acu_udp_stream.Az_Pos_Des",
            "acu.acu_udp_stream.Az_Pos_Err",
            "acu.acu_udp_stream.Corrected_Azimuth",
            "cryo-ls372-lsa21yc.temperatures.Channel_02_T",
        ]
        labels = ["Az_Pos_Des", "Az_Pos_Err", "Corrected_Azimuth", "fptemp"]
    if platform == "satp2":
        spec = hkdb.LoadSpec(
            cfg=cfg,
            fields=[
                "acu.acu_udp_stream.Az_Pos_Err",
                "acu.acu_udp_stream.Corrected_Azimuth",
                "acu.acu_udp_stream.Az_Pos_Des",
                "cryo-ls372-lsa22vx.temperatures.Channel_03_T",
            ],
            start=float(aman.timestamps[0]),
            end=float(aman.timestamps[-1]),
        )
        res = hkdb.load_hk(spec, show_pb=True)
        keys = [
            "acu.acu_udp_stream.Az_Pos_Des",
            "acu.acu_udp_stream.Az_Pos_Err",
            "acu.acu_udp_stream.Corrected_Azimuth",
            "cryo-ls372-lsa22vx.temperatures.Channel_03_T",
        ]
        labels = ["Az_Pos_Des", "Az_Pos_Err", "Corrected_Azimuth", "fptemp"]
    if platform == "satp3":
        spec = hkdb.LoadSpec(
            cfg=cfg,
            fields=[
                "acu.acu_udp_stream.Az_Pos_Err",
                "acu.acu_udp_stream.Corrected_Azimuth",
                "acu.acu_udp_stream.Az_Pos_Des",
                "cryo-ls372-lsa2dtc.temperatures.Channel_01_T",
            ],
            start=float(aman.timestamps[0]),
            end=float(aman.timestamps[-1]),
        )
        res = hkdb.load_hk(spec, show_pb=True)
        keys = [
            "acu.acu_udp_stream.Az_Pos_Des",
            "acu.acu_udp_stream.Az_Pos_Err",
            "acu.acu_udp_stream.Corrected_Azimuth",
            "cryo-ls372-lsa2dtc.temperatures.Channel_01_T",
        ]
        labels = ["Az_Pos_Des", "Az_Pos_Err", "Corrected_Azimuth", "fptemp"]
    for ikey, ilabel in zip(keys, labels):
        # ilabel = ikey.split('.')[-1]
        x, y = res.data[ikey]
        f = interp1d(x, y, kind="linear", fill_value="extrapolate")
        interpol = f(aman.timestamps)
        aman.wrap_new(ilabel, dtype="float32", shape=("samps",))
        aman[ilabel] = interpol


def make_binning(sigs, rs, r_bins):
    inds = np.array([np.argmin(np.abs(ir - r_bins)) for ir in rs])
    nbins = len(r_bins)

    binned_count = np.zeros(nbins)
    binned_signal = np.zeros(nbins)
    binned_squared = np.zeros(nbins)

    np.add.at(binned_count, inds, 1)
    np.add.at(binned_signal, inds, sigs)
    np.add.at(binned_squared, inds, sigs**2)

    binned_signal = binned_signal / binned_count
    binned_squared = binned_squared / binned_count
    return {
        "bin": r_bins,
        "sig": binned_signal,
        "sig_sqre": binned_squared,
        "count": binned_count,
    }


def exe_azbin(
    ctx,
    iobs,
    hkcfg_file,
    platform="satp1",
    cut=5,
):

    aman = exe_p2p_ana_load(ctx, iobs)
    load_az_temp(aman, hkcfg_file, platform)
    az_deg = aman.boresight.az / DEG
    azmax = round(np.max(aman.boresight.az / DEG), 1)
    azmin = round(np.min(aman.boresight.az / DEG), 1)
    az_bin = np.arange(azmin, azmax, 0.2)
    rmask_full = aman.flags.right_scan.mask()
    lmask_full = aman.flags.left_scan.mask()

    fp_full = aman.fptemp
    timestamps_full = aman.timestamps

    ###################
    # New way to load data which will make auto script better
    # This will trim the full TOD in two different ways
    # 1. For RMS and Gaussian statistics, only want to look at non-turnaround region
    # 2. For P2P only want turnaround region, which will be defined by a degree variable

    # mask out turnaround region
    turnaround_mask = (az_deg >= (azmin + cut)) & (az_deg <= (azmax - cut))

    turnaround_rmask = aman.flags.right_scan.mask() & turnaround_mask
    turnaround_lmask = aman.flags.left_scan.mask() & turnaround_mask

    fp_no_turnarounds = aman.fptemp[turnaround_mask]
    timestamps_no_turnarounds = aman.timestamps[turnaround_mask]
    az_deg_no_turnarounds = az_deg[turnaround_mask]

    rfpbin_no_turnarounds = make_binning(
        aman.fptemp[turnaround_rmask],
        aman.boresight.az[turnaround_rmask] / DEG,
        az_bin,
    )
    lfpbin_no_turnarounds = make_binning(
        aman.fptemp[turnaround_lmask],
        aman.boresight.az[turnaround_lmask] / DEG,
        az_bin,
    )
    razebin_no_turnarounds = make_binning(
        aman.Az_Pos_Err[turnaround_rmask],
        aman.boresight.az[turnaround_rmask] / DEG,
        az_bin,
    )
    lazebin_no_turnarounds = make_binning(
        aman.Az_Pos_Err[turnaround_lmask],
        aman.boresight.az[turnaround_lmask] / DEG,
        az_bin,
    )

    # mask out non-turnaround region
    near_min = (az_deg >= azmin) & (az_deg <= (azmin + cut))
    near_max = (az_deg <= azmax) & (az_deg >= (azmax - cut))
    non_turnaround_mask = near_min | near_max

    non_turnaround_rmask = aman.flags.right_scan.mask() & non_turnaround_mask
    non_turnaround_lmask = aman.flags.left_scan.mask() & non_turnaround_mask

    fp_only_turnarounds = aman.fptemp[non_turnaround_mask]
    timestamps_only_turnarounds = aman.timestamps[non_turnaround_mask]
    az_deg_only_turnarounds = az_deg[non_turnaround_mask]

    rfpbin_only_turnarounds = make_binning(
        aman.fptemp[non_turnaround_rmask],
        aman.boresight.az[non_turnaround_rmask] / DEG,
        az_bin,
    )
    lfpbin_only_turnarounds = make_binning(
        aman.fptemp[non_turnaround_lmask],
        aman.boresight.az[non_turnaround_lmask] / DEG,
        az_bin,
    )
    razebin_only_turnarounds = make_binning(
        aman.Az_Pos_Err[non_turnaround_rmask],
        aman.boresight.az[non_turnaround_rmask] / DEG,
        az_bin,
    )
    lazebin_only_turnarounds = make_binning(
        aman.Az_Pos_Err[non_turnaround_lmask],
        aman.boresight.az[non_turnaround_lmask] / DEG,
        az_bin,
    )

    ###################

    rfpbin_full = make_binning(
        aman.fptemp[rmask_full], aman.boresight.az[rmask_full] / DEG, az_bin
    )
    lfpbin_full = make_binning(
        aman.fptemp[lmask_full], aman.boresight.az[lmask_full] / DEG, az_bin
    )
    razebin_full = make_binning(
        aman.Az_Pos_Err[rmask_full], aman.boresight.az[rmask_full] / DEG, az_bin
    )
    lazebin_full = make_binning(
        aman.Az_Pos_Err[lmask_full], aman.boresight.az[lmask_full] / DEG, az_bin
    )

    ret = {}

    # Full data
    ret["az"] = az_bin
    ret["fp"] = fp_full
    ret["az_deg"] = az_deg
    ret["timestamps"] = timestamps_full
    ret["rcount"] = rfpbin_full["count"]
    ret["lcount"] = lfpbin_full["count"]
    ret["rfp"] = rfpbin_full["sig"]
    ret["rfp_sqre"] = rfpbin_full["sig_sqre"]
    ret["lfp"] = lfpbin_full["sig"]
    ret["lfp_sqre"] = lfpbin_full["sig_sqre"]
    ret["raze"] = razebin_full["sig"]
    ret["raze_sqre"] = razebin_full["sig_sqre"]
    ret["laze"] = lazebin_full["sig"]
    ret["laze_sqre"] = lazebin_full["sig_sqre"]
    # No turnaround data
    ret["fp_no_turnarounds"] = fp_no_turnarounds
    ret["az_deg_no_turnarounds"] = az_deg_no_turnarounds
    ret["timestamps_no_turnarounds"] = timestamps_no_turnarounds
    ret["rcount_no_turnarounds"] = rfpbin_no_turnarounds["count"]
    ret["lcount_no_turnarounds"] = lfpbin_no_turnarounds["count"]
    ret["rfp_no_turnarounds"] = rfpbin_no_turnarounds["sig"]
    ret["rfp_sqre_no_turnarounds"] = rfpbin_no_turnarounds["sig_sqre"]
    ret["lfp_no_turnarounds"] = lfpbin_no_turnarounds["sig"]
    ret["lfp_sqre_no_turnarounds"] = lfpbin_no_turnarounds["sig_sqre"]
    ret["raze_no_turnarounds"] = razebin_no_turnarounds["sig"]
    ret["raze_sqre_no_turnarounds"] = razebin_no_turnarounds["sig_sqre"]
    ret["laze_no_turnarounds"] = lazebin_no_turnarounds["sig"]
    ret["laze_sqre_no_turnarounds"] = lazebin_no_turnarounds["sig_sqre"]
    # Only turnaround data
    ret["fp_only_turnarounds"] = fp_only_turnarounds
    ret["az_deg_only_turnarounds"] = az_deg_only_turnarounds
    ret["timestamps_only_turnarounds"] = timestamps_only_turnarounds
    ret["rcount_only_turnarounds"] = rfpbin_only_turnarounds["count"]
    ret["lcount_only_turnarounds"] = lfpbin_only_turnarounds["count"]
    ret["rfp_only_turnarounds"] = rfpbin_only_turnarounds["sig"]
    ret["rfp_sqre_only_turnarounds"] = rfpbin_only_turnarounds["sig_sqre"]
    ret["lfp_only_turnarounds"] = lfpbin_only_turnarounds["sig"]
    ret["lfp_sqre_only_turnarounds"] = lfpbin_only_turnarounds["sig_sqre"]
    ret["raze_only_turnarounds"] = razebin_only_turnarounds["sig"]
    ret["raze_sqre_only_turnarounds"] = razebin_only_turnarounds["sig_sqre"]
    ret["laze_only_turnarounds"] = lazebin_only_turnarounds["sig"]
    ret["laze_sqre_only_turnarounds"] = lazebin_only_turnarounds["sig_sqre"]

    return ret, aman


def ana1(df):
    azbins = np.arange(0, 360, 0.2)
    # Full TOD
    rcounts = np.zeros(azbins.shape)
    rfp = np.zeros(azbins.shape)
    rfp_sqre = np.zeros(azbins.shape)
    rmc = np.zeros(azbins.shape)
    rmc_sqre = np.zeros(azbins.shape)
    raze = np.zeros(azbins.shape)
    raze_sqre = np.zeros(azbins.shape)

    lcounts = np.zeros(azbins.shape)
    lfp = np.zeros(azbins.shape)
    lfp_sqre = np.zeros(azbins.shape)
    lmc = np.zeros(azbins.shape)
    lmc_sqre = np.zeros(azbins.shape)
    laze = np.zeros(azbins.shape)
    laze_sqre = np.zeros(azbins.shape)

    # No Turnarounds
    rcounts_no_turnarounds = np.zeros(azbins.shape)
    rfp_no_turnarounds = np.zeros(azbins.shape)
    rfp_sqre_no_turnarounds = np.zeros(azbins.shape)
    raze_no_turnarounds = np.zeros(azbins.shape)
    raze_sqre_no_turnarounds = np.zeros(azbins.shape)

    lcounts_no_turnarounds = np.zeros(azbins.shape)
    lfp_no_turnarounds = np.zeros(azbins.shape)
    lfp_sqre_no_turnarounds = np.zeros(azbins.shape)
    laze_no_turnarounds = np.zeros(azbins.shape)
    laze_sqre_no_turnarounds = np.zeros(azbins.shape)

    # Only Turnarounds
    rcounts_only_turnarounds = np.zeros(azbins.shape)
    rfp_only_turnarounds = np.zeros(azbins.shape)
    rfp_sqre_only_turnarounds = np.zeros(azbins.shape)
    raze_only_turnarounds = np.zeros(azbins.shape)
    raze_sqre_only_turnarounds = np.zeros(azbins.shape)

    lcounts_only_turnarounds = np.zeros(azbins.shape)
    lfp_only_turnarounds = np.zeros(azbins.shape)
    lfp_sqre_only_turnarounds = np.zeros(azbins.shape)
    laze_only_turnarounds = np.zeros(azbins.shape)
    laze_sqre_only_turnarounds = np.zeros(azbins.shape)

    i = 0
    n = 0
    for i in tqdm(range(len(df))):
        iret = df["ret"][i]
        # Full TOD
        idxs = np.round(iret["az"] / 0.2).astype(int)

        np.add.at(rcounts, idxs, np.nan_to_num(iret["rcount"]))
        np.add.at(rfp, idxs, np.nan_to_num(iret["rfp"] * iret["rcount"]))
        np.add.at(rfp_sqre, idxs, np.nan_to_num(iret["rfp_sqre"] * iret["rcount"]))
        np.add.at(raze, idxs, np.nan_to_num(iret["raze"] * iret["rcount"]))
        np.add.at(raze_sqre, idxs, np.nan_to_num(iret["raze_sqre"] * iret["rcount"]))

        np.add.at(lcounts, idxs, np.nan_to_num(iret["lcount"]))
        np.add.at(lfp, idxs, np.nan_to_num(iret["lfp"] * iret["lcount"]))
        np.add.at(lfp_sqre, idxs, np.nan_to_num(iret["lfp_sqre"] * iret["lcount"]))
        np.add.at(laze, idxs, np.nan_to_num(iret["laze"] * iret["lcount"]))
        np.add.at(laze_sqre, idxs, np.nan_to_num(iret["laze_sqre"] * iret["lcount"]))

        # No turnaround TOD

        np.add.at(
            rcounts_no_turnarounds, idxs, np.nan_to_num(iret["rcount_no_turnarounds"])
        )
        np.add.at(
            rfp_no_turnarounds,
            idxs,
            np.nan_to_num(iret["rfp_no_turnarounds"] * iret["rcount_no_turnarounds"]),
        )
        np.add.at(
            rfp_sqre_no_turnarounds,
            idxs,
            np.nan_to_num(
                iret["rfp_sqre_no_turnarounds"] * iret["rcount_no_turnarounds"]
            ),
        )
        np.add.at(
            raze_no_turnarounds,
            idxs,
            np.nan_to_num(iret["raze_no_turnarounds"] * iret["rcount_no_turnarounds"]),
        )
        np.add.at(
            raze_sqre_no_turnarounds,
            idxs,
            np.nan_to_num(
                iret["raze_sqre_no_turnarounds"] * iret["rcount_no_turnarounds"]
            ),
        )

        np.add.at(
            lcounts_no_turnarounds, idxs, np.nan_to_num(iret["lcount_no_turnarounds"])
        )
        np.add.at(
            lfp_no_turnarounds,
            idxs,
            np.nan_to_num(iret["lfp_no_turnarounds"] * iret["lcount_no_turnarounds"]),
        )
        np.add.at(
            lfp_sqre_no_turnarounds,
            idxs,
            np.nan_to_num(
                iret["lfp_sqre_no_turnarounds"] * iret["lcount_no_turnarounds"]
            ),
        )

        np.add.at(
            laze_no_turnarounds,
            idxs,
            np.nan_to_num(iret["laze_no_turnarounds"] * iret["lcount_no_turnarounds"]),
        )
        np.add.at(
            laze_sqre_no_turnarounds,
            idxs,
            np.nan_to_num(
                iret["laze_sqre_no_turnarounds"] * iret["lcount_no_turnarounds"]
            ),
        )

        # Only turnaround TOD
        np.add.at(
            rcounts_only_turnarounds,
            idxs,
            np.nan_to_num(iret["rcount_only_turnarounds"]),
        )
        np.add.at(
            rfp_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["rfp_only_turnarounds"] * iret["rcount_only_turnarounds"]
            ),
        )
        np.add.at(
            rfp_sqre_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["rfp_sqre_only_turnarounds"] * iret["rcount_only_turnarounds"]
            ),
        )
        np.add.at(
            raze_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["raze_only_turnarounds"] * iret["rcount_only_turnarounds"]
            ),
        )
        np.add.at(
            raze_sqre_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["raze_sqre_only_turnarounds"] * iret["rcount_only_turnarounds"]
            ),
        )
        np.add.at(
            lcounts_only_turnarounds,
            idxs,
            np.nan_to_num(iret["lcount_only_turnarounds"]),
        )
        np.add.at(
            lfp_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["lfp_only_turnarounds"] * iret["lcount_only_turnarounds"]
            ),
        )
        np.add.at(
            lfp_sqre_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["lfp_sqre_only_turnarounds"] * iret["lcount_only_turnarounds"]
            ),
        )
        np.add.at(
            laze_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["laze_only_turnarounds"] * iret["lcount_only_turnarounds"]
            ),
        )
        np.add.at(
            laze_sqre_only_turnarounds,
            idxs,
            np.nan_to_num(
                iret["laze_sqre_only_turnarounds"] * iret["lcount_only_turnarounds"]
            ),
        )
        n += 1
    print(f"Number of observation that went into this plots = {n}")

    dfm = pd.DataFrame()
    dfm["az"] = azbins
    dfm["lmc"] = lmc / lcounts * 1e3
    dfm["lfp"] = lfp / lcounts * 1e3
    dfm["laz"] = laze / lcounts * 60

    dfm["lmcs"] = np.sqrt((lmc_sqre / lcounts - (lmc / lcounts) ** 2)) * 1e3
    dfm["lfps"] = np.sqrt((lfp_sqre / lcounts - (lfp / lcounts) ** 2)) * 1e3
    dfm["lazs"] = np.sqrt((laze_sqre / lcounts - (laze / lcounts) ** 2)) * 60

    dfm["lmce"] = dfm["lmcs"] / np.sqrt(lcounts)
    dfm["lfpe"] = dfm["lfps"] / np.sqrt(lcounts)
    dfm["laze"] = dfm["lazs"] / np.sqrt(lcounts)

    dfm["rmc"] = rmc / rcounts * 1e3
    dfm["rfp"] = rfp / rcounts * 1e3
    dfm["raz"] = raze / rcounts * 60

    dfm["rmcs"] = np.sqrt((rmc_sqre / rcounts - (rmc / rcounts) ** 2)) * 1e3
    dfm["rfps"] = np.sqrt((rfp_sqre / rcounts - (rfp / rcounts) ** 2)) * 1e3
    dfm["razs"] = np.sqrt((raze_sqre / rcounts - (raze / rcounts) ** 2)) * 60

    dfm["rmce"] = dfm["rmcs"] / np.sqrt(lcounts)
    dfm["rfpe"] = dfm["rfps"] / np.sqrt(lcounts)
    dfm["raze"] = dfm["razs"] / np.sqrt(lcounts)

    dfm["az1"] = dfm["az"].between(47, 91)
    dfm["az2"] = dfm["az"].between(109, 153)
    dfm["az3"] = dfm["az"].between(207, 251)
    dfm["az4"] = dfm["az"].between(269, 313)
    return dfm


def _gaussian(x, amp, mu, sigma):
    """Unnormalized Gaussian, used to model the temperature-residual histogram.

    Args:
        x (array_like): Points at which to evaluate the function.
        amp (float): Peak amplitude.
        mu (float): Mean (center).
        sigma (float): Standard deviation.

    Returns:
        np.ndarray: ``amp * exp(-(x - mu)^2 / (2 * sigma^2))`` evaluated at ``x``.
    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _fit_gaussian_to_residuals(variance, bins=10000):
    """Histogram temperature residuals and fit a Gaussian to the distribution.

    Args:
        variance (array_like): Temperature residuals (e.g. polynomial-detrended
            and median-subtracted), in mK.
        bins (int, optional): Number of histogram bins. Defaults to 10000.

    Returns:
        tuple: ``(popt, bin_centers, hist_vals)`` where ``popt`` is the
            ``(amp, mu, sigma)`` fit parameters from
            ``scipy.optimize.curve_fit``, and ``bin_centers``/``hist_vals``
            are the histogrammed residuals the fit was performed on. Raises
            ``RuntimeError`` if the fit fails to converge (propagated from
            ``curve_fit``).
    """
    hist_vals, bin_edges = np.histogram(variance, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    initial_guess = [np.max(hist_vals), np.mean(variance), np.std(variance)]
    popt, _ = curve_fit(_gaussian, bin_centers, hist_vals, p0=initial_guess)
    return popt, bin_centers, hist_vals


def _gaussian_r_squared(popt, bin_centers, hist_vals):
    """Compute the R^2 goodness-of-fit for a Gaussian fit to a histogram.

    Args:
        popt (tuple): ``(amp, mu, sigma)`` Gaussian fit parameters.
        bin_centers (array_like): Histogram bin centers the fit was evaluated on.
        hist_vals (array_like): Histogram density values at ``bin_centers``.

    Returns:
        float: Coefficient of determination (R^2) of the fit.
    """
    fit_at_bins = _gaussian(bin_centers, *popt)
    ss_res = np.sum((hist_vals - fit_at_bins) ** 2)
    ss_tot = np.sum((hist_vals - np.mean(hist_vals)) ** 2)
    return 1 - ss_res / ss_tot


def dataframe_maker(
    ctx,
    ys,
    ms,
    ds,
    ye,
    me,
    de,
    platform="satp1",
    tag=None,
    scan_type="cmb",
):
    """Create azimuth-binned pickle files for observations in a time range.

    The observation database in ``ctx`` is queried between the provided start
    and stop dates, then filtered by ``scan_type``. Each matching observation
    is passed to ``exe_azbin`` and the resulting pickle files are written to
    ``pkl_save_dir``.

    Args:
        ctx (sotodlib.core.Context): Context with an observation database.
        pkl_save_dir (str): Directory where output pickle files are saved.
        ys (int): Start year.
        ms (int): Start month.
        ds (int): Start day.
        ye (int): End year.
        me (int): End month.
        de (int): End day.
        platform (str): Telescope platform. Defaults to satp1.
        tag (str, optional): Observation tag to be entered. Defaults to None.
        scan_type (str, optional): Observation subtype to select. Defaults to
            'cmb'.

    Returns:
        str: The output directory path.
    """

    scan_start = dtime.datetime(ys, ms, ds, 0, 0, 0, 0, tzinfo=dtime.timezone.utc)
    scan_stop = dtime.datetime(
        ye, me, de, date.hour, date.minute, 0, 0, tzinfo=dtime.timezone.utc
    )
    if tag == None:
        query = {
            "query_text": f"start_time > {scan_start.timestamp()} and start_time < {scan_stop.timestamp()}",
        }
    else:
        query = {
            "tags": [f"{tag}=1"],
            "query_text": f"start_time > {scan_start.timestamp()} and start_time < {scan_stop.timestamp()}",
        }

    obslist = ctx.obsdb.query(**query)

    obss = []
    for iobs in obslist:
        if ctx.obsdb.get(iobs["obs_id"], tags=True)["subtype"] == scan_type:
            print(iobs["obs_id"], ctx.obsdb.get(iobs["obs_id"], tags=True)["tags"])
            obss.append(iobs["obs_id"])

    df = pd.DataFrame()
    for iobs in tqdm(obss):
        try:
            ret = exe_azbin(
                ctx,
                iobs,
                platform=platform,
            )
            df["obsid"] = iobs
            df["dt"] = [
                dtime.datetime.fromtimestamp(
                    int(iobs.split("_")[1]), tz=dtime.timezone.utc
                )
            ]
            df["ret"] = ret
        except Exception:
            print(f"exe_azbin failed for {iobs}:")
            traceback.print_exc()
    dfm = ana1(df)
    return dfm, df


def get_fpa_metrics(df, setpoint=75):
    """Calculate FPA temperature metrics from azimuth-binned pickle files.

    For each pickle file (one per observation) in ``pkl_save_dir``, this loads
    the FPA temperature time-ordered data, fits a 3rd order polynomial to
    remove slow trends, and fits a Gaussian to the residual distribution.
    Also computes the RMS/std of the temperature, its deviation from
    ``setpoint``, and the left/right turnaround peak-to-peak amplitude.

    Args:
        pkl_save_dir (str): Directory containing pickle files with FPA data,
            as produced by :func:`pkl`.
        setpoint (float, optional): Temperature setpoint, in mK (same units
            as the stored FPA temperature, which is converted from K to mK).
            Defaults to 75.

    Returns:
        dict: Dictionary keyed by observation ID, where each value is a dict
            with:
                - 'az_range': dict with 'min_az'/'max_az' (degrees).
                - 'gaussian_metrics': dict with 'x_fit'/'y_fit' (the fitted
                  Gaussian curve), 'fwhm', and 'variance' (the polynomial-
                  detrended, median-subtracted temperature residual, in mK).
                - 'popt': tuple, the ``(amp, mu, sigma)`` Gaussian fit params.
                - 'r_squared': float, goodness-of-fit (R^2) of the Gaussian fit.
                - 'rms': RMS temperature value, in mK.
                - 'std': Standard deviation of the temperature, in mK.
                - 'deviations': Absolute deviation of the RMS from ``setpoint``.
                - 'p2p': dict with 'left_p2p'/'right_p2p' turnaround
                  peak-to-peak amplitudes, in mK.

            Observations for which the Gaussian fit fails to converge, or
            which raise any other exception, are skipped (and logged) rather
            than included in the result.
    """

    # Calculate metrics
    ret = {}
    for idx in range(len(df["ret"])):
        try:
            temp = df["ret"][idx]["fp_no_turnarounds"] * 1e3
            t = df["ret"][idx]["timestamps_no_turnarounds"]
            x = (t - t[0]) / 60  # time in minutes
            y = temp

            # Calculate RMS, STD, and Deviation from setpoint
            rms_temp = np.sqrt(np.mean(np.square(y)))
            std = np.std(y)
            dev = abs(setpoint - float(f"{rms_temp:.8f}"))

            # Fit 3rd order polynomial to temperature TOD
            coeffs = np.polyfit(x, y, deg=3)
            poly_fit = np.polyval(coeffs, x)
            residuals = y - poly_fit
            variance = residuals - np.median(residuals)  # variance of residual, in mK

            try:
                popt, bin_centers, hist_vals = _fit_gaussian_to_residuals(variance)
            except RuntimeError:
                print(f"Fit failed for index {df['ret'][idx]}")
                continue

            amp, mu, sigma = popt
            fwhm = 2.355 * sigma
            r_squared = _gaussian_r_squared(popt, bin_centers, hist_vals)

            # Evaluate fit
            x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 500)
            y_fit = _gaussian(x_fit, *popt)

            # Calculate P2P with turnaround only data
            az = df["ret"][idx]["az"]
            rfp = (
                df["ret"][idx]["rfp_only_turnarounds"] * 1e3
            )  # K to mK conversion factor
            lfp = df["ret"][idx]["lfp_only_turnarounds"] * 1e3

            med_az = np.median(az)
            rfp = np.where(az < med_az, rfp, np.nan)
            lfp = np.where(az > med_az, lfp, np.nan)

            l_p2p = np.nanmax(lfp) - np.nanmin(lfp)
            r_p2p = np.nanmax(rfp) - np.nanmin(rfp)

            # Save to return dict
            obs_id = df["obsid"][idx]
            ret[obs_id] = {
                "az_range": {
                    "min_az": np.min(df["ret"][idx]["az_deg"]),
                    "max_az": np.max(df["ret"][idx]["az_deg"]),
                },
                "gaussian_metrics": {
                    "x_fit": x_fit,
                    "y_fit": y_fit,
                    "fwhm": fwhm,
                    "variance": variance,
                },
                "popt": popt,
                "r_squared": r_squared,
                "rms": rms_temp,
                "std": std,
                "deviations": dev,
                "p2p": {
                    "left_p2p": l_p2p,
                    "right_p2p": r_p2p,
                },
            }
        except Exception:
            print(f"get_fpa_metrics failed for index {idx}:")
            traceback.print_exc()
    return ret


def metric_analyzer(
    metric_dict,
    save_path=None,
    platform="satp1",
    show_plot=False,
):
    """Flag observations whose FPA metrics exceed platform-specific thresholds.

    Observations are binned into 4 fixed azimuth ranges (one per scan
    "field"); each bin has its own std/peak-to-peak/goodness-of-fit
    thresholds, calibrated per platform (satp1/satp2/satp3). For every
    observation whose azimuth range matches a bin, this checks the standard
    deviation, left/right turnaround peak-to-peak amplitude, and Gaussian fit
    R^2 (as computed by :func:`get_fpa_metrics`) against that bin's
    thresholds, printing a message for each exceedance. When the Gaussian R^2
    is below threshold, a diagnostic histogram + fit plot is also produced
    (saved to ``save_path`` if given).

    Args:
        metric_dict (dict): Per-observation metrics, as returned by
            :func:`get_fpa_metrics`.
        save_path (str, optional): Directory in which to save diagnostic
            Gaussian-fit plots for flagged observations. If None, plots are
            generated but not written to disk. Defaults to None.
        platform (str, optional): Telescope platform, used to select
            threshold values ("satp1", "satp2", or "satp3"). Defaults to
            "satp1".

    Returns:
        dict: Flag counts per azimuth bin (length-4 arrays), with keys:
            - 'std_cnt': Standard deviation threshold exceedances.
            - 'r_p2p_cnt': Right-turnaround peak-to-peak exceedances.
            - 'l_p2p_cnt': Left-turnaround peak-to-peak exceedances.
            - 'gof_cnt': Gaussian goodness-of-fit (R^2) exceedances.
    """
    minazs = [48.6, 110, 207, 268]
    maxazs = [91, 153, 250, 313]

    n_std = np.zeros(4, dtype=np.int_)
    n_r_p2p = np.zeros(4, dtype=np.int_)
    n_l_p2p = np.zeros(4, dtype=np.int_)
    n_gauss = np.zeros(4, dtype=np.int_)

    # set thresholds based on platform
    if platform == "satp1":
        std_threshold = [0.1, 0.13, 0.1, 0.12]
        r_p2p_threshold = [0.15, 0.12, 0.17, 0.15]
        l_p2p_threshold = [0.15, 0.3, 0.15, 0.12]
        r_sq_threshold = [0.9, 0.9, 0.9, 0.9]
    if platform == "satp2":
        std_threshold = [2.2e-1, 2.1e-1, 2.2e-1, 2.2e-1]
        r_p2p_threshold = [0.3, 0.35, 0.4, 0.4]
        l_p2p_threshold = [0.26, 0.35, 0.4, 0.35]
        r_sq_threshold = [0.8, 0.8, 0.8, 0.8]
    if platform == "satp3":
        std_threshold = [3.5e-2, 4.5e-2, 4e-2, 4.5e-2]
        r_p2p_threshold = [0.06, 0.08, 0.06, 0.08]
        l_p2p_threshold = [0.1, 0.1, 0.13, 0.15]
        r_sq_threshold = [0.9, 0.9, 0.9, 0.9]

    for j in range(4):
        minaz, maxaz = minazs[j], maxazs[j]
        for obs in metric_dict:
            std = metric_dict[obs]["std"]
            l_p2p = metric_dict[obs]["p2p"]["left_p2p"]
            r_p2p = metric_dict[obs]["p2p"]["right_p2p"]
            min_az, max_az = (
                metric_dict[obs]["az_range"]["min_az"],
                metric_dict[obs]["az_range"]["max_az"],
            )
            if np.isclose(min_az, minaz, rtol=1e-1) & np.isclose(
                max_az, maxaz, rtol=1e-1
            ):
                print(std, std_threshold[j])
                if std > std_threshold[j]:
                    print(
                        f"Standard deviation of observation {obs} is over threshold value of {std_threshold[j]} mK at {round(std, 2)} mK."
                    )
                    print(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_std[j] += 1
                if l_p2p > l_p2p_threshold[j]:
                    print(
                        f"Left going turnaround of observation {obs} is over threshold value of {l_p2p_threshold[j]} mK at {round(l_p2p, 2)} mK."
                    )
                    print(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_l_p2p[j] += 1
                if r_p2p > r_p2p_threshold[j]:
                    print(
                        f"Right going turnaround of observation {obs} is over threshold value of {r_p2p_threshold[j]} mK at {round(r_p2p, 2)} mK."
                    )
                    print(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_r_p2p[j] += 1

                # Gaussian thresholding (fit already computed by get_fpa_metrics)
                x_fit, y_fit, fwhm, variance = metric_dict[obs][
                    "gaussian_metrics"
                ].values()
                r_sq = metric_dict[obs]["r_squared"]

                if r_sq < r_sq_threshold[j]:
                    print(
                        f"Gaussian R squared value of observation {obs} is less than the threshold value of {r_sq_threshold[j]} at {round(r_sq, 2)}."
                    )
                    print(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_gauss[j] += 1
                    if show_plot:    
                        plt.hist(
                            variance,
                            density=True,
                            bins=50,
                            histtype="step",
                            label="Binned Data",
                        )
                        plt.plot(
                            x_fit, y_fit, label=f"Gaussian fit\nR_sq: {round(r_sq, 2)}"
                        )
                        plt.xlabel("Temperature Residual (mK)")
                        plt.ylabel("Probability Density")
                        plt.title("Gaussian fit to Deviation from Zero")
                        plt.legend()
                        if save_path != None:
                            plt.savefig(os.path.join(save_path + f"/{obs}_gaussian.png"))
                        plt.close()
        print(f"Number of Standard deviation flags: {n_std[j]}")
        print(f"Number of right turnaround P2P flags: {n_r_p2p[j]}")
        print(f"Number of left turnaround P2P flags: {n_l_p2p[j]}")
        print(f"Number of goodness of fit flags: {n_gauss[j]}")

    # Save flag counts to dict and return
    ret = {
        "std_cnt": n_std,
        "r_p2p_cnt": n_r_p2p,
        "l_p2p_cnt": n_l_p2p,
        "gof_cnt": n_gauss,
    }
    return ret

def az_circle_plotter(
    cnt_dict,
    dfm,
    df,
    ms,
    ds,
    me,
    de,
    platform="satp1",
    debug=False,
    save_path=None,
    show=True,
):
    """Plot a polar summary of azimuth scans, turnaround temperatures, and flags.

    Independently reloads the azimuth-binned pickle files for the given date
    range (from the platform's standard ``az_pkls`` output directory) via
    ``pkl_loader``, groups observations into 4 azimuth bins, and renders a
    polar plot showing: the scan azimuth range as a dashed circle, left/right
    turnaround FPA temperature vs. azimuth (blue/red traces, offset radially
    from the setpoint), the local horizon profile (from Cerro Toco survey
    data) and the LAT position, and the per-bin flag counts from
    :func:`metric_analyzer` (``cnt_dict``) annotated as text boxes.

    Args:
        cnt_dict (dict): Per-bin flag counts, as returned by
            :func:`metric_analyzer` (keys 'std_cnt', 'r_p2p_cnt',
            'l_p2p_cnt', 'gof_cnt', each a length-4 array).
        ys (int): Start year of the date range to reload pickle files for.
        ms (int): Start month.
        ds (int): Start day.
        ye (int): End year.
        me (int): End month.
        de (int): End day.
        platform (str, optional): Telescope platform; selects the pickle
            directory and temperature setpoint. Defaults to "satp1".
        debug (bool, optional): If True, print extra diagnostic info (azimuth
            bin counts, per-scan data) while building the plot. Defaults to
            False.
        save_path (str, optional): If given, path to save the figure to (in
            addition to optionally displaying it). Defaults to None.
        show (bool, optional): If True, display the figure with
            ``plt.show()``. Set to False for unattended/batch execution,
            since ``plt.show()`` blocks until the figure window is closed
            (or can error/hang on a headless node with an interactive
            backend and no display). Defaults to True.

    Returns:
        None. Optionally displays the figure and, if ``save_path`` is given,
        saves it to disk.
    """

    # Set figure params and geographical data
    offset = 0.8
    roff = -0.08
    loff = 0.08
    magnify = 1
    angle_offset = 90
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    center = (0, 0)
    radius = offset
    circle = plt.Circle(center, radius, fill=False, linestyle="--")
    ax.add_patch(circle)

    toco_az = np.array(
        [
            0,
            6,
            12,
            18,
            24,
            30,
            36,
            42,
            48,
            54,
            60,
            66,
            72,
            78,
            84,
            90,
            96,
            102,
            108,
            114,
            120,
            126,
            132,
            138,
            144,
            150,
            156,
            162,
            168,
            174,
            180,
            186,
            192,
            198,
            204,
            210,
            216,
            222,
            228,
            234,
            240,
            246,
            252,
            258,
            264,
            270,
            276,
            282,
            288,
            294,
            300,
            306,
            312,
            318,
            324,
            330,
            336,
            342,
            348,
            354,
        ]
    )
    toco_el = np.array(
        [
            0.47322,
            1.94546,
            5.34882,
            7.6002,
            10.038,
            12.86298,
            15.07134,
            14.8658,
            11.57238,
            9.86114,
            12.27982,
            12.09818,
            8.83822,
            7.4329,
            6.79716,
            5.88896,
            5.38706,
            5.05724,
            3.43682,
            3.42726,
            4.8278,
            4.8278,
            3.69494,
            1.90722,
            1.47702,
            1.66344,
            1.95024,
            2.09842,
            2.21314,
            0.84606,
            0.44932,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.44932,
            -0.33938,
            -0.5019,
            -0.42542,
            -0.44454,
            -0.60228,
            -0.8365,
            -1.41488,
            -1.673,
            -2.00282,
            -2.03628,
            -2.03628,
            -1.97892,
            -1.3384,
            -1.94546,
            1.94546,
            1.19978,
            2.19402,
            0.55926,
            1.82118,
            1.49136,
            0.0,
        ]
    )

    lat_az = 169  # https://docs.google.com/presentation/d/1XVWdp08v30p7zvJQtMD_piv6gAXbO6VMjkdIMFuSHX8/edit?slide=id.g334ac71e4d3_0_0#slide=id.g334ac71e4d3_0_0
    lat_el = 5  # Cerro Toco is at an angle of 15 deg and the top of the LAT is at an angle of 5 deg which is different than what I reported. email from Lyman

    nominal_minazs = [48.6, 110, 207, 268]
    nominal_maxazs = [91, 153, 250, 313]

    minazs = []
    maxazs = []
    for i in range(len(df["ret"])):
        ret = df["ret"][i]
        minaz, maxaz = np.min(ret["az"]), np.max(ret["az"])
        minazs.append(minaz)
        maxazs.append(maxaz)
    minazs = np.unique(minazs)
    maxazs = np.unique(maxazs)

    binned_minazs = [[] for _ in range(4)]
    binned_maxazs = [[] for _ in range(4)]
    for v in minazs:
        j = int(np.argmin(np.abs(np.array(nominal_minazs) - v)))
        binned_minazs[j].append(v)
    for v in maxazs:
        j = int(np.argmin(np.abs(np.array(nominal_maxazs) - v)))
        binned_maxazs[j].append(v)

    if debug:
        print(f"binned minazs: {binned_minazs}")
        print(f"binned maxazs: {binned_maxazs}")

    azidx = []
    for i in range(len(df)):
        iazmed = np.median(df["ret"][i]["az"])
        bl = False
        for j in range(4):
            if not binned_minazs[j] or not binned_maxazs[j]:
                continue  # no observations fell in this bin
            if any(iazmed > x for x in binned_minazs[j]) and any(
                iazmed < x for x in binned_maxazs[j]
            ):
                azidx.append(j)
                bl = True
                break
        if not bl:
            azidx.append(-1)
    df["azidx"] = azidx
    if debug:
        print(azidx)

    setpoints = {
        "satp1": 75,
        "satp2": 100,
        "satp3": 62,
    }

    # Number of points trimmed off each end of a turnaround trace before
    # drawing its direction arrow (avoids drawing the arrow through the
    # noisy edge of the trace).
    edge_trim = 50
    DEG = np.pi / 180
    colors = {
    "purple": "#9400D3",
    "blue": "#0066FF",
    "orange": "#FF8800",
    "red": "#FF2200",
    "green": "#00CC44",
    "yellow": "#FFD500",
    "cyan": "#00D5FF",
    "lime": "#A6E22E",
    "teal": "#00C2A8",
    }

    for i, imask in enumerate([dfm["az1"], dfm["az2"], dfm["az3"], dfm["az4"]]):
        if not imask.any():
            # No observations landed in this az bin over the requested date
            # range -- nothing to plot for it, so skip rather than operate
            # on empty arrays below.
            if debug:
                print(f"az_circle_plotter: no observations in az bin {i}, skipping")
            continue

        iaz = dfm["az"][imask] + angle_offset
        iraz = dfm["rfp"][imask]
        ilaz = dfm["lfp"][imask]

        if debug:
            print(iraz)
            print(ilaz)

        irfl = ~np.isnan(iraz.values)
        raz = iaz.values[irfl]
        if raz.size > 0:
            rrad = ((iraz.values[irfl] - setpoints[platform]) + offset + roff) * magnify
            irx = rrad * np.cos(raz * DEG)
            iry = rrad * np.sin(raz * DEG)
            ax.plot(irx, iry, color=colors["red"])

            # Only trim the edges if there are enough points left over to do
            # so; otherwise use the whole (short) trace.
            rs = (
                slice(edge_trim, -edge_trim)
                if raz.size > 2 * edge_trim
                else slice(None)
            )
            iarx = np.median(rrad) * 0.95 * np.cos(raz[rs] * DEG)
            iary = np.median(rrad) * 0.95 * np.sin(raz[rs] * DEG)
            ax.plot(iarx, iary, color=colors["red"], alpha=0.5)
            if iarx.size >= 2:
                ax.annotate(
                    "",
                    xy=[iarx[-1], iary[-1]],
                    xytext=[iarx[-2], iary[-2]],
                    arrowprops=dict(
                        shrink=0,
                        width=1,
                        headwidth=8,
                        headlength=10,
                        connectionstyle="arc3",
                        facecolor=colors["red"],
                        edgecolor=colors["red"],
                        alpha=0.8,
                    ),
                )
        elif debug:
            print(f"az_circle_plotter: no valid right-turnaround data in az bin {i}")

        ilfl = ~np.isnan(ilaz.values)
        laz = iaz.values[ilfl]
        if laz.size > 0:
            lrad = ((ilaz.values[ilfl] - setpoints[platform]) + offset + loff) * magnify
            ilx = lrad * np.cos(laz * DEG)
            ily = lrad * np.sin(laz * DEG)
            ax.plot(ilx, ily, color=colors["blue"])

            ls = (
                slice(edge_trim, -edge_trim)
                if laz.size > 2 * edge_trim
                else slice(None)
            )
            ialx = np.median(lrad) * 1.05 * np.cos(laz[ls] * DEG)
            ialy = np.median(lrad) * 1.05 * np.sin(laz[ls] * DEG)
            ax.plot(ialx, ialy, color=colors["blue"], alpha=0.5)
            if ialx.size >= 2:
                ax.annotate(
                    "",
                    xy=[ialx[0], ialy[0]],
                    xytext=[ialx[1], ialy[1]],
                    arrowprops=dict(
                        shrink=0,
                        width=1,
                        headwidth=8,
                        headlength=10,
                        connectionstyle="arc3",
                        facecolor=colors["blue"],
                        edgecolor=colors["blue"],
                        alpha=0.8,
                    ),
                )
        elif debug:
            print(f"az_circle_plotter: no valid left-turnaround data in az bin {i}")

        # arc = patches.Arc(center, 2*radius + 0.02, 2*radius + 0.02, angle=0, theta1=np.min(iaz), theta2=np.max(iaz), color='blue', linewidth=2)
        # ax.add_patch(arc)
        # arc = patches.Arc(center, 2*radius - 0.02, 2*radius - 0.02, angle=0, theta1=np.min(iaz), theta2=np.max(iaz), color='red', linewidth=2)
        # ax.add_patch(arc)

    azimuths = np.arange(0, 360, 30)
    for az in azimuths:
        rad = np.deg2rad(az + angle_offset)
        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)

        label_x = center[0] + 1 * radius * np.cos(rad)
        label_y = center[1] + 1 * radius * np.sin(rad)
        ax.text(
            label_x,
            label_y,
            f"{az}°",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
        )

    fig.tight_layout()
    ax.text(
        0.99,
        0.85,
        "Blue: Left-going (Az decreasing) scan",
        color=colors["blue"],
        fontsize=15,
    )
    ax.text(
        0.99,
        0.80,
        "Red: Right-going (Az increasing) scan",
        color=colors["red"],
        fontsize=15,
    )

    ifac = 4e-3
    ampoff = 200
    toco_rx = (toco_el + ampoff) * np.cos((toco_az + angle_offset) * DEG) * ifac
    toco_ry = (toco_el + ampoff) * np.sin((toco_az + angle_offset) * DEG) * ifac
    ax.plot(toco_rx, toco_ry, c=colors["orange"])
    lat_rx = (lat_el + ampoff) * np.cos((lat_az + angle_offset) * DEG) * ifac
    lat_ry = (lat_el + ampoff) * np.sin((lat_az + angle_offset) * DEG) * ifac
    ax.plot(lat_rx, lat_ry, "s", color="black")
    ax.text(
        -0.3,
        0.8,
        f"Orange: Horizon\nMax(EL of TOCO) is {np.max(toco_el):.01f} at Az = {toco_az[np.argmax(toco_el)]:.01f} deg",
        color=colors["orange"],
        fontsize=13,
    )
    ax.text(
        0.1,
        -0.9,
        f"Black square: LAT(EL = {lat_el:.01f}, Az = {lat_az:.01f} deg)",
        color="black",
        fontsize=13,
    )

    obsx = [-1, -1, 1, 1]
    obsy = [0.5, -0.5, -0.5, 0.5]
    std, r_p2p, l_p2p, gof = cnt_dict.values()

    for i in range(4):
        obslen = len(np.where(df["azidx"] == i)[0])
        # print(minazs[i], obslen)
        ax.text(
            obsx[i],
            obsy[i],
            f"#obs: {obslen}\nstd flags: {std[i]}\nr_p2p flags: {r_p2p[i]}\nl_p2p flags: {l_p2p[i]}\ngof flags: {gof[i]}",
            fontsize=15,
            va="center",
            ha="center",
            bbox={
                "facecolor": colors["cyan"],
                "alpha": 0.6,
                "pad": 7,
            },
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_xaxis()
    ax.axis("off")
    fig.suptitle(f"{platform}, {ms}/{ds} - {me}/{de}", y=0.93)
    if show:
        plt.show()
    if save_path != None:
        fig.savefig(save_path)


# Per-platform context config key and FPA temperature setpoint (mK).
PLATFORM_SETTINGS = {
    "satp1": {"ctx_key": "ctx_path1", "setpoint": 75},
    "satp2": {"ctx_key": "ctx_path2", "setpoint": 100},
    "satp3": {"ctx_key": "ctx_path3", "setpoint": 62},
}

def ensure_dir(path):
    """Create a directory if it doesn't already exist, logging the outcome.

    Unlike a bare ``os.makedirs(path, exist_ok=True)``, this prints a message
    distinguishing "created" from "already exists" (matching the previous
    behavior of this script), and reports permission errors without raising.

    Args:
        path (str): Directory path to create.

    Returns:
        str: ``path``, unchanged, for convenient chaining.
    """
    if os.path.isdir(path):
        print(f"Directory '{path}' already exists.")
        return path
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    except PermissionError:
        print(f"Permission Denied: unable to create directory '{path}'.")
    except Exception:
        print(f"An error occurred creating directory '{path}':")
        traceback.print_exc()
    return path

def get_parser(parser=None):
    """Parse command-line arguments for the FPA metrics pipeline.

    Args:
        argv (list[str], optional): Argument list to parse instead of
            ``sys.argv``. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument("config", help="Configuration yaml file.")
    parser.add_argument("--ys", type=int, default=date.year, help="Start year.")
    parser.add_argument("--ms", type=int, default=date.month, help="Start month.")
    parser.add_argument("--ds", type=int, default=1, help="Start day.")
    parser.add_argument("--ye", type=int, default=date.year, help="End year.")
    parser.add_argument("--me", type=int, default=date.month, help="End month.")
    parser.add_argument("--de", type=int, default=date.day, help="End day.")
    parser.add_argument(
        "-t", "--tag", type=str, default=None, help="Observation tag filter."
    )
    parser.add_argument(
        "--scan_type",
        type=str,
        default="cmb",
        help="Observation subtype to select.",
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="satp1",
        choices=sorted(PLATFORM_SETTINGS),
        help="Telescope platform.",
    )
    return parser


def main(
    config: str, 
    verbose: Optional[int] = 2,
    ys: Optional[int] = date.year,
    ms: Optional[int] = date.month,
    ds: Optional[int] = 1,
    ye: Optional[int] = date.year,
    me: Optional[int] = date.month,
    de: Optional[int] = date.day,
    tag: Optional[str] = None,
    scan_type: Optional[str] = "cmb",
    platform: Optional[str] = "satp1",
):
    """Run the full FPA metrics pipeline: bin, compute metrics, flag, and plot.

    Args:
        argv (list[str], optional): Argument list to parse instead of
            ``sys.argv`` (useful for testing). Defaults to None.
    """
    # Set verbose
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.INFO)
    elif verbose == 3:
        logger.setLevel(logging.DEBUG)
    
    configs = yaml.safe_load(open(config, "r"))

    settings = PLATFORM_SETTINGS[platform]
    ctx = core.Context(configs["contexts"][settings["ctx_key"]])
    setpoint = settings["setpoint"]
    dfm, df = dataframe_maker(
        ctx,
        ys=ys,
        ms=ms,
        ds=ds,
        ye=ye,
        me=me,
        de=de,
        platform=platform,
        tag=tag,
        scan_type=scan_type,
    )
    print("Made dataframes")

    df_ids = df["obsids"]
    # Get metrics from pkl files
    if not df_ids:
        print(
            f"No observations found for {platform} between "
            f"{ys}/{ms}/{ds} and {ye}/{me}/{de}."
        )
    print("trying to compute metrics")
    metric_dict = get_fpa_metrics(df=df, setpoint=setpoint)
    print("finished computing metrics")

    print("trying to analyze metrics")
    cnt_ret = metric_analyzer(metric_dict, save_path=None, platform=platform)
    
    if df_ids:
        print("trying to save cirlce plot")
        circle_plt_save_p = os.path.join(
            metric_save_dir, f"{args.ye}_{args.me}_{args.de}_circle_plot.png"
        )
        az_circle_plotter(
            cnt_dict=cnt_ret,
            dfm=dfm,
            df=df,
            ms=ms,
            ds=ds,
            me=me,
            de=de,
            platform=platform,
            save_path=circle_plt_save_p,
            show=False,
        )
    else:
        print("Skipping az_circle_plotter since there are no observations to plot.")
    print("finished analyzing metrics")


if __name__ == "__main__":
    main_launcher(main, get_parser)
