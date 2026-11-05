"""Compute and visualize focal-plane-array (FPA) temperature stability metrics.

Pipeline (see :func:`main`):

1. :func:`dataframe_maker` queries an observation database over a date range
   and runs :func:`exe_azbin` on each matching observation to produce
   azimuth-binned FPA temperature data, which are accumulated into a
   DataFrame (by :func:`ana1`) for plotting.
2. :func:`get_fpa_metrics` computes RMS, standard deviation, deviation from
   setpoint, peak-to-peak turnaround, and Gaussian-fit metrics for each
   observation.
3. :func:`metric_analyzer` flags observations whose metrics exceed
   platform-specific thresholds, binned by azimuth range.
4. :func:`az_circle_plotter` renders a polar summary plot of the azimuth
   scan range with flag counts overlaid.
"""

import argparse
import datetime as dtime
import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from sotodlib import core, tod_ops
from sotodlib.io import hkdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher

logger = init_logger('make_fpa_temp_metrics', 'make_fpa_temp_metrics: ')
date = dtime.datetime.now()
DEG = np.pi / 180

# Per-platform context config key and FPA temperature setpoint (mK).
PLATFORM_SETTINGS = {
    "satp1": {"ctx_key": "ctx_path1", "setpoint": 75},
    "satp2": {"ctx_key": "ctx_path2", "setpoint": 100},
    "satp3": {"ctx_key": "ctx_path3", "setpoint": 62},
}

# Cerro Toco horizon profile: elevation (deg) every 6 deg in azimuth.
TOCO_AZ = np.arange(0, 360, 6)
TOCO_EL = np.array(
    [
        0.47322, 1.94546, 5.34882, 7.6002, 10.038, 12.86298,
        15.07134, 14.8658, 11.57238, 9.86114, 12.27982, 12.09818,
        8.83822, 7.4329, 6.79716, 5.88896, 5.38706, 5.05724,
        3.43682, 3.42726, 4.8278, 4.8278, 3.69494, 1.90722,
        1.47702, 1.66344, 1.95024, 2.09842, 2.21314, 0.84606,
        0.44932, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -0.44932, -0.33938, -0.5019, -0.42542,
        -0.44454, -0.60228, -0.8365, -1.41488, -1.673, -2.00282,
        -2.03628, -2.03628, -1.97892, -1.3384, -1.94546, 1.94546,
        1.19978, 2.19402, 0.55926, 1.82118, 1.49136, 0.0,
    ]
)


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
        logger.info("Skip this data due to the data length is > 4000*200.")
        return np.nan, np.nan, np.nan, np.nan
    return aman


def load_az_temp(aman, hk_cfg):
    """Load HK azimuth and FPA temperature data and add it to ``aman``.

    The HK data are linearly interpolated onto ``aman.timestamps`` and wrapped
    into ``aman`` as float32 ``Az_Pos_Des``, ``Az_Pos_Err``,
    ``Corrected_Azimuth`` and ``fptemp``.

    Args:
        aman (sotodlib.core.AxisManager): Observation to add the HK data to.
        hk_cfg (dict): The ``hk`` block of the pipeline config. Contains the
            :class:`sotodlib.io.hkdb.HkConfig` keys (e.g. ``hk_root``,
            ``db_file``) plus ``fields``, the four HK fields to load, in the
            order azimuth setpoint, azimuth error, corrected azimuth, FPA
            temperature.
    """
    hk_cfg = dict(hk_cfg)
    keys = hk_cfg.pop("fields")
    cfg = hkdb.HkConfig.from_dict(hk_cfg)

    spec = hkdb.LoadSpec(
        cfg=cfg,
        fields=keys,
        start=float(aman.timestamps[0]),
        end=float(aman.timestamps[-1]),
    )
    res = hkdb.load_hk(spec, show_pb=True)
    labels = ["Az_Pos_Des", "Az_Pos_Err", "Corrected_Azimuth", "fptemp"]

    for ikey, ilabel in zip(keys, labels):
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


# Keys returned by _bin_scan_stats, in the order they appear in exe_azbin output.
_BINNED_KEYS = (
    "rcount", "lcount", "rfp", "rfp_sqre", "lfp", "lfp_sqre",
    "raze", "raze_sqre", "laze", "laze_sqre",
)


def _bin_scan_stats(aman, az_deg, az_bin, mask):
    """Bin FPA temperature and azimuth error by azimuth for each scan direction.

    Args:
        aman (sotodlib.core.AxisManager): Observation with ``fptemp``,
            ``Az_Pos_Err`` and left/right scan flags.
        az_deg (np.ndarray): Boresight azimuth, in degrees.
        az_bin (np.ndarray): Azimuth bin centers, in degrees.
        mask (np.ndarray): Boolean sample mask; only samples where it is True
            are binned.

    Returns:
        dict: Binned statistics for the right (``r``) and left (``l``)
            scans, keyed ``{side}count``, ``{side}fp``, ``{side}fp_sqre``,
            ``{side}aze`` and ``{side}aze_sqre``.
    """
    out = {}
    for side, scan in (("r", aman.flags.right_scan), ("l", aman.flags.left_scan)):
        smask = scan.mask() & mask
        fpbin = make_binning(aman.fptemp[smask], az_deg[smask], az_bin)
        azebin = make_binning(aman.Az_Pos_Err[smask], az_deg[smask], az_bin)
        out[f"{side}count"] = fpbin["count"]
        out[f"{side}fp"] = fpbin["sig"]
        out[f"{side}fp_sqre"] = fpbin["sig_sqre"]
        out[f"{side}aze"] = azebin["sig"]
        out[f"{side}aze_sqre"] = azebin["sig_sqre"]
    return out


def exe_azbin(
    ctx,
    iobs,
    hk_cfg,
    cut=5,
):
    """Load an observation and bin its FPA temperature in azimuth.

    The full TOD is trimmed in three ways: the full data (no suffix), the
    region excluding the ``cut`` degrees nearest each end of the scan
    (``_no_turnarounds``, used for RMS and Gaussian statistics), and only
    those end regions (``_only_turnarounds``, used for peak-to-peak).

    Args:
        ctx (sotodlib.core.Context): Context to load the observation from.
        iobs (str): Observation ID.
        hk_cfg (dict): HK loading config, passed to :func:`load_az_temp`.
        platform (str, optional): Telescope platform. Defaults to "satp1".
        cut (float, optional): Width of the turnaround region at each end of
            the scan, in degrees. Defaults to 5.

    Returns:
        tuple: ``(ret, aman)``. ``ret`` is a dict holding, for each of the
            three trimmings, the trimmed ``fp``, ``az_deg`` and
            ``timestamps`` arrays plus the :func:`_bin_scan_stats` outputs;
            ``ret["az"]`` is the azimuth bin array. ``aman`` is the loaded
            observation.
    """
    aman = exe_p2p_ana_load(ctx, iobs)
    load_az_temp(aman, hk_cfg)
    az_deg = aman.boresight.az / DEG
    azmax = round(np.max(az_deg), 1)
    azmin = round(np.min(az_deg), 1)
    az_bin = np.arange(azmin, azmax, 0.2)

    # Samples away from the scan ends, and samples within ``cut`` of them.
    away_from_ends = (az_deg >= (azmin + cut)) & (az_deg <= (azmax - cut))
    near_ends = ((az_deg >= azmin) & (az_deg <= (azmin + cut))) | (
        (az_deg <= azmax) & (az_deg >= (azmax - cut))
    )
    trimmings = (
        ("", np.ones(az_deg.shape, dtype=bool)),
        ("_no_turnarounds", away_from_ends),
        ("_only_turnarounds", near_ends),
    )

    ret = {"az": az_bin}
    for suffix, mask in trimmings:
        ret[f"fp{suffix}"] = aman.fptemp[mask]
        ret[f"az_deg{suffix}"] = az_deg[mask]
        ret[f"timestamps{suffix}"] = aman.timestamps[mask]
        binned = _bin_scan_stats(aman, az_deg, az_bin, mask)
        for key in _BINNED_KEYS:
            ret[f"{key}{suffix}"] = binned[key]

    return ret, aman


def ana1(df):
    """Accumulate per-observation azimuth bins into an overall azimuth profile.

    Args:
        df (pandas.DataFrame): Per-observation data, as built by
            :func:`dataframe_maker`; its ``ret`` column holds the
            :func:`exe_azbin` output for each observation.

    Returns:
        pandas.DataFrame: One row per 0.2 deg azimuth bin, with the mean
            (``fp``, ``az``), standard deviation (``*s``) and standard error
            (``*e``) of the FPA temperature (mK) and azimuth error
            (arcmin), for left (``l``) and right (``r``) scans. The ``mc``
            columns are not currently filled: they are 0 in bins with data
            and NaN in empty bins. ``az1``-``az4`` flag the four nominal
            azimuth ranges.
    """
    azbins = np.arange(0, 360, 0.2)
    suffixes = ("", "_no_turnarounds", "_only_turnarounds")
    stats = ("fp", "fp_sqre", "aze", "aze_sqre")

    # Count-weighted sums per scan direction and trimming (see exe_azbin).
    acc = {
        (side, sfx): {k: np.zeros(azbins.shape) for k in ("count",) + stats}
        for side in "rl"
        for sfx in suffixes
    }

    n = 0
    for i in tqdm(range(len(df))):
        iret = df["ret"][i]
        idxs = np.round(iret["az"] / 0.2).astype(int)
        for (side, sfx), a in acc.items():
            count = iret[f"{side}count{sfx}"]
            np.add.at(a["count"], idxs, np.nan_to_num(count))
            for k in stats:
                np.add.at(a[k], idxs, np.nan_to_num(iret[f"{side}{k}{sfx}"] * count))
        n += 1
    logger.info(f"Number of observation that went into this plots = {n}")

    # NOTE: the "mc" quantities are not accumulated yet, so the
    # corresponding columns are 0 (NaN in empty bins).
    zeros = np.zeros(azbins.shape)
    # NOTE: the standard errors of both scan directions are divided by the
    # left-scan counts.
    err_counts = acc["l", ""]["count"]

    dfm = pd.DataFrame()
    dfm["az"] = azbins
    for side in "lr":
        a = acc[side, ""]
        counts = a["count"]
        dfm[f"{side}mc"] = zeros / counts * 1e3
        dfm[f"{side}fp"] = a["fp"] / counts * 1e3
        dfm[f"{side}az"] = a["aze"] / counts * 60

        dfm[f"{side}mcs"] = np.sqrt((zeros / counts - (zeros / counts) ** 2)) * 1e3
        dfm[f"{side}fps"] = (
            np.sqrt((a["fp_sqre"] / counts - (a["fp"] / counts) ** 2)) * 1e3
        )
        dfm[f"{side}azs"] = (
            np.sqrt((a["aze_sqre"] / counts - (a["aze"] / counts) ** 2)) * 60
        )

        dfm[f"{side}mce"] = dfm[f"{side}mcs"] / np.sqrt(err_counts)
        dfm[f"{side}fpe"] = dfm[f"{side}fps"] / np.sqrt(err_counts)
        dfm[f"{side}aze"] = dfm[f"{side}azs"] / np.sqrt(err_counts)

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
    hk_cfg,
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
    """Create azimuth-binned data for observations in a time range.

    The observation database in ``ctx`` is queried between the provided start
    and stop dates, then filtered by ``scan_type``. Each matching observation
    is passed to ``exe_azbin`` and the results are collected in a DataFrame.

    Args:
        ctx (sotodlib.core.Context): Context with an observation database.
        hk_cfg (dict): HK loading config, passed to :func:`load_az_temp`.
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
        tuple: ``(dfm, df)``, the azimuth-binned summary DataFrame from
            :func:`ana1` and the per-observation DataFrame.
    """

    scan_start = dtime.datetime(ys, ms, ds, 0, 0, 0, 0, tzinfo=dtime.timezone.utc)
    scan_stop = dtime.datetime(
        ye, me, de, date.hour, date.minute, 0, 0, tzinfo=dtime.timezone.utc
    )
    if tag is None:
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
            tags = ctx.obsdb.get(iobs["obs_id"], tags=True)["tags"]
            logger.info(f"{iobs['obs_id']}: {tags}")
            obss.append(iobs["obs_id"])

    rows = []
    for iobs in tqdm(obss):
        try:
            ret, _ = exe_azbin(
                ctx,
                iobs,
                hk_cfg,
                platform=platform,
            )
            dt = dtime.datetime.fromtimestamp(
                int(iobs.split("_")[1]), tz=dtime.timezone.utc
            )
            rows.append({"obsid": iobs, "dt": dt, "ret": ret})
        except Exception:
            logger.exception(f"exe_azbin failed for {iobs}")
    df = pd.DataFrame(rows, columns=["obsid", "dt", "ret"])
    dfm = ana1(df)
    return dfm, df


def get_fpa_metrics(df, setpoint=75):
    """Calculate FPA temperature metrics from azimuth-binned observation data.

    For each observation in ``df``, this takes the FPA temperature
    time-ordered data, fits a 3rd order polynomial to
    remove slow trends, and fits a Gaussian to the residual distribution.
    Also computes the RMS/std of the temperature, its deviation from
    ``setpoint``, and the left/right turnaround peak-to-peak amplitude.

    Args:
        df (pandas.DataFrame): Per-observation data, as returned by
            :func:`dataframe_maker`.
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
                logger.error(f"Fit failed for index {df['ret'][idx]}")
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
            logger.exception(f"get_fpa_metrics failed for index {idx}")
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
    elif platform == "satp2":
        std_threshold = [2.2e-1, 2.1e-1, 2.2e-1, 2.2e-1]
        r_p2p_threshold = [0.3, 0.35, 0.4, 0.4]
        l_p2p_threshold = [0.26, 0.35, 0.4, 0.35]
        r_sq_threshold = [0.8, 0.8, 0.8, 0.8]
    elif platform == "satp3":
        std_threshold = [3.5e-2, 4.5e-2, 4e-2, 4.5e-2]
        r_p2p_threshold = [0.06, 0.08, 0.06, 0.08]
        l_p2p_threshold = [0.1, 0.1, 0.13, 0.15]
        r_sq_threshold = [0.9, 0.9, 0.9, 0.9]
    else:
        raise ValueError(f"Unknown platform: {platform}")

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
                logger.debug(f"{obs}: std {std}, threshold {std_threshold[j]}")
                if std > std_threshold[j]:
                    logger.info(
                        f"Standard deviation of observation {obs} is over threshold value of {std_threshold[j]} mK at {round(std, 2)} mK."
                    )
                    logger.info(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_std[j] += 1
                if l_p2p > l_p2p_threshold[j]:
                    logger.info(
                        f"Left going turnaround of observation {obs} is over threshold value of {l_p2p_threshold[j]} mK at {round(l_p2p, 2)} mK."
                    )
                    logger.info(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_l_p2p[j] += 1
                if r_p2p > r_p2p_threshold[j]:
                    logger.info(
                        f"Right going turnaround of observation {obs} is over threshold value of {r_p2p_threshold[j]} mK at {round(r_p2p, 2)} mK."
                    )
                    logger.info(
                        f"Az range of obs {obs}: {round(min_az, 1), round(max_az, 1)}"
                    )
                    n_r_p2p[j] += 1

                # Gaussian thresholding (fit already computed by get_fpa_metrics)
                x_fit, y_fit, fwhm, variance = metric_dict[obs][
                    "gaussian_metrics"
                ].values()
                r_sq = metric_dict[obs]["r_squared"]

                if r_sq < r_sq_threshold[j]:
                    logger.info(
                        f"Gaussian R squared value of observation {obs} is less than the threshold value of {r_sq_threshold[j]} at {round(r_sq, 2)}."
                    )
                    logger.info(
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
                        if save_path is not None:
                            plt.savefig(os.path.join(save_path + f"/{obs}_gaussian.png"))
                        plt.close()
        logger.info(f"Number of Standard deviation flags: {n_std[j]}")
        logger.info(f"Number of right turnaround P2P flags: {n_r_p2p[j]}")
        logger.info(f"Number of left turnaround P2P flags: {n_l_p2p[j]}")
        logger.info(f"Number of goodness of fit flags: {n_gauss[j]}")

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

    Groups the observations in ``df`` into 4 azimuth bins and renders a
    polar plot showing: the scan azimuth range as a dashed circle, left/right
    turnaround FPA temperature vs. azimuth (blue/red traces, offset radially
    from the setpoint), the local horizon profile (from Cerro Toco survey
    data) and the LAT position, and the per-bin flag counts from
    :func:`metric_analyzer` (``cnt_dict``) annotated as text boxes.

    Args:
        cnt_dict (dict): Per-bin flag counts, as returned by
            :func:`metric_analyzer` (keys 'std_cnt', 'r_p2p_cnt',
            'l_p2p_cnt', 'gof_cnt', each a length-4 array).
        dfm (pandas.DataFrame): Azimuth-binned summary, as returned by
            :func:`ana1`.
        df (pandas.DataFrame): Per-observation data, as returned by
            :func:`dataframe_maker`.
        ms (int): Start month (used in the plot title).
        ds (int): Start day.
        me (int): End month.
        de (int): End day.
        platform (str, optional): Telescope platform; selects the
            temperature setpoint. Defaults to "satp1".
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

    # LAT position and top-of-LAT elevation as seen from the SATs (deg).
    lat_az = 169
    lat_el = 5

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
        logger.info(f"binned minazs: {binned_minazs}")
        logger.info(f"binned maxazs: {binned_maxazs}")

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
        logger.info(azidx)

    setpoint = PLATFORM_SETTINGS[platform]["setpoint"]

    # Number of points trimmed off each end of a turnaround trace before
    # drawing its direction arrow (avoids drawing the arrow through the
    # noisy edge of the trace).
    edge_trim = 50
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
                logger.info(f"az_circle_plotter: no observations in az bin {i}, skipping")
            continue

        iaz = dfm["az"][imask] + angle_offset
        iraz = dfm["rfp"][imask]
        ilaz = dfm["lfp"][imask]

        if debug:
            logger.info(iraz)
            logger.info(ilaz)

        irfl = ~np.isnan(iraz.values)
        raz = iaz.values[irfl]
        if raz.size > 0:
            rrad = ((iraz.values[irfl] - setpoint) + offset + roff) * magnify
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
            logger.info(f"az_circle_plotter: no valid right-turnaround data in az bin {i}")

        ilfl = ~np.isnan(ilaz.values)
        laz = iaz.values[ilfl]
        if laz.size > 0:
            lrad = ((ilaz.values[ilfl] - setpoint) + offset + loff) * magnify
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
            logger.info(f"az_circle_plotter: no valid left-turnaround data in az bin {i}")

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
    toco_rx = (TOCO_EL + ampoff) * np.cos((TOCO_AZ + angle_offset) * DEG) * ifac
    toco_ry = (TOCO_EL + ampoff) * np.sin((TOCO_AZ + angle_offset) * DEG) * ifac
    ax.plot(toco_rx, toco_ry, c=colors["orange"])
    lat_rx = (lat_el + ampoff) * np.cos((lat_az + angle_offset) * DEG) * ifac
    lat_ry = (lat_el + ampoff) * np.sin((lat_az + angle_offset) * DEG) * ifac
    ax.plot(lat_rx, lat_ry, "s", color="black")
    ax.text(
        -0.3,
        0.8,
        f"Orange: Horizon\nMax(EL of TOCO) is {np.max(TOCO_EL):.01f} at Az = {TOCO_AZ[np.argmax(TOCO_EL)]:.01f} deg",
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
    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"{me}_{de}_circle_plot.png"))


def get_parser(parser=None):
    """Build the argument parser for the FPA metrics pipeline.

    Args:
        parser (argparse.ArgumentParser, optional): Parser to add arguments
            to. A new one is created if None.

    Returns:
        argparse.ArgumentParser: The parser with FPA metrics arguments added.
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", help="Configuration yaml file.")
    parser.add_argument(
        "--verbose",
        default=2,
        type=int,
        help="Increase output verbosity. 0: Error, 1: Warning, 2: Info(default), 3: Debug",
    )
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
    parser.add_argument(
        "--circle_plt_save_path",
        type=str,
        default=None,
        help="Directory in which to save the az circle summary plot.",
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
    circle_plt_save_path: Optional[str] = None,
    scan_type: Optional[str] = "cmb",
    platform: Optional[str] = "satp1",
):
    """Run the full FPA metrics pipeline: bin, compute metrics, flag, and plot.

    Args:
        config (str): Path to the configuration yaml file. Must contain
            ``contexts.ctx_path1/2/3`` entries giving the context file for
            each platform, and an ``hk`` block (see :func:`load_az_temp`).
        verbose (int, optional): Logging verbosity. 0: Error, 1: Warning,
            2: Info, 3: Debug. Defaults to 2.
        ys (int, optional): Start year. Defaults to the current year.
        ms (int, optional): Start month. Defaults to the current month.
        ds (int, optional): Start day. Defaults to 1.
        ye (int, optional): End year. Defaults to the current year.
        me (int, optional): End month. Defaults to the current month.
        de (int, optional): End day. Defaults to the current day.
        tag (str, optional): Observation tag filter. Defaults to None.
        circle_plt_save_path (str, optional): Directory in which to save the
            az circle summary plot. Defaults to None.
        scan_type (str, optional): Observation subtype to select. Defaults
            to "cmb".
        platform (str, optional): Telescope platform ("satp1", "satp2" or
            "satp3"). Defaults to "satp1".
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

    with open(config) as f:
        configs = yaml.safe_load(f)

    settings = PLATFORM_SETTINGS[platform]
    ctx = core.Context(configs["contexts"][settings["ctx_key"]])
    setpoint = settings["setpoint"]
    dfm, df = dataframe_maker(
        ctx,
        hk_cfg=configs["hk"],
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
    logger.info("Made dataframes")

    has_obs = not df.empty
    if not has_obs:
        logger.info(
            f"No observations found for {platform} between "
            f"{ys}/{ms}/{ds} and {ye}/{me}/{de}."
        )
    logger.info("trying to compute metrics")
    metric_dict = get_fpa_metrics(df=df, setpoint=setpoint)
    logger.info("finished computing metrics")

    logger.info("trying to analyze metrics")
    cnt_ret = metric_analyzer(metric_dict, save_path=None, platform=platform)

    if has_obs:
        logger.info("trying to save circle plot")
        az_circle_plotter(
            cnt_dict=cnt_ret,
            dfm=dfm,
            df=df,
            ms=ms,
            ds=ds,
            me=me,
            de=de,
            platform=platform,
            save_path=circle_plt_save_path,
            show=False,
        )
    else:
        logger.info("Skipping az_circle_plotter since there are no observations to plot.")
    logger.info("finished analyzing metrics")


if __name__ == "__main__":
    main_launcher(main, get_parser)
