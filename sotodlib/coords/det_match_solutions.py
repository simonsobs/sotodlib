from sotodlib.core import Context, AxisManager, LabelAxis
from sotodlib.coords import det_match as dm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import yaml
from scipy import interpolate
from scipy.stats import theilslopes
from tqdm.auto import tqdm, trange
from copy import deepcopy

import h5py
import sys


@dataclass
class SolutionsCfg:
    """
    Parameters
    ----------
    ctx_path : str
        Path to context file to use to pull tod metadata.
    pointing_results_dir : str
        Results to directory that contains pointing results. Files in
        directory should look like::

            focal_plane_<obs_id>_<wafer_slot>.hdf

    results_dir : str
        Directory where results should be stored.
    wafer_info_path : str
        Path to the wafer_info h5 file.
    platform: str
        Telescope platform.
    wafer_slots: list
        Optional list of wafer slots to make solutions for.
    base_obs_id: str
        Obs_id to use as a base for matching when merging multiple pointing
        obs_ids for a wafer.  Will default to the pointing obs_id with the
        greatest number of detectors above the min_R2 threshold.
    zemax_path: str
        If running for the LAT, the path to the zemax file must be specified.
    apply_roll: bool
        Whether or not to apply the obs_id roll angle.  Some pointing sets
        may already be corrected for roll angle.
    pointing_field : str
        Name of sub axis manager in pointing tune axis maanger containng the
        pointing information.
    site_pipeline_cfg_dir : str
        Path to site-pipeline-config dir. Defaults to the env var
        ``$SITE_PIPELINE_CONFIG_DIR``.
    finite_xi_thresh : int
        Minimum number of dets a pointing result must have to add it to the
        analysis.
    min_r2 : float
        Minimum R-squared for det pointing to be considered.
    sel_rad : float
        Selection radius for grid-based interpolation pointing offset subtraction.
    unassigned_slots : int
        Number of additional "unassigned" node to use per-side
    wafer_map_path : str
        Path to the wafer map file. Defaults to ``<site-pipeline-config>/shared/detmatpping/wafer_map.yaml``.
    match_pars : dict
        Dictionary of match parameters to use for pointing obs_id merging and
        each match iteration.  Should have the form::

            match_pars:
                pointing:
                    freq_width: 0.4
                    dist_width: 2.0
                match0:
                    freq_width: 200
                    dist_width: 0.4
                match1:
                    freq_width: 50
                    dist_width: 0.8
                match2:
                    freq_width: 5
                    dist_width: 0.1

    initial_pointing_offset : Tuple[float, float]
        Estimated pointing offset for the boresight. This should be
        (xi_offset, eta_offset) where both are in degrees.
    ufm_to_fp_path : str
        Path to file that maps wafer_slot to position on focal plane.
    fp_to_rx_path: str
        Path to file that maps tube slot to position on focal plane.
    freq_correct_by_muxband: bool
        If true, apply the same freq offset correction to all resonators in a mux-band.
    match_plot_color_by: str
        Color the residual scatter plots by "north_south" or "det_type".
    initial_offset_search_x_range_args, initial_offset_search_y_range_args: tuple
        Arguments passed to np.arange to establish the allowed xi/eta ranges.
        Offsets are additional shifts in degrees relative to the configured
        initial_pointing_offset.
    initial_offset_search_levels: int
        Number of 2D interval-halving steps. Each step tests at most 9 points.
        Search minimizes Match 0 matching_cost with frequency effectively disabled.
    """

    ctx_path: str
    pointing_results_dir: str
    results_dir: str
    wafer_info_path: str
    platform: str
    wafer_slots: Optional[list[str]] = None
    base_obs_id: Optional[str] = None
    zemax_path: Optional[str] = None
    apply_roll: bool = True

    pointing_field: str = "tod_pointing"
    site_pipeline_cfg_dir: str = "$SITE_PIPELINE_CONFIG_DIR"
    finite_xi_thresh: int = (
        500  # Min number of dets with finite xi to consider a pointing input
    )
    min_r2: float = 0.9
    sel_rad: float = 2.0
    unassigned_slots: int = 1200
    wafer_map_path: Optional[str] = None
    match_pars: Dict[str, dict] = field(default_factory=lambda: defaultdict(dict))

    initial_pointing_offset: Tuple[float, float] = (0, 0)
    ufm_to_fp_path: Optional[str] = None
    fp_to_rx_path: Optional[str] = None
    imprinter_path: Optional[str] = None
    freq_correct_by_muxband: bool = True
    # Correction to DESIGN frequencies, fitted from matched OPTC only.
    # "linear" fits a per-N/S, per-mux-band offset and slope.
    # "median" retains a constant per-band correction for comparison.
    freq_correction_model: str = "linear"  # "linear" or "median"
    freq_fit_min_matches: int = 20
    freq_fit_min_span_mhz: float = 20.0
    freq_fit_clip_sigma: float = 3.0
    freq_fit_clip_floor_mhz: float = 2.0
    freq_fit_max_iterations: int = 6
    freq_fit_max_abs_slope: float = 0.25
    show_match_plots: bool = True  # Display interactive diagnostics after each match
    match_plot_color_by: str = "north_south"  # "north_south" or "det_type"

    # Search xi/eta before Match 0 using progressively halved 2D intervals.
    # np.arange args establish the bounds; the step is not used as the search
    # spacing. Frequency is effectively ignored during the search.
    search_initial_pointing_offset: bool = True
    initial_offset_search_x_range_args: Tuple[float, float, float] = (-0.06, 0.061, 0.01)
    initial_offset_search_y_range_args: Tuple[float, float, float] = (-0.06, 0.061, 0.01)
    initial_offset_search_levels: int = 6

    ctx: Context = field(init=False)
    wafer_map: Dict[str, dict] = field(init=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolutionsCfg":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "SolutionsCfg":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def __post_init__(self):
        if self.match_plot_color_by not in ("north_south", "det_type"):
            raise ValueError(
                "match_plot_color_by must be 'north_south' or 'det_type'"
            )
        if self.freq_correction_model not in ("linear", "median"):
            raise ValueError("freq_correction_model must be 'linear' or 'median'")
        if self.freq_fit_min_matches < 2:
            raise ValueError("freq_fit_min_matches must be >= 2")
        if self.freq_fit_min_span_mhz <= 0:
            raise ValueError("freq_fit_min_span_mhz must be positive")
        if self.freq_fit_clip_sigma <= 0 or self.freq_fit_clip_floor_mhz < 0:
            raise ValueError("frequency fit clipping settings must be nonnegative")
        if self.freq_fit_max_iterations < 1:
            raise ValueError("freq_fit_max_iterations must be >= 1")
        if not 0 < self.freq_fit_max_abs_slope < 1:
            raise ValueError("freq_fit_max_abs_slope must be between 0 and 1")
        if self.search_initial_pointing_offset:
            if self.initial_offset_search_levels < 1:
                raise ValueError("initial_offset_search_levels must be >= 1")
            for name, args in (
                ("initial_offset_search_x_range_args", self.initial_offset_search_x_range_args),
                ("initial_offset_search_y_range_args", self.initial_offset_search_y_range_args),
            ):
                if not 1 <= len(args) <= 3:
                    raise ValueError(f"{name} must contain 1 to 3 np.arange arguments")
                try:
                    values = np.arange(*args)
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise ValueError(f"Invalid {name}: {args}") from exc
                if not values.size or not np.isfinite(values).all():
                    raise ValueError(f"{name} must produce a nonempty, finite array")
        self.ctx = Context(self.ctx_path)

        if self.wafer_slots is None:
            self.wafer_slots = ["ws.", "ws0", "ws1", "ws2", "ws3", "ws4", "ws5", "ws6"]

        if not os.path.exists(self.results_dir):
            if os.path.exists(os.path.split(self.results_dir)[0]):
                os.makedirs(self.results_dir)
            else:
                raise FileNotFoundError(
                    f"Could not find results dir or basedir: {self.results_dir}"
                )

        self.site_pipeline_cfg_dir = os.path.expandvars(self.site_pipeline_cfg_dir)
        if self.wafer_map_path is None:
            self.wafer_map_path = os.path.join(
                self.site_pipeline_cfg_dir, "shared/detmapping/wafer_map.yaml"
            )
        with open(self.wafer_map_path, "r") as f:
            self.wafer_map = yaml.safe_load(f)

        if self.ufm_to_fp_path is None:
            self.ufm_to_fp_path = os.path.join(
                self.site_pipeline_cfg_dir, "shared/focalplane/ufm_to_fp.yaml"
            )
        if self.fp_to_rx_path is None:
            self.fp_to_rx_path = os.path.join(
                self.site_pipeline_cfg_dir, "shared/focalplane/optics_tubes.yaml"
            )
        if self.imprinter_path is None:
            self.imprinter_path = os.path.join(
                self.site_pipeline_cfg_dir, f"{self.platform}/imprinter.yaml"
            )


@dataclass
class PointingInfo:
    pointing: np.ndarray
    obs_id: str
    obs: dict
    meta: AxisManager
    preprocessed: bool = False


_meta_cache = {}
def get_meta(cfg: SolutionsCfg, obs_id: str, wafer_slot: Optional[str] = None):
    if obs_id in _meta_cache:
        meta = _meta_cache[obs_id]
    else:
        meta = cfg.ctx.get_meta(obs_id)
        _meta_cache[obs_id] = meta
    meta = deepcopy(meta)
    if wafer_slot is not None:
        meta.restrict("dets", meta.det_info.wafer_slot == wafer_slot)
    return meta


def load_good_pointing_info(cfg: SolutionsCfg, wafer_slot: str) -> List[PointingInfo]:
    """
    Load pointing data for each pointing measurement on disk
    """
    files = []
    for f in os.listdir(cfg.pointing_results_dir):
        if os.path.splitext(f)[0].split("_")[-1] == wafer_slot:
            files.append(os.path.join(cfg.pointing_results_dir, f))

    pointing_info = []
    for f in tqdm(files):
        d = h5py.File(f)["focal_plane"]
        if np.sum(np.isfinite(d["xi"])) < cfg.finite_xi_thresh:
            continue
        obs_id = "_".join(os.path.basename(f).split("_")[2:-1])
        obs = cfg.ctx.obsdb.get(obs_id)
        pinfo = PointingInfo(
            pointing=d,
            obs_id=obs_id,
            obs=obs,
            meta=get_meta(cfg, obs_id, wafer_slot=wafer_slot),
        )
        pointing_info.append(pinfo)

    return pointing_info


def pointing_preprocess(cfg: SolutionsCfg, pinfo: PointingInfo):
    """
    Add tod_pointing to PointingInfo metadata, adjusting for boresight angle and
    pointing offset.
    """
    meta: AxisManager = pinfo.meta
    assert (meta.det_info.readout_id == pinfo.pointing["dets:readout_id"].astype(str)).all()

    tod_pointing = AxisManager(meta.dets)

    _xi = deepcopy(pinfo.pointing["xi"])
    _eta = deepcopy(pinfo.pointing["eta"])
    theta = 0
    offset = cfg.initial_pointing_offset

    _xi += np.radians(offset[0])
    _eta += np.radians(offset[1])
    xi = _xi * np.cos(theta) - _eta * np.sin(theta)
    eta = _xi * np.sin(theta) + _eta * np.cos(theta)

    tod_pointing.wrap("xi", xi)
    tod_pointing.wrap("eta", eta)
    tod_pointing.wrap("r2", pinfo.pointing["R2"])
    tod_pointing.xi[tod_pointing.r2 < cfg.min_r2] = np.nan
    tod_pointing.eta[tod_pointing.r2 < cfg.min_r2] = np.nan

    if 'tod_pointing' in meta:
        meta.move('tod_pointing', None)

    meta.wrap("tod_pointing", tod_pointing)
    return meta


def merge_pointing_info(cfg: SolutionsCfg, pinfos: List[PointingInfo], base_idx=0):
    """
    Combine all pointing measurements into a single resonator set, with the
    median pointing info from all. This requires a base_idx to be specified,
    which will be the index of the PointingInfo to use to create the ResSet
    template.  For all other PointingInfo objects, resonators will be matched to
    the base resset based on resonance frequency, to compile all pointing
    measurements for a given detector. The median of all measurements will be
    used as the real value.
    """
    for pinfo in pinfos:
        pointing_preprocess(cfg, pinfo)

    meta = pinfos[base_idx].meta
    stream_id = meta.det_info.stream_id[0]
    wafer_slot = meta.det_info.wafer_slot[0]

    base_resset = dm.ResSet.from_aman(
        meta,
        stream_id=stream_id,
        pointing=meta.tod_pointing,
    )

    pointing_map = {
        r.idx: [(r.xi, r.eta)] for r in base_resset
    }

    match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["pointing"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["pointing"]["dist_width"]),
    )

    for i in range(len(pinfos)):
        if i == base_idx:
            continue
        meta = pinfos[i].meta
        src = dm.ResSet.from_aman(
            meta,
            stream_id=stream_id,
            pointing=meta.tod_pointing,
        )
        dst = base_resset
        match = dm.Match(src, dst, match_pars=match_pars)
        for rsrc, rdst in match.get_match_iter(include_unmatched=False):
            pointing_map[rdst.idx].append((rsrc.xi, rsrc.eta))

    for r in base_resset:
        r.xi, r.eta = np.nanmedian(np.array(pointing_map[r.idx]).T, axis=1)

    return base_resset, pointing_map


def get_best_tod_pointing(cfg: SolutionsCfg, pinfos: List[PointingInfo]) -> AxisManager:
    _readout_ids = pinfos[0].pointing["dets:readout_id"]

    readout_ids = list(map(lambda bs: bs.decode(), _readout_ids))
    dets = LabelAxis("dets", readout_ids)
    ndets = dets.count

    for pinfo in pinfos:  # Shift and rotate xi/eta per pointing observation
        _xi = pinfo.pointing["xi"] + cfg.initial_pointing_offset[0]
        _eta = pinfo.pointing["eta"] + cfg.initial_pointing_offset[1]
        theta = np.deg2rad(pinfo.obs["roll_center"]) if cfg.apply_roll else 0.
        pinfo.xi = _xi * np.cos(theta) - _eta * np.sin(theta)
        pinfo.eta = _xi * np.sin(theta) + _eta * np.cos(theta)

    xis = np.full(ndets, np.nan)
    etas = np.full(ndets, np.nan)

    for i in trange(len(readout_ids)):  # Find optimal xi/eta per readout channel
        _xis = np.full(len(pinfos), np.nan)
        _etas = np.full(len(pinfos), np.nan)
        _r2s = np.full(len(pinfos), np.nan)
        for j, pi in enumerate(pinfos):
            rc = np.where(pi.pointing["dets:readout_id"] == readout_ids[i].encode())[0]
            if not rc:
                continue
            rc = rc[0]
            _xis[j] = pi.xi[rc]
            _etas[j] = pi.eta[rc]
            _r2s[j] = pi.pointing["R2"][rc]
        xis[i] = np.nanmean(_xis[_r2s > cfg.min_r2])
        etas[i] = np.nanmean(_etas[_r2s > cfg.min_r2])

    tod_pointing = AxisManager(dets)
    tod_pointing.wrap("xi", xis, [(0, "dets")])
    tod_pointing.wrap("eta", etas, [(0, "dets")])

    return tod_pointing


@dataclass
class MatchSolution:
    match: dm.Match
    am: AxisManager
    match_iterations: List[dm.Match] = field(default_factory=list)
    # Effective starting offset in degrees, including the grid-search adjustment.
    best_initial_pointing_offset: Optional[Tuple[float, float]] = None
    initial_offset_search: Optional[dict] = None


def _optical_pointing_pairs(match):
    """Yield matched OPTC design detectors with finite pointing on both sides.

    Only these pairs are permitted to determine pointing corrections.
    The resulting corrections are still applied to all measured resonators.
    """
    for rsrc, rdst in match.get_match_iter(include_unmatched=False):
        if _det_type_label(rdst.det_type) != "OPTC":
            continue
        if not np.isfinite([rsrc.xi, rsrc.eta, rdst.xi, rdst.eta]).all():
            continue
        yield rsrc, rdst


def get_pt_offset_interp(match, sel_rad=np.deg2rad(2)) -> Tuple[Any, Any]:
    """Interpolate measured minus design pointing offsets using OPTC pairs only.

    Other detector types never contribute to the interpolation, but the
    returned interpolators can be evaluated for any measured resonator with
    finite xi/eta. Empty local neighborhoods fall back to the median optical
    offset, so they do not propagate NaNs into other detectors' pointing.
    """
    if not np.isfinite(sel_rad) or sel_rad <= 0:
        raise ValueError("sel_rad must be finite and positive")

    pairs = list(_optical_pointing_pairs(match))
    if not pairs:
        raise ValueError(
            "Cannot estimate spatial pointing offsets: no matched OPTC "
            "detectors have finite measured and design xi/eta"
        )

    xis = np.array([rsrc.xi for rsrc, _ in pairs], dtype=float)
    etas = np.array([rsrc.eta for rsrc, _ in pairs], dtype=float)
    dxis = np.array([rsrc.xi - rdst.xi for rsrc, rdst in pairs], dtype=float)
    detas = np.array([rsrc.eta - rdst.eta for rsrc, rdst in pairs], dtype=float)

    step = sel_rad / 2
    xi_list = np.arange(np.min(xis), np.max(xis) + step, step)
    eta_list = np.arange(np.min(etas), np.max(etas) + step, step)
    if xi_list.size < 2:
        xi_list = np.array([xis[0] - step / 2, xis[0] + step / 2])
    if eta_list.size < 2:
        eta_list = np.array([etas[0] - step / 2, etas[0] + step / 2])

    xi_grid, eta_grid = np.meshgrid(xi_list, eta_list)
    dxi_data = np.empty_like(xi_grid)
    deta_data = np.empty_like(eta_grid)
    default_dxi = np.median(dxis)
    default_deta = np.median(detas)
    n_fallback = 0

    for i, j in np.ndindex(xi_grid.shape):
        select = np.hypot(
            xis - xi_grid[i, j], etas - eta_grid[i, j]
        ) < sel_rad
        if np.any(select):
            dxi_data[i, j] = np.median(dxis[select])
            deta_data[i, j] = np.median(detas[select])
        else:
            dxi_data[i, j] = default_dxi
            deta_data[i, j] = default_deta
            n_fallback += 1

    dxi_interp = interpolate.RegularGridInterpolator(
        (xi_list, eta_list), dxi_data.T, bounds_error=False, fill_value=None
    )
    deta_interp = interpolate.RegularGridInterpolator(
        (xi_list, eta_list), deta_data.T, bounds_error=False, fill_value=None
    )
    return dxi_interp, deta_interp


def get_foffset_interp(
    match, is_north, box_size=50, box_step=25
) -> Callable[[float], float]:
    df, f, is_norths = [], [], []
    for r1, r2 in match.get_match_iter(include_unmatched=False):
        df.append(r1.res_freq - r2.res_freq)
        f.append(r1.res_freq)
        is_norths.append(r1.is_north)
    df_arr = np.array(df)
    f_arr = np.array(f)
    is_north_arr = np.array(is_norths, dtype=bool)

    f0, f1 = np.min(f), np.max(f)
    df_meds = []
    fcs = []
    for fc in np.arange(f0, f1, box_step):
        sel = (f > fc - box_size / 2) & (f < fc + box_size / 2)
        sel &= is_north_arr == is_north
        df_meds.append(np.nanmedian(df_arr[sel]))
        fcs.append(fc)

    # Create interpolation
    f_func = interpolate.interp1d(fcs, df_meds, fill_value="extrapolate")
    return f_func


@dataclass
class MatchSolutionResult:
    results: Dict[str, Optional[MatchSolution]]
    am: Optional[AxisManager] = None
    traceback: Optional[str] = None


# Consistent colors across iterations and across the two diagnostic figures.
MATCH_PLOT_COLORS = {
    "north_south": {
        "South": "tab:blue",
        "North": "tab:orange",
    },
    "det_type": {
        "OPTC": "tab:blue",
        "DARK": "tab:orange",
        "SQID": "tab:green",
        "BARE": "tab:red",
        "UNRT": "tab:purple",
        "SLOT": "tab:brown",
        "NC": "tab:gray",
        "UNMATCHED": "0.7",
        "UNKNOWN": "tab:pink",
    },
}


def _det_type_label(value):
    """Normalize detector-type labels, including bytes from HDF5."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().upper() or "UNKNOWN"


def _plot_label(resonator, color_by):
    if color_by == "north_south":
        return "North" if resonator.is_north else "South"
    if color_by == "det_type":
        return _det_type_label(resonator.det_type)
    raise ValueError(f"Invalid color_by: {color_by}")


def _plot_color(label, color_by):
    return MATCH_PLOT_COLORS[color_by].get(label, "tab:cyan")


def _plot_groups(ax, xs, ys, labels, valid, color_by, **scatter_kwargs):
    """Scatter finite values by category, retaining a category legend."""
    labels = np.asarray(labels)
    for label in sorted(set(labels[valid])):
        sel = valid & (labels == label)
        ax.scatter(
            xs[sel], ys[sel],
            color=_plot_color(label, color_by),
            label=label,
            **scatter_kwargs,
        )



def _fit_mux_frequency_offset(measured, design, cfg):
    """Fit design - measured = offset + slope*(measured - f0).

    Uses an initial Theil-Sen fit and iteratively sigma-clipped least squares
    on matched OPTC pairs. Returns None for sparse or unreliable groups.
    """
    measured = np.asarray(measured, dtype=float)
    design = np.asarray(design, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(design)
    if valid.sum() < cfg.freq_fit_min_matches:
        return None

    f_meas = measured[valid]
    residual = (design - measured)[valid]
    span = np.ptp(f_meas)
    if cfg.freq_correction_model == "linear" and span < cfg.freq_fit_min_span_mhz:
        return None

    f0 = float(np.median(f_meas))
    x = f_meas - f0

    if cfg.freq_correction_model == "median":
        offset = float(np.median(residual))
        return dict(f0=f0, offset=offset, slope=0.0,
                    inliers=np.ones(len(x), dtype=bool))

    try:
        slope, offset, _, _ = theilslopes(residual, x)
    except (ValueError, FloatingPointError) as exc:
        return None
    if not np.isfinite([slope, offset]).all():
        return None

    inliers = np.ones(len(x), dtype=bool)
    for _ in range(cfg.freq_fit_max_iterations):
        fit_residual = residual - (offset + slope * x)
        center = np.median(fit_residual[inliers])
        mad = np.median(np.abs(fit_residual[inliers] - center))
        sigma = 1.4826 * mad
        threshold = max(cfg.freq_fit_clip_floor_mhz,
                        cfg.freq_fit_clip_sigma * sigma)
        new_inliers = np.abs(fit_residual - center) <= threshold
        if new_inliers.sum() < cfg.freq_fit_min_matches:
            return None
        if np.ptp(x[new_inliers]) < cfg.freq_fit_min_span_mhz:
            return None
        new_slope, new_offset = np.polyfit(x[new_inliers], residual[new_inliers], 1)
        converged = np.array_equal(new_inliers, inliers)
        inliers = new_inliers
        slope, offset = float(new_slope), float(new_offset)
        if converged:
            break

    if not np.isfinite([offset, slope]).all():
        return None
    if abs(slope) > cfg.freq_fit_max_abs_slope:
        return None
    if 1 + slope <= 0:
        return None

    return dict(f0=f0, offset=offset, slope=slope, inliers=inliers)


def _correct_design_frequency(f_design, fit):
    """Invert the fitted design(measured) relation to find measured frequency.

    design = measured + offset + slope*(measured - f0)
    => measured = f0 + (design - f0 - offset)/(1 + slope).
    This exact inversion matters when fitting a nonzero slope.
    """
    return fit["f0"] + (
        np.asarray(f_design) - fit["f0"] - fit["offset"]
    ) / (1.0 + fit["slope"])


def apply_muxband_frequency_correction(match, cfg, wafer_slot=""):
    """Fit on matched OPTC only; correct ALL design detector types per group.

    Does not rematch, so before/after diagnostics refer to the exact same
    original pairs. ``match._match()`` is called separately in Match 1.
    """
    groups = defaultdict(list)
    for src, dst in match.get_match_iter(include_unmatched=False):
        if _det_type_label(dst.det_type) != "OPTC":
            continue
        if dst.mux_band is None or int(dst.mux_band) < 0:
            continue
        if np.isfinite([src.res_freq, dst.res_freq]).all():
            groups[(bool(dst.is_north), int(dst.mux_band))].append(
                (float(src.res_freq), float(dst.res_freq))
            )

    # Include groups with no valid OPTC pair in the report, but don't alter them.
    dst_groups = defaultdict(list)
    for dst in match.dst:
        if dst.mux_band is None or int(dst.mux_band) < 0:
            continue
        dst_groups[(bool(dst.is_north), int(dst.mux_band))].append(dst)

    diagnostics = []
    for (north, mb), dests in sorted(dst_groups.items()):
        pairs = groups.get((north, mb), [])
        f_meas = np.array([x for x, _ in pairs], dtype=float)
        f_design = np.array([y for _, y in pairs], dtype=float)
        fit = _fit_mux_frequency_offset(f_meas, f_design, cfg)
        if fit is None:
            if len(pairs):
                diagnostics.append(dict(north=north, mux_band=mb,
                                        measured=f_meas, before=f_design - f_meas,
                                        after=f_design - f_meas, fit=None))
            continue

        # Apply to ALL detector types, not just OPTC.
        for dst in dests:
            if np.isfinite(dst.res_freq):
                dst.res_freq = float(_correct_design_frequency(dst.res_freq, fit))

        after = _correct_design_frequency(f_design, fit) - f_meas
        before = f_design - f_meas
        n_inliers = int(fit["inliers"].sum())

        diagnostics.append(dict(north=north, mux_band=mb,
                                measured=f_meas, before=before, after=after,
                                fit=fit))

    return diagnostics


def plot_muxband_frequency_correction(
    diagnostics,
    wafer_slot="",
    zoom_percentile=98.0,
):
    """Show same matched OPTC pairs before/after frequency correction.

    The BEFORE panel overlays conspicuous, outlined fitted lines on muted
    scatter points. Lines are drawn *after* all points so they cannot be
    obscured. Colors encode mux band; line style/markers encode N/S.

    Parameters
    ----------
    zoom_percentile : float or None
        If a float between 50 and 100, use the central percentile range of
        residuals for BOTH panels' y limits, clipping extreme outliers only
        in the display. Use None to show all residuals. This does not affect
        fitting or the frequency correction.
    """
    if not diagnostics:
        return

    fig, axs = plt.subplots(
        1, 2, figsize=(17, 6.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab20")
    n_fits = 0
    all_residuals = []

    # Draw scatter points first; intentionally subtle so the lines stand out.
    for record in diagnostics:
        x = np.asarray(record["measured"], dtype=float)
        mb = record["mux_band"]
        north = record["north"]
        color = cmap(mb % 20)
        marker = "^" if north else "o"

        for ax, name in zip(axs, ("before", "after")):
            y = np.asarray(record[name], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                all_residuals.append(y[valid])
                ax.scatter(
                    x[valid], y[valid], color=color, s=10, alpha=0.23,
                    marker=marker, linewidths=0, rasterized=True, zorder=2,
                )

    # Draw ALL fit lines last, with a thick contrasting outline above points.
    for record in diagnostics:
        fit = record["fit"]
        x = np.asarray(record["measured"], dtype=float)
        if fit is None or x.size < 2:
            continue

        # Show line only where optical inliers constrain the fit.
        inliers = np.asarray(fit["inliers"], dtype=bool)
        valid = np.isfinite(x) & inliers
        if valid.sum() < 2:
            continue
        xx = np.linspace(np.min(x[valid]), np.max(x[valid]), 200)
        yy = fit["offset"] + fit["slope"] * (xx - fit["f0"])

        color = cmap(record["mux_band"] % 20)
        north = record["north"]
        line, = axs[0].plot(
            xx, yy, color=color, lw=3.2,
            ls="-" if north else "--", alpha=1.0, zorder=20,
        )
        line.set_path_effects([
            pe.Stroke(linewidth=6.5, foreground="white"),
            pe.Normal(),
        ])
        n_fits += 1

    if all_residuals and zoom_percentile is not None:
        if not 50 < zoom_percentile < 100:
            raise ValueError("zoom_percentile must be between 50 and 100, or None")
        residuals = np.concatenate(all_residuals)
        lo, hi = np.percentile(residuals, [100 - zoom_percentile, zoom_percentile])
        lo, hi = min(lo, 0), max(hi, 0)
        if hi > lo:
            pad = 0.12 * (hi - lo)
            axs[0].set_ylim(lo - pad, hi + pad)

    for ax, label in zip(axs, ("Before", "After")):
        ax.axhline(0, color="0.45", lw=1, zorder=1)
        ax.set_xlabel("Measured resonance frequency (MHz)")
        ax.set_title(label)
        ax.grid(alpha=0.2, zorder=0)
    axs[0].set_ylabel("Design - measured (MHz)")

    # Small style legend: line colors are the corresponding mux-band colors.
    fit_handles = [
        Line2D([0], [0], color="0.2", lw=3, ls="-", label="North"),
        Line2D([0], [0], color="0.2", lw=3, ls="--", label="South"),
    ]
    axs[0].legend(handles=fit_handles, loc="upper left", fontsize=9)

    fig.suptitle(f"{wafer_slot} | Frequency fit | {n_fits} groups")
    if n_fits == 0:
        axs[0].text(
            0.5, 0.95, "No accepted fits",
            transform=axs[0].transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    plt.show()

def summarize_optical_frequency_slopes(match, stage, min_matches=20):
    """Report per-group frequency residual slopes AFTER assignments are recomputed."""
    groups = defaultdict(list)
    for src, dst in match.get_match_iter(include_unmatched=False):
        if _det_type_label(dst.det_type) != "OPTC" or dst.mux_band < 0:
            continue
        if np.isfinite([src.res_freq, dst.res_freq]).all():
            groups[(bool(dst.is_north), int(dst.mux_band))].append(
                (float(src.res_freq), float(dst.res_freq - src.res_freq))
            )
    for (north, mb), pairs in sorted(groups.items()):
        if len(pairs) < min_matches:
            continue
        xy = np.asarray(pairs)
        f, df = xy[:, 0], xy[:, 1]
        slope = np.polyfit(f - np.median(f), df, 1)[0]


def plot_match_offsets(
    match: dm.Match,
    stage: str,
    wafer_slot: str,
    bins: int = 50,
    color_by: str = "north_south",
):
    """Display pointing and frequency residuals, design minus measured.

    Scatter colors indicate N/S or the matched *design* detector type.
    Histograms remain combined across categories.
    """
    pairs = list(match.get_match_iter(include_unmatched=False))
    if not pairs:
        print(f"{wafer_slot} {stage}: no matched resonators to plot", flush=True)
        return

    dxi = np.rad2deg(np.array([r2.xi - r1.xi for r1, r2 in pairs])) * 3600
    deta = np.rad2deg(np.array([r2.eta - r1.eta for r1, r2 in pairs])) * 3600
    df = np.array([r2.res_freq - r1.res_freq for r1, r2 in pairs], dtype=float)
    fsrc = np.array([r1.res_freq for r1, _ in pairs], dtype=float)
    labels = np.array([
        _plot_label(r1 if color_by == "north_south" else r2, color_by)
        for r1, r2 in pairs
    ])

    valid_xy = np.isfinite(dxi) & np.isfinite(deta)
    valid_f = np.isfinite(fsrc) & np.isfinite(df)
    legend_title = "N/S" if color_by == "north_south" else "Detector type"

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axs[0, 0]
    _plot_groups(
        ax, dxi, deta, labels, valid_xy, color_by,
        s=9, alpha=0.45, rasterized=True,
    )
    ax.axvline(0, color="0.5", lw=1)
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel(r"$\Delta\xi$ (design - measured) [arcsec]")
    ax.set_ylabel(r"$\Delta\eta$ (design - measured) [arcsec]")
    if valid_xy.any():
        ax.set_title(f"Pointing (N={valid_xy.sum()})")
        ax.legend(title=legend_title, fontsize=8)
    else:
        ax.set_title("Pointing")

    ax = axs[0, 1]
    for values, label, color in (
        (dxi, r"$\Delta\xi$", "tab:blue"),
        (deta, r"$\Delta\eta$", "tab:orange"),
    ):
        selected = values[np.isfinite(values)]
        if selected.size:
            ax.hist(
                selected, bins=bins, histtype="step", lw=1.5,
                color=color, label=label,
            )
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel("Pointing offset (arcsec)")
    ax.set_ylabel("Matched detectors")
    if valid_xy.any():
        ax.legend()
    ax.set_title("Pointing histogram")

    ax = axs[1, 0]
    _plot_groups(
        ax, fsrc, df, labels, valid_f, color_by,
        s=9, alpha=0.45, rasterized=True,
    )
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel("Measured resonance frequency (MHz)")
    ax.set_ylabel(r"$\Delta f$ (design - measured) [MHz]")
    if valid_f.any():
        ax.legend(title=legend_title, fontsize=8)
        ax.set_title(f"Frequency (N={valid_f.sum()})")
    else:
        ax.set_title("Frequency")

    ax = axs[1, 1]
    if valid_f.any():
        ax.hist(
            df[valid_f], bins=bins, histtype="stepfilled",
            alpha=0.55, color="tab:green",
        )
        ax.set_title(f"Frequency histogram (median {np.median(df[valid_f]):+.1f} MHz)")
    else:
        ax.set_title("Frequency histogram")
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel(r"$\Delta f$ (design - measured) [MHz]")
    ax.set_ylabel("Matched detectors")

    for ax in axs.flat:
        ax.grid(alpha=0.2)

    fig.suptitle(f"{wafer_slot} | {stage}")
    plt.show()


def _design_band(resonator):
    """Optical frequency band of a design resonator (e.g. f090, f150)."""
    band = getattr(resonator, "det_bandpass", "")
    if isinstance(band, bytes):
        band = band.decode("utf-8", errors="replace")
    band = str(band).strip()
    if band and band.upper() not in {"NC", "NONE", "NAN", "UNKNOWN"}:
        return band

    # Older wafer-info records may have det_freq but no bandpass.
    freq = getattr(resonator, "det_freq", 0)
    try:
        freq = float(freq)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(freq) or freq <= 0:
        return None
    return f"f{int(round(freq)):03d}"


def plot_match_xieta(
    match: dm.Match,
    stage: str,
    wafer_slot: str,
    color_by: str = "north_south",
):
    """Side-by-side xi/eta matching plots, one panel per design band.

    Matched measured positions are labeled Pointing and shown in blue;
    design positions are orange. All bands use the same colors. The
    color_by argument is retained for compatibility with existing calls.
    """
    bands = sorted({
        band for r in match.dst
        if (band := _design_band(r)) is not None
    })
    if not bands:
        return

    matched_by_band = defaultdict(list)
    for rsrc, rdst in match.get_match_iter(include_unmatched=False):
        band = _design_band(rdst)
        if band is not None:
            matched_by_band[band].append((rsrc, rdst))

    pointing_color = "tab:blue"
    design_color = "tab:orange"
    link_color = "0.2"

    nband = len(bands)
    fig, axs = plt.subplots(
        1, nband,
        figsize=(max(5.5 * nband, 8.0), 5.0),
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs)

    for ax, band in zip(axs, bands):
        design = [r for r in match.dst if _design_band(r) == band]
        pairs = matched_by_band[band]
        measured = [rsrc for rsrc, _ in pairs]

        src_xy = np.rad2deg(np.array(
            [(r.xi, r.eta) for r in measured], dtype=float
        ).reshape(-1, 2))
        dst_xy = np.rad2deg(np.array(
            [(r.xi, r.eta) for r in design], dtype=float
        ).reshape(-1, 2))
        src_valid = np.isfinite(src_xy).all(axis=1)
        dst_valid = np.isfinite(dst_xy).all(axis=1)

        segments = []
        for rsrc, rdst in pairs:
            xy = np.rad2deg(np.array(
                [[rdst.xi, rdst.eta], [rsrc.xi, rsrc.eta]], dtype=float
            ))
            if np.isfinite(xy).all():
                segments.append(xy)

        if segments:
            lines = LineCollection(
                segments, colors=link_color, linewidths=1.8,
                alpha=0.8, zorder=3, rasterized=True,
            )
            lines.set_path_effects([
                pe.Stroke(linewidth=3.6, foreground="white"),
                pe.Normal(),
            ])
            ax.add_collection(lines)

        ax.scatter(
            src_xy[src_valid, 0], src_xy[src_valid, 1],
            s=16, marker="o", alpha=0.85, linewidths=0,
            color=pointing_color, rasterized=True, zorder=5,
        )
        ax.scatter(
            dst_xy[dst_valid, 0], dst_xy[dst_valid, 1],
            s=28, marker="+", alpha=0.95, linewidths=1.4,
            color=design_color, rasterized=True, zorder=6,
        )

        ax.legend(
            handles=[
                Line2D([0], [0], linestyle="None", marker="o",
                       color=pointing_color, markersize=4.5,
                       label=f"Pointing ({src_valid.sum()})"),
                Line2D([0], [0], linestyle="None", marker="+",
                       color=design_color, markersize=6.5,
                       label=f"Design ({dst_valid.sum()})"),
            ],
            loc="upper right", fontsize=8,
        )

        ax.set_title(band)
        ax.set_xlabel("xi (deg)")
        if ax is axs[0]:
            ax.set_ylabel("eta (deg)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)

    fig.suptitle(f"{wafer_slot} | {stage}")
    plt.show()



def search_best_initial_pointing_offset(src, dst, match_pars, cfg, wafer_slot=""):
    """Search for the best initial xi/eta shift by 2D interval halving.

    Evaluate a 3x3 stencil at each level, recenter on the lowest matching
    cost, and halve the stencil spacing. Frequencies are effectively ignored
    in the search (very large freq_width), but the final match uses the
    original match_pars. All trials use the same unshifted src coordinates.

    This is a local, coarse-to-fine optimizer, not a strict binary search:
    matching cost need not be monotonic or unimodal.
    """
    x_range = np.asarray(np.arange(*cfg.initial_offset_search_x_range_args), dtype=float)
    y_range = np.asarray(np.arange(*cfg.initial_offset_search_y_range_args), dtype=float)
    xmin, xmax = float(x_range.min()), float(x_range.max())
    ymin, ymax = float(y_range.min()), float(y_range.max())

    search_pars = deepcopy(match_pars)
    search_pars.freq_width = 1e12  # MHz: neutralize frequency cost during scan
    match = dm.Match(
        deepcopy(src), deepcopy(dst), match_pars=search_pars,
        apply_dst_pointing=False,
    )

    original_xy = np.asarray([(r.xi, r.eta) for r in match.src], dtype=float)
    valid_xy = np.isfinite(original_xy).all(axis=1)
    valid_indices = np.flatnonzero(valid_xy)

    def set_shift(dx_deg, dy_deg):
        dxi, deta = np.deg2rad([dx_deg, dy_deg])
        for i in valid_indices:
            match.src[i].xi = original_xy[i, 0] + dxi
            match.src[i].eta = original_xy[i, 1] + deta

    cache = {}

    def evaluate(dx_deg, dy_deg):
        # Cache duplicate samples between levels, including the zero baseline.
        key = (round(float(dx_deg), 12), round(float(dy_deg), 12))
        if key not in cache:
            set_shift(*key)
            match._match()
            cache[key] = float(match.matching_cost)
        return cache[key]

    def better(cost, dx, dy, best_cost, best_dx, best_dy):
        if not np.isfinite(cost):
            return False
        if not np.isfinite(best_cost):
            return True
        if np.isclose(cost, best_cost, rtol=1e-12, atol=1e-9):
            return np.hypot(dx, dy) < np.hypot(best_dx, best_dy)
        return cost < best_cost

    baseline_cost = evaluate(0.0, 0.0)
    best_cost = baseline_cost
    best_dx, best_dy = 0.0, 0.0

    # Start at the middle of the allowed rectangle. Retain the zero-offset
    # baseline as a separate candidate, including when outside the rectangle.
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    sx, sy = (xmax - xmin) / 2, (ymax - ymin) / 2
    levels = []

    for level in range(cfg.initial_offset_search_levels):
        xs = np.unique(np.clip([cx - sx, cx, cx + sx], xmin, xmax))
        ys = np.unique(np.clip([cy - sy, cy, cy + sy], ymin, ymax))
        costs = np.empty((len(ys), len(xs)), dtype=float)

        for iy, dy in enumerate(ys):
            for ix, dx in enumerate(xs):
                cost = evaluate(dx, dy)
                costs[iy, ix] = cost
                if better(cost, dx, dy, best_cost, best_dx, best_dy):
                    best_dx, best_dy = float(dx), float(dy)
                    best_cost = cost

        if not np.isfinite(best_cost):
            raise ValueError(f"{wafer_slot}: no finite pointing matching cost")

        levels.append({
            "x": xs.copy(), "y": ys.copy(), "costs": costs.copy(),
            "best_delta_deg": (best_dx, best_dy),
            "best_cost": best_cost,
            "step_deg": (sx, sy),
        })
        print(
            f"{wafer_slot} pointing search {level + 1}/{cfg.initial_offset_search_levels}: "
            f"best=({best_dx:+.6f}, {best_dy:+.6f}) deg, "
            f"cost={best_cost:.2f}, step=({sx:.6f}, {sy:.6f}) deg, "
            f"evaluated={len(cache)}",
            flush=True,
        )

        # If zero wins and is outside the configured range, project the center
        # onto the allowed box. Otherwise refine around the best offset.
        cx = float(np.clip(best_dx, xmin, xmax))
        cy = float(np.clip(best_dy, ymin, ymax))
        sx *= 0.5
        sy *= 0.5

    # Run Match 0 with its original frequency width and every other parameter.
    set_shift(best_dx, best_dy)
    match.match_pars = deepcopy(match_pars)
    match._match()

    base_x, base_y = cfg.initial_pointing_offset
    last = levels[-1]
    result = {
        "best_delta_deg": (best_dx, best_dy),
        "best_offset_deg": (float(base_x + best_dx), float(base_y + best_dy)),
        "baseline_cost": baseline_cost,
        "best_cost": best_cost,
        "x": last["x"],
        "y": last["y"],
        "costs": last["costs"],
        "levels": levels,
        "n_evaluations": len(cache),
        "search_freq_width_mhz": search_pars.freq_width,
        "baseline_metrics": {"matching_cost": baseline_cost},
        "best_metrics": {"matching_cost": best_cost},
    }
    print(
        f"{wafer_slot} pointing search: cost {baseline_cost:.2f} -> {best_cost:.2f}; "
        f"delta=({best_dx:+.6f}, {best_dy:+.6f}) deg; "
        f"initial offset=({result['best_offset_deg'][0]:+.6f}, "
        f"{result['best_offset_deg'][1]:+.6f}) deg; "
        f"{len(cache)} assignments",
        flush=True,
    )
    return match, result


def plot_initial_offset_search(search, wafer_slot=""):
    """Compare the first and final xi/eta search stencils."""
    levels = search["levels"]
    samples = [levels[0], levels[-1]] if len(levels) > 1 else [levels[0]]
    fig, axs = plt.subplots(
        1, len(samples), figsize=(5.2 * len(samples), 4.6),
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs)

    def edges(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 1:
            return values + np.array([-0.001, 0.001])
        mids = (values[:-1] + values[1:]) / 2
        return np.r_[values[0] - (mids[0] - values[0]), mids,
                     values[-1] + (values[-1] - mids[-1])]

    for i, (ax, level) in enumerate(zip(axs, samples)):
        xs, ys, costs = level["x"], level["y"], level["costs"]
        values = np.asarray(costs, dtype=float)
        positive = values[np.isfinite(values) & (values > 0)]
        log_cost = (len(positive) > 0 and
                    positive.max() / positive.min() > 100)
        if log_cost:
            values = np.log10(np.where(values > 0, values, np.nan))
        im = ax.pcolormesh(edges(xs), edges(ys), values,
                           cmap="viridis_r", shading="auto")
        # Mark the lowest-cost evaluated trial in this stencil.
        if np.isfinite(costs).any():
            iy, ix = np.unravel_index(np.nanargmin(costs), costs.shape)
            ax.plot(xs[ix], ys[iy], "rx", ms=10, mew=2.5)
        ax.set_title("First" if i == 0 else "Final")
        ax.set_xlabel("delta xi (deg)")
        ax.set_ylabel("delta eta (deg)")
        fig.colorbar(im, ax=ax, label="log10 cost" if log_cost else "Cost")

    bx, by = search["best_delta_deg"]
    fig.suptitle(
        f"{wafer_slot} | M0 offset search | "
        f"best ({bx:+.5f}, {by:+.5f}) deg"
    )
    plt.show()


def match_wafer(
    cfg: SolutionsCfg,
    am: AxisManager,
    stream_id: str,
    meas_rset: Optional[dm.ResSet],
) -> MatchSolution:
    """
    Create a match solution for a given wafer slot.

    Args
    ------
    cfg: SolutionsCfg
        Configuration object
    am: AxisManager
        Axis manager containing detector info about relevant wafer slot, along
        with measured pointing data.
    stream_id: str
        Stream Id of the wafer
    """
    match_iterations = []

    m = am.det_info.stream_id == stream_id
    wafer_slot = am.det_info.wafer_slot[m][0]
    tube_slot = am.obs_info.tube_slot

    if meas_rset is None:
        src = dm.ResSet.from_aman(
            am,
            stream_id,
            pointing=am[cfg.pointing_field],
        )
    else:
        src = meas_rset

    if wafer_slot == "ws.":
        wafer_slots = []
        wafers_dict = {}

        # get wafer and wafer_slot entry from imprinter
        with open(cfg.imprinter_path, "r") as f:
            imprinter = yaml.safe_load(f)

        for k, v in imprinter['tel_tubes'].items():
            if v['tube_slot'] == tube_slot:
                for ws in v['wafer_slots']:
                    if 'wafer' in ws.keys():
                        wafers_dict[ws['wafer_slot']] = (ws['wafer'].split('_')[-1].capitalize())
                        wafer_slots.append(ws['wafer_slot'])
    else:
        wafer_slots = [wafer_slot]
        wafers_dict = None

    dst_merged = []
    # loop through and get pointing information for all wafers
    for ws in wafer_slots:
        pt_cfg = dm.PointingConfig(
            fp_file=cfg.ufm_to_fp_path, rx_file=cfg.fp_to_rx_path,
            wafer_slot=ws, platform=cfg.platform,
            zemax_path=cfg.zemax_path,
            roll=np.deg2rad(am.obs_info.roll_center) if cfg.apply_roll else 0.0,
            tube_slot = am.obs_info.tube_slot
        )
        dst_wafer = dm.ResSet.from_wafer_info_file(
            cfg.wafer_info_path,
            stream_id,
            pt_cfg=pt_cfg,
        ).as_array()

        # extract current wafer
        if wafers_dict is not None:
            mask = [wafers_dict[ws] in d["det_id"].astype(str) for d in dst_wafer]
            dst_wafer = dst_wafer[mask]
        dst_wafer['wafer_slot'][:] = ws

        dst_merged.append(dst_wafer)
    dst_merged = np.concatenate(dst_merged)

    # get full wafer info and populate pointing and wafer_slot information
    # from merged dst (not necesary if matching a single wafer)
    if wafer_slot == "ws.":
        dst = dm.ResSet.from_wafer_info_file(
                cfg.wafer_info_path,
                stream_id,
                pt_cfg=None,
            ).as_array()

        merge_fields = ["xi", "eta", "wafer_slot"]
        for i, d in enumerate(dst_merged):
            idx = np.where(dst['det_id'] == d['det_id'])[0][0]
            if idx:
                for field in merge_fields:
                    dst[idx][field] = dst_merged[i][field]
        dst = dm.ResSet.from_array(dst)
    else:
        dst = dm.ResSet.from_array(dst_merged)

    # first match
    match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["match0"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["match0"]["dist_width"]),
        enforce_pointing_reqs=True,
        allow_unassigned_to_assigned=False,
        unassigned_slots=cfg.unassigned_slots,
    )
    offset_search = None
    if cfg.search_initial_pointing_offset:
        match, offset_search = search_best_initial_pointing_offset(
            src, dst, match_pars, cfg, wafer_slot=wafer_slot,
        )
        if cfg.show_match_plots:
            plot_initial_offset_search(offset_search, wafer_slot=wafer_slot)
    else:
        match = dm.Match(src, dst, match_pars=match_pars, apply_dst_pointing=False)

    match_iterations.append(deepcopy(match))
    if cfg.show_match_plots:
        plot_match_offsets(match, "M0 initial", wafer_slot, color_by=cfg.match_plot_color_by)
        plot_match_xieta(match, "M0 initial", wafer_slot, color_by=cfg.match_plot_color_by)

    # Determine the global shift from matched optical detectors only.
    optical_pairs = list(_optical_pointing_pairs(match))
    if not optical_pairs:
        raise ValueError(
            f"{wafer_slot}: cannot estimate global pointing shift: "
            "no matched OPTC pairs with finite xi/eta"
        )

    dxi = np.median([r2.xi - r1.xi for r1, r2 in optical_pairs])
    deta = np.median([r2.eta - r1.eta for r1, r2 in optical_pairs])

    for r in match.src:
        if np.isfinite([r.xi, r.eta]).all():
            r.xi += dxi
            r.eta += deta

    match._match()
    match_iterations.append(deepcopy(match))
    if cfg.show_match_plots:
        plot_match_offsets(match, "M0 shifted", wafer_slot, color_by=cfg.match_plot_color_by)
        plot_match_xieta(match, "M0 shifted", wafer_slot, color_by=cfg.match_plot_color_by)

    if cfg.freq_correct_by_muxband:
        freq_diagnostics = apply_muxband_frequency_correction(
            match, cfg, wafer_slot=wafer_slot
        )
        if cfg.show_match_plots:
            plot_muxband_frequency_correction(freq_diagnostics, wafer_slot)
    else:
        # Correct freq offset by box median interpolation
        foffset_north = get_foffset_interp(match, True)
        foffset_south = get_foffset_interp(match, False)
        for r in match.dst:
            if r.is_north:
                r.res_freq += foffset_north(r.res_freq)
            else:
                r.res_freq += foffset_south(r.res_freq)

    # Second match
    match.match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["match1"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["match1"]["dist_width"]),
        enforce_pointing_reqs=True,
        allow_unassigned_to_assigned=False,
        unassigned_slots=cfg.unassigned_slots
    )
    match._match()
    if cfg.show_match_plots:
        summarize_optical_frequency_slopes(match, "Match 1", cfg.freq_fit_min_matches)
        plot_match_offsets(match, "M1 frequency", wafer_slot, color_by=cfg.match_plot_color_by)
        plot_match_xieta(match, "M1 frequency", wafer_slot, color_by=cfg.match_plot_color_by)

    # Interpolators were estimated from matched OPTC detectors only.
    # Apply the spatial correction to ALL measured resonators with finite xi/eta.
    dxi_interp, deta_interp = get_pt_offset_interp(
        match, sel_rad=np.deg2rad(cfg.sel_rad)
    )
    n_corrected = 0
    for r in match.src:
        if not np.isfinite([r.xi, r.eta]).all():
            continue

        # Evaluate both offsets at the SAME original position.
        xy = (r.xi, r.eta)
        dxi = float(dxi_interp(xy).item())
        deta = float(deta_interp(xy).item())
        if not np.isfinite([dxi, deta]).all():
            raise ValueError(
                f"{wafer_slot}: non-finite spatial correction at xi/eta={xy}"
            )
        r.xi -= dxi
        r.eta -= deta
        n_corrected += 1

    # Third match
    match.match_pars.freq_width = cfg.match_pars["match2"]["freq_width"]
    match.match_pars.dist_width = np.deg2rad(cfg.match_pars["match2"]["dist_width"])

    match._match()
    match_iterations.append(deepcopy(match))
    if cfg.show_match_plots:
        summarize_optical_frequency_slopes(match, "Match 2", cfg.freq_fit_min_matches)
        plot_match_offsets(match, "M2 spatial", wafer_slot, color_by=cfg.match_plot_color_by)
        plot_match_xieta(match, "M2 spatial", wafer_slot, color_by=cfg.match_plot_color_by)

    return MatchSolution(
        match=match,
        match_iterations=match_iterations,
        am=am,
        best_initial_pointing_offset=(
            offset_search["best_offset_deg"] if offset_search is not None
            else tuple(cfg.initial_pointing_offset)
        ),
        initial_offset_search=offset_search,
    )


@dataclass
class FullWaferSolution:
    match_solution: MatchSolution
    pointing_results: List[PointingInfo]
    meta: AxisManager
    stream_id: str


def create_empty_match(cfg, am, wafer_slot, save=False):
    m = am.det_info.wafer_slot == wafer_slot
    stream_id = am.det_info.stream_id[m][0]
    src = dm.ResSet.from_aman(am, stream_id)

    if save:
        resset_file = os.path.join(cfg.results_dir, f"{stream_id}.npy")
        np.save(resset_file, src.as_array())

    return src

def save_wafer_solution(cfg: SolutionsCfg, solution: FullWaferSolution):
    solution.stream_id
    resset_file = os.path.join(cfg.results_dir, f"{solution.stream_id}.npy")
    match_file = os.path.join(cfg.results_dir, "matches", f"{solution.stream_id}.h5")
    if not os.path.exists(os.path.dirname(match_file)):
        os.makedirs(os.path.dirname(match_file))

    match = solution.match_solution.match
    np.save(resset_file, match.merged.as_array())
    match.save(match_file)


def get_wafer_solution(
    cfg: SolutionsCfg, wafer_slot: str, save=False
) -> Optional[FullWaferSolution]:
    pointing_results = load_good_pointing_info(cfg, wafer_slot)
    if len(pointing_results) == 0:
        return None

    if cfg.base_obs_id is not None:
        for i, pi in enumerate(pointing_results):
            if pi.obs_id == cfg.base_obs_id:
                base_idx = i
                break
        else:
            raise ValueError(f"Pointing info for base obs_id not found: {cfg.base_obs_id}")
    else:
        base_idx = 0

    meas_rset, pointing_map = merge_pointing_info(
        cfg, pointing_results, base_idx=base_idx,
    )

    meta = pointing_results[0].meta
    stream_id = meta.det_info.stream_id[0]

    match_solution = match_wafer(
        cfg, meta, stream_id, meas_rset=meas_rset,
    )

    solution = FullWaferSolution(
        pointing_results=pointing_results,
        match_solution=match_solution,
        meta=meta,
        stream_id=stream_id,
    )

    if save:
        save_wafer_solution(cfg, solution)

    return solution


def solve_all(cfg) -> Dict[str, Optional[FullWaferSolution]]:
    results = {ws: get_wafer_solution(cfg, ws, save=True) for ws in cfg.wafer_slots}
    return results


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    with open(cfg_file, "r") as f:
        cfg = SolutionsCfg(**yaml.safe_load(f))
    solve_all(cfg)
