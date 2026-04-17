from sotodlib.core import Context, AxisManager, LabelAxis
from sotodlib.coords import det_match as dm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any
from collections import defaultdict
import os
import numpy as np
import yaml
from copy import deepcopy
from scipy import interpolate
from tqdm.auto import tqdm, trange
from copy import deepcopy

import h5py

from importlib import reload
import yaml
import sys


@dataclass
class SolutionsCfg:
    """
    Args
    ------
    ctx_path: str
        Path to context file to use to pull tod metadata.
    pointing_results_dir: str
        Results to directory that contains pointing resutls from Tomoki's
        workflow. Files in directory should look like:
        ```
        focal_plane_<obs_id>_<wafer_slot>.hdf
        ```
    site_pipeline_cfg_dir: str
        Path to site-pipeline-config dir. Defaults to the env var
        ``$SITE_PIPELINE_CONFIG_DIR``.
    finite_xi_thresh: int
        Minimum number of dets a pointing result must have to add it to the
        analysis.
    min_r2: float
        Minimum R-squared for det pointing to be considered.
    unassigned_slots: int:
        Number of additional "unassigned" node to use per-side
    wafer_info_path: str
        Path to the wafer_info h5 file.
    wafer_map_path: str
        Path to the wafer map file. Defaults to ``<site-pipeline-config>/shared/detmatpping/wafer_map.yaml``.
    Initial pointing offset: Tuple[float, float]
        Estimated pointing offset for the boresight. This should be
        (xi_offset, eta_offset) where both are in radians.
    results_dir: str
        Directory where results should be stored.
    freq_correct_by_muxband: bool
        If true, apply the same freq offset correction to all resonators in a mux-band.
    tel_type: str
        Tel type for the optics model. Either "SAT" or "LAT"
    zemax_path: str
        If running for a "LAT" tel_type, the path to the zemax file must be specified.
    """

    ctx_path: str
    pointing_results_dir: str
    results_dir: str
    wafer_info_path: str
    tel_type: str
    base_obs_id: Optional[str] = None
    zemax_path: Optional[str] = None
    apply_roll: bool = True

    ctx: Context = field(init=False)
    pointing_field: str = "tod_pointing"
    site_pipeline_cfg_dir: str = "$SITE_PIPELINE_CONFIG_DIR"
    finite_xi_thresh: int = (
        500  # Min number of dets with finite xi to consider a pointing input
    )
    min_r2: float = 0.9
    unassigned_slots: int = 1200
    wafer_map_path: Optional[str] = None
    wafer_map: Dict[str, dict] = field(init=False)
    match_pars: Dict[str, dict] = field(default_factory=lambda: defaultdict(dict))

    initial_pointing_offset: Tuple[float, float] = (0, 0)
    ufm_to_fp_path: Optional[str] = None
    freq_correct_by_muxband: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolutionsCfg":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "SolutionsCfg":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def __post_init__(self):
        self.ctx = Context(self.ctx_path)

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

    _xi += offset[0]
    _eta += offset[1]
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
    base_resset = dm.ResSet.from_aman(meta, stream_id=stream_id, pointing=meta.tod_pointing)

    pointing_map = {
        r.idx: [(r.xi, r.eta)] for r in base_resset
    }

    match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["pointing"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["pointing"]["dist_width"])
    )

    for i in range(len(pinfos)):
        if i == base_idx:
            continue
        meta = pinfos[i].meta
        src = dm.ResSet.from_aman(meta, stream_id=stream_id, pointing=meta.tod_pointing)
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


def get_pt_offset_interp(match, sel_rad=np.deg2rad(2)) -> Tuple[Any, Any]:
    _xis, _etas, _dxis, _detas = [], [], [], []
    for r1, r2 in match.get_match_iter(include_unmatched=False):
        _xis.append(r1.xi)
        _etas.append(r1.eta)
        _dxis.append(r1.xi - r2.xi)
        _detas.append(r1.eta - r2.eta)
    xis = np.array(_xis)
    etas = np.array(_etas)
    dxis = np.array(_dxis)
    detas = np.array(_detas)

    xi_list = np.arange(np.nanmin(xis), np.nanmax(xis), sel_rad / 2)
    eta_list = np.arange(np.nanmin(etas), np.nanmax(etas), sel_rad / 2)
    xi_grid, eta_grid = np.meshgrid(xi_list, eta_list)
    dxi_data = np.full_like(xi_grid, np.nan)
    deta_data = np.full_like(eta_grid, np.nan)
    for i, j in np.ndindex(xi_grid.shape):
        sel = (
            np.sqrt((xis - xi_grid[i, j]) ** 2 + (etas - eta_grid[i, j]) ** 2) < sel_rad
        )
        sel &= np.isfinite(dxis) & np.isfinite(detas)
        dxi_data[i, j] = np.nanmedian(dxis[sel])
        deta_data[i, j] = np.nanmedian(detas[sel])

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


def match_wafer(
    cfg: SolutionsCfg,
    am: AxisManager,
    stream_id: str,
    meas_rset: Optional[dm.ResSet]
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

    if meas_rset is None:
        src = dm.ResSet.from_aman(am, stream_id, pointing=am[cfg.pointing_field])
    else:
        src = meas_rset

    pt_cfg = dm.PointingConfig(
        fp_file=cfg.ufm_to_fp_path, wafer_slot=wafer_slot, tel_type=cfg.tel_type,
        zemax_path=cfg.zemax_path,
        roll=np.deg2rad(am.obs_info.roll_center) if cfg.apply_roll else 0.0,
        tube_slot = am.obs_info.tube_slot
    )
    dst = dm.ResSet.from_wafer_info_file(cfg.wafer_info_path, stream_id, pt_cfg=pt_cfg)

    # first match
    match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["match0"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["match0"]["dist_width"]),
        enforce_pointing_reqs=True,
        allow_unassigned_to_assigned=False,
        unassigned_slots=cfg.unassigned_slots
    )
    match = dm.Match(src, dst, match_pars=match_pars, apply_dst_pointing=False)

    match_iterations.append(deepcopy(match))
    dxis, detas = [], []
    dfs = []
    is_north = []
    for r1, r2 in match.get_match_iter(include_unmatched=False):
        dxis.append(r2.xi - r1.xi)
        detas.append(r2.eta - r1.eta)
        dfs.append(r2.res_freq - r1.res_freq)
        is_north.append(r2.is_north)
    dxi = np.nanmedian(dxis)
    deta = np.nanmedian(detas)

    for r in match.src:
        r.xi += dxi
        r.eta += deta

    match._match()
    match_iterations.append(deepcopy(match))

    if cfg.freq_correct_by_muxband:
        da = match.dst.as_array()
        df_arr = np.full(len(da), np.nan)
        for i, r in enumerate(match.dst):
            if r.matched:
                df_arr[i] = r.res_freq - match.src[r.match_idx].res_freq

        for is_north in [0, 1]:
            for mb in np.unique(da["mux_band"]):
                mask = (da["mux_band"] == mb) & (da["is_north"] == is_north)
                df_med = np.nanmedian(df_arr[mask])
                for res_idx in np.where(mask)[0]:
                    match.dst[res_idx].res_freq -= df_med
    else:
        # Correct freq offset by box median interpolation
        foffset_north = get_foffset_interp(match, True)
        foffset_south = get_foffset_interp(match, False)
        for r in match.dst:
            if r.is_north:
                r.res_freq += foffset_north(r.res_freq)
            else:
                r.res_freq += foffset_south(r.res_freq)

    match.match_pars = dm.MatchParams(
        freq_width=cfg.match_pars["match1"]["freq_width"],
        dist_width=np.deg2rad(cfg.match_pars["match1"]["dist_width"]),
        enforce_pointing_reqs=True,
        allow_unassigned_to_assigned=False,
        unassigned_slots=cfg.unassigned_slots
    )
    match._match()

    dxi_interp, deta_interp = get_pt_offset_interp(match, sel_rad=np.deg2rad(2))
    for r in match.src:
        if np.isnan(r.xi):
            continue
        r.xi -= dxi_interp((r.xi, r.eta)).item()
        r.eta -= deta_interp((r.xi, r.eta)).item()

    match.match_pars.freq_width = cfg.match_pars["match2"]["freq_width"]
    match.match_pars.dist_width = np.deg2rad(cfg.match_pars["match2"]["dist_width"])

    match._match()

    match_iterations.append(deepcopy(match))

    return MatchSolution(
        match=match,
        match_iterations=match_iterations,
        am=am,
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

    meas_rset, pointing_map = merge_pointing_info(cfg, pointing_results, base_idx=base_idx)
    # tod_pointing = get_best_tod_pointing(cfg, pointing_results)

    meta = pointing_results[0].meta
    stream_id = meta.det_info.stream_id[0]

    match_solution = match_wafer(cfg, meta, stream_id, meas_rset=meas_rset)

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
    wafer_slots = ["ws0", "ws1", "ws2", "ws3", "ws4", "ws5", "ws6"]
    results = {ws: get_wafer_solution(cfg, ws, save=True) for ws in wafer_slots}
    return results


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    with open(cfg_file, "r") as f:
        cfg = SolutionsCfg(**yaml.safe_load(f))
    solve_all(cfg)
