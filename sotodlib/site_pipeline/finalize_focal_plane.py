import argparse as ap
from copy import deepcopy
import os
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import NDArray
from megham.transform import get_rigid  # TODO: add an equiv function to sotodlib
from scipy.cluster import vq
from scipy.optimize import minimize
from sotodlib.coords import affine as af
from sotodlib.coords import optics as op
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


@dataclass
class Transform:
    shift: NDArray[np.floating]  # (ndim,)
    xieta_affine: InitVar[NDArray[np.floating]]  # (ndim-1, ndim-1)
    gamma_scale: InitVar[float]
    affine: NDArray[np.floating] = field(init=False)  # (ndim, ndim)
    scale: NDArray[np.floating] = field(init=False)  # (ndim,)
    shear: float = field(init=False)
    rot: float = field(init=False)

    def __post_init__(self, xieta_affine, gamma_scale):
        self.affine = np.eye(len(xieta_affine) + 1)
        self.affine[: len(xieta_affine), : len(xieta_affine)] = xieta_affine
        self.affine[-1, -1] = gamma_scale
        self.decompose()

    @classmethod
    def identity(cls):
        return Transform(np.zeros(3), np.eye(2), 0)

    def decompose(self):
        xieta_affine = self.affine[:2, :2]
        gamma_scale = self.affine[-1, -1]
        scale, shear, rot = af.decompose_affine(xieta_affine)
        self.scale = np.array((*scale, gamma_scale))
        self.shear = shear.item()
        self.rot = af.decompose_rotation(rot)[-1]

    def save(self, f, path, append=""):
        if path not in f:
            f.create_group(path)
        _add_attrs(
            f[path],
            {
                f"shift{append}": self.shift,
                f"scale{append}": self.scale,
                f"shear{append}": self.shear,
                f"rot{append}": self.rot,
                f"affine{append}": self.affine,
            },
        )


@dataclass
class Template:
    det_ids: NDArray[np.str_]  # (ndet,)
    fp: NDArray[np.floating]  # (ndim, ndet)
    optical: NDArray[np.bool_]  # (ndet,)
    pointing_cfg: InitVar[Dict]
    center: NDArray[np.floating] = field(init=False)  # (ndim, 1)
    spacing: NDArray[np.floating] = field(init=False)  # (ndim,)

    def __post_init__(self, pointing_cfg):
        self.center = np.array(
            op.get_focal_plane(None, x=0, y=0, pol=0, **pointing_cfg)
        )
        xieta_spacing = af.get_spacing(self.fp[:2, self.optical])
        # For gamma rather than the spacing in real space we want the difference between bins
        # This is a rough estimate but good enough for us
        gamma_spacing = np.percentile(np.diff(np.sort(self.fp[2])), 99.9)
        self.spacing = np.array([xieta_spacing, xieta_spacing, gamma_spacing])


@dataclass
class FocalPlane:
    template: Template
    stream_id: str
    n_aman: InitVar[int]
    full_fp: NDArray[np.floating] = field(init=False)  # (ndim, ndet, n_aman)
    tot_weight: NDArray[np.floating] = field(init=False)  # (ndet,)
    avg_fp: NDArray[np.floating] = field(init=False)  # (ndim, ndet)
    weights: NDArray[np.floating] = field(init=False)  # (ndet,)
    transformed: NDArray[np.floating] = field(init=False)  # (ndim, ndet)
    center_transformed: NDArray[np.floating] = field(init=False)  # (ndim, 1)
    have_gamma: bool = field(init=False, default=False)
    n_point: NDArray[np.int_] = field(init=False)
    n_gamma: NDArray[np.int_] = field(init=False)
    transform: Transform = field(init=False, default_factory=Transform.identity)
    transform_nocm: Transform = field(init=False, default_factory=Transform.identity)

    def __post_init__(self, n_aman):
        self.full_fp = np.full(self.template.fp.shape + (n_aman,), np.nan)
        self.tot_weight = np.zeros(len(self.template.det_ids))
        self.avg_fp = np.full_like(self.template.fp, np.nan)
        self.weight = np.zeros(len(self.template.det_ids))
        self.transformed = self.template.fp.copy()
        self.center_transformed = self.template.center.copy()
        self.n_point = np.zeros_like(self.template.det_ids, dtype=int)
        self.n_gamma = np.zeros_like(self.template.det_ids, dtype=int)

    def map_to_template(self, aman):
        _, msk, template_msk = np.intersect1d(
            aman.det_info.det_id, self.template.det_ids, return_indices=True
        )
        if len(msk) != aman.dets.count:
            logger.warning("There are matched dets not found in the template")
        mapping = np.argsort(np.argsort(self.template.det_ids[template_msk]))
        srt = np.argsort(aman.det_info.det_id[msk])
        xi = aman.pointing.xi[msk][srt][mapping]
        eta = aman.pointing.eta[msk][srt][mapping]
        if "polarization" in aman:
            # name of field just a placeholder for now
            gamma = aman.polarization.polang[msk][srt][mapping]
        elif "gamma" in aman.pointing:
            gamma = aman.pointing.gamma[msk][srt][mapping]
        else:
            gamma = np.full(len(xi), np.nan)
        fp = np.vstack((xi, eta, gamma))
        return fp, template_msk

    def add_fp(self, i, fp, weights, template_msk):
        self.full_fp[:, template_msk, i] = fp * weights
        self.tot_weight[template_msk] += weights

    def save(self, f, db_info, group):
        ndets = len(self.template.det_ids)
        outdt = [
            ("dets:det_id", self.template.det_ids.dtype),
            ("xi", np.float32),
            ("eta", np.float32),
            ("gamma", np.float32),
        ]
        fpout = np.fromiter(
            zip(self.template.det_ids, *self.transformed), dtype=outdt, count=ndets
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpout),
            f,
            f"{group}/focal_plane",
            overwrite=True,
        )
        _add_attrs(f[f"{group}/focal_plane"], {"measured_gamma": self.have_gamma})
        entry = {"dets:stream_id": self.stream_id, "dataset": f"{group}/focal_plane"}
        entry.update(db_info[1])
        db_info[0].add_entry(entry, filename=os.path.basename(f.filename), replace=True)

        outdt_full = [
            ("dets:det_id", self.template.det_ids.dtype),
            ("xi_t", np.float32),
            ("eta_t", np.float32),
            ("gamma_t", np.float32),
            ("xi_m", np.float32),
            ("eta_m", np.float32),
            ("gamma_m", np.float32),
            ("weights", np.float32),
            ("n_point", np.int8),
            ("n_gamma", np.int8),
        ]
        fpfullout = np.fromiter(
            zip(
                self.template.det_ids,
                *self.transformed,
                *self.avg_fp,
                self.weights,
                self.n_point,
                self.n_gamma,
            ),
            dtype=outdt_full,
            count=ndets,
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpfullout),
            f,
            f"{group}/focal_plane_full",
            overwrite=True,
        )

        self.transform.save(f, f"{group}/transform")
        self.transform_nocm.save(f, f"{group}/transform", "_nocm")
        _add_attrs(
            f[f"{group}"],
            {
                "fit_centers": self.center_transformed,
                "template_centers": self.template.center,
            },
        )


@dataclass
class OpticsTube:
    pointing_cfg: InitVar[Dict]
    name: str = field(init=False)
    focal_planes: List[FocalPlane] = field(init=False, default_factory=list)
    center: NDArray[np.floating] = field(init=False)
    center_transformed: NDArray[np.floating] = field(init=False)
    transform: Transform = field(init=False, default_factory=Transform.identity)

    def __post_init__(self, pointing_cfg):
        self.name = pointing_cfg["tube_slot"]
        telescope_flavor = pointing_cfg["telescope_flavor"].upper()
        if telescope_flavor not in ["LAT", "SAT"]:
            raise ValueError("Telescope should be LAT or SAT")

        if telescope_flavor == "LAT":
            if pointing_cfg["zemax_path"] is None:
                raise ValueError("Must provide zemax_path for LAT")
            xi, eta, gamma = op.LAT_focal_plane(
                None,
                pointing_cfg["zemax_path"],
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                tube_slot=self.name,
            )
        else:
            xi, eta, gamma = op.SAT_focal_plane(
                None,
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                mapping_data=pointing_cfg.get("mapping_data", None),
            )
        self.center = np.array((xi, eta, gamma)).reshape((3, 1))
        self.center_transformed = self.center.copy()

    def save(self, f, db_info):
        f.create_group(self.name)
        _add_attrs(
            f[self.name],
            {"center": self.center, "center_transformed": self.center_transformed},
        )
        self.transform.save(f, f"{self.name}/transform")
        for focal_plane in self.focal_planes:
            focal_plane.save(f, db_info, f"{self.name}/{focal_plane.stream_id}")


def _create_db(filename, per_obs, obs_id):
    base = {}
    if per_obs:
        base = {"obs:obs_id": obs_id}
    if os.path.isfile(filename):
        return metadata.ManifestDb(filename), base
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:stream_id")
    if per_obs:
        scheme.add_exact_match("obs:obs_id")
    scheme.add_data_field("dataset")

    metadata.ManifestDb(scheme=scheme).to_file(filename)
    return metadata.ManifestDb(filename), base


def _avg_focalplane(full_fp, tot_weight):
    # Figure out how many good pointings we have for each det
    msk = np.isfinite(full_fp)
    n_obs = np.sum(np.any(msk, axis=0), axis=-1)
    n_point, _, n_gamma = tuple(np.sum(msk, axis=-1))
    tot_weight[tot_weight == 0] = np.nan
    avg_fp = np.nansum(full_fp, axis=-1) / tot_weight
    avg_weight = tot_weight / n_obs

    # nansum all all nans is 0, addressing that case here
    all_nan = ~np.any(np.isfinite(full_fp).reshape((len(full_fp), -1)), axis=1)
    avg_fp[all_nan] = np.nan

    return avg_fp, avg_weight, n_point, n_gamma


def _log_vals(shift, scale, shear, rot, axis):
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    for ax, s in zip(axis, shift):
        logger.info("\tShift along %s axis is %f", ax, s)
    for ax, s in zip(axis, scale):
        logger.info("\tScale along %s axis is %f", ax, s)
        if np.isclose(s, deg2rad):
            logger.warning(
                "\tScale factor for %s looks like a degrees to radians conversion", ax
            )
        elif np.isclose(s, rad2deg):
            logger.warning(
                "\tScale factor for %s looks like a radians to degrees conversion", ax
            )
    logger.info("\tShear param is %f", shear)
    logger.info("\tRotation of the %s-%s plane is %f radians", axis[0], axis[1], rot)


def _mk_plot(plot_dir, froot, nominal, measured, transformed):
    plt.style.use("tableau-colorblind10")
    # Plot pointing
    plt.scatter(
        nominal[0], nominal[1], alpha=0.4, color="blue", label="nominal", marker="P"
    )
    plt.scatter(
        transformed[0],
        transformed[1],
        alpha=0.4,
        color="black",
        label="transformed",
        marker="X",
    )
    plt.scatter(measured[0], measured[1], alpha=0.4, color="orange", label="fit")
    plt.xlabel("Xi (rad)")
    plt.ylabel("Eta (rad)")
    plt.legend()
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{froot}.png"))
        plt.cla()

    # Histogram of differences
    diff = measured - transformed
    dist = np.linalg.norm(diff[:2, np.isfinite(diff[0])], axis=0)
    bins = max(int(len(dist) / 20), 10)
    plt.hist(dist[dist < np.percentile(dist, 97)], bins=bins)
    plt.xlabel("Distance Between Measured and Transformed (rad)")
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{froot}_dist.png"))
        plt.clf()

    # tricontourf of residuals, subplots for xi, eta, gamma
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    flat_diff = np.abs(diff.ravel())
    max_diff = np.percentile(flat_diff[np.isfinite(flat_diff)], 99)
    im = None
    for i, name in enumerate(("xi", "eta", "gamma")):
        isfinite = np.isfinite(diff[i])
        axs[i].set_title(name)
        axs[i].set_xlim(np.nanmin(transformed[0]), np.nanmax(transformed[0]))
        axs[i].set_ylim(np.nanmin(transformed[1]), np.nanmax(transformed[1]))
        axs[i].set_aspect("equal")
        if np.sum(isfinite) == 0:
            continue
        im = axs[i].tricontourf(
            transformed[0, isfinite],
            transformed[1, isfinite],
            diff[i, isfinite],
            levels=20,
            vmin=-1 * max_diff,
            vmax=max_diff,
        )
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    axs[0].set_ylabel("Eta (rad)")
    axs[1].set_xlabel("Xi (rad)")
    fig.suptitle("Residuals from Fit")
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{froot}_res.png"), bbox_inches="tight")
        plt.close()


def gamma_fit(src, dst):
    """
    Fit the transformation for gamma.
    Note that the periodicity here assumes things are in radians.

    Arguments:

        src: Source gamma in radians

        dst: Destination gamma in radians

    Returns:

       scale: Scale applied to src

       shift: Shift applied to scale*src
    """

    def _gamma_min(pars, src, dst):
        scale, shift = pars
        transformed = np.sin(src * scale + shift)
        diff = np.sin(dst) - transformed

        return np.sqrt(np.mean(diff**2))

    res = minimize(_gamma_min, (1.0, 0.0), (src, dst))
    return res.x


def _load_template(template_path, ufm, pointing_cfg):
    template_rset = read_dataset(template_path, ufm)
    det_ids = template_rset["dets:det_id"]
    template = np.column_stack(
        (
            np.array(template_rset["xi"]),
            np.array(template_rset["eta"]),
            np.array(template_rset["gamma"]),
        )
    )
    template_optical = template_rset["is_optical"]

    return Template(
        np.array(det_ids), template.T, np.array(template_optical), pointing_cfg
    )


def _load_ctx(config):
    ctx = Context(config["context"]["path"])
    tod_pointing_name = config["context"].get("tod_pointing", "tod_pointing")
    map_pointing_name = config["context"].get("map_pointing", "map_pointing")
    pol_name = config["context"].get("polarization", "polarization")
    dm_name = config["context"].get("detmap", "detmap")
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    obs_ids = np.unique(obs_ids)
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")
    _config = config.copy()
    if "query" in _config["context"]:
        del _config["context"]["query"]
    amans = []
    dets = config["context"].get("dets", {})
    for obs_id in obs_ids:
        aman = ctx.get_meta(obs_id, dets=dets)
        if "det_info" not in aman:
            raise ValueError(f"No det_info in {obs_id}")
        if "wafer" not in aman.det_info and dm_name in aman:
            dm_aman = aman[dm_name].copy()
            aman.det_info.wrap("wafer", dm_aman)
            if "det_id" not in aman.det_info:
                aman.det_info.wrap(
                    "det_id", aman.det_info.wafer.det_id, [(0, aman.dets)]
                )
        if "det_id" in aman.det_info:
            aman.restrict("dets", ~np.isin(aman.det_info.det_id, ["", "NO_MATCH"]))
        else:
            raise ValueError(f"No detmap for {obs_id}")
        pol = pol_name in aman
        if pol:
            aman.move(pol_name, "polarization")
        else:
            logger.warning("No polarization data in context")

        if tod_pointing_name in aman:
            _aman = aman.copy()
            _aman.move(tod_pointing_name, "pointing")
            amans.append(_aman)
        if map_pointing_name in aman:
            _aman = aman.copy()
            _aman.move(map_pointing_name, "pointing")
            amans.append(_aman)
        elif tod_pointing_name not in aman:
            raise ValueError(f"No pointing found in {obs_id}")
    stream_ids = np.unique(np.concatenate([aman.det_info.stream_id for aman in amans]))

    return amans, obs_ids, stream_ids


def _load_rset_single(config):
    obs_id = config["resultsets"].get("obs_id", "")
    pointing_rset = read_dataset(*config["resultsets"]["pointing"])
    pointing_aman = pointing_rset.to_axismanager(axis_key="dets:readout_id")
    aman = AxisManager(pointing_aman.dets)
    aman = aman.wrap("pointing", pointing_aman)

    if "polarization" in config["resultsets"]:
        polarization_rset = read_dataset(*config["resultsets"]["polarization"])
        polarization_aman = polarization_rset.to_axismanager(axis_key="dets:readout_id")
        aman = aman.wrap("polarization", polarization_aman)

    det_info = AxisManager(aman.dets)
    dm_rset = read_dataset(*config["resultsets"]["detmap"])
    dm_aman = dm_rset.to_axismanager(axis_key="readout_id")
    det_info.wrap("wafer", dm_aman)
    det_info.wrap("readout_id", det_info.dets.vals, [(0, det_info.dets)])
    det_info.wrap("det_id", det_info.wafer.det_id, [(0, det_info.dets)])
    det_info.wrap(
        "stream_id",
        np.array([config["stream_id"].lower()] * det_info.dets.count),
        [(0, det_info.dets)],
    )
    det_info.wrap(
        "wafer_slot",
        np.array([config["wafer_slot"].lower()] * det_info.dets.count),
        [(0, det_info.dets)],
    )
    det_info.restrict("dets", det_info.dets.vals[det_info.det_id != ""])
    det_info.det_id = np.char.strip(det_info.det_id)  # Needed for some old results
    aman = aman.wrap("det_info", det_info)
    aman.restrict("dets", aman.dets.vals[aman.det_info.det_id != "NO_MATCH"])

    obs_info = AxisManager()
    obs_info.wrap("telescope_flavor", config["telescope_flavor"].lower())
    obs_info.wrap("tube_slot", config["tube_slot"].lower())
    aman.wrap("obs_info", obs_info)

    smurf = AxisManager(aman.dets)
    if "band" in aman.pointing:
        smurf.wrap("band", np.array(aman.pointing.band, dtype=int), [(0, smurf.dets)])
    elif "wafer" in det_info and "smurf_band" in det_info.wafer:
        smurf.wrap(
            "band", np.array(det_info.wafer.smurf_band, dtype=int), [(0, smurf.dets)]
        )
    if "channel" in aman.pointing:
        smurf.wrap(
            "channel", np.array(aman.pointing.channel, dtype=int), [(0, smurf.dets)]
        )
    elif "wafer" in det_info and "smurf_channel" in det_info.wafer:
        smurf.wrap(
            "channel",
            np.array(det_info.wafer.smurf_channel, dtype=int),
            [(0, smurf.dets)],
        )
    aman.det_info.wrap("smurf", smurf)

    return aman, obs_id


def _load_rset(config):
    stream_id = config["stream_id"]
    obs = config["resultsets"]
    _config = config.copy()
    obs_ids = np.array(list(obs.keys()))
    amans: List[Optional[AxisManager]] = [None] * len(obs_ids)
    obs_info = AxisManager()
    obs_info.wrap("stream_id", stream_id)
    for i, (obs_id, rsets) in enumerate(obs.items()):
        _config["resultsets"] = rsets
        _config["resultsets"]["obs_id"] = obs_id
        aman, _ = _load_rset_single(_config)
        if "det_info" not in aman or "det_id" not in aman.det_info:
            raise ValueError(f"No detmap for {obs_id}")
        amans[i] = aman

    return (
        amans,
        obs_ids,
        [
            stream_id,
        ],
    )


def _mk_pointing_config(telescope_flavor, tube_slot, wafer_slot, config):
    config_dir = config.get("pipeline_config_dir", os.environ["PIPELINE_CONFIG_DIR"])
    config_path = os.path.join(config_dir, "shared/focalplane/ufm_to_fp.yaml")
    zemax_path = config.get("zemax_path", None)

    pointing_cfg = {
        "telescope_flavor": telescope_flavor,
        "tube_slot": tube_slot,
        "wafer_slot": wafer_slot,
        "config_path": config_path,
        "zemax_path": zemax_path,
        "return_fp": False,
    }
    return pointing_cfg


def _restrict_inliers(aman, focal_plane):
    # TODO: Use gamma as well
    # Map to template
    fp, template_msk = focal_plane.map_to_template(aman)
    fp = fp[:2].T
    inliers = np.ones(len(fp), dtype=bool)

    rad_thresh = 1.05 * np.nanmax(
        np.linalg.norm(
            focal_plane.template.fp[:2] - focal_plane.template.center[:2], axis=0
        )
    )

    # Use kmeans to kill any ghosts
    fp_white = vq.whiten(fp[inliers])
    codebook, _ = vq.kmeans(fp_white, 2)
    codes, _ = vq.vq(fp_white, codebook)

    c0 = codes == 0
    c1 = codes == 1
    m0 = np.median(fp[inliers][c0], axis=0)
    m1 = np.median(fp[inliers][c1], axis=0)
    dist = np.linalg.norm(m0 - m1)

    # If centroids are too far from each other use the bigger one
    if dist < rad_thresh:
        cluster = c0 + c1
    elif np.sum(c0) >= np.sum(c1):
        cluster = c0
    else:
        cluster = c1

    # Flag anything too far away from the center
    cent = np.median(fp[inliers][cluster], axis=0)
    r = np.linalg.norm(fp[inliers] - cent, axis=1)
    inliers[inliers] *= cluster * (r <= rad_thresh)

    # Now kill dets that seem too far from their match
    fp[~inliers] = np.nan
    likelihood = af.gen_weights(fp.T, focal_plane.template.fp[:2, template_msk])
    inliers *= likelihood > 0.95  # ~2 sigma cut

    # Now restrict the AxisManager
    inlier_det_ids = focal_plane.template.det_ids[template_msk][inliers]
    return aman.restrict(
        "dets", aman.dets.vals[np.isin(aman.det_info.det_id, inlier_det_ids)]
    )


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load data
    if "context" in config:
        amans, obs_ids, stream_ids = _load_ctx(config)
    elif "resultsets" in config:
        amans, obs_ids, stream_ids = _load_rset(config)
    else:
        raise ValueError("No valid inputs provided")

    # Build output path
    append = config.get("append", "")
    per_obs = config.get("per_obs", False)
    froot = f"focal_plane{bool(append)*'_'}{append}{per_obs*('_'+obs_ids[0])}"
    dbroot = f"db{bool(append)*'_'}{append}"
    subdir = config.get("subdir", "")
    subdir = subdir + (subdir == "") * (
        per_obs * "per_obs" + (not per_obs) * "combined"
    )
    outpath = os.path.join(config["outdir"], subdir, f"{froot}.h5")
    dbpath = os.path.join(config["outdir"], subdir, f"{dbroot}.sqlite")
    outpath = os.path.abspath(outpath)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    weight_factor = config.get("weight_factor", 1000)
    gen_template = "template" not in config
    template_path = config.get("template", "nominal.h5")
    have_template = os.path.exists(template_path)
    if not gen_template and not have_template:
        logger.error("Provided template doesn't exist, trying to generate one")
        gen_template = True
    ots = {}
    for stream_id in stream_ids:
        logger.info("Working on %s", stream_id)

        # Limit ourselves to amans with this stream_id and restrict
        amans_restrict = [
            aman.copy().restrict(
                "dets", aman.dets.vals[aman.det_info.stream_id == stream_id]
            )
            for aman in amans
            if aman is not None and stream_id in aman.det_info.stream_id
        ]
        if len(amans_restrict) == 0:
            logger.error(
                "\tSomehow no AxisManagers with stream_id %s, skipping", stream_id
            )
            continue

        # Figure out where this UFM is installed and make pointing config
        tel = np.unique([aman.obs_info.telescope_flavor for aman in amans_restrict])
        ot = np.unique([aman.obs_info.tube_slot for aman in amans_restrict])
        ws = np.unique(
            np.concatenate([aman.det_info.wafer_slot for aman in amans_restrict])
        )
        if len(tel) > 1:
            raise ValueError(f"Multiple telescope flavors found for {stream_id}")
        if len(ot) > 1:
            raise ValueError(f"Multible tube slots found for {stream_id}")
        if len(ws) > 1:
            raise ValueError(f"Multiple wafer slots for {stream_id}")
        tel, ot, ws = tel[0], ot[0], ws[0]
        logger.info("\t%s is in %s %s %s", stream_id, tel, ot, ws)
        pointing_cfg = _mk_pointing_config(tel, ot, ws, config)
        if ot not in ots.keys():
            ots[ot] = OpticsTube(pointing_cfg)

        # If a template is provided load it, otherwise generate one
        if gen_template:
            logger.info(f"\tGenerating template for {stream_id}")
            if "wafer_info" not in config:
                raise ValueError("Need wafer_info to generate template")
            template_det_ids, template, is_optical = op.gen_template(
                config["wafer_info"], stream_id, **pointing_cfg
            )
            template = Template(template_det_ids, template.T, is_optical, pointing_cfg)
        elif have_template:
            logger.info("\tLoading template from %s", template_path)
            template = _load_template(template_path, stream_id, pointing_cfg)
        else:
            raise ValueError(
                "No template provided and unable to generate one for some reason"
            )

        focal_plane = FocalPlane(template, stream_id, len(amans))
        for i, (aman, obs_id) in enumerate(zip(amans_restrict, obs_ids)):
            logger.info("\tWorking on %s", obs_id)
            if aman.dets.count == 0:
                logger.info("\t\tNo dets found, skipping")
                continue

            # Restrict to optical dets
            optical = np.isin(
                aman.det_info.det_id, focal_plane.template.det_ids[template.optical]
            )
            aman.restrict("dets", aman.dets.vals[optical])
            if aman.dets.count == 0:
                logger.info("\t\tNo optical dets, skipping", stream_id)
                continue

            # Do some outlier cuts
            _restrict_inliers(aman, focal_plane)

            # Mapping to template
            fp, template_msk = focal_plane.map_to_template(aman)

            # Try an initial alignment and get weights
            try:
                aff, sft = af.get_affine(
                    fp[:2], focal_plane.template.fp[:2, template_msk]
                )
            except ValueError as e:
                logger.error("\t\t%s", e)
                continue
            aligned = aff @ fp[:2] + sft[..., None]
            if np.any(np.isfinite(fp[2])):
                gscale, gsft = gamma_fit(
                    fp[2], focal_plane.template.fp[2, template_msk]
                )
                weights = af.gen_weights(
                    np.vstack((aligned, gscale * fp[2] + gsft)),
                    focal_plane.template.fp[:, template_msk],
                    focal_plane.template.spacing.ravel() / weight_factor,
                )
            else:
                weights = af.gen_weights(
                    aligned,
                    focal_plane.template.fp[:2, template_msk],
                    focal_plane.template.spacing[:2].ravel() / weight_factor,
                )

            # Store weighted values
            focal_plane.add_fp(i, fp, weights, template_msk)

        # Compute the average focal plane with weights
        (
            focal_plane.avg_fp,
            focal_plane.weights,
            focal_plane.n_point,
            focal_plane.n_gamma,
        ) = _avg_focalplane(focal_plane.full_fp, focal_plane.tot_weight)

        # Compute transformation between the two nominal and measured pointing
        focal_plane.have_gamma = np.sum(focal_plane.n_gamma) > 0
        if focal_plane.have_gamma:
            gamma_scale, gamma_shift = gamma_fit(
                focal_plane.template.fp[2], focal_plane.avg_fp[2]
            )
            focal_plane.transformed[2] = (
                focal_plane.template.fp[2] * gamma_scale + gamma_shift
            )
            focal_plane.center_transformed[2] = (
                gamma_scale * focal_plane.template.center[2] + gamma_shift
            )
        else:
            logger.warning(
                "\tNo polarization data availible, gammas will be filled with the nominal values."
            )
            gamma_scale = 1.0
            gamma_shift = 0.0

        try:
            affine, shift = af.get_affine_two_stage(
                focal_plane.template.fp[:2], focal_plane.avg_fp[:2], focal_plane.weights
            )
        except ValueError as e:
            logger.error("\t%s", e)
            continue

        focal_plane.transformed[:2] = (
            affine @ focal_plane.template.fp[:2] + shift[..., None]
        )
        focal_plane.center_transformed[:2] = (
            affine @ focal_plane.template.center[:2] + shift[..., None]
        )

        rms = np.sqrt(np.nanmean((focal_plane.avg_fp - focal_plane.transformed) ** 2))
        logger.info("\tRMS after transformation is %f", rms)

        shift = np.array((*shift, gamma_shift))
        focal_plane.transform = Transform(shift, affine, gamma_scale)
        _log_vals(
            focal_plane.transform.shift,
            focal_plane.transform.scale,
            focal_plane.transform.shear,
            focal_plane.transform.rot,
            ("xi", "eta", "gamma"),
        )

        if config.get("plot", False):
            plot_dir = config.get("plot_dir", None)
            proot = f"{stream_id}{append}"
            if plot_dir is not None:
                plot_dir = os.path.join(plot_dir, subdir, per_obs * obs_ids[0])
                plot_dir = os.path.abspath(plot_dir)
                os.makedirs(plot_dir, exist_ok=True)
            _mk_plot(
                plot_dir,
                proot,
                focal_plane.template.fp,
                focal_plane.avg_fp,
                focal_plane.transformed,
            )
        ots[ot].focal_planes.append(focal_plane)

    # Per OT common mode
    for ot in ots.values():
        logger.info("Fitting common mode for %s", ot.name)
        centers = np.hstack([fp.template.center for fp in ot.focal_planes])
        centers_transformed = np.hstack(
            [fp.center_transformed for fp in ot.focal_planes]
        )
        if centers.shape[-1] < 3:
            logger.warning(
                "\tToo few wafers fit to compute common mode, transform will be approximated"
            )
            centers = np.hstack([ot.center, ot.center - 1, ot.center + 1])
            centers_transformed = np.mean(
                [
                    fp.transform.affine @ centers + fp.transform.shift[..., None]
                    for fp in ot.focal_planes
                ],
                axis=0,
            )
        rot, sft = get_rigid(centers[:2].T, centers_transformed[:2].T)
        rot = rot.T
        gamma_shift = np.mean(centers_transformed[2] - centers[2])
        ot.transform = Transform(np.array((*sft.ravel(), gamma_shift)), rot, 1.0)
        ot.center_transformed = (
            ot.transform.affine @ ot.center + ot.transform.shift[..., None]
        )
        _log_vals(
            ot.transform.shift,
            ot.transform.scale,
            ot.transform.shear,
            ot.transform.rot,
            ("xi", "eta", "gamma"),
        )

    # Full receiver common mode
    logger.info("Fitting receiver common mode")
    origin = np.zeros(3)[..., None]
    if len(ots) == 1:
        logger.info("\tOnly one OT found, receiver common mode will be from this tube")
        recv_transform = deepcopy(tuple(ots.values())[0].transform)
    else:
        centers = np.hstack([ot.center for ot in ots.values()])
        centers_transformed = np.hstack([ot.center_transformed for ot in ots.values()])
        if len(ots) < 3:
            logger.info(
                "\tNot enough OTs to fit receiver common mode, transform will be approximated"
            )
            centers = np.column_stack([np.roll(np.arange(3), i) for i in range(3)])
            centers_transformed = np.mean(
                [
                    ot.transform.affine @ centers + ot.transform.shift[..., None]
                    for ot in ots.values()
                ],
                axis=0,
            )
        rot, sft = get_rigid(centers[:2].T, centers_transformed[:2].T)
        rot = rot.T
        gamma_shift = np.mean(centers_transformed[2] - centers[2])
        recv_transform = Transform(np.array((*sft.ravel(), gamma_shift)), rot, 1.0)
    recv_center = recv_transform.affine @ origin + recv_transform.shift[..., None]
    _log_vals(
        recv_transform.shift,
        recv_transform.scale,
        recv_transform.shear,
        recv_transform.rot,
        ("xi", "eta", "gamma"),
    )

    # Now compute correction only transform for each ufm
    # Transforms are composed as ufm(ot(rx(focal_plane)))
    for ot in ots.values():
        # The full CM will end being the OT CM from above
        full_cm = deepcopy(ot.transform)

        # Now remove the receiver CM from the OT
        aff_inv = np.linalg.inv(recv_transform.affine)
        ot.transform.affine = ot.transform.affine @ aff_inv
        ot.transform.shift = (
            ot.transform.shift
            - (ot.transform.affine @ (recv_transform.shift)[..., None])[:, 0]
        )
        ot.transform.decompose()

        # Now for each fp remove the CM
        for fp in ot.focal_planes:
            aff_inv = np.linalg.inv(full_cm.affine)
            fp.transform_nocm.affine = fp.transform.affine @ aff_inv
            fp.transform_nocm.shift = (
                fp.transform.shift
                - (fp.transform.affine @ (full_cm.shift)[..., None])[:, 0]
            )
            fp.transform_nocm.decompose()

    # Make final outputs and save
    logger.info("Saving data to %s", outpath)
    logger.info("Writing to database at %s", dbpath)
    db, base = _create_db(dbpath, per_obs=per_obs, obs_id=obs_ids[0])
    with h5py.File(outpath, "w") as f:
        _add_attrs(f["/"], {"center": origin, "center_transformed": recv_center})
        f.create_group("transform")
        _add_attrs(
            f["transform"],
            {
                "shift": recv_transform.shift,
                "scale": recv_transform.scale,
                "shear": recv_transform.shear,
                "rot": recv_transform.rot,
                "affine": recv_transform.affine,
            },
        )
        for ot in ots.values():
            ot.save(f, (db, base))


if __name__ == "__main__":
    main()
