import argparse as ap
import os
from dataclasses import InitVar, dataclass, field
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.cluster import vq
from scipy.optimize import minimize
from scipy.spatial import transform
from sotodlib.coords import affine as af
from sotodlib.coords import optics as op
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__, "finalize_focal_plane: ")


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
    n_aman: int
    full_fp: NDArray[np.floating] = field(init=False)  # (ndim, ndet, n_aman)
    tot_weight: NDArray[np.floating] = field(init=False)  # (ndet,)
    avg_fp: NDArray[np.floating] = field(init=False)  # (ndim, ndet)
    weights: NDArray[np.floating] = field(init=False)  # (ndet,)
    transformed: NDArray[np.floating] = field(init=False)  # (ndim, ndet)
    centers_transformed: NDArray[np.floating] = field(init=False)  # (ndim, 1)

    def __post_init__(self):
        self.full_fp = np.nan + np.empty(self.template.fp.shape + (self.n_aman,))
        self.tot_weight = np.zeros(len(self.template.det_ids))
        self.avg_fp = np.nan + np.empty_like(self.template.fp)
        self.weight = np.zeros(len(self.template.det_ids))
        self.transformed = self.template.fp.copy()
        self.center_transformed = self.template.center.copy()

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
            gamma = np.nan + np.empty(len(xi))
        fp = np.vstack((xi, eta, gamma))
        return fp, template_msk

    def add_fp(self, i, fp, weights, template_msk):
        self.full_fp[:, template_msk, i] = fp * weights
        self.tot_weight[template_msk] += weights


def _avg_focalplane(full_fp, tot_weight, n_obs):
    tot_weight[tot_weight == 0] = np.nan
    avg_fp = np.nansum(full_fp, axis=-1) / tot_weight
    avg_weight = tot_weight / n_obs

    # nansum all all nans is 0, addressing that case here
    all_nan = ~np.any(np.isfinite(full_fp).reshape((len(full_fp), -1)), axis=1)
    avg_fp[all_nan] = np.nan

    return avg_fp, avg_weight


def _log_vals(shift, scale, shear, rot, axis):
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    for ax, s in zip(axis, shift):
        logger.info("Shift along %s axis is %f", ax, s)
    for ax, s in zip(axis, scale):
        logger.info("Scale along %s axis is %f", ax, s)
        if np.isclose(s, deg2rad):
            logger.warning(
                "Scale factor for %s looks like a degrees to radians conversion", ax
            )
        elif np.isclose(s, rad2deg):
            logger.warning(
                "Scale factor for %s looks like a radians to degrees conversion", ax
            )
    logger.info("Shear param is %f", shear)
    logger.info("Rotation of the %s-%s plane is %f radians", axis[0], axis[1], rot)


def _mk_fpout(det_id, transformed, measured):
    outdt = [
        ("dets:det_id", det_id.dtype),
        ("xi", np.float32),
        ("eta", np.float32),
        ("gamma", np.float32),
    ]
    fpout = np.fromiter(zip(det_id, *transformed), dtype=outdt, count=len(det_id))

    outdt_full = [
        ("dets:det_id", det_id.dtype),
        ("xi_t", np.float32),
        ("eta_t", np.float32),
        ("gamma_t", np.float32),
        ("xi_m", np.float32),
        ("eta_m", np.float32),
        ("gamma_m", np.float32),
    ]
    fpfullout = np.fromiter(
        zip(det_id, *transformed, *measured),
        dtype=outdt_full,
        count=len(det_id),
    )

    return metadata.ResultSet.from_friend(fpout), metadata.ResultSet.from_friend(
        fpfullout
    )


def _mk_tpout(xieta):
    outdt = [
        ("d_x", np.float32),
        ("d_y", np.float32),
        ("d_z", np.float32),
        ("s_x", np.float32),
        ("s_y", np.float32),
        ("s_z", np.float32),
        ("shear", np.float32),
        ("rot", np.float32),
    ]
    xieta = (*xieta[0], *xieta[1], *xieta[2:])
    tpout = np.array([xieta], outdt)

    return tpout


def _mk_refout(center, center_transformed):
    outdt = [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
    ]
    refout = np.array(
        [tuple(np.squeeze(center)), tuple(np.squeeze(center_transformed))], outdt
    )

    return refout


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


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

    # Historgram of differences
    diff = measured - transformed
    dist = np.linalg.norm(diff[:2, np.isfinite(diff[0])], axis=0)
    bins = int(len(dist) / 20)
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
        axs[i].set_xlim(np.nanmin(nominal[0]), np.nanmax(nominal[0]))
        axs[i].set_ylim(np.nanmin(nominal[1]), np.nanmax(nominal[1]))
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
        plt.clf()


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
    dets = {"stream_id": f"ufm_{config['ufm'].lower()}"}
    dets.update(config["context"].get("dets", {}))
    for obs_id in obs_ids:
        aman = ctx.get_meta(obs_id, dets=dets)
        if "wafer" not in aman.det_info and dm_name in aman:
            dm_aman = aman[dm_name].copy()
            aman.det_info.wrap("wafer", dm_aman)
            if "det_id" not in aman.det_info:
                aman.det_info.wrap(
                    "det_id", aman.det_info.wafer.det_id, [(0, aman.dets)]
                )
        if "det_id" in aman.det_info:
            aman.restrict("dets", aman.dets.vals[aman.det_info.det_id != ""])
            aman.restrict("dets", aman.dets.vals[aman.det_info.det_id != "NO_MATCH"])
        elif "det_info" not in aman or "det_id" not in aman.det_info:
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

    return (
        amans,
        obs_ids,
        amans[0].obs_info.telescope_flavor,
        amans[0].obs_info.tube_slot,
        amans[0].det_info.wafer_slot[0],
    )


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
    det_info.restrict("dets", det_info.dets.vals[det_info.det_id != ""])
    det_info.det_id = np.char.strip(det_info.det_id)  # Needed for some old results
    aman = aman.wrap("det_info", det_info)
    aman.restrict("dets", aman.dets.vals[aman.det_info.det_id != "NO_MATCH"])

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
    obs = config["resultsets"]
    _config = config.copy()
    obs_ids = np.array(list(obs.keys()))
    amans = [None] * len(obs_ids)
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
        config["telescope_flavor"],
        config["tube_slot"],
        config["wafer_slot"],
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
        amans, obs_ids, tel, ot, ws = _load_ctx(config)
    elif "resultsets" in config:
        amans, obs_ids, tel, ot, ws = _load_rset(config)
    else:
        raise ValueError("No valid inputs provided")

    # Generate pointing config
    pointing_cfg = _mk_pointing_config(tel, ot, ws, config)

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    froot = f"{ufm}{append}"
    subdir = config.get("subdir", None)
    if subdir is None:
        subdir = "combined"
        if len(obs_ids) == 1:
            subdir = obs_ids[0]
    outpath = os.path.join(config["outdir"], subdir, f"{froot}.h5")
    outpath = os.path.abspath(outpath)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # If a template is provided load it, otherwise generate one
    gen_template = "template" not in config
    template_path = config.get("template", "nominal.h5")
    have_template = os.path.exists(template_path)
    if not gen_template and not have_template:
        logger.error("Provided template doesn't exist, trying to generate one")
        gen_template = True
    if gen_template:
        logger.info(f"Generating template for {ufm}")
        if "wafer_info" not in config:
            raise ValueError("Need wafer_info to generate template")
        template_det_ids, template, is_optical = op.gen_template(
            config["wafer_info"], config["ufm"], **pointing_cfg
        )
        template = Template(template_det_ids, template.T, is_optical, pointing_cfg)
    elif have_template:
        logger.info("Loading template from %s", template_path)
        template = _load_template(template_path, ufm, pointing_cfg)
    else:
        raise ValueError(
            "No template provided and unable to generate one for some reason"
        )

    focal_plane = FocalPlane(template, len(amans))
    for i, (aman, obs_id) in enumerate(zip(amans, obs_ids)):
        logger.info("Working on %s", obs_id)
        if aman is None:
            raise ValueError("AxisManager doesn't exist?")

        # Restrict to optical dets
        optical = np.isin(
            aman.det_info.det_id, focal_plane.template.det_ids[template.optical]
        )
        aman.restrict("dets", aman.dets.vals[optical])

        # Do some outlier cuts
        _restrict_inliers(aman, focal_plane)

        # Mapping to template
        fp, template_msk = focal_plane.map_to_template(aman)

        # Try an initial alignment and get weights
        aff, sft = af.get_affine(fp[:2], focal_plane.template.fp[:2, template_msk])
        aligned = aff @ fp[:2] + sft[..., None]
        if np.any(np.isfinite(fp[2])):
            gscale, gsft = gamma_fit(fp[2], focal_plane.template.fp[2, template_msk])
            weights = af.gen_weights(
                np.vstack((aligned, gscale * fp[2] + gsft)),
                focal_plane.template.fp[:, template_msk],
                focal_plane.template.spacing.ravel() / 10,
            )
        else:
            weights = af.gen_weights(
                aligned,
                focal_plane.template.fp[:2, template_msk],
                focal_plane.template.spacing[:2].ravel() / 10,
            )

        # Store weighted values
        focal_plane.add_fp(i, fp, weights, template_msk)

    # Compute the average focal plane with weights
    focal_plane.avg_fp, focal_plane.weights = _avg_focalplane(
        focal_plane.full_fp, focal_plane.tot_weight, focal_plane.n_aman
    )

    # Compute transformation between the two nominal and measured pointing
    have_gamma = np.sum(np.isfinite(focal_plane.avg_fp[2]).astype(int)) > 10
    if have_gamma:
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
            "No polarization data availible, gammas will be filled with the nominal values."
        )
        gamma_scale = 1.0
        gamma_shift = 0.0

    nominal = focal_plane.template.fp[:2].copy()
    # Do an initial alignment without weights
    affine_0, shift_0 = af.get_affine(nominal, focal_plane.avg_fp[:2])
    init_align = affine_0 @ nominal + shift_0[..., None]
    # Now compute the actual transform
    affine, shift = af.get_affine_weighted(
        init_align, focal_plane.avg_fp[:2], focal_plane.weights
    )
    affine = affine @ affine_0
    shift += (affine @ shift_0[..., None])[:, 0]

    scale, shear, rot = af.decompose_affine(affine)
    shear = shear.item()
    rot = af.decompose_rotation(rot)[-1]
    focal_plane.transformed[:2] = affine @ nominal + shift[..., None]
    focal_plane.center_transformed[:2] = (
        affine @ focal_plane.template.center[:2] + shift[..., None]
    )

    rms = np.sqrt(np.nanmean((focal_plane.avg_fp - focal_plane.transformed) ** 2))
    logger.info("RMS after transformation is %f", rms)

    shift = (*shift, gamma_shift)
    scale = (*scale, gamma_scale)
    xieta = (shift, scale, shear, rot)
    _log_vals(shift, scale, shear, rot, ("xi", "eta", "gamma"))

    if config.get("plot", False):
        plot_dir = config.get("plot_dir", None)
        if plot_dir is not None:
            plot_dir = os.path.join(plot_dir, subdir)
            plot_dir = os.path.abspath(plot_dir)
            os.makedirs(plot_dir, exist_ok=True)
        _mk_plot(
            plot_dir,
            froot,
            focal_plane.template.fp,
            focal_plane.avg_fp,
            focal_plane.transformed,
        )

    # Make final outputs and save
    logger.info("Saving data to %s", outpath)
    fpout, fpfullout = _mk_fpout(
        focal_plane.template.det_ids, focal_plane.transformed, focal_plane.avg_fp
    )
    tpout = _mk_tpout(xieta)
    refout = _mk_refout(focal_plane.template.center, focal_plane.center_transformed)
    with h5py.File(outpath, "w") as f:
        write_dataset(fpout, f, "focal_plane", overwrite=True)
        _add_attrs(f["focal_plane"], {"measured_gamma": have_gamma})
        write_dataset(fpfullout, f, "focal_plane_full", overwrite=True)
        write_dataset(tpout, f, "offsets", overwrite=True)
        _add_attrs(f["offsets"], {"affine_xieta": affine})
        write_dataset(refout, f, "reference", overwrite=True)


if __name__ == "__main__":
    main()
