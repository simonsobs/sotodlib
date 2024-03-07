import argparse as ap
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.cluster import vq
from scipy.optimize import minimize
from sotodlib.coords import affine as af
from sotodlib.coords import optics as op
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _avg_focalplane(xi, eta, gamma, tot_weight):
    avg_xi = np.nansum(xi, axis=1) / tot_weight
    avg_eta = np.nansum(eta, axis=1) / tot_weight
    focal_plane = np.vstack((avg_xi, avg_eta))

    if np.any(np.isfinite(gamma)):
        avg_gamma = np.nansum(gamma, axis=1) / tot_weight
    else:
        avg_gamma = np.nan + np.zeros_like(avg_xi)

    n_obs = np.sum(np.isfinite(xi).astype(int), axis=1)
    avg_weight = tot_weight / n_obs

    return focal_plane, avg_gamma, avg_weight


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


def _mk_fpout(det_id, transformed, measured, measured_gamma):
    outdt = [
        ("dets:det_id", det_id.dtype),
        ("xi", np.float32),
        ("eta", np.float32),
        ("gamma", np.float32),
    ]
    fpout = np.fromiter(zip(det_id, *transformed.T), dtype=outdt, count=len(det_id))

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
        zip(det_id, *transformed.T, *measured, measured_gamma),
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


def _mk_refout(lever_arm):
    outdt = [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
    ]
    refout = np.array([tuple(np.squeeze(lever_arm))], outdt)

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

    def _gamma_min(scale, shift, gamma):
        src, dst = gamma
        transformed = np.sin(src * scale + shift)
        diff = np.sin(dst) - transformed

        return np.sqrt(np.mean(diff**2))

    res = minimize(_gamma_min, (1.0, 0.0), (src, dst))
    return res.x


def _load_template(template_path, ufm):
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

    return np.array(det_ids), template, np.array(template_optical)


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
    have_pol = []
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
        if not pol:
            logger.warning("No polarization data in context")

        if tod_pointing_name in aman:
            _aman = aman.copy()
            _aman.move(tod_pointing_name, "pointing")
            amans.append(_aman)
            have_pol.append(pol)
        if map_pointing_name in aman:
            _aman = aman.copy()
            _aman.move(map_pointing_name, "pointing")
            amans.append(_aman)
            have_pol.append(pol)
        elif tod_pointing_name not in aman:
            raise ValueError(f"No pointing found in {obs_id}")

    return (
        amans,
        obs_ids,
        have_pol,
        "pointing",
        pol_name,
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

    pol = False
    if "polarization" in config["resultsets"]:
        polarization_rset = read_dataset(*config["resultsets"]["polarization"])
        polarization_aman = polarization_rset.to_axismanager(axis_key="dets:readout_id")
        aman = aman.wrap("polarization", polarization_aman)
        pol = True

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

    return aman, obs_id, pol, "pointing", "polarization"


def _load_rset(config):
    obs = config["resultsets"]
    _config = config.copy()
    obs_ids = np.array(list(obs.keys()))
    amans = [None] * len(obs_ids)
    have_pol = [False] * len(obs_ids)
    for i, (obs_id, rsets) in enumerate(obs.items()):
        _config["resultsets"] = rsets
        _config["resultsets"]["obs_id"] = obs_id
        aman, _, pol, *_ = _load_rset_single(_config)
        if "det_info" not in aman or "det_id" not in aman.det_info:
            raise ValueError(f"No detmap for {obs_id}")
        amans[i] = aman
        have_pol[i] = pol

    return (
        amans,
        obs_ids,
        have_pol,
        "pointing",
        "polarization",
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


def _restrict_inliers(aman, template):
    focal_plane = np.column_stack((aman.pointing.xi, aman.pointing.eta))
    inliers = np.ones(len(focal_plane), dtype=bool)

    cent = np.nanmedian(template[:, :2], axis=0)
    rad_thresh = 1.05 * np.nanmax(np.linalg.norm(template[:, :2] - cent, axis=1))

    # Use kmeans to kill any ghosts
    fp_white = vq.whiten(focal_plane[inliers])
    codebook, _ = vq.kmeans(fp_white, 2)
    codes, _ = vq.vq(fp_white, codebook)

    c0 = codes == 0
    c1 = codes == 1
    m0 = np.median(focal_plane[inliers][c0], axis=0)
    m1 = np.median(focal_plane[inliers][c1], axis=0)
    dist = np.linalg.norm(m0 - m1)

    # If centroids are too far from each other use the bigger one
    if dist < rad_thresh:
        cluster = c0 + c1
    elif np.sum(c0) >= np.sum(c1):
        cluster = c0
    else:
        cluster = c1

    # Flag anything too far away from the center
    cent = np.median(focal_plane[inliers][cluster], axis=0)
    r = np.linalg.norm(focal_plane[inliers] - cent, axis=1)
    inliers[inliers] *= cluster * (r <= rad_thresh)

    # Now restrict the AxisManager
    return aman.restrict("dets", aman.dets.vals[inliers])


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
        amans, obs_ids, have_pol, pointing_name, pol_name, tel, ot, ws = _load_ctx(
            config
        )
    elif "resultsets" in config:
        amans, obs_ids, have_pol, pointing_name, pol_name, tel, ot, ws = _load_rset(
            config
        )
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
    (template_det_ids, template, is_optical) = (
        np.empty(0, dtype=str),
        np.empty((0, 0)),
        np.empty(0, dtype=bool),
    )  # Just to make pyright shut up
    gen_template = "template" not in config
    if not gen_template:
        template_path = config["template"]
        if os.path.exists(template_path):
            template_det_ids, template, is_optical = _load_template(template_path, ufm)
        else:
            logger.error("Provided template doesn't exist, trying to generate one")
            gen_template = True
    elif gen_template:
        logger.info(f"Generating template for {ufm}")
        if "wafer_info" not in config:
            raise ValueError("Need wafer_info to generate template")
        template_det_ids, template, is_optical = op.gen_template(
            config["wafer_info"], config["ufm"], **pointing_cfg
        )
    else:
        raise ValueError(
            "No template provided and unable to generate one for some reason"
        )
    optical_det_ids = template_det_ids[is_optical]
    template_spacing = af.get_spacing(template[is_optical, :2].T)

    xi = np.nan + np.zeros((len(template_det_ids), len(amans)))
    eta = np.nan + np.zeros((len(template_det_ids), len(amans)))
    gamma = np.nan + np.zeros((len(template_det_ids), len(amans)))
    tot_weight = np.zeros(len(template_det_ids))
    for i, (aman, obs_id, pol) in enumerate(zip(amans, obs_ids, have_pol)):
        logger.info("Working on %s", obs_id)
        if aman is None:
            raise ValueError("AxisManager doesn't exist?")

        # Restrict to optical dets
        optical = np.isin(aman.det_info.det_id, optical_det_ids)
        aman.restrict("dets", aman.dets.vals[optical])

        # Do some outlier cuts
        _restrict_inliers(aman, template)

        # Mapping to template
        det_ids = aman.det_info.det_id
        _, msk, template_msk = np.intersect1d(
            det_ids, template_det_ids, return_indices=True
        )
        if len(msk) != aman.dets.count:
            logger.warning("There are matched dets not found in the template")
        mapping = np.argsort(np.argsort(template_det_ids[template_msk]))
        srt = np.argsort(det_ids[msk])
        _xi = aman[pointing_name].xi[msk][srt][mapping]
        _eta = aman[pointing_name].eta[msk][srt][mapping]
        fp = np.vstack((_xi, _eta))

        # Kill dets that are really far from their matched det
        # TODO: If we include an initial rotation of the template this could just be a function of template spacing
        dist = np.linalg.norm(fp - template[template_msk, :2].T, axis=0)
        med_dist = np.nanmedian(dist)
        fp[:, dist > med_dist + 5 * np.nanstd(dist)] = np.nan
        logger.info("Median distance to matched det is %f", med_dist)
        ratio = med_dist / template_spacing
        logger.info("Median distance to matched det is %f the template spacing", ratio)

        # Try an initial alignment and get weights
        aff, sft = af.get_affine(fp, template[template_msk, :2].T)
        aligned = aff @ fp + sft[..., None]
        aligned = fp
        if pol:
            _gamma = aman[pol_name].polang[msk][mapping]
            gscale, gsft = gamma_fit(_gamma, template[template_msk, 2])
            weights = af.gen_weights(
                np.vstack((aligned, gscale * _gamma + gsft)),
                template[template_msk].T,
            )
        else:
            _gamma = np.nan + np.zeros_like(_xi)
            weights = af.gen_weights(aligned, template[template_msk, :2].T)

        # ~2 sigma cut
        weights[weights < 0.95] = 0

        # Store weighted values
        xi[template_msk, i] = fp[0] * weights
        eta[template_msk, i] = fp[1] * weights
        gamma[template_msk, i] = _gamma * weights
        tot_weight[template_msk] += weights
    tot_weight[tot_weight == 0] = np.nan

    # Compute the average focal plane while ignoring outliers
    measured, measured_gamma, weights = _avg_focalplane(xi, eta, gamma, tot_weight)

    # Compute the lever arm
    lever_arm = np.array(op.get_focal_plane(None, x=0, y=0, pol=0, **pointing_cfg))

    # Compute transformation between the two nominal and measured pointing
    fp_transformed = template.copy()
    have_gamma = np.sum(np.isfinite(measured_gamma).astype(int)) > 10
    if have_gamma:
        gamma_scale, gamma_shift = gamma_fit(template[:, 2], measured_gamma)
        fp_transformed[:, 2] = template[:, 2] * gamma_scale + gamma_shift
    else:
        logger.warning(
            "No polarization data availible, gammas will be filled with the nominal values."
        )
        gamma_scale = 1.0
        gamma_shift = 0.0

    nominal = template[:, :2].T.copy()
    # Do an initial alignment without weights
    affine_0, shift_0 = af.get_affine(nominal, measured)
    init_align = affine_0 @ nominal + shift_0[..., None]
    # Now compute the actual transform
    affine, shift = af.get_affine_weighted(init_align, measured, weights)
    affine = affine @ affine_0
    shift += (affine @ shift_0[..., None])[:, 0]

    scale, shear, rot = af.decompose_affine(affine)
    shear = shear.item()
    rot = af.decompose_rotation(rot)[-1]
    transformed = affine @ nominal + shift[..., None]
    fp_transformed[:, :2] = transformed.T

    rms = np.sqrt(np.nanmean((measured - transformed) ** 2))
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
            nominal,
            np.vstack((measured, measured_gamma)),
            fp_transformed.T,
        )

    # Make final outputs and save
    logger.info("Saving data to %s", outpath)
    fpout, fpfullout = _mk_fpout(
        template_det_ids, fp_transformed, measured, measured_gamma
    )
    tpout = _mk_tpout(xieta)
    refout = _mk_refout(lever_arm)
    with h5py.File(outpath, "w") as f:
        write_dataset(fpout, f, "focal_plane", overwrite=True)
        _add_attrs(f["focal_plane"], {"measured_gamma": measured_gamma})
        write_dataset(fpfullout, f, "focal_plane_full", overwrite=True)
        write_dataset(tpout, f, "offsets", overwrite=True)
        _add_attrs(f["offsets"], {"affine_xieta": affine})
        write_dataset(refout, f, "reference", overwrite=True)


if __name__ == "__main__":
    main()
