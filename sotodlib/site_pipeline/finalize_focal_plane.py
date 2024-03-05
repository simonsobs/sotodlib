import os
import argparse as ap
import h5py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yaml

from detmap.makemap import MapMaker
from so3g import proj
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util
from sotodlib.coords import optics as op
from sotodlib.coords import affine as af


logger = util.init_logger(__name__, "finalize_focal_plane: ")

valid_bg = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


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
    diff = measured - transformed
    diff = diff[:, np.isfinite(diff[0])]
    dist = np.linalg.norm(diff, axis=0)
    bins = int(len(dist) / 20)
    plt.hist(dist, bins=bins)
    plt.xlabel("Distance Between Measured and Transformed (rad)")
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{froot}_dist.png"))
        plt.cla()


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


def _get_wafer(ufm):
    # TODO: Switch to Matthew's code here
    try:
        wafer = MapMaker(north_is_highband=False, array_name=ufm, verbose=False)
    except ValueError:
        wafer = MapMaker(
            north_is_highband=False,
            array_name=ufm,
            verbose=False,
            use_solution_as_design=False,
        )
    det_x = []
    det_y = []
    polang = []
    det_ids = []
    template_bg = []
    is_north = []
    for det in wafer.grab_metadata():
        if not det.is_optical:
            continue
        det_x.append(det.det_x)
        det_y.append(det.det_y)
        polang.append(det.angle_actual_deg)
        det_ids.append(det.detector_id)
        template_bg.append(det.bias_line)
        is_north.append(det.is_north)
    template_bg = np.array(template_bg)
    msk = np.isin(template_bg, valid_bg)
    det_ids = np.array(det_ids)[msk]
    template_n = np.array(is_north)[msk]
    template = np.column_stack(
        (template_bg, np.array(det_x), np.array(det_y), np.array(polang))
    )[msk]

    return det_ids, template, template_n


class NoPointing(ValueError):
    pass


def _get_pointing(wafer, pointing_cfg):
    xi, eta, gamma = op.get_focal_plane(
        None, x=wafer[:, 1], y=wafer[:, 2], pol=wafer[:, 3], **pointing_cfg
    )
    pointing = wafer.copy()
    pointing[:, 1] = xi
    pointing[:, 2] = eta
    pointing[:, 3] = gamma

    return pointing


def _load_template(template_path, ufm):
    template_rset = read_dataset(template_path, ufm)
    bg = np.array(template_rset["bg"], dtype=int)
    msk = np.isin(bg, valid_bg)
    det_ids = template_rset["dets:det_id"][msk]
    template = np.column_stack(
        (
            bg.astype(float),
            np.array(template_rset["xi"]),
            np.array(template_rset["eta"]),
            np.array(template_rset["gamma"]),
        )
    )[msk]
    template_n = template_rset["is_north"][msk]

    return det_ids, template, template_n


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
            aman.append(_aman)
            have_pol.append(pol)
        if map_pointing_name in aman:
            _aman = aman.copy()
            _aman.move(map_pointing_name, "pointing")
            aman.append(_aman)
            have_pol.append(pol)
        elif tod_pointing_name not in aman:
            raise ValueError(f"No pointing found in {obs_id}")

    return amans, obs_ids, have_pol, "pointing", pol_name


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
    if "detmap" in config["resultsets"]:
        dm_rset = read_dataset(*config["resultsets"]["detmap"])
        dm_aman = dm_rset.to_axismanager(axis_key="readout_id")
        det_info.wrap("wafer", dm_aman)
        det_info.wrap("readout_id", det_info.dets.vals, [(0, det_info.dets)])
        det_info.wrap("det_id", det_info.wafer.det_id, [(0, det_info.dets)])
        det_info.restrict("dets", det_info.dets.vals[det_info.det_id != ""])
        aman.restrict("dets", aman.dets.vals[aman.det_info.det_id != "NO_MATCH"])
    aman = aman.wrap("det_info", det_info)

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

    return amans, obs_ids, have_pol, "pointing", "polarization"


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
        amans, obs_ids, have_pol, pointing_name, pol_name = _load_ctx(config)
    elif "resultsets" in config:
        amans, obs_ids, have_pol, pointing_name, pol_name = _load_rset(config)
    else:
        raise ValueError("No valid inputs provided")

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    os.makedirs(config["outdir"], exist_ok=True)
    froot = f"{ufm}{append}"
    outpath = os.path.join(config["outdir"], f"{froot}.h5")
    outpath = os.path.abspath(outpath)

    # If a template is provided load it, otherwise generate one
    (
        template_det_ids,
        template,
    ) = [], np.empty(
        (0, 0)
    )  # Just to make pyright shut up
    gen_template = "template" not in config
    if not gen_template:
        template_path = config["template"]
        if os.path.exists(template_path):
            template_det_ids, template, _ = _load_template(template_path, ufm)
        else:
            logger.error("Provided template doesn't exist, trying to generate one")
            gen_template = True
    elif gen_template:
        logger.info(f"Generating template for {ufm}")
        if "pointing_cfg" not in config:
            raise ValueError("Need pointing_cfg to generate template")
        template_det_ids, template, _ = _get_wafer(ufm)
        template = _get_pointing(template, config["pointing_cfg"])
    else:
        raise ValueError(
            "No template provided and unable to generate one for some reason"
        )

    xi = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    eta = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    gamma = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    tot_weight = np.zeros(len(template_det_ids))
    for i, (aman, obs_id, pol) in enumerate(zip(amans, obs_ids, have_pol)):
        logger.info("Working on %s", obs_id)
        if aman is None:
            raise ValueError("AxisManager doesn't exist?")

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

        # Do an initial alignment and get weights
        fp = np.vstack((_xi, _eta))
        aff, sft = af.get_affine(fp, template[template_msk, 0:3].T)
        aligned = aff @ fp + sft[..., None]
        if pol:
            _gamma = aman[pol_name].polang[msk][mapping]
            gscale, gsft = gamma_fit(_gamma, template[template_msk, 3])
            weights = af.gen_weights(
                np.vstack((aligned, gscale * _gamma + gsft)),
                template[template_msk, 1:].T,
            )
        else:
            _gamma = np.nan + np.zeros_like(_xi)
            weights = af.gen_weights(aligned, template[template_msk, 1:3].T)

        # Store weighted values
        xi[template_msk, i] = _xi * weights
        eta[template_msk, i] = _eta * weights
        gamma[template_msk, i] = _gamma * weights
        tot_weight[template_msk] += weights
    tot_weight[tot_weight == 0] = np.nan

    # Compute the average focal plane while ignoring outliers
    measured, measured_gamma, weights = _avg_focalplane(xi, eta, gamma, tot_weight)

    # Compute the lever arm
    lever_arm = np.array(
        op.get_focal_plane(None, x=0, y=0, pol=0, **config["coord_transform"])
    )

    # Compute transformation between the two nominal and measured pointing
    fp_transformed = template[:, 1:].copy()
    have_gamma = np.sum(np.isfinite(measured_gamma).astype(int)) > 10
    if have_gamma:
        gamma_scale, gamma_shift = gamma_fit(template[:, 3], measured_gamma)
        fp_transformed[:, 2] = template[:, 3] * gamma_scale + gamma_shift
    else:
        logger.warning(
            "No polarization data availible, gammas will be filled with the nominal values."
        )
        gamma_scale = 1.0
        gamma_shift = 0.0

    nominal = template[:, 1:3].T.copy()
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

    shift = (*shift, gamma_shift)
    scale = (*scale, gamma_scale)
    xieta = (shift, scale, shear, rot)
    _log_vals(shift, scale, shear, rot, ("xi", "eta", "gamma"))

    plot = config.get("plot", False)
    if plot:
        _mk_plot(config.get("plot_dir", None), froot, nominal, measured, transformed)

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
