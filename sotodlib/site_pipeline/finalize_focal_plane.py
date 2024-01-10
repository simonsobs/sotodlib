import os
import sys
import argparse as ap
import h5py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yaml

from so3g import proj
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util
from sotodlib.site_pipeline import make_position_match as mpm
from sotodlib.coords import optics as op
from sotodlib.coords import affine as af


logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _encs_notclose(az, el, bs):
    return not (
        np.isclose(az, az[0], equal_nan=True).all()
        and np.isclose(el, el[0], equal_nan=True).all()
        and np.isclose(bs, bs[0], equal_nan=True).all()
    )


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


def _mk_tpout(xieta, horiz):
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
    horiz = (*horiz[0], *horiz[1], *horiz[2:])
    tpout = np.array([xieta, horiz], outdt)

    return tpout


def _mk_refout(lever_arm, encoders):
    outdt = [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
    ]
    refout = np.array([tuple(np.squeeze(lever_arm)), tuple(encoders)], outdt)

    return refout


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


def _mk_plot(plot_dir, froot, nominal, measured, transformed):
    plt.style.use("tableau-colorblind10")
    plt.scatter(measured[0], measured[1], alpha=0.2, color="orange", label="fit")
    plt.scatter(nominal[0], nominal[1], alpha=0.2, color="blue", label="nominal")
    plt.scatter(
        transformed[0], transformed[1], alpha=0.2, color="black", label="transformed"
    )
    plt.xlabel("Xi (rad)")
    plt.ylabel("Eta (rad)")
    plt.legend()
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{froot}.png"))
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


def to_horiz(points, encoders):
    """
    Go from xieta coordinates to horizon coordinates.

    Arguments:

        points: (3, ndet) array of points.
                Rows should be (xi, eta, gamma).

        encoders: The encoder positions for the LOS.
                  Should be (az, el, bs).

    Returns:

        horiz : (3, ndet) array if points.
                Rows are (az, el, bs).
    """
    msk = np.isfinite(points).all(axis=0)
    good_points = points[:, msk]
    dets = np.arange(good_points.shape[1])
    fp = proj.FocalPlane.from_xieta(dets, *good_points)
    sight = proj.CelestialSightLine.for_horizon([0], *np.atleast_2d(encoders).T)
    asm = proj.Assembly.attach(sight, fp)
    output = np.zeros((len(dets), 1, 4))
    projectionist = proj.Projectionist()
    projectionist.get_coords(asm, output=output)

    # Get rid of the time axis and transpose
    output = np.squeeze(output, axis=1).T
    # Fix sign on az
    output[0] *= -1
    # Compute BS (need to check sign and 0 point)
    bs = np.arctan2(output[3], output[2]) % (2 * np.pi)

    _horiz = np.vstack((output[:2], bs))
    horiz = np.zeros((3, len(msk))) + np.nan
    horiz[:, msk] = _horiz

    return horiz


def _load_ctx(config):
    ctx = Context(config["context"]["path"])
    pm_name = config["context"].get("position_match", "position_match")
    pointing_name = config["context"].get("position_match", "pointing")
    pol_name = config["context"].get("position_match", "polarization")
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
    amans = [None] * len(obs_ids)
    have_pol = [False] * len(obs_ids)
    for i, obs_id in enumerate(obs_ids):
        _config["context"]["obs_ids"] = [obs_id]
        aman, _, pol, *_ = mpm._load_ctx(_config)
        if pm_name not in aman:
            raise ValueError(f"No position match in {obs_id}")
        if "det_info" not in aman or "det_id" not in aman.det_info:
            raise ValueError(f"No detmap for {obs_id}")
        amans[i] = aman
        have_pol[i] = pol

    return amans, obs_ids, have_pol, pointing_name, pol_name, pm_name


def _load_rset(config):
    obs = config["resultsets"]
    _config = config.copy()
    obs_ids = np.array(list(obs.keys()))
    amans = [None] * len(obs_ids)
    have_pol = [False] * len(obs_ids)
    for i, (obs_id, rsets) in enumerate(obs.items()):
        _config["resultsets"] = rsets
        _config["resultsets"]["obs_id"] = obs_id
        aman, _, pol, *_ = mpm._load_rset(_config)
        if "det_info" not in aman or "det_id" not in aman.det_info:
            raise ValueError(f"No detmap for {obs_id}")
        if "position_match" not in rsets:
            raise ValueError(f"No position match in {obs_id}")
        pm_aman = read_dataset(*rsets["position_match"]).to_axismanager(
            axis_key="dets:readout_id"
        )
        aman.wrap("position_match", pm_aman)
        amans[i] = aman
        have_pol[i] = pol

    return amans, obs_ids, have_pol, "pointing", "polarization", "position_match"


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
        amans, obs_ids, have_pol, pointing_name, pol_name, pm_name = _load_ctx(config)
    elif "resultsets" in config:
        amans, obs_ids, have_pol, pointing_name, pol_name, pm_name = _load_rset(config)
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
            template_det_ids, template, _ = mpm._load_template(template_path, ufm)
        else:
            logger.error("Provided template doesn't exist, trying to generate one")
            gen_template = True
    elif gen_template:
        logger.info(f"Generating template for {ufm}")
        if "pointing_cfg" not in config:
            raise ValueError("Need pointing_cfg to generate template")
        template_det_ids, template, _ = mpm._get_wafer(ufm)
        template = mpm._get_pointing(template, config["pointing_cfg"])
    else:
        raise ValueError(
            "No template provided and unable to generate one for some reason"
        )

    check_enc = config.get("check_enc", False)
    encoders = np.nan + np.zeros((3, len(obs_ids)))
    use_matched = config.get("use_matched", False)

    xi = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    eta = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    gamma = np.nan + np.zeros((len(template_det_ids), len(obs_ids)))
    tot_weight = np.zeros(len(template_det_ids))
    for i, (aman, obs_id, pol) in enumerate(zip(amans, obs_ids, have_pol)):
        logger.info("Working on %s", obs_id)
        if aman is None:
            raise ValueError("AxisManager doesn't exist?")
        # Cut outliers
        aman.restrict("dets", aman.dets.vals[~aman[pm_name].pointing_outlier])

        # Mapping to template
        if use_matched:
            det_ids = aman[pm_name].matched_det_id
        else:
            det_ids = aman.det_info.det_id
        _, msk, template_msk = np.intersect1d(
            det_ids, template_det_ids, return_indices=True
        )
        if np.sum(msk) != aman.dets.count:
            logger.warning("There are matched dets not found in the template")
        mapping = np.argsort(np.argsort(template_det_ids[template_msk]))
        srt = np.argsort(det_ids[msk])
        _xi = aman[pointing_name].xi[msk][srt][mapping]
        _eta = aman[pointing_name].eta[msk][srt][mapping]
        if pol:
            _gamma = aman[pol_name].polang[msk][mapping]
        else:
            _gamma = np.nan + np.zeros_like(_xi)
        weights = aman[pm_name].likelihood[msk][srt][mapping]

        # Store weighted values
        xi[template_msk, i] = _xi * weights
        eta[template_msk, i] = _eta * weights
        gamma[template_msk, i] = _gamma * weights
        tot_weight[template_msk] += weights

        if not check_enc:
            continue
        if (
            "az" not in aman[pointing_name]
            or "el" not in aman[pointing_name]
            or "roll" not in aman[pointing_name]
        ):
            raise ValueError("No encoder values included with pointing")
        az, el, roll = (
            aman[pointing_name].az,
            aman[pointing_name].el,
            aman[pointing_name].roll,
        )
        if _encs_notclose(az, el, roll):
            raise ValueError("Encoder values not close")
        encoders[0, i] = np.nanmedian(az)
        encoders[1, i] = np.nanmedian(el)
        encoders[2, i] = np.nanmedian(roll)
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
    # Center on UFM center
    nominal -= lever_arm[:2]
    _measured = measured.copy() - lever_arm[:2]
    # Compute transform, do a few iters to account for centers being off
    affine = np.eye(2)
    shift = af.weighted_shift(nominal, _measured, weights)
    for i in range(config.get("iters", 5)):
        affine, _ = af.get_affine(nominal, _measured - shift[..., None], centered=True)
        shift = af.weighted_shift(affine @ nominal, _measured, weights)
    nominal = template[:, 1:3].T.copy()

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

    # Compute nominal encoder vals
    if check_enc:
        if _encs_notclose(*encoders):
            raise ValueError("Not all encoder values are similar")
        else:
            encoders = np.nanmedian(encoders, axis=1)
            if np.isnan(encoders).any():
                logger.error("Some or all of the encoders are nan")
                horiz = ((np.nan,) * 3,) * 2 + (np.nan,) * 2
                horiz_affine = np.nan * np.empty_like(affine)
            else:
                # Put nominal and measured into horiz basis and compute transformation.
                nominal = np.vstack((nominal, template[:, 3]))
                _measured = np.vstack((measured - lever_arm[..., None], measured_gamma))
                nominal_horiz = to_horiz(nominal, encoders)
                measured_horiz = to_horiz(_measured, encoders)

                if have_gamma:
                    bs_scale, bs_shift = gamma_fit(nominal_horiz[2], measured_horiz[2])
                else:
                    bs_scale = 1.0
                    bs_shift = 0.0
                nominal_horiz = nominal_horiz[:2]
                measured_horiz = measured_horiz[:2]
                # Compute transform, do a few iters to account for centers being off
                horiz_affine = np.eye(2)
                horiz_shift = af.weighted_shift(nominal_horiz, measured_horiz, weights)
                for i in range(config.get("iters", 5)):
                    horiz_affine, _ = af.get_affine(
                        nominal, measured_horiz - horiz_shift[..., None], centered=True
                    )
                    horiz_shift = af.weighted_shift(
                        horiz_affine @ nominal, measured_horiz, weights
                    )
                horiz_scale, horiz_shear, horiz_rot = op.decompose_affine(horiz_affine)
                horiz_shear = horiz_shear.item()
                horiz_rot = op.decompose_rotation(horiz_rot)[-1]

                horiz_shift = (*horiz_shift, bs_shift)
                horiz_scale = (*horiz_scale, bs_scale)
                horiz = (horiz_shift, horiz_scale, horiz_shear, horiz_rot)
                _log_vals(
                    horiz_shift, horiz_scale, horiz_shear, horiz_rot, ("az", "el", "bs")
                )
    else:
        encoders = np.nan + np.zeros(3)
        horiz = ((np.nan,) * 3,) * 2 + (np.nan,) * 2
        horiz_affine = np.nan * np.empty_like(affine)

    # Make final outputs and save
    logger.info("Saving data to %s", outpath)
    fpout, fpfullout = _mk_fpout(
        template_det_ids, fp_transformed, measured, measured_gamma
    )
    tpout = _mk_tpout(xieta, horiz)
    refout = _mk_refout(lever_arm, encoders)
    with h5py.File(outpath, "w") as f:
        write_dataset(fpout, f, "focal_plane", overwrite=True)
        _add_attrs(f["focal_plane"], {"measured_gamma": measured_gamma})
        write_dataset(fpfullout, f, "focal_plane_full", overwrite=True)
        write_dataset(tpout, f, "offsets", overwrite=True)
        _add_attrs(f["offsets"], {"affine_xieta": affine, "affine_horiz": horiz_affine})
        write_dataset(refout, f, "reference", overwrite=True)


if __name__ == "__main__":
    main()
