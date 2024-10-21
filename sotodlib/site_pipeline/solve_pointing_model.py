import os, sys, pickle, math, h5py
import numpy as np
import argparse
import so3g.proj.quat as quat
import lmfit
from lmfit import minimize, Parameters
import yaml
import logging
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from sotodlib.site_pipeline import util as sp_util
from sotodlib import core
from sotodlib.coords import pointing_model as pm
from sotodlib.coords import fp_containers as fpc

DEG = np.pi / 180.0
ARCMIN = DEG / 60

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.5


def load_nom_centers(config):
    # Load Nominal UFM Center Locations from centered focal_plane
    ffp_path = config.get("ffp_path")
    ufms = config.get("ufms")
    nom_ufm_centers = np.zeros([7, 3]) * np.nan
    rx = fpc.Receiver.load_file(ffp_path)
    OT = rx["0"].optics_tubes[0]
    for ufm in range(len(OT.focal_planes)):
        index = ufms.index(OT.focal_planes[ufm].stream_id)
        nom_ufm_centers[index, :3] = OT.focal_planes[ufm].center

    return nom_ufm_centers


def _load_per_obs_data(config):
    # Load per-observation UFM center data points and weights
    # The per obs .h5 file a dict with obs_id for keys
    per_obs_fps = config.get("per_obs_fps")
    ufms = config.get("ufms")
    skip_tags = config.get("skip_tags", [])
    rxs = fpc.Receiver.load_file(per_obs_fps)
    filelist = list(rxs.keys())
    for skip in skip_tags:
        filelist = [obs for obs in filelist if skip not in obs]
    obs_ufm_centers = np.zeros([len(filelist), 7, 3]) * np.nan
    weights_ufm = np.zeros([len(filelist), 7])

    for i, ffp in enumerate(filelist):
        this_OT = rxs[ffp].optics_tubes[0]
        for u in range(len(this_OT.focal_planes)):
            index = ufms.index(this_OT.focal_planes[u].stream_id)
            obs_ufm_centers[i, index, :3] = this_OT.focal_planes[u].center_transformed
            weights_ufm[i, index] = np.nansum(this_OT.focal_planes[u].weights)
    weights_ufm = weights_ufm / 1720.0
    weights_ufm[weights_ufm < config.get("weight_cutoff")] = 0.0

    return filelist, obs_ufm_centers, weights_ufm


def _load_obs_boresight(config, filelist):
    # Load boresight elevation information from each observation
    # Put into an axis manager
    ctx = core.Context(config["context"]["path"])
    obs_info = [ctx.obsdb.get(obsid) for obsid in filelist]
    az_c = [obs["az_center"] for obs in obs_info]
    el_c = [obs["el_center"] for obs in obs_info]
    roll_c = [obs["roll_center"] for obs in obs_info]
    az_c = np.round(np.array(az_c), 4)
    el_c = np.round(np.array(el_c), 4)
    roll_c = np.round(np.array(roll_c), 4)
    roll_c[np.where(roll_c == 0)[0]] = 0  # rounding gives negative 0 sometimes.

    ancil = core.AxisManager(core.IndexAxis("samps"))
    ancil.wrap("az_enc", np.repeat(az_c, 7), [(0, "samps")])
    ancil.wrap("boresight_enc", np.repeat(-1 * roll_c, 7), [(0, "samps")])
    ancil.wrap("el_enc", np.repeat(el_c, 7), [(0, "samps")])

    return ancil, roll_c


def _init_fit_params(config):
    default_params = pm.defaults_sat_v1
    fixed_params = config.get("fixed_params")
    # Initialize lmfit Parameter object
    fit_params = Parameters()
    for p in list(default_params.keys()):
        fit_params.add(p, value=0.0, vary=True)
    # Turn off various parameters depending on platform
    for fix in fixed_params:
        fit_params[fix].set(vary=False)

    return fit_params


def chi_sq(weights, dist):
    N = np.identity(len(dist)) * weights
    chi2 = dist.T * N * dist
    return chi2


def model_template_xieta(params, pm_version, solver_aman):
    """
    Transform a measured (xi,eta) back into template position
    Data to Template -- modeling data as true template
    Quat math is based on this equation:
    q_nomodel * q_det_meas == q_model * q_det_true
    """
    xi_meas = solver_aman.ffp_ufm_center_fits[0]
    eta_meas = solver_aman.ffp_ufm_center_fits[1]
    if type(params) == lmfit.parameter.Parameters:
        params = params.valuesdict()
    params["version"] = pm_version
    if "sat" in pm_version:
        az, el, roll = pm._get_sat_enc_radians(solver_aman.ancil)
    boresight = pm.apply_pointing_model(solver_aman, pointing_model=params, wrap=False)
    az1, el1, roll1 = boresight.az, boresight.el, boresight.roll
    q_nomodel = quat.rotation_lonlat(-az, el, 0)
    q_model = quat.rotation_lonlat(-az1, el1, roll1)
    q_det_meas = quat.rotation_xieta(xi_meas, eta_meas, 0)
    xi_mod_true, eta_mod_true, _ = quat.decompose_xieta(
        ~q_model * q_nomodel * q_det_meas
    )

    return xi_mod_true, eta_mod_true


def model_measured_xieta(params, pm_version, solver_aman):
    """
    Transform template (xi,eta) to match measured (xi,eta).
    Template to Data -- modeling the template as measured data
    Quat math is based on this equation:
    q_nomodel * q_det_meas == q_model * q_det_true
    """
    if type(params) == lmfit.parameter.Parameters:
        params = params.valuesdict()
    params["version"] = pm_version
    xi_true, eta_true, gam_true = solver_aman.nom_ufm_centers
    if "sat" in pm_version:
        az, el, roll = pm._get_sat_enc_radians(solver_aman.ancil)
    boresight = pm.apply_pointing_model(solver_aman, pointing_model=params, wrap=False)
    az1, el1, roll1 = boresight.az, boresight.el, boresight.roll

    q_nomodel = quat.rotation_lonlat(-az, el, 0)
    q_model = quat.rotation_lonlat(-az1, el1, roll1)
    q_det_true = quat.rotation_xieta(xi_true, eta_true, 0)
    xi_mod_meas, eta_mod_meas, _ = quat.decompose_xieta(
        ~q_nomodel * q_model * q_det_true
    )

    return xi_mod_meas, eta_mod_meas


def objective_model_func_lmfit(
    params, pm_version, solver_aman, xieta_model, weights=True
):
    if xieta_model == "measured":
        xi_mod, eta_mod = model_measured_xieta(params, pm_version, solver_aman)
        xi_ref, eta_ref, _ = solver_aman.ffp_ufm_center_fits
    elif xieta_model == "template":
        xi_mod, eta_mod = model_template_xieta(params, pm_version, solver_aman)
        xi_ref, eta_ref, _ = solver_aman.nom_ufm_centers

    dist = []
    for i in range(len(xi_mod)):
        dist.append(math.dist([xi_ref[i], eta_ref[i]], [xi_mod[i], eta_mod[i]]))
    if weights:
        return chi_sq(solver_aman.weights, np.array(dist))
    else:
        return chi_sq(np.ones(len(dist)), np.array(dist))


def get_RMS(model_xieta, ref_xieta, weights):
    diff = (model_xieta[0] / ARCMIN - ref_xieta[0] / ARCMIN) ** 2 + (
        model_xieta[1] / ARCMIN - ref_xieta[1] / ARCMIN
    ) ** 2
    return (np.nansum(diff * weights) / np.nansum(weights)) ** 0.5


def _round_params(param_dict, decimal):
    P = {}
    for k in list(param_dict.keys()):
        P[k] = np.round(param_dict[k], decimal)
    return P


def _create_db(filename, save_dir):
    db_filename = os.path.join(save_dir, filename)
    # Get Database ready
    if os.path.exists(db_filename):
        return core.metadata.ManifestDb(db_filename)
    else:
        os.makedirs(save_dir, exist_ok=True)
    scheme = core.metadata.ManifestScheme()
    scheme.add_data_field("dataset")
    return core.metadata.ManifestDb(db_filename, scheme=scheme)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to Configuration File")
    return parser


def build_param_fit_stat_aman(output):
    # takes output of lmfit.minimize()
    parameter_fit_stats = core.AxisManager(core.IndexAxis("parameters"))
    parameter_fit_stats.wrap(
        "name", np.array([output.params[p].name for p in output.params])
    )
    parameter_fit_stats.wrap(
        "value", np.array([output.params[p].value for p in output.params])
    )
    parameter_fit_stats.wrap(
        "vary", np.array([output.params[p].vary for p in output.params])
    )
    parameter_fit_stats.wrap(
        "min", np.array([output.params[p].min for p in output.params])
    )
    parameter_fit_stats.wrap(
        "max", np.array([output.params[p].max for p in output.params])
    )
    parameter_fit_stats.wrap(
        "stderr", np.array([output.params[p].stderr for p in output.params])
    )
    parameter_fit_stats.wrap(
        "correl", np.array([output.params[p].correl for p in output.params])
    )
    return parameter_fit_stats


def main(config_path: str):
    # Read relevant config file info
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    platform = config.get("platform")  # e.g. satp1
    pm_version = config.get("pm_version")  # e.g. sat_v1
    sv_tag = config.get("solution_version_tag")  # e.g. YYMMDDr#
    xieta_model = config.get("xieta_model", "measured")
    xe_tag = f"{xieta_model}_xieta"
    iterate_cutoff = config.get("iterate_cutoff", None)
    append = config.get("append", "")
    append_tag = f"{bool(append)*'_'}{append}"
    save_dir = os.path.join(
        config.get("outdir"), f"{platform}_pointing_model_{sv_tag}", f"{xe_tag}{append_tag}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Logger
    logger = sp_util.init_logger(__name__, "Solve pointing_model")
    logpath = os.path.join(save_dir, "pointing_model.log")
    logfile = logging.FileHandler(logpath)
    logger.addHandler(logfile)

    if xieta_model != "measured" and xieta_model != "template":
        logger.error(
            'Not recognized xieta_model. \
                     Only "measured" or "template" accepted'
        )
        exit
    logger.info(
        "Pointing model will try to replicate (model) the %s data.", xieta_model
    )

    # Load in focal_plane and boresigt data
    nom_ufm_centers = load_nom_centers(config)
    logger.info("Loaded nominal UFM centers from %s: ", config.get("ffp_path"))
    logger.info(nom_ufm_centers)

    filelist, obs_ufm_centers, weights_ufm = _load_per_obs_data(config)
    logger.info("Loaded per-obs FFP data from %s: ", config.get("per_obs_fps"))
    logger.info("Including data from these obs:")
    logger.info(filelist)

    ancil, roll_c = _load_obs_boresight(config, filelist)
    logger.info("Loaded boresight data from obs ids.")

    # Build Axis Managers
    obs_info = core.AxisManager()
    obs_info.wrap("obs_ids", np.array(filelist))

    solver_aman = core.AxisManager(core.IndexAxis("samps"))
    solver_aman.wrap("ancil", ancil)
    solver_aman.wrap("obs_info", obs_info)
    solver_aman.wrap("roll_c", np.repeat(roll_c, 7), [(0, "samps")])
    solver_aman.wrap(
        "nom_ufm_centers",
        np.repeat([nom_ufm_centers], len(filelist), axis=0)
        .reshape(len(filelist) * 7, 3)
        .T,
        [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
        [(1, "samps")],
    )
    solver_aman.wrap(
        "ffp_ufm_center_fits",
        obs_ufm_centers.reshape(len(filelist) * 7, 3).T,
        [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
        [(1, "samps")],
    )
    solver_aman.wrap("weights", weights_ufm.reshape(-1), [(0, "samps")])

    # Make weights/data cuts
    # solver_aman.weights[solver_aman.ancil.az_enc > 360] = 0.0
    logger.info("Built axis manager")

    # Initialize Parameters to Fit with Model
    fit_params = _init_fit_params(config)
    if xieta_model == "measured":
        model_reference = solver_aman.ffp_ufm_center_fits
    elif xieta_model == "template":
        model_reference = solver_aman.nom_ufm_centers

    # Solve for Model Parameters
    # use chosen xieta_model to solve for parameters
    use_weights = True
    model_solved_params = lmfit.minimize(
        objective_model_func_lmfit,
        fit_params,
        method="nelder",
        nan_policy="omit",
        args=(pm_version, solver_aman, xieta_model, use_weights),
    )

    test_params = _round_params(model_solved_params.params.valuesdict(), 8)
    test_params["version"] = pm_version
    logger.info("Found best-fit pointing model parameters")
    logger.info(model_solved_params.params.pretty_print(precision=5, colwidth=11))

    # save pointing model parameters to axis manager
    param_aman = core.AxisManager()
    for k in list(test_params.keys()):
        param_aman.wrap(k, test_params[k])
    solver_aman.wrap("pointing_model", param_aman)

    # parameter_fit_stats = build_param_fit_stat_aman(model_solved_params)
    # solver_aman.wrap("parameter_fit_stats", parameter_fit_stats)

    # Model template and measured points using parameters found above
    if xieta_model == "measured":
        model_fits = model_measured_xieta(
            solver_aman.pointing_model, pm_version, solver_aman
        )
    elif xieta_model == "template":
        model_fits = model_template_xieta(
            solver_aman.pointing_model, pm_version, solver_aman
        )

    logger.info(
        "RMS on fit: %f", get_RMS(model_fits, model_reference, solver_aman.weights)
    )
    
    fit_residual_i1 = np.array(
        [
            math.dist(
                [model_reference[0][i], model_reference[1][i]],
                [model_fits[0][i], model_fits[1][i]],
            )
            for i in range(len(model_fits[0]))
        ]
    )

    # Save fit results to the axis manager
    modelfit_aman = core.AxisManager()
    modelfit_aman.wrap("xi", model_fits[0], overwrite=True)
    modelfit_aman.wrap("eta", model_fits[1], overwrite=True)
    solver_aman.wrap("model_fits", modelfit_aman, overwrite=True)
    solver_aman.wrap(
        "fit_rms",
        get_RMS(model_fits, model_reference, solver_aman.weights),
        overwrite=True,
    )

    if config.get("make_plots"):
        tag = "_i1"
        plot_ws0_model_fits(solver_aman, config, save_dir, tag)
        plot_template_space_fits_per_wafer(solver_aman, config, save_dir, tag)
        plot_residuals_vs_ancil(solver_aman, config, save_dir, tag)
        plot_xieta_cross_residuals(solver_aman, config, save_dir, tag)
        plot_xieta_residuals(solver_aman, config, save_dir, tag)
        

    if iterate_cutoff is not None:
        logger.info("Iterating parameter solution")

        cutoff = np.nanstd(fit_residual_i1) + np.nanmedian(fit_residual_i1)
        logger.info(f"1 std away from residual Median: {cutoff / ARCMIN} arcmin")
        logger.info(f"Using {iterate_cutoff} as cutoff")
        bad_fit_mask = np.where((fit_residual_i1 / ARCMIN) > iterate_cutoff)[0]
        logger.info("Bad fit indices:")
        logger.info(bad_fit_mask)
        logger.info(
            "%f data points are higher than %s arcmin",
            len(bad_fit_mask),
            iterate_cutoff,
        )
        bad_filename = bad_fit_mask // 7
        bad_wafer = bad_fit_mask % 7
        for mask_ind, full_ind in enumerate(bad_fit_mask):
            logger.info(
                f"{filelist[bad_filename[mask_ind]]} ws{bad_wafer[mask_ind]} is bad. Roll {solver_aman.roll_c[full_ind]}, El {solver_aman.ancil.el_enc[full_ind]}"
            )

        solver_aman.weights[bad_fit_mask] = 0.0
        use_weights = True
        model_solved_params = lmfit.minimize(
            objective_model_func_lmfit,
            fit_params,
            method="nelder",
            nan_policy="omit",
            args=(pm_version, solver_aman, xieta_model, use_weights),
        )

        test_params = _round_params(model_solved_params.params.valuesdict(), 8)
        test_params["version"] = pm_version
        logger.info("Found best-fit pointing model parameters, second iteration")
        logger.info(model_solved_params.params.pretty_print(precision=5, colwidth=11))

        # save pointing model parameters to axis manager
        param_aman = core.AxisManager()
        for k in list(test_params.keys()):
            param_aman.wrap(k, test_params[k])
        solver_aman.wrap("pointing_model", param_aman, overwrite=True)

        # parameter_fit_stats = build_param_fit_stat_aman(model_solved_params)
        # solver_aman.wrap("parameter_fit_stats", parameter_fit_stats, overwrite=True)

        # Recalculate best fit modeled points
        if xieta_model == "measured":
            model_fits = model_measured_xieta(
                solver_aman.pointing_model, pm_version, solver_aman
            )
        elif xieta_model == "template":
            model_fits = model_template_xieta(
                solver_aman.pointing_model, pm_version, solver_aman
            )
        logger.info(
            "RMS on fit: %f", get_RMS(model_fits, model_reference, solver_aman.weights)
        )
        fit_residual_i2 = np.array(
            [
                math.dist(
                    [model_reference[0][i], model_reference[1][i]],
                    [model_fits[0][i], model_fits[1][i]],
                )
                for i in range(len(model_fits[0]))
            ]
        )
        # Save fit results to the axis manager
        modelfit_aman = core.AxisManager()
        modelfit_aman.wrap("xi", model_fits[0], overwrite=True)
        modelfit_aman.wrap("eta", model_fits[1], overwrite=True)
        solver_aman.wrap("model_fits", modelfit_aman, overwrite=True)
        solver_aman.wrap(
            "fit_rms",
            get_RMS(model_fits, model_reference, solver_aman.weights),
            overwrite=True,
        )
        if config.get("make_plots"):
            tag = "_i2"
            plot_ws0_model_fits(solver_aman, config, save_dir, tag)
            plot_template_space_fits_per_wafer(solver_aman, config, save_dir, tag)
            plot_residuals_vs_ancil(solver_aman, config, save_dir, tag)
            plot_xieta_cross_residuals(solver_aman, config, save_dir, tag)
            plot_xieta_residuals(solver_aman, config, save_dir, tag)
            plot_total_residuals(solver_aman, config, save_dir,  tag, fit_residual_i1, fit_residual_i2, bad_fit_mask)
    else:
        if config.get("make_plots"):
            plot_total_residuals(solver_aman, config, save_dir,  tag='', fit_residual_i1=fit_residual_i1)
            

    if config.get("save_output"):
        # Save .h5 and ManifestDb
        h5_rel = "pointing_model_data.h5"
        h5_filename = os.path.join(save_dir, h5_rel)
        solver_aman.save(h5_filename, overwrite=True)
        dbfile = "db.sqlite"
        db = _create_db(dbfile, save_dir)
        db.add_entry({"dataset": "pointing_model"}, filename=h5_rel, replace=True)
        db.to_file(os.path.join(save_dir, dbfile))

        
####################
# Plotting Functions
####################

def plot_ws0_model_fits(solver_aman, config, save_dir, tag=""):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    platform = config.get("platform")
    plotmask = np.where(solver_aman.weights)
    rms = np.round(solver_aman.fit_rms, 4)
    xi_model_fit = solver_aman.model_fits.xi
    eta_model_fit = solver_aman.model_fits.eta
    if config.get("xieta_model") == "measured":
        xi_ref, eta_ref, _ = solver_aman.ffp_ufm_center_fits
    elif config.get("xieta_model") == "template":
        xi_ref, eta_ref, _ = solver_aman.nom_ufm_centers
    markercolor = solver_aman.ancil.el_enc
    coloredby = "El"
    scale_weights = solver_aman.weights / np.nanmax(solver_aman.weights)

    ####
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])
    ax.plot(
        solver_aman["nom_ufm_centers"][0, : 7 + 1] / DEG,
        solver_aman["nom_ufm_centers"][1, : 7 + 1] / DEG,
        "rx",
        label="Nominal Center",
    )
    ax.scatter(
        xi_ref[plotmask] / DEG,
        eta_ref[plotmask] / DEG,
        c=markercolor[plotmask],
        alpha=0.5,
        label="Data",
        edgecolors="k",
        linewidths=0.4,
        s=130 * scale_weights[plotmask],
        cmap="jet",
        vmax=65,
    )
    im = ax.scatter(
        xi_model_fit / DEG,
        eta_model_fit / DEG,
        marker="*",
        c=markercolor,
        cmap="jet",
        edgecolor="gray",
        lw=0.3,
        s=130,
        label=f"Model, RMS = {rms}",
        vmax=65,
    )
    ax.legend(loc=1, fontsize="small")
    ax.set_xlabel("Xi (deg)")
    ax.set_ylabel("Eta (deg)")
    plt.colorbar(im, location="top", fraction=0.046, pad=0.04)
    ax.set_title(f"Fits, Colored by {coloredby} (deg)\n\n\n")

    # Plot lines connecting data to modeled data point
    xitoxi = np.empty((len(xi_model_fit), 2))
    xitoxi[:, 0] = xi_ref / DEG
    xitoxi[:, 1] = xi_model_fit / DEG
    etatoeta = np.empty((len(eta_model_fit), 2))
    etatoeta[:, 0] = eta_ref / DEG
    etatoeta[:, 1] = eta_model_fit / DEG
    ax.plot(xitoxi.T, etatoeta.T, "k", lw=0.4)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    plt.subplots_adjust(left=0.1, right=0.90, bottom=0.05, hspace=0.3)
    plt.savefig(f"{plot_dir}/{platform}_ws0_model_fits{tag}.png", dpi=350)
    plt.close()


def plot_template_space_fits_per_wafer(solver_aman, config, save_dir, tag=""):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    pm_version = config.get("pm_version")
    platform = config.get("platform")
    scale_weights = solver_aman.weights / np.nanmax(solver_aman.weights)
    xi_unmod, eta_unmod = model_template_xieta(
        solver_aman.pointing_model, pm_version, solver_aman
    )
    xi0, eta0 = model_template_xieta(pm.defaults_sat_v1, pm_version, solver_aman)

    fig, ax = plt.subplots(2, 4, figsize=(9, 6))
    for i in range(7):
        ax[i // 4, i % 4].plot(0, 0, "kx", label="Nominal Center")
        im = ax[i // 4, i % 4].scatter(
            xi_unmod[i::7] / ARCMIN - solver_aman.nom_ufm_centers[0, i] / ARCMIN,
            eta_unmod[i::7] / ARCMIN - solver_aman.nom_ufm_centers[1, i] / ARCMIN,
            c=solver_aman.ancil.el_enc[i::7],
            s=scale_weights[i::7] * 80,
            marker="o",
            lw=0,
            alpha=0.5,
            cmap="jet",
        )
    plt.colorbar(im, ax[1, 3], label="Elevation (deg)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_elevation{tag}.png", dpi=350)
    plt.close()

    fig, ax = plt.subplots(2, 4, figsize=(9, 6))
    for i in range(7):
        ax[i // 4, i % 4].plot(0, 0, "kx", label="Nominal Center")
        im = ax[i // 4, i % 4].scatter(
            xi_unmod[i::7] / ARCMIN - solver_aman.nom_ufm_centers[0, i] / ARCMIN,
            eta_unmod[i::7] / ARCMIN - solver_aman.nom_ufm_centers[1, i] / ARCMIN,
            c=solver_aman.ancil.boresight_enc[i::7],
            s=scale_weights[i::7] * 80,
            marker="o",
            lw=0,
            alpha=0.5,
            cmap="jet",
        )
        ax[i // 4, i % 4].set_xlim(-15, 15)
        ax[i // 4, i % 4].set_ylim(-15, 15)
        ax[i // 4, i % 4].set_title(f"ws{i}")
    plt.colorbar(im, ax[1, 3], label="Boresight (deg)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_boresight{tag}.png", dpi=350)
    plt.close()


def plot_residuals_vs_ancil(solver_aman, config, save_dir, tag):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    platform = config.get("platform")
    scale_weights = solver_aman.weights / np.nanmax(solver_aman.weights)
    plotmask = np.where(solver_aman.weights)
    xi_model_fit = solver_aman.model_fits.xi
    eta_model_fit = solver_aman.model_fits.eta
    if config.get("xieta_model") == "measured":
        xi_ref, eta_ref, _ = solver_aman.ffp_ufm_center_fits
    elif config.get("xieta_model") == "template":
        xi_ref, eta_ref, _ = solver_aman.nom_ufm_centers

    fig, ax = plt.subplots(2, 3, figsize=(8, 6), sharex="col", sharey="row")
    plt.setp(ax[0, 1].get_yticklabels(), visible=False)
    plt.suptitle(r"$\delta \xi$, $\delta \eta$ vs Az, El, Boresight")
    for k in range(6):
        i = k // 3
        j = k % 3
        if i == 0:
            model = xi_model_fit
            ref = xi_ref
        elif i == 1:
            model = eta_model_fit
            ref = eta_ref
        if j == 0:
            x = solver_aman.ancil.az_enc % 360 
        elif j == 1:
            x = solver_aman.ancil.el_enc
        elif j == 2:
            x = solver_aman.ancil.boresight_enc
        
        ax[i, j].scatter(
            x[plotmask],
            (model - ref)[plotmask] / ARCMIN,
            color="k",
            marker=".",
            alpha=0.3,
            lw=0,
            s=scale_weights[plotmask] * 80,
        )
        ax[i, j].axhline(0, xmin=0, xmax=1, color="k", lw=2, alpha=0.5)
        mxb = np.polyfit(
            x[plotmask],
            (model[plotmask] - ref[plotmask]) / ARCMIN,
            1,
            w=scale_weights[plotmask],
        )
        xrange = np.arange(np.nanmin(x), np.nanmax(x))
        ax[i, j].plot(xrange, mxb[0] * xrange + mxb[1], "r", lw=1)
    ax[0, 0].set_ylabel("dXi [arcmin]")
    ax[1, 0].set_ylabel("dEta [arcmin]")
    ax[1, 0].set_xlabel("Azimuth [deg]")
    ax[1, 1].set_xlabel("Elevation [deg]")
    ax[1, 2].set_xlabel("Boresight [deg]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{platform}_residuals_vs_ancillary{tag}.png", dpi=350)
    plt.close()
    
def plot_total_residuals(solver_aman, config, save_dir,  tag, fit_residual_i1, fit_residual_i2=None, bad_fit_mask=None):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    iterate_cutoff = config.get("iterate_cutoff", None)
    platform = config.get("platform")
    
    if fit_residual_i2 is not None:
        fig = plt.figure(figsize=(6,4))
        gs = fig.add_gridspec(7,1)
        ax1 = fig.add_subplot(gs[0:-2,:])
        ax2 = fig.add_subplot(gs[-2:,:])
        #Plot first fit iteration residuals
        ax1.plot(np.arange(len(fit_residual_i1)), fit_residual_i1 / ARCMIN,
                'r.', mew=0, alpha=0.6, lw=0, label = '1st Fit')
        ax1.set_ylabel(f'Fit Residual $\left|\Delta$(xi, eta)$\right|$ [arcmin]')
        ax1.set_xlabel('Data point')
        ax1.axhline(iterate_cutoff, xmin=0, xmax=1, color="k", linestyle = ':',
                   lw=0.8, label = 'Cutoff')
        #Plot second fit iteration residuals
        ax1.plot(np.arange(len(fit_residual_i2)),
                fit_residual_i2 / ARCMIN, 'b*',
                alpha=0.5, lw=0, mew=0, label = '2nd fit') 
        ax1.axhline(0, xmin=0, xmax=1, color="k", alpha=0.5, lw=0.8)
        ax1.set_ylabel(r'Fit Residual $\left|\Delta\text{(xi, eta)}\right|$ [arcmin]')      
        ax1.legend(loc=2)
        
        ax2.plot(np.arange(len(fit_residual_i1)), (fit_residual_i2 - fit_residual_i1) / ARCMIN,
                 'k.', mew=0, alpha = 0.6, label = "Res i2 - Res i1")
        ax2.plot(np.arange(len(fit_residual_i1))[bad_fit_mask], (fit_residual_i2 - fit_residual_i1)[bad_fit_mask] / ARCMIN,
                 'kx', lw=0.2, alpha = 0.6, label = "Excl. from i2 fit") 
        ax2.axhline(0, xmin=0, xmax=1, color="k", alpha=0.5, lw=0.8)
        ax2.legend(fontsize='x-small')
        ax2.set_xlabel('Data points')
        ax2.set_ylabel(r'$\Delta$ Residuals')
        plt.savefig(f"{plot_dir}/{platform}_total_residuals{tag}.png", dpi=350)
        
    else:
        fig, ax = plt.subplots()
        #Plot first fit iteration residuals
        ax.plot(np.arange(len(fit_residual_i1)), fit_residual_i1 / ARCMIN,
                'r.', mew=0, alpha=0.6, lw=0, label = '1st Fit')
        ax.set_ylabel(r'Fit Residual $\left|\Delta\text{(xi, eta)}\right|$ [arcmin]')
        ax.set_xlabel('Data points')
        plt.legend(loc=2)
        plt.savefig(f"{plot_dir}/{platform}_total_residuals{tag}.png", dpi=350)
    plt.close()

def plot_xieta_residuals(solver_aman, config, save_dir, tag):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    scale_weights = solver_aman.weights / np.nanmax(solver_aman.weights)
    plotmask = np.where(solver_aman.weights)
    platform = config.get("platform")
    xi_model_fit = solver_aman.model_fits.xi
    eta_model_fit = solver_aman.model_fits.eta
    if config.get("xieta_model") == "measured":
        xi_ref, eta_ref, _ = solver_aman.ffp_ufm_center_fits
    elif config.get("xieta_model") == "template":
        xi_ref, eta_ref, _ = solver_aman.nom_ufm_centers
    
    fig, ax = plt.subplots(2, 1)
    for i, xe in enumerate(['Xi','Eta']):
        if xe == 'Xi':
            xaxis_ref = xi_ref
            xlabel = 'Xi'
            yref = xi_ref
            ymodel = xi_model_fit
            ylabel = 'dXi'        
        elif xe  == 'Eta':
            xaxis_ref = eta_ref
            xlabel = 'Eta'
            yref = eta_ref
            ymodel = eta_model_fit
            ylabel = 'dEta'
        #xi residuals vs xi
        im = ax[i].scatter(
            xaxis_ref[plotmask] / DEG,
            (ymodel - yref)[plotmask] / ARCMIN,
            marker="*",
            c=xaxis_ref[plotmask],
            cmap="jet",
            s=100 * scale_weights[plotmask],
            alpha=scale_weights[plotmask],
            edgecolors="k",
            linewidths=0.4,
        )
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.set_title(xlabel)
        ax[i].axhline(0, xmin=0, xmax=1, color="k", lw=0.8, alpha=0.6)
        ax[i].axvline(0, ymin=0, ymax=1, color="k", lw=0.8, alpha=0.5)
        ax[i].set_ylim(-10, 20)
        ax[i].set_xlabel(f"{xlabel} (deg)", fontsize="small")
        ax[i].set_ylabel(f"{ylabel} [arcmin]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{platform}_xieta_residuals{tag}.png", dpi=350)
    plt.close()


def plot_xieta_cross_residuals(solver_aman, config, save_dir, tag):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    scale_weights = solver_aman.weights / np.nanmax(solver_aman.weights)
    plotmask = np.where(solver_aman.weights)
    platform = config.get("platform")
    xi_model_fit = solver_aman.model_fits.xi
    eta_model_fit = solver_aman.model_fits.eta
    if config.get("xieta_model") == "measured":
        xi_ref, eta_ref, _ = solver_aman.ffp_ufm_center_fits
    elif config.get("xieta_model") == "template":
        xi_ref, eta_ref, _ = solver_aman.nom_ufm_centers
        
    fig, ax = plt.subplots(2, 1)
    for i, xe in enumerate(['Xi','Eta']):
        if xe == 'Xi':
            xaxis_ref = eta_ref
            xlabel = 'Eta'
            yref = xi_ref
            ymodel = xi_model_fit
            ylabel = 'dXi'        
        elif xe  == 'Eta':
            xaxis_ref = xi_ref
            xlabel = 'Xi'
            yref = eta_ref
            ymodel = eta_model_fit
            ylabel = 'dEta'
        #xi residuals vs xi
        im = ax[i].scatter(
            xaxis_ref[plotmask] / DEG,
            (ymodel - yref)[plotmask] / ARCMIN,
            marker="*",
            c=xaxis_ref[plotmask],
            cmap="jet",
            s=100 * scale_weights[plotmask],
            alpha=scale_weights[plotmask],
            edgecolors="k",
            linewidths=0.4,
        )
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.set_title(xlabel)
        ax[i].axhline(0, xmin=0, xmax=1, color="k", lw=0.8, alpha=0.6)
        ax[i].axvline(0, ymin=0, ymax=1, color="k", lw=0.8, alpha=0.5)
        ax[i].set_ylim(-10, 20)
        ax[i].set_xlabel(f"{xlabel} (deg)", fontsize="small")
        ax[i].set_ylabel(f"{ylabel} [arcmin]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{platform}_xieta_cross_residuals{tag}.png", dpi=350)
    plt.close()



############

if __name__ == "__main__":
    sp_util.main_launcher(main, get_parser)
