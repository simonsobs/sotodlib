import os, sys, pickle, math, h5py
import numpy as np
import argparse as ap
import so3g.proj.quat as quat
import lmfit
from lmfit import minimize, Parameters
import yaml
import logging

from sotodlib.site_pipeline import util
from sotodlib import core
from sotodlib.coords import pointing_model as pm
from sotodlib.coords import fp_containers as fpc

DEG = np.pi / 180.0
ARCMIN = DEG / 60


def _load_nom_centers(config):
    # Load Nominal UFM Center Locations from centered focal_plane
    ffp_path = config.get("ffp_path")
    ufms = config.get("ufms")
    nom_ufm_centers = np.zeros([1, 7, 3]) * np.nan
    rx = fpc.Receiver.load_file(ffp_path)
    OT = rx["0"].optics_tubes[0]
    for ufm in range(len(OT.focal_planes)):
        index = ufms.index(OT.focal_planes[ufm].stream_id)
        nom_ufm_centers[0, index, :3] = OT.focal_planes[ufm].center
    nom_ufm_centers = nom_ufm_centers[0]

    return nom_ufm_centers


# obsdb_entries = [ctx.obsdb.get(obsid) for obsid in filelist]
def _load_per_obs_data(config):
    # Load per-observation UFM center data points and weights
    # The per obs .h5 file a dict with obs_id for keys
    per_obs_fps = config.get("per_obs_fps")
    ufms = config.get("ufms")
    rxs = fpc.Receiver.load_file(per_obs_fps)
    if config.get("platform") == "satp1":
        filelist = list(rxs.keys())
        # the following are known to be bad fits:
        filelist = [
            item
            for item in filelist
            if "_1713" not in item and "1716423951" not in item
        ]
    else:
        filelist = list(rxs.keys())
    obsidnum = np.array(
        [filelist[id].split("_")[1] for id, _ in enumerate(filelist)], dtype=int
    )
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
    az_c = [ctx.obsdb.get(obsid)["az_center"] for obsid in filelist]
    el_c = [ctx.obsdb.get(obsid)["el_center"] for obsid in filelist]
    roll_c = [ctx.obsdb.get(obsid)["roll_center"] for obsid in filelist]
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


def objective_model_func_lmfit(params, solver_aman, return_fit=False, weights=True):
    if type(params) == lmfit.parameter.Parameters:
        params = params.valuesdict()
    xi_nom, eta_nom, gam_nom = solver_aman.nom_ufm_centers
    az, el, roll = pm._get_sat_enc_radians(solver_aman.ancil)
    az1, el1, roll1 = pm.model_sat_v1(params, az, el, roll)
    ## Quat math is based on this equation: q_nomodel * q_det_data == q_model * q_det_true
    q_nomodel = quat.rotation_lonlat(-az, el, 0)
    q_model = quat.rotation_lonlat(-az1, el1, roll1)
    q_det_true = quat.rotation_xieta(xi_nom, eta_nom, 0)
    xi_mod, eta_mod, gamma_mod = quat.decompose_xieta(~q_nomodel * q_model * q_det_true)
    xi_ffp, eta_ffp, gamma_ffp = solver_aman.ffp_ufm_center_fits
    if return_fit:
        return xi_mod, eta_mod, gamma_mod
    else:
        dist = []
        for i in range(len(xi_mod)):
            dist.append(math.dist([xi_ffp[i], eta_ffp[i]], [xi_mod[i], eta_mod[i]]))
        if weights:
            return chi_sq(solver_aman.weights, np.array(dist))
        else:
            return chi_sq(np.ones(len(dist)), np.array(dist))


def get_RMS(model_xieta, data_xieta, weights):
    diff = (model_xieta[0] / ARCMIN - data_xieta[0] / DEG * 60) ** 2 + (
        model_xieta[1] / ARCMIN - data_xieta[1] / DEG * 60
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


def main():
    # Read input parameters
    parser = ap.ArgumentParser()
    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Read relevant config file info
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    platform = config.get("platform")  # e.g. satp1
    pm_version = config.get("pm_version")  # e.g. sat_v1
    solution_version_tag = config.get("solution_version_tag")  # e.g. YYMMDDr#
    save_dir = os.path.join(
        config.get("outdir"), f"{platform}_pointing_model_{solution_version_tag}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # savemeta_dir = os.path.join(config.get("savemeta_dir"), solution_version_tag)
    # if not os.path.exists(savemeta_dir):
    #    os.makedirs(savemeta_dir, exists_ok=False)

    # Initialize Logger
    logger = util.init_logger(__name__, "Solve pointing_model")
    logpath = os.path.join(save_dir, "pointing_model.log")
    logfile = logging.FileHandler(logpath)
    logger.addHandler(logfile)

    # Load in focal_plane and boresigt data
    nom_ufm_centers = _load_nom_centers(config)
    logger.info("Loaded nominal UFM centers from %s: ", config.get("ffp_path"))
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

    weights_mask = np.where(solver_aman["weights"] == 0)[0]
    solver_aman["ffp_ufm_center_fits"][:, weights_mask] = np.nan

    logger.info("Built axis manager")

    # Initialize Parameters to Fit with Model
    fit_params = _init_fit_params(config)

    # Solve for Model Paramters
    model_solved_params = lmfit.minimize(
        objective_model_func_lmfit,
        fit_params,
        method="nelder",
        nan_policy="omit",
        args=(solver_aman, False, True),
    )
    model_fits = objective_model_func_lmfit(
        model_solved_params.params, solver_aman, return_fit=True
    )

    test_params = _round_params(model_solved_params.params.valuesdict(), 8)
    test_params["version"] = pm_version

    logger.info("Found best-fit pointing model parameters")
    logger.info(test_params)
    logger.info(
        "RMS on fit: %f",
        get_RMS(model_fits, solver_aman.ffp_ufm_center_fits, solver_aman.weights),
    )

    # Save fit results to the axis manager
    modelfit_aman = core.AxisManager()
    modelfit_aman.wrap("xi", model_fits[0])
    modelfit_aman.wrap("eta", model_fits[1])
    # modelfit_aman.wrap("gamma", model_fits[2])
    solver_aman.wrap("model_fits", modelfit_aman)

    param_aman = core.AxisManager()
    for k in list(test_params.keys()):
        param_aman.wrap(k, test_params[k])
    solver_aman.wrap("pointing_model", param_aman)
    solver_aman.wrap(
        "fit_rms",
        get_RMS(model_fits, solver_aman.ffp_ufm_center_fits, solver_aman.weights),
    )

    # Save .h5 and ManifestDb
    h5_rel = "pointing_model_data.h5"
    h5_filename = os.path.join(save_dir, h5_rel)
    solver_aman.save(h5_filename, overwrite=True)
    dbfile = "db.sqlite"
    db = _create_db(dbfile, save_dir)
    db.add_entry({"dataset": "pointing_model"}, filename=h5_rel, replace=True)
    db.to_file(os.path.join(save_dir, dbfile))

############

if __name__ == "__main__":
    main()
