import os
import pickle
import math
import h5py
import argparse
import yaml
import logging
import numpy as np
import so3g.proj.quat as quat
import pdb
# import lmfit
import lmfit
from lmfit import minimize, Parameters, fit_report
import time
import shutil

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sotodlib.coords.helpers import _valid_arg
from sotodlib.site_pipeline import util as sp_util
from sotodlib import core
from sotodlib.coords import pointing_model as pm
from sotodlib.coords import fp_containers as fpc

DEG = np.pi / 180.0
ARCMIN = DEG / 60

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.5


def load_nom_ufm_centers(config):
    # Load Nominal UFM Center Locations from centered focal_plane
    ffp_path = config.get("ffp_path")
    ufms = config.get("ufms")
    nom_ufm_centers = np.zeros([7, 3]) * np.nan
    rx = fpc.Receiver.load_file(ffp_path)
    OT = rx["0"].optics_tubes[0]
    for ufm in range(len(OT.focal_planes)):
        try:
            index = ufms.index(OT.focal_planes[ufm].stream_id)
        except:
            temp_ufms = config.get("temp_ufms")
            index = temp_ufms.index(OT.focal_planes[ufm].stream_id) 
        nom_ufm_centers[index, :3] = OT.focal_planes[ufm].center

    return nom_ufm_centers

def load_per_obs_data(config):
    # Load per-observation UFM center data points and weights
    # The per obs .h5 file a dict with obs_id for keys
    per_obs_fps = config.get("per_obs_fps")
    ufms = config.get("ufms")
    skip_tags = config.get("skip_tags", [])
    t0 = config.get("begin_timerange", 0)
    tf = config.get("end_timerange", 3000000000)
    rxs = fpc.Receiver.load_file(per_obs_fps)

    filelist = [obs for obs in rxs.keys() if all(skip not in obs for skip in skip_tags)]
    filelist = [obs for obs in filelist if int(obs.split("_")[1]) > t0 and int(obs.split("_")[1]) < tf]
    if config.get("use_these_files") is not None:
        filelist = [filelist[i] for i in config.get("use_these_files")]
        
    obs_ufm_centers = np.zeros([len(filelist), 7, 3]) * np.nan
    weights_ufm = np.zeros([len(filelist), 7])
    obs_index = []

    for i, ffp in enumerate(filelist):
        this_OT = rxs[ffp].optics_tubes[0]
        for u in range(len(this_OT.focal_planes)):
            index = ufms.index(this_OT.focal_planes[u].stream_id)
            obs_ufm_centers[i, index, :3] = this_OT.focal_planes[u].center_transformed
            weights_ufm[i, index] = np.nansum(this_OT.focal_planes[u].weights)
        obs_index.append(np.repeat(i, 7))
           
    weights_ufm = weights_ufm / 1720.0
    weights_ufm[weights_ufm < config.get("weight_cutoff")] = 0.0
    initial_weights_mask = np.where(weights_ufm == 0)
    obs_ufm_centers[initial_weights_mask] = np.nan
    obs_index = np.concatenate(obs_index)
    #obs_index[initial_weights_mask] = np.nan

    return filelist, obs_ufm_centers, weights_ufm, obs_index

def load_nom_focal_plane_full(config, ufm):
    which_template = config.get("use_as_template", "ffp")
    if which_template == "nominal":
        ffp_path = config.get("nominal")
        with h5py.File(ffp_path, "r") as template_fp:
            det_ids = template_fp[ufm]["dets:det_id"][:]
            xi = template_fp[ufm]["xi"][:]
            eta = template_fp[ufm]["eta"][:]
            gamma = template_fp[ufm]["gamma"][:]
            nom_det_array = np.stack((xi, eta, gamma), axis=1)

    elif which_template == "ffp":
        ffp_path = config.get("ffp_path")
        with h5py.File(ffp_path, "r") as template_fp:
            OT = template_fp["0/st1"]
            fpf = np.array(OT[ufm]["focal_plane_full"][:])
            # Extracting specific columns using structured arrays
            det_ids = fpf[
                "dets:det_id"
            ]  # Assuming 'f0' corresponds to the detector IDs
            xi = fpf["xi_t"]  # Assuming 'f1' corresponds to xi
            eta = fpf["eta_t"]  # Assuming 'f2' corresponds to eta
            gamma = fpf["gamma_t"]  # Assuming 'f3' corresponds to gamma
            nom_det_array = np.stack((xi, eta, gamma), axis=1)

    return det_ids, nom_det_array
    
def create_size_mask(obs_index):
    #create comparably sized datasets of all obs.
    unique, counts = np.unique(obs_index, return_counts=True)
    min_count = min(counts)
    limiting_mask = np.zeros_like(obs_index, dtype=bool)
    for dataset in unique:
        indices = np.where(obs_index == dataset)[0]
        selected_indices = np.random.choice(indices, min_count, replace=False)
        limiting_mask[selected_indices] = True
    return limiting_mask
    
def create_culling_mask(obs_index, cull_dets):
    # Remove a random fraction 1/cull_dets of each dataset 
    unique, counts = np.unique(obs_index, return_counts=True)
    #min_count = min(counts)
    culling_mask = np.zeros_like(obs_index, dtype=bool)
    for dataset, count in zip(unique, counts):
        indices = np.where(obs_index == dataset)[0]
        selected_indices = np.random.choice(indices, (cull_dets - 1)* count // cull_dets, replace=False)
        culling_mask[selected_indices] = True
    return culling_mask 
    
    
def load_per_detector_data(config, no_downsample_set=False, return_all_dets=False):
    per_obs_fps = config.get("per_obs_fps")
    skip_tags = config.get("skip_tags", [])
    t0 = config.get("begin_timerange", 0)
    tf = config.get("end_timerange", int(time.time()))
    rxs = fpc.Receiver.load_file(per_obs_fps)

    if return_all_dets:
        band = None
        cull_dets = config.get("cull_dets", None)
        cull_twice = config.get("cull_twice", False)
        even_obs_size = False
        which_ufm = None
        filelist = [obs for obs in rxs.keys() if all(skip not in obs for skip in skip_tags)]
        filelist = [obs for obs in filelist if int(obs.split("_")[1]) > t0 and int(obs.split("_")[1]) < tf]

    else:
        band = config.get("band")
        if band is not None:
            band = band.encode("utf-8")  
        cull_dets = config.get("cull_dets", None)
        cull_twice = config.get("cull_twice", False)
        even_obs_size = config.get("even_obs_size", False)
        which_ufm = config.get("which_ufm", None)
        filelist = [obs for obs in rxs.keys() if all(skip not in obs for skip in skip_tags)]
        filelist = [obs for obs in filelist if int(obs.split("_")[1]) > t0 and int(obs.split("_")[1]) < tf]
        if config.get("use_these_files") is not None:
            filelist = [filelist[i] for i in config.get("use_these_files")]

    weights_dets, obs_dets_fits, stream_id_list, obs_index = [], [], [], []
    #which_ufm = config.get("which_ufm", None)
    which_data = config.get("use_as_data")
    which_weights = config.get("use_as_weights", None)
    
    for i, ffp in enumerate(filelist):
        this_OT = rxs[ffp].optics_tubes[0]
        for ufm in this_OT.focal_planes:
            if which_ufm is not None and ufm.stream_id not in which_ufm:
                continue
            stream_id_list.append(ufm.stream_id)
            weights = ufm.weights[:, 1] if which_weights == "r2" else ufm.weights[:, 0]
            data = ufm.avg_fp if which_data == "raw" else ufm.transformed
            weights_dets.append(weights)
            obs_dets_fits.append(data)
            obs_index.append(np.repeat(i, len(ufm.weights)))
            
    nom_data = [load_nom_focal_plane_full(config, s) for s in stream_id_list]
    all_det_ids, all_nom_det_array = map(np.concatenate, zip(*nom_data))
    weights_dets = np.concatenate(weights_dets)
    obs_dets_fits = np.concatenate(obs_dets_fits, axis=0)
    obs_index = np.concatenate(obs_index)

    weights_dets[weights_dets < config.get("weight_cutoff")] = 0.0
    obs_dets_fits[np.where(weights_dets == 0)] = np.nan
    mask = ~np.isnan(weights_dets)  
    
    if no_downsample_set:
        #plotting use-case to compare subset fits with the entire dataset.
        return (
            filelist,
            obs_dets_fits[mask],
            weights_dets[mask],
            all_nom_det_array[mask],
            all_det_ids[mask],
            obs_index[mask],
        )

    else:
        # Reduce detector counts for computation
        if band is not None:
            mask &= np.array([band in det for det in all_det_ids])
        #apply weights and band mask
        obs_dets_fits = obs_dets_fits[mask]
        weights_dets = weights_dets[mask]
        all_nom_det_array = all_nom_det_array[mask]
        all_det_ids = all_det_ids[mask]
        obs_index = obs_index[mask]
        if even_obs_size:
            mask = create_size_mask(obs_index)
            obs_dets_fits = obs_dets_fits[mask]
            weights_dets = weights_dets[mask]
            all_nom_det_array = all_nom_det_array[mask]
            all_det_ids = all_det_ids[mask]
            obs_index = obs_index[mask]
        if cull_dets is not None:
            for _ in range(2 if cull_twice else 1):
                mask = create_culling_mask(obs_index, cull_dets)
                obs_dets_fits = obs_dets_fits[mask]
                weights_dets = weights_dets[mask]
                all_nom_det_array = all_nom_det_array[mask]
                all_det_ids = all_det_ids[mask]
                obs_index = obs_index[mask]
        return (
            filelist, 
            obs_dets_fits,
            weights_dets,
            all_nom_det_array,
            all_det_ids,
            obs_index
        )

def load_obs_boresight(config, filelist):
    # Load boresight elevation information from each observation
    # Put into an axis manager
    pm_version = config.get("pm_version")
    ctx = core.Context(config["context"]["path"])
    obs_info = [ctx.obsdb.get(obsid) for obsid in filelist]
    az_c = np.array([obs["az_center"] for obs in obs_info])
    el_c = np.array([obs["el_center"] for obs in obs_info])
    roll_c = np.array([obs["roll_center"] for obs in obs_info])
    #az_c = np.round(np.array(az_c), 4)
    #el_c = np.round(np.array(el_c), 4)
    #roll_c = np.round(np.array(roll_c), 4)
    #roll_c[np.where(roll_c == 0)[0]] = 0  # rounding gives negative 0 sometimes.

    ancil = core.AxisManager(core.IndexAxis("samps"))
    ancil.wrap("az_enc", np.repeat(az_c, 7), [(0, "samps")])
    ancil.wrap("el_enc", np.repeat(el_c, 7), [(0, "samps")])
    if "lat" in pm_version:
        ancil.wrap("corotator_enc", np.repeat((el_c - 60. - roll_c), 7), [(0, "samps")])
    if "sat" in pm_version:   
        ancil.wrap("boresight_enc", np.repeat(-1 * roll_c, 7), [(0, "samps")])
    return ancil, roll_c


def load_obs_boresight_per_detector(config, filelist, obs_ind):
    # Load boresight elevation information from each observation
    # Put into an axis manager
    platform = config.get("platform")
    ctx = core.Context(config["context"]["path"])
    obs_info = [ctx.obsdb.get(obsid) for obsid in filelist]
    az_c = np.array([obs["az_center"] for obs in obs_info])
    el_c = np.array([obs["el_center"] for obs in obs_info])
    roll_c = np.array([obs["roll_center"] for obs in obs_info])

    ancil = core.AxisManager(core.IndexAxis("samps"))
    if platform == 'lat':
        roll_c = np.array([roll_c[i] for i in obs_ind])       
        ancil.wrap("az_enc", np.array([az_c[i] for i in obs_ind]), [(0, "samps")])
        ancil.wrap("el_enc", np.array([el_c[i] for i in obs_ind]), [(0, "samps")])
        ancil.wrap("corotator_enc", ancil.el_enc - 60. - roll_c, [(0, "samps")])  
    else:   
        roll_c = np.array([roll_c[i] for i in obs_ind])
        ancil.wrap("az_enc", np.array([az_c[i] for i in obs_ind]), [(0, "samps")])
        ancil.wrap("el_enc", np.array([el_c[i] for i in obs_ind]), [(0, "samps")])
        ancil.wrap("boresight_enc", -1 * roll_c, [(0, "samps")])

    return ancil, roll_c


def _init_fit_params(config):
    pm_version = config.get("pm_version")
    init_params = config.get("initial_params", pm.param_defaults[pm_version])
    fixed_params = config.get("fixed_params",None)
    # Initialize lmfit Parameter object
    fit_params = Parameters()
    for p in init_params.keys():
        fit_params.add(p, value=init_params[p], vary=True)
    # Turn off various parameters depending on platform
    if fixed_params is not None:
        for fix in fixed_params:
            fit_params[fix].set(vary=False)

    return fit_params

def objective_model_func_lmfit(
    params, pm_version, solver_aman, xieta_model, weights=True
):
    if xieta_model == "measured":
        xi_mod, eta_mod = model_measured_xieta(params, pm_version, solver_aman)
        xi_ref, eta_ref, _ = solver_aman.measured_xieta_data
    elif xieta_model == "template":
        xi_mod, eta_mod = model_template_xieta(params, pm_version, solver_aman)
        xi_ref, eta_ref, _ = solver_aman.nominal_xieta_locs
    dist = np.sqrt((xi_ref - xi_mod) ** 2 + (eta_ref - eta_mod) ** 2)
    #print(np.nansum(dist))
    weights_array = solver_aman.weights if weights else np.ones(len(dist))
    return chi_sq(weights_array, dist)

def chi_sq(weights, dist):
    #N = np.identity(len(dist)) * weights
    #chi2 = dist.T @ N @ dist
    chi2 = np.nansum(dist ** 2 * weights)
    return chi2

def model_template_xieta(params, pm_version, aman):
    """
    Transform a measured (xi,eta) back into template position
    Data to Template -- modeling data as true template
    Quat math is based on this equation:
    q_nomodel * q_det_meas == q_model * q_det_true
    """
    xi_meas = aman.measured_xieta_data[0]
    eta_meas = aman.measured_xieta_data[1]
    params = params.valuesdict() if isinstance(params, Parameters) else params
    params["version"] = pm_version
    if "sat" in pm_version:
        az, el, roll = pm._get_sat_enc_radians(aman.ancil)
        q_nomodel = quat.rotation_lonlat(-az, el, 0)
    if "lat" in pm_version:
        az, el, roll = pm._get_lat_enc_radians(aman.ancil)
        q_nomodel = quat.rotation_lonlat(-az, el, roll)
    boresight = pm.apply_pointing_model(aman, pointing_model=params, wrap=False)
    az1, el1, roll1 = boresight.az, boresight.el, boresight.roll
    q_model = quat.rotation_lonlat(-az1, el1, roll1)
    q_det_meas = quat.rotation_xieta(xi_meas, eta_meas, 0)
    xi_mod_true, eta_mod_true, _ = quat.decompose_xieta(
        ~q_model * q_nomodel * q_det_meas
    )
    return xi_mod_true, eta_mod_true
    
def model_measured_xieta(params, pm_version, aman):
    """
    Transform template (xi,eta) to match measured (xi,eta).
    Template to Data -- modeling the template as measured data
    Quat math is based on this equation:
    q_nomodel * q_det_meas == q_model * q_det_true
    """
    params = params.valuesdict() if isinstance(params, Parameters) else params
    params["version"] = pm_version
    xi_true, eta_true, gam_true = aman.nominal_xieta_locs
    if "sat" in pm_version:
        az, el, roll = pm._get_sat_enc_radians(aman.ancil)
        q_nomodel = quat.rotation_lonlat(-az, el, 0) 
    if "lat" in pm_version:
        az, el, roll = pm._get_lat_enc_radians(aman.ancil)
        q_nomodel = quat.rotation_lonlat(-az, el, roll)

    boresight = pm.apply_pointing_model(aman, pointing_model=params, wrap=False)
    az1, el1, roll1 = boresight.az, boresight.el, boresight.roll
    q_model = quat.rotation_lonlat(-az1, el1, roll1)
    q_det_true = quat.rotation_xieta(xi_true, eta_true, 0)
    xi_mod_meas, eta_mod_meas, _ = quat.decompose_xieta(
        ~q_nomodel * q_model * q_det_true
    )

    return xi_mod_meas, eta_mod_meas


def calc_RMS_and_residuals(modeled_fits, model_reference, weights, use_inds=None):
    diff = ((modeled_fits[0] - model_reference[0]) / ARCMIN) ** 2 +\
           ((modeled_fits[1] - model_reference[1]) / ARCMIN) ** 2
    if use_inds is not None:
        diff = diff[use_inds]
        weights = weights[use_inds]
    return np.sqrt(np.nansum(diff * weights) / np.nansum(weights)), diff**0.5

    
def apply_model_params(xieta_model, pointing_model, pm_version, aman, use_inds=None):
    #fetch_RMS_and_residuals
    #Apply PM parameters to either template or data points, calc RMS, return residuals.
    if xieta_model == "measured":
        model_reference = aman.measured_xieta_data
        modeled_fits = model_measured_xieta(
            pointing_model, pm_version, aman
        )
    elif xieta_model == "template":
        model_reference = aman.nominal_xieta_locs  
        modeled_fits = model_template_xieta(
            pointing_model, pm_version, aman
        )      
    rms, fit_residuals = calc_RMS_and_residuals(modeled_fits, model_reference, aman.weights, use_inds=use_inds)
    return modeled_fits, fit_residuals, rms, model_reference


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
    scheme.add_range_match("obs:obs_timestamp")
    scheme.add_data_field("dataset")
    return core.metadata.ManifestDb(db_filename, scheme=scheme)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to Configuration File")
    return parser


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
    plotlims = config.get("plotlims", 20)
    append_string = config.get("append", "")
    if append_string is not None:
        append_tag = f"{bool(append_string)*'_'}{append_string}"
    else:
        append_tag = ""
    which_ufm = config.get("which_ufm", "")
    if which_ufm is not None:
        which_ufm = which_ufm if isinstance(which_ufm, list) else [which_ufm]
        suffixes = [ufm.split("_")[-1] for ufm in which_ufm]
        ufm_tag = "_" + "_".join(suffixes)
    else:
        ufm_tag = ""  
    band = config.get("band", None)
    if config.get("band") is not None:
        band_tag = f"_{band}"
    else:
        band_tag = ""
    save_dir = os.path.join(
        config.get("outdir"),
        f"{platform}_pointing_model_{sv_tag}",
        f"{xe_tag}{append_tag}{ufm_tag}{band_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))
    
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

    ##########################################################
    ### Begin split for per-detector or per-UFM center fitting
    ##########################################################

    fit_type = config.get("fit_type", "detector")
    if fit_type == "detector":

        which_ufm = config.get("which_ufm", None)
        if which_ufm is not None:
            logger.info(
                "Performing per-detector fits for subset of UFMs: %s. ", which_ufm
            )
        else:
            logger.info("Performing per-detector fits for all UFM data.")

        which_data = config.get("use_as_data")
        use_weights = config.get("use_weights", True)

        #Make axis manager with full detector set.
        # Keep wafer/band/obs cuts but do not further downsample.
        fitcheck_aman = core.AxisManager(core.IndexAxis("samps"))
        (
            filelist,
            obs_dets_fits,
            weights_dets,
            all_nom_det_array,
            all_det_ids,
            obs_index,
        ) = load_per_detector_data(config, no_downsample_set=True)
        ancil, roll_c = load_obs_boresight_per_detector(config, filelist, obs_index)
        
        fitcheck_aman.wrap("ancil", ancil)
        fitcheck_aman.wrap(
            "nominal_xieta_locs", all_nom_det_array.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        fitcheck_aman.wrap(
            "measured_xieta_data", obs_dets_fits.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        fitcheck_aman.wrap("weights", weights_dets, [(0, "samps")])
        fitcheck_aman.wrap("obs_index", obs_index, [(0, "samps")])
        logger.info("Loaded %s fit check data points", len(weights_dets))

        #Now make axis manager that has down sampled data for computation
        solver_aman = core.AxisManager(core.IndexAxis("samps"))
        (
            filelist,
            obs_dets_fits,
            weights_dets,
            all_nom_det_array,
            all_det_ids,
            obs_index,
        ) = load_per_detector_data(config)
        logger.info("Loaded %s data points", len(weights_dets))
        ancil, roll_c = load_obs_boresight_per_detector(config, filelist, obs_index)

        # Build Axis Managers        
        solver_aman.wrap("ancil", ancil)
        obs_info = core.AxisManager()
        obs_info.wrap("obs_ids", np.array(filelist))
        solver_aman.wrap("obs_info", obs_info)
        solver_aman.wrap("roll_c", roll_c, [(0, "samps")])
        solver_aman.wrap(
            "nominal_xieta_locs",
            all_nom_det_array.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        solver_aman.wrap(
            "measured_xieta_data",
            obs_dets_fits.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        solver_aman.wrap("weights", weights_dets, [(0, "samps")])
        solver_aman.wrap("obs_index", obs_index)
        logger.info("Built axis manager")

    ###########################################################################
    elif fit_type == "ufm_center":
        use_weights = config.get("use_weights", True)
        # Load in focal_plane and boresight data
        nom_ufm_centers = load_nom_ufm_centers(config)
        logger.info("Loaded nominal UFM centers from %s: ", config.get("ffp_path"))
        logger.info(nom_ufm_centers)

        filelist, obs_ufm_centers, weights_ufm, obs_index = load_per_obs_data(config)
        logger.info("Loaded per-obs FFP data from %s: ", config.get("per_obs_fps"))
        logger.info("Including data from these obs:")
        logger.info(filelist)

        ancil, roll_c = load_obs_boresight(config, filelist)
        logger.info("Loaded boresight data from obs ids.")

        # Build Axis Managers
        obs_info = core.AxisManager()
        obs_info.wrap("obs_ids", np.array(filelist))

        solver_aman = core.AxisManager(core.IndexAxis("samps"))
        solver_aman.wrap("ancil", ancil)
        solver_aman.wrap("obs_info", obs_info)
        solver_aman.wrap("roll_c", np.repeat(roll_c, 7), [(0, "samps")])
        solver_aman.wrap(
            "nominal_xieta_locs",
            np.repeat([nom_ufm_centers], len(filelist), axis=0)
            .reshape(len(filelist) * 7, 3)
            .T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        solver_aman.wrap(
            "measured_xieta_data",
            obs_ufm_centers.reshape(len(filelist) * 7, 3).T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        solver_aman.wrap("weights", weights_ufm.reshape(-1), [(0, "samps")])
        solver_aman.wrap("obs_index", obs_index)
        # Make weights/data cuts
        logger.info("Built axis manager")

    ################################
    # END of SPLIT: Now fit the parameters
    ################################

    # Initialize Parameters to Fit with Model
    fit_params = _init_fit_params(config)
    logger.info("Initialized fit parameters")

    # Solve for Model Parameters
    # use chosen xieta_model to solve for parameters
    model_solved_params = minimize(
        objective_model_func_lmfit,
        fit_params,
        method="nelder",
        nan_policy="omit",
        args=(pm_version, solver_aman, xieta_model, use_weights),
    )
    logger.info("Ran 1st Minimization")

    test_params = _round_params(model_solved_params.params.valuesdict(), 8)
    test_params["version"] = pm_version
    logger.info("Found best-fit pointing model parameters")
    logger.info(test_params)
    logger.info(model_solved_params.params.pretty_print(precision=5, colwidth=11))
    logger.info("Fit Report:")
    logger.info(fit_report(model_solved_params))

    # save pointing model parameters to axis manager
    param_aman = core.AxisManager()
    for k in list(test_params.keys()):
        param_aman.wrap(k, test_params[k])
    solver_aman.wrap("pointing_model", param_aman)

    # save errors to axis manager
    error_aman = core.AxisManager()
    for k in list(model_solved_params.params.values()):
        error_aman.wrap(k.name, k.stderr)
    solver_aman.wrap("pointing_model_errors", error_aman)

    # parameter_fit_stats = build_param_fit_stat_aman(model_solved_params)
    # solver_aman.wrap("parameter_fit_stats", parameter_fit_stats)

    # Model template and measured points using parameters found above
    modeled_fits, fit_residuals_i1, rms_i1, model_reference = apply_model_params(xieta_model, solver_aman.pointing_model, pm_version, solver_aman)
    logger.info("RMS on fit: %f arcmin", rms_i1)

    # Save fit results to the axis manager
    modelfit_aman = core.AxisManager()
    modelfit_aman.wrap("xi", modeled_fits[0], overwrite=True)
    modelfit_aman.wrap("eta", modeled_fits[1], overwrite=True)
    solver_aman.wrap("modeled_fits", modelfit_aman, overwrite=True)
    solver_aman.wrap("fit_residuals", fit_residuals_i1, overwrite=True)
    solver_aman.wrap("fit_rms", rms_i1, overwrite=True)
    
    if fit_type == "detector":
        _, fit_residuals_full, rms_full, _ = apply_model_params(xieta_model, solver_aman.pointing_model, pm_version, fitcheck_aman)
        logger.info("RMS on FULL detector set: %f arcmin", rms_full)
        solver_aman.wrap("fit_residuals_full", fit_residuals_full, overwrite=True)
        solver_aman.wrap("fit_rms_full", rms_full, overwrite=True)
        solver_aman.wrap("obs_index_full", fitcheck_aman.obs_index)
        
    cutoff = np.nanstd(fit_residuals_i1)*2 + np.nanmedian(fit_residuals_i1)
    logger.info(f"2 stdev away from residual Median: {cutoff:.2f} arcmin")

    if config.get("make_plots"):
        tag = "_i1"
        plotter = ModelFitsPlotter(solver_aman=solver_aman,
                                   config=config,
                                   save_dir=save_dir,
                                   iteration_tag=tag,
                                   save_figure=True,
                                   plotlims=plotlims)
        if fit_type == "ufm_center":
            plotter.plot_ws0_modeled_fits()
            plotter.plot_template_space_fits_per_wafer()
            plotter.plot_residuals_vs_ancil()
            plotter.plot_xieta_cross_residuals()
            plotter.plot_xieta_residuals()
        else:
            plotter.plot_modeled_fits()
            plotter.plot_template_space_fits_per_detector()
            plotter.plot_residuals_vs_ancil()
            plotter.plot_residuals_histograms()
            plotter.plot_dets_in_these_obs()

    if iterate_cutoff is not None:
        logger.info("Iterating parameter solution")
        logger.info(f"Using {iterate_cutoff} as cutoff")
        if iterate_cutoff == "auto":
            iterate_cutoff = np.nanstd(fit_residuals_i1)*2 + np.nanmedian(fit_residuals_i1)
            logger.info(f"Using {iterate_cutoff} as cutoff")
        bad_fit_inds = np.where(fit_residuals_i1 > iterate_cutoff)[0]
        logger.info("Bad fit indices:")
        logger.info(bad_fit_inds)
        logger.info(
            "%f data points are higher than %s arcmin",
            len(bad_fit_inds),
            iterate_cutoff,
        )

        if len(bad_fit_inds) != 0:
            if fit_type == "ufm_center":
                bad_filename = bad_fit_inds // 7
                bad_wafer = bad_fit_inds % 7
                logger.info("Outliers:")
                for i, full_i in enumerate(bad_fit_inds):
                    logger.info(
                        f"{filelist[bad_filename[i]]}; ws{bad_wafer[i]}; Resid. {np.round(fit_residuals_i1[full_i], 4)}"
                    )
                    logger.info(
                        f"--- Roll {solver_aman.roll_c[full_i]}; El {solver_aman.ancil.el_enc[full_i]}; weight {np.round(solver_aman.weights[full_i],4)}"
                    )

            # Print RMS of initial fits without outlying data points before
            # zero-ing the weights.
            good_fit_inds = np.where(fit_residuals_i1 < iterate_cutoff)[0]
            _, _, masked_rms, _ = apply_model_params(xieta_model, 
                                                    solver_aman.pointing_model, 
                                                    pm_version, 
                                                    solver_aman,
                                                    use_inds=good_fit_inds)
            
            logger.info("RMS on initial fit without outliers: %f arcmin", masked_rms)
            solver_aman.wrap('bad_fit_inds', bad_fit_inds)
            solver_aman.weights[bad_fit_inds] = 0.0

            model_solved_params = minimize(
                objective_model_func_lmfit,
                fit_params,
                method="nelder",
                nan_policy="omit",
                args=(pm_version, solver_aman, xieta_model, use_weights),
            )

            test_params = _round_params(model_solved_params.params.valuesdict(), 8)
            test_params["version"] = pm_version
            logger.info("Found best-fit pointing model parameters, second iteration")
            logger.info(test_params)
            logger.info(
                model_solved_params.params.pretty_print(precision=5, colwidth=11)
            )
            logger.info("Fit Report:")
            logger.info(lmfit.fit_report(model_solved_params))

            # save pointing model parameters to axis manager
            solver_aman.move("pointing_model", "pointing_model_i1")
            param_aman = core.AxisManager()
            for k in list(test_params.keys()):
                param_aman.wrap(k, test_params[k])
            solver_aman.wrap("pointing_model", param_aman, overwrite=True)

            # save errors to axis manager
            solver_aman.move("pointing_model_errors", "pointing_model_errors_i1")
            error_aman = core.AxisManager()
            for k in list(model_solved_params.params.values()):
                error_aman.wrap(k.name, k.stderr)
            solver_aman.wrap("pointing_model_errors", error_aman, overwrite=True)
            
            # parameter_fit_stats = build_param_fit_stat_aman(model_solved_params)
            # solver_aman.wrap("parameter_fit_stats", parameter_fit_stats, overwrite=True)

            # Recalculate best-fit modeled points
            modeled_fits, fit_residuals_i2, rms_i2, model_reference = apply_model_params(xieta_model,
                                                              solver_aman.pointing_model,
                                                              pm_version,
                                                              solver_aman)

            logger.info("RMS on secondary fit: %f arcmin", rms_i2)

            # Save fit results to the axis manager
            modelfit_aman = core.AxisManager()
            modelfit_aman.wrap("xi", modeled_fits[0], overwrite=True)
            modelfit_aman.wrap("eta", modeled_fits[1], overwrite=True)
            solver_aman.wrap("modeled_fits", modelfit_aman, overwrite=True)
            solver_aman.move("fit_residuals", "fit_residuals_i1") 
            solver_aman.wrap("fit_residuals", fit_residuals_i2, overwrite=True)
            solver_aman.move("fit_rms", "fit_rms_i1")
            solver_aman.wrap("fit_rms", rms_i2, overwrite=True)

            if fit_type == "detector":
                _, fit_residuals_full, rms_full, _ = pm.apply_model_params(xieta_model, solver_aman.pointing_model, pm_version, fitcheck_aman)
                logger.info("RMS on FULL detector set: %f arcmin", rms_full)
                solver_aman.move("fit_residuals_full", "fit_residuals_full_i1")
                solver_aman.move("fit_rms_full", "fit_rms_full_i1")
                solver_aman.wrap("fit_residuals_full", fit_residuals_full, overwrite=True)
                solver_aman.wrap("fit_rms_full", rms_full, overwrite=True)         
                
            if config.get("make_plots"):
                tag = "_i2"
                plotter = ModelFitsPlotter(solver_aman=solver_aman,
                                           config=config,
                                           save_dir=save_dir,
                                           iteration_tag=tag,
                                           save_figure=True,
                                           plotlims=plotlims)
                plotter.plot_total_residuals()
                plotter.plot_residuals_vs_ancil()
                if fit_type == "ufm_center":
                    plotter.plot_ws0_modeled_fits()
                    plotter.plot_template_space_fits_per_wafer()
                    plotter.plot_xieta_cross_residuals()
                    plotter.plot_xieta_residuals()
                else:
                    plotter.plot_modeled_fits()
                    plotter.plot_template_space_fits_per_detector()
                    plotter.plot_residuals_histograms()
                    plotter.plot_dets_in_these_obs()
    else:
        if config.get("make_plots"):
            plotter = ModelFitsPlotter(solver_aman=solver_aman,
                                       config=config,
                                       save_dir=save_dir,
                                       iteration_tag="",
                                       save_figure=True,
                                       plotlims=plotlims)
            plotter.plot_total_residuals()

    if config.get("save_output"):
        # Save .h5 and ManifestDb
        h5_rel = "pointing_model_data.h5"
        h5_filename = os.path.join(save_dir, h5_rel)
        solver_aman.save(h5_filename, overwrite=True)
        dbfile = "db.sqlite"
        t0 = config.get("begin_timerange", 0)
        t1 = config.get("end_timerange", int(time.time()))
        Epoch_Name = config.get("epoch_name")
        db = _create_db(dbfile, save_dir)
        db.add_entry(
            {"obs:obs_timestamp": (t0, t1), "dataset": f"{Epoch_Name}_parameters"},
            filename=h5_rel,
            replace=True,
        )
        db.to_file(os.path.join(save_dir, dbfile))

    #Optional extra plotting
    if config.get("make_full_analysis_plots", True):
        #Fill up axis manager with ALL the data (only cuts from culling and time stamps remain)
        
        (
            filelist,
            obs_dets_fits,
            weights_dets,
            all_nom_det_array,
            all_det_ids,
            obs_index,
        ) = load_per_detector_data(config, return_all_dets=True)
        ancil, roll_c = load_obs_boresight_per_detector(config, filelist, obs_index)
        ufm_list = [ufm.split("_")[1] for ufm in config.get('ufms')]
        
        obs_info = core.AxisManager()
        obs_info.wrap("obs_ids", np.array(filelist))
        
        full_aman = core.AxisManager(core.IndexAxis("samps"))
        full_aman.wrap("obs_info", obs_info)
        full_aman.wrap("ancil", ancil)
        full_aman.wrap(
            "nominal_xieta_locs", all_nom_det_array.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        full_aman.wrap(
            "measured_xieta_data", obs_dets_fits.T,
            [(0, core.LabelAxis("xietagamma", ["xi", "eta", "gamma"]))],
            [(1, "samps")],
        )
        full_aman.wrap("weights", weights_dets, [(0, "samps")])
        full_aman.wrap("obs_index", obs_index)
        full_aman.wrap("roll_c", roll_c, [(0, "samps")])
        full_aman.wrap("det_ids", all_det_ids,  [(0, "samps")])
        full_aman.wrap("radial", 
                       np.sqrt(full_aman.nominal_xieta_locs[0]**2 + full_aman.nominal_xieta_locs[1]**2)/DEG,
                       [(0, "samps")])
        full_aman.wrap("det_ufm", 
                       np.array([detid.decode('utf-8').split('_')[0].lower() for detid in full_aman.det_ids])
                       , [(0, "samps")])
        full_aman.wrap("det_wafer", np.array([ufm_list.index(d) for d in full_aman.det_ufm]), [(0, "samps")])
        
        try:
            full_modeled, full_residuals, rms, _ = apply_model_params("template",
                                                                    solver_aman.pointing_model_i1,
                                                                    config.get("pm_version"),
                                                                    full_aman)
        except:
            full_modeled, fit_residuals, rms, _ = apply_model_params("template",
                                                                    solver_aman.pointing_model,
                                                                    config.get("pm_version"),
                                                                    full_aman)
        
        full_aman.wrap("full_modeled", np.array(full_modeled),
                       [(0, core.LabelAxis("xieta", ["xi", "eta"]))],
                       [(1, "samps")])
        full_aman.wrap("fit_residuals", fit_residuals, [(0, "samps")])
        del(full_modeled)
        del(fit_residuals)
        
        (obs_az, obs_el, obs_roll, 
         obs_resid, obs_dxi, obs_deta,
         obs_std_xi, obs_std_eta
        ) = [], [], [], [], [], [], [], []
        (all_ufm_az, all_ufm_el, all_ufm_roll,
         all_ufm_resid, all_ufm_dxi, all_ufm_deta,
         all_ufm_std_xi, all_ufm_std_eta, all_ufm_wafer_num
        ) = [], [], [], [], [], [], [], [], []
        for ob in np.unique(full_aman.obs_index):
            inds = np.where(full_aman.obs_index == ob)[0] 
            obs_az.append(np.nanmedian(full_aman.ancil.az_enc[inds]))
            obs_el.append(np.nanmedian(full_aman.ancil.el_enc[inds]))
            obs_roll.append(np.nanmedian(full_aman.roll_c[inds]))
            obs_resid.append(np.nanmean(full_aman.fit_residuals[inds]))
            obs_dxi.append(np.nanmean((full_aman.full_modeled[0] -
                                       full_aman.nominal_xieta_locs[0])[inds]/DEG*60))
            obs_deta.append(np.nanmean((full_aman.full_modeled[1] -
                                        full_aman.nominal_xieta_locs[1])[inds]/DEG*60))
            obs_std_xi.append(np.nanstd((full_aman.full_modeled[0] -
                                         full_aman.nominal_xieta_locs[0])[inds]/DEG*60))
            obs_std_eta.append(np.nanstd((full_aman.full_modeled[1] -
                                          full_aman.nominal_xieta_locs[1])[inds]/DEG*60))     
            (ufm_az, ufm_el, ufm_roll, ufm_resid, 
             ufm_dxi, ufm_deta, ufm_std_xi,
             ufm_std_eta, ufm_wafer_num
            )= [], [], [], [], [], [], [], [], []
            for ufm in ufm_list:
                ufm_inds = np.where(full_aman.det_ufm[inds] == ufm)[0]    
                ufm_az.append(np.nanmedian(full_aman.ancil.az_enc[inds][ufm_inds]))
                ufm_el.append(np.nanmedian(full_aman.ancil.el_enc[inds][ufm_inds]))
                ufm_roll.append(np.nanmedian(full_aman.roll_c[inds][ufm_inds]))
                ufm_resid.append(np.nanmean(full_aman.fit_residuals[inds][ufm_inds]))
                ufm_dxi.append(np.nanmean((full_aman.full_modeled[0] - 
                                           full_aman.nominal_xieta_locs[0])[inds][ufm_inds]/DEG*60))
                ufm_deta.append(np.nanmean((full_aman.full_modeled[1] -
                                            full_aman.nominal_xieta_locs[1])[inds][ufm_inds]/DEG*60))
                ufm_std_xi.append(np.nanstd((full_aman.full_modeled[0] -
                                             full_aman.nominal_xieta_locs[0])[inds][ufm_inds]/DEG*60))
                ufm_std_eta.append(np.nanstd((full_aman.full_modeled[1] -
                                              full_aman.nominal_xieta_locs[1])[inds][ufm_inds]/DEG*60))
                ufm_wafer_num.append(np.nanmedian(full_aman.det_wafer[inds][ufm_inds]))           
            all_ufm_az.append(ufm_az)
            all_ufm_el.append(ufm_el)
            all_ufm_roll.append(ufm_roll)
            all_ufm_resid.append(ufm_resid)  
            all_ufm_deta.append(ufm_deta)
            all_ufm_dxi.append(ufm_dxi)
            all_ufm_std_xi.append(ufm_std_xi)
            all_ufm_std_eta.append(ufm_std_eta)
            all_ufm_wafer_num.append(ufm_wafer_num)
            
        per_ufm_stats = core.AxisManager()
        per_obs_stats = core.AxisManager()
        
        per_obs_stats.wrap("el", np.array(obs_el))
        per_obs_stats.wrap("roll", np.array(obs_roll))
        per_obs_stats.wrap("az", np.array(obs_az))
        per_obs_stats.wrap("resid", np.array(obs_resid))
        per_obs_stats.wrap("dxi", np.array(obs_dxi))
        per_obs_stats.wrap("deta", np.array(obs_deta))
        per_obs_stats.wrap("std_xi", np.array(obs_std_xi))
        per_obs_stats.wrap("std_eta", np.array(obs_std_eta))

        per_ufm_stats.wrap("az", np.array(all_ufm_az))
        per_ufm_stats.wrap("el", np.array(all_ufm_el))
        per_ufm_stats.wrap("roll", np.array(all_ufm_roll))
        per_ufm_stats.wrap("resid", np.array(all_ufm_resid))
        per_ufm_stats.wrap("dxi", np.array(all_ufm_dxi))
        per_ufm_stats.wrap("deta", np.array(all_ufm_deta))
        per_ufm_stats.wrap("std_xi", np.array(all_ufm_std_xi))
        per_ufm_stats.wrap("std_eta", np.array(all_ufm_std_eta))
        per_ufm_stats.wrap("wafer_num", np.array(all_ufm_wafer_num))
        
        if platform == "lat":
            obs_cr = []
            all_ufm_cr = []
            for ob in np.unique(full_aman.obs_index):
                inds = np.where(full_aman.obs_index == ob)[0]
                obs_cr.append(np.nanmedian(full_aman.ancil.corotator_enc[inds]))
                ufm_cr = []
                for ufm in ufm_list:
                    ufm_cr.append(np.nanmedian(full_aman.ancil.corotator_enc[inds][ufm_inds]))
                all_ufm_cr.append(ufm_cr)
            per_obs_stats.wrap("cr", np.array(obs_cr))
            per_ufm_stats.wrap("cr", np.array(all_ufm_cr))
        
        full_aman.wrap("dxi", (full_aman.full_modeled[0] - 
                               full_aman.nominal_xieta_locs[0])/DEG*60, [(0, "samps")])
        full_aman.wrap("deta", (full_aman.full_modeled[1] - 
                                full_aman.nominal_xieta_locs[1])/DEG*60, [(0, "samps")])
        
        #full_dxi_av = np.nanmean(full_dxi)
        #full_deta_av = np.nanmean(full_deta)
        obsids=np.array([int(D.split('_')[1]) for D in full_aman.obs_info.obs_ids])
        per_obs_stats.wrap("obsids", obsids)
        per_ufm_stats.wrap("obsids", np.repeat(obsids, np.shape(per_ufm_stats["dxi"])[1]))
        full_aman.wrap("obsids", obsids[full_aman.obs_index])
        
        #Calculate RMSs
        per_obs_stats.wrap("rms", np.sqrt(np.nanmean(per_obs_stats["dxi"]**2 + per_obs_stats["deta"]**2)))
        per_ufm_stats.wrap("rms", np.sqrt(np.nanmean(per_ufm_stats["dxi"]**2 + per_ufm_stats["deta"]**2)))
        full_aman.wrap("rms", np.sqrt(np.nanmean(full_aman["dxi"]**2 + full_aman["deta"]**2)))
        full_aman.wrap("per_ufm_stats", per_ufm_stats)
        full_aman.wrap("per_obs_stats", per_obs_stats)
        
        plotter = ModelFitsPlotter(solver_aman=full_aman,
                                       config=config,
                                       save_dir=save_dir,
                                       iteration_tag="",
                                       save_figure=True,
                                       plotlims=plotlims)
        plotter.plot_full_residuals_across_focalplane()
        plotter.plot_full_histogram()
        plotter.plot_full_unmodeled_residuals()
        
    logger.info("Done")


####################
# Plotting Functions
####################

class ModelFitsPlotter:
    def __init__(self, solver_aman, config, save_dir, iteration_tag="", save_figure=True, plotlims=None):

        self.aman = solver_aman
        self.config = config
        self.tag = iteration_tag
        self.save_figure = save_figure
        if plotlims is not None:
            self.plotlims = plotlims
        else:
            self.plotlims =  config.get("plotlims", 20)       
        plot_dir = os.path.join(save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = plot_dir

        self.platform = config.get("platform")
        self.pm_version = config.get("pm_version")
        self.ufms = config.get("ufms")
        self.which_ufm = config.get("which_ufm",None)
        self.xieta_model = config.get("xieta_model", "measured")
        self.append_string = config.get("append","")
        self.iterate_cutoff = config.get("iterate_cutoff", None)

    def plot_full_unmodeled_residuals(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        append = self.append_string        

        ancil = self.aman.ancil
        modeled = self.aman.full_modeled
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        per_ufm_stats = self.aman.per_ufm_stats
        per_obs_stats = self.aman.per_obs_stats
        roll_c = self.aman.roll_c
        full_dxi_av = np.nanmean(self.aman.dxi)
        full_deta_av = np.nanmean(self.aman.deta)
        if "sat" in platform:
            elmin, elmax = 45, 65
            rollmin, rollmax = -45, 45
            azmin, azmax = 0, 360
            if platform=="satp2":
                plotlim = self.plotlims*0.7
            else:
                plotlim = 8
        else:
            plotlim = self.plotlims
            elmin, elmax = None, None
            rollmin, rollmax = -70, 40
            azmin, azmax = 0, 360
            
        plt.figure(figsize=(7,5))
        plt.scatter((modeled[0] - nominal_xieta_locs[0])/DEG*60,
                    (modeled[1] - nominal_xieta_locs[1])/DEG*60,
                    c=roll_c, s=2.7,
                    alpha=0.15, marker='.', cmap='jet',
                    vmin=rollmin, vmax=rollmax)
        plt.scatter(per_ufm_stats.dxi, per_ufm_stats.deta,
                    c=per_ufm_stats.roll,
                    alpha=0.6, marker='o', cmap='jet',
                    s=25, edgecolor='k', linewidth=0.7,
                    vmin=rollmin, vmax=rollmax, label='UFM Avg')
        plt.scatter(per_obs_stats.dxi, per_obs_stats.deta, 
                    c = per_obs_stats.roll, 
                    alpha=0.99, marker='X', cmap='jet',
                    s=45, edgecolor='k', linewidth=0.7,
                    vmin=rollmin, vmax=rollmax, label='Obs Avg.')
        plt.colorbar(label='Roll')
        plt.xlabel('Xi Error (arcmin)')
        plt.ylabel('Eta Error (arcmin)')
        plt.axvline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.xlim(-plotlim,plotlim); plt.ylim(-plotlim,plotlim)
        plt.scatter(full_dxi_av, full_deta_av, color='r', marker='o', edgecolor='k', label='All Data Avg. Offset')
        plt.legend(fontsize='small')        
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_full_2D_Residuals_Roll{tag}.png", dpi=350)
        plt.close()
            

        plt.figure(figsize=(7,5))
        plt.scatter((modeled[0] - nominal_xieta_locs[0])/DEG*60,
                    (modeled[1] - nominal_xieta_locs[1])/DEG*60,
                    c=ancil.el_enc, s=2.7,
                    alpha=0.15, marker='.', cmap='jet',
                    vmin=elmin, vmax=elmax)
        plt.scatter(per_ufm_stats.dxi, per_ufm_stats.deta,
                    c=per_ufm_stats.el,
                    alpha=0.6, marker='o', cmap='jet',
                    s=25, edgecolor='k', linewidth=0.7,
                    vmin=elmin, vmax=elmax, label='UFM Avg')
        plt.scatter(per_obs_stats.dxi, per_obs_stats.deta, 
                    c = per_obs_stats.el, 
                    alpha=0.99, marker='X', cmap='jet',
                    s=45, edgecolor='k', linewidth=0.7,
                    vmin=elmin, vmax=elmax, label='Obs Avg.')
        plt.colorbar(label='Elevation')
        plt.xlabel('Xi Error (arcmin)')
        plt.ylabel('Eta Error (arcmin)')
        plt.axvline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.xlim(-plotlim,plotlim); plt.ylim(-plotlim,plotlim)
        plt.scatter(full_dxi_av, full_deta_av, color='r', marker='o', edgecolor='k', label='All Data Avg. Offset')
        plt.legend(fontsize='small')        
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_full_2D_Residuals_El{tag}.png", dpi=350)
        plt.close()

        plt.figure(figsize=(7,5))
        plt.scatter((modeled[0] - nominal_xieta_locs[0])/DEG*60,
                    (modeled[1] - nominal_xieta_locs[1])/DEG*60,
                    c=ancil.az_enc%360, s=2.7,
                    alpha=0.15, marker='.', cmap='jet',
                    vmin=azmin, vmax=azmax)
        plt.scatter(per_ufm_stats.dxi, per_ufm_stats.deta,
                    c=per_ufm_stats.az%360,
                    alpha=0.6, marker='o', cmap='jet',
                    s=25, edgecolor='k', linewidth=0.7,
                    vmin=azmin, vmax=azmax, label='UFM Avg')
        plt.scatter(per_obs_stats.dxi, per_obs_stats.deta, 
                    c = per_obs_stats.az%360, 
                    alpha=0.99, marker='X', cmap='jet',
                    s=45, edgecolor='k', linewidth=0.7,
                    vmin=azmin, vmax=azmax, label='Obs Avg.')
        plt.colorbar(label='Azimuth')
        plt.xlabel('Xi Error (arcmin)')
        plt.ylabel('Eta Error (arcmin)')
        plt.axvline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.axhline(0,0,1, color='k', alpha=0.4)
        plt.xlim(-plotlim,plotlim); plt.ylim(-plotlim,plotlim)
        plt.scatter(full_dxi_av, full_deta_av, color='r', marker='o', edgecolor='k', label='All Data Avg. Offset')
        plt.legend(fontsize='small')        
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_full_2D_Residuals_Az{tag}.png", dpi=350)
        plt.close()
   
            
    def plot_full_residuals_across_focalplane(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        append = self.append_string
        ancil = self.aman.ancil
        
        weights = self.aman.weights
        fit_residuals = self.aman.fit_residuals
        nominal_xieta_locs = self.aman.nominal_xieta_locs

        fig, ax = plt.subplots()
        im = ax.scatter(nominal_xieta_locs[0], nominal_xieta_locs[1],
                        c=fit_residuals, alpha=0.11, cmap='jet',
                        linewidth=0, s=15, vmax=self.plotlims)
        sm = cm.ScalarMappable(cmap=im.cmap, norm=im.norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Fit Residual (arcmin)')
        ax.set_xlabel('Xi (rad)')
        ax.set_ylabel('Eta (rad)')
        plt.title('Fit Residuals across Focal Plane\n(Not averaged per det)')
        if platform == 'lat':
            plt.xlim(-.042, .042);plt.ylim(-.042, .042)
        else:
            plt.xlim(-.31, .31);plt.ylim(-.31, .31)
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_full_FocalPlane_colored_FitResiduals{tag}.png", dpi=350)

    def plot_full_histogram(self):
        platform = self.platform
        plotlims = self.plotlims
        plot_dir = self.plot_dir
        tag = self.tag
        append = self.append_string
        
        det_rms = self.aman.rms
        ufm_rms = self.aman.per_ufm_stats.rms
        obs_rms = self.aman.per_obs_stats.rms
        fit_residuals = self.aman.fit_residuals
        
        plt.figure()
        plt.hist(fit_residuals, bins=50, range=(0, plotlims))
        plt.axvline(det_rms , 0, 1, color='k',
                    label=f'Full RMS {det_rms:.2f} arcmin')
        plt.axvline(obs_rms, 0, 1, color='c',
                    label = f'Obs RMS {obs_rms:.2f} arcmin')
        plt.axvline(ufm_rms, 0, 1, color='m',
                    label=f'UFM RMS {ufm_rms:.2f} arcmin')
        plt.legend(fontsize='medium')
        plt.title(platform + ' 1D residuals')
        plt.xlabel('arcmin')
            
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_full_Hist_Residuals{tag}.png", dpi=350)     
            
    def plot_dets_in_these_obs(self):  
        platform = self.platform
        plot_dir = self.plot_dir
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights
        roll_c = self.aman.roll_c
        elev = self.aman.ancil.el_enc
        azim = self.aman.ancil.az_enc
        if platform == 'lat':
            elmin=10; elmax=90
        else:
            elmin=45; elmax=65
        
        plt.figure()
        fig, ax = plt.subplots(2,2,figsize=(11,10))
        ax[0,0].set_title('color by boresight.roll', fontsize='medium')
        a0 = ax[0,0].scatter(measured_xieta_data[0], measured_xieta_data[1], 
                           c=roll_c, alpha=0.5, s=10,
                           cmap='jet',vmin=-45,vmax=45)
        ax[1,0].set_title('color by fit Elevation', fontsize='medium')
        a1= ax[1,0].scatter(measured_xieta_data[0], measured_xieta_data[1], 
                          c=elev, alpha=0.3, s=10, cmap='jet', vmin=elmin, vmax=elmax)
        ax[0,1].set_title('color by fit Azimuth', fontsize='medium')
        a2= ax[0,1].scatter(measured_xieta_data[0], measured_xieta_data[1], 
                          c=azim, alpha=0.3, s=10, cmap='jet', vmin=0, vmax=420)
        ax[1,1].set_title('color by fit weights', fontsize='medium')
        a3= ax[1,1].scatter(measured_xieta_data[0], measured_xieta_data[1], 
                          c=weights, alpha=0.3, s=10, cmap='jet', vmin=0.8, vmax=1)
        c0 = plt.colorbar(a0)
        c1 = plt.colorbar(a1)
        c2 = plt.colorbar(a2)
        c3 = plt.colorbar(a3)
                
        plt.suptitle('Detectors hit in these observations')
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_dets_in_these_obs.png", dpi=350)
            plt.close()
    
    def plot_modeled_fits(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        ancil = self.aman.ancil
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights
        modeled_fits = self.aman.modeled_fits
        fit_rms = self.aman.fit_rms
        if self.which_ufm is not None:
            if isinstance(self.which_ufm, list):
                ufm_list = self.which_ufm
            else:
                ufm_list = [self.which_ufm]
        else:
            ufm_list = self.ufms
        plotmask = np.where(weights)
        nom_array = np.concatenate(
            [load_nom_focal_plane_full(self.config, ufm)[1] for ufm in ufm_list],
            axis=0,
        )
        rms = np.round(fit_rms, 4)
        xi_model_fit = modeled_fits.xi
        eta_model_fit = modeled_fits.eta
        if self.xieta_model == "measured":
            xi_ref, eta_ref, _ = measured_xieta_data
        elif self.xieta_model == "template":
            xi_ref, eta_ref, _ = nominal_xieta_locs
        if 'sat' in platform:
            markercolor = ancil.boresight_enc.copy()
            coloredby = "Boresight"
        elif 'lat' in platform:
            markercolor = ancil.corotator_enc.copy()
            coloredby = "Corotator"  
        scale_weights = weights / np.nanmax(weights)   
        ####
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[:, :])
        ax.scatter(
            xi_ref[plotmask] / DEG,
            eta_ref[plotmask] / DEG,
            c=markercolor[plotmask],
            alpha=0.4,
            label="Data",
            edgecolors="k",
            linewidths=0.4,
            s=130 * scale_weights[plotmask],
            cmap="jet",
            vmax=65,
        )
        sc3 = ax.scatter(
            xi_model_fit / DEG,
            eta_model_fit / DEG,
            marker="*",
            c=markercolor,
            cmap="jet",
            alpha=0.4,
            edgecolor="k",
            lw=0.4,
            s=65,
            label=f"Model, RMS = {rms}",
            vmax=65,
        )
        ax.scatter(
            nom_array[:, 0] / DEG,
            nom_array[:, 1] / DEG,
            marker=".",
            color="r",
            alpha=0.2,
            label="Nominal Center",
        )
        offsets = sc3.get_offsets()
        #xmin, ymin = offsets.min(axis=0)
        #xmax, ymax = offsets.max(axis=0)
        ax.legend(loc=1, fontsize="small")
        ax.set_xlabel("Xi (deg)")
        ax.set_ylabel("Eta (deg)")
        cb = plt.colorbar(sc3, fraction=0.046, pad=0.04)
        cb.ax.set_title(coloredby)
        ax.set_title(f"Fits, Colored by {coloredby} (deg)")
    
        # Plot lines connecting data to modeled data point
        xitoxi = np.empty((len(xi_model_fit), 2))
        xitoxi[:, 0] = xi_ref / DEG
        xitoxi[:, 1] = xi_model_fit / DEG
        etatoeta = np.empty((len(eta_model_fit), 2))
        etatoeta[:, 0] = eta_ref / DEG
        etatoeta[:, 1] = eta_model_fit / DEG
        ax.plot(xitoxi.T, etatoeta.T, "k", lw=0.4)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_model_fits{tag}.png", dpi=350)
            plt.close()
    
    
    def plot_ws0_modeled_fits(self): 
        platform = self.platform
        plot_dir = self.plot_dir
        xieta_model = self.xieta_model
        tag = self.tag
        ancil = self.aman.ancil
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights
        modeled_fits = self.aman.modeled_fits
        fit_rms = self.aman.fit_rms
        
        xi_model_fit = modeled_fits.xi
        eta_model_fit = modeled_fits.eta
        if xieta_model == "measured":
            xi_ref, eta_ref, _ = measured_xieta_data
        elif xieta_model == "template":
            xi_ref, eta_ref, _ = nominal_xieta_locs
        markercolor = [ancil.el_enc.copy(), ancil.boresight_enc.copy(), ancil.az_enc.copy()]
        coloredby = ["El", "Boresight", "Az"]
        vmins = [40, -50, 0]
        vmaxs = [65, 50, 360]
        scale_weights = weights / np.nanmax(weights)
        plotmask = np.where(weights)[0]
        rms = np.round(fit_rms, 4)
        
        # Calculate lines connecting data to modeled data point
        xitoxi = np.empty((len(xi_model_fit), 2))
        xitoxi[:, 0] = xi_ref / DEG
        xitoxi[:, 1] = xi_model_fit / DEG
        etatoeta = np.empty((len(eta_model_fit), 2))
        etatoeta[:, 0] = eta_ref / DEG
        etatoeta[:, 1] = eta_model_fit / DEG     
        
        ####
        #fig = plt.figure(figsize=(6, 6))
        #gs = fig.add_gridspec(2, 2)
        #ax = fig.add_subplot(gs[:, :])
        fig, ax = plt.subplots(1,3, figsize=(9,4), sharey=True)
        for x in range(3):
            ax[x].plot(
                nominal_xieta_locs[0, : 7 + 1] / DEG,
                nominal_xieta_locs[1, : 7 + 1] / DEG,
                "rx",
                label="Nominal Center",
                )
            ax[x].scatter(
                xi_ref[plotmask] / DEG,
                eta_ref[plotmask] / DEG,
                c=markercolor[x][plotmask],
                alpha=0.4,
                label="Data",
                edgecolors="k",
                linewidths=0.4,
                s=130 * scale_weights[plotmask],
                cmap="jet",
                vmin=vmins[x],
                vmax=vmaxs[x],
            )
            im = ax[x].scatter(
                xi_model_fit[plotmask] / DEG,
                eta_model_fit[plotmask] / DEG,
                marker="*",
                c=markercolor[x][plotmask],
                cmap="jet",
                edgecolor="gray",
                lw=0.3,
                s=130,
                label=f"Model, RMS = {rms}",
                vmin=vmins[x],
                vmax=vmaxs[x],
            )
            ax[x].legend(loc=1, fontsize="small")
            ax[x].set_xlabel("Xi (deg)")
            ax[x].set_ylabel("Eta (deg)")
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.set_title(coloredby[x])
            ax[x].set_title(f"Fits, Colored by {coloredby[x]} (deg)")
            ax[x].plot(xitoxi.T, etatoeta.T, "k", lw=0.4,alpha=0.5)
            ax[x].set_xlim(-1.05, 0.25)
            ax[x].set_ylim(-0.2, 0.2)
        # plt.subplots_adjust(left=0.1, right=0.90, bottom=0.05, hspace=0.3)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_ws0_model_fits{tag}.png", dpi=350)
            plt.close()
    

    def plot_template_space_fits_per_wafer(self):
        platform = self.platform
        plot_dir = self.plot_dir
        plotlims = self.plotlims
        pm_version = self.pm_version
        tag = self.tag
        ancil = self.aman.ancil
        roll_c = self.aman.roll_c
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights      
        pointing_model = self.aman.pointing_model
        modeled_fits = self.aman.modeled_fits
        fit_rms = self.aman.fit_rms    
        
        scale_weights = weights / np.nanmax(weights)
        xi_unmod, eta_unmod = model_template_xieta(
            pointing_model,
            pm_version,
            self.aman
        )
        xi0, eta0 = model_template_xieta(
            pm.param_defaults[pm_version],
            pm_version,
            self.aman
        )
        #Plot with Elevation as colorbar
        fig, ax = plt.subplots(2, 4, figsize=(9, 6))
        for i in range(7):
            ax[i // 4, i % 4].plot(0, 0, "kx", label="Nominal Center")
            im = ax[i // 4, i % 4].scatter(
                xi_unmod[i::7] / ARCMIN - nominal_xieta_locs[0, i] / ARCMIN,
                eta_unmod[i::7] / ARCMIN - nominal_xieta_locs[1, i] / ARCMIN,
                c=ancil.el_enc[i::7],
                s=scale_weights[i::7] * 80,
                edgecolor="gray",
                lw=0.3,
                marker="o",
                alpha=0.5,
                cmap="jet",
            )
            ax[i // 4, i % 4].set_xlim(-1 * plotlims, plotlims)
            ax[i // 4, i % 4].set_ylim(-1 * plotlims, plotlims)
            ax[i // 4, i % 4].set_title(f"ws{i}")
            ax[i // 4, i % 4].set_aspect('equal', adjustable='box')
        plt.colorbar(im, ax[1, 3], label="Elevation (deg)", fraction=0.046, pad=0.04)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_elevation{tag}.png", dpi=350)
            plt.close()
        #Plot with Roll as colorbar
        fig, ax = plt.subplots(2, 4, figsize=(9, 6))
        if "sat" in pm_version:
            markercolor = -1*ancil.boresight_enc[i::7].copy()
        if "lat" in pm_version:
            markercolor = roll_c[i::7].copy()
        coloredby = "roll"
        for i in range(7):
            ax[i // 4, i % 4].plot(0, 0, "kx", label="Nominal Center")
            im = ax[i // 4, i % 4].scatter(
                xi_unmod[i::7] / ARCMIN - nominal_xieta_locs[0, i] / ARCMIN,
                eta_unmod[i::7] / ARCMIN - nominal_xieta_locs[1, i] / ARCMIN,
                c=markercolor,
                s=scale_weights[i::7] * 80,
                edgecolor="gray",
                lw=0.3,
                marker="o",
                alpha=0.5,
                cmap="jet",
            )
            ax[i // 4, i % 4].set_xlim(-1 * plotlims, plotlims)
            ax[i // 4, i % 4].set_ylim(-1 * plotlims, plotlims)
            ax[i // 4, i % 4].set_title(f"ws{i}")
            ax[i // 4, i % 4].set_aspect('equal', adjustable='box')
        plt.colorbar(im, ax[1, 3], label=f"{coloredby} (deg)", fraction=0.046, pad=0.04)
        if self.save_figure:
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_roll{tag}.png", dpi=350)
    
        if "lat" in pm_version:
            markercolor = ancil.corotator_enc[i::7].copy()
            coloredby = "corotator"
            fig, ax = plt.subplots(2, 4, figsize=(9, 6))
            for i in range(7):
                ax[i // 4, i % 4].plot(0, 0, "kx", label="Nominal Center")
                im = ax[i // 4, i % 4].scatter(
                    xi_unmod[i::7] / ARCMIN - nominal_xieta_locs[0, i] / ARCMIN,
                    eta_unmod[i::7] / ARCMIN - nominal_xieta_locs[1, i] / ARCMIN,
                    c=ancil.boresight_enc[i::7],
                    s=scale_weights[i::7] * 80,
                    edgecolor="gray",
                    lw=0.3,
                    marker="o",
                    alpha=0.5,
                    cmap="jet",
                )
                ax[i // 4, i % 4].set_xlim(-1 * plotlims, plotlims)
                ax[i // 4, i % 4].set_ylim(-1 * plotlims, plotlims)
                ax[i // 4, i % 4].set_title(f"ws{i}")
                ax[i // 4, i % 4].set_aspect('equal', adjustable='box')
            plt.colorbar(im, ax[1, 3], label=f"{coloredby} (deg)", fraction=0.046, pad=0.04)
            plt.tight_layout()
            if self.save_figure:
                plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_corotator{tag}.png", dpi=350)
                plt.close()


    def plot_template_space_fits_per_detector(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        plotlims = self.plotlims
        pm_version = self.pm_version        
        ancil = self.aman.ancil
        roll_c = self.aman.roll_c
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights
        pointing_model = self.aman.pointing_model
        modeled_fits = self.aman.modeled_fits
        fit_rms = self.aman.fit_rms   
        
        scale_weights = weights / np.nanmax(weights)
        xi_unmod, eta_unmod = model_template_xieta(
            pointing_model,
            pm_version,
            self.aman
        )
        xi0, eta0 = model_template_xieta(
            pm.param_defaults[pm_version],
            pm_version,
            self.aman
        )
        #plot with weights as colorbar
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(0, 0, "kx", label="Nominal Center")
        im = ax.scatter(
            xi_unmod / ARCMIN - nominal_xieta_locs[0] / ARCMIN,
            eta_unmod / ARCMIN - nominal_xieta_locs[1] / ARCMIN,
            c=weights,
            s=scale_weights * 5,
            edgecolor="gray",
            lw=0.3,
            marker="o",
            alpha=0.2,
            cmap="viridis",
            vmin=self.config.get("weight_cutoff"),
            vmax=1
        )
        ax.set_xlim(-1 * plotlims, plotlims)
        ax.set_ylim(-1 * plotlims, plotlims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Unmodeled fits, by fit weight")
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_weights{tag}.png", dpi=350)
            plt.close()   
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(0, 0, "kx", label="Nominal Center")
        im = ax.scatter(
            xi_unmod / ARCMIN - nominal_xieta_locs[0] / ARCMIN,
            eta_unmod / ARCMIN - nominal_xieta_locs[1] / ARCMIN,
            c=ancil.el_enc,
            s=scale_weights * 5,
            edgecolor="gray",
            lw=0.3,
            marker="o",
            alpha=0.2,
            cmap="jet",
        )
        ax.set_xlim(-1 * plotlims, plotlims)
        ax.set_ylim(-1 * plotlims, plotlims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Unmodeled fits, by elevation color")
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_elevation{tag}.png", dpi=350)
            plt.close()
    
        fig, ax = plt.subplots(figsize=(9, 6))
        if "sat" in pm_version:
            markercolor = -1*ancil.boresight_enc
        if "lat" in pm_version:
            markercolor = roll_c
        coloredby = "roll"
        ax.plot(0, 0, "kx", label="Nominal Center")
        im = ax.scatter(
            xi_unmod / ARCMIN - nominal_xieta_locs[0] / ARCMIN,
            eta_unmod / ARCMIN - nominal_xieta_locs[1] / ARCMIN,
            c=markercolor,
            s=scale_weights * 5,
            edgecolor="gray",
            lw=0.3,
            marker="o",
            alpha=0.2,
            cmap="jet",
        )
        ax.set_xlim(-1 * plotlims, plotlims)
        ax.set_ylim(-1 * plotlims, plotlims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Unmodeled fits, colored by {coloredby} angle")
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_roll{tag}.png", dpi=350)
            plt.close()
    
        if "lat" in pm_version:
            markercolor = ancil.corotator_enc.copy()
            coloredby = "corotator"
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(0, 0, "kx", label="Nominal Center")
            im = ax.scatter(
                xi_unmod / ARCMIN - nominal_xieta_locs[0] / ARCMIN,
                eta_unmod / ARCMIN - nominal_xieta_locs[1] / ARCMIN,
                c=markercolor,
                s=scale_weights * 5,
                edgecolor="gray",
                lw=0.3,
                marker="o",
                alpha=0.2,
                cmap="jet",
            )
            ax.set_xlim(-1 * plotlims, plotlims)
            ax.set_ylim(-1 * plotlims, plotlims)
            ax.set_title(f"Unmodeled fits, colored by {coloredby} angle")
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            if self.save_figure:
                plt.savefig(f"{plot_dir}/{platform}_unmodeled_fits_WS_corotator{tag}.png", dpi=350)
                plt.close()
    
    
    def plot_residuals_vs_ancil(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        pm_version = self.pm_version
        xieta_model = self.xieta_model
        plotlims = self.plotlims
        ancil = self.aman.ancil
        roll_c = self.aman.roll_c
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        weights = self.aman.weights
        pointing_model = self.aman.pointing_model
        modeled_fits = self.aman.modeled_fits
        fit_rms = self.aman.fit_rms   
        
        scale_weights = weights / np.nanmax(weights)
        plotmask = np.where(weights)
        xi_model_fit = modeled_fits.xi
        eta_model_fit = modeled_fits.eta
        if xieta_model == "measured":
            xi_ref, eta_ref, _ = measured_xieta_data
        elif xieta_model == "template":
            xi_ref, eta_ref, _ = nominal_xieta_locs
    
        if "sat" in platform:
            third_enc = ancil.boresight_enc.copy()
            third_enc_name = "Boresight"
        elif "lat" in platform:
            third_enc = ancil.corotator_enc.copy()
            third_enc_name = "Corotator"        
        fig, ax = plt.subplots(2, 3, figsize=(8, 6), sharex="col", sharey="row")
        plt.setp(ax[0, 1].get_yticklabels(), visible=False)
        plt.suptitle(r"$\delta \xi$, $\delta \eta$" + f" vs Az, El, {third_enc_name}")
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
                x = ancil.az_enc % 360
            elif j == 1:
                x = ancil.el_enc
            elif j == 2:
                x = third_enc
            ax[i, j].scatter(
                x[plotmask],
                (model - ref)[plotmask] / ARCMIN,
                color="k",
                marker=".",
                alpha=0.1,
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
            ax[i, j].plot(
                xrange,
                mxb[0] * xrange + mxb[1],
                "r",
                lw=1,
                label=f"Slope {np.round(mxb[0],4)}\n [arcmin/deg]",
            )
            ax[i, j].legend(fontsize="small")
        ax[0, 0].set_ylabel("dXi [arcmin]")
        ax[1, 0].set_ylabel("dEta [arcmin]")
        ax[1, 0].set_xlabel("Azimuth [deg]")
        ax[1, 1].set_xlabel("Elevation [deg]")
        ax[1, 2].set_xlabel(f"{third_enc_name} [deg]")
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_residuals_vs_ancillary{tag}.png", dpi=350)
            plt.close()
     
    def plot_total_residuals(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        plotlims = self.plotlims
        pm_version = self.pm_version
        ancil = self.aman.ancil
        roll_c = self.aman.roll_c
        weights = self.aman.weights
        fit_rms = self.aman.fit_rms   
        obs_index = self.aman.obs_index
        
        scale_weights = weights / np.nanmax(weights) 
        effobs =np.where(np.diff(np.append(obs_index, obs_index[-1]+1))>0)[0]
        try:
            two_fits = np.any(_valid_arg("fit_residuals_i1", 'signal', src=self.aman))
        except:
            two_fits = False
        if two_fits:
            iterate_cutoff = self.config.get("iterate_cutoff")
            if iterate_cutoff == "auto":
                iterate_cutoff = np.nanstd(self.aman.fit_residuals_i1)*2 + np.nanmedian(self.aman.fit_residuals_i1)
            bad_fit_inds = self.aman.bad_fit_inds
            fit_residuals_i1 = self.aman.fit_residuals_i1
            fit_residuals_i2 = self.aman.fit_residuals
            fig = plt.figure(figsize=(6, 4))
            gs = fig.add_gridspec(7, 1)
            ax1 = fig.add_subplot(gs[0:-2, :])
            ax2 = fig.add_subplot(gs[-2:, :])
            # Plot first fit iteration residuals
            ax1.plot(
                np.arange(len(fit_residuals_i1)),
                fit_residuals_i1,
                "r.",
                mew=0,
                alpha=0.6,
                lw=0,
                label="1st Fit",
            )
            ax1.set_ylabel(r"Fit Residual $\left|\Delta$(xi, eta)$\right|$ [arcmin]")
            ax1.set_xlabel("Data point")
            ax1.axhline(
                iterate_cutoff,
                #xmin=0,
                #xmax=1,
                color="k",
                linestyle=":",
                lw=0.8,
                label="Cutoff",
            )
            # Plot second fit iteration residuals
            ax1.plot(
                np.arange(len(fit_residuals_i2)),
                fit_residuals_i2,
                "b*",
                alpha=0.5,
                lw=0,
                mew=0,
                label="2nd fit",
            )
            xtox = np.empty((len(fit_residuals_i2), 2))
            xtox[:, 0] = np.arange(len(fit_residuals_i1))
            xtox[:, 1] = np.arange(len(fit_residuals_i2))
            ytoy = np.empty((len(fit_residuals_i1), 2))
            ytoy[:, 0] = fit_residuals_i1 
            ytoy[:, 1] = fit_residuals_i2 
            ax1.plot(xtox.T, ytoy.T, "k", lw=0.4)
    
            ax1.axhline(0, xmin=0, xmax=1, color="k", alpha=0.5, lw=0.8)
            ax1.set_ylabel(r"Fit Residual $\left|\Delta\text{(xi, eta)}\right|$ [arcmin]")
            ax1.legend(loc=2, fontsize="small")

            ax2.scatter(
                np.arange(len(fit_residuals_i1)),
                (fit_residuals_i2 - fit_residuals_i1),
                c="k",
                marker="o",
                s=scale_weights * 50,
                lw=0,
                alpha=0.6,
                label="Res i2 - Res i1",
            )
            ax2.plot(
                np.arange(len(fit_residuals_i1))[bad_fit_inds],
                (fit_residuals_i2 - fit_residuals_i1)[bad_fit_inds],
                "kx",
                ms=7,
                lw=0.2,
                alpha=0.6,
                label="Excl. from i2 fit",
            )
            ax2.axhline(0, xmin=0, xmax=1, color="k", alpha=0.5, lw=0.8)
            for e in effobs:
                ax1.axvline(e,linestyle=':',color='k') 
                ax2.axvline(e,linestyle=':',color='k')  
            ax2.legend(fontsize="x-small")
            ax2.set_xlabel("Data points")
            ax2.set_ylabel(r"$\Delta$ Residuals")
            if self.save_figure:
                plt.savefig(f"{plot_dir}/{platform}_total_residuals{tag}.png", dpi=350)
                plt.close()
        else:
            # Plot first fit iteration residuals only
            fit_residuals_i1 = self.aman.fit_residuals
            fig, ax1 = plt.subplots()
            im = ax1.scatter(
                np.arange(len(fit_residuals_i1)),
                fit_residuals_i1,
                s=scale_weights * 50,
                #c=obs_index,
                c=weights,
                lw=0,
                alpha=0.4,
                label="1st Fit",
                vmax=1,
            )
            for e in effobs:
                ax1.axvline(e,linestyle=':',color='k')
            plt.colorbar(im, label='fit weights', fraction=0.046, pad=0.04)
            ax1.set_ylabel(r"Fit Residual $\left|\Delta\text{(xi, eta)}\right|$ [arcmin]")
            ax1.set_xlabel("Data points")
            plt.title("Fit Residuals")
            plt.legend(loc=2)
            if self.save_figure:
                plt.savefig(f"{plot_dir}/{platform}_total_residuals{tag}.png", dpi=350)
                plt.close()
            
    
    def plot_residuals_histograms(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        append = self.append_string
        ancil = self.aman.ancil
        weights = self.aman.weights
        fit_rms = self.aman.fit_rms
        fit_residuals = self.aman.fit_residuals
        fit_rms_full = self.aman.fit_rms_full
        fit_residuals_full = self.aman.fit_residuals_full

        xmax = np.nanmax(fit_residuals_full) * 1.1
        title = f"{append} {tag}"
        plt.figure()
        plt.hist(fit_residuals_full,
                 bins = 25, 
                 range = (0, xmax),
                 alpha=0.7,
                 label='All dets above weight threshold'
                )
        plt.hist(fit_residuals,
                 bins = 25, 
                 range = (0, xmax),
                 alpha=0.7, 
                 label='Subset in Fit'
                )
        plt.axvline(fit_rms_full,color='C0',
                    label=f'Weighted RMSE (all): {fit_rms_full:.3f} arcmin')
        plt.axvline(fit_rms, color='C1',
                    label=f'Weighted RMSE (set): {fit_rms:.3f} arcmin')
        plt.xlabel('Fit Residuals (arcmin)')
        plt.ylabel('# Detectors')
        plt.title(append)
        plt.legend()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_residuals_hists{tag}.png", dpi=350)
            plt.close()
        
    def plot_xieta_residuals(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        xieta_model = self.xieta_model
        weights = self.aman.weights
        modeled_fits = self.aman.modeled_fits
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        
        scale_weights = weights / np.nanmax(weights)
        plotmask = np.where(weights)
        xi_model_fit = modeled_fits.xi
        eta_model_fit = modeled_fits.eta
        if xieta_model == "measured":
            xi_ref, eta_ref, _ = measured_xieta_data
        elif xieta_model == "template":
            xi_ref, eta_ref, _ = nominal_xieta_locs
    
        fig, ax = plt.subplots(2, 1)
        for i, xe in enumerate(["Xi", "Eta"]):
            if xe == "Xi":
                xaxis_ref = xi_ref
                xlabel = "Xi"
                yref = xi_ref
                ymodel = xi_model_fit
                ylabel = "dXi"
            elif xe == "Eta":
                xaxis_ref = eta_ref
                xlabel = "Eta"
                yref = eta_ref
                ymodel = eta_model_fit
                ylabel = "dEta"
            # xi residuals vs xi
            im = ax[i].scatter(
                xaxis_ref[plotmask] / DEG,
                (ymodel - yref)[plotmask] / ARCMIN,
                marker=".",
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
            ax[i].set_ylim(-self.plotlims, self.plotlims)
            ax[i].set_xlabel(f"{xlabel} (deg)", fontsize="small")
            ax[i].set_ylabel(f"{ylabel} [arcmin]")
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_xieta_residuals{tag}.png", dpi=350)
            plt.close()
    
    
    def plot_xieta_cross_residuals(self):
        platform = self.platform
        plot_dir = self.plot_dir
        tag = self.tag
        xieta_model = self.xieta_model
        weights = self.aman.weights
        modeled_fits = self.aman.modeled_fits
        nominal_xieta_locs = self.aman.nominal_xieta_locs
        measured_xieta_data = self.aman.measured_xieta_data
        
        scale_weights = weights / np.nanmax(weights)
        plotmask = np.where(weights)
        xi_model_fit = modeled_fits.xi
        eta_model_fit = modeled_fits.eta
        if xieta_model == "measured":
            xi_ref, eta_ref, _ = measured_xieta_data
        elif xieta_model == "template":
            xi_ref, eta_ref, _ = nominal_xieta_locs

        fig, ax = plt.subplots(2, 1)
        for i, xe in enumerate(["Xi", "Eta"]):
            if xe == "Xi":
                xaxis_ref = eta_ref
                xlabel = "Eta"
                yref = xi_ref
                ymodel = xi_model_fit
                ylabel = "dXi"
            elif xe == "Eta":
                xaxis_ref = xi_ref
                xlabel = "Xi"
                yref = eta_ref
                ymodel = eta_model_fit
                ylabel = "dEta"
            # xi residuals vs xi
            im = ax[i].scatter(
                xaxis_ref[plotmask] / DEG,
                (ymodel - yref)[plotmask] / ARCMIN,
                marker=".",
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
            ax[i].set_ylim(-self.plotlims, self.plotlims)
            ax[i].set_xlabel(f"{xlabel} (deg)", fontsize="small")
        ax[i].set_ylabel(f"{ylabel} [arcmin]")
        plt.tight_layout()
        if self.save_figure:
            plt.savefig(f"{plot_dir}/{platform}_xieta_cross_residuals{tag}.png", dpi=350)
            plt.close()

   
############

if __name__ == "__main__":
    sp_util.main_launcher(main, get_parser)
