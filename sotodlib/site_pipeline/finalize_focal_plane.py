import argparse as ap
import datetime as dt
import logging
import os
from copy import deepcopy
from importlib import import_module
from typing import List, Optional

import h5py
import megham.transform as mt
import megham.utils as mu
import numpy as np
import yaml
from scipy.cluster import vq
from scipy.optimize import minimize
from so3g.proj import quat
from sotodlib.coords import optics as op
from sotodlib.coords.fp_containers import (
    FocalPlane,
    OpticsTube,
    Receiver,
    Template,
    Transform,
    plot_by_gamma,
    plot_ot,
    plot_receiver,
    plot_ufm,
)
from sotodlib.coords.pointing_model import apply_pointing_model
from sotodlib.core import AxisManager, Context, IndexAxis, metadata
from sotodlib.io.metadata import read_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _create_db(filename, per_obs, obs_id, start_time, stop_time):
    if per_obs:
        base = {"obs:obs_id": obs_id}
        group = obs_id
    else:
        base = {"obs:timestamp": (start_time, stop_time)}
        group = str(start_time)
    if os.path.isfile(filename):
        return metadata.ManifestDb(filename), base, group
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:stream_id")
    if per_obs:
        scheme.add_exact_match("obs:obs_id")
    else:
        scheme.add_range_match("obs:timestamp")
    scheme.add_data_field("dataset")

    metadata.ManifestDb(scheme=scheme).to_file(filename)
    return metadata.ManifestDb(filename), base, group


def _avg_focalplane(full_fp, tot_weight):
    # Figure out how many good pointings we have for each det
    msk = np.isfinite(full_fp)
    n_obs = np.sum(np.any(msk, axis=1), axis=-1)
    n_point, _, n_gamma = tuple(np.sum(msk, axis=-1).T)
    tot_weight[tot_weight[:, 0] == 0] = np.nan
    avg_fp = np.nansum(full_fp, axis=-1) / tot_weight[:, 0][..., None]
    avg_weight = tot_weight / n_obs[..., None]

    # nansum all all nans is 0, addressing that case here
    all_nan = ~np.any(
        np.isfinite(np.swapaxes(full_fp, 0, 1)).reshape((full_fp.shape[1], -1)), axis=1
    )
    avg_fp[:, all_nan] = np.nan

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
        np.array(det_ids), template, np.array(template_optical), pointing_cfg
    )


def _get_obs_ids(ctx, metalist, start_time, stop_time, query=None, obs_ids=[], tags=[]):
    all_obs = obs_ids
    query_obs = []
    if len(obs_ids) == 0:
        query_all = query
        if query is None:
            query_all = (
                f"type=='obs' and start_time>{start_time} and stop_time<{stop_time}"
            )
        if ctx.obsdb is None:
            raise ValueError("No obsdb!")
        all_obs = ctx.obsdb.query(query_all, tags=tags)["obs_id"]
    dbs = [
        metadata.ManifestDb(md["db"])
        for md in ctx["metadata"]
        if md.get("name", "") in metalist or md.get("label", "") in metalist
    ]
    with_meta = np.unique(
        np.hstack(
            [np.array([entry["obs:obs_id"] for entry in db.inspect()]) for db in dbs]
        )
    )
    all_obs = np.intersect1d(all_obs, with_meta)

    if query is not None:
        query_obs = ctx.obsdb.query(query)["obs_id"]
    obs_ids += query_obs

    if len(obs_ids) == 0 and query is None:
        return all_obs
    return np.intersect1d(obs_ids, all_obs)


def _load_ctx(config):
    ctx = Context(config["context"]["path"])
    if ctx.obsdb is None:
        raise ValueError("No obsdb!")
    tod_pointing_name = config["context"].get("tod_pointing", "tod_pointing")
    map_pointing_name = config["context"].get("map_pointing", "map_pointing")
    pol_name = config["context"].get("polarization", "polarization")
    dm_name = config["context"].get("detmap", "detmap")
    roll_range = config.get("roll_range", [-1 * np.inf, np.inf])
    obs_ids = _get_obs_ids(
        ctx,
        [tod_pointing_name, map_pointing_name, pol_name],
        config["start_time"],
        config["stop_time"],
        config["context"].get("query", None),
        config["context"].get("obs_ids", []),
        config["context"].get("tags", ["timing_issues=0"]),
    )
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")
    amans = []
    dets = config["context"].get("dets", {})
    for obs_id in obs_ids:
        roll = ctx.obsdb.get(obs_id)["roll_center"]
        if roll < roll_range[0] or roll > roll_range[1]:
            logger.info("%s has a roll that is out of range", obs_id)
            continue
        try:
            aman = ctx.get_meta(obs_id, dets=dets)
        except metadata.loader.LoaderError:
            logger.error("Failed to load %s, skipping", obs_id)
            continue
        if aman.obs_info.tube_slot == "stp1":
            aman.obs_info.tube_slot = "st1"
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
    obs_ids = [aman.obs_info.obs_id for aman in amans]
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
        [stream_id],
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
    fp, _, template_msk = focal_plane.map_by_det_id(aman)
    fp = fp[:, :2]
    inliers = np.ones(len(fp), dtype=bool)

    rad_thresh = 1.05 * np.nanmax(
        np.linalg.norm(
            focal_plane.template.fp[:, :2] - focal_plane.template.center[:, :2], axis=1
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
    rot, sft = mt.get_rigid(fp, focal_plane.template.fp[template_msk, :2])
    fp_aligned = mt.apply_transform(fp, rot, sft)
    likelihood = mu.gen_weights(fp_aligned, focal_plane.template.fp[template_msk, :2])
    inliers *= likelihood > 0.61  # ~1 sigma cut

    # Now restrict the AxisManager
    inlier_det_ids = focal_plane.template.det_ids[template_msk][inliers]
    return aman.restrict(
        "dets", aman.dets.vals[np.isin(aman.det_info.det_id, inlier_det_ids)]
    )


def _apply_pointing_model(config, aman):
    if "pointing_model" not in config:
        logger.info("\t\tNo pointing model specified!")
        return aman
    if not config["pointing_model"].get("apply", False):
        logger.info("\t\tNot applying pointing model")
        return aman
    if "function" not in config["pointing_model"]:
        logger.info("\t\tUsing default pointing model function")
        func = apply_pointing_model
    else:
        func = getattr(
            import_module(config["pointing_model"]["function"][0]),
            config["pointing_model"]["function"][1],
        )
    if "az" not in aman.pointing:
        raise ValueError("Need to have az in pointing fits to apply pointing model")
    if "el" not in aman.pointing:
        raise ValueError("Need to have el in pointing fits to apply pointing model")
    if "roll" not in aman.pointing:
        raise ValueError("Need to have roll in pointing fits to apply pointing model")

    params = config["pointing_model"].get("params", {})
    if "pointing_model" in aman:
        for key, val in params.items():
            if key in aman.pointing_model:
                aman.pointing_model[key] = val
            else:
                aman.pointing_model.wrap(key, val)
        params = aman.pointing_model
    ancil = AxisManager(IndexAxis("samps", aman.dets.count))
    ancil.wrap("az_enc", np.rad2deg(aman.pointing.az))
    ancil.wrap("el_enc", np.rad2deg(aman.pointing.el))
    ancil.wrap("roll_enc", np.rad2deg(aman.pointing.roll))
    bs = func(aman, params, ancil, False)
    q_fp = quat.rotation_xieta(aman.pointing.xi, aman.pointing.eta)
    have_gamma = False
    if "gamma" in aman.pointing:
        if np.any(np.isnan(aman.pointing.gamma)):
            logger.warning(
                "\t\tnans in gamma, not including in pointing model correction"
            )
        else:
            q_fp = quat.rotation_xieta(
                aman.pointing.xi, aman.pointing.eta, aman.pointing.gamma
            )
            have_gamma = True

    xi, eta, gamma = quat.decompose_xieta(
        ~quat.euler(2, bs.roll)
        * ~quat.rotation_lonlat(-bs.az, bs.el)
        * quat.rotation_lonlat(-1 * aman.pointing.az, aman.pointing.el)
        * quat.euler(2, aman.pointing.roll)
        * q_fp
    )

    aman.pointing.xi[:] = xi
    aman.pointing.eta[:] = eta
    if have_gamma:
        aman.pointing.gamma[:] = gamma

    return aman


def _reverse_roll(fp, aff, sft, aman):
    if "obs_info" not in aman:
        raise ValueError("Can't reverse roll without obs information")
    if "roll_center" not in aman.obs_info:
        raise ValueError("Can't reverse roll without roll information")
    roll = -1 * np.deg2rad(aman.obs_info.roll_center)

    # We want to shift so we rotating about the origin
    # To get to nominal we do fp@aff + sft
    # So if we just want to recenter we do fp + sft@aff^-1
    inv_aff, _ = mt.invert_transform(aff, np.zeros_like(sft))
    sft_adj = sft @ inv_aff
    fp_sft = fp[:, :2] + sft_adj

    # Now lets reverse the roll
    # The transpose is the inverse
    rot = np.array([[np.cos(roll), -1 * np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    fp_rot = fp_sft @ rot

    # And undo the shift, keeping track of rotations
    fp_rot -= sft_adj @ rot

    # Make sure its set
    fp[:, :2] = fp_rot

    # For gamma lets just shift by the roll
    fp[:, 2] -= roll

    return fp


def main():
    # Read in input pars
    parser = ap.ArgumentParser()
    parser.add_argument("config_path", help="Location of the config file")
    parser.add_argument(
        "--per_obs", "-p", action="store_true", help="Run in per observation mode"
    )
    parser.add_argument(
        "--include_cm",
        "-i",
        action="store_true",
        help="Include the common mode in the final detector positions",
    )
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    per_obs = config.get("per_obs", args.per_obs)
    include_cm = config.get("include_cm", args.include_cm)

    # Build output path
    append = config.get("append", "")
    dbroot = f"db{bool(append)*'_'}{append}"
    froot = f"focal_plane{bool(append)*'_'}{append}"
    subdir = config.get("subdir", "")
    subdir = subdir + (subdir == "") * (
        per_obs * "per_obs" + (not per_obs) * "combined"
    )
    outdir = os.path.join(config["outdir"], subdir)
    outpath = os.path.abspath(os.path.join(outdir, f"{froot}.h5"))
    dbpath = os.path.join(outdir, f"{dbroot}.sqlite")
    logpath = os.path.join(outdir, f"{froot}.log")
    os.makedirs(outdir, exist_ok=True)
    plot_dir_base = config.get("plot_dir", None)
    if plot_dir_base is not None:
        plot_dir_base = os.path.join(
            plot_dir_base, subdir + bool(append) * "_" + append
        )
        plot_dir_base = os.path.abspath(plot_dir_base)
        os.makedirs(plot_dir_base, exist_ok=True)

    # Log file
    logfile = logging.FileHandler(logpath)
    logger.addHandler(logfile)

    # Time range
    config["start_time"] = config.get("start_time", 0)
    config["stop_time"] = config.get("stop_time", 2**32)
    logger.info(
        "Running on time range %s to %s",
        dt.datetime.fromtimestamp(config["start_time"]),
        dt.datetime.fromtimestamp(config["stop_time"]),
    )

    # Load data
    if "context" in config:
        amans, obs_ids, stream_ids = _load_ctx(config)
    elif "resultsets" in config:
        amans, obs_ids, stream_ids = _load_rset(config)
    else:
        raise ValueError("No valid inputs provided")

    weight_factor = config.get("weight_factor", 1000)
    min_points = config.get("min_points", 50)
    gen_template = "template" not in config
    template_path = config.get("template", "nominal.h5")
    have_template = os.path.exists(template_path)
    if not gen_template and not have_template:
        logger.error("Provided template doesn't exist, trying to generate one")
        gen_template = True

    # Need to move installed OT and WS of array to templace for this
    # if config.get("pad", False):
    #     logger.info("Padding missing arrays with template, getting complete list of arrays from template")
    #     if not have_template:
    #         logger.warning("\tNo template provided, arrays not found in any observations will be missing")
    #     with h5py.File(template_path) as f:
    #         stream_ids = list(f.keys()) 

    # Split up into batches
    # Right now either per_obs or all at once
    # Maybe allow for batch my encoder angle later?
    if per_obs:
        logger.info("Running in per_obs mode")
        batches = [([aman], [obs_id]) for aman, obs_id in zip(amans, obs_ids)]
    else:
        batches = [(amans, obs_ids)]
    for amans, obs_ids in batches:
        plot_dir = plot_dir_base
        if per_obs:
            plot_dir = os.path.join(plot_dir_base, obs_ids[0])
        else:
            plot_dir = os.path.join(plot_dir_base, str(config["start_time"]))
        os.makedirs(plot_dir, exist_ok=True)
        logger.info("Working on batch containing: %s", str(obs_ids))
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
                message = "\tSomehow no AxisManagers with stream_id %s, skipping"
                if per_obs:
                    logger.info(message, stream_id)
                else:
                    logger.error(message, stream_id)
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
                ots[ot] = OpticsTube.from_pointing_cfg(pointing_cfg)

            # If a template is provided load it, otherwise generate one
            if gen_template:
                logger.info(f"\tGenerating template for {stream_id}")
                if "wafer_info" not in config:
                    raise ValueError("Need wafer_info to generate template")
                template_det_ids, template, is_optical = op.gen_template(
                    config["wafer_info"], stream_id, **pointing_cfg
                )
                template = Template(
                    template_det_ids,
                    template,
                    is_optical,
                    pointing_cfg,
                )
            elif have_template:
                logger.info("\tLoading template from %s", template_path)
                template = _load_template(template_path, stream_id, pointing_cfg)
            else:
                raise ValueError(
                    "No template provided and unable to generate one for some reason"
                )

            focal_plane = FocalPlane.empty(template, stream_id, ws, len(amans))
            if focal_plane.template is None:
                raise ValueError("Template is somehow None")

            for i, (aman, obs_id) in enumerate(zip(amans_restrict, obs_ids)):
                logger.info("\tWorking on %s", obs_id)
                if aman.dets.count < min_points:
                    logger.info("\t\tToo few dets found, skipping")
                    continue

                if config.get("faked_gamma", False):
                    aman.pointing.gamma[:] = np.nan

                # Restrict to optical dets
                optical = np.isin(
                    aman.det_info.det_id, focal_plane.template.det_ids[template.optical]
                )
                aman.restrict("dets", aman.dets.vals[optical])
                if aman.dets.count == 0:
                    logger.info("\t\tNo optical dets, skipping...")
                    continue

                # Apply pointing model if we want to
                aman = _apply_pointing_model(config, aman)

                # Do some outlier cuts
                if "hits" in aman.pointing:
                    aman.restrict(
                        "dets", aman.pointing.hits > config.get("min_hits", 5)
                    )
                    if aman.dets.count == 0:
                        logger.info("\t\tNo high hits dets, skipping...")
                        continue
                if aman.dets.count < min_points:
                    logger.info("\t\tToo few dets found, skipping")
                    continue
                _restrict_inliers(aman, focal_plane)

                # Mapping to template
                fp, r2, template_msk = focal_plane.map_by_det_id(aman)
                focal_plane.template.add_wafer_info(aman, template_msk)

                # Try an initial alignment and get weights
                try:
                    aff, sft = mt.get_rigid(
                        fp[:, :2], focal_plane.template.fp[template_msk, :2]
                    )
                except ValueError as e:
                    logger.error("\t\t%s", e)
                    continue
                aligned = mt.apply_transform(fp[:, :2], aff, sft)

                if config.get("reverse_roll", False):
                    fp = _reverse_roll(fp, aff, sft, aman)
                if np.any(np.isfinite(fp[:, 2])):
                    gscale, gsft = gamma_fit(
                        fp[:, 2], focal_plane.template.fp[template_msk, 2]
                    )
                    weights = mu.gen_weights(
                        np.column_stack((aligned, gscale * fp[:, 2] + gsft)),
                        focal_plane.template.fp[template_msk],
                        focal_plane.template.spacing.ravel() / weight_factor,
                    )
                else:
                    weights = mu.gen_weights(
                        aligned,
                        focal_plane.template.fp[template_msk, :2],
                        focal_plane.template.spacing[:2].ravel() / weight_factor,
                    )
                # ~1 sigma cut
                weights[weights < 0.61] = np.nan
                if np.sum(np.isfinite(weights)) < min_points / 2:
                    logger.error("\t\tToo few points! Skipping...")

                # Store weighted values
                weights = np.column_stack((weights, r2))
                focal_plane.add_fp(i, fp, weights, template_msk)

            # Compute the average focal plane with weights
            (
                focal_plane.avg_fp,
                focal_plane.weights,
                focal_plane.n_point,
                focal_plane.n_gamma,
            ) = _avg_focalplane(focal_plane.full_fp, focal_plane.tot_weight)
            tot_points = np.sum((focal_plane.n_point > 0).astype(int))
            logger.info("\t%d points in fit", tot_points)
            if tot_points < min_points:
                logger.error("\tToo few points! Skipping...")
                if config.get("pad", False):
                    logger.info("\tPadding output with template")
                    focal_plane.transformed = focal_plane.template.fp
                    focal_plane.tot_weight = None
                    ots[ot].focal_planes.append(focal_plane)
                continue

            try:
                affine, shift = mt.get_affine_two_stage(
                    focal_plane.template.fp[:, :2],
                    focal_plane.avg_fp[:, :2],
                    focal_plane.weights[:, 0],
                )
            except ValueError as e:
                logger.error("\t%s", e)
                continue

            focal_plane.transformed[:, :2] = mt.apply_transform(
                focal_plane.template.fp[:, :2], affine, shift
            )
            focal_plane.center_transformed[:, :2] = mt.apply_transform(
                focal_plane.template.center[:, :2], affine, shift
            )

            # Compute transformation between the two nominal and measured pointing
            focal_plane.have_gamma = np.sum(focal_plane.n_gamma) > 0
            if focal_plane.have_gamma:
                gamma_scale, gamma_shift = gamma_fit(
                    focal_plane.template.fp[:, 2], focal_plane.avg_fp[:, 2]
                )
            else:
                logger.warning(
                    "\tNo polarization data availible, gammas will be based on the nominal values."
                )
                logger.warning(
                    "\tSetting gamma shift to the xi-eta rotation and scale to 1.0"
                )
                transform = Transform.from_split(np.array((*shift, 0.0)), affine, 1.0)
                gamma_scale = 1.0
                gamma_shift = transform.rot
            focal_plane.transformed[:, 2] = (
                focal_plane.template.fp[:, 2] * gamma_scale + gamma_shift
            )
            focal_plane.center_transformed[:, 2] = (
                focal_plane.template.center[:, 2] * gamma_scale + gamma_shift
            )

            rms = np.sqrt(
                np.nanmean(
                    (
                        focal_plane.avg_fp[:, : (2 + focal_plane.have_gamma)]
                        - focal_plane.transformed[:, : (2 + focal_plane.have_gamma)]
                    )
                    ** 2
                )
            )
            logger.info("\tRMS after transformation is %f", rms)

            shift = np.array((*shift, gamma_shift))
            focal_plane.transform = Transform.from_split(shift, affine, gamma_scale)
            _log_vals(
                focal_plane.transform.shift,
                focal_plane.transform.scale,
                focal_plane.transform.shear,
                focal_plane.transform.rot,
                ("xi", "eta", "gamma"),
            )

            if config.get("plot", False):
                plot_ufm(focal_plane, plot_dir)
                plot_by_gamma(focal_plane, plot_dir)
            ots[ot].focal_planes.append(focal_plane)

        # Per OT common mode
        todel = []
        for name, ot in ots.items():
            logger.info("Fitting common mode for %s", ot.name)
            if ot.num_fps == 0:
                logger.error("\tNo focal planes found! Skipping...")
                if not config.get("pad", False):
                    todel.append(name)
                continue
            centers = np.vstack([fp.template.center for fp in ot.focal_planes if fp.tot_weight is not None])
            centers_transformed = np.vstack(
                [fp.center_transformed for fp in ot.focal_planes if fp.tot_weight is not None]
            )
            plot_ot(ot, plot_dir)
            if centers.shape[0] < 3:
                logger.warning(
                    "\tToo few wafers fit to compute common mode, transform will be approximated"
                )
                centers = np.vstack([ot.center, ot.center - 1, ot.center + 1])
                centers_transformed = np.vstack(
                    [
                        mt.apply_transform(
                            centers, fp.transform.affine, fp.transform.shift
                        )
                        for fp in ot.focal_planes
                    ],
                )
                centers = np.repeat(centers, len(ot.focal_planes), 0)
            rot, sft = mt.get_rigid(centers[:, :2], centers_transformed[:, :2])
            gamma_shift = np.mean(centers_transformed[:, 2] - centers[:, 2])
            ot.transform_fullcm = Transform.from_split(
                np.array((*sft.ravel(), gamma_shift)), rot, 1.0
            )
            ot.center_transformed = mt.apply_transform(
                ot.center, ot.transform_fullcm.affine, ot.transform_fullcm.shift
            )
            _log_vals(
                ot.transform_fullcm.shift,
                ot.transform_fullcm.scale,
                ot.transform_fullcm.shear,
                ot.transform_fullcm.rot,
                ("xi", "eta", "gamma"),
            )
        for ot in todel:
            del ots[ot]

        # Full receiver common mode
        logger.info("Fitting receiver common mode")
        if len(ots) == 0:
            logger.error("\tNo optics tubes found! Skipping...")
            continue
        elif len(ots) == 1:
            logger.info(
                "\tOnly one OT found, receiver common mode will be from this tube"
            )
            recv_transform = deepcopy(tuple(ots.values())[0].transform_fullcm)
        else:
            centers = np.vstack([ot.center for ot in ots.values() if ot.num_fps > 0])
            centers_transformed = np.vstack(
                [ot.center_transformed for ot in ots.values() if ot.num_fps > 0]
            )
            if len(ots) < 3:
                logger.info(
                    "\tNot enough OTs to fit receiver common mode, transform will be approximated"
                )
                centers = np.vstack(
                    [np.roll(np.arange(3, dtype=float), i) for i in range(3)]
                )
                centers_transformed = np.vstack(
                    [
                        mt.apply_transform(
                            centers, ot.transform.affine, ot.transform.shift
                        )
                        for ot in ots.values()
                    ],
                )
                centers = np.repeat(centers, len(ots), 0)
            rot, sft = mt.get_rigid(centers[:, :2], centers_transformed[:, :2])
            gamma_shift = np.mean(centers_transformed[:, 2] - centers[:, 2])
            recv_transform = Transform.from_split(
                np.array((*sft.ravel(), gamma_shift)), rot, 1.0
            )
        receiver = Receiver(
            list(ots.values()), transform=recv_transform, include_cm=include_cm
        )
        _log_vals(
            recv_transform.shift,
            recv_transform.scale,
            recv_transform.shear,
            recv_transform.rot,
            ("xi", "eta", "gamma"),
        )
        plot_receiver(receiver, plot_dir)

        # Now compute correction only transform for each ufm
        # Transforms are composed as ufm(ot(rx(focal_plane)))
        for ot in receiver.optics_tubes:
            # Now remove the receiver CM from the OT
            ot.transform.affine, ot.transform.shift = mt.decompose_transform(
                ot.transform_fullcm.affine,
                ot.transform_fullcm.shift,
                recv_transform.affine,
                recv_transform.shift,
            )
            ot.transform.decompose()

            # Now for each fp remove the CM
            for fp in ot.focal_planes:
                (
                    fp.transform_nocm.affine,
                    fp.transform_nocm.shift,
                ) = mt.decompose_transform(
                    fp.transform.affine,
                    fp.transform.shift,
                    ot.transform_fullcm.affine,
                    ot.transform_fullcm.shift,
                )
                fp.transform_nocm.decompose()
                # Remove the common mode if desired
                if not include_cm and fp.template is not None:
                    fp.transformed = mt.apply_transform(
                        fp.template.fp,
                        fp.transform_nocm.affine,
                        fp.transform_nocm.shift,
                    )

        # Make final outputs and save
        logger.info("Saving data to %s", outpath)
        logger.info("Writing to database at %s", dbpath)
        db, base, group = _create_db(
            dbpath,
            per_obs=per_obs,
            obs_id=obs_ids[0],
            start_time=config["start_time"],
            stop_time=config["stop_time"],
        )
        if config.get("pad", False):
            logger.info("Padding missing arrays with values from template")
        with h5py.File(outpath, "a") as f:
            if group in f:
                del f[group]
            f.create_group(group)
            receiver.save(f, (db, base), group)


if __name__ == "__main__":
    main()
