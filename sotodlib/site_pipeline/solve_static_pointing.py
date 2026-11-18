"""
Jointly solve for the static pointing model
and detector offset. Should eventually also
include the ability to incorporate external
datasets (starcamera, tiltmeter, etc.).
"""

import argparse as ap
import datetime as dt
import hashlib
import os
import shutil
from enum import Enum
from typing import cast
from copy import deepcopy
import corner
import matplotlib.pyplot as plt
from functools import partial

import emcee
import git
import h5py
import numpy as np
import pyinstrument
import sotodlib.coords.fp_containers as fpc
import sotodlib.coords.fp_plots as fpp
import sotodlib.utils.jdb_utils as ju
import sqlalchemy as sqy
import yaml
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from sotodlib.coords import optics
from sotodlib.core import Context, metadata
from sotodlib.io.metadata import read_dataset
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.utils import epochs
from sotodlib.utils.config import load_config_namespace

logger = init_logger(__name__, "solve_static_pointing: ")
default_config = {
    "tel": "LAT",
    "fake_gamma": True,
    "force_zero_roll": False,
    "rx_groups": [],
    "pm_groups": [],
    "pad_epochs": [],
    "pm_ver_override": {},
    "pm_defaults_overrides": {},
    "par_groups": {},
    "root_dir": "~",
    "project_dir": "pointing",
    "append": "",
    "point_meta": "focal_plane",
    "obs_whitelist": [],
    "obs_blacklist": [],
    "weight_factor": 1000,
    "min_frac_dets": 0.1,
    "max_dscale": 0.1,
    "max_shear": 0.05,
    "min_hits": 3,
    "min_r2": 0.3,
    "pad_empty": True,
    "parameters": {},
    "n_rounds": 3,
    "outlier_sigma": 20,
    "mcmc_steps": 0,
    "nwalkers": 32,
    "save_every": 10,
    "nuisance_params": {},
    "obs_time_thresh": 180,
    "nuisance_scale": 10,
    "activate_nuisance": 1,
    "activate_nuisance_transforms": 1,
    "least_squares": True,
    "cm_fac": 1e-3,
    "proposal_factor": 1e-2,
}


class ErrCode(Enum):
    NO_ERR = 0
    LOADER = 1
    MIN_DETS = 2
    AFF_ERR = 3
    HIGH_SCALE = 4
    HIGH_SHEAR = 5
    PLOT = 6
    FPINIT = 7



def least_squares_fixed(fun, x0, bounds, x_scale, jac_sparsity, args=(), **kwargs):
    x0 = np.asarray(x0, dtype=float)
    lb = np.asarray(bounds[0], dtype=float)
    ub = np.asarray(bounds[1], dtype=float)
    fixed_mask = lb == ub
    free_mask = ~fixed_mask
    x_fixed = x0.copy()
    x_fixed[fixed_mask] = lb[fixed_mask]
    if jac_sparsity is not None:
        jac_sparsity = jac_sparsity[:, free_mask]

    def wrapped_fun(x_free, *args):
        x = x_fixed.copy()
        x[free_mask] = x_free
        return fun(x, *args)

    result = least_squares(
        wrapped_fun,
        x0[free_mask],
        args=args,
        bounds=(lb[free_mask], ub[free_mask]),
        x_scale=x_scale[free_mask],
        jac_sparsity=jac_sparsity,
        **kwargs,
    )

    x_full = x_fixed.copy()
    x_full[free_mask] = result.x
    result.x = x_full
    result.fixed_mask = fixed_mask

    return result


def _report_fit(res, cfg, system, npar, x0, report_static, report_nuisance):
    logger.info(res.message)
    logger.info("Success: %s", res.success)
    logger.info("Objective: %g", res.cost if cfg.least_squares else res.fun)
    if cfg.least_squares:
        logger.info("Optimality: %g", res.optimality)
    if report_static:
        changes = system.parameters - x0[:npar]
        header = f"{'Parameter Name':<35} {'Optimized Value':>18} {'Change':>18}"
        logger.info("Parameter changes:\n%s", header)
        logger.info("-" * len(header))
        for name, value, change in zip(system.parameter_names, system.parameters, changes):
            logger.info(f"{name:<35} {value:>18.8g} {change:>18.8g}")
    if report_nuisance:
        nn = len(cfg.nuisance_params)
        cur = system.nuisance_params.params_flat.reshape((-1, nn))
        avg = np.mean(cur, axis=0)
        tot = np.sum(cur, axis=0)
        changes = (system.nuisance_params.params_flat - x0[npar:]).reshape((-1, nn))
        avg_change = np.mean(changes, axis=0)
        tot_change = np.sum(changes, axis=0)
        header = f"{'Nuisance Name':<35} {'Average Value':>18} {'Average Change':>18} {'Net Value':>18} {'Net Change':>18}"
        logger.info("Parameter changes:\n%s", header)
        logger.info("-" * len(header))
        for name, a, ac, t, tc in zip(cfg.nuisance_params.keys(), avg, avg_change, tot, tot_change):
            logger.info(f"{name:<35} {a:>18.8g} {ac:>18.8g} {t:>18.8g} {tc:>18.8g}")

def _fit_round(system, x0, bounds, scales, nresid, use_least_squares, jac_sparsity, cm_fac, **kwargs):
    if use_least_squares:
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        res = least_squares_fixed(
            system.objective_resid,
            x0,
            bounds=(lb, ub),
            x_scale=scales,
            jac_sparsity=jac_sparsity,
            args=(nresid, cm_fac),
            method="trf",
            loss="linear", #"soft_l1",
            tr_solver="lsmr",
            verbose=2,
            **kwargs,
        )

        return res, res.x

    scaled_bounds = [(lo / scale, hi / scale) for (lo, hi), scale in zip(bounds, scales)]
    def scaled_objective(z):
        return system.objective(z * scales, nresid, cm_fac)

    res = minimize(
        scaled_objective,
        x0 / scales,
        method="L-BFGS-B",
        bounds=scaled_bounds,
        **kwargs,
    )

    return res, res.x * scales


def _run_mcmc(system, nresid, cfg, bounds, scales, data_dir, plot_dir):
    system.set_static(False)
    system.set_thaw(True)

    x0 = system.parameters_full.copy()
    ndim = len(x0)
    npar = len(system.parameters)
    nnuisance = ndim - npar

    scales = np.asarray(scales, dtype=float)
    lb = np.asarray([b[0] for b in bounds], dtype=float)
    ub = np.asarray([b[1] for b in bounds], dtype=float)

    if len(scales) != ndim or len(bounds) != ndim:
        raise ValueError("Inconsistent parameter dimensions.")

    fixed_mask = lb == ub
    free_mask = ~fixed_mask

    # Divide the parameters into two blocks
    physical_mask = np.zeros(ndim, dtype=bool)
    physical_mask[:npar] = True
    nuisance_mask = ~physical_mask
    physical_free = physical_mask * free_mask
    nuisance_free = nuisance_mask * free_mask
    nphysical = np.sum(physical_free)
    nnuisance_free = np.sum(nuisance_free)

    if nphysical + nnuisance_free == 0:
        raise ValueError("All parameters are fixed; nothing to sample.")

    # Enforce fixed parameter values
    x = x0.copy()
    x[fixed_mask] = lb[fixed_mask]

    # Proposal scales
    proposal_scales = scales * cfg.proposal_factor
    proposal_scales[npar:] /= 10
    if np.any(proposal_scales[free_mask] <= 0):
        raise ValueError("All free parameters must have positive proposal scales.")

    logger.info( "Gibbs MCMC: %d physical, %d nuisance, %d fixed parameters", nphysical, nnuisance_free, np.sum(fixed_mask))
    logger.info( "Running %d Gibbs iterations", cfg.mcmc_steps)

    # Full joint log probability
    def log_prob(x):
        if np.any(x < lb) or np.any(x > ub):
            return -np.inf
        try:
            value = system.log_prob(x, nresid, cfg.cm_fac)
        except (
            RuntimeError,
            AssertionError,
            ValueError,
            FloatingPointError,
        ):
            return -np.inf
        if not np.isfinite(value):
            return -np.inf
        return value
    current_logp = log_prob(x)

    if not np.isfinite(current_logp):
        raise RuntimeError("Initial parameter vector has invalid log probability.")
    diagnostics = {
         "physical": {
             "bounds": 0,
             "nonfinite": 0,
             "errors": 0,
             "metropolis": 0,
             "accepted": 0,
             "log_alpha": [],
         },
         "nuisance": {
             "bounds": 0,
             "nonfinite": 0,
             "errors": 0,
             "metropolis": 0,
             "accepted": 0,
             "log_alpha": [],
         },
     }
    def metropolis_step(x, current_logp, mask, block_name):
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return x, current_logp, False

        proposal = x.copy()

        proposal[indices] += (
            rng.normal(size=len(indices))
            * proposal_scales[indices]
        )

        if np.any(proposal[indices] < lb[indices]) or np.any(
            proposal[indices] > ub[indices]
        ):
            diagnostics[block_name]["bounds"] += 1
            return x, current_logp, False

        try:
            proposal_logp = log_prob(proposal)
        except Exception:
            diagnostics[block_name]["errors"] += 1
            raise

        if not np.isfinite(proposal_logp):
            diagnostics[block_name]["nonfinite"] += 1
            return x, current_logp, False

        log_alpha = proposal_logp - current_logp

        diagnostics[block_name]["log_alpha"].append(log_alpha)

        if np.log(rng.random()) < min(0.0, log_alpha):
            diagnostics[block_name]["accepted"] += 1
            return proposal, proposal_logp, True

        diagnostics[block_name]["metropolis"] += 1
        return x, current_logp, False

    rng = np.random.default_rng()

    # Initialize chain storage
    samples = np.empty((cfg.mcmc_steps, ndim))
    physical_accept = 0
    nuisance_accept = 0

    # Gibbs sampling
    for i in range(cfg.mcmc_steps):
        # Sample physical parameters conditional on the current nuisance parameters
        x, current_logp, accepted = metropolis_step(x, current_logp, physical_free, "physical")
        physical_accept += accepted

        # Sample nuisance parameters conditional on the newly sampled physical parameters.
        x, current_logp, accepted = metropolis_step(x, current_logp, nuisance_free, "nuisance")
        nuisance_accept += accepted

        samples[i] = x

        if (i + 1) % cfg.save_every == 0:
            logger.info(
                "Gibbs iteration %d/%d | "
                "logp = %.5f | "
                "physical acceptance = %.3f | "
                "nuisance acceptance = %.3f",
                i + 1,
                cfg.mcmc_steps,
                current_logp,
                physical_accept / (i + 1),
                nuisance_accept / (i + 1),
            )

    for name, d in diagnostics.items():
        logger.info(
            "%s diagnostics: bounds=%d, nonfinite=%d, "
            "errors=%d, metropolis=%d, accepted=%d",
            name,
            d["bounds"],
            d["nonfinite"],
            d["errors"],
            d["metropolis"],
            d["accepted"],
        )

        if d["log_alpha"]:
            logger.info(
                "%s log_alpha: median=%.3g, min=%.3g, max=%.3g",
                name,
                np.median(d["log_alpha"]),
                np.min(d["log_alpha"]),
                np.max(d["log_alpha"]),
            )

    # Acceptance diagnostics
    physical_fraction = physical_accept / cfg.mcmc_steps
    nuisance_fraction = nuisance_accept / cfg.mcmc_steps

    logger.info("Physical acceptance fraction: %.3f", physical_fraction)
    logger.info("Nuisance acceptance fraction: %.3f", nuisance_fraction)

    try:
        tau = emcee.autocorr.integrated_time(samples, quiet=True)
        discard = int(2 * np.nanmax(tau))
        logger.info("Autocorrelation times: %s", tau)
        logger.info( "Estimated effective sample sizes: %s", np.round(cfg.mcmc_steps / tau).astype(int))
    except emcee.autocorr.AutocorrError:
        logger.warning( "Chain too short to estimate autocorrelation time.")
        discard = int(0.1 * cfg.mcmc_steps)
    except ValueError:
        logger.warning("Estimating autocorrelation time raised a ValueError.")
        discard = int(0.1 * cfg.mcmc_steps)

    if discard >= cfg.mcmc_steps:
        raise RuntimeError( "MCMC chain too short relative to autocorrelation time.")
    logger.info("Discarding %d burn-in iterations", discard)

    posterior_samples = samples[discard:]

    if len(posterior_samples) == 0:
        raise RuntimeError("No samples remaining after burn-in.")

    # Posterior summaries
    median = np.median(posterior_samples, axis=0)
    lower = np.percentile(posterior_samples, 16, axis=0)
    upper = np.percentile(posterior_samples, 84, axis=0)

    system.update_parameters(median)
    system.set_thaw(True)
    system.set_freeze(True)

    # Log physical parameter estimates.
    header = (
        f"{'Parameter Name':<35}"
        f"{'Median':>18}"
        f"{'Lower':>18}"
        f"{'Upper':>18}"
    )

    logger.info("Physical parameter estimates:\n%s", header)
    logger.info("-" * len(header))

    for i, (name, med, lo, hi) in enumerate(
        zip(
            system.parameter_names,
            median[:npar],
            lower[:npar],
            upper[:npar],
        )
    ):
        suffix = " (fixed)" if fixed_mask[i] else ""

        logger.info(
            f"{name + suffix:<35}"
            f"{med:>18.8g}"
            f"{lo:>18.8g}"
            f"{hi:>18.8g}"
        )

    # Log nuisance parameter estimates
    if nnuisance > 0:
        nuisance_names = list(cfg.nuisance_params.keys())

        logger.info("Nuisance parameter estimates:")

        for j, name in enumerate(nuisance_names):
            idx = npar + j

            logger.info(
                "%-35s %18.8g %18.8g %18.8g",
                name,
                median[idx],
                lower[idx],
                upper[idx],
            )

    # Corner plot for physical parameters.
    plot_mask = physical_free
    physical_samples = posterior_samples[:, plot_mask]
    physical_names = [name for name, include in zip(system.parameter_names, plot_mask[:npar]) if include]
    if physical_samples.shape[1] > 0:
        fig = corner.corner(
            physical_samples,
            labels=physical_names,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".4g",
            truths=median[plot_mask],
        )

        fig.savefig( os.path.join(plot_dir, "mcmc_corner.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    _plot_fit(system, plot_dir, "mcmc", cfg)
    logger.info("Gibbs MCMC complete")

    # Save the chain.
    logger.info("Saving MCMC")
    system.set_static(True)
    outfile = os.path.join(data_dir, "static_pointing_mcmc.h5")
    with h5py.File(outfile, "a") as f:
        system.save(f, "/", True)
        group = f.create_group("chain")
        group.create_dataset( "samples", data=samples, compression="gzip")
        group.create_dataset( "posterior_samples", data=posterior_samples, compression="gzip")
        group.create_dataset("fixed_mask", data=fixed_mask)
        group.attrs["ndim"] = ndim
        group.attrs["npar"] = npar
        group.attrs["nuisance_params"] = nnuisance
        group.attrs["discard"] = discard
        group.attrs["physical_acceptance"] = physical_fraction
        group.attrs["nuisance_acceptance"] = nuisance_fraction

    return samples, posterior_samples


def _get_jobdict_po(jdb):
    return {
        f"{job.tags['obs_id']}+{job.tags['wafer_slot']}+{job.tags['array']}": job
        for job in jdb.get_jobs(jclass="per_obs")
    }


def _get_jobit_po(jdb, obslist, system):
    _ = jdb
    jobit = []
    for obs in obslist:
        try:
            rx = system.search("receivers", obs["timestamp"])
            if rx is None:
                continue
            ot = rx[obs["tube_slot"]]
        except IndexError:
            continue
        for ws, fp in zip(ot.wafer_slots, ot.focal_planes):
            jobit += [(obs, ws, fp.name)]
    return jobit


def _get_jobstr_po(info):
    obs, wafer_slot, array = info
    return f"{obs['obs_id']}+{wafer_slot}+{array}"


def _get_tags_po(info):
    obs, wafer_slot, array = info
    return {
        "obs_id": obs["obs_id"],
        "ot": obs["tube_slot"],
        "timestamp": obs["timestamp"],
        "array": array,
        "wafer_slot": wafer_slot,
        "errcode": 0,
        "message": "",
    }


def _setup_paths(root_dir, project, tel, append=""):
    plot_dir = os.path.join(
        root_dir, "plots", project, tel, "solve_static_pointing" + append
    )
    data_dir = os.path.join(
        root_dir, "data", project, tel, "solve_static_pointing" + append
    )
    plot_dir = os.path.expanduser(plot_dir)
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return plot_dir, data_dir


def _compute_templates_and_copy_old(system, cfg, old_system, old_cfg, ctx, plot_dir):
    logger.info("Computing templates and copying old data if compatible")
    dbs = [
        (metadata.ManifestDb(md["db"]), os.path.dirname(md["db"]))
        for md in ctx["metadata"]
        if "wafer_info" in (md.get("name", ""), md.get("label", ""))
    ]
    if len(dbs) == 0:
        raise ValueError("wafer_info not in context!")
    db, db_dir = dbs[0]
    if db.scheme.cols != [
        ("dets:stream_id", "out", "exact", "numeric"),
        ("dataset", "out", "exact", "numeric"),
    ]:
        raise ValueError(
            "wafer_info has unexpected ManifestScheme! Please annoy one of the usual suspects to modify this code!"
        )

    if "zemax_path" in cfg.optics_config:
        system.state_meta["zemax_hash"] = hashlib.md5(
            open(cfg.optics_config["zemax_path"], "rb").read()
        ).hexdigest()

    with open(cfg.optics_config["fp_to_ot"]) as f:
        ot_locs = yaml.safe_load(f)

    check_old = old_system is not None and old_cfg is not None
    if check_old and old_cfg is not None:
        check_old = {
            key: val for key, val in cfg.optics_config.items() if key != "zemax_path"
        } == {
            key: val
            for key, val in old_cfg.optics_config.items()
            if key != "zemax_path"
        }
        if "zemax_path" in cfg.optics_config and old_system is not None:
            if "zemax_hash" not in old_system.state_meta:
                check_old = False
            else:
                check_old *= (
                    system.state_meta["zemax_hash"]
                    == old_system.state_meta["zemax_hash"]
                )
    if check_old is False:
        logger.info("Old system has incompatible optics config")
    epc_dict = {epoch.name: epoch for epoch in system.era.epochs}
    for rx in system.receivers:
        logger.info("Working on templates for %s", f"{rx.name}_{rx.epoch}")
        epoch = epc_dict[rx.epoch.split("+")[0]]
        ws_mapping = epoch._internal.data["ws_mapping"]  # ot : ws : (stream_id, array)
        old_rx = None
        if check_old and old_system is not None:
            matches = [
                orx
                for orx in old_system.receivers
                if f"{orx.name}_{orx.epoch}" == f"{rx.name}_{rx.epoch}"
            ]
            if len(matches) == 1:
                old_rx = matches[0]
        for ot in rx.optics_tubes:
            old_ot = None
            if old_rx is not None:
                matches = [
                    oot for oot in old_rx.optics_tubes if f"{oot.name}" == f"{ot.name}"
                ]
                if len(matches) == 1:
                    old_ot = matches[0]
            for i, (fp, ws) in enumerate(zip(ot.focal_planes, ot.wafer_slots)):
                fp_str = f"{rx.name}_{rx.epoch}:{ot.name}:{ws}:{fp.name}"
                old_fp = None
                if old_ot is not None:
                    matches = [
                        ofp
                        for ofp, ows in zip(old_ot.focal_planes, old_ot.wafer_slots)
                        if f"{ofp.name}" == f"{fp.name}" and ows == ws
                    ]
                    if len(matches) == 1:
                        old_fp = matches[0]
                if old_fp is not None:
                    logger.info(
                        "%s: Matching focal plane found in old data! Copying template and data.",
                        fp_str,
                    )
                    ot.focal_planes[i] = old_fp
                    continue
                logger.info("%s: computing fresh template", fp_str)
                sid, arr = ws_mapping[ot.name][ws]
                res = db.inspect({"dets:stream_id": sid})
                if len(res) == 0:
                    raise ValueError("%s not found in wafer_info!", sid)
                elif len(res) > 1:
                    raise ValueError("%s found multiple times in wafer_info?", sid)
                wafer = read_dataset(
                    os.path.join(db_dir, res[0]["filename"]), res[0]["dataset"]
                )
                idx = np.where(np.isin(np.asarray(wafer["dets:wafer.array"]), [arr]))[0]
                wafer = wafer.subset(rows=idx)
                det_ids = np.asarray(wafer["dets:det_id"])
                det_x = wafer["dets:wafer.x"]
                det_y = wafer["dets:wafer.y"]
                det_pol = wafer["dets:wafer.angle"]
                split = np.array([f"{t}_{b}_{p}" for t, b, p in zip(wafer["dets:wafer.type"], wafer["dets:wafer.bandpass"], wafer["dets:wafer.pol"])])  # type: ignore

                focal_plane_args = (
                    0,
                    cfg.tel[:3].upper(),
                    ot.name,
                    ws,
                    cfg.optics_config["ufm_to_fp"],
                    None,
                    cfg.optics_config["fp_to_ot"],
                    None,
                    cfg.optics_config.get("zemax_path"),
                    None,
                    True,
                )

                # fmt: off
                coords = optics.get_focal_plane(None, det_x, det_y, det_pol, *focal_plane_args)  # type: ignore
                centers = optics.get_focal_plane(None, np.zeros(1), np.zeros(1), np.zeros(1), *focal_plane_args)  # type: ignore
                xi, eta, gamma, x_fp, y_fp, pol_fp, x_ot, y_ot, pol_ot = coords # type: ignore
                xi_c, eta_c, gamma_c, x_fp_c, y_fp_c, pol_fp_c, x_ot_c, y_ot_c, pol_ot_c = centers # type: ignore
                # fmt: on

                for name, values, center in (
                    ("xieta", (xi, eta, gamma), (xi_c, eta_c, gamma_c)),
                    ("fp", (x_fp, y_fp, pol_fp), (x_fp_c, y_fp_c, pol_fp_c)),
                    ("ot", (x_ot, y_ot, pol_ot), (x_ot_c, y_ot_c, pol_ot_c)),
                ):
                    setattr(
                        fp,
                        f"template_{name}" if name != "xieta" else "template",
                        fpc.DetectorOffsets(
                            np.column_stack(values),
                            det_ids,
                            split,
                            np.asarray(center).ravel(),
                            name,
                        ),
                    )
        logger.info("Plotting %s", f"{rx.name}_{rx.epoch}")
        fpp.plot_receiver_templates(rx, ot_locs, plot_dir)


def _get_old(outfile, overwrite, backup):
    old_system = None
    old_cfg = None
    if not os.path.isfile(outfile):
        return old_system, old_cfg
    logger.info("Existing file found at %s", outfile)
    if backup:
        with h5py.File(outfile) as f:
            ts = f["state"].attrs["timestamp"]
        new_path = f"{outfile}.{ts}"
        logger.info("Copying old file to %s", new_path)
        if overwrite:
            shutil.move(outfile, new_path)
        else:
            shutil.copyfile(outfile, new_path)
    if overwrite:
        return old_system, old_cfg
    with h5py.File(outfile) as f:
        old_system = fpc.PointingSystem.load(f, "/")
    old_cfg, _ = load_config_namespace(old_system.state_meta["config"])
    logger.warning(
        "Running with overwrite=False and an existing file. This mean that wherever possible we will fall back on existing data. Some compatibility checks will be done but if a key piece of metadata (wafer_info, det_match, etc.) has changed you will wan't to do a fresh run!"
    )
    return old_system, old_cfg


def _get_obslist(ctx, cfg, system):
    if ctx.obsdb is None:
        raise ValueError("No obsdb!")

    # Make sure we have the metadata we want
    point_db = None
    required = ["assignment", "det_info", "wafer_info", cfg.point_meta]
    for req in required:
        dbs = [
            md
            for md in ctx["metadata"]
            if md.get("name", "") == req or md.get("label", "") == req
        ]
        if len(dbs) == 0 or len(dbs) > 1:
            raise ValueError(
                f"Found {len(dbs)} entries in Context for required metadata: {req}"
            )
        if req == cfg.point_meta:
            point_db = metadata.ManifestDb(dbs[0]["db"])
    if point_db is None:
        raise ValueError("Pointing metadata not found")

    # Get all of the obs ids in our pointing db
    cols = [c[0] for c in point_db.scheme.cols]
    if "obs:obs_id" not in cols:
        raise ValueError("Pointing metadata not indexed on obsdb!")
    obsids = np.unique(np.array([entry["obs:obs_id"] for entry in point_db.inspect()]))
    # obsids = ["obs_1782128967_lati1_111"]
    # obsids = ["obs_1777500417_lati6_111"]

    # Validate against black and whitelist
    if len(cfg.obs_whitelist) > 0:
        obsids = np.intersect1d(obsids, cfg.obs_whitelist)
    if len(cfg.obs_blacklist) > 0:
        obsids = np.setdiff1d(obsids, cfg.obs_blacklist)

    # Load obsdb rows and filter by timestamp
    obslist = [ctx.obsdb.get(obs_id) for obs_id in obsids]
    obslist = [
        obs
        for obs in obslist
        if system.search("receivers", obs["timestamp"]) is not None
    ]

    return obslist


def _load_obs(aman, obs_id, array, ctx, system, cfg, job):
    obs = ctx.obsdb.get(obs_id)
    rx = system.search("receivers", obs["timestamp"])
    pm = system.search("pointing_models", obs["timestamp"])
    fp = rx.find_fpc(obs["tube_slot"], array)
    if fp is None:
        raise ValueError("Array {array} not found in {rx.name} {rx.epoch}")
    # Cuts
    if "hits" in aman.focal_plane:
        aman.restrict("dets", aman.focal_plane.hits >= cfg.min_hits)
    if "R2" in aman.focal_plane:
        aman.restrict("dets", aman.focal_plane.R2 > cfg.min_r2)
    dfrac = aman.dets.count / len(fp.template.xyz)
    if dfrac < cfg.min_frac_dets:
        msg = f"Only {100*np.round(dfrac, 2)}% of dets uncut"
        ju.fail(job, ErrCode.MIN_DETS, msg, logger)
        return None, rx, fp
    loaded = fpc.FocalPlane.from_aman(
        fp.name,
        obs["obs_id"],
        aman,
        fp,
        pm,
        fp.autofreeze,
        fp.fake_gamma,
        cfg.weight_factor,
    )

    # Not adding to the system so that we can leave ourselves open to parallelism
    return loaded, rx, fp


def _run_per_obs(open_jobs, done_jobs, jdb, ctx, system, cfg, plot_dir, outfile):
    job = None
    njobs = len(open_jobs)
    if njobs == 0:
        logger.info("No new per-obs jobs to run!")
        return
    logger.info(
        "Starting per-obs mode with %d open jobs and %d already done",
        njobs,
        len(done_jobs),
    )
    open_jobs += [None]
    pm_pars = deepcopy(system.parameters) # Cache a copy of the pars
    system.set_pm_to_defaults(cfg)
    for i, j in enumerate(open_jobs):
        if job is not None:
            logger.debug("Writing to db")
            ju.update_jobs_retry(jdb, [job], 10, logger)
        job = None
        if j is not None:
            with jdb.session_scope() as session:
                job = session.get(jobdb.Job, j.id)
                if job is not None:
                    session.expunge(job)
        if job is None:
            continue

        job.mark_visited()
        obs_id = job.tags["obs_id"]
        ws = job.tags["wafer_slot"]
        array = job.tags["array"]
        jstr = f"{obs_id}+{ws}+{array}"
        logger.info("Loading and processing %s (%d/%d)", jstr, i + 1, njobs)

        # Load
        try:
            aman = ctx.get_meta(obs_id, dets={"wafer.array": array})
        except Exception as e:
            msg = f"Failed to load with error: {e}"
            ju.fail(job, ErrCode.LOADER, msg, logger)
            continue

        # Init the fp
        try:
            meas, rx, fp = _load_obs(aman, obs_id, array, ctx, system, cfg, job)
        except Exception as e:
            msg = f"Failed to build focal plane with error: {e}"
            ju.fail(job, ErrCode.FPINIT, msg, logger)
            continue
        if meas is None:
            continue

        # Check min dets
        dfrac = np.nansum(meas.weights) / len(meas.weights)
        if dfrac < cfg.min_frac_dets:
            msg = f"Only {100*np.round(dfrac, 2)}% of weighted dets uncut"
            ju.fail(job, ErrCode.MIN_DETS, msg, logger)
            continue

        # Lets make sure transformed is initid
        try:
            meas.freeze(True, False)
        except Exception as e:
            msg = f"Failed to compute affine with error: {e}"
            ju.fail(job, ErrCode.AFF_ERR, msg, logger)
            continue

        # Cut on scale and shear
        try:
            scale, shear, _ = meas.transform.decompose
        except Exception as e:
            msg = f"Failed to compute affine with error: {e}"
            ju.fail(job, ErrCode.AFF_ERR, msg, logger)
            continue
        if abs(1 - np.max(scale)) > cfg.max_dscale:
            msg = f"Scale too high: {str(scale)}"
            ju.fail(job, ErrCode.HIGH_SCALE, msg, logger)
            continue
        if shear > cfg.max_shear:
            msg = f"Shear too high: {np.round(shear, 2)}"
            ju.fail(job, ErrCode.HIGH_SHEAR, msg, logger)
            continue

        # Plot
        pdir = os.path.join(
            plot_dir, rx.epoch, "per-obs", str(job.tags["timestamp"])[:5]
        )
        os.makedirs(pdir, exist_ok=True)
        try:
            fpp.plot_ufm(meas, pdir)
        except Exception as e:
            msg = f"Failed to plot with error: {e}"
            ju.fail(job, ErrCode.PLOT, msg, logger)
            continue

        # Add to system
        fp.focal_planes += [meas]
        job.jstate = cast(sqy.Column[str], jobdb.JState.done)

        if i > 0 and i % cfg.save_every == 10:
            with h5py.File(outfile, "a") as f:
                system.save(f, "/", True)
    system.update_parameters(pm_pars)
    logger.info("Done with per-obs mode!")


def _check_jobs(system, jdb, done_jobs, open_jobs):
    to_open = []
    in_system = []
    for rx in system.receivers:
        for ot in rx.optics_tubes:
            for fp in ot.focal_planes:
                in_system += [f"{meas.meas_id}+{meas.name}" for meas in fp.focal_planes]
    done_ids = [f"{job.tags['obs_id']}+{job.tags['array']}" for job in done_jobs]
    open_ids = [f"{job.tags['obs_id']}+{job.tags['array']}" for job in open_jobs]
    all_ids = done_ids + open_ids

    # Check if there are any measurements without a job
    not_in_jdb = np.setdiff1d(in_system, all_ids)
    if len(not_in_jdb) > 0:
        raise ValueError(
            f"{len(not_in_jdb)} jobs in current system but not found in jdb!\n{not_in_jdb}"
        )

    # Check if there are any jobs marked done not in the system and reopen them
    not_in_system = np.setdiff1d(done_ids, in_system)
    to_open = np.array(done_jobs)[np.isin(done_ids, not_in_system)]
    if len(not_in_system) > 0:
        for job in to_open:
            job.jstate = cast(sqy.Column[str], jobdb.JState.open)
        logger.info("Opening %s old jobs", len(to_open))
        jdb.update_jobs(to_open)
        with jdb.session_scope() as session:
            for job in to_open:  # updated_jobs:
                jid = job.id
                refreshed_job = session.get(jobdb.Job, jid)
                session.expunge(refreshed_job)
                open_jobs.append(refreshed_job)
    done_jobs = [job for job, jid in zip(done_jobs, done_ids) if jid in not_in_system]

    # Anything that is an open job needs to be removed from the system
    open_ids += [f"{job.tags['obs_id']}+{job.tags['array']}" for job in to_open]
    for rx in system.receivers:
        for ot in rx.optics_tubes:
            for fp in ot.focal_planes:
                fp.focal_planes = [
                    meas for meas in fp.focal_planes if f"{meas.meas_id}+{meas.name}" not in open_ids
                ]

    return open_jobs, done_jobs


def _plot_fit(system, plot_dir, i, cfg):
    logger.info("Plotting fitted focal-plane collections for round %s", i)
    for rx in system.receivers:
        if rx.allpad:
            continue
        pdir = os.path.join(plot_dir, rx.epoch, f"fit_{i}")
        os.makedirs(pdir, exist_ok=True)
        fpp.plot_receiver(rx, pdir)
        fpp.plot_receiver_displacements(rx, [(None, None)], pdir)
        fpp.plot_receiver_nuisance_displacements(rx, system.nuisance_params, cfg.nuisanse_plot, [(None, None)], pdir)
        fpp.plot_receiver_nuisance_transform_displacements(rx, system.nuisance_params, [(None, None)], pdir)
        fpp.plot_receiver_resid_vs_enc(rx, pdir)
        fpp.plot_receiver_residuals(rx, pdir)
        for ot in rx.optics_tubes:
            if ot.allpad:
                continue
            fpp.plot_optics_tube(ot, pdir)
            for fp in ot.focal_planes:
                if fp.pad:
                    continue
                fpp.plot_ufm(fp, pdir)


def run(config_path: str, overwrite: bool, backup: bool, timestamp: str):
    # Load config
    require = (
        "optics_config",
        "calendar",
        "era",
        "context",
    )  # Not bothering with things we have defaults for
    cfg, cfg_str = load_config_namespace(config_path, default_config, None, require)
    cal = epochs.Calendar.load(cfg.calendar)
    ctx = Context(cfg.context)
    with open(cfg.calendar) as f:
        cal_str = f.read()
    with open(cfg.context) as f:
        ctx_str = f.read()

    # Figure out paths
    plot_dir, data_dir = _setup_paths(
        cfg.root_dir, cfg.project_dir, cfg.tel, cfg.append
    )
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    outfile = os.path.join(data_dir, "static_pointing.h5")

    # Setup output
    old_system, old_cfg = _get_old(outfile, overwrite, backup)
    system = fpc.PointingSystem.empty(cal.eras[cfg.era], cfg, cal_str)
    system.state_meta["timestamp"] = timestamp
    system.state_meta["config"] = cfg_str
    system.state_meta["context"] = ctx_str
    repo = git.Repo(
        os.path.abspath(os.path.dirname(__file__)), search_parent_directories=True
    )
    system.state_meta["git_sha"] = (
        f"{repo.head.object.hexsha}{'_dirty'*repo.is_dirty()}"
    )

    # Make templates and load old data if we can
    _compute_templates_and_copy_old(system, cfg, old_system, old_cfg, ctx, plot_dir)

    # Save our template
    # logger.info("Saving templates")
    # with h5py.File(outfile, "a") as f:
    #     system.save(f, "/", True)
    # fpc.make_manifestdb(
    #     system, "template", os.path.join(data_dir, "template.sqlite"), outfile
    # )

    # Mark pad epochs as pad
    logger.info("Padding epochs")
    system.set_pad(cfg.pad_epochs, False)

    # Get obs list
    logger.info("Getting obslist")
    obslist = _get_obslist(ctx, cfg, system)

    # Set up per-obs jobs
    logger.info("Setting up per-obs jobs")
    jdb_path = os.path.join(data_dir, "jobdb.sqlite")
    jdb, open_jobs = ju.setup_jobs(
        None,
        jdb_path,
        "per_obs",
        _get_jobdict_po,
        partial(_get_jobit_po, obslist=obslist, system=system),
        _get_jobstr_po,
        _get_tags_po,
        [],
        overwrite,
        False,
        None,
        0,
        False,
        logger,
    )
    done_jobs = jdb.get_jobs(jclass="per_obs", jstate="done")
    open_jobs, done_jobs = _check_jobs(system, jdb, done_jobs, open_jobs)

    # Load in all obs and populate in per-obs mode, probably want some sort of parallelism here
    _run_per_obs(open_jobs, done_jobs, jdb, ctx, system, cfg, plot_dir, outfile)

    # Pad empty if needed
    if cfg.pad_empty:
        logger.info("Padding empty")
        system.set_pad(cfg.pad_epochs, True)

    # Force pointing model consistancy
    pm_epcs = [pm.epoch for pm in system.pointing_models]
    for rx in system.receivers:
        for ot in rx.optics_tubes:
            for fp in ot.focal_planes:
                for meas in fp.focal_planes:
                    try:
                        pm_idx = pm_epcs.index(meas.pointing_model.epoch)
                    except ValueError:
                        raise ValueError(
                            f"Can't find pointing model with epoch {meas.pointing_model.epoch} (in {meas.name}+{meas.meas_id})"
                        )
                    meas.pointing_model = system.pointing_models[pm_idx]

    # Setup nuisance
    nuisance_params = fpc.NuisanceParams.from_system(system, cfg.nuisance_params, cfg.obs_time_thresh)
    system.nuisance_params = nuisance_params
    for pm in system.pointing_models:
        pm.nuisance_params = nuisance_params

    # Save our per-obs results
    # logger.info("Saving per-obs")
    # system.set_static(True)
    # with h5py.File(outfile, "a") as f:
    #     system.save(f, "/", True)
    # fpc.make_per_obs_manifestdb(
    #     system, "transformed", os.path.join(data_dir, "per_obs.sqlite"), outfile
    # )

    # Setup fit params
    params, bounds, scales = fpc.setup_fit_params(system, cfg, system.nuisance_params)
    system.update_parameters(params)
    logger.info(params[:len(system.parameters)])
    npar = len(system.parameters)
    nresid = len(system.residuals)
    nresid_full = len(system.objective_resid(system.parameters_full, nresid, cfg.cm_fac))
    jac_sparsity = None
    if len(cfg.nuisance_params) > 0 and cfg.least_squares and cfg.n_rounds > 0:
        jac_sparsity = np.zeros((len(system.parameters_full), nresid_full), dtype=int)
        jac_n = jac_sparsity[npar:].reshape((-1, len(cfg.nuisance_params), nresid_full))
        n = 0
        for rx in system.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    for meas in fp.focal_planes:
                        idx = system.nuisance_params.obs_mapping[meas.meas_id]
                        ndet = len(meas.weights)
                        jac_n[idx, :, n:(n+ndet)] = meas.wmsk.astype(int)
                        n += ndet
        jac_sparsity[npar:,nresid:] = 1
        jac_sparsity = csr_array(jac_sparsity.T, dtype=bool)
    
    system.set_thaw(True)
    system.set_freeze(True)
    logger.info("Initial chisq is %.5f", system.chisq)
    # _plot_fit(system, plot_dir, 0)
    
    # Fit for pointing model and focal planes
    for i in range(cfg.n_rounds):
        logger.info("Starting fit round %d", i + 1)
        system.set_thaw(True)
        x0 = system.parameters_full.copy()

        # Just static
        logger.info("Fitting static for round %d", i + 1)
        x0 = x0[:npar]
        scales_use = scales[:npar]
        bounds_use = bounds[:npar]
        res, params_fit = _fit_round(system, x0, bounds_use, scales_use, nresid, cfg.least_squares, None, cfg.cm_fac)
        system.update_parameters(params_fit)
        _report_fit(res, cfg, system, npar, x0, True, False)
        system.set_thaw(True)

        # Now nuisance
        if len(cfg.nuisance_params) > 0 and i+1 >= cfg.activate_nuisance:
            logger.info("Fitting nuisance for round %d", i + 1)
            x0 = system.parameters_full.copy()
            bounds_fixed = [(val, val) for val in x0[:npar]]
            bounds_use = bounds_fixed + bounds[npar:]
            scales_use = scales
            res, params_fit = _fit_round(system, x0, bounds_use, scales_use, nresid, cfg.least_squares, jac_sparsity, cfg.cm_fac, max_nfev=25, xtol=1e-6, ftol=1e-4)
            system.update_parameters(params_fit)
            _report_fit(res, cfg, system, npar, x0, False, True)
            system.set_thaw(True)

        # Now do the fp nuisance transforms
        if i+1 >= cfg.activate_nuisance_transforms:
            system.nuisance_params.compute_transforms(system)

        # Freeze and plot
        system.set_freeze(True)
        _plot_fit(system, plot_dir, "i + 1", cfg)
    
        # Cut outliers
        tot_cut = 0
        for rx in system.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    if fp.pad:
                        continue
                    all_errs = []
                    wmsks = []
                    for meas in fp.focal_planes:
                        wmsk = np.isfinite(meas.weights) * (meas.weights > 0)
                        if meas.fake_gamma:
                            err = np.linalg.norm(meas.fpc_resid[:, :2], axis=1)
                        else:
                            err = np.linalg.norm(meas.fpc_resid, axis=1)
                        err[~wmsk] = np.nan
                        all_errs.append(err)
                        wmsks.append(wmsk)
                    errs_full = np.hstack(all_errs)
                    err_cutoff = np.nanmean(errs_full) + cfg.outlier_sigma * np.nanstd(errs_full)
                    print(err_cutoff)
                    for meas, err, wmsk in zip(fp.focal_planes, all_errs, wmsks):
                        mrmsk = err > err_cutoff
                        # meas.weights[mrmsk] = 0
                        meas.wmsk[mrmsk] = False
                        meas.wmsk = meas.wmsk.copy() # reset cache
                        tot_cut += np.sum(mrmsk[wmsk])

        if tot_cut == 0 and i >= cfg.activate_nuisance:
            logger.info("None cut! Stopping optimization.")
            break
    logger.info("Done fitting!")

    # Save
    logger.info("Saving final fit")
    system.set_static(True)
    with h5py.File(outfile, "a") as f:
        system.save(f, "/", True)
    fpc.make_manifestdb(system, "transformed", os.path.join(data_dir, "focal_plane.sqlite"), outfile)

    # Run MCMC if we want
    if cfg.mcmc_steps > 0:
        _ = _run_mcmc(system, nresid, cfg, bounds, scales, data_dir, plot_dir)

def main():
    # Load arguments
    parser = ap.ArgumentParser()
    parser.add_argument("config_path", help="Location of the config file")
    parser.add_argument("--profile", "-p", action="store_true", help="Run a profiler")
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing data"
    )
    parser.add_argument(
        "--backup", "-b", action="store_true", help="Backup existing data"
    )
    args = parser.parse_args()

    profiler = None
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("Starting at %s", timestamp)
    if args.profile:
        logger.info("Running with profiler")
        profiler = pyinstrument.Profiler()
        profiler.start()

    try:
        run(args.config_path, args.overwrite, args.backup, timestamp)
    finally:
        if args.profile and profiler is not None:
            prof_out = f"solve_static_pointing_{timestamp}.html"
            profiler.stop()
            profiler.write_html(prof_out)
            logger.info("Saving profile to %s", prof_out)


if __name__ == "__main__":
    main()
