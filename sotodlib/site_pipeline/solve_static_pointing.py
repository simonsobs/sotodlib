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

import git
import h5py
import numpy as np
import pyinstrument
import sotodlib.coords.fp_containers as fpc
import yaml
from sotodlib.coords import optics
from sotodlib.core import Context, metadata
from sotodlib.io.metadata import read_dataset
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
    "pm_ver_override": {},
    "par_groups": {},
    "root_dir": "~",
    "project_dir": "solve_static_pointing",
    "append": "",
}


def _setup_paths(root_dir, project, tel, append=""):
    plot_dir = os.path.join(root_dir, "plots", project, tel, append)
    data_dir = os.path.join(root_dir, "data", project, tel, append)
    os.makedirs(os.path.expanduser(plot_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(data_dir), exist_ok=True)

    return plot_dir, data_dir


def _compute_templates_and_copy_old(system, cfg, old_system, old_cfg, ctx):
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

    check_old = old_system is not None and old_cfg is not None
    if check_old:
        check_old = {
            key: val for key, val in cfg.optics_config if key != "zemax_path"
        } == {key: val for key, val in old_cfg.optics_config if key != "zemax_path"}
        if "zemax_path" in cfg.optics_config:
            if "zemax_hash" not in old_system.state_meta:
                check_old = False
            else:
                check_old *= (
                    system.state_meta["zemax_hash"]
                    == old_system.state_meta["zemax_hash"]
                )
    if check_old:
        logger.info("Old system has incompatible optics config")
    epc_dict = {epoch.name: epoch for epoch in system.era.epochs}
    for rx in system.receivers:
        logger.info("Working on templates for %s", f"{rx.name}_{rx.epoch}")
        epoch = epc_dict[rx.epoch.split("+")[0]]
        ws_mapping = epoch._internal.data["ws_mapping"]  # ot : ws : (stream_id, array)
        old_rx = None
        if check_old:
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
                    oot for oot in old_rx.optics_tubes if f"{oot.name}" == f"{oot.name}"
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
                        if f"{ofp.name}" == f"{ofp.name}" and ows == ws
                    ]
                    if len(matches) == 1:
                        old_ot = matches[0]
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
                    None,
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

                coords = optics.get_focal_plane(det_x, det_y, det_pol, *focal_plane_args)  # type: ignore
                centers = optics.get_focal_plane(np.zeros(1), np.zeros(1), np.zeros(1), *focal_plane_args)  # type: ignore

                xi, eta, gamma, x_fp, y_fp, pol_fp, x_ot, y_ot, pol_ot = coords
                (
                    xi_c,
                    eta_c,
                    gamma_c,
                    x_fp_c,
                    y_fp_c,
                    pol_fp_c,
                    x_ot_c,
                    y_ot_c,
                    pol_ot_c,
                ) = centers

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


def _get_old(outfile, overwrite):
    old_system = None
    old_cfg = None
    if not os.path.isfile(outfile):
        return old_system, old_cfg
    logger.info("Existing file found at %s")
    with h5py.File(outfile) as f:
        ts = f["state"].attrs["timestamp"]
    new_path = f"{outfile}.{ts}"
    logger.info("Copying old file to %s", new_path)
    if overwrite:
        shutil.move(outfile, new_path)
        return old_system, old_cfg
    shutil.copyfile(outfile, new_path)
    with h5py.File(outfile) as f:
        old_system = fpc.PointingSystem.load(f, "/")
    old_cfg, _ = load_config_namespace(old_system.state_meta["config"])
    logger.warning(
        "Running with overwrite=False and an existing file. This mean that wherever possible we will fall back on existing data. Some compatibility checks will be done but if a key piece of metadata (wafer_info, det_match, etc.) has changed you will wan't to do a fresh run!"
    )
    return old_system, old_cfg


def run(config_path: str, overwrite: bool, timestamp: str):
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

    # Figure out paths
    plot_dir, data_dir = _setup_paths(
        cfg.root_dir, cfg.project_dir, cfg.tel, cfg.append
    )
    outfile = os.path.join(data_dir, "static_pointing.h5")

    # Setup output
    old_system, old_cfg = _get_old(outfile, overwrite)
    system = fpc.PointingSystem.empty(
        cal.eras[cfg.era], cfg, yaml.dump(yaml.safe_load(cfg.calendar))
    )
    system.state_meta["timestamp"] = timestamp
    system.state_meta["config"] = cfg_str
    system.state_meta["context"] = yaml.dump(yaml.safe_load(cfg.context))
    repo = git.Repo(
        os.path.abspath(os.path.dirname(__file__)), search_parent_directories=True
    )
    system.state_meta["git_sha"] = (
        f"{repo.head.object.hexsha}{'_dirty'*repo.is_dirty()}"
    )

    # Make templates and load old data if we can
    _compute_templates_and_copy_old(system, cfg, old_system, old_cfg, ctx)

    # TODO: Mark pad epochs as pad

    # TODO: Load in all obs and populate in per-obs mode, probably want some sort of parallelism here

    # TODO: Fit for pointing model and focal planes

    # Save
    # TODO: Setup databases!
    with h5py.File(outfile, "a") as f:
        system.save(f, "/", True)


def main():
    # Load arguments
    parser = ap.ArgumentParser()
    parser.add_argument("config_path", help="Location of the config file")
    parser.add_argument("--profile", "-p", action="store_true", help="Run a profiler")
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing data"
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
        run(args.config_path, args.overwrite, timestamp)
    finally:
        if args.profile and profiler is not None:
            prof_out = f"solve_static_pointing_{timestamp}.html"
            profiler.stop()
            profiler.write_html(prof_out)
            logger.info("Saving profile to %s", prof_out)


if __name__ == "__main__":
    main()
