import argparse
import multiprocessing
import os
import json
import sqlite3
from contextlib import closing
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED,
)
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm

from sotodlib import core
from sotodlib.io import hkdb
from sotodlib.io.ancil import utils
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher


MISSING = 5500
SITE_FIELDS = ["pwv", "ambient_temp", "uv", "wind_spd", "wind_dir"]
PLAT_FIELDS = ["hwp_rate1", "hwp_rate2", "hwp_direction"]
DATA_FIELDS = (
    "pwv", "dpwv", "ambient_temp", "uv", "wind_spd", "wind_dir",
    "hwp_rate", "pwv_start", "pwv_end", "dhwp_rate", "hwp_direction",
    "az_acc", "az_vel",
)


def cli_error(e, field, logger):
    tb = "".join(traceback.format_tb(e.__traceback__))
    logger.error(f"ERROR getting {field} data:\n{type(e)}: {e}\n{tb}")


def append_error(errlog, label, message):
    with open(errlog, "a") as f:
        f.write(f"\n{time.time()}, {label}\n{message}\n")


def load_numpy_dict(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            return {key: data[key] for key in data.files}
    return data.item()


def get_t_mask(t0, t1, t):
    t = np.asarray(t)
    return (t > t0) & (t < t1)


def finite_or_missing(value):
    return float(value) if np.isfinite(value) else MISSING


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def get_weather(t0, t1, weather_data, factor=1):
    timestamps, values = weather_data
    m = get_t_mask(t0, t1, timestamps)
    values = finite_values(np.asarray(values)[m])
    return float(np.median(values) * factor) if values.size else MISSING


def get_pwv(t0, t1, pwv_data):
    timestamps = pwv_data.get("timestamps", None)
    if timestamps is None:
        raise KeyError("PWV data must contain timestamps")
    m = get_t_mask(t0, t1, timestamps)
    values = np.asarray(pwv_data["pwv"], dtype=float)[m]
    valid = finite_values(values)
    if not valid.size:
        return (MISSING,) * 4
    return (
        float(np.median(valid)),
        finite_or_missing(values[0]),
        finite_or_missing(values[-1]),
        float(np.ptp(valid)),
    )


def get_hwp_spin(t0, t1, platform_data):
    rates = []
    for key, (timestamps, values) in platform_data.items():
        if "hwp-bbb" in key:
            m = get_t_mask(t0, t1, timestamps)
            rates.append(finite_values(np.asarray(values)[m]))
    if not rates:
        return MISSING, MISSING
    rates = np.concatenate(rates)
    if not rates.size:
        return MISSING, MISSING
    return float(np.median(rates)), float(np.ptp(rates))


def get_hwp_dir(t0, t1, platform_data):
    timestamps, values = platform_data["hwp-pid.hwppid.direction"]
    m = get_t_mask(t0, t1, timestamps)
    values = finite_values(np.asarray(values)[m])
    if not values.size:
        return MISSING
    return 2 * int(np.median(values)) - 1


def get_scan_params():
    """Placeholder for future scan velocity and acceleration calculations."""
    return MISSING, MISSING


def get_man_db(cfg):
    path = cfg["output_path"]
    if os.path.exists(path):
        return core.metadata.ManifestDb(path)
    scheme = core.metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    for field in DATA_FIELDS:
        scheme.add_data_field(field)
    return core.metadata.ManifestDb(path, scheme=scheme)


def open_frame_offset_db(path):
    """Open the SQLite cache, creating the table if necessary."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frame_offsets (
                    obs_id TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source IN ("site", "plat")),
                    offsets_json TEXT NOT NULL,
                    PRIMARY KEY (obs_id, source)
                )
            """)
    except Exception:
        conn.close()
        raise
    return conn


def load_frame_offsets(conn, obs_ids):
    """Load cached offsets only for the requested observations."""
    offsets = {"site": {}, "plat": {}}
    obs_ids = list(dict.fromkeys(obs_ids))
    # Batch IN queries to stay below the SQLite bind parameter limit.
    for i in range(0, len(obs_ids), 500):
        batch = obs_ids[i:i + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT obs_id, source, offsets_json FROM frame_offsets "
            f"WHERE obs_id IN ({placeholders})", batch
        )
        for oid, source, payload in rows:
            offsets[source][oid] = json.loads(payload)
    return offsets


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize frame offset value: {type(value)}")


def save_frame_offsets(conn, rows):
    """Commit newly gathered offsets in a single transaction."""
    if not rows:
        return
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO frame_offsets "
            "(obs_id, source, offsets_json) VALUES (?, ?, ?)", rows
        )


# Each offset worker owns its own HKDB engines and connection pools.
_OFFSET_WORKER = {}


def init_offset_worker(cfg, sources):
    global _OFFSET_WORKER
    configs = {
        key: hkdb.HkConfig.from_yaml(cfg[f"{key}_hkdb_cfg"])
        for key in sources
    }
    _OFFSET_WORKER = {
        key: (config, hkdb.HkDb(config))
        for key, config in configs.items()
    }


def get_obs_frame_offsets(oid, start, stop, sources):
    """Query only missing sources; return successful entries and errors."""
    fields = {"site": SITE_FIELDS, "plat": PLAT_FIELDS}
    rows, errors = [], []
    for key in sources:
        try:
            config, db = _OFFSET_WORKER[key]
            result = hkdb.get_frame_offsets(
                config, start, stop, fields[key], db,
            )
            if not isinstance(result, dict):
                raise TypeError("Expected a mapping of HK file paths to offsets")
            payload = json.dumps(
                result, default=_json_default, separators=(",", ":"),
                allow_nan=False,
            )
            rows.append((oid, key, payload))
        except Exception:
            errors.append((key, traceback.format_exc()))
    return rows, errors


def gather_frame_offsets(cfg, obs_list, logger, errlog):
    """Query HK databases in parallel; only the parent writes the cache."""
    path = cfg["frame_offset_path"]
    save_every = int(cfg.get("frame_offset_save_every", 1000))
    nproc = int(cfg.get("frame_offset_nproc", 4))
    if save_every < 1 or nproc < 1:
        raise ValueError("frame_offset_save_every and frame_offset_nproc must be positive")

    with closing(open_frame_offset_db(path)) as conn:
        offsets = load_frame_offsets(conn, [o["obs_id"] for o in obs_list])
        missing = []
        for obs in obs_list:
            oid = obs["obs_id"]
            sources = tuple(key for key in ("site", "plat") if oid not in offsets[key])
            if sources:
                missing.append((oid, obs["start_time"], obs["stop_time"], sources))
        logger.info(f"Frame offsets: {len(obs_list) - len(missing)} cached, "
                    f"{len(missing)} to gather")
        if not missing:
            return offsets["site"], offsets["plat"]

        nproc = min(nproc, len(missing))
        sources = tuple(key for key in ("site", "plat")
                        if any(key in task[3] for task in missing))
        logger.info(f"Gathering offsets with {nproc} workers, "
                    f"committing every {save_every} completed observations")
        pending = []
        tasks = iter(missing)
        futures = {}
        completed = 0
        start_time = time.perf_counter()
        write_time = 0.0

        def submit_next(exe):
            task = next(tasks, None)
            if task is not None:
                futures[exe.submit(get_obs_frame_offsets, *task)] = task[0]

        def flush():
            nonlocal write_time
            if pending:
                t0 = time.perf_counter()
                save_frame_offsets(conn, pending)
                elapsed = time.perf_counter() - t0
                write_time += elapsed
                logger.info(f"Committed {len(pending)} offset entries in {elapsed:.3f} s")
                pending.clear()

        with ProcessPoolExecutor(
            max_workers=nproc,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_offset_worker,
            initargs=(cfg, sources),
        ) as exe:
            try:
                for _ in range(min(len(missing), 4 * nproc)):
                    submit_next(exe)
                with tqdm(total=len(missing), desc="Gathering frame offsets") as pb:
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            oid = futures.pop(future)
                            rows, errors = future.result()
                            for row_oid, key, payload in rows:
                                offsets[key][row_oid] = json.loads(payload)
                                pending.append((row_oid, key, payload))
                            for key, message in errors:
                                logger.error(f"ERROR getting {key} frame offsets for {oid}:\n{message}")
                                append_error(errlog, f"{key} frame offsets for {oid}", message)
                            completed += 1
                            if completed % save_every == 0:
                                flush()
                            pb.update(1)
                            submit_next(exe)
            finally:
                # Save results already collected, even on a worker failure.
                for future in futures:
                    future.cancel()
                flush()
        elapsed = time.perf_counter() - start_time
        logger.info(f"Frame-offset gathering: {elapsed:.2f} s total; "
                    f"{write_time:.2f} s in SQLite writes (including commits)")

    return offsets["site"], offsets["plat"]


_WORKER = {}


def init_worker(cfg):
    global _WORKER
    _WORKER = {
        "site_cfg": hkdb.HkConfig.from_yaml(cfg["site_hkdb_cfg"]),
        "plat_cfg": hkdb.HkConfig.from_yaml(cfg["plat_hkdb_cfg"]),
        "pwv_data": load_numpy_dict(cfg["pwv_file"]),
    }


def get_split_metadata(obs, frame_offsets_site, frame_offsets_plat):
    logger = init_logger("split_info")
    oid = obs["obs_id"]
    t0, t1 = obs["start_time"], obs["stop_time"]
    try:
        site_spec = hkdb.LoadSpec(
            start=t0, end=t1, cfg=_WORKER["site_cfg"],
            fields=SITE_FIELDS, frame_offsets=frame_offsets_site,
        )
        plat_spec = hkdb.LoadSpec(
            start=t0, end=t1, cfg=_WORKER["plat_cfg"],
            fields=PLAT_FIELDS, frame_offsets=frame_offsets_plat,
        )
        weather_data = hkdb.load_hk(site_spec, show_pb=False).data
        platform_data = hkdb.load_hk(plat_spec, show_pb=False).data

        entry = {"obs:obs_id": oid}
        entry.update({field: MISSING for field in DATA_FIELDS})

        try:
            pwv, pwv_start, pwv_end, dpwv = get_pwv(t0, t1, _WORKER["pwv_data"])
            entry.update(pwv=pwv, pwv_start=pwv_start, pwv_end=pwv_end, dpwv=dpwv)
        except Exception as e:
            cli_error(e, "pwv", logger)

        weather_fields = {
            "ambient_temp": ("env-vantage.weather_data.temp_outside", 1),
            "uv": ("env-vantage.weather_data.UV", 0.1),
            "wind_spd": ("env-vantage.weather_data.wind_speed", 1.609),
            "wind_dir": ("env-vantage.weather_data.wind_dir", 1),
        }
        for field, (key, factor) in weather_fields.items():
            try:
                entry[field] = get_weather(t0, t1, weather_data[key], factor)
            except Exception as e:
                cli_error(e, field, logger)

        try:
            entry["hwp_rate"], entry["dhwp_rate"] = get_hwp_spin(t0, t1, platform_data)
        except Exception as e:
            cli_error(e, "hwp spin", logger)

        try:
            entry["hwp_direction"] = get_hwp_dir(t0, t1, platform_data)
        except Exception as e:
            cli_error(e, "hwp direction", logger)

        entry["az_vel"], entry["az_acc"] = get_scan_params()

        return None, utils.denumpy(entry)
    except Exception:
        # Do not mark observations with failed HK loads as completed.
        message = traceback.format_exc()
        logger.error(f"ERROR processing {oid}:\n{message}")
        return "Fail", message


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config File")
    parser.add_argument("--query", type=str, help="Query to pass to obsdb")
    parser.add_argument("--nproc", type=int, default=4)
    return parser


def main(config: str, query: Optional[str] = None, nproc: Optional[int] = 4):
    logger = init_logger("split_info", verbosity=3)
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(os.path.dirname(os.path.abspath(cfg["output_path"])), exist_ok=True)
    errlog = os.path.join(os.path.dirname(cfg["output_path"]), "errlog.txt")
    ctx = core.Context(cfg["context_file"])
    man_db = get_man_db(cfg)
    completed = {entry["obs:obs_id"] for entry in man_db.inspect()}
    obslist = [dict(obs) for obs in ctx.obsdb.query(query)
               if obs["obs_id"] not in completed]
    logger.info(f"{len(obslist)} observations to process")
    if not obslist:
        return

    frame_offsets_site, frame_offsets_plat = gather_frame_offsets(
        cfg, obslist, logger, errlog
    )
    ready = [obs for obs in obslist
             if obs["obs_id"] in frame_offsets_site
             and obs["obs_id"] in frame_offsets_plat]
    if len(ready) != len(obslist):
        logger.warning(f"Skipping {len(obslist) - len(ready)} observations "
                       "with missing frame offsets; they can be retried on the next run")
    if not ready:
        return

    logger.info("Launching multiproc pool...")
    with ProcessPoolExecutor(
        max_workers=nproc,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(cfg,),
    ) as exe:
        futures = {
            exe.submit(get_split_metadata, obs,
                       frame_offsets_site[obs["obs_id"]],
                       frame_offsets_plat[obs["obs_id"]]): obs["obs_id"]
            for obs in ready
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing observations"):
            oid = futures.pop(future)
            try:
                err, entry = future.result()
                if err:
                    append_error(errlog, f"processing error for {oid}", entry)
                    continue
                man_db.add_entry(entry)
            except Exception:
                message = traceback.format_exc()
                logger.error(f"ERROR processing {oid}:\n{message}")
                append_error(errlog, f"processing or DB error for {oid}", message)


if __name__ == "__main__":
    main_launcher(main, get_parser)
