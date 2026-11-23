import h5py
import numpy as np
import os
import yaml
from copy import deepcopy
from typing import Optional, Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass
from tqdm.auto import tqdm
import logging
import argparse
import time
import uuid

from sotodlib.coords import det_match, optics
from sotodlib.coords.det_match_solutions import SolutionsCfg, solve_all
from sotodlib import core
from sotodlib.core.metadata import ManifestDb
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.utils.procs_pool import get_exec_env


logger = init_logger("update_det_match", "update-det-match: ")


def get_detset_time(detset: str) -> float:
    """
    Gets timestamp associated with a detset. Will parse this from the detset
    name, assuming it is of the form <stream_id>_<time>_tune.
    """
    return float(detset.split('_')[-2])


@dataclass
class UpdateDetMatchesConfig:
    """
    Configuration for update script

    Args
    ------
    results_path: str
        Path to directory where results such as matches, manifestdbs, and
        h5 files will be stored.
    context_path: str
        Path to context file. This must contain detset and det_cal metadata.
    wafer_info_path: str
        Path to wafer info metadata file.
    site_pipeline_root: str
        Path to root of site-pipeline-configs. If ``$SITE_PIPELINE_CONFIG_DIR``
        is set in the environment, that will be used as the default.
    wafer_map_path: str
        Path to wafer-map to be used to find det-match solution files. If not specified,
        defaults to ``<site_pipeline_root>/shared/detmapping/wafer_map.yaml``.
    freq_offset_range_args: Optional[Tuple[float, float, float]]
        If this is not None, for each match, we will scan over a range of
        freq-offsets to determine the optimal offset to use. If set, must
        contain a tuple of floats, containing ([start,] stop, [step,]) that will
        be passed directly to ``np.arange``. If it is None, will just run with
        the match with freq_offset_mhz=0.
    match_pars: Optional[Dict]
        If not None, will be passed directly to ``det_match.MatchParams`` that
        is used by the det-match function.
    detset_meta_name: str
        Name of the metadata entry in the context that contains detset info.
    detcal_meta_name: str
        Name of the metadata entry in the context that contains det_cal info.
    ufms: Optional[List[str]]
        List of ufm names to run update_det_match on.  Will run on all ufms
        for which detsets exist if None.
    show_pb: bool
        Will show progress bar when scanning freq-offset.
    apply_solution_pointing: bool
        If True, pointing information computed from design-detector positions
        will be used in the ``merged`` detset of the match.
    write_relpath: bool
        If True, will use the relative path to the h5 file (relative to the db
        path) when writing to the manifestdb
    solution_type: str
        Type of solutions to use. Must be one of ['kaiwen_handmade',
        'resonator_set'].  If 'kaiwen_handmade', will use the handmade solutions
        from Kaiwen pulled from the wafer_map file in the site-pipeline-configs.
        If `resonator_set`, must also specify the ``resonator_set_dir`` to pull
        solutions from.
    resonator_set_dir: Optional[str]
        If ``solution_type`` is 'resonator_set', this must be specified and
        contain the path to the resonator-set solutions. This directory must
        have a res-set npy file for each stream_id that is expected in the
        matching, formatted like ``<resonator_set_dir>/<stream_id>.npy``, which
        contains the result from ``np.save(fname, match.merged.as_array())``.
    time_before_cache_failure: float
        Time in seconds before a failed detset will be added to the cache. This
        is to prevent new detsets still acquiring data from being added right
        away.

    Attributes
    -------------
    freq_offsets : Optional[np.ndarray]
        If not None, contains freq_offsets determined by
        ``freq_offset_range_args`` which will be scanned over.
    """
    results_path: str
    context_path: str
    wafer_info_path: str
    site_pipeline_root:str = os.environ.get('SITE_PIPELINE_CONFIG_DIR')
    wafer_map_path: Optional[str] = None
    freq_offset_range_args: Optional[tuple[float, float, float]] = (-4, 4, 0.3)
    match_pars: Optional[Dict] = None
    detset_meta_name : str = 'smurf'
    detcal_meta_name: str = 'det_cal'
    ufms: Optional[List[str]] = None
    show_pb: bool = False
    apply_solution_pointing: bool = True
    write_relpath: bool = True
    solution_type: str = 'kaiwen_handmade'
    resonator_set_dir: Optional[str] = None
    time_before_cache_failure: float = float(3600 * 24 * 7)
    start_time: int = 0
    stop_time: int = 2**32

    def __post_init__(self):
        if self.site_pipeline_root is None:
            raise ValueError("Must set site_pipeline_root, or SITE_PIPELINE_CONFIG_DIR env var")

        if self.wafer_map_path is None:
            self.wafer_map_path = os.path.join(
                self.site_pipeline_root, 'shared/detmapping/wafer_map.yaml')

        if self.freq_offset_range_args is not None:
            self.freq_offsets = np.arange(*self.freq_offset_range_args)
        else:
            self.freq_offsets = None

        if self.match_pars is None:
            self.match_pars = {}

        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results dir does not exist: {self.results_path}")

        allowed_solution_types = ['kaiwen_handmade', 'resonator_set']
        if self.solution_type not in allowed_solution_types:
            raise ValueError(
                f"Solution type ({self.solution_type}) must be a member of: {allowed_solution_types}")

        if self.solution_type == 'resonator_set':
            if self.resonator_set_dir is None:
                raise ValueError("Must specify resonator_set_dir for solution_type='resonator_set'")

        if self.ufms is not None:
            self.ufms = [u.lower() for u in self.ufms]


class Runner:
    def __init__(self, cfg: UpdateDetMatchesConfig):
        self.cfg = cfg
        self.ctx = core.Context(cfg.context_path)
        self.detset_db = None
        self.detcal_db = None
        self.wafer_info_dataset: defaultdict[str, Any] = defaultdict(lambda: None)
        with open(self.cfg.wafer_map_path, 'r') as f:
            self.wafer_map = yaml.safe_load(f)

        self.ufm_to_fp_file = os.path.join(
            cfg.site_pipeline_root, 'shared/focalplane/ufm_to_fp.yaml')

        for d in self.ctx['metadata']:
            if 'name' in d:
                entry_name = d['name']
            elif 'label' in d:
                entry_name = d['label']
            else:
                continue
            if entry_name == cfg.detset_meta_name:
                self.detset_db = core.metadata.ManifestDb(d['db'])
            elif entry_name == cfg.detcal_meta_name:
                self.detcal_db = core.metadata.ManifestDb(d['db'])

        if self.detset_db is None:
            raise Exception(
                f"Could not find detset metadata entry with name: {cfg.detset_meta_name}")
        if self.detcal_db is None:
            raise Exception(
                f"Could not find detcal metadata entry with name: {cfg.detcal_meta_name}")

        with h5py.File(cfg.wafer_info_path) as h5f:
            for ufm in list(h5f.keys()):
                self.wafer_info_dataset[ufm] = h5f[ufm][:]

        self.failed_detset_cache_path = os.path.join(cfg.results_path, 'failed_detsets.yaml')
        self.match_dir = os.path.join(cfg.results_path, 'matches')
        if not os.path.exists(self.match_dir):
            os.mkdir(self.match_dir)

    def get_remaining_detsets(self) -> List[str]:
        detsets_all = set(self.detset_db.get_entries(['dataset'])['dataset'])
        if self.cfg.ufms:
            detsets_all = set([d for d in detsets_all if '_'.join(d.split('_')[:2]) in self.cfg.ufms])
        ts = [get_detset_time(d) for d in detsets_all]
        detsets_all = set([d for t, d in zip(ts, detsets_all) if t > self.cfg.start_time and t < self.cfg.stop_time])
        failed_detsets = set(get_failed_detsets(self.failed_detset_cache_path))
        finished_detsets = set([os.path.splitext(f)[0] for f in os.listdir(self.match_dir)])
        remaining_detsets = np.sort(list(detsets_all - failed_detsets - finished_detsets))
        return remaining_detsets

def load_solution_set(runner: Runner, stream_id: str, wafer_slot=None):
    cfg = runner.cfg
    if cfg.solution_type == 'kaiwen_handmade':
        sol_file = os.path.join(
            os.path.dirname(runner.cfg.wafer_map_path),
            runner.wafer_map[stream_id]['solution']
        )
        teltype = runner.wafer_map[stream_id]['tel_type']
        if wafer_slot is None:  # Pull from detmapping cfg
            wafer_slot = runner.wafer_map[stream_id]['wafer_slot']
        fp_pars = optics.get_ufm_to_fp_pars(teltype, wafer_slot, runner.ufm_to_fp_file)
        rs = det_match.ResSet.from_solutions(sol_file, fp_pars=fp_pars, platform=teltype)
        rs.name = 'sol'
        return rs

    elif cfg.solution_type == 'resonator_set':
        sol_file = os.path.join(cfg.resonator_set_dir, f"{stream_id}.npy")
        rs_arr = np.load(sol_file)
        rs = det_match.ResSet.from_array(rs_arr)
        rs.name = 'sol'
        return rs

def add_to_failed_cache(cache_file, detset, msg, cfg: UpdateDetMatchesConfig):
    if time.time() - get_detset_time(detset) < cfg.time_before_cache_failure:
        logger.info(f"{detset} is too recent to add to failed cache")
        return

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            x = yaml.safe_load(f) or {}
    else:
        x = {}

    x[str(detset)] = str(msg)
    with open(cache_file, 'w') as f:
        yaml.dump(x, f)

def get_failed_detsets(cache_file):
    if not os.path.exists(cache_file):
        return []

    with open(cache_file, 'r') as f:
        x = yaml.safe_load(f)

    return list(x.keys())

def run_match_aman(runner: Runner, aman, detset, wafer_slot=None):
    stream_id = aman.det_info.stream_id[aman.det_info.detset == detset][0]

    rs0 = det_match.ResSet.from_aman(aman, stream_id)
    rs0.name = 'meas'

    rs1 = load_solution_set(runner, stream_id, wafer_slot=wafer_slot)
    rs1.name = 'sol'

    match_pars = det_match.MatchParams(**runner.cfg.match_pars)
    freq_offsets = runner.cfg.freq_offsets
    if freq_offsets is not None:
        costs, opt_freq = scan_for_freq_offset(
            rs0, rs1, freq_offsets, show_pb=runner.cfg.show_pb,
            match_pars=match_pars,
        )
        match_pars.freq_offset_mhz = opt_freq
    match = det_match.Match(rs0, rs1, match_pars=match_pars,
                     apply_dst_pointing=runner.cfg.apply_solution_pointing)

    return match

def _run_match_core(runner: Runner, detset: str):
    """
    Find the observation with the highest number of non-NC detectors
    for the requested detset and run detector matching.

    Returns
    -------
    success, failure_reason : tuple[bool, str or None]
        Failure reason is only set when the old code would add the detset
        to failed_detsets.yaml.  The caller owns cache updates.
    """

    fpath = os.path.join(runner.match_dir, f"{detset}.h5")

    if os.path.exists(fpath):
        logger.info(f"Match already exists for {detset}")
        return True, None

    obs_all = set(
        runner.ctx.obsdb.query("type=='obs'")["obs_id"]
    )

    cur = runner.ctx.obsfiledb.conn.execute(
        "SELECT DISTINCT obs_id FROM files WHERE detset = ?",
        (detset,),
    )

    obs_dset = {r[0] for r in cur}

    obsids_with_cal = set(
        runner.detcal_db.get_entries(["dataset"])["dataset"]
    )

    obs_ids = sorted(
        obs_all.intersection(obs_dset).intersection(obsids_with_cal),
        key=lambda s: int(s.split("_")[1]),
        reverse=True,
    )

    if not obs_ids:
        logger.error(f"Cannot find obsid for detset {detset}")
        return False, "NO_OBSID_WITH_CAL"

    logger.info(
        f"Searching {len(obs_ids)} observations for detset {detset}"
    )

    best_obs_id = None
    best_aman = None
    best_n_good = -1
    best_n_total = 0

    for obs_id in tqdm(
        obs_ids,
        desc=f"Searching {detset}",
        disable=not runner.cfg.show_pb,
    ):
        try:
            aman = runner.ctx.get_meta(
                obs_id,
                dets={"detset": detset},
                ignore_missing=True,
            )

            stream_id = aman.det_info.stream_id[0]

            bandpass = np.asarray(
                aman.det_cal.bandpass
            ).astype(str)

            good = (bandpass != "NC")

            # Some bg files give yields higher than design.  Skip those since
            # there will be many spurious entries with wrong bgs.
            bad_bg = False
            for bp in np.unique(bandpass[good]):
                design_yield = np.count_nonzero(runner.wafer_info_dataset[stream_id]['dets:wafer.bandpass'].astype(str) == bp)
                det_cal_yield = np.count_nonzero(bandpass[good] == bp)
                logger.info(f"{obs_id} {bp} yields: {det_cal_yield} / {design_yield}")
                if det_cal_yield > design_yield:
                    bad_bg = True
                    break

            if bad_bg:
                logger.warning(
                    f"{obs_id} and {detset} yield surpasses design yield; "
                    "skipping."
                )
                continue

            n_good = int(np.count_nonzero(good))
            n_total = aman.dets.count

            if n_total == 0:
                continue

            logger.debug(
                f"{obs_id} | {detset} | "
                f"{n_good}/{n_total} non-NC detectors"
            )

            if n_good > best_n_good:
                best_obs_id = obs_id
                best_aman = aman
                best_n_good = n_good
                best_n_total = n_total

                logger.info(
                    f"New best obs_id: {best_obs_id} | "
                    f"{best_n_good}/{best_n_total} non-NC detectors"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load metadata for {obs_id} "
                f"and {detset}: {e}"
            )
            continue

    if best_aman is None:
        logger.error(
            f"No usable detector calibration metadata found for {detset}"
        )
        return False, None

    logger.info(
        f"Selected obs_id {best_obs_id} for {detset}: "
        f"{best_n_good}/{best_n_total} non-NC detectors "
        f"({100 * best_n_good / best_n_total:.2f}%)"
    )

    stream_id = str(
        best_aman.det_info.stream_id[0]
    )

    wafer_slot = str(
        best_aman.det_info.wafer_slot[0]
    )

    logger.info(
        f"Running match for {detset} | "
        f"{stream_id} | {wafer_slot} | {best_obs_id}"
    )

    try:
        match = run_match_aman(
            runner,
            best_aman,
            detset,
            wafer_slot=wafer_slot,
        )

        # Publish only complete files.  This prevents the parent process
        # from indexing an HDF5 file still being written by a worker.
        pending_dir = os.path.join(runner.cfg.results_path, ".pending_matches")
        os.makedirs(pending_dir, exist_ok=True)
        pending_file = os.path.join(
            pending_dir, f"{detset}.{uuid.uuid4().hex}.h5"
        )
        try:
            match.save(pending_file)
            os.replace(pending_file, fpath)
        finally:
            if os.path.exists(pending_file):
                os.remove(pending_file)

        logger.info(f"Saved match to file: {fpath}")
    except Exception as e:
        logger.warning(
            f"run_match_aman failed for {detset}: {e}"
        )
        return False, None

    return True, None


def run_match(runner: Runner, detset: str) -> bool:
    """Run a single detset serially, retaining the original bool API."""
    success, failure_reason = _run_match_core(runner, detset)
    if failure_reason is not None:
        add_to_failed_cache(
            runner.failed_detset_cache_path, detset, failure_reason, runner.cfg
        )
    return success


# Worker processes keep their own Context / SQLite connections; never pass
# a parent Runner into a process pool.  Reuse the Runner across detsets that
# land on the same worker, rather than reloading wafer data for each task.
_WORKER_RUNNER = None
_WORKER_CONFIG_FILE = None


def _run_match_worker(config_file: str, detset: str):
    """Execute one detset without writing any shared manifests or cache."""
    global _WORKER_RUNNER, _WORKER_CONFIG_FILE

    if _WORKER_RUNNER is None or _WORKER_CONFIG_FILE != config_file:
        with open(config_file) as f:
            cfg = UpdateDetMatchesConfig(**yaml.safe_load(f))
        _WORKER_RUNNER = Runner(cfg)
        _WORKER_CONFIG_FILE = config_file

    success, reason = _run_match_core(_WORKER_RUNNER, detset)
    return detset, success, reason


def scan_for_freq_offset(rs0, rs1, freq_offsets, match_pars=None, show_pb=True):
    """
    Scans through a list of frequency offsets to find optimal match between two
    res-sets.

    Returns
    ----------
    costs : np.ndarray
        Costs of size ``len(freq_offsets)`` containing the matching cost
        at each frequency
    opt_freq : float
        Optimal offset-frequency for match.
    """
    if match_pars is None:
        match_pars = det_match.MatchParams()
    else:
        match_pars = deepcopy(match_pars)
    freq_offsets = np.array(freq_offsets)

    rs0 = deepcopy(rs0)
    rs1 = deepcopy(rs1)

    costs = np.full_like(freq_offsets, np.nan)
    for i, offset in enumerate(tqdm(freq_offsets, disable=(not show_pb))):
        match_pars.freq_offset_mhz = offset
        match = det_match.Match(rs0, rs1, match_pars=match_pars)
        costs[i] = match.matching_cost

    imin = np.argmin(costs)
    opt_freq = freq_offsets[imin]
    logger.info(f"Found freq offset: {opt_freq}")
    return costs, opt_freq


def update_manifests(runner: Runner, detset):
    det_match_idx = os.path.join(runner.cfg.results_path, 'det_match.sqlite')
    det_match_h5 = os.path.join(runner.cfg.results_path, 'det_match.h5')
    assignment_idx = os.path.join(runner.cfg.results_path, 'assignment.sqlite')
    assignment_h5 = os.path.join(runner.cfg.results_path, 'assignment.h5')

    match_file = os.path.join(runner.cfg.results_path, f'matches/{detset}.h5')
    with h5py.File(match_file, 'r') as f:
        ra = np.array(f['merged'])

    names = list(ra.dtype.names)
    names[names.index('readout_id')] = 'dets:readout_id'
    names[names.index('det_id')] = 'dets:det_id'
    ra.dtype.names = tuple(names)
    assignment = ra[['dets:readout_id', 'dets:det_id']]

    def add_to_db(arr, db_path, h5_path, detset, write_relpath=True):
        write_dataset(core.metadata.ResultSet.from_friend(arr), h5_path, detset, overwrite=True)
        if not os.path.exists(db_path):
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')
            db = core.metadata.ManifestDb(db_path, scheme=scheme)

        if write_relpath:
            h5_path = os.path.relpath(h5_path, start=os.path.dirname(db_path))

        db = core.metadata.ManifestDb(db_path)
        if detset not in db.get_entries(['dataset'])['dataset']:
            db.add_entry({
                'dets:detset': detset,
                'dataset': detset
            }, h5_path)
        else:
            logger.warning(f"Dataset {detset} already exists in db: {db_path}")

    write_relpath = runner.cfg.write_relpath
    add_to_db(ra, det_match_idx, det_match_h5, detset,
              write_relpath=write_relpath)
    add_to_db(assignment, assignment_idx, assignment_h5, detset,
              write_relpath=write_relpath)


def update_manifests_all(runner):
    det_match_idx = os.path.join(runner.cfg.results_path, 'det_match.sqlite')
    assignment_idx = os.path.join(runner.cfg.results_path, 'assignment.sqlite')
    if os.path.exists(det_match_idx):
        det_match_db = ManifestDb(det_match_idx)
        det_match_detsets = det_match_db.get_entries(['dataset'])['dataset']
    else:
        det_match_detsets = []
    if os.path.exists(assignment_idx):
        assignment_db = ManifestDb(assignment_idx)
        assignment_detsets = assignment_db.get_entries(['dataset'])['dataset']
    else:
        assignment_detsets = []

    indexed_detsets = set(det_match_detsets).intersection(set(assignment_detsets))
    completed_detsets = {
        os.path.splitext(f)[0] for f in os.listdir(runner.match_dir)
        if f.endswith(".h5") and os.path.isfile(os.path.join(runner.match_dir, f))
    }

    for ds in (completed_detsets - indexed_detsets):
        logger.info(f"Adding {ds} to manifests")
        update_manifests(runner, ds)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='path to config file')
    parser.add_argument('--run-all', action='store_true', help='run all detsets')
    parser.add_argument('--nproc', type=int, default=4,
                        help='number of worker processes for --run-all (default: 4)')
    parser.add_argument('--solutions-mode', action='store_true',
                         help='create a det_match solution set')
    return parser


def _main_detsets(config_file: str, run_all: bool, executor=None,
                  as_completed_callable=None):
    """Parent process owns shared manifests and failed-detset cache updates."""
    with open(config_file) as f:
        cfg = UpdateDetMatchesConfig(**yaml.safe_load(f))

    runner = Runner(cfg)
    remaining_detsets = [str(d) for d in runner.get_remaining_detsets()]
    logger.info(f"{len(remaining_detsets)} detsets to match")

    # Complete any indexing left from prior runs before starting workers.
    if run_all:
        update_manifests_all(runner)

    # Preserve the prior run_all=False behavior: stop at first success.
    if not run_all:
        for detset in remaining_detsets:
            success = run_match(runner, detset)
            update_manifests_all(runner)
            if success:
                break
        return

    futures = {
        executor.submit(_run_match_worker, config_file, detset): detset
        for detset in remaining_detsets
    }
    n_success = 0
    n_failed = 0
    n_manifest_failed = 0

    for future in tqdm(
        as_completed_callable(futures), total=len(futures),
        desc="Matching detsets", disable=not bool(futures),
    ):
        detset = futures[future]
        try:
            returned_detset, success, failure_reason = future.result()
            if returned_detset != detset:
                raise RuntimeError(
                    f"Worker returned detset {returned_detset} for {detset}"
                )
        except Exception:
            n_failed += 1
            logger.exception(f"Worker failed for {detset}")
            continue

        # Workers only write matches/<detset>.h5.  The parent alone edits
        # failed_detsets.yaml, det_match.h5, assignment.h5 and their SQLite DBs.
        if failure_reason is not None:
            add_to_failed_cache(
                runner.failed_detset_cache_path, detset, failure_reason, cfg
            )

        if success:
            n_success += 1
            try:
                # Index only this COMPLETED match, not other workers'
                # still-open files that may be visible in matches/.
                update_manifests(runner, detset)
            except Exception:
                n_manifest_failed += 1
                logger.exception(f"Manifest update failed for {detset}")
        else:
            n_failed += 1

    # Index any successfully written results whose manifest updates failed,
    # and catch up on pre-existing finished outputs. All tasks are complete.
    update_manifests_all(runner)
    logger.info(
        f"Detset matching finished: {n_success} succeeded, "
        f"{n_failed} failed, {n_manifest_failed} manifest update errors"
    )


def main(config_file: str, run_all: bool = False,
         solutions_mode: bool = False, nproc: int = 4):
    """Match detsets, parallelizing --run-all using sotodlib's executor."""
    if solutions_mode:
        # Unchanged: solutions mode is not parallelized here.
        with open(config_file) as f:
            cfg = SolutionsCfg(**yaml.safe_load(f))
        solve_all(cfg)
        return

    config_file = os.path.abspath(config_file)

    if not run_all:
        _main_detsets(config_file, run_all=False)
        return

    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        try:
            _main_detsets(
                config_file, run_all=True, executor=executor,
                as_completed_callable=as_completed_callable,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(
        args.config_file, run_all=args.run_all,
        solutions_mode=args.solutions_mode, nproc=args.nproc,
    )
