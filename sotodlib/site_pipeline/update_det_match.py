import h5py
import numpy as np
import os
import yaml
from copy import deepcopy
from typing import Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import logging
import argparse

from sotodlib.coords import det_match, optics
from sotodlib import core
from sotodlib.core.metadata import ManifestDb
from sotodlib.io.metadata import write_dataset


logger = logging.getLogger("update_det_match")
if len(logger.handlers) == 0:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class UpdateDetMatchesConfig:
    """
    Configuration for update script

    Args
    ----
    results_path: str
        Path to directory where results such as matches, manifestdbs, and
        h5 files will be stored.
    context_path: str
        Path to context file. This must contain detset and det_cal metadata.
    site_pipeline_root: str
        Path to root of site-pipeline-configs. If ``$SITE_PIPELINE_CONFIG_DIR``
        is set in the environment, that will be used as the default.
    run_freq_offset_scan: bool
        If true, this will scan through a range of freq-offsets (between src and
        dst res-sets) to find an optimal offset for each match.
    freq_offset_range_args: Optional[Tuple[float, float, float]]
        If this is not None, for each match, we will scan over a range of
        freq-offsets to determine the optimal offset to use. If set, must
        contain a tuple of floats, containing ([start,] stop, [step,]) that will
        be passed directly to ``np.arange``. If it is None, will just run with
        the match with freq_offset_mhz=0.
    detset_meta_name: str
        Name of the metadata entry in the context that contains detset info.
    detcal_meta_name: str
        Name of the metadata entry in the context that contains det_cal info.
    show_pb: bool
        Will show progress bar when scanning freq-offset.
    apply_solution_pointing: bool
        If True, pointing information computed from design-detector positions
        will be used in the ``merged`` detset of the match.
    
    Attriutes
    -------------
    freq_offsets : Optional[np.ndarray]
        If not None, contains freq_offsets determined by
        ``freq_offset_range_args`` which will be scanned over.
    """
    results_path: str
    context_path: str
    site_pipeline_root:str = os.environ.get('SITE_PIPELINE_CONFIG_DIR')
    freq_offset_range_args: Optional[tuple[float, float, float]] = (-4, 4, 0.3)
    detset_meta_name : str = 'smurf'
    detcal_meta_name: str = 'det_cal'
    show_pb: bool = False
    apply_solution_pointing: bool = True

    def __post_init__(self):
        if self.site_pipeline_root is None:
            raise ValueError("Must set site_pipeline_root, or SITE_PIPELINE_CONFIG_DIR env var")
        
        if self.freq_offset_range_args is not None:
            self.freq_offsets = np.arange(*self.freq_offset_range_args)
        else:
            self.freq_offsets = None


class Runner:
    def __init__(self, cfg: UpdateDetMatchesConfig):
        self.cfg = cfg
        self.ctx = core.Context(cfg.context_path)
        dmap_cfg_file = os.path.join(
            cfg.site_pipeline_root, 'shared/detmapping/wafer_map.yaml'
        )
        with open(dmap_cfg_file, 'r') as f:
            self.wafer_map = yaml.safe_load(f)
        
        self.detset_db = None
        self.detcal_db = None

        self.detmap_cfg_path = os.path.join(
            cfg.site_pipeline_root, 'shared/detmapping/wafer_map.yaml')
        with open(self.detmap_cfg_path, 'r') as f:
            self.detmap_cfg = yaml.safe_load(f)
        
        self.ufm_to_fp_file = os.path.join(
            cfg.site_pipeline_root, 'shared/focalplane/ufm_to_fp.yaml')

        for d in self.ctx['metadata']:
            if d['name'] == cfg.detset_meta_name:
                self.detset_db = core.metadata.ManifestDb(d['db'])
            elif d['name'] == cfg.detcal_meta_name:
                self.detcal_db = core.metadata.ManifestDb(d['db'])
        if self.detset_db is None:
            raise Exception(
                f"Could not find detset metadtata entry with name: {cfg.detset_meta_name}")
        if self.detcal_db is None:
            raise Exception(
                f"Could not find detset metadtata entry with name: {cfg.detcal_meta_name}")
        
        self.failed_detset_cache_path = os.path.join(cfg.results_path, 'failed_detsets.yaml')
        self.match_dir = os.path.join(cfg.results_path, 'matches')
        
    def run_next_match(self):
        detsets_all = set(self.detset_db.get_entries(['dataset'])['dataset'])
        failed_detsets = set(get_failed_detsets(self.failed_detset_cache_path))
        finished_detsets = set([os.path.splitext(f)[0] for f in os.listdir(self.match_dir)])
        remaining_detsets = list(detsets_all - failed_detsets - finished_detsets)
        if len(remaining_detsets) == 0:
            return False
        logger.info(f"Number of detsets remaining: {len(remaining_detsets)}")
        run_match(self, remaining_detsets[0])
        return True

def add_to_failed_cache(cache_file, detset, msg):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            x = yaml.safe_load(f)
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
    sol_file = os.path.join(
        os.path.dirname(runner.detmap_cfg_path), 
        runner.detmap_cfg[stream_id]['solution']
    )

    rs0 = det_match.ResSet.from_aman(aman, stream_id)
    rs0.name = 'meas'

    teltype = runner.detmap_cfg[stream_id]['tel_type']
    if wafer_slot is None:  # Pull from detmapping cfg
        wafer_slot = runner.detmap_cfg[stream_id]['wafer_slot']
    fp_pars = optics.get_ufm_to_fp_pars(teltype, wafer_slot, runner.ufm_to_fp_file)
    rs1 = det_match.ResSet.from_solutions(sol_file, fp_pars=fp_pars, platform=teltype)
    rs1.name = 'sol'

    match_pars = det_match.MatchParams()
    freq_offsets = runner.cfg.freq_offsets
    if freq_offsets is not None:
        costs, opt_freq = scan_for_freq_offset(rs0, rs1, freq_offsets, show_pb=runner.cfg.show_pb)
        match_pars.freq_offset_mhz = opt_freq
    match = det_match.Match(rs0, rs1, match_pars=match_pars, 
                     apply_dst_pointing=runner.cfg.apply_solution_pointing)
    return match

def run_match(runner: Runner, detset: str):
    """
    Creates match files for specified detset, along with any other unmatched
    detsets in the loaded observation. If match fails for a known reason, this
    will add it to the failed_detset_cache so that it is not re-attempted.
    """
    # Find obs-id with cal info
    obs_all = set(runner.ctx.obsdb.query("type=='obs'")['obs_id'])
    cur = runner.ctx.obsfiledb.conn.execute(f"""
        select distinct obs_id from files where detset=='{detset}'
    """)
    obsids_with_cal = set(runner.detcal_db.get_entries(['dataset'])['dataset'])
    obs_dset = {r[0] for r in cur}
    obs_ids = sorted(list(
        obs_all.intersection(obs_dset).intersection(obsids_with_cal)), 
        key=lambda s:s.split('_')[1])[::-1]
    if len(obs_ids) == 0:
        add_to_failed_cache(
            runner.failed_detset_cache_path, detset, "NO_OBSID_WITH_CAL"
        )
        logger.error(f"Cannot find obsid for detset {detset}")
        return None

    obs_id = obs_ids[0]

    book_dir = os.path.dirname(runner.ctx.obsfiledb.get_files(obs_id)[detset][0][0])
    book_idx_file = os.path.join(book_dir, 'M_index.yaml')
    with open(book_idx_file, 'r') as f:
        book_idx = yaml.safe_load(f)

    aman = runner.ctx.get_meta(obs_id)
    finished_detsets = set([os.path.splitext(f)[0] for f in os.listdir(runner.match_dir)])
    new_detsets = []
    for detset in np.unique(aman.det_info.detset):
        if detset not in finished_detsets:
            new_detsets.append(detset)

    logger.info(f"Loaded obs_id {obs_id}. Running matches for detsets:")
    for ds in new_detsets:
        logger.info(f"    - {ds}")

    for ds in new_detsets:

        stream_id = aman.det_info.stream_id[aman.det_info.detset == ds][0]
        # Try to get wafer slot info from book idx
        if 'wafer_slots' in book_idx:
            for ws in book_idx['wafer_slots']:  
                if ws['stream_id'] == stream_id:
                    wafer_slot = ws['wafer_slot']
                    break
            else:
                logger.error(
                    f"Could not find wafer_slot from book index for ds={detset}, "
                    f"obs_id={obs_id}"
                )
                raise Exception("Could not find wafer-slot")
        else:
            wafer_slot = None

        match = run_match_aman(runner, aman, ds, wafer_slot=wafer_slot)
        fpath = os.path.join(runner.match_dir, f"{ds}.h5")
        match.save(fpath)
        logger.info(f"Saved match to file: {fpath}")

    return aman


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
    idx = names.index('readout_id')
    names[idx] = 'dets:readout_id'
    ra.dtype.names = tuple(names)
    assignment = ra[['dets:readout_id', 'det_id']]

    def add_to_db(arr, db_path, h5_path, detset):
        write_dataset(core.metadata.ResultSet.from_friend(arr), h5_path, detset, overwrite=True)
        if not os.path.exists(db_path):
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')
            db = core.metadata.ManifestDb(db_path, scheme=scheme)

        db = core.metadata.ManifestDb(db_path)
        if detset not in db.get_entries(['dataset'])['dataset']:
            db.add_entry({
                'dets:detset': detset,
                'dataset': detset
            }, h5_path)
        else:
            logger.warning(f"Dataset {detset} already exists in db: {db_path}")
    
    add_to_db(ra, det_match_idx, det_match_h5, detset)
    add_to_db(assignment, assignment_idx, assignment_h5, detset)


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
    completed_detsets = set([os.path.splitext(f)[0] for f in os.listdir(runner.match_dir)])

    for ds in (completed_detsets - indexed_detsets):
        logger.info(f"Adding {ds} to manifests")
        update_manifests(runner, ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--all', action='store_true', help='run all detsets')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = UpdateDetMatchesConfig(**yaml.safe_load(f))

    runner = Runner(cfg)

    if args.all:
        update_manifests_all(runner)
        while runner.run_next_match():
            update_manifests_all(runner)
    else:
        runner.run_next_match()
        update_manifests_all(runner)