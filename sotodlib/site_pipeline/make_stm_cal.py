import numpy as np
import os
import time
import traceback
import yaml
import argparse
from typing import Optional, List, Callable
from collections import defaultdict

from sotodlib import core
from sotodlib.io.hkdb import HkConfig
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.stimulator.stimulator import (
    get_hk,
    calc_gain,
    calc_timeconstant,
    preprocessing,
)

_OBS_TYPES = ('gain', 'time_constant', 'gain_and_timeconstant')
_DB_TYPES = ('gain', 'time_constant', 'readout_delay', 'gain_with_tau_correction')
_PRODUCTS = {
    'gain': ['gain'],
    'time_constant': ['time_constant', 'readout_delay'],
    'gain_and_timeconstant': ['gain', 'time_constant',
                              'readout_delay', 'gain_with_tau_correction'],
}

def run(
    logger,
    context_path,
    stm_config_dict,
    metadata_list,
    obs_id,
    obs_type,
    n_split,
):
    """Process one stimulator observation.

    Dispatches to calc_gain / calc_timeconstant based on obs_type.
    Returns (obs_id, stm_cal AxisManager, error info)
    where stm_cal AxisManager holds output of the stimulator calibration analysis.
    """
    hkdb_cfg = HkConfig.from_yaml(stm_config_dict['hkdb_cfg'])

    try:
        ctx = core.Context(context_path, metadata_list=metadata_list)
        meta = ctx.get_meta(obs_id, ignore_missing=True)

        logger.info(f'Processing {obs_id} ({obs_type}), {meta.dets.count} dets')

        nper = int(np.ceil(meta.dets.count / n_split))
        hkdata = None
        stm_cal = None

        for i in range(n_split):
            dets = meta.dets.vals[i * nper:(i + 1) * nper]
            if len(dets) == 0:
                continue
            meta_chunk = meta.restrict('dets', dets, in_place=False)
            tod = ctx.get_obs(meta_chunk)

            if hkdata is None:
                hkdata = get_hk(hkdb_cfg, aman=tod)

            valid_gain, valid_timeconstant = preprocessing(tod, hkdata)

            if valid_gain:
                calc_gain(tod)

            if valid_timeconstant:
                calc_timeconstant(tod, idxs=None)

            if stm_cal is None:
                stm_cal = tod.stm_cal
            else:
                stm_cal = core.AxisManager.concatenate(
                    [stm_cal, tod.stm_cal], axis='dets', other_fields='first')

        assert stm_cal.dets.count == meta.dets.count
        return obs_id, stm_cal, None

    except (LoaderError, OSError) as e:
        logger.error(f'Failed to load {obs_id}: {e}')
        return obs_id, None, (type(e).__name__, str(e), traceback.format_exc())
    except Exception as e:
        logger.error(f'Failed to process {obs_id}: {e}')
        return obs_id,  None, (type(e).__name__, str(e), traceback.format_exc())


def _get_stm_h5_path(output_dir, obs_id):
    oid_spl = obs_id.split('_')
    unix = oid_spl[1][:5]
    tube_slot = oid_spl[2]

    return os.path.join(output_dir, f'stm_cal_{tube_slot}_{unix}.h5')


def _open_stm_manifest_dbs(output_dir, logger):
    dbs = {}
    for key in _DB_TYPES:
        db_path = os.path.join(output_dir, f'stm_{key}.sqlite')
        if os.path.exists(db_path):
            logger.info(f'Mapping {db_path}')
            dbs[key] = core.metadata.ManifestDb(db_path)
        else:
            logger.info(f'Creating {db_path}')
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_exact_match('dets:detset')
            scheme.add_data_field('dataset')
            dbs[key] = core.metadata.ManifestDb(db_path, scheme=scheme)
    return dbs


def _publish_self(dbs, output_dir, obs_id, obs_type, stm_cal, detset, overwrite):
    h5_path = _get_stm_h5_path(output_dir, obs_id)

    for product in _PRODUCTS[obs_type]:
        if product == 'gain' and 'stm_gain' not in stm_cal._fields:
            continue
        if product == 'time_constant' and 'stm_tau' not in stm_cal._fields:
            continue
        if product == 'readout_delay' and 'readout_delay' not in stm_cal._fields:
            continue
        if product == 'gain_with_tau_correction' and \
            'stm_gain_with_tau_correction' not in stm_cal._fields:
            continue

        dbs[product].add_entry(
            {'obs:obs_id': obs_id,
             'dets:detset': detset,
             'dataset': obs_id},
            filename=h5_path,
            replace=overwrite,
        )


def _detsets_by_obsid(obsfiledb, obsids):
    return {
        obs_id: set(obsfiledb.get_detsets(obs_id))
        for obs_id in obsids
    }


def _build_stm_cal_index(ctx, stm_rows, product):
    stm_detsets = _detsets_by_obsid(
        ctx.obsfiledb,
        [row['obs_id'] for row in stm_rows],
    )

    cal_index = defaultdict(lambda: {
        'times': [],
        'obs_ids': [],
    })

    for row in stm_rows:
        if product not in _PRODUCTS[_tag_resolver(row)]:
            continue

        stm_obs_id = row['obs_id']
        start_time = row['start_time']

        for detset in stm_detsets[stm_obs_id]:
            cal_index[detset]['times'].append(start_time)
            cal_index[detset]['obs_ids'].append(stm_obs_id)

    # List -> np.array for faster search later
    out = {}
    for detset, items in cal_index.items():
        out[detset] = {
            'times': np.asarray(items['times'], dtype=float),
            'obs_ids': items['obs_ids']
        }

    return out


def _find_latest_stm_cal(cal_index, detset, obs_start_time, max_days_before):
    if detset not in cal_index:
        return None

    times = cal_index[detset]['times']
    obs_ids = cal_index[detset]['obs_ids']

    idx = np.searchsorted(times, obs_start_time, side='right') - 1
    if idx < 0:
        return None

    max_age = 3600 * 24 * max_days_before
    if obs_start_time - times[idx] > max_age:
        return None

    return obs_ids[idx]


def _tag_resolver(row):
    if row['gain_and_timeconstant']:
        return 'gain_and_timeconstant'
    if row['gain'] and row['time_constant']:
        return 'gain_and_timeconstant'
    if row['gain']:
        return 'gain'
    if row['time_constant']:
        return 'time_constant'
    else:
        raise ValueError(f'Row {row} has no valid obs_type tag.')


def _publish_obs_relation(db, obs_rows, obs_detsets, cal_index, max_days_before, output_dir):
    for row in obs_rows:
        obs_id = row['obs_id']
        obs_start_time = row['start_time']

        for detset in obs_detsets.get(obs_id, []):
            query = {'obs:obs_id': obs_id, 'dets:detset': detset}
            existing = db.inspect(query, strict=False)
            # Remove any existing entries
            for entry in existing:
                db.remove_entry(entry['_id'], commit=False)

            stm_obs_id = _find_latest_stm_cal(
                cal_index,
                detset,
                obs_start_time,
                max_days_before,
            )

            if stm_obs_id is not None:
                db.add_entry(
                    {
                        'obs:obs_id': obs_id,
                        'dets:detset': detset,
                        'dataset': stm_obs_id,
                    },
                    filename=_get_stm_h5_path(output_dir, stm_obs_id),
                    replace=True,
                    commit=False,
                )

        db.conn.commit()


def load_stimulator_cal(db: core.metadata.ManifestDb):
    """Return processed stimulator calibration entries from a ManifestDb.
    """
    available = {}

    for entry in db.inspect({}):
        obs_id = entry['obs:obs_id']
        dataset = entry['dataset']

        if obs_id != dataset:
            continue

        available[dataset] = entry

    return available


def _main(
    executor,
    as_completed_callable: Callable,
    context_path: str,
    stm_config: dict,
    output_dir: str,
    metadata_list: Optional[List[str]] = 'all',
    verbosity: Optional[int] = 2,
    overwrite: Optional[bool] = False,
    obs_type_tags: Optional[List[str]] = None,
    obs_id: Optional[List[str]] = None,
    n_split: Optional[int] = 1,
    nprocs: Optional[int] = 1,
    max_retry: Optional[int] = 3,
    stale: Optional[float] = 60.,
    max_days_before: Optional[float] = 1.0,
    update_obs_corresp: Optional[bool] = True,
):
    """Main function for making stimulator calibration metadata.

    Arguments
    ---------
    context_path : str
        Path to context file.
    stm_config : dict
        Stimulator configuration dict (loaded from stm_config yaml).
    output_dir : str
        Directory for HDF5 files and ManifestDb sqlite files.
    metadata_list : str or list of str (default 'all')
        Metadata labels to load when building context.
    verbosity : int (default 2)
        0: Error, 1: Warning, 2: Info, 3: Debug
    overwrite : bool (default False)
        If True, reprocess obs already in the database.
    obs_type_tags : list of str (default all three types)
        Subset of ('gain', 'time_constant', 'gain_and_timeconstant') to process.
    obs_id : list of str (default None)
        Explicit list of obs-ids to process; if None all matching obs are used.
    n_split : int (default 1)
        Number of detector chunks per observation (for memory management).
    nprocs : int (default 1)
        Number of parallel worker processes.
    max_retry : int (default 3)
        Maximum attempts before marking a job as failed.
    stale : float (default 60.)
        Jobs locked longer than this many seconds are unlocked before starting.
    max_days_before : float (default 1.0)
        Maximum age of stimulator calibration to use for a given observation.
    update_obs_corresp : bool (default True)
        If True, update the ManifestDb entries for the correspondence between
        stimulator calibration and observations.
    """
    logger = init_logger(__name__, 'make_stm_cal: ', verbosity=verbosity)
    errlog = os.path.join(output_dir, 'errlog.txt')

    if obs_type_tags is None:
        obs_type_tags = list(_OBS_TYPES)

    ctx = core.Context(context_path, metadata_list=metadata_list)

    # Collect (obs_id, obs_type) pairs to process
    stm_pairs = []
    obs_type_query = ' or '.join(f'`{tag}`=1' for tag in obs_type_tags)
    stm_rows = ctx.obsdb.query(obs_type_query, tags=list(_OBS_TYPES),
                               sort=['start_time'])

    if obs_id is not None:
        rows_by_obs_id = {
            row['obs_id']: row
            for row in stm_rows
        }

        for oid in obs_id:
            row = rows_by_obs_id.get(oid, None)
            if row is None:
                logger.warning(f'obs_id {oid} not found in obsdb, skipping.')
                continue

            stm_pairs.append((oid, _tag_resolver(row)))
    else:
        for row in stm_rows:
            stm_pairs.append((row['obs_id'], _tag_resolver(row)))

    # ManifestDb: one per calibration product
    dbs = _open_stm_manifest_dbs(output_dir, logger)

    jclass = 'stm_cal'
    jdb_path = os.path.join(output_dir, 'jobdb.sqlite')
    jdb = jobdb.JobManager(sqlite_file=jdb_path)

    for oid, otype in stm_pairs:
        if len(jdb.get_jobs(jclass=jclass, tags={'obs_id': oid, 'obs_type': otype})) == 0:
            jdb.create_job(jclass, tags={'obs_id': oid, 'obs_type': otype})

    for j in jdb.get_jobs(jclass=jclass):
        if j.lock and (time.time() - j.lock) > stale:
            logger.info(f'Clearing old lock on {j}')
            jdb.unlock(j.id)

    jstate = 'all' if overwrite else 'open'
    to_do = jdb.get_jobs(jclass=jclass, jstate=jstate, locked=False)

    futures = []
    with jdb.locked(to_do, count=len(to_do)) as jobs:
        for job in jobs:
            job.mark_visited()
            futures.append(executor.submit(
                run,
                logger,
                context_path,
                stm_config,
                metadata_list,
                obs_id=job.tags['obs_id'],
                obs_type=job.tags['obs_type'],
                n_split=n_split,
            ))

        for future in as_completed_callable(futures):
            oid, stm_cal, error_info = future.result()
            detsets = ctx.obsfiledb.get_detsets(oid)
            for job in jobs:
                if job.tags['obs_id'] == oid:
                    break
            else:
                # This should not happen but just in case.
                logger.error(f'No job found for obs_id={oid}, skipping.')
                continue

            obs_type = job.tags['obs_type']

            if error_info is not None:
                with open(errlog, 'a') as f:
                    f.write(f"{time.time()}, {oid}, {obs_type}, {error_info[0]}\n")
                    f.write("\t" + error_info[1] + "\n")
                    f.write("\t" + error_info[2] + "\n")

            if stm_cal is not None:
                try:
                    logger.info(f'Saving {oid}...')
                    h5_path = _get_stm_h5_path(output_dir, oid)
                    stm_cal.save(h5_path, overwrite=overwrite,
                                 compression='gzip', group=oid)
                    for detset in detsets:
                        _publish_self(dbs, output_dir, oid, obs_type, stm_cal, detset, overwrite)
                    job.jstate = 'done'
                    continue
                except Exception as e:
                    logger.error(f'Failed to save {oid}: {e}')
                    with open(errlog, 'a') as f:
                        f.write(f"{time.time()}, {oid}, {obs_type}, {type(e).__name__}\n")
                        f.write("\t" + str(e) + "\n")
                        f.write("\t" + traceback.format_exc() + "\n")

            if job.visit_count > max_retry:
                logger.error(f'Mark {oid} as failed.')
                job.jstate = 'failed'
            else:
                logger.error(f'Failed {oid}, try again later')

        if update_obs_corresp:
            obs_rows = ctx.obsdb.query(
                "type=='obs'",
                sort=['start_time'],
            )[::-1]
            obs_detsets = _detsets_by_obsid(
                ctx.obsfiledb,
                [row['obs_id'] for row in obs_rows],
            )
            query_stm_all = ' or '.join(f'`{tag}`=1' for tag in _OBS_TYPES)
            stm_all = ctx.obsdb.query(query_stm_all, tags=list(_OBS_TYPES),
                                sort=['start_time'])

            for product in _DB_TYPES:
                stm_cal_mandb = load_stimulator_cal(dbs[product])
                stm_rows_available = [
                    row for row in stm_all
                    if row['obs_id'] in stm_cal_mandb
                ]
                cal_index = _build_stm_cal_index(ctx, stm_rows_available, product)
                _publish_obs_relation(dbs[product], obs_rows, obs_detsets,
                                    cal_index, max_days_before, output_dir)


def main(pipeline_config, stm_config):
    with open(pipeline_config, 'r') as f:
        pp_cfg = yaml.safe_load(f)
    with open(stm_config, 'r') as f:
        stm_cfg = yaml.safe_load(f)

    rank, executor, as_completed_callable = \
        get_exec_env(nprocs=pp_cfg['nprocs'])
    if rank == 0:
        _main(
            executor=executor,
            as_completed_callable=as_completed_callable,
            stm_config=stm_cfg,
            **pp_cfg,
        )


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config',
                        help='Path to pipeline configuration yaml file.')
    parser.add_argument('stm_config',
                        help='Path to stimulator configuration yaml file.')
    return parser


if __name__ == '__main__':
    main_launcher(main, get_parser)
