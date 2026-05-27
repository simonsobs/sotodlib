import numpy as np
import os
import time
import traceback
import yaml
import argparse
from typing import Optional, List, Callable

from sotodlib import core
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.stimulator.stimulator import (
    get_hk,
    calc_gain,
    calc_timeconstant,
)

_OBS_TYPES = ('gain', 'time_constant', 'gain_and_timeconstant')
_DEFAULT_R_FRAC_MIN = 0.2
_DEFAULT_R_FRAC_MAX = 0.8


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
    Returns (obs_id, result AxisManager) where result holds stm_gain and/or
    stm_timeconstant as float arrays aligned to the dets axis.
    """
    hkdb_cfg = stm_config_dict['hkdb_cfg']
    r_frac_min = stm_config_dict.get('r_frac_min', _DEFAULT_R_FRAC_MIN)
    r_frac_max = stm_config_dict.get('r_frac_max', _DEFAULT_R_FRAC_MAX)

    try:
        ctx = core.Context(context_path, metadata_list=metadata_list)
        meta = ctx.get_meta(obs_id, ignore_missing=True)

        if hasattr(meta, 'det_cal') and hasattr(meta.det_cal, 'r_frac'):
            meta.restrict(
                'dets',
                (meta.det_cal.r_frac > r_frac_min)
                & (meta.det_cal.r_frac < r_frac_max),
            )

        logger.info(f'Processing {obs_id} ({obs_type}), {meta.dets.count} dets')

        nper = int(np.ceil(meta.dets.count / n_split))
        hkdata = None
        result = None

        for i in range(n_split):
            dets = meta.dets.vals[i * nper:(i + 1) * nper]
            if len(dets) == 0:
                continue
            meta_chunk = meta.restrict('dets', dets, in_place=False)
            tod = ctx.get_obs(meta_chunk)

            if hkdata is None:
                hkdata = get_hk(hkdb_cfg, aman=tod)

            if obs_type in ('gain', 'gain_and_timeconstant'):
                if calc_gain(tod, hkdata) is False:
                    raise RuntimeError(f'Invalid stimulator HK data for {obs_id}')
            if obs_type in ('time_constant', 'gain_and_timeconstant'):
                preprocess = obs_type != 'gain_and_timeconstant'
                if calc_timeconstant(
                    tod, hkdata, preprocess=preprocess
                ) is False:
                    raise RuntimeError(f'Invalid stimulator HK data for {obs_id}')

            chunk = core.AxisManager(meta_chunk.dets)
            if obs_type in ('gain', 'gain_and_timeconstant'):
                chunk.wrap('stm_gain', tod.det_cal.stm_gain, [(0, 'dets')])
            if obs_type in ('time_constant', 'gain_and_timeconstant'):
                chunk.wrap(
                    'stm_timeconstant', tod.det_cal.stm_tau, [(0, 'dets')])

            if result is None:
                result = chunk
            else:
                result = core.AxisManager.concatenate(
                    [result, chunk], axis='dets', other_fields='first')

        assert result.dets.count == meta.dets.count
        return obs_id, result, None

    except (LoaderError, OSError) as e:
        logger.error(f'Failed to load {obs_id}: {e}')
        return obs_id, None, (type(e).__name__, str(e), traceback.format_exc())
    except Exception as e:
        logger.error(f'Failed to process {obs_id}: {e}')
        return obs_id, None, (type(e).__name__, str(e), traceback.format_exc())


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
    """
    logger = init_logger(__name__, 'make_stm_cal: ', verbosity=verbosity)
    errlog = os.path.join(output_dir, 'errlog.txt')

    if obs_type_tags is None:
        obs_type_tags = list(_OBS_TYPES)

    ctx = core.Context(context_path, metadata_list=metadata_list)

    # Collect (obs_id, obs_type) pairs to process
    obs_pairs = []
    obs_type_query = ' or '.join(f'`{tag}`=1' for tag in obs_type_tags)
    obs_rows = ctx.obsdb.query(obs_type_query, tags=obs_type_tags)
    if obs_id is not None:
        rows_by_obs_id = {}
        for row in obs_rows:
            rows_by_obs_id.setdefault(row['obs_id'], []).append(row)
        for oid in obs_id:
            for row in rows_by_obs_id.get(oid, []):
                for otype in obs_type_tags:
                    if row[otype]:
                        obs_pairs.append((oid, otype))
    else:
        for otype in obs_type_tags:
            for row in obs_rows:
                if row[otype]:
                    obs_pairs.append((row['obs_id'], otype))

    # ManifestDb: one per calibration product
    dbs = {}
    for key in ('gain', 'timeconstant'):
        db_path = os.path.join(output_dir, f'stm_{key}.sqlite')
        if os.path.exists(db_path):
            logger.info(f'Mapping {db_path}')
            dbs[key] = core.metadata.ManifestDb(db_path)
        else:
            logger.info(f'Creating {db_path}')
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('dataset')
            dbs[key] = core.metadata.ManifestDb(db_path, scheme=scheme)

    jclass = 'stm_cal'
    jdb_path = os.path.join(output_dir, 'jobdb.sqlite')
    jdb = jobdb.JobManager(sqlite_file=jdb_path)

    for oid, otype in obs_pairs:
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
            oid, result, error_info = future.result()
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

            if result is not None:
                try:
                    logger.info(f'Saving {oid}...')
                    unix = oid.split('_')[1][:4]

                    output_keys = []
                    if obs_type in ('gain', 'gain_and_timeconstant'):
                        output_keys.append('gain')
                    if obs_type in ('time_constant', 'gain_and_timeconstant'):
                        output_keys.append('timeconstant')

                    for key in output_keys:
                        field_name = 'stm_gain' if key == 'gain' else 'stm_timeconstant'
                        sub = core.AxisManager(result.dets)
                        sub.wrap(field_name, result[field_name], [(0, 'dets')])

                        h5_fn = f'stm_{key}_{unix}.h5'
                        h5_path = os.path.join(output_dir, h5_fn)
                        sub.save(h5_path, overwrite=overwrite,
                                 compression='gzip', group=oid)
                        dbs[key].add_entry(
                            {'obs:obs_id': oid, 'dataset': oid},
                            filename=h5_fn, replace=overwrite,
                        )
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
