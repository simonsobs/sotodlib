import numpy as np
import os
import time
import yaml
import argparse
from typing import Optional, List, Callable

from sotodlib import core
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.hwp.hwp_angle_model import apply_hwp_angle_model
import sotodlib.hwp.hwp as hwp
from sotodlib.tod_ops import filters, fourier_filter
from sotodlib.site_pipeline.calibration.wiregrid import wrap_wg_hk, \
    find_operation_range, calc_calibration_data_set, fit_with_circle, \
    get_cal_gamma, load_data, wg_config


def run(
    logger,
    context_path,
    wiregrid_config,
    metadata_list,
    obs_id,
    n_split,
):
    """
    Function to run process pipeline for wire grid calibration.
    This split detectors into `n_split` chunks and process serially.
    Returns obs_id and ResultSet.
    """
    wg_cfg = wg_config(**wiregrid_config)

    try:
        ctx = core.Context(context_path, metadata_list=metadata_list)

        meta = ctx.get_meta(obs_id)
        nper = int(np.ceil(meta.dets.count/n_split))
        logger.info(f'process {obs_id}, dets {meta.dets.count}')
        for i in range(n_split):
            dets = meta.dets.vals[i*nper:(i+1)*nper]
            meta_short = meta.restrict('dets', dets, in_place=False)

            tod = ctx.get_obs(meta_short)

            if i == 0:
                raw_data_dict_wg = load_data(wg_cfg, tod.timestamps[0],
                                             tod.timestamps[-1])

            # ---- Preprocess start ---- #
            apply_hwp_angle_model(tod)
            iir_filt = filters.iir_filter(tod, invert=True)
            tod.signal = fourier_filter(tod, iir_filt)
            hwp.demod_tod(tod, signal='signal')
            # ---- Preprocess end ---- #

            # ---- Wire grid calibration start ---- #

            wrap_wg_hk(tod, raw_data_dict=raw_data_dict_wg)
            idx_steps_starts, idx_steps_ends = find_operation_range(tod)
            calc_calibration_data_set(tod, idx_steps_starts, idx_steps_ends)
            fit_with_circle(tod)
            get_cal_gamma(tod)
            # ---- Wire grid calibration end ---- #

            if i == 0:
                result = tod.wg.copy()
            else:
                result = core.AxisManager.concatenate(
                    [result, tod.wg], axis='dets', other_fields='first')

        assert result.dets.count == meta.dets.count
        return obs_id, result
    except (LoaderError, OSError) as e:
        logger.error(f'Failed to load data {obs_id} {e}')
        return obs_id, None
    except Exception as e:
        logger.error(f'Failed to process {obs_id} {e}')
        return obs_id, None


def _main(
    executor,
    as_completed_callable: Callable,
    context_path: str,
    wiregrid_config: str,
    output_dir: str,
    metadata_list: Optional[List[str]] = 'all',
    verbosity: Optional[int] = 2,
    overwrite: Optional[bool] = False,
    tags: Optional[List[str]] = None,
    obs_id: Optional[List[str]] = None,
    n_split: Optional[int] = 10,
    nprocs: Optional[int] = 1,
    max_retry: Optional[int] = 3,
    stale: Optional[float] = 60.
):
    """
    Main function for making wg_cal metadata

    Arguments
    ---------
    context_path: str
        Path to context file.
    process_pipe: str
        Dictionary of process pipeline config
    output_dir: str
        Path to the output directory
    metadata_list: str or list of str (default 'all')
        List of metadata labels to load
    verbosity: int (default 2)
        0: Error, 1: Warning, 2: Info, 3: Debug
    overwrite: bool (default False)
        If true, overwrites existing entries in the database
    tags: List of str (default None)
        List of tag to use for quering observations
    obs_id: List of str (default None)
        List of obs-ids of particular observations that you want to run.
    n_split: int (default 10)
        Number of splits for the serial processing of a single observation.
    nprocs: int (default 1)
        Number of processes to use. 1 at site computing.
    max_retry: int (default 3)
        Maximum number to retry job.
    stale: float (default 60.)
        Jobs locked for longer than this time will be treated as stale
        and unlocked.
    """

    logger = init_logger(
        __name__, 'make_wg_cal: ', verbosity=verbosity)

    ctx = core.Context(context_path, metadata_list=metadata_list)
    obs_ids = []
    for tag in tags:
        obslist = ctx.obsdb.query("subtype = 'cal'", tags=[tag])
        for obs in obslist:
            obs_ids.append(obs['obs_id'])

    db_path = os.path.join(output_dir, 'wg_cal.sqlite')
    if os.path.exists(db_path):
        logger.info(f"Mapping {db_path} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(db_path)
    else:
        logger.info(f"Creating {db_path} for the "
                    "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            db_path,
            scheme=scheme
        )

    jclass = 'wg_cal'
    jdb_path = os.path.join(output_dir, 'jobdb.sqlite')
    jdb = jobdb.JobManager(sqlite_file=jdb_path)

    for obs_id in obs_ids:
        jobs = jdb.get_jobs(jclass=jclass, tags={'obs_id': obs_id})
        if len(jobs) == 0:
            jdb.create_job(jclass, tags={'obs_id': obs_id})

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
                wiregrid_config,
                metadata_list,
                obs_id=job.tags['obs_id'],
                n_split=n_split,
            ))
        for future in futures:
            obs_id, result = future.result()
            for job in jobs:
                if job.tags['obs_id'] == obs_id:
                    break
            if result is not None:
                try:
                    logger.info(f'saving {obs_id}...')
                    unix = obs_id.split('_')[1][:4]  # first 4 digits
                    h5_fn = f'wg_cal_{unix}.h5'
                    h5_path = os.path.join(output_dir, h5_fn)
                    result.save(h5_path, overwrite=overwrite,
                                compression='gzip', group=obs_id)
                    db.add_entry(
                        {'obs:obs_id': obs_id, 'dataset': obs_id},
                        filename=h5_fn, replace=overwrite,
                    )
                    job.jstate = 'done'
                    continue
                except Exception as e:
                    logger.error(f'Failed to save {obs_id} {e}')
            if job.visit_count > max_retry:
                logger.error(f'Mark {obs_id} as failed.')
                job.jstate = 'failed'
            else:
                logger.error(f'Failed {obs_id}, try again later')


def main(pipeline_config, wiregrid_config):
    with open(pipeline_config, "r") as f:
        pp_cfg = yaml.safe_load(f)
    with open(wiregrid_config, "r") as f:
        wg_cfg = yaml.safe_load(f)

    rank, executor, as_completed_callable = \
        get_exec_env(nprocs=pp_cfg['nprocs'])
    if rank == 0:
        _main(
            executor=executor,
            as_completed_callable=as_completed_callable,
            wiregrid_config=wg_cfg,
            **pp_cfg
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config',
                        help="Path to configuration yaml file.")
    parser.add_argument('wiregrid_config',
                        help="Path to wire grid configuration yaml file.")
    return parser


if __name__ == '__main__':
    main_launcher(main, get_parser)
