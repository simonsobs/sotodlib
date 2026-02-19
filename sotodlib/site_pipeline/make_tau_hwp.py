import numpy as np
import os
import time
import yaml
import argparse
from typing import Optional, List, Callable

from sotodlib import core, preprocess
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.core.metadata.loader import LoaderError


dtype = [
    ('dets:readout_id', '<U40'),
    ('tau_hwp', 'f4'),
    ('tau_hwp_error', 'f4'),
    ('AQ', 'f4'),
    ('AQ_error', 'f4'),
    ('AU', 'f4'),
    ('AU_error', 'f4'),
    ('redchi2s', 'f4'),
]


def run(
    logger,
    context_path,
    process_pipe,
    metadata_list,
    obs_id,
    n_split,
):
    """
    Function to run process pipeline for tau_hwp.
    This split detectors into `n_split` chunks and process serially.
    Returns obs_id and ResultSet.
    """
    res_arr = np.array([], dtype=dtype)
    rset = core.metadata.ResultSet.from_friend(res_arr)

    try:
        ctx = core.Context(context_path, metadata_list=metadata_list)
        meta = ctx.get_meta(obs_id)
        nper = int(np.ceil(meta.dets.count/n_split))
        logger.info(f'process {obs_id}, dets {meta.dets.count}')
        for i in range(n_split):
            dets = meta.dets.vals[i*nper:(i+1)*nper]
            meta_short = meta.restrict('dets', dets, in_place=False)
            tod = ctx.get_obs(meta_short)
            pipe = preprocess.Pipeline(process_pipe)
            proc = core.AxisManager(tod.dets, tod.samps)
            for process in pipe:
                logger.info(f'{process.name}, dets {tod.dets.count}')
                process.process(tod, proc)
                process.calc_and_save(tod, proc)
            for i in range(tod.dets.count):
                dic = {'dets:readout_id': tod.dets.vals[i]}
                dic.update({v[0]: tod.tau_hwp[v[0]][i] for v in dtype[1:]})
                rset.append(dic)
        assert len(rset) == meta.dets.count
        return obs_id, rset
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
    process_pipe: dict,
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
    Main function for making tau_hwp metadata

    Arguments
    ---------
    context_path: str
        Path to context file.
    process_pipe: str
        Dictionary of process pipeline config
    output_dir: str
        Path to the output directory
    metadata_list: str or list of str
        List of metadata labels to load
    verbosity: int
        0: Error, 1: Warning, 2: Info, 3: Debug
    overwrite: bool
        If true, overwrites existing entries in the database
    tags: List of str
        List of tag to use for quering observations
    obs_id: List of str
        List of obs-ids of particular observations that you want to run.
    n_split:
        Number of splits for the serial processing of a single observation.
    nprocs: int
        Number of processes to use. 1 at site computing.
    max_retry: int
        Maximum number to retry job.
    stale: float
        Jobs locked for longer than this time will be treated as stale
        and unlocked.
    """

    logger = init_logger(
        __name__, 'make_tau_hwp: ', verbosity=verbosity)

    ctx = core.Context(context_path, metadata_list=metadata_list)
    obs_ids = []
    for tag in tags:
        obslist = ctx.obsdb.query("subtype = 'cal'", tags=[tag])
        for obs in obslist:
            obs_ids.append(obs['obs_id'])

    db_path = os.path.join(output_dir, 'tau_hwp.sqlite')
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

    jclass = 'tau_hwp'
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
                process_pipe,
                metadata_list,
                obs_id=job.tags['obs_id'],
                n_split=n_split,
            ))
        for future in futures:
            obs_id, rset = future.result()
            for job in jobs:
                if job.tags['obs_id'] == obs_id:
                    break
            if rset is not None:
                try:
                    logger.info(f'saving {obs_id}...')
                    unix = obs_id.split('_')[1][:4]  # first 4 digits
                    h5_fn = f'tau_hwp_{unix}.h5'
                    h5_path = os.path.join(output_dir, h5_fn)
                    write_dataset(
                        data=rset,
                        filename=h5_path,
                        address=obs_id,
                        overwrite=True
                    )
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


def main(config):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    rank, executor, as_completed_callable = get_exec_env(nprocs=cfg['nprocs'])
    if rank == 0:
        _main(
            executor=executor,
            as_completed_callable=as_completed_callable,
            **cfg
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to configuration yaml file.")
    return parser


if __name__ == '__main__':
    main_launcher(main, get_parser)
