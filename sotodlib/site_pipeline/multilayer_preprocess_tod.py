import os
import yaml
import time
import logging
from typing import Optional, Union, Callable, List
import numpy as np
import argparse
import traceback
from typing import Optional
from sotodlib.utils.procs_pool import get_exec_env
import h5py
import copy
from tqdm import tqdm
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib import core
from sotodlib.site_pipeline.jobdb import JobManager
from sotodlib.preprocess import _Preprocess, Pipeline, processes
import sotodlib.preprocess.preprocess_util as pp_util
from sotodlib.preprocess.preprocess_util import PreprocessErrors
import sotodlib.site_pipeline.util as sp_util


logger = pp_util.init_logger("preprocess")


def multilayer_preprocess_tod(obs_id: str,
                              configs_init: Union[str, dict],
                              configs_proc: Union[str, dict],
                              group: list,
                              verbosity: int = 0,
                              overwrite: bool = False):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and group combination
    and saves the results in the ManifestDb specified in the configs.

    Arguments
    ----------
    obs_id : str or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs_init : str or dict
        Config file or loaded config dictionary for first layer database.
    configs_proc : str or dict
        Config file or loaded config dictionary for second layer database.
    group : list
        The group to be run.  For example, this might be ['ws0', 'f090']
        if ``group_by`` (specified by the subobs->use key in the preprocess
        config) is ['wafer_slot', 'wafer.bandpass'].
    overwrite : bool
         If True, overwrite contents of temporary h5 files.
    verbosity : str
        The log level to use.  0 = error, 1 = warn, 2 = info, 3 = debug.

    Returns
    -------
    out_dict_init : dict or None
        Dictionary output for init config from get_preproc_group_out_dict
        if preprocessing ran successfully for init layer or ``None`` if
        preprocessing was loaded or ``preproc_or_load_group`` failed.
    out_dict_proc : dict or None
        Dictionary output for proc config from get_preproc_group_out_dict
        if preprocessing ran successfully for proc layer or ``None`` if
        preprocessing was loaded, that layer was not run or loaded, or
        ``preproc_or_load_group`` failed.
    errors : tuple
        A tuple containing the error from PreprocessError, an error message,
        and the traceback. Each will be None if preproc_or_load_group finished
        successfully.
    """
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))
    dets = {gb:gg for gb, gg in zip(group_by, group)}
    aman, out_dict_init, out_dict_proc, errors = pp_util.preproc_or_load_group(
        obs_id=obs_id,
        configs_init=configs_init,
        dets=dets,
        configs_proc=configs_proc,
        logger=logger,
        overwrite=overwrite,
        save_archive=False
    )

    return out_dict_init, out_dict_proc, errors


def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          configs_init: str,
          configs_proc: str,
          query: str = '',
          obs_id: Optional[str] = None,
          overwrite: bool = False,
          min_ctime: Optional[int] = None,
          max_ctime: Optional[int] = None,
          update_delay: Optional[int] = None,
          tags: Optional[str] = None,
          planet_obs: bool = False,
          verbosity: Optional[int] = None,
          nproc: Optional[int] = 4,
          raise_error: Optional[bool] = False):

    init_temp_subdir = "temp"
    proc_temp_subdir = "temp_proc"

    tqdm.monitor_interval = 0

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)

    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    os.makedirs(os.path.dirname(configs_init['archive']['policy']['filename']),
                exist_ok=True)
    os.makedirs(os.path.dirname(configs_proc['archive']['policy']['filename']),
                exist_ok=True)

    # jobdb
    jobdb_path = configs_proc.get("jobdb", None)
    if jobdb_path is not None:
        jdb = JobManager(sqlite_file=jobdb_path)

    errlog = os.path.join(os.path.dirname(configs_proc['archive']['index']),
                          'errlog_proc.txt')

    obs_list = sp_util.get_obslist(context_proc, query=query, obs_id=obs_id,
                                   min_ctime=min_ctime, max_ctime=max_ctime,
                                   update_delay=update_delay, tags=tags,
                                   planet_obs=planet_obs)

    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")
        return

    # clean up lingering files from previous incomplete runs
    init_policy_dir = os.path.join(os.path.dirname(
            configs_init['archive']['policy']['filename']),
            init_temp_subdir
    )
    proc_policy_dir = os.path.join(os.path.dirname(
        configs_proc['archive']['policy']['filename']),
        proc_temp_subdir
    )

    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.cleanup_obs(obs_id, init_policy_dir, errlog, configs_init,
                            context_init, subdir=init_temp_subdir, remove=overwrite)
        pp_util.cleanup_obs(obs_id, proc_policy_dir, errlog, configs_proc,
                            context_proc, subdir=proc_temp_subdir, remove=overwrite)

    # remove datasets from final archive file not found in init db
    pp_util.cleanup_archive(configs_init, logger)
    pp_util.cleanup_archive(configs_proc, logger)

    run_list = []

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

    if overwrite or not os.path.exists(configs_proc['archive']['index']):
        db = None
    else:
        db = core.metadata.ManifestDb(configs_proc['archive']['index'])

    futures = []
    futures_dict = {}
    for obs in obs_list:
        futures.append(executor.submit(pp_util.get_groups, obs["obs_id"], configs_proc))
        futures_dict[futures[-1]] = obs

    for future in tqdm(as_completed_callable(futures), total=len(futures),
                       desc="building run list from obs list"):
        obs = futures_dict[future]
        _, groups, _ = future.result()

        if db is not None and not overwrite:
            x = db.inspect({'obs:obs_id': obs['obs_id']})
            if x is not None and len(x) != 0 and len(x) != len(groups):
                [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]

        for group in groups:
            if 'NC' not in group:
                # Did obs_id and group fail in init pipeline
                failed_jobs = None
                if jobdb_path is not None:
                    tags = {}
                    tags['obs:obs_id'] = obs['obs_id']
                    for gb, g in zip(group_by, group):
                        tags['dets:' + gb] = g
                    failed_jobs = jdb.get_jobs(jclass="init",
                                               tags=tags,
                                               jstate=["failed"])
                if not failed_jobs or overwrite:
                    run_list.append((obs, group))

    # filter by jobdb status
    if jobdb_path is not None:
        run_list, jobs = pp_util.filter_preproc_runlist_by_jobdb(
            jdb=jdb,
            jclass="proc",
            run_list=run_list,
            group_by=group_by,
            overwrite=overwrite,
            logger=logger
        )
    else:
        jobs = [None for r in run_list]

    if len(run_list) == 0:
        logger.info("Nothing to run")
        return
    logger.info(f'Run list created with {len(run_list)} obsid groups')

    # ensure dbs exist up front to prevent race conditions
    pp_util.get_preprocess_db(configs_init, group_by, logger)
    pp_util.get_preprocess_db(configs_proc, group_by, logger)

    futures = []
    futures_dict = {}
    obs_errors = {}

    for r, j in zip(run_list, jobs):
        futures.append(
            executor.submit(multilayer_preprocess_tod,
                obs_id=r[0]['obs_id'],
                configs_init=configs_init,
                configs_proc=configs_proc,
                group=r[1],
                verbosity=verbosity,
                overwrite=overwrite
            )
        )

        futures_dict[futures[-1]] = (r[0]['obs_id'], r[1], j)
        if r[0]['obs_id'] not in obs_errors:
            obs_errors[r[0]['obs_id']] = []

    total = len(futures)

    with open('progress_bar.txt', 'w') as f:
        for future in tqdm(as_completed_callable(futures), total=total,
                               desc="multilayer_preprocess_tod", file=f,
                               miniters=max(1, total // 100)):
            obs_id, group, job = futures_dict[future]
            out_meta = (obs_id, group)
            try:
                out_dict_init, out_dict_proc, errors = future.result()
                obs_errors[obs_id].append({'group': group, 'error': errors[0]})
                logger.info(f"{obs_id}: {group} extracted successfully")
            except Exception as e:
                errmsg, tb = PreprocessErrors.get_errors(e)
                logger.error(f"Executor Future Result Error for {obs_id}: {group}:\n{errmsg}\n{tb}")
                obs_errors[obs_id].append({'group': group, 'error': PreprocessErrors.ExecutorFutureError})
                out_dict_init = None
                out_dict_proc = None
                errors = (PreprocessErrors.ExecutorFutureError, errmsg, tb)

            futures.remove(future)

            # only run if first layer was run
            if out_dict_init is not None:
                logger.info(f"Adding future result to init db for {obs_id}: {group}")
                pp_util.cleanup_mandb(out_dict_init, out_meta, errors,
                                      configs_init, logger, overwrite)
            logger.info(f"Adding future result to proc db for {obs_id}: {group}")
            pp_util.cleanup_mandb(out_dict_proc, out_meta, errors,
                                  configs_proc, logger, overwrite)

            if jobdb_path is not None:
                with jdb.locked(job) as j:
                    j.mark_visited()
                    if errors[0] is not None:
                        j.jstate = "failed"
                        j.tags["error"] = errors[0]
                    else:
                        j.jstate = "done"

    n_obs_fail = 0
    n_groups_fail = 0
    for obs_id, out_meta in obs_errors.items():
        if all(entry['error'] is not None for entry in out_meta):
            n_obs_fail += 1
            n_groups_fail += len(out_meta)
        else:
            for entry in out_meta:
                if entry['error'] is not None:
                    n_groups_fail += 1

    if raise_error and (n_groups_fail > 0):
        raise RuntimeError(f"multilayer_preprocess_tod ended with {n_obs_fail}/{len(obs_errors)} "
                           f"failed obsids and {n_groups_fail}/{len(run_list)} failed groups")
    else:
        logger.info("multilayer_preprocess_tod is done")


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('configs_init', help="Preprocessing Configuration File for first layer database")
    parser.add_argument('configs_proc', help="Preprocessing Configuration File for second layer database")
    parser.add_argument(
        '--query',
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
        type=str
    )
    parser.add_argument(
        '--obs-id',
        help="obs-id of particular observation if we want to run on just one"
    )
    parser.add_argument(
        '--overwrite',
        help="If true, overwrites existing entries in the database",
        action='store_true',
    )
    parser.add_argument(
        '--min-ctime',
        help="Minimum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--max-ctime',
        help="Maximum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--update-delay',
        help="Number of days (unit is days) in the past to start observation list.",
        type=int
    )
    parser.add_argument(
        '--tags',
        help="Observation tags. Ex: --tags 'jupiter' 'setting'",
        nargs='*',
        type=str
    )
    parser.add_argument(
        '--planet-obs',
        help="If true, takes all planet tags as logical OR and adjusts related configs",
        action='store_true',
    )
    parser.add_argument(
        '--verbosity',
        help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
        default=2,
        type=int
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )
    parser.add_argument(
        '--raise-error',
        help="Raise an error upon completion if any obsids or groups fail.",
        type=bool,
        default=False
    )
    return parser


def main(configs_init: str,
         configs_proc: str,
         query: Optional[str] = None,
         obs_id: Optional[str] = None,
         overwrite: bool = False,
         min_ctime: Optional[int] = None,
         max_ctime: Optional[int] = None,
         update_delay: Optional[int] = None,
         tags: Optional[List[str]] = None,
         planet_obs: bool = False,
         verbosity: Optional[int] = None,
         nproc: Optional[int] = 4,
         raise_error: Optional[bool] = False):

    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor=executor,
              as_completed_callable=as_completed_callable,
              configs_init=configs_init,
              configs_proc=configs_proc,
              query=query,
              obs_id=obs_id,
              overwrite=overwrite,
              min_ctime=min_ctime,
              max_ctime=max_ctime,
              update_delay=update_delay,
              tags=tags,
              planet_obs=planet_obs,
              verbosity=verbosity,
              nproc=nproc,
              raise_error=raise_error)


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
