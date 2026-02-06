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
from sotodlib.site_pipeline.jobdb import JobManager, JState
from sotodlib.preprocess import _Preprocess, Pipeline, processes
import sotodlib.preprocess.preprocess_util as pp_util
from sotodlib.preprocess.preprocess_util import PreprocessErrors
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.site_pipeline.utils.obsdb import get_obslist


logger = pp_util.init_logger("preprocess")


def multilayer_preprocess_tod(obs_id: str,
                              configs_init: Union[str, dict],
                              configs_proc: Union[str, dict],
                              group: list,
                              verbosity: int = 0,
                              compress: bool = False,
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
    verbosity : str
        The log level to use.  0 = error, 1 = warn, 2 = info, 3 = debug.
    compress : bool
        Whether or not to compress the preprocessing h5 files.
    overwrite : bool
         If True, overwrite contents of temporary h5 files.

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
    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))
    dets = {gb:gg for gb, gg in zip(group_by, group)}
    aman, out_dict_init, out_dict_proc, errors = pp_util.preproc_or_load_group(
        obs_id=obs_id,
        configs_init=configs_init,
        dets=dets,
        configs_proc=configs_proc,
        logger=logger,
        overwrite=overwrite,
        save_archive=False,
        compress=compress,
    )

    return out_dict_init, out_dict_proc, errors


def _check_init_jobdb(
    jdb,
    init_db,
    init_jobs_map,
    obs_id,
    group,
    group_by,
    overwrite=False,
):
    """
    Check if a job exists in the init jobdb.  Create it if it doesn't or open
    it if overwritting or it is not in the init db.

    Arguments
    ----------
    jdb : JobDB
        JobDB instance.
    init_db : ManifestDb or None
        Init preproc database.
    init_jobs_map : dict
        Mapping from jobdb tags to existing init jobs.
    obs_id : str
        The obs_id for jobdb entry.
    group : list
        The group for the jobdb entry.
    group_by : list
        Keys defining detector grouping (e.g. ["wafer", "band"]).
    overwrite : bool, optional
        Whether to reopen jobs even if they already exist in init_db.

    Returns
    -------
    failed_job : bool
        True if a failed job should be recorded as failed.
    job : jobdb.Job or None
        New job to add to init db. None if none are to be added.
    """
    # build tags
    tags = {'obs:obs_id': obs_id}
    for gb, g in zip(group_by, group):
        tags[f'dets:{gb}'] = g

    init_job = init_jobs_map.get(frozenset(tags.items()))

    # make job if it doesn't exist
    if init_job is None:
        tags["error"] = None
        init_job = jdb.create_job(
            jclass="init",
            tags=tags,
            commit=False,
            check_existing=False,
        )
        return False, init_job

    # if job is open, don't need to do anything
    if init_job.jstate not in (JState.done, JState.failed):
        return False, None

    # check whether this job exists in init_db
    found = True
    if init_job.jstate == JState.done and init_db is not None:
        x = init_db.inspect({'obs:obs_id': obs_id})
        found = any(
            [a[f'dets:{gb}'] for gb in group_by] == group
            for a in x
        )

    # reopen job if overwriting or missing from init_db
    if overwrite or not found:
        with jdb.locked(init_job) as j:
            j.jstate = "open"
            for _t in j._tags:
                if _t.key == "error":
                    _t.value = None
        return False, None

    # otherwise mark job as failed so we don't rerun it
    if init_job.jstate == JState.failed:
        return True, None

    return False, None


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
          nproc: int = 4,
          compress: bool = False,
          run_from_jobdb: bool = False,
          raise_error: bool = False):

    init_temp_subdir = "temp"
    proc_temp_subdir = "temp_proc"

    tqdm.monitor_interval = 0

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    for fname in (configs_init['archive']['policy']['filename'],
                  configs_proc['archive']['policy']['filename']
                 ):
        os.makedirs(os.path.dirname(fname),
                    exist_ok=True)

    # jobdb
    jobdb_path = configs_proc.get("jobdb", None)
    if jobdb_path is not None:
        jdb = JobManager(sqlite_file=jobdb_path)
         # get init jobs
        new_init_jobs = []
        init_jobs = jdb.get_jobs(jclass="init")
        init_jobs_map = {
            frozenset({k: v for k, v in j.tags.items() if k != 'error'}.items()): j
            for j in init_jobs
        }
    elif run_from_jobdb:
        raise ValueError(f"Need a jobdb if using it to make a run_list.")

    errlog = os.path.join(os.path.dirname(configs_proc['archive']['index']),
                          'errlog_proc.txt')

    if jobdb_path is not None and run_from_jobdb:
        if not overwrite:
            job_list = jdb.get_jobs("proc", jstate=JState.open)
        else:
            job_list = jdb.get_jobs("proc")
        obs_list = list(set([j.tags['obs:obs_id'] for j in job_list]))
        if len(obs_list) == 0:
            logger.warning(f"No open jobs in jobdb.")
            return
    else:
        obs_list = get_obslist(context_proc, query=query, obs_id=obs_id,
                            min_ctime=min_ctime, max_ctime=max_ctime,
                            update_delay=update_delay, tags=tags,
                            planet_obs=planet_obs)

        obs_list = [obs['obs_id'] for obs in obs_list]

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

    for obs_id in obs_list:
        pp_util.cleanup_obs(obs_id, init_policy_dir, errlog, configs_init,
                            context_init, subdir=init_temp_subdir, remove=overwrite)
        pp_util.cleanup_obs(obs_id, proc_policy_dir, errlog, configs_proc,
                            context_proc, subdir=proc_temp_subdir, remove=overwrite)

    # remove datasets from final archive file not found in init db
    for config in (configs_init, configs_proc):
        pp_util.cleanup_archive(config, logger)

    run_list = []

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

    if overwrite or not os.path.exists(configs_init['archive']['index']):
        init_db = None
    else:
        init_db = core.metadata.ManifestDb(configs_init['archive']['index'])
    if overwrite or not os.path.exists(configs_proc['archive']['index']):
        proc_db = None
    else:
         proc_db = core.metadata.ManifestDb(configs_proc['archive']['index'])

    if not run_from_jobdb:
        futures = []
        futures_dict = {}
        for obs_id in obs_list:
            futures.append(executor.submit(pp_util.get_groups, obs_id, configs_proc))
            futures_dict[futures[-1]] = obs_id

        for future in tqdm(as_completed_callable(futures), total=len(futures),
                           desc="building run list from obs list"):
            obs_id = futures_dict[future]
            _, groups, _ = future.result()

            if proc_db is not None and not overwrite:
                x = proc_db.inspect({'obs:obs_id': obs_id})
                if x is not None and len(x) != 0 and len(x) != len(groups):
                    [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]
            for group in groups:
                if 'NC' not in group:
                    failed_job = False
                    if jobdb_path is not None:
                        failed_job, new_init_job = _check_init_jobdb(
                                                        jdb,
                                                        init_db,
                                                        init_jobs_map,
                                                        obs_id,
                                                        group,
                                                        group_by,
                                                        overwrite=False,
                                                    )
                        if new_init_job:
                            new_init_jobs.append(new_init_job)

                    if not failed_job or overwrite:
                        run_list.append((obs_id, group))


        # filter by jobdb status
        if jobdb_path is not None:
            jdb.commit_jobs(new_init_jobs)

            run_list = pp_util.filter_preproc_runlist_by_jobdb(
                jdb=jdb,
                jclass="proc",
                db=proc_db,
                run_list=run_list,
                group_by=group_by,
                overwrite=overwrite,
                logger=logger
            )
    else:
        for job in job_list:
            obs_id = job.tags["obs:obs_id"]
            groups = [[job.tags['dets:'+g] for g in group_by]]
            if overwrite:
                with jdb.locked(job) as j:
                    j.jstate = JState.open
                    for _t in j._tags:
                        if _t.key == "error":
                            _t.value = None
            elif proc_db is not None:
                x = proc_db.inspect({'obs:obs_id': obs_id})
                if x is not None and len(x) != 0 and len(x) != len(groups):
                    for a in x:
                        entry = [a[f'dets:{gb}'] for gb in group_by]
                        if entry in groups:
                            groups.remove(entry)
                            # if it was found in the db but is still in open
                            # jobs, then it was added by cleanup_obs and
                            # should be set to done.
                            with jdb.locked(job) as j:
                                j.jstate = JState.done
            for group in groups:
                failed_job = False
                if jobdb_path is not None:
                    failed_job, new_init_job = _check_init_jobdb(
                                                    jdb,
                                                    init_db,
                                                    init_jobs_map,
                                                    obs_id,
                                                    group,
                                                    group_by,
                                                    overwrite=False,
                                                )
                    if new_init_job:
                        new_init_jobs.append(new_init_job)

                if not failed_job or overwrite:
                    run_list.append((obs_id, group))

        jdb.commit_jobs(new_init_jobs)

    if len(run_list) == 0:
        logger.info("Nothing to run")
        return
    logger.info(f'Run list created with {len(run_list)} obsid groups')

    # ensure dbs exist up front to prevent race conditions
    db_init = pp_util.get_preprocess_db(configs_init, group_by, logger)
    db_proc = pp_util.get_preprocess_db(configs_proc, group_by, logger)

    futures = []
    futures_dict = {}
    obs_errors = {}

    for r in run_list:
        futures.append(
            executor.submit(multilayer_preprocess_tod,
                obs_id=r[0],
                configs_init=configs_init,
                configs_proc=configs_proc,
                group=r[1],
                verbosity=verbosity,
                compress=compress,
                overwrite=overwrite,
            )
        )

        futures_dict[futures[-1]] = (r[0], r[1])
        if r[0] not in obs_errors:
            obs_errors[r[0]] = []

    total = len(futures)

    batch_size_init = configs_init['archive'].get('batch_size', 1)
    batch_size_proc = configs_proc['archive'].get('batch_size', 1)

    pb_name = f"pb_{str(int(time.time()))}.txt"
    with open(pb_name, 'w') as f:
        with MultiDbBatchManager(
            [db_init, db_proc], batch_sizes=[batch_size_init, batch_size_proc], logger=logger
        ) as (db_mgr_init, db_mgr_proc):
            for future in tqdm(as_completed_callable(futures), total=total,
                                desc="multilayer_preprocess_tod", file=f,
                                miniters=max(1, total // 100)):
                obs_id, group = futures_dict[future]
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
                                        configs_init, logger, overwrite,
                                        db_manager=db_mgr_init)
                logger.info(f"Adding future result to proc db for {obs_id}: {group}")
                pp_util.cleanup_mandb(out_dict_proc, out_meta, errors,
                                    configs_proc, logger, overwrite,
                                    db_manager=db_mgr_proc)

                # update jobdb
                if jobdb_path is not None:
                    tags = {}
                    tags["obs:obs_id"] = obs_id
                    for gb, g in zip(group_by, group):
                        tags['dets:' + gb] = g
                    jobs = jdb.get_jobs(jstate=JState.open, tags=tags)

                    for job in jobs:
                        # init layer state will be JState.done if already run
                        if job.jstate == JState.open:
                            with jdb.locked(job) as j:
                                j.mark_visited()
                                if errors[0] is not None:
                                    j.jstate = JState.failed
                                    for _t in j._tags:
                                        if _t.key == "error":
                                            _t.value = errors[0]
                                else:
                                    j.jstate = JState.done
    if raise_error:
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

        if n_groups_fail > 0:
            raise RuntimeError(f"preprocess_tod ended with {n_obs_fail}/{len(obs_errors)} "
                               f"failed obsids and {n_groups_fail}/{len(run_list)} failed groups")
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
        '--compress',
        help="Compress preprocessing database.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--run-from-jobdb',
        help="If True, use open jobs in jobdb as the run_list.",
        default=False,
        action='store_true',
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
         compress: bool = False,
         nproc: int = 4,
         run_from_jobdb: bool = False,
         raise_error: bool = False):

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
              compress=compress,
              run_from_jobdb=run_from_jobdb,
              raise_error=raise_error)


if __name__ == '__main__':
    main_launcher(main, get_parser)
