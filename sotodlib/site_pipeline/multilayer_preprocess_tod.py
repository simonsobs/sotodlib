import os
import yaml
import time
import logging
from typing import Optional, Union, Callable
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


def multilayer_preprocess_tod(obs_id,
                              configs_init,
                              configs_proc,
                              group,
                              verbosity=0,
                              overwrite=False,
                              run_parallel=False):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and group combination
    and saves the results in the ManifestDb specified in the configs.

    Arguments
    ----------
    obs_id: string or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs_init: string or dictionary
        config file or loaded config directory for existing database
    configs_proc: string or dictionary
        config file or loaded config directory for processing database
    group: list
        Group to run if you only want to run a partial update
    overwrite: bool
        if True, overwrite existing entries in ManifestDb
    verbosity: log level
        0 = error, 1 = warn, 2 = info, 3 = debug
    run_parallel: Bool
        If true preprocess_tod is called in a parallel process which returns
        dB info and errors and does no sqlite writing inside the function.
    """
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    if configs_init.get("lmsi_config", None) is not None or configs_proc.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))
    dets = {gb:gg for gb, gg in zip(group_by, group)}
    aman, out_dict_init, out_dict_proc, error = pp_util.preproc_or_load_group(obs_id=obs_id,
                                                                              configs_init=configs_init,
                                                                              dets=dets,
                                                                              configs_proc=configs_proc,
                                                                              logger=logger,
                                                                              overwrite=overwrite,
                                                                              save_archive=not run_parallel)
    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi
        for configs in [configs_init, configs_proc]:
            new_plots = os.path.join(configs["plot_dir"],
                                     f'{str(aman.timestamps[0])[:5]}',
                                     aman.obs_info.obs_id)

            if os.path.exists(new_plots):
                lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                          Path(configs["lmsi_config"]),
                          Path(os.path.join(new_plots, 'index.html')))

    return out_dict_init, out_dict_proc, error


def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          configs_init: str,
          configs_proc: str,
          query: Optional[str] = None,
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

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)

    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    os.makedirs(os.path.dirname(configs_proc['archive']['policy']['filename']),
                exist_ok=True)

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
    init_policy_dir = os.path.join(os.path.dirname(configs_init['archive']['policy']['filename']),
                                  init_temp_subdir)
    proc_policy_dir = os.path.join(os.path.dirname(configs_proc['archive']['policy']['filename']),
                                   proc_temp_subdir)

    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.cleanup_obs(obs_id, init_policy_dir, errlog, configs_init,
                            context_init, subdir=init_temp_subdir, remove=overwrite)
        pp_util.cleanup_obs(obs_id, proc_policy_dir, errlog, configs_proc,
                            context_proc, subdir=proc_temp_subdir, remove=overwrite)

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
                run_list.append((obs, group))

    if jobdb_path is not None:
        run_list_skipped = []
        jobs = []
        for r in run_list:
            jclass = r[0]['obs_id']
            for gb, g in zip(group_by, r[1]):
                if gb == 'detset':
                    jclass += "_" + g
                else:
                    jclass += "_" + gb + "_" + str(g)
            if jdb.get_jobs(jclass=jclass, jstate=["done", "failed", "ignored"]):
                run_list_skipped.append(r)
            else:
                open_jobs = jdb.get_jobs(jclass=jclass, jstate=["open"])
                if open_jobs:
                    job = open_jobs[0]
                else:
                    job = jdb.create_job(jclass)

                jobs.append(job)

        logger.info(f"skipping {len(run_list_skipped)} jobs from jobdb")

        run_list = [r for r in run_list if r not in run_list_skipped]
    else:
        jobs = [None for r in run_list]
    logger.info(f'Run list created with {len(run_list)} obsid groups')

    futures = []
    futures_dict = {}
    obs_errors = {}

    run_parallel = True #nproc > 1

    # Run write_block obs-ids in parallel at once then write all to the sqlite db.
    for r, j in zip(run_list, jobs):
        futures.append(executor.submit(multilayer_preprocess_tod,
                                       obs_id=r[0]['obs_id'],
                                       configs_init=configs_init,
                                       configs_proc=configs_proc,
                                       group=r[1],
                                       verbosity=verbosity,
                                       overwrite=overwrite,
                                       run_parallel=run_parallel))

        futures_dict[futures[-1]] = (r[0]['obs_id'], r[1], j)
        if r[0]['obs_id'] not in obs_errors:
            obs_errors[r[0]['obs_id']] = []

    for future in tqdm(as_completed_callable(futures), total=len(futures),
                           desc="multilayer_preprocess_tod"):
        obs_id, group, job = futures_dict[future]
        out_meta = (obs_id, group)
        try:
            out_dict_init, out_dict_proc, error = future.result()
            obs_errors[obs_id].append({'group': group, 'error': error})
            logger.info(f"{obs_id}: {group} extracted successfully")
        except Exception as e:
            errmsg, tb = PreprocessErrors.get_errors(e)
            logger.error(f"Executor Future Result Error for {obs_id}: {group}:\n{errmsg}\n{tb}")
            obs_errors[obs_id].append({'group': group, 'error': PreprocessErrors.ExecutorFutureError})
            out_dict = None
            error = PreprocessErrors.ExecutorFutureError

            if jobdb_path is not None:
                with jdb.locked(job) as j:
                    j.mark_visited()
                    j.jstate = "failed"

        futures.remove(future)

        if run_parallel:
            logger.info(f"Adding future result to init db for {obs_id}: {group}")
            pp_util.cleanup_mandb(out_dict_init, out_meta, error,
                                  configs_init, logger)
            logger.info(f"Adding future result to proc db for {obs_id}: {group}")
            pp_util.cleanup_mandb(out_dict_proc, out_meta, error,
                                  configs_proc, logger)

        if jobdb_path is not None:
            with jdb.locked(job) as j:
                if j.visit_count == 0:
                    j.mark_visited()
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
         tags: Optional[str] = None,
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
