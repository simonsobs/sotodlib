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
from sotodlib.core.metadata.manifest import DbBatchManager
from sotodlib.site_pipeline.jobdb import JobManager, JState
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.site_pipeline.utils.obsdb import get_obslist
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess.preprocess_util import PreprocessErrors
from sotodlib.preprocess import _Preprocess, Pipeline, processes


logger = pp_util.init_logger("preprocess")


def load_preprocess_tod_sim(obs_id,
                            sim_map,
                            configs="preprocess_configs.yaml",
                            context=None,
                            dets=None,
                            meta=None,
                            modulated=True,
                            logger=logger):
    """Loads the saved information from the preprocessing pipeline and runs the
    processing section of the pipeline on simulated data

    Assumes preprocess_tod has already been run on the requested observation.

    Arguments
    ----------
    obs_id : multiple
        passed to ``context.get_obs`` to load AxisManager, see Notes for
        `context.get_obs`
    sim_map : pixell.enmap.ndmap
        signal map containing (T, Q, U) fields
    configs : string or dictionary
        config file or loaded config directory
    dets : dict
        dets to restrict on from info in det_info. See context.get_meta.
    meta : AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    modulated : bool
        If True, apply the HWP angle model and scan the simulation
        into a modulated signal.
        If False, scan the simulation into demodulated timestreams.

    Returns
    -------
    aman : core.AxisManager
        Axis manager after running through the preprocessing steps.  Returns
        ``None`` if all detectors are cut.
    """
    configs, context = pp_util.get_preprocess_context(configs, context)
    if dets is not None:
        meta.restrict("dets", dets)
    det_vals = pp_util.load_preprocess_det_select(
        obs_id, configs=configs, context=context, logger=logger
    )
    meta.restrict("dets", [d for d in meta.dets.vals if d in det_vals])

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta, no_signal=True)
        if modulated:
            # Apply the HWP angle model here
            # WARNING : should be turned off in the config file
            # to filter simulations
            aman = hwp_angle_model.apply_hwp_angle_model(aman)
            aman.move("signal", None)
        demod_mm.from_map(aman, sim_map, wrap=True, modulated=modulated)
        pipe.run(aman, aman.preprocess, sim=True)
        return aman


def preprocess_tod(configs: Union[str, dict],
                   obs_id: str,
                   group: dict,
                   verbosity: int = 0,
                   compress: bool = False,
                   overwrite: bool = False,
                  ):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and group combination
    and saves the results in the ManifestDb specified in the configs.

    Arguments
    ----------
    configs : str or dict
         Config file or loaded config dictionary.
    obs_id : str or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs.
    group : list
        The group to be run.  For example, this might be ['ws0', 'f090']
        if ``group_by`` (specified by the subobs->use key in the preprocess
        config) is ['wafer_slot', 'wafer.bandpass'].
    verbosity : str
        Log level. 0 = error, 1 = warn, 2 = info, 3 = debug.
    compress : bool
        Whether or not to compress the preprocessing h5 files.
    overwrite : bool
        If True, overwrite contents of temporary h5 files.

    Returns
    -------
    out_dict : dict or None
        Dictionary output for init config from get_preproc_group_out_dict
        if preprocessing ran successfully for init layer or ``None`` if
        preprocessing was loaded or ``preproc_or_load_group`` failed.
    errors : tuple
        A tuple containing the error from PreprocessError, an error message,
        and the traceback. Each will be None if preproc_or_load_group finished
        successfully.
    """
    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    dets = {gb:gg for gb, gg in zip(group_by, group)}
    aman, out_dict, _, errors = pp_util.preproc_or_load_group(
        obs_id=obs_id,
        configs_init=configs,
        dets=dets,
        configs_proc=None,
        logger=logger,
        overwrite=overwrite,
        save_archive=False,
        compress=compress,
    )

    return out_dict, errors


def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          configs: str,
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

    temp_subdir = "temp"

    tqdm.monitor_interval = 0

    configs, context = pp_util.get_preprocess_context(configs)
    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    os.makedirs(os.path.dirname(configs['archive']['policy']['filename']),
                exist_ok=True)

    jobdb_path = configs.get("jobdb", None)
    if jobdb_path is not None:
        jdb = JobManager(sqlite_file=jobdb_path)
    elif run_from_jobdb:
        raise ValueError(f"Need a jobdb if using it to make a run_list.")

    errlog = os.path.join(os.path.dirname(configs['archive']['index']),
                          'errlog.txt')

    if run_from_jobdb:
        if not overwrite:
            job_list = jdb.get_jobs("init", jstate=JState.open)
        else:
            job_list = jdb.get_jobs("init")
        obs_list = list(set([j.tags['obs:obs_id'] for j in job_list]))
        if len(obs_list) == 0:
            logger.warning(f"No jobs found in jobdb.")
            return
    else:
        obs_list = get_obslist(context, query=query, obs_id=obs_id,
                            min_ctime=min_ctime, max_ctime=max_ctime,
                            update_delay=update_delay, tags=tags,
                            planet_obs=planet_obs)
        obs_list = [obs['obs_id'] for obs in obs_list]
        if len(obs_list) == 0:
            logger.warning(f"No observations returned from query: {query}")
            return

    # clean up lingering files from previous incomplete runs
    policy_dir = os.path.join(os.path.dirname(
            configs['archive']['policy']['filename']),
            temp_subdir
    )
    for obs_id in obs_list:
        pp_util.cleanup_obs(obs_id, policy_dir, errlog, configs, context,
                            subdir=temp_subdir, remove=overwrite)

    # remove datasets from final archive file not found in db
    pp_util.cleanup_archive(configs, logger)

    run_list = []

    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))

    if overwrite or not os.path.exists(configs['archive']['index']):
        db = None
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])

    if not run_from_jobdb:
        futures = []
        futures_dict = {}
        for obs_id in obs_list:
            futures.append(executor.submit(pp_util.get_groups, obs_id, configs))
            futures_dict[futures[-1]] = obs_id

        for future in tqdm(as_completed_callable(futures), total=len(futures),
                           desc="building run list from obs list"):
            obs_id = futures_dict[future]
            _, groups, _ = future.result()

            if db is not None and not overwrite:
                x = db.inspect({'obs:obs_id': obs_id})
                if x is not None and len(x) != 0 and len(x) != len(groups):
                    [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]

            for group in groups:
                if 'NC' not in group:
                    run_list.append((obs_id, group))

        if jobdb_path is not None:
            run_list = pp_util.filter_preproc_runlist_by_jobdb(
                jdb=jdb,
                jclass="init",
                db=db,
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
            elif db is not None:
                x = db.inspect({'obs:obs_id': obs_id})
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
                run_list.append((obs_id, group))

    if len(run_list) == 0:
        logger.info("Nothing to run")
        return
    logger.info(f'Run list created with {len(run_list)} obsid groups')

    # ensure db exists up front to prevent race conditions
    db = pp_util.get_preprocess_db(configs, group_by, logger)

    futures = []
    futures_dict = {}
    obs_errors = {}

    for r in run_list:
        futures.append(
            executor.submit(
                preprocess_tod,
                obs_id=r[0],
                configs=configs,
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

    # batch updates to ManifestDb
    batch_size = configs['archive'].get('batch_size', 1)

    pb_name = f"pb_{str(int(time.time()))}.txt"
    with open(pb_name, 'w') as f:
        with DbBatchManager(db, batch_size=batch_size, logger=logger) as db_manager:
            for future in tqdm(as_completed_callable(futures), total=total,
                            desc="preprocess_tod", file=f,
                            miniters=max(1, total // 100)):
                obs_id, group = futures_dict[future]
                out_meta = (obs_id, group)
                try:
                    out_dict, errors = future.result()
                    obs_errors[obs_id].append({'group': group, 'error': errors[0]})
                    logger.info(f"{obs_id}: {group} extracted successfully")
                except Exception as e:
                    errmsg, tb = PreprocessErrors.get_errors(e)
                    logger.error(f"Executor Future Result Error for {obs_id}: {group}:\n{errmsg}\n{tb}")
                    obs_errors[obs_id].append({'group': group, 'error': PreprocessErrors.ExecutorFutureError})
                    out_dict = None
                    errors = (PreprocessErrors.ExecutorFutureError, errmsg, tb)

                futures.remove(future)

                pp_util.cleanup_mandb(out_dict, out_meta, errors, configs,
                                    logger, overwrite, db_manager=db_manager)

                # update jobdb
                if jobdb_path is not None:
                    tags = {}
                    tags["obs:obs_id"] = obs_id
                    for gb, g in zip(group_by, group):
                        tags['dets:' + gb] = g
                    job = jdb.get_jobs(jclass="init", jstate=JState.open, tags=tags)
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
    logger.info("preprocess_tod is done")


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('configs', help="Preprocessing Configuration File")
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


def main(configs: str,
         query: str = '',
         obs_id: Optional[str] = None,
         overwrite: bool = False,
         min_ctime: Optional[int] = None,
         max_ctime: Optional[int] = None,
         update_delay: Optional[int] = None,
         tags: Optional[List[str]] = None,
         planet_obs: bool = False,
         verbosity: Optional[int] = None,
         nproc: int = 4,
         compress: bool = False,
         run_from_jobdb: bool = False,
         raise_error: bool = False):

    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor=executor,
              as_completed_callable=as_completed_callable,
              configs=configs,
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
