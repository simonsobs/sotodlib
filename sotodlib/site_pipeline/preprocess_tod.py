import os
import yaml
import time
import logging
from typing import Optional, Union, Callable
import numpy as np
import argparse
import traceback
from typing import Optional
import cProfile, pstats, io
from sotodlib.utils.procs_pool import get_exec_env
import h5py
import copy
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")


def dummy_preproc(obs_id, group_list, logger,
                  configs, overwrite, run_parallel):
    """
    Dummy function that can be put in place of preprocess_tod in the
    main function for testing issues in the processpoolexecutor
    (multiprocessing).
    """
    error = None
    outputs = []
    context = core.Context(configs["context_file"])
    group_by, groups, error = pp_util.get_groups(obs_id, configs, context)
    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)
    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")
        dets = {gb:gg for gb, gg in zip(group_by, group)}
        proc_aman = core.AxisManager(core.LabelAxis('dets', ['det%i' % i for i in range(3)]),
                                     core.OffsetAxis('samps', 1000))
        proc_aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')
        proc_aman.wrap_new('timestamps', ('samps',))[:] = (np.arange(proc_aman.samps.count) / 200)

        outputs_grp = pp_util.save_group(obs_id, configs, dets, context, subdir='temp')
        logger.info(f"Saving data to {outputs_grp['temp_file']}:{outputs_grp['db_data']['dataset']}")
        proc_aman.save(outputs_grp['temp_file'], outputs_grp['db_data']['dataset'], overwrite)

        if run_parallel:
            outputs.append(outputs_grp)

    if run_parallel:
        return error, outputs


def preprocess_tod(obs_id,
                   configs,
                   verbosity=0,
                   group_list=None,
                   overwrite=False,
                   run_parallel=False,
                   run_tracemalloc=False):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and saves the results in
    the ManifestDb specified in the configs.

    Arguments
    ----------
    obs_id: string or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs: string or dictionary
        config file or loaded config directory
    group_list: None or list
        list of groups to run if you only want to run a partial update
    overwrite: bool
        if True, overwrite existing entries in ManifestDb
    verbosity: log level
        0 = error, 1 = warn, 2 = info, 3 = debug
    run_parallel: Bool
        If true preprocess_tod is called in a parallel process which returns
        dB info and errors and does no sqlite writing inside the function.
    """

    outputs = []
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups, error = pp_util.get_groups(obs_id, configs, context)

    if error is not None:
        if run_parallel:
            return error[0], [None, None]
        else:
            return

    all_groups = groups.copy()
    for g in all_groups:
        if group_list is not None:
            if g not in group_list:
                groups.remove(g)
                continue
        if 'wafer.bandpass' in group_by:
            if 'NC' in g:
                groups.remove(g)
                continue
        try:
            meta = context.get_meta(obs_id, dets = {gb:gg for gb, gg in zip(group_by, g)})
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {g}\n{errmsg}\n{tb}")
            groups.remove(g)
            continue

        if meta.dets.count == 0:
            groups.remove(g)

    if len(groups) == 0:
        logger.warning(f"group_list:{group_list} contains no overlap with "
                       f"groups in observation: {obs_id}:{all_groups}. "
                       f"No analysis to run.")
        error = 'no_group_overlap'
        if run_parallel:
            return error, [None, None]
        else:
            return

    if not(run_parallel):
        db = pp_util.get_preprocess_db(configs, group_by)

    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)

    if configs.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False

    n_fail = 0
    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")
        dets = {gb:gg for gb, gg in zip(group_by, group)}
        try:
            aman = context.get_obs(obs_id, dets=dets)
            tags = np.array(context.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            aman.wrap('tags', tags)
            if run_tracemalloc:
                proc_aman, success, snapshots_process, snapshots_calc = pipe.run(aman, run_tracemalloc=run_tracemalloc)

                dest_dataset = obs_id
                for gb, g in zip(group_by, group):
                    if gb == 'detset':
                        dest_dataset += "_" + g
                    else:
                        dest_dataset += "_" + gb + "_" + str(g)
                trace_dir = os.path.join(os.path.dirname(configs['archive']['policy']['filename']), "trace")
                for i, snap in enumerate(snapshots_process):
                    snap[1].dump(os.path.join(trace_dir, f"{dest_dataset}_snapshot_process_{snap[0]}_{i}.pkl"))
                for i, snap in enumerate(snapshots_calc):
                    snap[1].dump(os.path.join(trace_dir, f"{dest_dataset}_snapshot_calc_{snap[0]}_{i}.pkl"))
            else:
                proc_aman, success = pipe.run(aman, run_tracemalloc=run_tracemalloc)

            if make_lmsi:
                new_plots = os.path.join(configs["plot_dir"],
                                         f'{str(aman.timestamps[0])[:5]}',
                                         aman.obs_info.obs_id)
        except Exception as e:
            #error = f'{obs_id} {group}'
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {group}\n{errmsg}\n{tb}")
            # return error, None, [errmsg, tb]
            # need a better way to log if just one group fails.
            n_fail += 1
            continue
        if success != 'end':
            # If a single group fails we don't log anywhere just mis an entry in the db.
            logger.info(f"ERROR: {obs_id} {group}\nFailed at step {success}")
            n_fail += 1
            continue

        outputs_grp = pp_util.save_group(obs_id, configs, dets, context, subdir='temp')
        logger.info(f"Saving data to {outputs_grp['temp_file']}:{outputs_grp['db_data']['dataset']}")
        proc_aman.save(outputs_grp['temp_file'], outputs_grp['db_data']['dataset'], overwrite)

        if run_parallel:
            outputs.append(outputs_grp)
        else:
            logger.info(f"Saving to database under {outputs_grp['db_data']}")
            if len(db.inspect(outputs_grp['db_data'])) == 0:
                h5_path = os.path.relpath(outputs_grp['temp_file'],
                        start=os.path.dirname(configs['archive']['index']))
                db.add_entry(outputs_grp['db_data'], h5_path)

    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi

        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))

    if run_parallel:
        if n_fail == len(groups):
            # If no groups make it to the end of the processing return error.
            logger.info(f'ERROR: all groups failed for {obs_id}')
            error = 'all_fail'
            return error, [obs_id, 'all groups']
        else:
            logger.info('Returning data to futures')
            error = None
            return error, outputs


def load_preprocess_tod_sim(obs_id, sim_map,
                            configs="preprocess_configs.yaml",
                            context=None, dets=None,
                            meta=None, modulated=True, logger=logger):
    """Loads the saved information from the preprocessing pipeline and runs the
    processing section of the pipeline on simulated data

    Assumes preprocess_tod has already been run on the requested observation.

    Arguments
    ----------
    obs_id: multiple
        passed to ``context.get_obs`` to load AxisManager, see Notes for
        `context.get_obs`
    sim_map: pixell.enmap.ndmap
        signal map containing (T, Q, U) fields
    configs: string or dictionary
        config file or loaded config directory
    dets: dict
        dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    modulated: bool
        If True, apply the HWP angle model and scan the simulation
        into a modulated signal.
        If False, scan the simulation into demodulated timestreams.
    """
    configs, context = pp_util.get_preprocess_context(configs, context)
    if dets is not None:
        meta.restrict("dets", dets)
    meta = pp_util.load_preprocess_det_select(
        obs_id, configs=configs, context=context, meta=meta, logger=logger
    )

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
        '--profile',
        help="Run profiling.",
        type=bool,
        default=False
    )
    parser.add_argument(
        '--tracemalloc',
        help="Run tracemalloc.",
        type=bool,
        default=False
    )
    parser.add_argument(
        '--raise-error',
        help="Raise an error upon completion if any obsids or groups fail.",
        type=bool,
        default=False
    )
    return parser


def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          configs: str,
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
          run_profiling: Optional[bool] = False,
          run_tracemalloc: Optional[bool] = False,
          raise_error: Optional[bool] = False):

    configs, context = pp_util.get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    errlog = os.path.join(os.path.dirname(configs['archive']['index']),
                          'errlog.txt')

    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id, min_ctime=min_ctime,
                                   max_ctime=max_ctime, update_delay=update_delay, tags=tags,
                                   planet_obs=planet_obs)

    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")

    # clean up lingering files from previous incomplete runs
    policy_dir = os.path.join(os.path.dirname(configs['archive']['policy']['filename']), 'temp')
    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.cleanup_obs(obs_id, policy_dir, errlog, configs, context,
                            subdir='temp', remove=overwrite)

    run_list = []

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = [ (o,None) for o in obs_list]
        group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            if x is None or len(x) == 0:
                run_list.append( (obs, None) )
            else:
                group_by, groups, _ = pp_util.get_groups(obs["obs_id"], configs, context)
                if len(x) != len(groups):
                    [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]
                    run_list.append( (obs, groups) )

    logger.info(f'Run list created with {len(run_list)} obsids')

    n_fail = 0

    if run_profiling:
        profile_dir = os.path.join(os.path.dirname(configs['archive']['policy']['filename']), 'prof')
        if not(os.path.exists(profile_dir)):
            os.makedirs(profile_dir)
    else:
        profile_dir = None

    if run_tracemalloc:
        trace_dir = os.path.join(os.path.dirname(configs['archive']['policy']['filename']), 'trace')
        if not(os.path.exists(trace_dir)):
            os.makedirs(trace_dir)

    # Run write_block obs-ids in parallel at once then write all to the sqlite db.
    futures = [executor.submit(pp_util.profile_function,
                    func=preprocess_tod,
                    profile_path=os.path.join(profile_dir, f'{r[0]["obs_id"]}.prof') if profile_dir is not None else None,
                    obs_id=r[0]['obs_id'],
                    group_list=r[1], verbosity=verbosity,
                    configs=configs,
                    overwrite=overwrite,
                    run_parallel=True,
                    run_tracemalloc=run_tracemalloc) for r in run_list]
    for future in as_completed_callable(futures):
        logger.info('New future as_completed result')
        try:
            err, db_datasets = future.result()
            if err is not None:
                n_fail += 1
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
            f = open(errlog, 'a')
            f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            f.close()
            n_fail+=1
            continue
        futures.remove(future)

        if db_datasets:
            if err is None:
                logger.info(f'Processing future result db_dataset: {db_datasets}')
                for db_dataset in db_datasets:
                    pp_util.cleanup_mandb(err, db_dataset, configs, logger)
            else:
                pp_util.cleanup_mandb(err, db_datasets, configs, logger)

    if run_profiling:
        combined_profile_dir = os.path.join(profile_dir, 'combined_profile.prof')
        if os.path.exists(combined_profile_dir):
            combined_stats = pstats.Stats(combined_profile_dir)
        else:
            combined_stats = None
        for r in run_list:
            profile_file = os.path.join(profile_dir, f'{r[0]["obs_id"]}.prof')
            if os.path.exists(profile_file):
                try:
                    stats = pstats.Stats(profile_file)
                    if combined_stats is None:
                        combined_stats = stats
                    else:
                        combined_stats.add(stats)
                except Exception as e:
                    logger.error(f"cannot get stats for {r[0]['obs_id']}: {e}")
        if combined_stats is not None:
            combined_stats.dump_stats(combined_profile_dir)

    if raise_error and n_fail > 0:
        raise RuntimeError(f"preprocess_tod: {n_fail}/{len(run_list)} obs_ids failed")

def main(configs: str,
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
         run_profiling: Optional[bool] = False,
         run_tracemalloc: Optional[bool] = False,
         raise_error: Optional[bool] = False):

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
              run_profiling=run_profiling,
              run_tracemalloc=run_tracemalloc,
              raise_error=raise_error)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)