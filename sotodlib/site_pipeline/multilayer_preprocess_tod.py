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
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib import core
from sotodlib.preprocess import _Preprocess, Pipeline, processes
import sotodlib.preprocess.preprocess_util as pp_util
import sotodlib.site_pipeline.util as sp_util

logger = pp_util.init_logger("preprocess")

def multilayer_preprocess_tod(obs_id, 
                              configs_init,
                              configs_proc,
                              verbosity=0,
                              group_list=None, 
                              overwrite=False,
                              run_parallel=False):
    """Meant to be run as part of a batched script. Given a single
    Observation ID, this function uses an existing ManifestDb generated
    from a previous runof the processing pipeline, runs the pipeline using
    a second config, and outputs a new ManifestDb. Det groups must exist in
    the first dB to be included in the pipeline run on the second config.

    Arguments
    ----------
    obs_id: string or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs_init: string or dictionary
        config file or loaded config directory for existing database
    configs_proc: string or dictionary
        config file or loaded config directory for processing database
        to be output
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

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    # list to hold error, destination file, and db data
    outputs_init = []
    outputs_proc = []

    if type(configs_init) == str:
        configs_init = yaml.safe_load(open(configs_init, "r"))
    context_init = core.Context(configs_init["context_file"])

    if type(configs_proc) == str:
        configs_proc = yaml.safe_load(open(configs_proc, "r"))
    context_proc = core.Context(configs_proc["context_file"])

    group_by_init, groups_init, error_init = pp_util.get_groups(obs_id, configs_init, context_init)
    group_by_proc, groups_proc, error_proc = pp_util.get_groups(obs_id, configs_proc, context_proc)

    if error_init is not None:
        if run_parallel:
            return error_init[0], [None, None], [None, None]
        else:
            return

    if error_proc is not None:
        if run_parallel:
            return error_proc[0], [None, None], [None, None]
        else:
            return

    if len(groups_init) > 0 and len(groups_proc) > 0:
        if (group_by_init != group_by_proc).any():
            raise ValueError('init and proc groups do not match')

    all_groups_proc = groups_proc.copy()
    for g in all_groups_proc:
        if g not in groups_init:
            groups_proc.remove(g)
            continue
        if group_list is not None:
            if g not in group_list:
                groups_proc.remove(g)
                continue
        if 'wafer.bandpass' in group_by_proc:
            if 'NC' in g:
                groups_proc.remove(g)
                continue
        try:
            meta = context_proc.get_meta(obs_id, dets = {gb:gg for gb, gg in zip(group_by_proc, g)})
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {g}\n{errmsg}\n{tb}")
            groups_proc.remove(g)
            continue

        if meta.dets.count == 0:
            groups_proc.remove(g)

    if len(groups_proc) == 0:
        logger.warning(f"group_list:{group_list} contains no overlap with "
                       f"groups in observation: {obs_id}:{all_groups_proc}. "
                       f"No analysis to run.")
        error = 'no_group_overlap'
        if run_parallel:
            return error, [None, None], [None, None]
        else:
            return

    if not(run_parallel):
        db_init = pp_util.get_preprocess_db(configs_init, group_by_proc)
        db_proc = pp_util.get_preprocess_db(configs_proc, group_by_proc)

    # pipeline for init config
    pipe_init = Pipeline(configs_init["process_pipe"],
                         plot_dir=configs_init["plot_dir"], logger=logger)
    # pipeline for processing config
    pipe_proc = Pipeline(configs_proc["process_pipe"],
                         plot_dir=configs_proc["plot_dir"], logger=logger)

    if configs_proc.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False

    # loop through and reduce each group
    n_fail = 0
    for group in groups_proc:
        logger.info(f"Beginning run for {obs_id}:{group}")
        dets = {gb:gg for gb, gg in zip(group_by_proc, group)}
        try:
            error, outputs_grp_init, _, aman = pp_util.preproc_or_load_group(obs_id, configs_init,
                                                                             dets=dets, logger=logger)
            if error is None:
                outputs_init.append(outputs_grp_init)

            init_fields = aman.preprocess._fields.copy()

            outputs_grp_proc = pp_util.save_group(obs_id, configs_proc, dets,
                                      context_proc, subdir='temp_proc')

            # tags from context proc
            tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            if "tags" in aman._fields:
                aman.move("tags", None)
            aman.wrap('tags', tags_proc)

            # now run the pipeline on the processed axis manager
            logger.info(f"Beginning processing pipeline for {obs_id}:{group}")
            proc_aman, success = pipe_proc.run(aman)
            proc_aman.wrap('pcfg_ref', pp_util.get_pcfg_check_aman(pipe_init))

            # remove fields found in aman.preprocess from proc_aman
            for fld_init in init_fields:
                if fld_init in proc_aman:
                    proc_aman.move(fld_init, None)

        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {group}\n{errmsg}\n{tb}")
            n_fail += 1
            continue
        if success != 'end':
            # If a single group fails we don't log anywhere just mis an entry in the db.
            logger.info(f"ERROR: {obs_id} {group}\nFailed at step {success}") 
            n_fail += 1
            continue

        logger.info(f"Saving data to {outputs_grp_proc['temp_file']}:{outputs_grp_proc['db_data']['dataset']}")
        proc_aman.save(outputs_grp_proc['temp_file'], outputs_grp_proc['db_data']['dataset'], overwrite)

        if run_parallel:
            outputs_proc.append(outputs_grp_proc)
        else:
            logger.info(f"Saving to database under {outputs_grp_proc['db_data']}")
            if len(db_proc.inspect(outputs_grp_proc['db_data'])) == 0:
                h5_path = os.path.relpath(outputs_grp_proc['temp_file'],
                        start=os.path.dirname(configs_proc['archive']['index']))
                db_proc.add_entry(outputs_grp_proc['db_data'], h5_path)

    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi

        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs1["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))

    if run_parallel:
        if n_fail == len(groups_proc):
            # If no groups make it to the end of the processing return error.
            logger.info(f'ERROR: all groups failed for {obs_id}')
            error = 'all_fail'
            return error, [obs_id, 'all groups'], [obs_id, 'all groups']
        else:
            logger.info('Returning data to futures')
            error = None
            return error, outputs_init, outputs_proc


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('configs_init', help="Preprocessing Configuration File for existing database")
    parser.add_argument('configs_proc', help="Preprocessing Configuration File for new database")
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
    return parser

def main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
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
         nproc: Optional[int] = 4):

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)

    errlog = os.path.join(os.path.dirname(configs_proc['archive']['index']),
                          'errlog.txt')

    obs_list = sp_util.get_obslist(context_proc, query=query, obs_id=obs_id, min_ctime=min_ctime,
                                   max_ctime=max_ctime, update_delay=update_delay, tags=tags,
                                   planet_obs=planet_obs)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")

    # clean up lingering files from previous incomplete runs
    policy_dir_init = os.path.join(os.path.dirname(configs_init['archive']['policy']['filename']), 'temp')
    policy_dir_proc = os.path.join(os.path.dirname(configs_proc['archive']['policy']['filename']), 'temp_proc')
    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.cleanup_obs(obs_id, policy_dir_init, errlog, configs_init, context_init,
                            subdir='temp', remove=overwrite)
        pp_util.cleanup_obs(obs_id, policy_dir_proc, errlog, configs_proc, context_proc,
                            subdir='temp_proc', remove=overwrite)

    run_list = []

    if overwrite or not os.path.exists(configs_proc['archive']['index']):
        # run on all if database doesn't exist
        for obs in obs_list:
            #run on all if database doesn't exist
            run_list = [ (o,None) for o in obs_list]
            group_by_proc = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))
    else:
        db = core.metadata.ManifestDb(configs_proc['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            if x is None or len(x) == 0:
                run_list.append( (obs, None) )
            else:
                group_by_proc, groups_proc, _ = pp_util.get_groups(obs["obs_id"], configs_proc, context_proc)
                if len(x) != len(groups_proc):
                    [groups_proc.remove([a[f'dets:{gb}'] for gb in group_by_proc]) for a in x]
                    run_list.append( (obs, groups_proc) )

    logger.info(f'Run list created with {len(run_list)} obsids')

    # run write_block obs-ids in parallel at once then write all to the sqlite db.
    futures = [executor.submit(multilayer_preprocess_tod, obs_id=r[0]['obs_id'],
                group_list=r[1], verbosity=verbosity,
                configs_init=configs_init,
                configs_proc=configs_proc,
                overwrite=overwrite, run_parallel=True) for r in run_list]
    for future in as_completed_callable(futures):
        logger.info('New future as_completed result')
        try:
            err, db_datasets_init, db_datasets_proc = future.result()
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
            f = open(errlog, 'a')
            f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            f.close()
            continue
        futures.remove(future)

        if db_datasets_init:
            if err is None:
                for db_dataset in db_datasets_init:
                    logger.info(f'Processing future result db_dataset: {db_datasets_init}')
                    pp_util.cleanup_mandb(err, db_dataset, configs_init, logger, overwrite)
            else:
                pp_util.cleanup_mandb(err, db_datasets_init, configs_init, logger, overwrite)

        if db_datasets_proc:
            if err is None:
                logger.info(f'Processing future dependent result db_dataset: {db_datasets_proc}')
                for db_dataset in db_datasets_proc:
                    pp_util.cleanup_mandb(err, db_dataset, configs_proc, logger, overwrite)
            else:
                pp_util.cleanup_mandb(err, db_datasets_proc, configs_proc, logger, overwrite)

if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **args)
