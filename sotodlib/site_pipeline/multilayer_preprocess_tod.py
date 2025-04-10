import os
import yaml
import time
import logging
import numpy as np
import argparse
import traceback
from typing import Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import copy
from tqdm import tqdm
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib import core
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
    init_temp_subdir = "temp"
    proc_temp_subdir = "temp_proc"

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)
    logger.info(f"Starting preprocess run of {obs_id}: {group}")

    context_init = core.Context(configs_init["context_file"])
    context_proc = core.Context(configs_proc["context_file"])

    group_by = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

    try:
        meta_proc = context_proc.get_meta(obs_id, dets = {gb:gg for gb,
                                            gg in zip(group_by, group)})
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Get Metadata failure: {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, None, (obs_id, group), (obs_id, group), PreprocessErrors.MetaDataError

    if meta_proc.dets.count == 0:
        logger.warning(f"No analysis to run for {obs_id}: {group}")
        return None, None, (obs_id, group), (obs_id, group), PreprocessErrors.NoDetsRemainError

    if configs_proc.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False

    dets = {gb:gg for gb, gg in zip(group_by, group)}

    try:
        pipe_init = Pipeline(configs_init["process_pipe"],
                             plot_dir=configs_init["plot_dir"], logger=logger)
        pipe_proc = Pipeline(configs_proc["process_pipe"],
                             plot_dir=configs_proc["plot_dir"], logger=logger)

        error, out_dict_init, _, aman = pp_util.preproc_or_load_group(obs_id,
                                                                      configs_init,
                                                                      dets=dets,
                                                                      logger=logger,
                                                                      overwrite=overwrite)
        if error == 'load_success':
            out_dict_init = None
        init_fields = aman.preprocess._fields.copy()
        out_dict_proc = pp_util.save_group(obs_id, configs_proc,
                                           dets, context_proc,
                                           subdir='temp_proc')
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Pipeline Run Error for {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, None, (obs_id, group), (obs_id, group), PreprocessErrors.InitPipeLineRunError

    try:
        logger.info(f"Starting proc pipeline on {obs_id}: {group}")
        tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id,
                                                    tags=True)['tags'])
        if "tags" in aman._fields:
            aman.move("tags", None)
        aman.wrap('tags', tags_proc)

        proc_aman, success = pipe_proc.run(aman)
        proc_aman.wrap('pcfg_ref', pp_util.get_pcfg_check_aman(pipe_init))

        for init_field in init_fields:
            if init_field in proc_aman:
                proc_aman.move(init_field, None)
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Pipeline Run Error for {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, None, (obs_id, group), (obs_id, group), PreprocessErrors.ProcPipeLineRunError
    if success != 'end':
        logger.error(f"Pipeline Step Error for {obs_id} {group}\nFailed at step {success}")
        return None, None, (obs_id, group), (obs_id, group), PreprocessErrors.PipeLineStepError

    logger.info(f"Saving data to {out_dict_proc['temp_file']}:{out_dict_proc['db_data']['dataset']}")
    proc_aman.save(out_dict_proc['temp_file'], out_dict_proc['db_data']['dataset'],
                   overwrite)

    if not run_parallel:
        logger.info(f"Saving {obs_id}: {group} to database under {out_dict_proc['db_data']}")
        db = pp_util.get_preprocess_db(configs, group_by)
        if len(db.inspect(out_dict_proc['db_data'])) == 0:
            h5_path = os.path.relpath(out_dict_proc['temp_file'],
                    start=os.path.dirname(configs['archive']['index']))
            db.add_entry(out_dict_proc['db_data'], h5_path)

    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi

        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs_proc["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))

    return out_dict_init, out_dict_proc, (obs_id, group), (obs_id, group), None

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

    multiprocessing.set_start_method('spawn')

    init_temp_subdir = "temp"
    proc_temp_subdir = "temp_proc"

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)

    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

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

    futures_dict = {}

    if overwrite or not os.path.exists(configs_proc['archive']['index']):
        db = None
    else:
        db = core.metadata.ManifestDb(configs_proc['archive']['index'])

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(pp_util.get_groups,
                              obs["obs_id"],
                              configs_proc) for obs in obs_list]
        for obs, future in zip(obs_list, futures):
            futures_dict[future] = obs["obs_id"]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="building run list from obs list"):
            obs_id = futures_dict[future]
            _, groups, _ = future.result()

            if db is not None:
                x = db.inspect({'obs:obs_id': obs_id})
                if x is not None and len(x) != 0 and len(x) != len(groups):
                    [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]

            for group in groups:
                if 'NC' not in group:
                    run_list.append((obs, group))

    logger.info(f'Run list created with {len(run_list)} obsid groups')

    futures_dict = {}
    obs_errors = {}

    # run write_block obs-ids in parallel at once then write all to the sqlite db.
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(multilayer_preprocess_tod,
                      obs_id=r[0]['obs_id'],
                      group=r[1], verbosity=verbosity,
                      configs_init=configs_init,
                      configs_proc=configs_proc,
                      overwrite=overwrite,
                      run_parallel=True) for r in run_list]

        for r, future in zip(run_list, futures):
            futures_dict[future] = (r[0]['obs_id'], r[1])
            if r[0]['obs_id'] not in obs_errors:
                    obs_errors[r[0]['obs_id']] = []

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="multilayer_preprocess_tod"):
            obs_id, group = futures_dict[future]
            logger.info(f"{obs_id}: {group} extracted successfully")
            try:
                out_dict_init, out_dict_proc, out_meta_init, out_meta_proc, error = future.result()
                obs_errors[obs_id].append({'group': group, 'error': error})
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.error(f"Multiprocess Future Result Error for {obs_id}: {group}:\n{errmsg}\n{tb}")
                obs_errors[obs_id].append({'group': group, 'error': PreprocessErrors.MultiProcessFutureError})
                out_dict_init = None
                out_dict_proc = None
                out_meta_init = (obs_id, group)
                out_meta_proc = (obs_id, group)
                error = PreprocessErrors.MultiProcessFutureError
            futures.remove(future)

            logger.info(f"Adding future result to db for {obs_id}: {group}")
            pp_util.cleanup_mandb(out_dict_init, out_meta_init, error,
                                  configs_init, logger)
            pp_util.cleanup_mandb(out_dict_proc, out_meta_proc, error,
                                  configs_proc, logger)

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

    logger.warn(f"{n_obs_fail}/{len(obs_errors)} observations failed entirely")
    logger.warn(f"{n_groups_fail}/{len(run_list)} groups failed")

    pp_util.create_error_db(configs_proc, obs_errors)

    if raise_error:
        raise RuntimeError("multilayer_preprocess_tod ended with failed obsids")
    else:
        logger.info("multilayer_preprocess_tod is done")


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
