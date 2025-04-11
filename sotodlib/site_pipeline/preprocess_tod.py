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
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess.preprocess_util import PreprocessErrors
from sotodlib.preprocess import _Preprocess, Pipeline, processes


logger = pp_util.init_logger("preprocess")

def dummy_preproc(obs_id,
                   configs,
                   group,
                   verbosity=0,
                   overwrite=False,
                   run_parallel=False):
    """
    Dummy function that can be put in place of preprocess_tod in the
    main function for testing issues in the processpoolexecutor
    (multiprocessing).
    """
    temp_subdir = "temp"
    logger = pp_util.init_logger("preprocess", verbosity=verbosity)
    logger.info(f"Starting preprocess run of {obs_id}: {group}")

    context = core.Context(configs["context_file"])

    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))

    error = None
    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"],
                    logger=logger)
    dets = {gb:gg for gb, gg in zip(group_by, group)}
    proc_aman = core.AxisManager(core.LabelAxis('dets', ['det%i' % i for i in range(3)]),
                                 core.OffsetAxis('samps', 1000))
    proc_aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')
    proc_aman.wrap_new('timestamps', ('samps',))[:] = (np.arange(proc_aman.samps.count) / 200)

    out_dict = pp_util.save_group(obs_id, configs, dets, context,
                                  subdir=temp_subdir)
    logger.info(f"Saving data to {out_dict['temp_file']}:{out_dict['db_data']['dataset']}")
    proc_aman.save(out_dict['temp_file'], out_dict['db_data']['dataset'],
                   overwrite)

    return out_dict, (obs_id, group), error


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


def preprocess_tod(obs_id,
                   configs,
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
    configs: string or dictionary
        config file or loaded config directory
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

    temp_subdir = "temp"

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)
    logger.info(f"Starting preprocess run of {obs_id}: {group}")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])

    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))

    try:
        meta = context.get_meta(obs_id, dets = {gb:gg for gb,
                                                gg in zip(group_by, group)})
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Get Metadata failure: {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, (obs_id, group), PreprocessErrors.MetaDataError

    if meta.dets.count == 0:
        logger.warning(f"No analysis to run for {obsid}: {group}")
        return None, (obs_id, group), PreprocessErrors.NoDetsRemainError

    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"],
                    logger=logger)

    if configs.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False

    dets = {gb:gg for gb, gg in zip(group_by, group)}

    try:
        aman = context.get_obs(obs_id, dets=dets)
        aman.wrap('tags', np.array(context.obsdb.get(aman.obs_info.obs_id,
                                                     tags=True)['tags']))
        proc_aman, success = pipe.run(aman)

        if make_lmsi:
            new_plots = os.path.join(configs["plot_dir"],
                                     f'{str(aman.timestamps[0])[:5]}',
                                     aman.obs_info.obs_id
                                    )
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Pipeline Run Error for {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, (obs_id, group), PreprocessErrors.PipeLineRunError
    if success != 'end':
        logger.error(f"Pipeline Step Error for {obs_id}: {group}\nFailed at step {success}")
        return None, (obs_id, group), PreprocessErrors.PipeLineStepError

    out_dict = pp_util.save_group(obs_id, configs, dets, context,
                                  subdir=temp_subdir)
    logger.info(f"Saving data to {out_dict['temp_file']}:{out_dict['db_data']['dataset']}")
    proc_aman.save(out_dict['temp_file'], out_dict['db_data']['dataset'],
                   overwrite)

    if not run_parallel:
        logger.info(f"Saving {obs_id}: {group} to database under {out_dict['db_data']}")
        db = pp_util.get_preprocess_db(configs, group_by)
        if len(db.inspect(out_dict['db_data'])) == 0:
            h5_path = os.path.relpath(out_dict['temp_file'],
                    start=os.path.dirname(configs['archive']['index']))
            db.add_entry(out_dict['db_data'], h5_path)

    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi

        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))

    return out_dict, (obs_id, group), None


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
        '--raise-error',
        help="Raise an error upon completion if any obsids or groups fail.",
        type=bool,
        default=False
    )
    return parser


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
         raise_error: Optional[bool] = False):

    multiprocessing.set_start_method('spawn')

    temp_subdir = "temp"

    configs, context = pp_util.get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    errlog = os.path.join(os.path.dirname(configs['archive']['index']),
                          'errlog.txt')

    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id,
                                   min_ctime=min_ctime, max_ctime=max_ctime,
                                   update_delay=update_delay, tags=tags,
                                   planet_obs=planet_obs)
    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")
        return

    # clean up lingering files from previous incomplete runs
    policy_dir = os.path.join(os.path.dirname(configs['archive']['policy']['filename']),
                              temp_subdir)
    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.cleanup_obs(obs_id, policy_dir, errlog, configs, context,
                            subdir=temp_subdir, remove=overwrite)

    run_list = []

    futures_dict = {}

    if overwrite or not os.path.exists(configs['archive']['index']):
        db = None
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(pp_util.get_groups,
                              obs["obs_id"],
                              configs) for obs in obs_list]
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

    # Run write_block obs-ids in parallel at once then write all to the sqlite db.
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(preprocess_tod,
                     obs_id=r[0]['obs_id'],
                     group=r[1],
                     configs=configs,
                     verbosity=verbosity,
                     overwrite=overwrite,
                     run_parallel=True) for r in run_list
                  ]

        for r, future in zip(run_list, futures):
            futures_dict[future] = (r[0]['obs_id'], r[1])
            if r[0]['obs_id'] not in obs_errors:
                    obs_errors[r[0]['obs_id']] = []

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="preprocess_tod"):
            obs_id, group = futures_dict[future]
            logger.info(f"{obs_id}: {group} extracted successfully")
            try:
                out_dict, out_meta, error = future.result()
                obs_errors[obs_id].append({'group': group, 'error': error})
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.error(f"Multiprocess Future Result Error for {obs_id}: {group}:\n{errmsg}\n{tb}")
                obs_errors[obs_id].append({'group': group, 'error': PreprocessErrors.MultiProcessFutureError})
                out_dict = None
                out_meta = (obs_id, group)
                error = PreprocessErrors.MultiProcessFutureError
            futures.remove(future)

            logger.info(f"Adding future result to db for {obs_id}: {group}")
            pp_util.cleanup_mandb(out_dict, out_meta, error, configs, logger)

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

    pp_util.create_error_db(configs, obs_errors)

    if raise_error:
        raise RuntimeError("preprocess_tod ended with failed obsids")
    else:
        logger.info("preprocess_tod is done")


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)