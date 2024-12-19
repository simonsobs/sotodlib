import os
import yaml
import time
import logging
import numpy as np
import argparse
import traceback
from typing import Optional, List
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import copy
from sotodlib import core
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib.preprocess import _Preprocess, Pipeline, processes
import sotodlib.preprocess.preprocess_util as pp_util
import sotodlib.site_pipeline.util as sp_util

logger = pp_util.init_logger("preprocess")

def preprocess_obs(
    obs_id, 
    configs,  
    overwrite=False, 
    logger=None
):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and saves the results in
    the ManifestDb specified in the configs.   

    Arguments
    ----------
    obs_id: string or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs: string or dictionary
        config file or loaded config directory
    overwrite: bool
        if True, overwrite existing entries in ManifestDb
    logger: logging instance
        the logger to print to
    """

    if logger is None: 
        logger = pp_util.init_logger("preprocess")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])

    if os.path.exists(configs['archive']['index']):
        logger.info(f"Mapping {configs['archive']['index']} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(configs['archive']['index'])
    else:
        logger.info(f"Creating {configs['archive']['index']} for the "
                     "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            configs['archive']['index'],
            scheme=scheme
        )

    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)

    logger.info(f"Beginning run for {obs_id}")

    aman = context.get_obs(obs_id, no_signal=True)
    proc_aman, success = pipe.run(aman)
    if success != 'end':
        return

    policy = pp_util.ArchivePolicy.from_params(configs['archive']['policy'])
    dest_file, dest_dataset = policy.get_dest(obs_id)
    logger.info(f"Saving data to {dest_file}:{dest_dataset}")
    proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

    # Update the index.
    db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}

    logger.info(f"Saving to database under {db_data}")
    if len(db.inspect(db_data)) == 0:
        h5_path = os.path.relpath(dest_file,
                start=os.path.dirname(configs['archive']['index']))
        db.add_entry(db_data, h5_path)

    if configs.get("lmsi_config", None) is not None:
        from pathlib import Path
        import lmsi.core as lmsi

        new_plots = os.path.join(configs["plot_dir"],
                                 f'{str(aman.timestamps[0])[:5]}',
                                 aman.obs_info.obs_id)
        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))


def load_preprocess_tod_sim(obs_id, sim_map,
                            configs="preprocess_configs.yaml",
                            context=None, dets=None,
                            meta=None, modulated=True):
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
    meta = pp_util.load_preprocess_det_select(obs_id, configs=configs,
                                              context=context, dets=dets,
                                              meta=meta)

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
    group_by, groups = pp_util.get_groups(obs_id, configs, context)
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
                   run_parallel=False):
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
    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups = pp_util.get_groups(obs_id, configs, context)
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
            proc_aman, success = pipe.run(aman)

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


def multilayer_preprocess_tod(obs_id, 
                              configs_init,
                              configs_proc,
                              verbosity=0,
                              group_list=None, 
                              overwrite=False,
                              run_parallel=False):
    """Meant to be run as part of a batched script. Given a single
    Observation ID, this function runs the pipeline using two different
    configs. If the first ManifestDb exists, it will load it, otherwise it
    will run the pipeline on the first config, run the pipeline using
    the second config, and output a new ManifestDb. Det groups must exist in
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

    group_by_proc, groups_proc = pp_util.get_groups(obs_id, configs_proc, context_proc)

    all_groups_proc = groups_proc.copy()
    for g in all_groups_proc:
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
                                                                             dets=dets, logger=logger,
                                                                             context_init=context_init)
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
        '--no-signal',
        help="Whether or not to load signal (False calls preprocess_obs)",
        type=bool,
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


def main(configs_init: str,
         configs_proc: Optional[str] = None,
         query: Optional[str] = None, 
         obs_id: Optional[str] = None,
         no_signal: Optional[bool] = False,
         overwrite: bool = False,
         min_ctime: Optional[int] = None,
         max_ctime: Optional[int] = None,
         update_delay: Optional[int] = None,
         tags: Optional[str] = None,
         planet_obs: bool = False,
         verbosity: Optional[int] = None,
         nproc: Optional[int] = 4):

    logger = pp_util.init_logger("preprocess", verbosity=verbosity)

    # whether we're running multiple configs or not
    run_multi = False

    if configs_proc is not None:
        if no_signal == False:
            run_multi = True
        else:
            logger.warning('no_signal is True. ignoring second config file')

    configs_init, context_init = pp_util.get_preprocess_context(configs_init)

    if run_multi:
        configs_proc, context_proc = pp_util.get_preprocess_context(configs_proc)
        configs = configs_proc
        context = context_proc
    else:
        configs = configs_init
        context = context_init

    errlog = os.path.join(os.path.dirname(configs['archive']['index']),'errlog.txt')

    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id, min_ctime=min_ctime, 
                                   max_ctime=max_ctime, update_delay=update_delay, tags=tags, 
                                   planet_obs=planet_obs)

    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")

    # clean up lingering files from previous incomplete runs
    for obs in obs_list:
        obs_id = obs['obs_id']
        pp_util.save_group_and_cleanup(obs_id, configs_init, context_init,
                                       subdir='temp', remove=overwrite)
        if run_multi:
            pp_util.save_group_and_cleanup(obs_id, configs_proc, context_proc,
                                           subdir='temp_proc', remove=overwrite)

    run_list = []

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        for obs in obs_list:
            group_by, groups = pp_util.get_groups(obs["obs_id"], configs_init, context_init)

            if run_multi:
                group_by_proc, groups_proc = pp_util.get_groups(obs["obs_id"], configs_proc, context_proc)

                if (group_by != group_by_proc).any():
                    raise ValueError('init and proc groups do not match')

                all_groups_proc = groups_proc.copy()
                for g in all_groups_proc:
                    if g not in groups:
                        groups_proc.remove(g)
                groups = groups_proc
                group_by = group_by_proc

            run_list.append( (obs, groups) )

    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            group_by, groups = pp_util.get_groups(obs["obs_id"], configs_init, context_init)

            if run_multi:
                group_by_proc, groups_proc = pp_util.get_groups(obs["obs_id"], configs_proc, context_proc)

                if (group_by != group_by_proc).any():
                    raise ValueError('init and proc groups do not match')

                all_groups_proc = groups_proc.copy()
                for g in all_groups_proc:
                    if g not in groups:
                        groups_proc.remove(g)
                groups = groups_proc
                group_by = group_by_proc

            if x is None or len(x) == 0:
                run_list.append( (obs, groups) )
            elif len(x) != len(groups):
                [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]
                run_list.append( (obs, groups) )


    logger.info(f'Run list created with {len(run_list)} obsids')

    if no_signal:
        logger.info('Running preprocess_obs')
        for obs in run_list:
            logger.info(f"Processing obs_id: {obs_id}")
            try:
                preprocess_obs(obs["obs_id"], configs, overwrite=overwrite, logger=logger)
            except Exception as e:
                logger.info(f"preprocess_obs failed wi{type(e)}: {e}")
                logger.info(''.join(traceback.format_tb(e.__traceback__)))
                logger.info(f'Skiping obs:{obs["obs_id"]} and moving to the next')
                continue
    else:
        multiprocessing.set_start_method('spawn')

        with ProcessPoolExecutor(nproc) as exe:
            if not run_multi:
                futures = [exe.submit(preprocess_tod, obs_id=r[0]['obs_id'],
                     group_list=r[1], verbosity=verbosity,
                     configs=configs,
                     overwrite=overwrite, run_parallel=True) for r in run_list]
            else:
                futures = [exe.submit(multilayer_preprocess_tod, obs_id=r[0]['obs_id'],
                            group_list=r[1], verbosity=verbosity,
                            configs_init=configs_init,
                            configs_proc=configs_proc,
                            overwrite=overwrite, run_parallel=True) for r in run_list]

            for future in as_completed(futures):
                logger.info('New future as_completed result')
                try:
                    if not run_multi:
                        err, db_datasets_init = future.result()
                    else:
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

                logger.info(f'Processing future result db_dataset: {db_datasets_init}')
                if err is None:
                    if db_datasets_init:
                        for db_dataset in db_datasets_init:
                            pp_util.cleanup_mandb(err, db_dataset, configs, logger)
                    if run_multi and db_datasets_proc:
                        logger.info(f'Processing future dependent result db_dataset: {db_datasets_proc}')
                        for db_dataset in db_datasets_proc:
                            pp_util.cleanup_mandb(err, db_dataset, configs_proc, logger, overwrite)


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
