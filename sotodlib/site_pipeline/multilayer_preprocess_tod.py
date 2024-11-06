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
from sotodlib.coords import demod as demod_mm
from sotodlib.hwp import hwp_angle_model
from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
import sotodlib.site_pipeline.preprocess_common as sp_com
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")

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

    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    # list to hold error, destination file, and db data
    outputs = []

    if type(configs_init) == str:
        configs_init = yaml.safe_load(open(configs_init, "r"))
    context_init = core.Context(configs_init["context_file"])

    if type(configs_proc) == str:
        configs_proc = yaml.safe_load(open(configs_proc, "r"))
    context_proc = core.Context(configs_proc["context_file"])

    # get groups
    group_by_init, groups_init = sp_util.get_groups(obs_id, configs_init, context_init)
    group_by_proc, groups_proc = sp_util.get_groups(obs_id, configs_proc, context_proc)

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
        if g not in groups_init:
            groups_proc.remove(g)
        try:
            # all proc groups should be in init context
            meta = context_init.get_meta(obs_id, dets = {gb:gg for gb, gg in zip(group_by_proc, g)})
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {g}\n{errmsg}\n{tb}")
            groups_proc.remove(g)
            continue
        if meta.dets.count == 0:
            groups_proc.remove(g)

    # if no processing groups remain
    if len(groups_proc) == 0:
        logger.warning(f"group_list:{group_list} contains no overlap with "
                       f"groups in observation: {obs_id}:{all_groups_proc}. "
                       f"No analysis to run.")
        error = 'no_group_overlap'
        if run_parallel:
            return error, None, [None, None]
        else:
            return

    # get or create the processing database
    if not(run_parallel):
        db = sp_util.get_preprocess_db(configs_proc, group_by_proc, logger)

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
        try:
            # load and process the axis manager from the init config
            aman = sp_com.load_preprocess_tod(obs_id=obs_id, dets={gb:gg for gb, gg in zip(group_by_proc, group)},
                                              configs=configs_init, context=context_init)

            # tags from context proc
            tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            if "tags" in aman._fields:
                aman.move("tags", None)
            aman.wrap('tags', tags_proc)

            # now run the pipeline on the processed axis manager
            logger.info(f"Beginning processing pipeline for {obs_id}:{group}")
            proc_aman, success = pipe_proc.run(aman)

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

        # get the destination files
        policy = sp_util.ArchivePolicy.from_params(configs_proc['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)

        for gb, g in zip(group_by_proc, group):
            if gb == 'detset':
                dest_dataset += "_" + g
            else:
                dest_dataset += "_" + gb + "_" + str(g)
        logger.info(f"Saving data to {dest_file}:{dest_dataset}")
        proc_aman.save(dest_file, dest_dataset, overwrite)

        # Collect index info.
        db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
        for gb, g in zip(group_by_proc, group):
            db_data['dets:'+gb] = g
        if run_parallel:
            outputs.append(db_data)
        else:
            logger.info(f"Saving to database under {db_data}")
            if len(db.inspect(db_data)) == 0:
                h5_path = os.path.relpath(dest_file,
                        start=os.path.dirname(configs_proc['archive']['index']))
                db.add_entry(db_data, h5_path)

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
            return error, None, [obs_id, 'all groups']
        else:
            logger.info('Returning data to futures')
            error = None
            return error, dest_file, outputs

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
         nproc: Optional[int] = 4):

    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    # Get configs and contexts
    configs_init, context_init = sp_util.get_preprocess_context(configs_init)
    configs_proc, context_proc = sp_util.get_preprocess_context(configs_proc)

    errlog_proc = os.path.join(os.path.dirname(configs_proc['archive']['index']),'errlog.txt')

    multiprocessing.set_start_method('spawn')

    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) - update_delay*86400

    if obs_id is not None:
        tot_query = f"obs_id=='{obs_id}'"
    else:
        tot_query = "and "
        if min_ctime is not None:
            tot_query += f"timestamp>={min_ctime} and "
        if max_ctime is not None:
            tot_query += f"timestamp<={max_ctime} and "
        if query is not None:
            tot_query += query + " and "
        tot_query = tot_query[4:-4]
        if tot_query=="":
            tot_query="1"

    if not(tags is None):
        for i, tag in enumerate(tags):
            tags[i] = tag.lower()
            if '=' not in tag:
                tags[i] += '=1'

    if planet_obs:
        obs_list = []
        for tag in tags:
            obs_list.extend(context_init.obsdb.query(tot_query, tags=[tag]))
    else:
        obs_list = context_init.obsdb.query(tot_query, tags=tags)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")

    # obs_id's to run for proc db
    run_list_proc = []

    # error if init db does not exist (maybe run preprocess_tod)
    if not os.path.exists(configs_init['archive']['index']):
        raise ValueError("First configuration file database does not exist")
    else:
        # get init db
        db_init = core.metadata.ManifestDb(configs_init['archive']['index'])
        
        # check if proc db exists
        db_proc_exists = False
        if not overwrite and os.path.exists(configs_proc['archive']['index']):
            db_proc_exists = True

        # get processing db if it exists
        if db_proc_exists:
            db_proc = core.metadata.ManifestDb(configs_proc['archive']['index'])

        # loop over obs_id's in obs_list
        for obs in obs_list:
            group_by_init, groups_init = sp_util.get_groups(obs["obs_id"], configs_init, context_init)
            group_by_proc, groups_proc = sp_util.get_groups(obs["obs_id"], configs_proc, context_proc)

            found_groups_init = []
            found_groups_proc = []

            # find groups in init db
            x_init = db_init.inspect({'obs:obs_id': obs["obs_id"]})
            [found_groups_init.append([a[f'dets:{gb}'] for gb in group_by_init]) for a in x_init]

            if db_proc_exists:
                # find groups in proc db
                x_proc = db_proc.inspect({'obs:obs_id': obs["obs_id"]})
                [found_groups_proc.append([a[f'dets:{gb}'] for gb in group_by_proc]) for a in x_proc]

            all_groups_proc = groups_proc.copy()
            # remove groups not in init db and in proc db
            for group in all_groups_proc:
                if 'NC' not in group:
                    if group not in found_groups_init:
                        groups_proc.remove(group)
                    elif group in found_groups_proc:
                        groups_proc.remove(group)
            run_list_proc.append( (obs, groups_proc ))

        logger.info(f'Run list created with {len(run_list_proc)} obsids')

        nfile = 0
        folder = os.path.dirname(configs_proc['archive']['policy']['filename'])
        basename = os.path.splitext(configs_proc['archive']['policy']['filename'])[0]
        dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
        if not(os.path.exists(folder)):
                os.makedirs(folder)
        while os.path.exists(dest_file) and os.path.getsize(dest_file) > 10e9:
            nfile += 1
            dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'

        logger.info(f'Starting dest_file set to {dest_file}')

        # run write_block obs-ids in parallel at once then write all to the sqlite db.
        with ProcessPoolExecutor(nproc) as exe:
            futures = [exe.submit(multilayer_preprocess_tod, obs_id=r[0]['obs_id'],
                        group_list=r[1], verbosity=verbosity,
                        configs_init=configs_init, 
                        configs_proc=sp_util.swap_archive(configs_proc, f'temp/{r[0]["obs_id"]}.h5'),
                        overwrite=overwrite, run_parallel=True) for r in run_list_proc]
            for future in as_completed(futures):
                logger.info('New future as_completed result')
                try:
                    err, src_file, db_datasets = future.result()
                except Exception as e:
                    errmsg = f'{type(e)}: {e}'
                    tb = ''.join(traceback.format_tb(e.__traceback__))
                    logger.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
                    f = open(errlog_proc, 'a')
                    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                    f.close()
                    continue
                futures.remove(future)

                logger.info(f'Processing future result db_dataset: {db_datasets}')
                db = sp_util.get_preprocess_db(configs_proc, group_by_proc, logger)

                logger.info('Database connected')
                if os.path.exists(dest_file) and os.path.getsize(dest_file) >= 10e9:
                    nfile += 1
                    dest_file = basename + '_'+str(nfile).zfill(3)+'.h5'
                    logger.info('Starting a new h5 file.')

                h5_path = os.path.relpath(dest_file,
                                start=os.path.dirname(configs_proc['archive']['index']))

                if err is None:
                    logger.info(f'Moving files from temp to final destination.')
                    with h5py.File(dest_file,'a') as f_dest:
                        with h5py.File(src_file,'r') as f_src:
                            for dts in f_src.keys():
                                f_src.copy(f_src[f'{dts}'], f_dest, f'{dts}')
                                for member in f_src[dts]:
                                    if isinstance(f_src[f'{dts}/{member}'], h5py.Dataset):
                                        f_src.copy(f_src[f'{dts}/{member}'], f_dest[f'{dts}'], f'{dts}/{member}')
                    for db_data in db_datasets:
                        logger.info(f"Saving to database under {db_data}")
                        if len(db.inspect(db_data)) == 0:
                            db.add_entry(db_data, h5_path)
                    logger.info(f'Deleting {src_file}.')
                    os.remove(src_file)
                else:
                    logger.info(f'Writing {db_datasets[0]} to error log')
                    f = open(errlog_proc, 'a')
                    f.write(f'\n{time.time()}, {err}, {db_datasets[0]}\n{db_datasets[1]}\n')
                    f.close()

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)