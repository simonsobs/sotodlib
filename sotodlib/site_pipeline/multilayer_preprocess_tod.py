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
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")

def _get_preprocess_context(configs, context=None):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    
    if context is None:
        context = core.Context(configs["context_file"])
        
    if type(context) == str:
        context = core.Context(context)
    
    # if context doesn't have the preprocess archive it in add it
    # allows us to use same context before and after calculations
    found=False
    if context.get("metadata") is None:
        context["metadata"] = []

    for key in context.get("metadata"):
        if key.get("unpack") == "preprocess":
            found=True
            break
    if not found:
        context["metadata"].append( 
            {
                "db" : configs["archive"]["index"],
                "unpack" : "preprocess"
            }
        )
    return configs, context

def _get_groups(obs_id, configs, context):
    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    for i, gb in enumerate(group_by):
        if gb.startswith('dets:'):
            group_by[i] = gb.split(':',1)[1]

        if (gb == 'detset') and (len(group_by) == 1):
            groups = context.obsfiledb.get_detsets(obs_id)
            return group_by, [[g] for g in groups]
        
    det_info = context.get_det_info(obs_id)
    rs = det_info.subset(keys=group_by).distinct()
    groups = [[b for a,b in r.items()] for r in rs]
    return group_by, groups

def _get_preprocess_db(configs, group_by):
    if os.path.exists(configs['archive']['index']):
        logger.info(f"Mapping {configs['archive']['index']} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(configs['archive']['index'])
    else:
        logger.info(f"Creating {configs['archive']['index']} for the "
                     "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        for gb in group_by:
            scheme.add_exact_match('dets:' + gb)
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            configs['archive']['index'],
            scheme=scheme
        )
    return db

def swap_archive(config, fpath):
    tc = copy.deepcopy(config)
    tc['archive']['policy']['filename'] = os.path.join(os.path.dirname(tc['archive']['policy']['filename']), fpath)
    dname = os.path.dirname(tc['archive']['policy']['filename'])
    if not(os.path.exists(dname)):
        os.makedirs(dname)
    return tc

def load_preprocess_det_select(obs_id, configs, context=None,
                               dets=None, meta=None):
    """ Loads the metadata information for the Observation and runs through any
    data selection specified by the Preprocessing Pipeline.

    Arguments
    ----------
    obs_id: multiple
        passed to `context.get_obs` to load AxisManager, see Notes for 
        `context.get_obs`
    configs: string or dictionary
        config file or loaded config directory
    dets: dict
        dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    """
    configs, context = _get_preprocess_context(configs, context)
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    
    meta = context.get_meta(obs_id, dets=dets, meta=meta)
    logger.info(f"Cutting on the last process: {pipe[-1].name}")
    pipe[-1].select(meta)
    return meta

def load_preprocess_tod(obs_id, configs="preprocess_configs.yaml",
                        context=None, dets=None, meta=None):
    """ Loads the saved information from the preprocessing pipeline and runs the
    processing section of the pipeline. 

    Assumes preprocess_tod has already been run on the requested observation. 
    
    Arguments
    ----------
    obs_id: multiple
        passed to `context.get_obs` to load AxisManager, see Notes for 
        `context.get_obs`
    configs: string or dictionary
        config file or loaded config directory
    dets: dict
        dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    """
    configs, context = _get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context, dets=dets, meta=meta)

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta)
        pipe.run(aman, aman.preprocess)
        return aman

def multilayer_preprocess_tod(obs_id, 
                              configs, 
                              verbosity=0,
                              group_list=None, 
                              overwrite=False,
                              run_parallel=False):
    
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)
    
    outputs = []
    
    if not isinstance(configs, list):
        raise ValueError("configs input should be a list")
    
    if len(configs) < 2:
        raise ValueError("at least 2 config files should be provided")
    
    # first config
    if type(configs[0]) == str:
        configs0 = yaml.safe_load(open(configs[0], "r"))
    else:
        configs0 = configs[0]
    
    # first context
    context0 = core.Context(configs0["context_file"])
    
    # second config
    if type(configs[1]) == str:
        configs1 = yaml.safe_load(open(configs[1], "r"))
    else:
        configs1 = configs[1]
    
    # second context
    context1 = core.Context(configs1["context_file"])
    
    # get groups1
    group_by1, groups1 = _get_groups(obs_id, configs1, context1)
    all_groups1 = groups1.copy()
    for g in all_groups1:
        if group_list is not None:
            if g not in group_list:
                groups1.remove(g)
                continue
        if 'wafer.bandpass' in group_by1:
            if 'NC' in g:
                groups1.remove(g)
                continue
        try:
            meta = context0.get_meta(obs_id, dets = {gb:gg for gb, gg in zip(group_by1, g)})
            print(meta)
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {g}\n{errmsg}\n{tb}")
            groups1.remove(g)
            continue
        if meta.dets.count == 0:
            groups1.remove(g)
    
    # if no groups1 remain
    if len(groups1) == 0:
        logger.warning(f"group_list:{group_list} contains no overlap with "
                       f"groups in observation: {obs_id}:{all_groups1}. "
                       f"No analysis to run.")
        error = 'no_group_overlap'
        if run_parallel:
            return error, None, [None, None]
        else:
            return

    if not(run_parallel):
        db = _get_preprocess_db(configs1, group_by1)
        
    # pipeline for config1
    pipe = Pipeline(configs1["process_pipe"], plot_dir=configs1["plot_dir"], logger=logger)
    
    if configs1.get("lmsi_config", None) is not None:
        make_lmsi = True
    else:
        make_lmsi = False
    
    # loop through and reduce each group
    n_fail = 0
    for group in groups1:
        logger.info(f"Beginning run for {obs_id}:{group}")
        try:
            
            aman = load_preprocess_tod(obs_id=obs_id, dets={gb:gg for gb, gg in zip(group_by1, group)},
                                       configs=configs0, context=context0)
            
            logger.info(f"Beginning second pipeline for {obs_id}:{group}")
            proc_aman, success = pipe.run(aman)

            if make_lmsi:
                new_plots = os.path.join(configs1["plot_dir"],
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

        policy = sp_util.ArchivePolicy.from_params(configs1['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        for gb, g in zip(group_by1, group):
            if gb == 'detset':
                dest_dataset += "_" + g
            else:
                dest_dataset += "_" + gb + "_" + str(g)
        logger.info(f"Saving data to {dest_file}:{dest_dataset}")
        proc_aman.save(dest_file, dest_dataset, overwrite)

        # Collect index info.
        db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
        for gb, g in zip(group_by1, group):
            db_data['dets:'+gb] = g
        if run_parallel:
            outputs.append(db_data)
        else:
            logger.info(f"Saving to database under {db_data}")
            if len(db.inspect(db_data)) == 0:
                h5_path = os.path.relpath(dest_file,
                        start=os.path.dirname(configs1['archive']['index']))
                db.add_entry(db_data, h5_path)

    if make_lmsi:
        from pathlib import Path
        import lmsi.core as lmsi

        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs1["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))
    
    if run_parallel:
        if n_fail == len(groups1):
            # If no groups1 make it to the end of the processing return error.
            logger.info(f'ERROR: all groups1 failed for {obs_id}')
            error = 'all_fail'
            return error, None, [obs_id, 'all groups1']
        else:
            logger.info('Returning data to futures')
            error = None
            return error, dest_file, outputs
def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+', help="List of Preprocessing Configuration Files")
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

def main(
        configs: list,
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
    
    if not isinstance(configs, list):
        raise ValueError("Config input should be a list")
    
    if len(configs) < 2:
        raise ValueError("At least 2 configuration files should be provided")
    
    configs0, context0 = _get_preprocess_context(configs[0])
    configs1, context1 = _get_preprocess_context(configs[1])
    
    errlog0 = os.path.join(os.path.dirname(configs0['archive']['index']),
                          'errlog.txt')
    errlog1 = os.path.join(os.path.dirname(configs1['archive']['index']),
                          'errlog.txt')
    
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
            obs_list.extend(context0.obsdb.query(tot_query, tags=[tag]))
    else:
        obs_list = context0.obsdb.query(tot_query, tags=tags)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list0 = []
    run_list1 = []
    
    # error if config0 db does not exist
    if not os.path.exists(configs0['archive']['index']):
        raise ValueError("First configuration file database does not exist")
    # if db0 exists
    else:
        # get db0
        db0 = core.metadata.ManifestDb(configs0['archive']['index'])
        dbexists = False
        # if db1 exists
        if not overwrite and os.path.exists(configs1['archive']['index']):
            db1 = core.metadata.ManifestDb(configs1['archive']['index'])
            dbexists = True
        # loop over obs
        for obs in obs_list:
            # get obs from db0
            x0 = db0.inspect({'obs:obs_id': obs["obs_id"]})
            group_by0, groups0 = _get_groups(obs["obs_id"], configs0, context0)
            group_by1, groups1 = _get_groups(obs["obs_id"], configs1, context1)

            # if obs found in x0 and not empty
            if x0 is not None and len(x0) != 0:
                if len(x0) == len(groups0):
                    run_list0.append((obs, None))
                # if mismatch between db0 obs and groups0
                elif len(x0) != len(groups0):
                    [groups0.remove([a[f'dets:{gb}'] for gb in group_by0]) for a in x0]
                    run_list0.append( (obs, groups0) )
                # if db1 exists
                if dbexists:
                    # get obs from db1
                    x1 = db1.inspect({'obs:obs_id': obs["obs_id"]})
                     # if obs not in x0 or empty
                    if x1 is None or len(x1) == 0:
                        all_groups = groups1.copy()
                        for g in all_groups:
                            if 'wafer.bandpass' in group_by1:
                                if g in groups0 and 'NC' not in g:
                                    groups1.remove(g)
                                    continue
                        
                        run_list1.append( (obs, groups1) )
                    elif len(x1) != len(groups1):
                        [groups1.remove([a[f'dets:{gb}'] for gb in group_by1]) for a in x1]
                        
                        all_groups = groups1.copy()
                        for g in all_groups:
                            if 'wafer.bandpass' in group_by1:
                                if g in groups0 and 'NC' not in g:
                                    groups1.remove(g)
                                    continue
                        run_list1.append( (obs, groups1) )
                else:
                    all_groups = groups1.copy()
                    for g in all_groups:
                        if 'wafer.bandpass' in group_by1:
                            if g in groups0 and 'NC' not in g:
                                groups1.remove(g)
                                continue
                    run_list1.append( (obs, groups1) )
    
    logger.info(f'Run list created with {len(run_list0)} obsids')
    logger.info(f'Run list created with {len(run_list1)} obsids')
    
    # Expects archive policy filename to be <path>/<filename>.h5 and then this adds
    # <path>/<filename>_<xxx>.h5 where xxx is a number that increments up from 0 
    # whenever the file size exceeds 10 GB.
    nfile = 0
    folder = os.path.dirname(configs1['archive']['policy']['filename'])
    basename = os.path.splitext(configs1['archive']['policy']['filename'])[0]
    dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
    if not(os.path.exists(folder)):
            os.makedirs(folder)
    while os.path.exists(dest_file) and os.path.getsize(dest_file) > 10e9:
        nfile += 1
        dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'

    logger.info(f'Starting dest_file set to {dest_file}')

    # Run write_block obs-ids in parallel at once then write all to the sqlite db.
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(multilayer_preprocess_tod, obs_id=r[0]['obs_id'],
                    group_list=r[1], verbosity=verbosity,
                    configs=[configs0, swap_archive(configs1, f'temp/{r[0]["obs_id"]}.h5')],
                    overwrite=overwrite, run_parallel=True) for r in run_list1]
        for future in as_completed(futures):
            logger.info('New future as_completed result')
            try:
                err, src_file, db_datasets = future.result()
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
                f = open(errlog1, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            logger.info(f'Processing future result db_dataset: {db_datasets}')
            db = _get_preprocess_db(configs1, group_by1)
            
            logger.info('Database connected')
            if os.path.exists(dest_file) and os.path.getsize(dest_file) >= 10e9:
                nfile += 1
                dest_file = basename + '_'+str(nfile).zfill(3)+'.h5'
                logger.info('Starting a new h5 file.')

            h5_path = os.path.relpath(dest_file,
                            start=os.path.dirname(configs1['archive']['index']))
            
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
                f = open(errlog1, 'a')
                f.write(f'\n{time.time()}, {err}, {db_datasets[0]}\n{db_datasets[1]}\n')
                f.close()

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)