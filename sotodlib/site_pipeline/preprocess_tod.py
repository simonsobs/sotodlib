import os
import yaml
import time
import numpy as np
import argparse
import traceback
from typing import Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import copy

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
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
    group_by, groups = _get_groups(obs_id, configs, context)
    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)
    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")
        proc_aman = core.AxisManager(core.LabelAxis('dets', ['det%i' % i for i in range(3)]),
                                     core.OffsetAxis('samps', 1000))
        proc_aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')
        proc_aman.wrap_new('timestamps', ('samps',))[:] = (np.arange(proc_aman.samps.count) / 200)
        policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        for gb, g in zip(group_by, group):
            if gb == 'detset':
                dest_dataset += "_" + g
            else:
                dest_dataset += "_" + gb + "_" + str(g)
        logger.info(f"Saving data to {dest_file}:{dest_dataset}")
        proc_aman.save(dest_file, dest_dataset, overwrite)        
        
        # Collect index info.
        db_data = {'obs:obs_id': obs_id,
                   'dataset': dest_dataset}
        for gb, g in zip(group_by, group):
            db_data['dets:'+gb] = g
        if run_parallel:
            outputs.append(db_data)
    if run_parallel:
        return error, dest_file, outputs

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

def preproc_or_load_mpi(obs_id, configs, dets, logger=None, 
                        context=None, overwrite=False):
    """
    Function to be dropped into the ML mapmaker run in an MPI framework.
    This function is expected to receive a single obs_id, and dets dictionary
    which is expected to be processed by a single MPI rank. The dets dictionary
    must match the grouping specified in the preprocess config file. If the
    preprocess database entry for this obsid-dets group already exists then this
    function will just load back the processed tod calling the ``load_preprocess_tod``
    function. If the db entry does not exist of the overwrite flag is set to True then
    the full preprocessing steps defined in the configs are run and the outputs are
    written to a unique h5 file. Any errors, the info to populate the database, 
    the file path of the h5 file, and the process tod are returned from this function.
    This function is expected to be run in conjunction with the ``cleanup_mandb_mpi``
    function which consumes all of the outputs (except the processed tod) and writes
    to the database, and moves the multiple h5 files into fewer h5 files (each <= 10 GB).

    Arguments
    ---------
    obs_id: str
        Obs id to process or load
    configs: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets: dict
        Dictionary specifying which detectors/wafers to load see ``Context.obsdb.get_obs``.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.
    context: fpath or core.Context
        Optional. Filepath or context object used for data loading/querying.
    overwrite: bool
        Optional. Whether or not to overwrite existing entries in the preprocess manifest db.

    Returns
    -------
    error: str
        String indicating if the function succeeded in its execution or failed.
        If ``None`` then it succeeded in processing and the mandB should be updated.
        If ``'load_success'`` then axis manager was successfully loaded from existing preproc db.
        If any other string then processing failed and output will be logged in the error log.
    output: list
        Varies depending on the value of ``error``.
        If ``error == None`` then output is the info needed to update the manifest db.
        If ``error == 'load_success'`` then output is just ``[obs_id, dets]``.
        If ``error`` is anything else then output stores what to save in the error log.
    aman: Core.AxisManager
        Processed axis manager only returned if ``error`` is ``None`` or ``'load_success'``.
    """
    error = None
    outputs = {}
    if logger is None: 
        logger = sp_util.init_logger("preprocess")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups = _get_groups(obs_id, configs, context)
    all_groups = groups.copy()
    cur_groups = [list(np.fromiter(dets.values(), dtype='<U32'))]
    for g in all_groups:
        if g not in cur_groups:
            groups.remove(g)

        if len(groups) == 0:
            logger.warning(f"group_list:{cur_groups} contains no overlap with "
                           f"groups in observation: {obs_id}:{cur_groups}. "
                           f"No analysis to run.")
            error = 'no_group_overlap'
            return error, [obs_id, dets], None

    dbexist = True
    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])
        dbix = {'obs:obs_id':obs_id}
        for gb, g in zip(group_by, cur_groups[0]):
            dbix[f'dets:{gb}'] = g
        print(dbix)
        if len(db.inspect(dbix)) == 0:
            dbexist = False
    else:
        dbexist = False

    if dbexist and (not overwrite):
        logger.info(f"db exists for {obs_id} {dets} loading data and applying preprocessing.")
        aman = load_preprocess_tod(obs_id=obs_id, dets=dets,
                                   configs=configs, context=context)
        error = 'load_success'
        return error, [obs_id, dets], aman
    else:
        logger.info(f"Generating new preproc db entry for {obs_id} {dets}")
        pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)
        try:
            aman = context.get_obs(obs_id, dets=dets)
            tags = np.array(context.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            aman.wrap('tags', tags)
            proc_aman, success = pipe.run(aman)
            aman.wrap('preprocess', proc_aman)
        except Exception as e:
            error = f'Failed to load: {obs_id} {dets}'
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"{error}\n{errmsg}\n{tb}")
            return error, [errmsg, tb], None
        if success != 'end':
            # If a single group fails we don't log anywhere just mis an entry in the db.
            return success, [obs_id, dets], None
        newpath = f'temp/{obs_id}'
        for cg in cur_groups[0]:
            newpath += f'_{cg}'
        temp_config = swap_archive(configs, newpath+'.h5')
        policy = sp_util.ArchivePolicy.from_params(temp_config['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        for gb, g in zip(group_by, cur_groups[0]):
            if gb == 'detset':
                dest_dataset += "_" + g
            else:
                dest_dataset += "_" + gb + "_" + str(g)

        proc_aman.save(dest_file, dest_dataset, overwrite)
        # Collect info for saving h5 file.
        outputs['temp_file'] = dest_file
        
        # Collect index info.
        db_data = {'obs:obs_id': obs_id,
                    'dataset': dest_dataset}
        for gb, g in zip(group_by, cur_groups[0]):
            db_data['dets:'+gb] = g
        outputs['db_data'] = db_data
        return error, outputs, aman
          
def cleanup_mandb_mpi(error, outputs, configs, logger):
    """
    Function to update the manifest db when database generation run in parrallel using MPI.
    This function is expected to be run from rank 0 after a ``comm.gather`` call and consumes
    the outputs from ``preproc_or_load_mpi``. See the ``preproc_or_load_mpi`` docstring for
    the varying expected values of ``error`` and the associated ``outputs``. This function
    will either: 
    
    1) Update the mandb sqlite file and move the h5 archive from its temporary
    location to its permanent path if error is ``None``.
    
    2) Return nothing if error is ``load_success``.
    
    3) Update the error log if error is anything else.
    """
    if error is None:
        # Expects archive policy filename to be <path>/<filename>.h5 and then this adds
        # <path>/<filename>_<xxx>.h5 where xxx is a number that increments up from 0 
        # whenever the file size exceeds 10 GB.
        nfile = 0
        folder = os.path.dirname(configs['archive']['policy']['filename'])
        basename = os.path.splitext(configs['archive']['policy']['filename'])[0]
        dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
        if not(os.path.exists(folder)):
                os.makedirs(folder)
        while os.path.exists(dest_file) and os.path.getsize(dest_file) > 10e9:
            nfile += 1
            dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
        group_by =  [k.split(':')[-1] for k in outputs['db_data'].keys() if 'dets' in k]
        db = _get_preprocess_db(configs, group_by)
        h5_path = os.path.relpath(dest_file,
                                  start=os.path.dirname(configs['archive']['index']))

        src_file = outputs['temp_file']
        with h5py.File(dest_file,'a') as f_dest:
            with h5py.File(src_file,'r') as f_src:
                for dts in f_src.keys():
                    f_src.copy(f_src[f'{dts}'], f_dest, f'{dts}')
                    for member in f_src[dts]:
                        if isinstance(f_src[f'{dts}/{member}'], h5py.Dataset):
                            f_src.copy(f_src[f'{dts}/{member}'], f_dest[f'{dts}'], f'{dts}/{member}')
        logger.info(f"Saving to database under {outputs['db_data']}")
        if len(db.inspect(outputs['db_data'])) == 0:
            db.add_entry(outputs['db_data'], h5_path)
        os.remove(src_file)
    elif error == 'load_success':
        return
    else:
        folder = os.path.dirname(configs['archive']['index'])
        if not(os.path.exists(folder)):
            os.makedirs(folder)
        errlog = os.path.join(folder, 'errlog.txt')
        f = open(errlog, 'a')
        f.write(f'{time.time()}, {error}\n')
        f.write(f'\t{outputs[0]}\n\t{outputs[1]}\n')
        f.close()

def preprocess_tod(obs_id, 
                    configs, 
                    logger,
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
    logger: logging instance
        the logger to print to
    run_parallel: Bool
        If true preprocess_tod is called in a parallel process which returns
        dB info and errors and does no sqlite writing inside the function.
    """
    error = None
    outputs = []
    if logger is None: 
        logger = sp_util.init_logger("preprocess")
    
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups = _get_groups(obs_id, configs, context)
    all_groups = groups.copy()
    if group_list is not None:
        for g in all_groups:
            if g not in group_list:
                groups.remove(g)

        if len(groups) == 0:
            logger.warning(f"group_list:{group_list} contains no overlap with "
                        f"groups in observation: {obs_id}:{all_groups}. "
                        f"No analysis to run.")
            error = 'no_group_overlap'
            if run_parallel:
                return error, None, [None, None]
            else:
                return
    
    if not(run_parallel):
        db = _get_preprocess_db(configs, group_by)
    
    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)

    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")
        try:
            aman = context.get_obs(obs_id, dets={gb:g for gb, g in zip(group_by, group)})
            tags = np.array(context.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            aman.wrap('tags', tags)
            proc_aman, success = pipe.run(aman)
        except Exception as e:
            error = f'{obs_id} {group}'
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"{error}\n{errmsg}\n{tb}")
            return error, None, [errmsg, tb]
        if success != 'end':
            # If a single group fails we don't log anywhere just mis an entry in the db.
            continue

        policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        for gb, g in zip(group_by, group):
            if gb == 'detset':
                dest_dataset += "_" + g
            else:
                dest_dataset += "_" + gb + "_" + str(g)
        logger.info(f"Saving data to {dest_file}:{dest_dataset}")
        proc_aman.save(dest_file, dest_dataset, overwrite)

        # Collect index info.
        db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
        for gb, g in zip(group_by, group):
            db_data['dets:'+gb] = g
        if run_parallel:
            outputs.append(db_data)
        else:
            logger.info(f"Saving to database under {db_data}")
            if len(db.inspect(db_data)) == 0:
                h5_path = os.path.relpath(dest_file,
                        start=os.path.dirname(configs['archive']['index']))
                db.add_entry(db_data, h5_path)
    if run_parallel:
        return error, dest_file, outputs        

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
        # select applied in load_preprocess_det_select
        pipe.run(aman, aman.preprocess, select=False)
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
    return parser

def main(
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
        nproc: Optional[int] = 4
 ):
    configs, context = _get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    errlog = os.path.join(os.path.dirname(configs['archive']['index']),
                          'errlog.txt')

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
            obs_list.extend(context.obsdb.query(tot_query, tags=[tag]))
    else:
        obs_list = context.obsdb.query(tot_query, tags=tags)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = [ (o,None) for o in obs_list]
        group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            group_by, groups = _get_groups(obs["obs_id"], configs, context)
            if x is None or len(x) == 0:
                run_list.append( (obs, None) )
            elif len(x) != len(groups):
                [groups.remove([a[f'dets:{gb}'] for gb in group_by]) for a in x]
                run_list.append( (obs, groups) )

    # Expects archive policy filename to be <path>/<filename>.h5 and then this adds
    # <path>/<filename>_<xxx>.h5 where xxx is a number that increments up from 0 
    # whenever the file size exceeds 10 GB.
    nfile = 0
    folder = os.path.dirname(configs['archive']['policy']['filename'])
    basename = os.path.splitext(configs['archive']['policy']['filename'])[0]
    dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
    if not(os.path.exists(folder)):
            os.makedirs(folder)
    while os.path.exists(dest_file) and os.path.getsize(dest_file) > 10e9:
        nfile += 1
        dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'

    # Run write_block obs-ids in parallel at once then write all to the sqlite db.
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(preprocess_tod, obs_id=r[0]['obs_id'],
                     group_list=r[1], logger=logger,
                     configs=swap_archive(configs, f'temp/{r[0]["obs_id"]}.h5'),
                     overwrite=overwrite, run_parallel=True) for r in run_list]
        for future in as_completed(futures):
            err, src_file, db_datasets = future.result()
            db = _get_preprocess_db(configs, group_by)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) >= 10e9:
                nfile += 1
                dest_file = basename + '_'+str(nfile).zfill(3)+'.h5'

            h5_path = os.path.relpath(dest_file,
                            start=os.path.dirname(configs['archive']['index']))

            if err is None:
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
                os.remove(src_file)
            else:
                f = open(errlog, 'a')
                f.write(f'{time.time()}, {err}, {db_datasets[0]}\n{db_datasets[1]}')
                f.close()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    sp_util.main_launcher(main, get_parser)
