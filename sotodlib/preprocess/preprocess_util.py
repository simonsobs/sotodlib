import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")

def load_preprocess_det_select(obs_id, configs, context=None,
                               dets=None, meta=None):
    """Loads the metadata information for the Observation and runs through any
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
    configs, context = sp_util.get_preprocess_context(configs, context)
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    
    meta = context.get_meta(obs_id, dets=dets, meta=meta)
    logger.info(f"Cutting on the last process: {pipe[-1].name}")
    pipe[-1].select(meta)
    return meta

def load_preprocess(obs_id, configs, context=None, dets=None, meta=None,
                    no_signal=None):
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
    no_signal: bool 
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    """
    configs, context = sp_util.get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context,
                                      dets=dets, meta=meta)

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta, no_signal=no_signal)
        pipe.run(aman, aman.preprocess)
        return aman
    
def preproc_or_load_group(obs_id, configs, dets, logger=None, 
                          context=None, overwrite=False):
    """
    This function is expected to receive a single obs_id, and dets dictionary.
    The dets dictionary must match the grouping specified in the preprocess
    config file. If the preprocess database entry for this obsid-dets group
    already exists then this function will just load back the processed tod
    calling the ``load_preprocess`` function. If the db entry does not
    exist of the overwrite flag is set to True then the full preprocessing
    steps defined in the configs are run and the outputs are written to a
    unique h5 file. Any errors, the info to populate the database, the file
    path of the h5 file, and the process tod are returned from this function.
    This function is expected to be run in conjunction with the
    ``cleanup_mandb`` function which consumes all of the outputs (except the
    processed tod), writes to the database, and moves the multiple h5 files
    into fewer h5 files (each <= 10 GB).

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
    group_by, groups = sp_util.get_groups(obs_id, configs, context)
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
        aman = load_preprocess(obs_id=obs_id, dets=dets, configs=configs, context=context)
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
        temp_config = sp_util.swap_archive(configs, newpath+'.h5')
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

def cleanup_mandb(error, outputs, configs, logger):
    """
    Function to update the manifest db when data is collected from the
    ``preproc_or_load_group`` function. If used in an mpi framework this
    function is expected to be run from rank 0 after a ``comm.gather``.
    See the ``preproc_or_load_group`` docstring for the varying expected
    values of ``error`` and the associated ``outputs``. This function will
    either: 
    
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
        group_by = [k.split(':')[-1] for k in outputs['db_data'].keys() if 'dets' in k]
        db = sp_util.get_preprocess_db(configs, group_by, logger)
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
