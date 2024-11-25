import os
import logging
import time
import sys
import copy
import yaml
import numpy as np
import h5py
import traceback

from .. import core

from . import _Preprocess, Pipeline, processes

class ArchivePolicy:
    """Storage policy assistance.  Helps to determine the HDF5
    filename and dataset name for a result.

    Make me better!

    """
    @staticmethod
    def from_params(params):
        if params['type'] == 'simple':
            return ArchivePolicy(**params)
        if params['type'] == 'directory':
            return DirectoryArchivePolicy(**params)
        raise ValueError('No handler for "type"="%s"' % params['type'])

    def __init__(self, **kwargs):
        self.filename = kwargs['filename']

    def get_dest(self, product_id):
        """Returns (hdf_filename, dataset_addr).

        """
        return self.filename, product_id


class DirectoryArchivePolicy:
    """Storage policy for stuff organized directly on the filesystem.

    """
    def __init__(self, **kwargs):
        self.root_dir = kwargs['root_dir']
        self.pattern = kwargs['pattern']

    def get_dest(self, **kw):
        """Returns full path to destination directory.

        """
        return os.path.join(self.root_dir, self.pattern.format(**kw))


class _ReltimeFormatter(logging.Formatter):
    def __init__(self, *args, t0=None, **kw):
        super().__init__(*args, **kw)
        if t0 is None:
            t0 = time.time()
        self.start_time = t0

    def formatTime(self, record, datefmt=None):
        if datefmt is None:
            datefmt = '%8.3f'
        return datefmt % (record.created - self.start_time)


def init_logger(name, announce='', verbosity=2):
    """Configure and return a logger for site_pipeline elements.  It is
    disconnected from general sotodlib (propagate=False) and displays
    relative instead of absolute timestamps.

    """
    logger = logging.getLogger(name)

    if verbosity == 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.WARNING
    elif verbosity == 2:
        level = logging.INFO
    elif verbosity == 3:
        level = logging.DEBUG

    # add handler only if it doesn't exist
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler(sys.stdout)
        formatter = _ReltimeFormatter('%(asctime)s: %(message)s (%(levelname)s)')

        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        i, r = formatter.start_time // 1, formatter.start_time % 1
        text = (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(i))
              + (',%03d' % (r*1000)))
        logger.info(f'{announce}Log timestamps are relative to {text}')
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
                break

    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    return logger


def get_preprocess_context(configs, context=None):
    """Load the provided config file and context file. To be used in
    ``preprocess_*.py`` site pipeline scripts. If the provided context
    file does not have a metadata entry for preprocess then one will
    be added based on the definition in the config file.

    Parameters
    ----------
    configs : str or dict
        The configuration file or dictionary.
    context : str or core.Context, optional
        The context to use. If None, it is created from the configuration file.

    Returns
    -------
    configs : dict
        The configuration dictionary.
    context : core.Context
        The context file.
    """
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
        if key.get("name") == "preprocess":
            found=True
            break
    if not found:
        context["metadata"].append(
            {
                "db" : configs["archive"]["index"],
                "name" : "preprocess"
            }
        )
    return configs, context


def get_groups(obs_id, configs, context):
    """Get subobs group method and groups. To be used in
    ``preprocess_*.py`` site pipeline scripts.

    Parameters
    ----------
    obs_id : str
        The obsid.
    configs : dict
        The configuration dictionary.
    context : core.Context
        The Context file to use.

    Returns
    -------
    group_by : list of str
        The list of keys used to group the detectors.
    groups : list of list of int
        The list of groups of detectors.
    """
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


def get_preprocess_db(configs, group_by, logger=None):
    """Get or create a ManifestDb found for a given
    config.

    Arguments
    ----------
    configs : dict
        The configuration dictionary.
    group_by : list of str
        The list of keys used to group the detectors.
    logger : PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.

    Returns
    -------
    db : ManifestDb
        ManifestDb object
    """

    if logger is None:
        logger = init_logger("preprocess_db")

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
    """Update the configuration archive policy filename,
    create an output archive directory if it doesn't exist,
    and return a copy of the config.

    Arguments
    ----------
    configs : dict
        The configuration dictionary.
    fpath : str
        The archive policy filename to write to.

    Returns
    -------
    tc : dict
        Copy of the configuration file with an updated archive policy filename
    """
    tc = copy.deepcopy(config)
    tc['archive']['policy']['filename'] = os.path.join(os.path.dirname(tc['archive']['policy']['filename']), fpath)
    dname = os.path.dirname(tc['archive']['policy']['filename'])
    if not(os.path.exists(dname)):
        os.makedirs(dname)
    return tc


def load_preprocess_det_select(obs_id, configs, context=None,
                               dets=None, meta=None, logger=None):
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
    logger : PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.
    """
    
    if logger is None:
        logger = init_logger("preprocess")

    configs, context = get_preprocess_context(configs, context)
    pipe = Pipeline(configs["process_pipe"], logger=logger)

    meta = context.get_meta(obs_id, dets=dets, meta=meta)
    logger.info(f"Cutting on the last process: {pipe[-1].name}")
    pipe[-1].select(meta)
    return meta

def load_and_preprocess(obs_id, configs, context=None, dets=None, meta=None,
                        no_signal=None, logger=None):
    """ Loads the saved information from the preprocessing pipeline and runs
    the processing section of the pipeline.

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
    logger : PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.
    """
    
    if logger is None:
        logger = init_logger("preprocess")
    
    configs, context = get_preprocess_context(configs, context)
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
    calling the ``load_and_preprocess`` function. If the db entry does not
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
    if logger is None:
        logger = init_logger("preprocess")
    
    error = None
    outputs = {}

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups = get_groups(obs_id, configs, context)
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
        aman = load_and_preprocess(obs_id=obs_id, dets=dets, configs=configs, context=context)
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
        policy = ArchivePolicy.from_params(temp_config['archive']['policy'])
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


def cleanup_mandb(error, outputs, configs, logger=None):
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
    if logger is None:
        logger = init_logger("preprocess")
    
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
        db = get_preprocess_db(configs, group_by, logger)
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
