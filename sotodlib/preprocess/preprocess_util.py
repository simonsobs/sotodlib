import os
import logging
import time
import sys
import copy
import yaml
import numpy as np
import h5py
import traceback
import inspect
from sotodlib.hwp import hwp_angle_model
from sotodlib.coords import demod as demod_mm
from sotodlib.tod_ops import t2pleakage

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

    Arguments
    ----------
    name : str
        The name of the logger
    announce : str
        Initial message to be displayed after logger is instantiated.
    verbosity : int
        Level of logger output
        0: Error
        1: Warning
        2: Info
        3: Debug

    Returns
    -------
    logger : PythonLogger
        The initialized logger object
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

    Arguments
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
        if key.get("name") == "preprocess" or key.get("label") == "preprocess":
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

    Arguments
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

    try:
        group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
        for i, gb in enumerate(group_by):
            if gb.startswith('dets:'):
                group_by[i] = gb.split(':',1)[1]

            if (gb == 'detset') and (len(group_by) == 1):
                groups = context.obsfiledb.get_detsets(obs_id)
                return group_by, [[g] for g in groups], None

        det_info = context.get_det_info(obs_id)
        rs = det_info.subset(keys=group_by).distinct()
        groups = [[b for a,b in r.items()] for r in rs]
        return group_by, groups, None
    except Exception as e:
        error = f'Failed get groups for: {obs_id}'
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return [], [], [error, errmsg, tb]


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
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs: string or dictionary
        Config file or loaded config directory
    context: core.Context
        The Context file to use.
    dets: dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    logger: PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs, context = get_preprocess_context(configs, context)
    pipe = Pipeline(configs["process_pipe"], logger=logger)

    meta = context.get_meta(obs_id, dets=dets, meta=meta)
    logger.info("Restricting detectors on all processes")
    keep_all = np.ones(meta.dets.count,dtype=bool)
    for process in pipe[:]:
        keep = process.select(meta, in_place=False)
        if isinstance(keep, np.ndarray):
            keep_all &= keep
    meta.restrict("dets", meta.dets.vals[keep_all])
    return meta


def load_and_preprocess(obs_id, configs, context=None, dets=None, meta=None,
                        no_signal=None, logger=None):
    """Loads the saved information from the preprocessing pipeline and runs
    the processing section of the pipeline.

    Assumes preprocess_tod has already been run on the requested observation.

    Arguments
    ----------
    obs_id: multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs: string or dictionary
        Config file or loaded config directory
    context: core.Context
        Optional. The Context file to use.
    dets: dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal: bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger: PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs, context = get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context,
                                      dets=dets, meta=meta, logger=logger)

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta, no_signal=no_signal)
        pipe.run(aman, aman.preprocess, select=False)
        return aman


def multilayer_load_and_preprocess(obs_id, configs_init, configs_proc,
                                   dets=None, meta=None, no_signal=None,
                                   logger=None, init_only=False):
    """Loads the saved information from the preprocessing pipeline from a
    reference and a dependent database and runs the processing section of
    the pipeline for each.

    Assumes preprocess_tod and multilayer_preprocess_tod have already been run
    on the requested observation.

    Arguments
    ----------
    obs_id: multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs_init: string or dictionary
        Config file or loaded config directory
    configs_proc: string or dictionary
        Second config file or loaded config dictionary to load
        dependent databases generated using multilayer_preprocess_tod.py.
    dets: dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal: bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.
    init_only: bool
        Optional. If True, do not run the dependent pipeline.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs_init, context_init = get_preprocess_context(configs_init)
    meta_init = context_init.get_meta(obs_id, dets=dets, meta=meta)

    configs_proc, context_proc = get_preprocess_context(configs_proc)
    meta_proc = context_proc.get_meta(obs_id, dets=dets, meta=meta)

    group_by_init, groups_init, error_init = get_groups(obs_id, configs_init, context_init)
    group_by_proc, groups_proc, error_proc = get_groups(obs_id, configs_proc, context_proc)

    if error_init is not None:
        raise ValueError(f"{error_init[0]}\n{error_init[1]}\n{error_init[2]}")

    if error_proc is not None:
        raise ValueError(f"{error_proc[0]}\n{error_proc[1]}\n{error_proc[2]}")

    if (group_by_init != group_by_proc).any():
        raise ValueError('init and proc groups do not match')

    if meta_init.dets.count == 0 or meta_proc.dets.count == 0:
        logger.info(f"No detectors in obs {obs_id}")
        return None
    else:
        pipe_init = Pipeline(configs_init["process_pipe"], logger=logger)
        aman_cfgs_ref = get_pcfg_check_aman(pipe_init)

        if check_cfg_match(aman_cfgs_ref, meta_proc.preprocess['pcfg_ref'],
                           logger=logger):
            pipe_proc = Pipeline(configs_proc["process_pipe"], logger=logger)

            logger.info("Restricting detectors on all proc pipeline processes")
            keep_all = np.ones(meta_proc.dets.count, dtype=bool)
            for process in pipe_proc[:]:
                keep = process.select(meta_proc, in_place=False)
                if isinstance(keep, np.ndarray):
                    keep_all &= keep
            meta_proc.restrict("dets", meta_proc.dets.vals[keep_all])
            meta_init.restrict('dets', meta_proc.dets.vals)

            aman = context_init.get_obs(meta_init, no_signal=no_signal)
            logger.info("Running initial pipeline")
            pipe_init.run(aman, aman.preprocess, select=False)
            if init_only:
                return aman

            logger.info("Running dependent pipeline")
            proc_aman = context_proc.get_meta(obs_id, meta=aman)

            aman.preprocess.merge(proc_aman.preprocess)

            pipe_proc.run(aman, aman.preprocess, select=False)

            return aman
        else:
            raise ValueError('Dependency check between configs failed.')


def multilayer_load_and_preprocess_sim(obs_id, configs_init, configs_proc,
                                       sim_map, meta=None,
                                       logger=None, init_only=False,
                                       t2ptemplate_aman=None):
    """Loads the saved information from the preprocessing pipeline from a
    reference and a dependent database, loads the signal from a (simulated)
    map into the AxisManager and runs the processing section of the pipeline
    for both databases.

    Assumes preprocess_tod and multilayer_preprocess_tod have already been run
    on the requested observation.

    Arguments
    ----------
    obs_id: multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs_init: string or dictionary
        Config file or loaded config directory
    configs_proc: string or dictionary
        Second config file or loaded config dictionary to load
        dependent databases generated using multilayer_preprocess_tod.py.
    sim_map: numpy.ndmap or enmap.ndmap
        Input simulated map to be observed
    meta: AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal: bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.
    init_only: bool
        Optional. Whether or not to run the dependent pipeline.
    t2ptemplate_aman: AxisManager
        Optional. AxisManager to use as a template for t2p leakage
        deprojection.
    """
    if logger is None:
        logger = init_logger("preprocess")

    configs_init, context_init = get_preprocess_context(configs_init)
    meta_init = context_init.get_meta(obs_id, meta=meta)

    configs_proc, context_proc = get_preprocess_context(configs_proc)
    meta_proc = context_proc.get_meta(obs_id, meta=meta)

    group_by_init, groups_init, error_init = get_groups(obs_id, configs_init, context_init)
    group_by_proc, groups_proc, error_proc = get_groups(obs_id, configs_proc, context_proc)

    if error_init is not None:
        raise ValueError(f"{error_init[0]}\n{error_init[1]}\n{error_init[2]}")

    if error_proc is not None:
        raise ValueError(f"{error_proc[0]}\n{error_proc[1]}\n{error_proc[2]}")

    if (group_by_init != group_by_proc).any():
        raise ValueError('init and proc groups do not match')

    if meta_init.dets.count == 0 or meta_proc.dets.count == 0:
        logger.info(f"No detectors in obs {obs_id}")
        return None
    else:
        pipe_init = Pipeline(configs_init["process_pipe"], logger=logger)
        aman_cfgs_ref = get_pcfg_check_aman(pipe_init)

        if check_cfg_match(aman_cfgs_ref, meta_proc.preprocess['pcfg_ref'],
                           logger=logger):
            pipe_proc = Pipeline(configs_proc["process_pipe"], logger=logger)

            logger.info("Restricting detectors on all proc pipeline processes")
            keep_all = np.ones(meta_proc.dets.count, dtype=bool)
            for process in pipe_proc[:]:
                keep = process.select(meta_proc, in_place=False)
                if isinstance(keep, np.ndarray):
                    keep_all &= keep
            meta_proc.restrict("dets", meta_proc.dets.vals[keep_all])
            meta_init.restrict('dets', meta_proc.dets.vals)
            aman = context_init.get_obs(meta_proc, no_signal=True)
            aman = hwp_angle_model.apply_hwp_angle_model(aman)
            aman.move("signal", None)

            logger.info("Reading in simulated map")
            demod_mm.from_map(aman, sim_map, wrap=True, modulated=True)

            logger.info("Running initial pipeline")
            pipe_init.run(aman, aman.preprocess, sim=True)

            if init_only:
                return aman
            
            if t2ptemplate_aman is not None:
                # Replace Q,U with simulated timestreams
                t2ptemplate_aman.wrap("demodQ", aman.demodQ, [(0, 'dets'), (1, 'samps')], overwrite=True)
                t2ptemplate_aman.wrap("demodU", aman.demodU, [(0, 'dets'), (1, 'samps')], overwrite=True)

                t2p_aman = t2pleakage.get_t2p_coeffs(
                    t2ptemplate_aman,
                    merge_stats=False
                )
                t2pleakage.subtract_t2p(
                    aman,
                    t2p_aman,
                    T_signal=t2ptemplate_aman.dsT
                )

            logger.info("Running dependent pipeline")
            proc_aman = context_proc.get_meta(obs_id, meta=aman)
            aman.preprocess.merge(proc_aman.preprocess)
            pipe_proc.run(aman, aman.preprocess, sim=True)

            return aman
        else:
            raise ValueError('Dependency check between configs failed.')


def find_db(obs_id, configs, dets, context=None, logger=None):
    """This function checks if the manifest db from
    a config file exists and searches if it contains
    an entry for the provided Obs id and set of detectors.

    Arguments
    ----------
    obs_id: str
        Obs id to process or load
    configs: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets: dict
        Dictionary specifying which detectors/wafers to load see ``Context.obsdb.get_obs``.
    context: core.Context
        Optional. Context object used for data loading/querying.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.

    Returns
    -------
    dbexist : bool
        True if db exists and entry for input detectors is found.
    """

    if logger is None:
        logger = init_logger("preprocess")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    if context is None:
        context = core.Context(configs["context_file"])
    group_by, _, _ = get_groups(obs_id, configs, context)
    cur_groups = [list(np.fromiter(dets.values(), dtype='<U32'))]
    dbexist = True
    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])
        dbix = {'obs:obs_id':obs_id}
        for gb, g in zip(group_by, cur_groups[0]):
            dbix[f'dets:{gb}'] = g
        logger.info(f'find_db found: {dbix}')
        if len(db.inspect(dbix)) == 0:
            dbexist = False
    else:
        dbexist = False

    return dbexist


def save_group(obs_id, configs, dets, context=None, subdir='temp'):
    """This function returns a dictionary containing the data destination filename
    and the values to populate the manifest db.

    Arguments
    ----------
    obs_id: str
        Obs id to process or load
    configs: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets: dict
        Dictionary specifying which detectors/wafers to load see ``Context.obsdb.get_obs``.
    context: core.Context
        Optional. Context object used for data loading/querying.
    subdir: str
        Optional. Subdirectory to save the output files into.  If it does not exist, it is created.

    Returns
    -------
    outputs : dict
        Dictionary including output filename of data file and information for corresponding
        database entry.
    """

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    if context is None:
        context = core.Context(configs["context_file"])

    cur_groups = [list(np.fromiter(dets.values(), dtype='<U32'))]
    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    newpath = f'{subdir}/{obs_id}'
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

    # Collect info for saving h5 file.
    outputs = {}
    outputs['temp_file'] = dest_file

    # Collect index info.
    db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
    for gb, g in zip(group_by, cur_groups[0]):
        db_data['dets:'+gb] = g
    outputs['db_data'] = db_data

    return outputs


def save_group_and_cleanup(obs_id, configs, context=None, subdir='temp',
                           logger=None, remove=False):
    """This function checks if any temporary files exist from a preprocessing
     run and will either add them to the config policy file and create an entry
     in the manifest db by calling ``cleanup_mandb``.  If the file exists but
     cannot be opened or if remove is True, the file will be deleted. Remove
     is intended to be to allow for overwrite=True in ``preprocess_tod.py``
     and ``multilayer_preprocess_tod.py``.

    Arguments
    ----------
    obs_id: str
        Obs id to process or load
    configs: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    context: core.Context
        Optional. Context object used for data loading/querying.
    subdir: str
        Optional. Subdirectory to save the output files into.  If it does not exist, it is created.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.
    remove: bool
        Optional. Default is False. Whether to remove a file if found.
        Used when ``overwrite`` is True in driving functions.
    """

    if logger is None:
        logger = init_logger("preprocess")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    if context is None:
        context = core.Context(configs["context_file"])

    group_by, groups, error = get_groups(obs_id, configs, context)

    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])
        x = db.inspect({'obs:obs_id': obs_id})
    else:
        x = None

    all_groups = groups.copy()
    for g in all_groups:
        if 'wafer.bandpass' in group_by:
            if 'NC' in g:
                groups.remove(g)
                continue

    # get groups in db
    db_groups = []
    if x is not None and len(x) > 0:
        [db_groups.append([a[f'dets:{gb}'] for gb in group_by]) for a in x]

    for g in groups:
        dets = {gb:gg for gb, gg in zip(group_by, g)}
        outputs_grp = save_group(obs_id, configs, dets, context, subdir)

        if os.path.exists(outputs_grp['temp_file']):
            try:
                if not remove and g not in db_groups:
                    cleanup_mandb(None, outputs_grp, configs, logger)
                else:
                    # if we're overwriting
                    if remove:
                        logger.info(f"remove={remove}: removing {outputs_grp['temp_file']}")
                    # if found in database already
                    elif g in db_groups:
                        logger.info(f"{outputs_grp['temp_file']} found in db, removing")
                    os.remove(outputs_grp['temp_file'])
            except OSError as e:
                # remove if it can't be opened
                os.remove(outputs_grp['temp_file'])
    return error


def cleanup_obs(obs_id, policy_dir, errlog, configs, context=None,
                subdir='temp', remove=False):
    """
    For a given obs id, this function will search the policy_dir directory
    if it exists for any files with that obsnum in their filename. If any are
    found, it will run save_group_and_cleanup for that obs id.

    Arguments
    ---------
    obs_id: str
        Obs id to check and clean up
    policy_dir: str
        Directory to temp per-group output files
    errlog: fpath
        Filepath to error logging file.
    configs: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    context: core.Context
        Optional. Context object used for data loading/querying.
    subdir: str
        Optional. Subdirectory to save the output files into.
    remove: bool
        Optional. Default is False. Whether to remove a file if found.
        Used when ``overwrite`` is True in driving functions.
    """

    if os.path.exists(policy_dir):
        found = False
        for f in os.listdir(policy_dir):
            if obs_id in f:
                found = True
                break

        if found:
            error = save_group_and_cleanup(obs_id, configs, context,
                                           subdir=subdir, remove=remove)
            if error is not None:
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, cleanup error\n{error[0]}\n{error[2]}\n')
                f.close()


def preproc_or_load_group(obs_id, configs_init, dets, configs_proc=None,
                          logger=None, overwrite=False):
    """
    This function is expected to receive a single obs_id, and dets dictionary.
    The dets dictionary must match the grouping specified in the preprocess
    config files. It accepts either one or two config strings or dicts representing
    an initial and a dependent pipeline stage. If the preprocess database entry for
    this obsid-dets group already exists then this function will just load back the
    processed tod calling either the ``load_and_preprocess`` or
    ``multilayer_load_and_preprocess`` functions. If the db entry does not exist or
    the overwrite flag is set to True then the full preprocessing steps defined in
    the configs are run and the outputs are written to a unique h5 file. Any errors,
    the info to populate the database, the file path of the h5 file, and the process
    tod are returned from this function. This function is expected to be run in
    conjunction with the ``cleanup_mandb`` function which consumes all of the outputs
    (except the processed tod), writes to the database, and moves the multiple h5 files
    into fewer h5 files (each <= 10 GB).

    Arguments
    ---------
    obs_id: str
        Obs id to process or load
    configs_init: fpath or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets: dict
        Dictionary specifying which detectors/wafers to load see ``Context.obsdb.get_obs``.
    configs_proc: fpath or dict
        Filepath or dictionary containing a dependent preprocess configuration file.
    logger: PythonLogger
        Optional. Logger object or None will generate a new one.
    overwrite: bool
        Optional. Whether or not to overwrite existing entries in the preprocess manifest db.

    Returns
    -------
    error: str
        String indicating if the function succeeded in its execution or failed.
        If ``None`` then it succeeded in processing and the mandB should be updated.
        If ``'load_success'`` then axis manager was successfully loaded from existing preproc db.
        If any other string then processing failed and output will be logged in the error log.
    output_init: list
        Varies depending on the value of ``error``.
        If ``error == None`` then output is the info needed to update the manifest db.
        If ``error == 'load_success'`` then output is just ``[obs_id, dets]``.
        If ``error`` is anything else then output stores what to save in the error log.
    output_proc: list:
        See output_init for possible values.
    aman: Core.AxisManager
        Processed axis manager only returned if ``error`` is ``None`` or ``'load_success'``.
    """

    if logger is None:
        logger = init_logger("preprocess")

    error = None

    if type(configs_init) == str:
        configs_init = yaml.safe_load(open(configs_init, "r"))

    context_init = core.Context(configs_init["context_file"])

    if configs_proc is not None:
        if type(configs_proc) == str:
            configs_proc = yaml.safe_load(open(configs_proc, "r"))
        context_proc = core.Context(configs_proc["context_file"])

        group_by, groups, error = get_groups(obs_id, configs_proc, context_proc)
    else:
        group_by, groups, error = get_groups(obs_id, configs_init, context_init)

    if error is not None:
        return error[0], [error[1], error[2]], [error[1], error[2]], None

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
            return error, [obs_id, dets], [obs_id, dets], None

    db_init_exist = find_db(obs_id, configs_init, dets, context_init,
                            logger=logger)

    db_proc_exist = False
    if configs_proc is not None:
        db_proc_exist = find_db(obs_id, configs_proc, dets, context_proc,
                                logger=logger)

    if (not db_init_exist) and db_proc_exist and (not overwrite):
        logger.info('dependent db requires initial db if not overwriting')
        error = 'no_init_db'
        return error, [obs_id, dets], [obs_id, dets], None

    if db_init_exist and (not overwrite):
        if db_proc_exist:
            try:
                logger.info(f"both db and depdendent db exist for {obs_id} {dets} loading data and applying preprocessing.")
                aman = multilayer_load_and_preprocess(obs_id=obs_id, dets=dets, configs_init=configs_init,
                                                      configs_proc=configs_proc, logger=logger)
                error = 'load_success'
                return error, [obs_id, dets], [obs_id, dets], aman
            except Exception as e:
                error = f'Failed to load: {obs_id} {dets}'
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.info(f"{error}\n{errmsg}\n{tb}")
                return error, [errmsg, tb], [errmsg, tb], None
        else:
            try:
                logger.info(f"init db exists for {obs_id} {dets} loading data and applying preprocessing.")
                aman = load_and_preprocess(obs_id=obs_id, dets=dets, configs=configs_init,
                                           context=context_init, logger=logger)
            except Exception as e:
                error = f'Failed to load: {obs_id} {dets}'
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.info(f"{error}\n{errmsg}\n{tb}")
                return error, [errmsg, tb], [errmsg, tb], None

            if configs_proc is None:
                error = 'load_success'
                return error, [obs_id, dets], [obs_id, dets], aman
            else:
                try:
                    outputs_proc = save_group(obs_id, configs_proc, dets, context_proc, subdir='temp_proc')
                    init_fields = aman.preprocess._fields.copy()
                    logger.info(f"Generating new dependent preproc db entry for {obs_id} {dets}")
                    # pipeline for init config
                    pipe_init = Pipeline(configs_init["process_pipe"], plot_dir=configs_init["plot_dir"], logger=logger)
                    aman_cfgs_ref = get_pcfg_check_aman(pipe_init)
                    # pipeline for processing config
                    pipe_proc = Pipeline(configs_proc["process_pipe"], plot_dir=configs_proc["plot_dir"], logger=logger)

                    # tags from context proc
                    tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])

                    if "tags" in aman._fields:
                        aman.move("tags", None)
                    aman.wrap('tags', tags_proc)

                    proc_aman, success = pipe_proc.run(aman)

                    # remove fields found in aman.preprocess from proc_aman
                    for fld_init in init_fields:
                        if fld_init in proc_aman:
                            proc_aman.move(fld_init, None)

                    proc_aman.wrap('pcfg_ref', aman_cfgs_ref)

                except Exception as e:
                    error = f'Failed to run dependent processing pipeline: {obs_id} {dets}'
                    errmsg = f'Dependent pipeline failed with {type(e)}: {e}'
                    tb = ''.join(traceback.format_tb(e.__traceback__))
                    logger.info(f"{error}\n{errmsg}\n{tb}")
                    return error, [errmsg, tb], [errmsg, tb], None
                if success != 'end':
                    # If a single group fails we don't log anywhere just mis an entry in the db.
                    return success, [obs_id, dets], [obs_id, dets], None

                logger.info(f"Saving data to {outputs_proc['temp_file']}:{outputs_proc['db_data']['dataset']}")
                proc_aman.save(outputs_proc['temp_file'], outputs_proc['db_data']['dataset'], overwrite)

                aman.preprocess.merge(proc_aman)

                return error, [obs_id, dets], outputs_proc, aman
    else:
        # pipeline for init config
        logger.info(f"Generating new preproc db entry for {obs_id} {dets}")
        try:
            pipe_init = Pipeline(configs_init["process_pipe"], plot_dir=configs_init["plot_dir"], logger=logger)
            aman_cfgs_ref = get_pcfg_check_aman(pipe_init)
            outputs_init = save_group(obs_id, configs_init, dets, context_init, subdir='temp')
            aman = context_init.get_obs(obs_id, dets=dets)
            tags = np.array(context_init.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])
            aman.wrap('tags', tags)
            proc_aman, success = pipe_init.run(aman)
            aman.wrap('preprocess', proc_aman)
        except Exception as e:
            error = f'Failed to run initial pipeline: {obs_id} {dets}'
            errmsg = f'Initial pipeline failed with {type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"{error}\n{errmsg}\n{tb}")
            return error, [errmsg, tb], [errmsg, tb], None
        if success != 'end':
            # If a single group fails we don't log anywhere just mis an entry in the db.
            return success, [obs_id, dets], [obs_id, dets], None

        logger.info(f"Saving data to {outputs_init['temp_file']}:{outputs_init['db_data']['dataset']}")
        proc_aman.save(outputs_init['temp_file'], outputs_init['db_data']['dataset'], overwrite)

        if configs_proc is None:
            return error, outputs_init, [obs_id, dets], aman
        else:
            try:
                outputs_proc = save_group(obs_id, configs_proc, dets, context_proc, subdir='temp_proc')
                init_fields = aman.preprocess._fields.copy()
                logger.info(f"Generating new dependent preproc db entry for {obs_id} {dets}")
                # pipeline for processing config
                pipe_proc = Pipeline(configs_proc["process_pipe"], plot_dir=configs_proc["plot_dir"], logger=logger)
                 # tags from context proc
                tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id, tags=True)['tags'])

                if "tags" in aman._fields:
                    aman.move("tags", None)
                aman.wrap('tags', tags_proc)

                proc_aman, success = pipe_proc.run(aman)

                # remove fields found in aman.preprocess from proc_aman
                for fld_init in init_fields:
                    if fld_init in proc_aman:
                        proc_aman.move(fld_init, None)

                proc_aman.wrap('pcfg_ref', aman_cfgs_ref)

            except Exception as e:
                error = f'Failed to run dependent processing pipeline: {obs_id} {dets}'
                errmsg = f'Dependent pipeline failed with {type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.info(f"{error}\n{errmsg}\n{tb}")
                return error, [errmsg, tb], [errmsg, tb], None
            if success != 'end':
                # If a single group fails we don't log anywhere just mis an entry in the db.
                return success, [obs_id, dets], [obs_id, dets], None

            logger.info(f"Saving data to {outputs_proc['temp_file']}:{outputs_proc['db_data']['dataset']}")
            proc_aman.save(outputs_proc['temp_file'], outputs_proc['db_data']['dataset'], overwrite)

            aman.preprocess.merge(proc_aman)

            return error, outputs_init, outputs_proc, aman


def cleanup_mandb(error, outputs, configs, logger=None, overwrite=False):
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

    Arguments
    ---------
    error : str
        Error message output form preprocessing functions
    outputs : dict
        Dictionary including entries for the temporary h5 filename
        ('temp_file') and the obs_id group metadata and db entry (db_data).
        See save_group for more info.
    configs : dict
        Preprocessing configuration dictionary
    logger : PythonLogger
        Optional.  Python logger.
    overwrite : bool
        Optional. Delete the entry in the archive file if it exists and
        replace it with the new entry.
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
        if os.path.isabs(folder) and not(os.path.exists(folder)):
            os.makedirs(folder)
        while os.path.exists(dest_file) and os.path.getsize(dest_file) > 10e9:
            nfile += 1
            dest_file = basename + '_' + str(nfile).zfill(3) + '.h5'
        group_by = [k.split(':')[-1] for k in outputs['db_data'].keys() if 'dets' in k]
        h5_path = os.path.relpath(dest_file,
                                  start=os.path.dirname(configs['archive']['index']))

        src_file = outputs['temp_file']

        logger.debug(f"Source file: {src_file}")
        logger.debug(f"Destination file: {dest_file}")

        with h5py.File(dest_file,'a') as f_dest:
            with h5py.File(src_file,'r') as f_src:
                for dts in f_src.keys():
                    logger.debug(f"\t{dts}")
                    # If the dataset or group already exists, delete it to overwrite
                    if overwrite and dts in f_dest:
                        del f_dest[dts]
                    f_src.copy(f_src[f'{dts}'], f_dest, f'{dts}')
                    for member in f_src[dts]:
                        logger.debug(f"\t{dts}/{member}")
                        if isinstance(f_src[f'{dts}/{member}'], h5py.Dataset):
                            f_src.copy(f_src[f'{dts}/{member}'], f_dest[f'{dts}'], f'{dts}/{member}')
        logger.info(f"Saving to database under {outputs['db_data']}")
        db = get_preprocess_db(configs, group_by, logger)
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
        if outputs is not None:
            f.write(f'\t{outputs[0]}\n\t{outputs[1]}\n')
        f.close()


def get_pcfg_check_aman(pipe):
    """
    Given a preprocess pipeline class return an axis manager containing
    the ordered steps of the pipeline with all arguments for each step.

    Arguments
    ---------
    pipe: _Preprocess class
        Preprocess pipeline class from which to build the step argument axis
        manager.
    """

    pcfg_ref = core.AxisManager()
    for i, pp in enumerate(pipe):
        pcfg_ref.wrap(f'{i}_{pp.name}', core.AxisManager())
        for memb in inspect.getmembers(pp, lambda a:not(inspect.isroutine(a))):
            if not memb[0][0] == '_':
                if type(memb[1]) is dict:
                    pcfg_ref[f'{i}_{pp.name}'].wrap(memb[0], core.AxisManager())
                    for itm in memb[1].items():
                        pcfg_ref[f'{i}_{pp.name}'][memb[0]].wrap(itm[0], str(itm[1]))
                else:
                    pcfg_ref[f'{i}_{pp.name}'].wrap(memb[0], memb[1])
    return pcfg_ref


def _check_assignment_length(a, b):
    """
    Helper function to check if the set of assignments in axis manager ``a`` matches
    the length of assignments in axis manager ``b``.

    Arguments
    ---------
    a: AxisManager
        Primary axis manager to cross check assignments with.
    b: AxisManager
        Secondary axis manager to cross check assignments with
    """

    aa = np.fromiter(a._assignments.keys(), dtype='<U32')
    bb = np.fromiter(b._assignments.keys(), dtype='<U32')

    if len(aa) != len(bb):
        return False, None, None
    else:
        return True, aa, bb


def check_cfg_match(ref, loaded, logger=None):
    """
    Checks that the ``ref`` and ``loaded`` axis managers containing the ordered
    preprocess pipelines match one another.

    Arguments
    ---------
    ref : AxisManager
        Reference axis manager for cross checking
    loaded : AxisManager
        Loaded axis manager for cross checking.
    logger : PythonLogger
        Optional. Python logger object.
    """

    if logger is None:
        logger = init_logger("preprocess")
    check, ref_items, loaded_items = _check_assignment_length(ref, loaded)
    if check:
        for ri, li in zip (ref_items, loaded_items):
            if ri != li:
                logger.warning('Config check fails due to ordered pipeline element names not matching.')
                return False
            else:
                if type(ref[ri]) is core.AxisManager and type(loaded[li]) is core.AxisManager:
                    check_cfg_match(ref[ri], loaded[li], logger)
                elif ref[ri] == loaded[li]:
                    continue
                else:
                    logger.warning(f'Config check fails due to arguments of {li} not matching')
                    return False
        return True
    else:
        logger.warning('Config check fails due to pipeline list not being of equal length')
        return False
