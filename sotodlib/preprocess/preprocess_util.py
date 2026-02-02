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
from pathlib import Path
import re
from tqdm import tqdm
from sotodlib.hwp import hwp_angle_model
from sotodlib.coords import demod as demod_mm
from sotodlib.tod_ops import t2pleakage
from sotodlib.core.flagman import has_any_cuts
from sotodlib.site_pipeline.jobdb import JState
from sotodlib.core.util import H5ContextManager

from .. import core
from . import Pipeline


class PreprocessErrors:
    """Stores the various errors that can occur from the preprocessing
    functions.
    """
    LoadSuccess = "load_success"
    GetGroupsError = "get_groups_error"
    MetaDataError = "get_meta_data_error"
    NoDetsRemainError = "no_dets_remain_error"
    NoGroupOverlapError = "no_group_overlap_error"
    MultilayerPipelineLoadError = "multilayer_load_and_preprocess_error"
    SingleLayerPipelineLoadError = "single_layer_load_and_preprocess_error"
    PipeLineRunError = "pipeline_run_error"
    InitPipeLineRunError = "init_pipeline_run_error"
    ProcPipeLineRunError = "proc_pipeline_run_error"
    PipeLineStepError = "pipeline_step_error"
    NoInitDbError = "no_init_db_error"
    GroupOutputError = "group_output_error"
    ExecutorFutureError = "executor_future_error"
    SkipMissingError = "skip_missing_error"

    @classmethod
    def get_errors(cls, e):
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))

        return errmsg, tb


def _get_aman_encodings(encodings, field):
    """Encodings for flacarray compression."""
    if (
        isinstance(field, np.ndarray)
        and np.issubdtype(field.dtype, np.number)
        and not np.isnan(field).any()
    ):
        encodings["type"] = "flacarray"

        if np.issubdtype(field.dtype, np.floating):
            if field.dtype == np.float32:
                quanta = 1e-7
            else:
                quanta = 1e-10

            encodings["args"] = {
                "level": 5,
                "quanta": quanta,
            }
        return

    elif isinstance(field, core.AxisManager):
        for name in field._assignments:
            subfield = field[name]
            encodings[name] = {}
            _get_aman_encodings(encodings[name], subfield)


def filter_preproc_runlist_by_jobdb(jdb, jclass, db, run_list, group_by,
                                    overwrite=False, logger=None):
    """Given a preprocess_tod or multilayer_preprocess_tod run list, checks
    whether that entry exists in the preprocess jobdb. If it failed or is done
    and overwrite is False, add it to the list of skipped obs_ids.  If it
    doesn't exist, is open, or is done but overwite is True, add an open job
    to the jobdb.

    Arguments
    ---------

    jdb : JobDB
        The preprocessing jobdb class.
    jclass : str
        The job name.
    db : ManifestDb or None
        Preprocessing database.
    run_list : list
        List of (obs_id, group) tuples.
    group_by : list
        How grouping is being done for preprocessing.  Specified in the
        preprocessing config through the subobs.use entry.
    overwrite : bool
        Whether or not to overwrite entries in the preprocessing db.
    logger : PythonLogger
        A python logger.

    Returns
    -------
    run_list : list
        Run list with the subset of skipped entries removed.
    """
    if logger is None:
        logger = init_logger("preprocess")

    run_list_skipped = []

    failed = 0
    done = 0

    existing_jobs = jdb.get_jobs(jclass)
    tags_to_job = {
        frozenset({k: v for k, v in j.tags.items() if k != 'error'}.items()): j
        for j in existing_jobs
    }

    new_jobs = []
    for r in tqdm(run_list, total=len(run_list),
                 desc="filtering by jobdb"):
        tags = {}
        tags["obs:obs_id"] = r[0]
        for gb, g in zip(group_by, r[1]):
            tags['dets:' + gb] = g

        job = tags_to_job.get(frozenset(tags.items()), None)
        if not job:
            tags["error"] = None
            new_jobs.append(jdb.create_job(jclass, tags=tags, commit=False,
                                           check_existing=False))
        elif job.jstate in [JState.done, JState.failed]:
            found = True
            if job.jstate == JState.done and db is not None:
                x = db.inspect({'obs:obs_id': r[0]})
                found = any(
                    [a[f'dets:{gb}'] for gb in group_by] == r[1]
                    for a in x
                )
            if overwrite or not found:
                with jdb.locked(job) as j:
                    if overwrite == True:
                        j.jstate = "open"
                        j.tags["error"] = None
            else:
                if job.jstate == JState.done:
                    done += 1
                elif job.jstate == JState.failed:
                    failed += 1
                run_list_skipped.append(r)

    jdb.commit_jobs(new_jobs)

    logger.info(f"skipping {done} done jobs and {failed} failed jobs from jobdb")
    run_list = [r for r in run_list if r not in run_list_skipped]

    return run_list


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


def get_groups(obs_id, configs, context=None):
    """Get subobs group method and groups. To be used in
    ``preprocess_*.py`` site pipeline scripts.

    Arguments
    ----------
    obs_id : str
        The obsid.
    configs : str or dict
        The configuration dictionary.
    context : core.Context
        The Context file to use.

    Returns
    -------
    group_by : list of str
        The list of keys used to group the detectors.
    groups : list of list of int
        The list of groups of detectors.
    errors : tuple
        Tuple of errors or Nones.
    """
    try:
        if type(configs) == str:
            configs = yaml.safe_load(open(configs, "r"))
        if context is None:
            context = core.Context(configs["context_file"])
        group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
        for i, gb in enumerate(group_by):
            if gb.startswith('dets:'):
                group_by[i] = gb.split(':',1)[1]

            if (gb == 'detset') and (len(group_by) == 1):
                groups = context.obsfiledb.get_detsets(obs_id)
                return group_by, [[g] for g in groups], (None, None, None)

        det_info = context.get_det_info(obs_id)
        rs = det_info.subset(keys=group_by).distinct()
        groups = [[b for a,b in r.items()] for r in rs]
        return group_by, groups, (None, None, None)
    except Exception as e:
        error = PreprocessErrors.GetGroupsError
        errmsg, tb = PreprocessErrors.get_errors(e)
        return [], [], (error, errmsg, tb)


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
        logger = init_logger("preprocess")

    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])
    else:
        logger.debug(f"Creating {configs['archive']['index']} as the "
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
    os.makedirs(dname, exist_ok=True)
    return tc


def load_preprocess_det_select(obs_id, configs, context=None,
                               dets=None, meta=None, logger=None):
    """Loads the metadata information for the Observation and runs through any
    data selection specified by the Preprocessing Pipeline.

    Arguments
    ----------
    obs_id : multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs : string or dictionary
        Config file or loaded config directory
    context : core.Context
        The Context file to use.
    dets : dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta : AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    logger : PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.

    Returns
    -------
    list
        Restricted list of detector vals.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs, context = get_preprocess_context(configs, context)
    pipe = Pipeline(configs["process_pipe"], logger=logger)

    if meta is None:
        meta = context.get_meta(obs_id, dets=dets)
    logger.info("Restricting detectors on all processes")
    keep_all = np.ones(meta.dets.count,dtype=bool)
    for process in pipe[:]:
        keep = process.select(meta, in_place=False)
        if isinstance(keep, np.ndarray):
            keep_all &= keep
    return meta.dets.vals[keep_all]


def load_and_preprocess(obs_id, configs, context=None, dets=None, meta=None,
                        no_signal=None, logger=None):
    """Loads the saved information from the preprocessing pipeline and runs
    the processing section of the pipeline.

    Assumes preprocess_tod has already been run on the requested observation.

    Arguments
    ----------
    obs_id : multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs : string or dictionary
        Config file or loaded config directory
    context : core.Context
        Optional. The Context file to use.
    dets : dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta : AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal : bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger : PythonLogger
        Optional. Logger object.  If None, a new logger
        is created.

    Returns
    -------
    aman : core.AxisManager or None
        Loaded and restricted axis manager with preprocessing metadata. Returns
        ``None`` if all detectors cut.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs, context = get_preprocess_context(configs, context)
    meta = context.get_meta(obs_id, dets=dets, meta=meta)
    if (
        'valid_data' in meta.preprocess and
        isinstance(meta.preprocess.valid_data, core.AxisManager)
       ):
        keep = has_any_cuts(meta.preprocess.valid_data.valid_data)
        meta.restrict("dets", keep)
    else:
        det_vals = load_preprocess_det_select(obs_id, configs=configs, context=context,
                                              dets=dets, meta=meta, logger=logger)
        meta.restrict("dets", [d for d in meta.dets.vals if d in det_vals])

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
                                   logger=None, init_only=False,
                                   ignore_cfg_check=False):
    """Loads the saved information from the preprocessing pipeline from a
    reference and a dependent database and runs the processing section of
    the pipeline for each.

    Assumes preprocess_tod and multilayer_preprocess_tod have already been run
    on the requested observation.

    Arguments
    ----------
    obs_id : multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs_init : string or dictionary
        Config file or loaded config directory
    configs_proc : string or dictionary
        Second config file or loaded config dictionary to load
        dependent databases generated using multilayer_preprocess_tod.py.
    dets : dict
        Dets to restrict on from info in det_info. See context.get_meta.
    meta : AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal : bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger : PythonLogger
        Optional. Logger object or None will generate a new one.
    init_only : bool
        Optional. If True, do not run the dependent pipeline.
    ignore_cfg_check : bool
        If True, do not attempt to validate that configs_init is the same as
        the config used to create the existing init db.

    Returns
    -------
    aman : core.AxisManager or None
        Loaded and restricted axis manager with preprocessing metadata. Returns
        ``None`` if all detectors cut.
    """

    if logger is None:
        logger = init_logger("preprocess")

    configs_init, context_init = get_preprocess_context(configs_init)
    meta_init = context_init.get_meta(obs_id, dets=dets, meta=meta)

    configs_proc, context_proc = get_preprocess_context(configs_proc)
    meta_proc = context_proc.get_meta(obs_id, dets=dets, meta=meta)

    group_by_init = np.atleast_1d(configs_init['subobs'].get('use', 'detset'))
    group_by_proc = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

    if (group_by_init != group_by_proc).any():
        raise ValueError('init and proc groups do not match')

    if meta_init.dets.count == 0 or meta_proc.dets.count == 0:
        logger.info(f"No detectors in obs {obs_id}")
        return None
    else:
        pipe_init = Pipeline(configs_init["process_pipe"], logger=logger)
        aman_cfgs_ref = get_pcfg_check_aman(pipe_init)

        if (
            ignore_cfg_check or
            check_cfg_match(aman_cfgs_ref, meta_proc.preprocess['pcfg_ref'],
                           logger=logger)
        ):
            pipe_proc = Pipeline(configs_proc["process_pipe"], logger=logger)

            logger.info("Restricting detectors on all proc pipeline processes")
            if (
                'valid_data' in meta_proc.preprocess and
                isinstance(meta_proc.preprocess.valid_data, core.AxisManager)
               ):
                keep_all = has_any_cuts(meta_proc.preprocess.valid_data.valid_data)
            else:
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

            if 'valid_data' in aman.preprocess:
                aman.preprocess.move('valid_data', None)
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
    obs_id : multiple
        Passed to `context.get_obs` to load AxisManager, see Notes for
        `context.get_obs`
    configs_init : string or dictionary
        Config file or loaded config directory
    configs_proc : string or dictionary
        Second config file or loaded config dictionary to load
        dependent databases generated using multilayer_preprocess_tod.py.
    sim_map : numpy.ndmap or enmap.ndmap
        Input simulated map to be observed
    meta : AxisManager
        Contains supporting metadata to use for loading.
        Can be pre-restricted in any way. See context.get_meta.
    no_signal : bool
        If True, signal will be set to None.
        This is a way to get the axes and pointing info without
        the (large) TOD blob.  Not all loaders may support this.
    logger : PythonLogger
        Optional. Logger object or None will generate a new one.
    init_only : bool
        Optional. Whether or not to run the dependent pipeline.
    t2ptemplate_aman : AxisManager
        Optional. AxisManager to use as a template for t2p leakage
        deprojection.

    Returns
    -------
    aman : core.AxisManager or None
        Loaded and restricted axis manager with preprocessing metadata. Returns
        ``None`` if all detectors cut.
    """
    if logger is None:
        logger = init_logger("preprocess")

    configs_init, context_init = get_preprocess_context(configs_init)
    meta_init = context_init.get_meta(obs_id, meta=meta)

    configs_proc, context_proc = get_preprocess_context(configs_proc)
    meta_proc = context_proc.get_meta(obs_id, meta=meta)

    group_by_init = np.atleast_1d(configs_init['subobs'].get('use', 'detset'))
    group_by_proc = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

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
            if 'valid_data' in aman.preprocess:
                aman.preprocess.move('valid_data', None)
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
    obs_id : str
        Obs id to process or load
    configs : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets : dict
        Dictionary specifying which detectors/wafers to load see ``Context.obsdb.get_obs``.
    context : core.Context
        Optional. Context object used for data loading/querying.
    logger : PythonLogger
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
    group_by = np.atleast_1d(configs['subobs'].get('use', 'detset'))
    cur_groups = [list(np.fromiter(dets.values(), dtype='<U32'))]
    dbexist = True
    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])
        dbix = {'obs:obs_id':obs_id}
        for gb, g in zip(group_by, cur_groups[0]):
            dbix[f'dets:{gb}'] = g
        if len(db.inspect(dbix)) == 0:
            dbexist = False
            logger.debug(f"Entry {dbix} not found in {configs['archive']['index']}")
        else:
            logger.debug(f"Entry {dbix} found in {configs['archive']['index']}")
    else:
        dbexist = False

    return dbexist


def cleanup_archive(configs, logger=None):
    """This function finds the final preprocess archive file and deletes any
    datasets that are not found in the preprocess database.  This helps avoid
    cases where the database writing was interrupted in a previous run.

    Arguments
    ----------
    configs : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    logger : PythonLogger
        Optional. Logger object or None will generate a new one.
    """

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    if logger is None:
        logger = init_logger("preprocess")

    if os.path.exists(configs['archive']['index']):
        db = core.metadata.ManifestDb(configs['archive']['index'])

        basename = os.path.splitext(os.path.basename(configs["archive"]["policy"]["filename"]))[0]

        # remove datasets from last archive file if they are not in db
        archive_files = list(
            Path(os.path.dirname(configs["archive"]["policy"]["filename"])).rglob(f"{basename}*.h5")
        )
        pattern = re.compile(r"\d+")
        archive_files = [p for p in archive_files if pattern.findall(p.stem)]

        if archive_files:
            latest_file = max([(int(pattern.findall(p.stem)[-1]), p)
                               for p in archive_files if pattern.findall(p.stem)],
                              key=lambda t: t[0])[1]

            db_datasets = [d['dataset'] for d in db.inspect()]
            with H5ContextManager(latest_file, mode="r+") as f:
                keys = list(f.keys())
                for key in keys:
                    if key not in db_datasets:
                        logger.debug(f"{key} not found in db. deleting it from {latest_file}.")
                        del f[key]

        db.conn.close()


def get_preproc_group_out_dict(obs_id, configs, dets, context=None, subdir='temp'):
    """This function returns a dictionary containing the data destination filename
    and the values to populate the manifest db.

    Arguments
    ----------
    obs_id : str
        Obs id to process or load
    configs : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets : dict
        Dictionary specifying which detectors/wafers to load see
        ``Context.obsdb.get_obs``.
    context : core.Context
        Optional. Context object used for data loading/querying.
    subdir : str
        Optional. Subdirectory to save the output files into.  If it does not
        exist, it is created.

    Returns
    -------
    outputs : dict
        Dictionary including output filename of data file and information for
        corresponding database entry.
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
    obs_id : str
        Obs id to process or load
    configs : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    context : core.Context
        Optional. Context object used for data loading/querying.
    subdir : str
        Optional. Subdirectory to save the output files into.
        If it does not exist, it is created.
    logger : PythonLogger
        Optional. Logger object or None will generate a new one.
    remove : bool
        Optional. Default is False. Whether to remove a file if found.
        Used when ``overwrite`` is True in driving functions.

    Returns
    -------
    errors : tuple
        Error from get_groups.
    """

    if logger is None:
        logger = init_logger("preprocess")

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    if context is None:
        context = core.Context(configs["context_file"])

    group_by, groups, errors = get_groups(obs_id, configs)

    all_groups = groups.copy()
    for g in all_groups:
        if 'wafer.bandpass' in group_by:
            if 'NC' in g:
                groups.remove(g)
                continue

    for g in groups:
        dets = {gb:gg for gb, gg in zip(group_by, g)}
        outputs_grp = get_preproc_group_out_dict(obs_id, configs,
                                                 dets, subdir=subdir)

        if os.path.exists(outputs_grp['temp_file']):
            try:
                if not remove:
                    cleanup_mandb(outputs_grp, (obs_id, g),
                                  (None, None, None), configs, logger)
                else:
                    # if we're overwriting, remove file so it will re-run
                    os.remove(outputs_grp['temp_file'])
            except OSError as e:
                # remove if it can't be opened
                os.remove(outputs_grp['temp_file'])
    return errors


def cleanup_obs(obs_id, policy_dir, errlog, configs, context=None,
                subdir='temp', remove=False):
    """For a given obs id, this function will search the policy_dir directory
    if it exists for any files with that obsnum in their filename. If any are
    found, it will run save_group_and_cleanup for that obs id.

    Arguments
    ---------
    obs_id : str
        Obs id to check and clean up
    policy_dir : str
        Directory to temp per-group output files
    errlog : str
        Filepath to error logging file.
    configs : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    context : core.Context
        Optional. Context object used for data loading/querying.
    subdir : str
        Optional. Subdirectory to save the output files into.
    remove : bool
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
            errors = save_group_and_cleanup(obs_id, configs, context,
                                           subdir=subdir, remove=remove)

            if errors[0] is not None:
                with open(errlog, 'a') as f:
                    f.write(f"{time.time()}, {obs_id}, n/a', {errors[0]}\n")
                    f.write("\t" + (errors[1] or "") + (errors[2] or "") + "\n")


def preproc_or_load_group(obs_id, configs_init, dets, configs_proc=None,
                         logger=None, overwrite=False, save_archive=False,
                         save_proc_aman=True, compress=False,
                         skip_missing=False, ignore_cfg_check=False):
    """
    This function is expected to receive a single obs_id, and dets dictionary.
    The dets dictionary must match the grouping specified in the preprocess
    config files. It accepts either one or two config strings or dicts representing
    an initial and a dependent pipeline stage. If the preprocess database entry for
    this obsid-dets group already exists then this function will just load back the
    processed tod calling either the ``load_and_preprocess`` or
    ``multilayer_load_and_preprocess`` functions. If the db entry does not exist or
    the overwrite flag is set to True then the full preprocessing steps defined in
    the configs are run and if save_proc_aman is True, the outputs are written to a
    unique h5 file. Any errors, the info to populate the database, the file path of
    the h5 file, and the process tod are returned from this function. Processed axis
    managers can be written to an archive and database by using cleanup_mandb
    (or setting save_archive to True) which consumes all of the outputs
    (except the processed tod), writes to the database, and moves the multiple h5
    files into fewer h5 files (each <= 10 GB).

    Arguments
    ---------
    obs_id : str
        Obs id to process or load
    configs_init : str or dict
        Filepath or dictionary containing the preprocess configuration file.
    dets : dict
        Dictionary specifying which detectors/wafers to load see
        ``Context.obsdb.get_obs``.
    configs_proc : str or dict
        Filepath or dictionary containing a dependent preprocess configuration
        file.
    logger : PythonLogger
        Optional. Logger object or None will generate a new one.
    overwrite : bool
        Optional. Whether or not to overwrite existing entries in the
        preprocess manifest db.
     save_archive : bool
        Call cleanup_mandb if True to save to the archive and database files
        in configs_init and configs_proc. Should be False if preproc_or_load_group
        is being called from within a parallelized script (i.e. python multiprocessing or MPI).
    save_proc_aman : bool
        Whether or not to save the preprocessing axis manager.  Required if saving into
        a preprocessing archive.
    compress : bool
        Whether or not to compress the preprocessing data.  Uses flacarray compression.
    skip_missing : bool
        Do not attempt to run preprocessing pipeline if either of the preproc dbs
        don't exist or the obs_id and group combination is not found.
    ignore_cfg_check : bool
        If True, do not attempt to validate that configs_init is the same as the config
        used to create the existing init db when running ``multilayer_load_and_preprocess``.

    Returns
    -------
    aman : AxisManager or None
        Preprocessed axis manager if preproc_or_load_group finished
        successfully or None if it failed.
    out_dict_init : dict or None
        Dictionary output for init config from get_preproc_group_out_dict
        if preprocessing ran successfully for init layer or ``None`` if
        preprocessing was loaded or ``preproc_or_load_group`` failed.
    out_dict_proc : dict or None
        Dictionary output for proc config from get_preproc_group_out_dict
        if preprocessing ran successfully for proc layer or ``None`` if
        preprocessing was loaded, that layer was not run or loaded, or
        ``preproc_or_load_group`` failed.
    errors : tuple
        A tuple containing the error from PreprocessError, an error message,
        and the traceback. Each will be None if preproc_or_load_group finished
        successfully.
    """
    init_temp_subdir = "temp"
    proc_temp_subdir = "temp_proc"

    if compress == True:
        compress = "gzip"
    else:
        compress = None

    if logger is None:
        logger = init_logger("preprocess")

    group = [list(np.fromiter(dets.values(), dtype='<U32'))][0]

    # Do a try except around config and meta reading to catch metadata failures
    try:
        if type(configs_init) == str:
            configs_init = yaml.safe_load(open(configs_init, "r"))

        context_init = core.Context(configs_init["context_file"])
        make_lmsi_init = configs_init.get("lmsi_config") is not None

        if configs_proc is not None:
            if type(configs_proc) == str:
                configs_proc = yaml.safe_load(open(configs_proc, "r"))
            context_proc = core.Context(configs_proc["context_file"])
            make_lmsi_proc = configs_proc.get("lmsi_config") is not None

            # Ensure grouping matches between init and proc configs
            group_by_init = np.atleast_1d(configs_init['subobs'].get('use', 'detset'))
            group_by_proc = np.atleast_1d(configs_proc['subobs'].get('use', 'detset'))

            if (group_by_init != group_by_proc).any():
                raise ValueError('init and proc groups do not match')

    except Exception as e:
        errmsg, tb = PreprocessErrors.get_errors(e)
        logger.error(f"Get configs/context failed for {obs_id}: {group}\n{errmsg}\n{tb}")
        return None, None, None, (PreprocessErrors.MetaDataError, errmsg, tb)

    db_init_exist = find_db(obs_id, configs_init, dets, logger=logger)
    if configs_proc is not None:
        db_proc_exist = find_db(obs_id, configs_proc, dets, logger=logger)
    else:
        db_proc_exist = False

    # Skip entries not in either db if the db exists but entry is not found.
    # Setting overwrite to True will bypass this and re-run.
    if not overwrite and skip_missing:
        # init db exists but entry not in init db
        if (
            os.path.exists(configs_init['archive']['index'])
            and not db_init_exist
        ):
            logger.warn(f"{obs_id}: {group} not found in init db and skip missing={skip_missing}")
            return None, None, None, (PreprocessErrors.SkipMissingError, None, None)
        # proc db exists but entry not in proc db
        if (
            configs_proc is not None
            and os.path.exists(configs_proc['archive']['index'])
            and not db_proc_exist
        ):
            logger.warn(f"{obs_id}: {group} not found in proc db and skip missing={skip_missing}")
            return None, None, None, (PreprocessErrors.SkipMissingError, None, None)

    # Cannot run if proc db exists but init db does not
    if db_proc_exist and not db_init_exist and not overwrite:
        logger.error("loading from proc db requires init db if overwrite is False")
        return None, None, None, (PreprocessErrors.NoInitDbError, None, None)

    # Load first layer only
    if not overwrite:
        if db_init_exist and not db_proc_exist:
            out_dict_init = None
            try:
                logger.info(f"Loading and applying preprocessing for initial layer db on {obs_id}:{group}")
                aman = load_and_preprocess(obs_id=obs_id, dets=dets, configs=configs_init,
                                           logger=logger)
            except Exception as e:
                errmsg, tb = PreprocessErrors.get_errors(e)
                logger.error(f"Initial layer Pipeline Load Error for {obs_id}: {group}\n{errmsg}\n{tb}")
                return None, None, None, (PreprocessErrors.SingleLayerPipelineLoadError, errmsg, tb)

            # Return if not running proc db
            if configs_proc is None:
                logger.info(f"preproc_or_load_group finished successfully for {obs_id}:{group}")
                return aman, out_dict_init, None, (PreprocessErrors.LoadSuccess, None, None)

        # Load first and second layer
        elif db_init_exist and db_proc_exist:
            try:
                logger.info(f"Loading and applying preprocessing for both dbs on {obs_id}:{group}")
                aman = multilayer_load_and_preprocess(obs_id=obs_id, dets=dets, configs_init=configs_init,
                                                      configs_proc=configs_proc, logger=logger,
                                                      ignore_cfg_check=ignore_cfg_check)
                logger.info(f"preproc_or_load_group finished successfully for {obs_id}:{group}")
                return aman, None, None, (PreprocessErrors.LoadSuccess, None, None)
            except Exception as e:
                errmsg, tb = PreprocessErrors.get_errors(e)
                logger.error(f"Multilayer Pipeline Load Error for {obs_id}: {group}\n{errmsg}\n{tb}")
                return None, None, None, (PreprocessErrors.MultilayerPipelineLoadError, errmsg, tb)

    # Run first layer
    if not db_init_exist or overwrite:
        try:
            logger.info(f"Generating new init db entry for {obs_id}: {group}")
            pipe_init = Pipeline(configs_init["process_pipe"],
                                 plot_dir=configs_init["plot_dir"],
                                 logger=logger)
            aman_cfgs_ref = get_pcfg_check_aman(pipe_init)

            out_dict_init = get_preproc_group_out_dict(obs_id,
                                                       configs_init,
                                                       dets,
                                                       subdir=init_temp_subdir)

            aman = context_init.get_obs(obs_id, dets=dets)
            tags = np.array(context_init.obsdb.get(aman.obs_info.obs_id,
                                                   tags=True)['tags'])
            aman.wrap('tags', tags)
            proc_aman, success = pipe_init.run(aman)
            aman.wrap('preprocess', proc_aman)
        except Exception as e:
            errmsg, tb = PreprocessErrors.get_errors(e)
            logger.error(f"Pipeline Run Error for {obs_id}: {group}\n{errmsg}\n{tb}")
            return None, None, None, (PreprocessErrors.InitPipeLineRunError, errmsg, tb)
        if success != 'end':
            logger.error(f"Init Pipeline Step Error for {obs_id}: {group}\nFailed at step {success}")
            return None, None, None, (PreprocessErrors.PipeLineStepError, success, None)

        if save_proc_aman:
            logger.info(f"Saving preprocessing axis manager to "
                        f"{out_dict_init['temp_file']}:{out_dict_init['db_data']['dataset']}")
            encodings = {}
            if compress is not None:
                _get_aman_encodings(encodings, proc_aman)
            proc_aman.save(out_dict_init['temp_file'],
                           out_dict_init['db_data']['dataset'],
                           compression=compress,
                           encodings=encodings,
                           overwrite=overwrite)
            if save_archive:
                logger.info(f"Adding result to init db for {obs_id}: {group}")
                cleanup_mandb(out_dict_init, (obs_id, group), (None, None, None),
                              configs_init, logger=logger, overwrite=overwrite)
        # Make init plots
        if make_lmsi_init:
            new_plots = os.path.join(configs_init["plot_dir"],
                                 f'{str(aman.timestamps[0])[:5]}',
                                 aman.obs_info.obs_id)
            from pathlib import Path
            import lmsi.core as lmsi

            if os.path.exists(new_plots):
                lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                          Path(configs_init["lmsi_config"]),
                          Path(os.path.join(new_plots, 'index.html')))

        # Return if not running proc db
        if configs_proc is None:
            logger.info(f"preproc_or_load_group finished successfully for {obs_id}:{group}")
            return aman, out_dict_init, None, (None, None, None)

    # Run second layer
    if (not db_proc_exist or overwrite) and configs_proc is not None:
        try:
            logger.info(f"Generating new proc db entry for {obs_id}: {group}")
            init_fields = aman.preprocess._fields.copy()
            init_fields.pop('valid_data', None)
            tags_proc = np.array(context_proc.obsdb.get(aman.obs_info.obs_id,
                                                        tags=True)['tags'])
            if "tags" in aman._fields:
                aman.move("tags", None)
            aman.wrap('tags', tags_proc)

            out_dict_proc = get_preproc_group_out_dict(obs_id,
                                                       configs_proc,
                                                       dets=dets,
                                                       subdir=proc_temp_subdir)

            pipe_proc = Pipeline(configs_proc["process_pipe"],
                                 plot_dir=configs_proc["plot_dir"], logger=logger)
            proc_aman, success = pipe_proc.run(aman)
            pipe_init = Pipeline(configs_init["process_pipe"],
                                 plot_dir=configs_init["plot_dir"],
                                 logger=logger)
            proc_aman.wrap('pcfg_ref', get_pcfg_check_aman(pipe_init))

            for init_field in init_fields:
                if init_field in proc_aman:
                    proc_aman.move(init_field, None)
        except Exception as e:
            errmsg, tb = PreprocessErrors.get_errors(e)
            logger.error(f"Pipeline Run Error for {obs_id}: {group}\n{errmsg}\n{tb}")
            return None, out_dict_init, None, (PreprocessErrors.ProcPipeLineRunError, errmsg, tb)
        if success != 'end':
            logger.error(f"Proc Pipeline Step Error for {obs_id}: {group}\nFailed at step {success}")
            return None, out_dict_init, None, (PreprocessErrors.PipeLineStepError, success, None)

        if save_proc_aman:
            logger.info(f"Saving proc axis manager to "
                        f"{out_dict_proc['temp_file']}:{out_dict_proc['db_data']['dataset']}")
            encodings = {}
            if compress is not None:
                _get_aman_encodings(encodings, proc_aman)
            proc_aman.save(out_dict_proc['temp_file'],
                           out_dict_proc['db_data']['dataset'],
                           compression=compress,
                           encodings=encodings,
                           overwrite=overwrite)
            if save_archive:
                logger.info(f"Adding result to proc db for {obs_id}: {group}")
                cleanup_mandb(out_dict_proc, (obs_id, group), (None, None, None),
                              configs_proc, logger=logger, overwrite=overwrite)
        if 'valid_data' in aman.preprocess:
            aman.preprocess.move('valid_data', None)
        aman.preprocess.merge(proc_aman)

        # Make proc plots
        if make_lmsi_proc:
            new_plots = os.path.join(configs_proc["plot_dir"],
                                 f'{str(aman.timestamps[0])[:5]}',
                                 aman.obs_info.obs_id)

            from pathlib import Path
            import lmsi.core as lmsi

            if os.path.exists(new_plots):
                lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                          Path(configs_proc["lmsi_config"]),
                          Path(os.path.join(new_plots, 'index.html')))

    logger.info(f"preproc_or_load_group finished successfully for {obs_id}:{group}")
    return aman, out_dict_init, out_dict_proc, (None, None, None)


def cleanup_mandb(out_dict, out_meta, errors, configs, logger=None, overwrite=False, db_manager=None):
    """Function to update the manifest db when data is collected from the
    ``preproc_or_load_group`` function. If used in an mpi framework this
    function is expected to be run from rank 0 after a ``comm.gather``.
    See the ``preproc_or_load_group`` docstring for the varying expected
    values of ``errors`` and the associated ``out_dict``. This function will
    either:

    1) Update the ManifestDb sqlite file and move the h5 archive from its
    temporary location to its permanent path if errors[0] is ``None``, out_dict
    is not``None``. Deletes the temporary h5 file.

    2) Return nothing if errors[0] is ``PreprocessErrors.LoadSuccess`` or both it
    and out_dict are None.

    3) Otherwise, update the error log.

    Arguments
    ---------
    errors : tuple
         A tuple containing the error from PreprocessError, an error message,
        and the traceback. Each will be None if preproc_or_load_group finished
        successfully.
    out_meta : tuple
        The tuple (obs_id, group).
    outputs : dict
        Dictionary including entries for the temporary h5 filename
        ('temp_file') and the obs_id group metadata and db entry (db_data).
        See save_group for more info.
    configs : dict
        Preprocessing configuration dictionary.
    logger : PythonLogger
        Optional.  Python logger.
    overwrite : bool
        Optional. Delete the entry in the archive file if it exists and
        replace it with the new entry.
    db_manager : DbBatchManager, optional
        External database batch manager for optimized operations.
        If provided, uses the manager instead of creating individual connections.
    """

    if logger is None:
        logger = init_logger("preprocess")

    if out_dict is not None and os.path.isfile(out_dict['temp_file']):
        obs_id, group = out_meta
        logger.info(f"Adding future result to db for {obs_id}: {group}")

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
        group_by = [k.split(':')[-1] for k in out_dict['db_data'].keys() if 'dets' in k]
        h5_path = os.path.relpath(dest_file,
                                  start=os.path.dirname(configs['archive']['index']))

        src_file = out_dict['temp_file']

        with H5ContextManager(dest_file, mode='a') as f_dest:
            with H5ContextManager(src_file, mode='r') as f_src:
                for dts in f_src.keys():
                    # If the dataset or group already exists, delete it to overwrite
                    if overwrite and dts in f_dest:
                        del f_dest[dts]
                    f_src.copy(f_src[f'{dts}'], f_dest, f'{dts}')
                    for member in f_src[dts]:
                        if isinstance(f_src[f'{dts}/{member}'], h5py.Dataset):
                            f_src.copy(f_src[f'{dts}/{member}'], f_dest[f'{dts}'], f'{dts}/{member}')

        if db_manager is not None:
            # Use the batch manager
            db_manager.add_entry(out_dict['db_data'], h5_path)
        else:
            # Use the original approach for backward compatibility
            db = get_preprocess_db(configs, group_by, logger)
            if len(db.inspect(out_dict['db_data'])) == 0:
                db.add_entry(out_dict['db_data'], h5_path)
            # make sure we close the db each time
            db.conn.close()

        os.remove(src_file)
    elif (
        errors[0] == PreprocessErrors.LoadSuccess or
        (errors[0] is None and out_dict is None)
    ):
        return
    else:
        folder = os.path.dirname(configs['archive']['index'])
        if not(os.path.exists(folder)):
            os.makedirs(folder)
        errlog = os.path.join(folder, 'errlog.txt')
        with open(errlog, 'a') as f:
            f.write(f"{time.time()}, {out_meta[0]}, {out_meta[1]}, {errors[0]}\n")
            f.write("\t" + (errors[1] or "") + (errors[2] or "") + "\n")


def get_pcfg_check_aman(pipe):
    """
    Given a preprocess pipeline class return an axis manager containing
    the ordered steps of the pipeline with all arguments for each step.

    Arguments
    ---------
    pipe : _Preprocess class
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
    a : AxisManager
        Primary axis manager to cross check assignments with.
    b : AxisManager
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
