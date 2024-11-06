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

def load_preprocess_tod(obs_id, configs="preprocess_configs.yaml",
                        context=None, dets=None, meta=None):
    """Loads the saved information from the preprocessing pipeline and runs the
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
    configs, context = sp_util.get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context,
                                      dets=dets, meta=meta)

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta)
        pipe.run(aman, aman.preprocess)
        return aman