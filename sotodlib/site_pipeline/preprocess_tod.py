import os
import yaml
import numpy as np

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import _Preprocess, PIPELINE, processes

logger = sp_util.init_logger("preprocess")

def _build_pipe_from_configs(configs):
    pipe = []
    for process in configs["process_pipe"]:
        name = process.get("name")
        if name is None:
            raise ValueError(f"Every process step must have a 'name' key")
        cls = PIPELINE.get(name)
        if cls is None:
            logger.warning(f"'{name}' not registered as a pipeline element,"
                            "ignoring")
            continue
        pipe.append(cls(process))
    return pipe

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

def preprocess_tod(obs_id, configs, overwrite=False):
    """
    Arguments
    ----------
    obs_id: obs_id or obs entry (passed to context.get_obs)
    configs: config file or loaded config directory
    """

    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])

    group_by = configs['subobs'].get('use', 'detset')
    if group_by.startswith('dets:'):
        group_by = group_by.split(':',1)[1]

    if group_by == 'detset':
        groups = context.obsfiledb.get_detsets(obs_id)
    else:
        det_info = context.get_det_info(obs_id)
        groups = det_info.subset(keys=[group_by]).distinct()[group_by]

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

    pipe = _build_pipe_from_configs(configs)

    for group in groups:

        aman = context.get_obs(obs_id, dets={group_by:group})
        proc_aman = core.AxisManager( aman.dets, aman.samps)

        for process in pipe:
            logger.info(f"Processing {process.name}")

            process.process(aman) ## make changes to aman
            process.calc_and_save(aman, proc_aman) ## calculate data products

        policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        if group_by == 'detset':
            dest_dataset += '_' + group
        proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

        logger.info("Saving to database")
        # Update the index.
        db_data = {'obs:obs_id': obs_id,
                   'dataset': dest_dataset}
        if group_by != 'detset':
            db_data['dets:'+group_by] = group
        if db.match(db_data) is None:
            db.add_entry(db_data, dest_file)

def load_preprocess_det_select(obs_id, configs, context=None):
    configs, context = _get_preprocess_context(configs, context)
    
    pipe = _build_pipe_from_configs(configs)
    meta = context.get_meta(obs_id)

    for process in pipe:
        logger.info(f"Processing {process.name}")
        process.select(meta)
    return meta

def load_preprocess_tod(obs_id, configs="preprocess_configs.yaml", context=None ):
    configs, context = _get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context)
    
    pipe = _build_pipe_from_configs(configs)
    aman = context.get_obs(meta)
    for process in pipe:
        logger.info(f"Processing {process.name}")
        process.process(aman)
    return aman
