import os
import yaml
import numpy as np
import argparse

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import _Preprocess, PIPELINE, processes

logger = sp_util.init_logger("preprocess")

def _build_pipe_from_configs(configs, logger):
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

def _get_groups(obs_id, configs, context):
    group_by = configs['subobs'].get('use', 'detset')
    if group_by.startswith('dets:'):
        group_by = group_by.split(':',1)[1]

    if group_by == 'detset':
        groups = context.obsfiledb.get_detsets(obs_id)
    else:
        det_info = context.get_det_info(obs_id)
        groups = det_info.subset(keys=[group_by]).distinct()[group_by]
    return group_by, groups

def preprocess_tod(obs_id, configs, overwrite=False, logger=None):
    """Meant to be run as part of a batched script, this function calls the
    preprocessing pipeline a specific Observation ID and saves the results in
    the ManifestDb specified in the configs.   

    Arguments
    ----------
    obs_id: string or ResultSet entry
        obs_id or obs entry that is passed to context.get_obs
    configs: string or dictionary
        config file or loaded config directory
    logger: logging instance
        the logger to print to
    """

    if logger is None: 
        logger = sp_util.init_logger("preprocess")
    
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])
    group_by, groups = _get_groups(obs_id, configs, context)

    if os.path.exists(configs['archive']['index']):
        logger.info(f"Mapping {configs['archive']['index']} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(configs['archive']['index'])
    else:
        logger.info(f"Creating {configs['archive']['index']} for the "
                     "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dets:' + group_by)
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            configs['archive']['index'],
            scheme=scheme
        )

    pipe = _build_pipe_from_configs(configs, logger)

    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")

        aman = context.get_obs(obs_id, dets={group_by:group})
        aman, proc_aman = run_preprocess(aman, pipe, logger=logger)

        policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        if group_by == 'detset':
            dest_dataset += '_' + group
        else:
            dest_dataset += "_" + group_by + "_" + str(group)
        proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

        logger.info("Saving to database")
        # Update the index.
        db_data = {'obs:obs_id': obs_id,
                   'dataset': dest_dataset}
        db_data['dets:'+group_by] = group
        
        if db.match(db_data) is None:
            db.add_entry(db_data, dest_file)

def run_preprocess(aman, pipe=None, configs=None, logger=None):
    """Run preprocessing on any loaded AxisManager. Broken out so
    the pipeline can be easily run without databases.

    Arguments
    ---------
    aman: AxisManager
        loaded AxisManager
    pipe: list
        pipeline list as built by _build_pipe_from_configs 
    configs: string or dict 
        a preprocessing config file or loaded config dictionary
    """
    if logger is None: 
        logger = sp_util.init_logger("preprocess")
    
    if pipe is None:
        if configs is None:
            raise ValueError("Either pipe or configs must be specified")
        pipe = _build_pipe_from_configs(configs)

    proc_aman = core.AxisManager( aman.dets, aman.samps)

    for process in pipe:
        logger.info(f"Processing {process.name}")

        process.process(aman, proc_aman) ## make changes to aman
        process.calc_and_save(aman, proc_aman) ## calculate data products
    logger.info("Finished Processing")

    return aman, proc_aman

def load_preprocess_det_select(obs_id, configs, context=None):
    """ Loads the metadata information for the Observation and runs through any
    data selection specified by the Preprocessing Pipeline.

    Arguments
    ----------
    obs_id: multiple
        passed to `context.get_obs` to load AxisManager, see Notes for 
        `context.get_obs`
    configs: string or dictionary
        config file or loaded config directory
    """
    configs, context = _get_preprocess_context(configs, context)
    
    pipe = _build_pipe_from_configs(configs)
    meta = context.get_meta(obs_id)

    for process in pipe:
        logger.info(f"Selecting On {process.name}")
        process.select(meta)
    return meta

def load_preprocess_tod(obs_id, configs="preprocess_configs.yaml", context=None ):
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
    """

    configs, context = _get_preprocess_context(configs, context)
    meta = load_preprocess_det_select(obs_id, configs=configs, context=context)
    
    pipe = _build_pipe_from_configs(configs)
    aman = context.get_obs(meta)
    for process in pipe:
        logger.info(f"Processing {process.name}")
        process.process(aman, aman.preprocess)
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
    return parser

def main(
    configs, 
    query=None, 
    obs_id=None, 
    overwrite=False,
    min_ctime=None,
    max_ctime=None,
    logger=None,
 ):
    configs, context = _get_preprocess_context(configs)
    if logger is None: 
        logger = sp_util.init_logger("preprocess")
    globals()['logger'] = logger

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
    
    obs_list = context.obsdb.query(tot_query)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []

    if not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = obs_list
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.match({'obs:obs_id': obs["obs_id"]}, multi=True)
            group_by, groups = _get_groups(obs["obs_id"], configs, context)
            if overwrite or (x is None or len(x) != len(groups)):
                run_list.append(obs)

    logger.info(f"Beginning to run preprocessing on {len(run_list)} observations")
    for obs in run_list:
        logger.info(f"Processing obs_id: {obs_id}")
        preprocess_tod(obs["obs_id"], configs, overwrite=overwrite,logger=logger)
            

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
