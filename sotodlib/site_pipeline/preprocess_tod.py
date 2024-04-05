import os
import yaml
import numpy as np
import argparse
import traceback

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
    return group_by, list(groups)

def preprocess_tod(
    obs_id, 
    configs, 
    group_list=None, 
    overwrite=False, 
    logger=None
):
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
    """

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
            return
 
    if os.path.exists(configs['archive']['index']):
        logger.info(f"Mapping {configs['archive']['index']} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(configs['archive']['index'])
    else:
        logger.info(f"Creating {configs['archive']['index']} for the "
                     "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_exact_match('dets:' + group_by)
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            configs['archive']['index'],
            scheme=scheme
        )

    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)

    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")

        aman = context.get_obs(obs_id, dets={group_by:group})
        proc_aman, success = pipe.run(aman)
        if success != 'end':
            continue

        policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        if group_by == 'detset':
            dest_dataset += '_' + group
        else:
            dest_dataset += "_" + group_by + "_" + str(group)
        logger.info(f"Saving data to {dest_file}:{dest_dataset}")
        proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

        # Update the index.
        db_data = {'obs:obs_id': obs_id,
                   'dataset': dest_dataset}
        db_data['dets:'+group_by] = group
        
        logger.info(f"Saving to database under {db_data}")
        if len(db.inspect(db_data)) == 0:
            db.add_entry(db_data, dest_file)

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
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    
    meta = context.get_meta(obs_id)
    logger.info(f"Cutting on the last process: {pipe[-1].name}")
    pipe[-1].select(meta)
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

    if meta.dets.count == 0:
        logger.info(f"No detectors left after cuts in obs {obs_id}")
        return None
    else:
        pipe = Pipeline(configs["process_pipe"], logger=logger)
        aman = context.get_obs(meta)
        pipe.run(aman, aman.preprocess)
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
        configs: str,
        query: Optional[str] = None, 
        obs_id: Optional[str] = None, 
        overwrite: bool = False,
        min_ctime: Optional[int] = None,
        max_ctime: Optional[int] = None,
 ):
    configs, context = _get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess")

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

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = [ (o,None) for o in obs_list]
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            group_by, groups = _get_groups(obs["obs_id"], configs, context)
            if x is None or len(x) == 0:
                run_list.append( (obs, None) )
            elif len(x) != len(groups):
                [groups.remove(a[f'dets:{group_by}']) for a in x]
                run_list.append( (obs, groups) )

    logger.info(f"Beginning to run preprocessing on {len(run_list)} observations")
    for obs, groups in run_list:
        logger.info(f"Processing obs_id: {obs_id}")
        try:
            preprocess_tod(obs["obs_id"], configs, overwrite=overwrite,
                           group_list=groups, logger=logger)
        except Exception as e:
            logger.info(f"{type(e)}: {e}")
            logger.info(''.join(traceback.format_tb(e.__traceback__)))
            logger.info(f'Skiping obs:{obs["obs_id"]} and moving to the next')
            continue
            

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
