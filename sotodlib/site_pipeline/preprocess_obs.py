import os
import yaml
import time
import numpy as np
import argparse
import traceback
from typing import Optional

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

def preprocess_obs(
    obs_id, 
    configs,  
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

    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)

    logger.info(f"Beginning run for {obs_id}")

    aman = context.get_obs(obs_id, no_signal=True)
    proc_aman, success = pipe.run(aman)
    if success != 'end':
        return

    policy = sp_util.ArchivePolicy.from_params(configs['archive']['policy'])
    dest_file, dest_dataset = policy.get_dest(obs_id)
    logger.info(f"Saving data to {dest_file}:{dest_dataset}")
    proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

    # Update the index.
    db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
    
    logger.info(f"Saving to database under {db_data}")
    if len(db.inspect(db_data)) == 0:
        db.add_entry(db_data, dest_file)

def load_preprocess_obs(obs_id, configs="preprocess_obs_configs.yaml", context=None ):
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
    
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    aman = context.get_obs(meta, no_signal=True)
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
 ):
    configs, context = _get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess")
    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) + update_delay

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
    
    for i, tag in enumerate(tags):
        tags[i] = tags[i].lower()
        if '=' not in tags[i]:
            tags[i] += '=1'

    obs_list = context.obsdb.query(tot_query, tags=tags)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = [o for o in obs_list]
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            if x is None or len(x) == 0:
                run_list.append(obs)

    logger.info(f"Beginning to run preprocessing on {len(run_list)} observations")
    for obs in run_list:
        logger.info(f"Processing obs_id: {obs_id}")
        try:
            preprocess_obs(obs["obs_id"], configs, overwrite=overwrite, logger=logger)
        except Exception as e:
            logger.info(f"{type(e)}: {e}")
            logger.info(''.join(traceback.format_tb(e.__traceback__)))
            logger.info(f'Skiping obs:{obs["obs_id"]} and moving to the next')
            continue
            

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
