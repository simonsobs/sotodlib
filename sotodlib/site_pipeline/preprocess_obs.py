import os
import yaml
import time
import numpy as np
import argparse
import traceback
from typing import Optional, List

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")

def preprocess_obs(
    obs_id, 
    configs,  
    overwrite=False, 
    logger=None,
    obs_group=None
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
    obs_group: list of strings
        List of obs_ids within group
    """

    if logger is None: 
        logger = sp_util.init_logger("preprocess")
    
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))

    context = core.Context(configs["context_file"])

    source_list = configs.get('source_list', None)
    source_names = []
    for _s in source_list:
        if isinstance(_s, str):
            source_names.append(_s)
        elif len(_s) == 3:
            source_names.append(_s[0])
        else:
            raise ValueError('Invalid style of source')
 
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
        scheme.add_data_field('coverage')
        scheme.add_data_field('source_distance')
        if obs_group:
            scheme.add_data_field('obs_group')
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

    policy = pp_util.ArchivePolicy.from_params(configs['archive']['policy'])
    dest_file, dest_dataset = policy.get_dest(obs_id)
    logger.info(f"Saving data to {dest_file}:{dest_dataset}")
    proc_aman.save(dest_file, dest_dataset, overwrite=overwrite)

    # Update the index.
    db_data = {'obs:obs_id': obs_id,
                'dataset': dest_dataset}
    
    sso_footprint_process = False
    for process in configs['process_pipe']:
        if 'name' in process and 'sso_footprint' == process['name']:
            sso_footprint_process = True

    if sso_footprint_process:
        logger.info(f"Saving per source to database {db_data}")
        nearby_source_names = []
        for _source in proc_aman.sso_footprint._assignments.keys():
            nearby_source_names.append(_source)
        coverage = []
        distances = []
        for source_name in source_names:
            if source_name in nearby_source_names:
                for key in proc_aman.sso_footprint[source_name]._assignments.keys():
                    if 'ws' in key:
                        if proc_aman.sso_footprint[source_name][key]:
                            coverage.append(f"{source_name}:{key}")

                distances.append(f"{source_name}:{proc_aman.sso_footprint[source_name]['mean_distance']}")
        db_data['coverage'] = ','.join(coverage)
        db_data['source_distance'] = ','.join(distances)
    else:
        db_data['coverage'] = None
        db_data['source_distance'] = None

    if obs_group:
        db_data['obs_group'] = ','.join(obs_group)
    
    logger.info(f"Saving to database under {db_data}")
    if len(db.inspect(db_data)) == 0:
        h5_path = os.path.relpath(dest_file,
                start=os.path.dirname(configs['archive']['index']))
        db.add_entry(db_data, h5_path)

    if configs.get("lmsi_config", None) is not None:
        from pathlib import Path
        import lmsi.core as lmsi

        new_plots = os.path.join(configs["plot_dir"],
                                 f'{str(aman.timestamps[0])[:5]}',
                                 aman.obs_info.obs_id)
        if os.path.exists(new_plots):
            lmsi.core([Path(x.name) for x in Path(new_plots).glob("*.png")],
                      Path(configs["lmsi_config"]),
                      Path(os.path.join(new_plots, 'index.html')))

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
        '--lat',
        help="If true, filter obs list to only keep the first obs_ids of those with \
              timestamps within 10 seconds, since lat obs_ids are split by optics tube. \
              We only need one for preprocess_obs loads the entire aman without signal.",
        action='store_true',
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
    tags: Optional[List[str]] = None,
    planet_obs: bool = False,
    verbosity: Optional[int] = None,
    lat: bool = False,
 ):
    configs, context = pp_util.get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)
    
    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id, min_ctime=min_ctime, 
                                   max_ctime=max_ctime, update_delay=update_delay, tags=tags, 
                                   planet_obs=planet_obs)
    
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []
    
    if lat:
        time_tolerance = 10
        kept = []
        seen_ctimes = []
        obs_groups = []
        current_group = []

        for s in obs_list:
            try:
                ctime_str = s['obs_id'].split('_')[1]
                ctime = int(ctime_str)
            except (IndexError, ValueError):
                continue

            is_close = any(abs(ctime - seen) <= time_tolerance for seen in seen_ctimes)

            if not is_close:
                if current_group:
                    obs_groups.append(current_group)
                current_group = [s['obs_id']]
                kept.append(s)
                seen_ctimes = [ctime]
            else:
                current_group.append(s['obs_id'])
                seen_ctimes.append(ctime)

        if current_group:
            obs_groups.append(current_group)

        obs_list = kept
    else:
        obs_groups = None

    if overwrite or not os.path.exists(configs['archive']['index']):
        #run on all if database doesn't exist
        run_list = [o for o in obs_list]
    else:
        mask = []
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            if x is None or len(x) == 0:
                run_list.append(obs)
                mask.append(True)
            else:
                mask.append(False)
        
        if obs_groups:
            obs_groups = [group for keep, group in zip(mask, obs_groups) if keep]

    logger.info(f"Beginning to run preprocessing on {len(run_list)} observations")
    for i, obs in enumerate(run_list):
        logger.info(f"Processing obs_id: {obs['obs_id']}")
        try:
            preprocess_obs(obs["obs_id"], configs, overwrite=overwrite, logger=logger, 
                           obs_group=None if obs_groups is None else obs_groups[i])
        except Exception as e:
            logger.info(f"{type(e)}: {e}")
            logger.info(''.join(traceback.format_tb(e.__traceback__)))
            logger.info(f'Skiping obs:{obs["obs_id"]} and moving to the next')
            continue
            

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
