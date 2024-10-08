import os
import yaml
import time
import logging
import numpy as np
import argparse
import traceback
from typing import Optional
import copy

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes

logger = sp_util.init_logger("preprocess")

def plot_preprocess_tod(obs_id, configs, context, group_list=None, verbosity=2):
    """ Loads the saved information from the preprocessing pipeline and runs the
    plotting section of the pipeline. 

    Assumes preprocess_tod has already been run on the requested observation. 
    
    Arguments
    ----------
    obs_id: multiple
        passed to `context.get_obs` to load AxisManager, see Notes for 
        `context.get_obs`
    configs: string or dictionary
        config file or loaded config directory
    """
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)
    
    group_by, groups = sp_util.get_groups(obs_id, configs, context)
    all_groups = groups.copy()
    for g in all_groups:
        if group_list is not None:
            if g not in group_list:
                groups.remove(g)
                continue
        if 'wafer.bandpass' in group_by:
            if 'NC' in g:
                groups.remove(g)
                continue
        try:
            meta = context.get_meta(obs_id, dets = {gb:gg for gb, gg in zip(group_by, g)})
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {g}\n{errmsg}\n{tb}")
            groups.remove(g)
            continue

        if meta.dets.count == 0:
            groups.remove(g)
        
    if len(groups) == 0:
        logger.warning(f"group_list:{group_list} contains no overlap with "
                       f"groups in observation: {obs_id}:{all_groups}. "
                       f"No analysis to run.")
        return
    
    pipe = Pipeline(configs["process_pipe"], plot_dir=configs["plot_dir"], logger=logger)
    
    for group in groups:
        logger.info(f"Beginning run for {obs_id}:{group}")
        try:
            meta = context.get_meta(obs_id, dets={gb:g for gb, g in zip(group_by, group)})
            aman = context.get_obs(meta)
            pipe.run(aman, aman.preprocess, update_plot=True)
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.info(f"ERROR: {obs_id} {group}\n{errmsg}\n{tb}")
            continue


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
        planet_obs: bool = False,
        verbosity: Optional[int] = None,
 ):
    """The intended use case of this script is: if preprocessing has been previously
    run on a set of observations but plotting was not enabled during that run. Now,
    we re-run preprocessing on those observations to produce the plots, without having
    to fully re-run all the calc steps. Provide the preprocess config file that was
    initially used, this time enabling plotting in the pipeline where necessary. The
    config should point to the existing database with the preprocess data. A query
    should be provided similar to preprocess_tod to define the obs you want to run
    plotting on.

    """
    configs, context = sp_util.get_preprocess_context(configs)
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)

    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) - update_delay*86400

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

    if not(tags is None):
        for i, tag in enumerate(tags):
            tags[i] = tag.lower()
            if '=' not in tag:
                tags[i] += '=1'

    if planet_obs:
        obs_list = []
        for tag in tags:
            obs_list.extend(context.obsdb.query(tot_query, tags=[tag]))
    else:
        obs_list = context.obsdb.query(tot_query, tags=tags)
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []

    if not os.path.exists(configs['archive']['index']):
        # don't run if database doesn't exist
        raise Exception(f"No database found at {configs['archive']['index']}.")
    else:
        db = core.metadata.ManifestDb(configs['archive']['index'])
        for obs in obs_list:
            x = db.inspect({'obs:obs_id': obs["obs_id"]})
            group_by, groups = sp_util.get_groups(obs["obs_id"], configs, context)
            if x is None or len(x) == 0:
                logger.warning(f"Obs_id {obs['obs_id']} not found in database.")
            else:
                run_list.append( (obs, None) )

    logger.info(f"Beginning to run preprocessing on {len(run_list)} observations")
    for obs, groups in run_list:
        obsid = obs["obs_id"]
        ctime = obsid[4:9]
        plot_path = os.path.join(configs['plot_dir'], f'{ctime}/{obsid}')
        logger.info(f"Processing obs_id: {obsid}")
        if not overwrite and os.path.exists(plot_path):
            logger.warning(f"Plots found at {plot_path}. Set overwrite=True to overwrite them.")
            continue
        try:
            plot_preprocess_tod(obsid, configs, context, group_list=groups, verbosity=verbosity)
        except Exception as e:
            logger.info(f"{type(e)}: {e}")
            logger.info(''.join(traceback.format_tb(e.__traceback__)))
            logger.info(f'Skiping obs:{obs["obs_id"]} and moving to the next')
            continue

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
