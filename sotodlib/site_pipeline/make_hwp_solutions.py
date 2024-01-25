"""
Script to make hwp angle of CalDB.
This script will run:
1. automatically if data are acquired as preliminary version.
2. New versions of the metadata can be created as the analysis evolves.
"""
import sys
import os
import argparse
import logging
import yaml
import datetime as dt
import scipy 
from typing import Optional

import sotodlib
from sotodlib import core
from sotodlib.hwp.g3thwp import G3tHWP
from sotodlib.site_pipeline import util
default_logger = util.init_logger(__name__, 'make-hwp-solutions: ')

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        'context', 
        help="Path to context yaml file to define observation\
        for which to generate hwp angle.")
    parser.add_argument(
        'HWPconfig', 
        help="Path to HWP configuration yaml file.")
    parser.add_argument(
        '-o', '--output-dir', action='store', default=None, type=str,
        help='output data directory, overwrite config output_dir')
    parser.add_argument(
        '--verbose', default=2, type=int,
        help="increase output verbosity. \
        0: Error, 1: Warning, 2: Info(default), 3: Debug")
    parser.add_argument(
        '--overwrite',
        help="If true, overwrites existing entries in the database",
        action='store_true',
    )
    parser.add_argument(
        '--query', 
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",  
        type=str
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
        '--obs-id',
        help="obs-id of particular observation if we want to run on just one",
    )
    return parser

def main(
    context: str, 
    HWPconfig: str, 
    output_dir:Optional[str] = None, 
    verbose:Optional[int] = 2,
    overwrite:Optional[bool] = False,
    query:Optional[str] = None,
    min_ctime:Optional[float] = None,
    max_ctime:Optional[float] = None,
    obs_id:Optional[str] = None,    
    logger=None,
 ):
    if logger is None:
        logger = default_logger
    logger.info(f"Using context {context} and HWPconfig {HWPconfig}")
    
    configs = yaml.safe_load(open(HWPconfig, "r"))
    logger.info("Starting make_hwp_solutions")

    # Specify output directory
    if output_dir is None:
        output_dir = configs["output_dir"]

    if not os.path.exists(output_dir):
        logger.info(f"Making output directory {output_dir}")
        os.mkdir(output_dir)
    
    # Set verbose
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.INFO)
    elif verbose == 3:
        logger.setLevel(logging.DEBUG)
        
    ctx = core.Context(context)

    # Get file + dataset from policy.
    # policy = util.ArchivePolicy.from_params(config['archive']['policy'])
    # dest_file, dest_dataset = policy.get_dest(obs_id)
    # Use 'output_dir' argument for now    
    h5_filename = 'hwp_angle.h5'
    man_db_filename = os.path.join(output_dir, 'hwp_angle.sqlite')
    output_filename = os.path.join(output_dir, h5_filename)
    
    if os.path.exists(man_db_filename):
        logger.info(f"Mapping {man_db_filename} for the "
                    "archive index.")
        man_db = core.metadata.ManifestDb(man_db_filename)
    else: 
        logger.info(f"Creating {man_db_filename} for the "
                     "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        man_db = core.metadata.ManifestDb(
            man_db_filename,
            scheme=scheme
        )

    # load observation data
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

    logger.debug(f"Sending query to obsdb: {tot_query}")
    obs_list = ctx.obsdb.query(tot_query)
    print(obs_list['obs_id'])
        
    if len(obs_list)==0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []
    completed = man_db.get_entries(['dataset'])['dataset']

    for obs in obs_list:
        if overwrite or not obs['obs_id'] in completed:
            run_list.append(obs)

    #write solutions
    for obs in run_list:
        h5_address = obs["obs_id"]
        logger.info(f"Calculating Angles for {h5_address}")
        ctx = core.Context(context)
        tod = ctx.get_obs(obs, no_signal=True)
        
        # make angle solutions
        g3thwp = G3tHWP(HWPconfig)
        g3thwp.write_solution_h5(
            tod, 
            output=output_filename, 
            h5_address=h5_address
        )
        
        del g3thwp
        
        # Add an entry to the database
        man_db.add_entry(
            {'obs:obs_id': obs["obs_id"], 'dataset': h5_address}, filename=h5_filename,
        )   
    return
    
    
if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
