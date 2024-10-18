"""
Script to make hwp angle metadata.
This script will run:
1. automatically if data are acquired as preliminary version.
2. New versions of the metadata can be created as the analysis evolves.
"""
import os
import time
import argparse
import logging
import yaml
import traceback
from typing import Optional

from sotodlib import core
from sotodlib.hwp.g3thwp import G3tHWP
from sotodlib.site_pipeline import util

logger = util.init_logger('make_hwp_solutions', 'make-hwp-solutions: ')


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
        '-o', '--solution-output-dir', action='store', default=None, type=str,
        help='output directory of solution metadata, \
        overwrite config solution_output_dir')
    parser.add_argument(
        '--encoder-output-dir', action='store', default=None, type=str,
        help='output directory of encoder metadata, \
        overwrite config encoder_output_dir')
    parser.add_argument(
        '--verbose', default=2, type=int,
        help="increase output verbosity. \
        0: Error, 1: Warning, 2: Info(default), 3: Debug")
    parser.add_argument(
        '--overwrite', action='store_true',
        help="If true, overwrites existing entries in the database",
    )
    parser.add_argument(
        '--load-h5', action='store_true',
        help="If true, try to load raw encoder data from h5 file",
    )
    parser.add_argument(
        '--query', type=str,
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
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
        '--obs-id', type=str, nargs='+',
        help="List of obs-ids of particular observations that you want to run",
    )
    return parser


def make_db(db_filename):
    """
    Make sqlite database
    Get file + dataset from policy.
    policy = util.ArchivePolicy.from_params(config['archive']['policy'])
    dest_file, dest_dataset = policy.get_dest(obs_id)
    Use 'output_dir' argument for now
    """

    if os.path.exists(db_filename):
        logger.info(f"Mapping {db_filename} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(db_filename)
    else:
        logger.info(f"Creating {db_filename} for the "
                    "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            db_filename,
            scheme=scheme
        )
    return db


def save(aman, db, h5_filename, output_dir, obs_id, overwrite, compression):
    """
    Save HDF5 file and add entry to sqlite database
    """
    max_trial = 5
    wait_time = 5
    for i in range(1, max_trial + 1):
        try:
            aman.save(os.path.join(output_dir, h5_filename), obs_id, overwrite, compression)
            logger.info("Saved aman")
            db.add_entry(
                {'obs:obs_id': obs_id, 'dataset': obs_id}, filename=h5_filename, replace=overwrite,
            )
            logger.info("Added entry to db")
            return
        except BlockingIOError:
            logger.warn(f"Cannot save aman because HDF5 is temporary locked, try again in {wait_time} seconds, trial {i}/{max_trial}")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while saving aman")
            print(traceback.format_exc())
            break

    logger.error("Cannot save aman, give up.")


def main(
    context: str,
    HWPconfig: str,
    solution_output_dir: Optional[str] = None,
    encoder_output_dir: Optional[str] = None,
    verbose: Optional[int] = 2,
    overwrite: Optional[bool] = False,
    query: Optional[str] = None,
    min_ctime: Optional[float] = None,
    max_ctime: Optional[float] = None,
    obs_id: Optional[str] = None,
    load_h5: Optional[bool] = False,
 ):
    logger.info(f"Using context {context} and HWPconfig {HWPconfig}")

    configs = yaml.safe_load(open(HWPconfig, "r"))
    logger.info("Starting make_hwp_solutions")

    # Specify output directory
    if solution_output_dir is None:
        solution_output_dir = configs["solution_output_dir"]
    if encoder_output_dir is None:
        encoder_output_dir = configs["encoder_output_dir"]

    if not os.path.exists(solution_output_dir):
        logger.info(f"Making output directory {solution_output_dir}")
        os.mkdir(solution_output_dir)
    if not os.path.exists(encoder_output_dir):
        logger.info(f"Making output directory {encoder_output_dir}")
        os.mkdir(encoder_output_dir)

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

    db_encoder = make_db(os.path.join(encoder_output_dir, 'hwp_encoder.sqlite'))
    db_solution = make_db(os.path.join(solution_output_dir, 'hwp_angle.sqlite'))

    # load observation data
    if obs_id is not None:
        tot_query = ' or '.join([f"(obs_id=='{o}')" for o in obs_id])
    else:
        tot_query = "and "
        if min_ctime is not None:
            tot_query += f"start_time>={min_ctime} and "
        if max_ctime is not None:
            tot_query += f"start_time<={max_ctime} and "
        if query is not None:
            tot_query += query + " and "
        tot_query = tot_query[4:-4]
        if tot_query == "":
            tot_query = "1"

    logger.debug(f"Sending query to obsdb: {tot_query}")
    obs_list = ctx.obsdb.query(tot_query)

    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")
    run_list = []
    completed = db_solution.get_entries(['dataset'])['dataset']

    for obs in obs_list:
        if overwrite or not obs['obs_id'] in completed:
            run_list.append(obs)

    # write solutions
    for obs in run_list:
        obs_id = obs["obs_id"]
        logger.info(f"Calculating Angles for {obs_id}")
        ctx = core.Context(context)
        tod = ctx.get_obs(obs, dets=[])

        # split h5 file by first 5 digits of unixtime
        unix = obs_id.split('_')[1][:5]
        h5_encoder = f'hwp_encoder_{unix}.h5'
        h5_solution = f'hwp_angle_{unix}.h5'

        # make angle solutions
        g3thwp = G3tHWP(HWPconfig)

        if load_h5:
            aman_encoder = g3thwp.set_data(tod, h5_filename=os.path.join(encoder_output_dir, h5_encoder))
        else:
            aman_encoder = g3thwp.set_data(tod)
        logger.info("Saving hwp_encoder")
        save(aman_encoder, db_encoder, h5_encoder, encoder_output_dir, obs_id, overwrite, 'gzip')
        del aman_encoder

        aman_solution = g3thwp.make_solution(tod)
        logger.info("Saving hwp_angle")
        save(aman_solution, db_solution, h5_solution, solution_output_dir, obs_id, overwrite, 'gzip')
        del aman_solution
        del g3thwp

    return


if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))
