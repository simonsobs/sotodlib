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
import sqlite3
from typing import Optional, Union, Callable

from sotodlib import core
from sotodlib.hwp.g3thwp import G3tHWP
from sotodlib.site_pipeline import util
from sotodlib.utils.procs_pool import get_exec_env

max_trial = 5
wait_time = 5

# encodings for flacarray compression
encodings = {
    'hwp_angle': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver1_1': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver2_1': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver3_1': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver1_2': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver2_2': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_angle_ver3_2': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_rate_1': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'hwp_rate_2': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 1.0e-8}},
    'timestamps': {'type': 'flacarray', 'args': {'level': 5, 'quanta': 5.0e-5}},
    'quad_1': {'type': 'flacarray'},
    'quad_2': {'type': 'flacarray'},
}

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
    parser.add_argument(
        '--off-site', action='store_true',
        help="This must be true when you run this off site. "
             "If true, this does not try to load L2 data"
    )
    parser.add_argument(
        '--nprocs', default=1, type=int,
        help="number of processes to use. We use nprocs=1"
             " at site computing"
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


def save(aman, db, h5_filename, output_dir, obs_id, overwrite, compression, encodings=None):
    """
    Save HDF5 file and add entry to sqlite database
    Return True if save is succeeded, otherwise return False
    """
    aman_saved = False
    db_saved = False

    for i in range(1, max_trial + 1):
        try:
            aman.save(os.path.join(output_dir, h5_filename), obs_id, overwrite,
                      compression, encodings=encodings)
            logger.info("Saved aman")
            aman_saved = True
            break
        except BlockingIOError:
            logger.warning("Cannot save aman because HDF5 is temporary locked, "
                           f"try again in {wait_time} seconds, trial {i}/{max_trial}")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while saving aman")
            print(traceback.format_exc())
            break
    if not aman_saved:
        logger.error("Cannot save aman, give up.")
        return False

    for i in range(1, max_trial + 1):
        try:
            db.add_entry(
                {'obs:obs_id': obs_id, 'dataset': obs_id}, filename=h5_filename, replace=overwrite,
            )
            logger.info("Added entry to db")
            db_saved = True
            break
        except sqlite3.OperationalError:
            logger.warning(f"Cannot save db, try again in {wait_time} seconds, trial {i}/{max_trial}")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while saving aman")
            print(traceback.format_exc())
            break
    if not db_saved:
        logger.error("Cannot save db, give up.")
        return False

    return True


def get_solution(obs_id, encoder_completed, context, HWPconfig,
                 encoder_output_dir, off_site=False):
    """ calculate hwp angle solution.
    Load encoder L3 data if it exists.
    If this is run off site, load L2 data, otherwise skip it.

    Args
        obs_id: str
        encoder_completed:
            list of obs_id which has L3 encoder data

    Return
        obs_id: str
        aman_encoder: AxisManager
            axis manager of hwp encoder
        aman_solution: AxisManager
            axis manager of hwp angle solution
    """
    logger.info(f"Calculating Angles for {obs_id}")
    # split h5 file by first 5 digits of unixtime
    unix = obs_id.split('_')[1][:5]
    h5_encoder = os.path.join(encoder_output_dir, f'hwp_encoder_{unix}.h5')
    ctx = core.Context(context)
    g3thwp = G3tHWP(HWPconfig)

    # make angle solutions
    tod = ctx.get_obs(obs_id, dets=[])
    # Load L3 data if it exists
    if obs_id in encoder_completed:
        g3thwp.set_data(tod, h5_filename=h5_encoder)
        aman_encoder = None
    # Do nothing if L3 data does not exists off-site
    elif off_site:
        logger.info(f"Skip {obs_id} beause L3 data do not exist off site")
        return obs_id, None, None
    # Load L2 data at site
    else:
        aman_encoder = g3thwp.set_data(tod)

    aman_solution = g3thwp.make_solution(tod)
    return obs_id, aman_encoder, aman_solution


def _main(
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable,
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
    off_site: Optional[bool] = False,
    nprocs: Optional[int] = 1,
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

    db_solution = make_db(os.path.join(solution_output_dir, 'hwp_angle.sqlite'))
    db_encoder = make_db(os.path.join(encoder_output_dir, 'hwp_encoder.sqlite'))
    encoder_completed = db_encoder.get_entries(['dataset'])['dataset']

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
    ctx = core.Context(context)
    obs_list = ctx.obsdb.query(tot_query)
    obs_list = [obs['obs_id'] for obs in obs_list]
    run_list = []

    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")

    completed = db_solution.get_entries(['dataset'])['dataset']

    for obs_id in obs_list:
        if overwrite or obs_id not in completed:
            run_list.append(obs_id)

    # write solutions
    futures = []
    for obs_id in run_list:
        futures.append(executor.submit(get_solution, obs_id, encoder_completed,
                                       context, HWPconfig, encoder_output_dir,
                                       off_site))

    for future in as_completed_callable(futures):
        try:
            obs_id, aman_encoder, aman_solution = future.result()
            # split h5 file by first 5 digits of unixtime
            unix = obs_id.split('_')[1][:5]
            h5_encoder = f'hwp_encoder_{unix}.h5'
            h5_solution = f'hwp_angle_{unix}.h5'
            if aman_encoder is not None:
                logger.info(f"Saving hwp_encoder: {obs_id}")
                success = save(aman_encoder, db_encoder, h5_encoder,
                               encoder_output_dir,
                               obs_id, overwrite, 'gzip')
                if not success:
                    logger.error(f"Failed to save hwp_encoder: {obs_id}, skip saving hwp_angle")
                    continue
            if aman_solution is not None:
                logger.info(f"Saving hwp_angle: {obs_id}")
                success = save(aman_solution, db_solution, h5_solution,
                               solution_output_dir, obs_id, overwrite,
                               compression='gzip', encodings=encodings)
                if not success:
                    logger.error(f"Failed to save hwp_angle: {obs_id}")
        except Exception as e:
            logger.error(e)
    return


def main(args):
    rank, executor, as_completed_callable = get_exec_env(nprocs=args.nprocs)
    if rank == 0:
        _main(executor, as_completed_callable, **vars(args))


if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(args)
