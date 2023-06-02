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
import sotodlib
from sotodlib import core
from sotodlib.hwp.g3thwp import G3tHWP
from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'make-hwp-solutinos: ')


def get_parser(parser=None):
    if parser is None:
        
        parser = argparse.ArgumentParser()
    parser.add_argument('--context',required=True, help=
                        "Path to context yaml file to define observation for which to generate hwp angle.")
    parser.add_argument('--config-file', required=True, help=
                        "Path to HWP configuration yaml file.")
    parser.add_argument(
        '-o', '--output-dir', action='store', default=None, type=str,
        help='output data directory, overwrite config output_dir')
    parser.add_argument("--verbose", default=2, type=int,
                        help="increase output verbosity. \
                        0: Error, 1: Warning, 2: Info(default), 3: Debug")

    return parser

def main(context=None, config_file=None, output_dir=None, verbose=None):
    
    print(context, config_file)
    configs = yaml.safe_load(open(config_file, "r"))
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = configs["output_dir"]
    
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

    scheme = core.metadata.ManifestScheme()
    scheme.add_exact_match('obs:obs_id')
    scheme.add_data_field('dataset')
    man_db = core.metadata.ManifestDb(scheme=scheme)
    
    # Get file + dataset from policy.
    # policy = util.ArchivePolicy.from_params(config['archive']['policy'])
    # dest_file, dest_dataset = policy.get_dest(obs_id)
    # Use 'output_dir' argument for now    
    man_db_filename = os.path.join(output_dir, 'hwp_angle.sqlite')
    output_filename = os.path.join(output_dir, 'hwp_angle.h5')
    
    # temporary for debugging
    obs_range = 1
    obs = ctx.obsdb.get()[:obs_range]
    for obs_id in obs['obs_id']:
        print(obs_id)
        tod = ctx.get_obs(obs_id, no_signal=True)
        # Wrap result into AxisManager for HDF5 off-load.
        aman = core.AxisManager(tod.dets, tod.samps)
        aman.wrap_new('hwp_angle', shape=('samps', ))
        aman.wrap_new('hwp_angle_eval', shape=('samps', ))
        
        #### Angle calculation ####
        start = int(tod.timestamps[0])
        end = int(tod.timestamps[-1])
        g3thwp = G3tHWP(config_file)
        data = g3thwp.load_data(start, end)

        if len(data)==0:
            logger.info(f"Found no HWP data in {obs_id}")
            continue

        logger.debug("analyze")
        solved = g3thwp.analyze(data)
        
        # template subtracted angle
        try:
            g3thwp.eval_angle(solved)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while the template subtraction")
            continue    
        g3thwp.write_solution_h5(solved, tod, output=output_filename, h5_address=obs_id)
        
        del g3thwp
        """
        aman.hwp_angle = scipy.interpolate.interp1d(
            solved['fast_time'],
            solved['angle'],
            kind='linear',
            fill_value='extrapolate')(tod.timestamps)
        
        # template subtracted angle --- need to fix bug inside
        g3thwp.eval_angle(solved)
        aman.hwp_angle_eval = scipy.interpolate.interp1d(
            solved['fast_time'],
            solved['angle'],
            kind='linear',
            fill_value='extrapolate')(tod.timestamps)

       
        del g3thwp
        #### End of angle calculation ####

        h5_address = obs_id
        aman.save(output_filename, h5_address, overwrite=True)
        """     
        h5_address = obs_id
        # Add an entry to the database
        man_db.add_entry({'obs:obs_id': obs_id, 'dataset': h5_address}, filename=output_filename)

        # Commit the ManifestDb to file.
        man_db.to_file(man_db_filename)
   
    return
    
    
if __name__ == '__main__':
    parser = get_parser()
    util.main_launcher(main, get_parser)
