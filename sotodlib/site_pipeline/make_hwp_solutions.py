"""
Script to make hwp angle of CalDB.
This script will run:
1. automatically if data are acquired as preliminary version.
2. New versions of the metadata can be created as the analysis evolves.
The output format is sqlite or HDF5.
"""
import os
import argparse
import logging
import yaml

import datetime as dt
from sotodlib.hwp.g3thwp import G3tHWP

from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'make-hwp-solutinos: ')

def get_parser():

    parser = argparse.ArgumentParser(
        description='Output HWP angle as versioned metadata to load using CalDB')
    parser.add_argument(
        '-c', '--config-file', default=None, type=str, required=True,
        help="Configuration file for running make_hwp_solutions")
    parser.add_argument(
        '-d', '--data-dir', action='store', default=None, type=str,
        help='input data directory, overwrite config data_dir')
    parser.add_argument(
        '-o', '--output-dir', action='store', default=None, type=str,
        help='output data directory, overwrite config output_dir')
    parser.add_argument('--file', default=None, action='append',
        help="Force processing of a specific file, overriding the "
             "standard selection process.  The file must be in the "
             "usual data tree, though.  You may specify either the file "
             "basename (1234567890.g3) or the full path.")
    parser.add_argument("--verbose", default=2, type=int,
                        help="increase output verbosity. \
                        0: Error, 1: Warning, 2: Info(default), 3: Debug")

    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    logger.info("Starting make_hwp_solutions")

    # Load output files from update_hwp_angle.py
    configs = yaml.safe_load(open(args.config_file, "r"))
    if args.data_dir is None:
        args.data_dir = configs["data_dir"]

    # Specify output directory
    if args.output_dir is None:
        args.output_dir = configs["output_dir"]

    if not os.path.exists(args.output_dir):
        logger.info(f"Making output directory {args.output_dir}")
        os.mkdir(args.output_dir)
    
    # Set verbose
    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    elif args.verbose == 3:
        logger.setLevel(logging.DEBUG)

    if args.file is not None and len(files) != len(args.file):
        logger.warning(f"Matched only {len(files)} of the {len(args.file)} "
                       "files provided with --file.")

    ########## Put main process here ###########
