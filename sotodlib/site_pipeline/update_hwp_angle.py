"""
Script for running updates on (or creating) a hwp angle g3 file. 
This script will run periodically even when hwp is not spinning.
Meaning is designed to work from something like a cronjob. 
The output hwp angle should be synchronized to SMuRF timing 
outside this script.
"""
import os
import argparse
import logging
import yaml

import datetime as dt
from sotodlib.hwp.g3thwp import G3tHWP

from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'update-hwp-angle: ')


def get_parser():

    parser = argparse.ArgumentParser(
        description='Analyze HWP encoder data from level-2 HK data, \
                        and produce HWP angle solution for all times.')
    parser.add_argument(
        '-c', '--config-file', default=None, type=str, required=True,
        help="Configuration File for running update_hwp_angle")
    parser.add_argument(
        '-d', '--data-dir', action='store', default=None, type=str,
        help='input data directory, overwrite config data_dir')
    parser.add_argument(
        '-o', '--output-dir', action='store', default=None, type=str,
        help='output data directory, overwrite config output_dir')
    parser.add_argument('--update-delay', default=2, type=float,
        help="Days to subtract from now to set as minimum ctime. \
              Set to 0 to build from scratch",)
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

    logger.info("Starting update_hwp_angle")

    configs = yaml.safe_load(open(args.config_file, "r"))
    if args.data_dir is None:
        args.data_dir = configs["data_dir"]
    
    if args.output_dir is None:
        args.output_dir = configs["output_dir"]

    if not os.path.exists(args.output_dir):
        logger.info(f"Making output directory {args.output_dir}")
        os.mkdir(args.output_dir)
    

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    elif args.verbose == 3:
        logger.setLevel(logging.DEBUG)
    
    if args.update_delay > 0:
        min_time = dt.datetime.now() - dt.timedelta(days=args.update_delay)
        min_ctime = min_time.timestamp()
    else:
        min_ctime = None

    files = []
    existing = []
    for root, _, fs in os.walk(args.data_dir):
        for f in fs:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, args.data_dir)
            if args.file is not None:
                if f in args.file or \
                   any([os.path.exists(_f) and os.path.samefile(full_path, _f)
                        for _f in args.file]):
                    files.append(rel_path)
                continue

            if min_ctime is not None:
                filetime = int(f.split(".")[0])
                if filetime < min_ctime:
                    continue

            out_file = os.path.join(args.output_dir, rel_path)
            if os.path.exists( out_file):
                existing.append(rel_path)
            else:
                files.append(rel_path)
                
    if args.file is not None and len(files) != len(args.file):
        logger.warning(f"Matched only {len(files)} of the {len(args.file)} "
                       "files provided with --file.")

    ## assume the last existing file was incomplete
    if len(existing) > 0:
        files.append( sorted(existing)[-1])

    for f in sorted(files):
        in_file = os.path.join(args.data_dir, f)
        out_file = os.path.join(args.output_dir, f)
        logger.info(f"Processing {in_file}")

        try:
            logger.debug("instance G3tHWP class")
            hwp = G3tHWP(args.config_file)
            data = hwp.load_file(in_file)
            if len(data)==0:
                logger.info(f"Found no HWP data in {f}")
                continue
        
            logger.debug("analyze")
            solved = hwp.analyze(data)

            logger.info(f"writing solution {out_file}")
            if not os.path.exists( os.path.split(out_file)[0]):
                os.mkdir(os.path.split(out_file)[0])
            hwp.write_solution(solved, out_file)

        except Exception as e:
            logger.error(f"Exception '{e}' thrown while processing {in_file}")
            continue

