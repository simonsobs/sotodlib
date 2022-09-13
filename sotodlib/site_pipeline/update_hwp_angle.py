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

from sotodlib.hwp.g3thwp import G3tHWP

logger = logging.getLogger(__name__)


def get_parser():

    parser = argparse.ArgumentParser(
        description='Analyze HWP encoder data from level-2 HK data, \
                        and produce HWP angle solution for all times.')
    parser.add_argument(
        '-c', '--config-file', default=None, type=str, required=True,
        help="Configuration File for running update_hwp_angle")
    parser.add_argument(
        '-t', '--time', action='store', default=None, type=int,
        help='time range (ctime integers), overwrite config time range. \
            file name/list will be ignored if you specify this. \
            ex) --time [start timestamp] [end timestamp]',
        nargs=2)
    parser.add_argument(
        '-d', '--data-dir', action='store', default=None, type=str,
        help='input data directory, overwrite config data_dir')
    parser.add_argument(
        '-f', '--file', action='store', default=None, type=str, nargs='*',
        help='filename or list of filenames (to be loaded in order). \
            overwrite yaml file list. \n
            ignored if you specify time range by argument.')
    parser.add_argument(
        '-o', '--output', action='store', default=None, type=str,
        help='path to output g3 file')
    parser.add_argument("--verbose", default=2, type=int,
                        help="increase output verbosity. \
                        0: Error, 1: Warning, 2: Info(default), 3: Debug")

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    logger.info("Starting update_hwp_angle")

    configs = yaml.safe_load(open(args.config_file, "r"))

    logger.debug("instance G3tHWP class")
    hwp = G3tHWP(args.config_file)

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    elif args.verbose == 3:
        logger.setLevel(logging.DEBUG)

    # Load data from arguments or config file
    data = None
    if args.time is not None and args.data_dir is not None:
        data = hwp.load_data(
            args.time[0], args.time[1], archive_path=args.data_dir)
    elif args.time is not None and args.data_dir is None:
        data = hwp.load_data(args.time[0], args.time[1])
    elif args.file is not None:
        data = hwp.load_file(args.file)
    elif 'start' in configs.keys() and 'end' in configs.keys():
        data = hwp.load_data(configs['start'], configs['end'])
    elif 'file' in configs.keys():
        data = hwp.load_file(configs['file'])
    else:
        logger.error("Not specified time range and filenames")
        sys.exit(1)

    logger.debug("analyze")
    solved = hwp.analyze(data)

    logger.debug("write_solution")
    if args.output is not None:
        output = args.output
    elif 'output' in configs.keys():
        output = configs['output']
    hwp.write_solution(solved, output)

    logger.info("output file: " + output)
    logger.info("Finised update_hwp_angle")
