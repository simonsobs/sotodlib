import os
import sys
import yaml
import logging
import numpy as np
from argparse import ArgumentParser

import sotodlib
from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io import so_ufm
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__)


def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('config_file',
        help="Configuration file name.")
    parser.add_argument('target', nargs='*', default=None,
                        help="Target override: array_name stream_id.")
    parser.add_argument('--overwrite', action='store_true', 
        help="Overwrite existing entries.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log-file', help="Output log filename")
    return parser

def get_array_name(stream_id):
    # ufm_mv56 -> Mv56
    return stream_id.split('_')[1].capitalize()

def main(config_file=None, target=None, overwrite=False, debug=False, log_file=None):
    if debug:
        logger.setLevel("DEBUG")
    
    if log_file is not None:
        formatter = util._ReltimeFormatter('%(asctime)s: %(message)s (%(levelname)s)')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    config = yaml.safe_load(open(config_file, "r"))

    if target is not None and len(target) > 0:
        assert(len(target) == 2)  # array_name stream_id
        targets = [tuple(target)]
    else:
        targets = [(get_array_name(stream_id), stream_id) for stream_id in config['stream_ids']]

    array_names = [target[0] for target in targets]
    logger.info(f"Requested Det Info for UFMs: {','.join([array for array in array_names])}")

    # Outputs
    db_file = os.path.join(config['output_dir'], 'wafer_info.sqlite')
    h5_rel = 'wafer_info.h5'
    h5_file = os.path.join(config['output_dir'], h5_rel)

    if os.path.exists(db_file):
        logger.info(f"Det Info {db_file} exists, looking for updates")
        db = core.metadata.ManifestDb(db_file)
        existing = list(db.get_entries(["dataset"])["dataset"])
    else:
        os.makedirs(config['output_dir'], exist_ok=True)
        logger.info(f"Creating Det Info {db_file}.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_data_field('dets:stream_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(db_file, scheme=scheme)
        existing = []

    for array_name, stream_id in targets:

        # make result set per array
        if array_name in existing and not overwrite:
            logger.info(f"{array_name} exists in database, pass --overwrite to"
            " overwrite existing information.")
            continue
        else:
            logger.info(f"Creating entry for {array_name}.")

        # Get one row per resonator
        rows = so_ufm.get_wafer_info(array_name, config, {},
                                     include_no_match=True)

        # Output key names are formatted for det_info consumption.
        prefix = "dets:wafer."
        key_map = {}
        for k in rows[0].keys():
            if k == 'det_id':
                key_map[k] = 'dets:' + k
            else:
                key_map[k] = prefix + k

        det_rs = core.metadata.ResultSet(keys=list(key_map.values()))
        for row in rows:
            det_rs.append({key_map[k]: v for k, v in row.items()})

        write_dataset(det_rs, h5_file, array_name, overwrite)

        # Update the index if it's a new entry
        if not array_name in existing:
            db_data = {'dets:stream_id': stream_id,
                       'dataset': array_name}
            db.add_entry(db_data, h5_rel)


def replace_none(val, replace_val=np.nan):
    if val is None:
        return replace_val
    return val


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
