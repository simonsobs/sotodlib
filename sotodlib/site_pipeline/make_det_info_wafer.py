import os
import sys
import yaml
import logging
import numpy as np
from argparse import ArgumentParser

from detmap.makemap import MapMaker
import sotodlib
from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__)

def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('config_file',
        help="Configuration file name.")
    parser.add_argument('--overwrite', action='store_true', 
        help="Overwrite existing entries.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log-file', help="Output log filename")
    return parser

def main(config_file=None, overwrite=False, debug=False, log_file=None):
    if debug:
        logger.setLevel("DEBUG")
    
    if log_file is not None:
        formatter = util._ReltimeFormatter('%(asctime)s: %(message)s (%(levelname)s)')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    configs = yaml.safe_load(open(config_file, "r"))
    array_names = [array["name"] for array in configs["arrays"]]
    logger.info(f"Creating Det Info for UFMs-{','.join([array for array in array_names])}")

    if os.path.exists(configs["det_db"]):
        logger.info(f"Det Info {configs['det_db']} exists, looking for updates")
        db = core.metadata.ManifestDb(configs["det_db"])
        existing = list(db.get_entries(["dataset"])["dataset"])
    else:
        logger.info(f"Creating Det Info {configs['det_db']}.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_data_field('dets:stream_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(configs["det_db"], scheme=scheme)
        existing = []

    w = "dets:wafer."
    keys = [
        "dets:det_id",
        w + "array",
        w + "bond_pad",
        w + "mux_band",
        w + "mux_channel",
        w + "mux_subband",
        w + "mux_position",
        w + "design_freq_mhz",
        w + "bias_line",
        w + "pol",
        w + "bandpass",
        w + "det_row",
        w + "det_col",
        w + "rhombus",
        w + "type",
        w + "det_x",
        w + "det_y",
        w + "angle",
        w + "coax",
    ]

    for array_cfg in configs['arrays']:
        array_name = array_cfg['name']
        stream_id = array_cfg['stream_id']

        # make result set per array
        if array_name in existing and not overwrite:
            logger.info(f"{array_name} exists in database, pass --overwrite to"
            " overwrite existing information.")
            continue
        det_rs = core.metadata.ResultSet(keys=keys)
        
        # Initialize a detmap.makemap.MapMaker() instance 
        # that will have ideal/design metadata for this array.
        map_maker = MapMaker(north_is_highband=False,                              
                             array_name=array_name,
                             verbose=False)

        # iterate over the ideal/design metadata for this array
        for tune in map_maker.grab_metadata():
            
            if tune.bandpass is None or (type(tune.bandpass)==str and "NC" in tune.bandpass):
                bp = "NC"    
            else:
                bp = f"f{str(tune.bandpass).rjust(3,'0')}" 

            # some dark detectors are reporting non-nan angles            
            if str(tune.det_type) == "OPTC":
                angle = np.radians(replace_none(tune.angle_actual_deg))
            else:
                angle = np.nan

            # recapitalize Mv in det_id
            det_id = tune.detector_id
            det_id = det_id[0].upper() + det_id[1:]

            # add detector name to database
            det_rs.append({
                "dets:det_id": det_id,
                w + "array": array_name,
                w + "bond_pad": tune.bond_pad,
                w + "mux_band": str(tune.mux_band),
                w + "mux_channel": replace_none(tune.mux_channel, -1),
                w + "mux_subband": replace_none(tune.mux_subband, -1),
                w + "mux_position": replace_none(tune.mux_layout_position, -1),
                w + "design_freq_mhz": replace_none(tune.design_freq_mhz),
                w + "bias_line": replace_none(tune.bias_line, -1),
                w + "pol": str(tune.pol),
                w + "bandpass": bp,
                w + "det_row": replace_none(tune.det_row, -1),
                w + "det_col": replace_none(tune.det_col, -1),
                w + "rhombus": str(tune.rhomb),
                w + "type": str(tune.det_type),
                w + "det_x": replace_none(tune.det_x),
                w + "det_y": replace_none(tune.det_y),
                w + "angle": angle,
                w + "coax" : "N" if tune.is_north else "S",
            })

        det_rs.append({
                "dets:det_id": "NO_MATCH",
                w + "array": array_name,
                w + "bond_pad": -1,
                w + "mux_band": -1,
                w + "mux_channel": -1,
                w + "mux_subband": -1,
                w + "mux_position": -1,
                w + "design_freq_mhz": np.nan,
                w + "bias_line": -1,
                w + "pol": "NC",
                w + "bandpass": "NC",
                w + "det_row": -1,
                w + "det_col": -1,
                w + "rhombus": "NC",
                w + "type": "NC",
                w + "det_x": np.nan,
                w + "det_y": np.nan,
                w + "angle": np.nan,
                w + "coax" : "X",
        })
            
        write_dataset(det_rs, configs["det_info"], array_name, overwrite)
        # Update the index if it's a new entry
        if not array_name in existing:
            db_data = {'dets:stream_id': stream_id,
                       'dataset': array_name}
            db.add_entry(db_data, configs["det_info"])


def replace_none(val, replace_val=np.nan):
    if val is None:
        return replace_val
    return val


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
