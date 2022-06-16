import os
import sys
import yaml
import logging
import numpy as np
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib import core
from sotodlib.io.metadata import write_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', help="Configuration File for running DetMap")
    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    configs = yaml.safe_load(open(args.config_file, "r"))

    array_names = [array["name"] for array in configs["arrays"]]
    logger.info(f"Creating Det Info for UFMs-{','.join([array for array in array_names])}")

    scheme = core.metadata.ManifestScheme()
    scheme.add_range_match('obs:timestamp')
    scheme.add_data_field('dataset')
    db = core.metadata.ManifestDb(configs["det_db"], scheme=scheme)

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
    ]
    det_rs = core.metadata.ResultSet(keys=keys)

    for array_name in array_names:
        # Initialize a detmap.makemap.MapMaker() instance 
        # that will have ideal/design metadata for this array.
        map_maker = MapMaker(north_is_highband=False,                              
                             array_name=array_name,
                             verbose=False)

        # iterate over the ideal/design metadata for this array
        for tune in map_maker.grab_metadata():
            
            # add detector name to database
            det_rs.append({
                "dets:det_id": tune.detector_id,
                w + "array": array_name,
                w + "bond_pad": tune.bond_pad,
                w + "mux_band": str(tune.mux_band),
                w + "mux_channel": replace_none(tune.mux_channel, -1),
                w + "mux_subband": replace_none(tune.mux_subband, -1),
                w + "mux_position": replace_none(tune.mux_layout_position, -1),
                w + "design_freq_mhz": replace_none(tune.design_freq_mhz),
                w + "bias_line": replace_none(tune.bias_line, -1),
                w + "pol": str(tune.pol),
                w + "bandpass": f"f{tune.bandpass}" if tune.bandpass is not None else "NC",
                w + "det_row": replace_none(tune.det_row, -1),
                w + "det_col": replace_none(tune.det_col, -1),
                w + "rhombus": str(tune.rhomb),
                w + "type": str(tune.det_type),
                w + "det_x": replace_none(tune.det_x),
                w + "det_y": replace_none(tune.det_y),
                w + "angle": np.radians(replace_none(tune.angle_actual_deg)),
            })

    dest_dataset = ",".join(array_names)
    write_dataset(det_rs, configs["det_info"], dest_dataset)
    # Update the index.
    db_data = {'obs:timestamp': [0, 2e11],
               'dataset': dest_dataset}
    db.add_entry(db_data, configs["det_info"])

    return None

def replace_none(val, replace_val=np.nan):
    if val is None:
        return replace_val
    return val


if __name__ == '__main__':
    aman = main()
