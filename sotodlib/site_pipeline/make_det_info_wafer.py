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
    parser.add_argument('-c', '--config-file',help=
                        "Configuration File for running DetMap")
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
    
    out_dir, out_file = os.path.split(configs["det_info"])
    
    w = "dets:wafer."
    keys = [
        "dets:det_id",
        w+"array",
        w+"bond_pad",
        w+"mux_band",
        w+"mux_channel",
        w+"mux_subband",
        w+"mux_position",
        w+"design_freq_mhz",
        w+"bias_line",
        w+"pol",
        w+"bandpass",
        w+"det_row",
        w+"det_col",
        w+"rhombus",
        w+"type",
        w+"det_x",
        w+"det_y",
        w+"angle",
    ]
    det_rs = core.metadata.ResultSet(keys=keys)
    
    for array_name in array_names:
        # Generate the OperateTuneData for the Array
        map_maker = MapMaker( north_is_highband=False, ## should not matter for this
                              array_name=array_name,
                              dark_bias_lines=None,
                              output_parent_dir=out_dir,
                              verbose=False)
        otd, layout = map_maker.load_metadata()
        otd.map_layout_data(layout)

        ## we CANNOT use otd.tune_data for the for loop.
        ## probably need docs/change control on the IDs in DetMap if that
        ## the source of the IDs
        for tune in otd: 
            #print(tune.detector_id, tune.is_optical)
            # add detector name to database
            det_rs.append({
                "dets:det_id": tune.detector_id,
                w+"array": array_name,
                w+"bond_pad": tune.bond_pad,
                w+"mux_band": str(tune.mux_band),
                w+"mux_channel": none_to_nan(tune.mux_channel),
                w+"mux_subband": none_to_nan(tune.mux_subband),
                w+"mux_position": none_to_nan(tune.mux_layout_position),
                w+"design_freq_mhz": none_to_nan(tune.design_freq_mhz),
                w+"bias_line": none_to_nan(tune.bias_line),
                w+"pol": str(tune.pol),
                w+"bandpass": f"f{tune.bandpass}" if tune.bandpass is not None else "NC",
                w+"det_row": none_to_nan(tune.det_row),
                w+"det_col": none_to_nan(tune.det_col),
                w+"rhombus": str(tune.rhomb),
                w+"type": str(tune.det_type),
                w+"det_x": none_to_nan(tune.det_x),
                w+"det_y": none_to_nan(tune.det_y),
                w+"angle": np.radians(none_to_nan(tune.angle_actual_deg)),
            })
    
    dest_dataset = ",".join(array_names)
    write_dataset(det_rs, configs["det_info"], dest_dataset)
    # Update the index.
    db_data = {'obs:timestamp': [0, 2e11],
               'dataset': dest_dataset}
    db.add_entry(db_data, configs["det_info"])

def none_to_nan(val):
    if val is None:
        return np.nan
    return val

if __name__=='__main__':
    aman = main() 
