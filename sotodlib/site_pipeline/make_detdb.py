import os
import sys
import logging
import numpy as np
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib.core import AxisManager
from sotodlib.core.metadata import ResultSet

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-a', '--array-names',help=
                        "Comma-Separated array names to generate for this"
                        " DetInfo.")
    parser.add_argument('-o', '--out-file', help=
                        "File name to save det info.")
    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse_args()
    
    array_names = args.array_names.split(',')
    logger.info(f"Creating Det Info for UFMs-{','.join([array for array in array_names])}")
    
    if args.out_file is None:
        out_file = f"det_info.h5"
        out_dir = '.'
        out_path = out_file
    else:
        out_path = args.out_file
        out_dir, out_file = os.path.split(args.out_file)
        if out_dir == '':
            out_dir = '.'
    
    
    keys = [
        "dets",
        "array",
        "bond_pad",
        "mux_band",
        "mux_channel",
        "mux_subband",
        "mux_position",
        "design_freq_mhz",
        "bias_line",
        "pol",
        "bandpass",
        "det_row",
        "det_col",
        "rhombus",
        "type",
        "det_x",
        "det_y",
        "angle",
    ]
    det_rs = ResultSet(keys=keys)
    
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
                "dets": tune.detector_id,
                "array": array_name,
                "bond_pad": tune.bond_pad,
                "mux_band": str(tune.mux_band),
                "mux_channel": none_to_nan(tune.mux_channel),
                "mux_subband": none_to_nan(tune.mux_subband),
                "mux_position": none_to_nan(tune.mux_layout_position),
                "design_freq_mhz": none_to_nan(tune.design_freq_mhz),
                "bias_line": none_to_nan(tune.bias_line),
                "pol": str(tune.pol),
                "bandpass": f"f{tune.bandpass}" if tune.bandpass is not None else "NC",
                "det_row": none_to_nan(tune.det_row),
                "det_col": none_to_nan(tune.det_col),
                "rhombus": str(tune.rhomb),
                "type": str(tune.det_type),
                "det_x": none_to_nan(tune.det_x),
                "det_y": none_to_nan(tune.det_y),
                "angle": np.radians(none_to_nan(tune.angle_actual_deg)),
            })
    
    aman = det_rs.to_axismanager()
    aman.save(out_path)
    return aman

def none_to_nan(val):
    if val is None:
        return np.nan
    return val

if __name__=='__main__':
    aman = main() 
