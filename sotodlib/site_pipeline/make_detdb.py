import os
import sys
import logging
import numpy as np
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib.core.metadata import DetDb, common

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-a', '--array-names',help=
                        "Comma-Separated array names to generate for this DetDb.")
    parser.add_argument('-o', '--out-file', help=
                        "File name to save detdb.")
    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse_args()
    
    array_names = args.array_names.split(',')
    logger.info(f"Creating DetDb for UFMs-{','.join([array for array in array_names])}")
    
    if args.out_file is None:
        out_file = f"detdb.db"
        out_dir = '.'
        out_path = out_file
    else:
        out_path = args.out_file
        out_dir, out_file = os.path.split(args.out_file)
        if out_dir == '':
            out_dir = '.'
    

    detdb = DetDb(map_file=out_path)
    column_defs = [
        "'bond_pad' int",
        "'mux_band' int",
        "'mux_channel' int",
        "'mux_subband' str",
        "'mux_position' int",
        "'design_freq_mhz' float",
        "'bias_line' int",
        "'pol' str",
        "'bandpass' str",
        "'det_row' int",
        "'det_col' int",
        "'rhombus' str",
        "'type' str",
        "'det_x' float",
        "'det_y' float",
        "'angle' float",
    ]
    detdb.create_table('base', column_defs)

    for array_name in array_names:
        # Generate the OperateTuneData for the Array
        map_maker = MapMaker( north_is_highband=False, ## should not matter for this
                              array_name=array_name,
                              dark_bias_lines=None,
                              output_parent_dir=out_dir,
                              verbose=False)
        otd, layout = map_maker.load_metadata()
        otd.map_layout_data(layout)

        for tune in otd.tune_data:
            ## follow tod2maps docs convention to create detector IDs
            duid = None
            dtype = None

            if tune.is_optical is None:
                if tune.bond_pad == -1:
                    duid = f"{array_name}_BARE_Mp{tune.mux_layout_position:02}bNCD"  
                    dtype = 'BARE'
                elif tune.bond_pad == 64:
                    duid = f"{array_name}_SQID_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
                    dtype = 'SQID'
                elif tune.det_row is None:
                    duid = f"{array_name}_UNRT_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
                    dtype = 'UNRT'
                bandpass = None
            elif not tune.is_optical:
                if tune.bandpass == 'NC': ## these are pin/slot resonators
                    duid = f"{array_name}_SLOT_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D" 
                    dtype = 'SLOT'
                else:
                    duid = f"{array_name}_DARK_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
                    dtype = 'DARK'
                bandpass = None
            else:
                duid = f"{array_name}_f{int(tune.bandpass):03}_"+ \
                        f"{tune.rhomb}r{tune.det_row:02}c{tune.det_col:02}{tune.pol}" 
                dtype='OPTC'
                bandpass =  f"f{int(tune.bandpass):03}"

            if duid is None:
                raise ValueError(f"Detector ID not assigned for {repr(tune)}")

            # add detector name to database
            detdb.get_id(duid)
            detdb.add_props('base', duid,
                bond_pad = tune.bond_pad,
                mux_band = tune.mux_band,
                mux_channel = tune.mux_channel,
                mux_subband = tune.mux_subband,
                mux_position = tune.mux_layout_position,
                design_freq_mhz = tune.design_freq_mhz,
                bias_line = tune.bias_line,
                pol = tune.pol,
                bandpass = bandpass,
                det_row = tune.det_row,
                det_col = tune.det_col,
                rhombus = tune.rhomb,
                type = dtype,
                det_x = tune.det_x,
                det_y = tune.det_y,
                angle = np.radians(tune.angle_actual_deg) if tune.angle_actual_deg is not None else None,
                commit=False,
            )
        detdb.conn.commit()


if __name__=='__main__':
    main() 
