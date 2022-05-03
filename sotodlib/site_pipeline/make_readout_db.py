import os
import sys
import yaml
import logging
import numpy as np
import datetime as dt
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib.core.metadata import DetDb
from sotodlib.io.load_smurf import G3tSmurf, TuneSets

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file',help=
                        "Configuration File for running DetMap")
    parser.add_argument('--min-ctime', type=float, help=
                        "Minimum ctime to search for TuneSets")
    parser.add_argument('--max-ctime', type=float, help=
                        "Maximum ctime to search for TuneSets")
    args = parser.parse_args()
    return args

def get_detector_id(array_name, tune):
    ## follow tod2maps docs convention to create detector IDs
    ## will be replaced by DetMap function in the near future
    duid = None

    if tune.is_optical is None:
        if tune.bond_pad == -1:
            duid = f"{array_name}_BARE_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"  
        elif tune.bond_pad == 64:
            duid = f"{array_name}_SQID_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
        elif tune.det_row is None:
            duid = f"{array_name}_UNRT_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
    elif not tune.is_optical:
        if tune.bandpass == 'NC': ## these are pin/slot resonators
            duid = f"{array_name}_SLOT_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D" 
        else:
            duid = f"{array_name}_DARK_Mp{tune.mux_layout_position:02}b{tune.bond_pad:02}D"
    else:
        duid = f"{array_name}_f{int(tune.bandpass):03}_"+ \
                f"{tune.rhomb}r{tune.det_row:02}c{tune.det_col:02}{tune.pol}" 
    return duid

def create_table(readout_db, name):
    column_defs = [
        "'readout_id' str",
        "'tuneset' str"
    ]
    if name not in readout_db._get_property_tables():
        logger.debug(f"Creating Table {name}")
        readout_db.create_table(name, column_defs)
    else:
        logger.debug(f"Table {name} exists")

def setup_readout_db(configs):
    
    if not os.path.exists(configs['readout_db']):
        logger.info(f"Created Readout ID database at {configs['readout_db']}")
        readout_db = DetDb(map_file=configs["readout_db"])
    else:
        readout_db = DetDb(map_file=configs["readout_db"])

    # copy detectors from det_db into readout_db
    det_db = DetDb.from_file(configs["det_db"], force_new_db=False)
    for name in det_db.dets()["name"]:
        x = readout_db.get_id(name, commit=False)
    readout_db.conn.commit()
    
    # add tables for different mapping configurations
    for array in configs['arrays']:
        for mapping in array['mapping']:
            tbl_name = f"{array['name']}_v{mapping['version']}"
            create_table(readout_db, tbl_name)
    return readout_db


def main(args=None):
    if args is None:
        args = parse_args()


    configs = yaml.safe_load(open(args.config_file, "r"))

    readout_db = setup_readout_db(configs)
    SMURF = G3tSmurf(os.path.join(configs["data_prefix"], "timestreams"),
                     configs["g3tsmurf_db"],
                     meta_path=os.path.join(configs["data_prefix"], "smurf"))
    session = SMURF.Session()

    for array in configs['arrays']:
        ## Find TuneSets for Each Array
        tunesets = session.query(TuneSets).filter(TuneSets.stream_id==array["stream_id"])
        if args.min_ctime is not None:
            tunesets.filter(TuneSets.start >= dt.datetime.utcfromtimestamp(args.min_ctime))
        if args.max_ctime is not None:
            tunesets.filter(TuneSets.start <= dt.datetime.utcfromtimestamp(args.max_ctime))
        tunesets = tunesets.all()
        logger.info(f"Found {len(tunesets)} TuneSets for Array {array['name']}")

        
        for mapping in array["mapping"]:
            ## Setup Mapping for Each Mapping Type
            tbl_name = f"{array['name']}_v{mapping['version']}"
            prop = f"{tbl_name}.readout_id"

            map_maker = MapMaker(
                north_is_highband=array["north_is_highband"],
                array_name = array["name"],
                mapping_strategy = mapping["strategy"],
                dark_bias_lines=array["dark_bias_lines"],
                **mapping["params"]
            )

            complete_matches = np.unique(
                readout_db.props(props=[f"{tbl_name}.tuneset"])[f"{tbl_name}.tuneset"]
            )

            ## Run mapping on each TuneSet.
            for ts in tunesets:
                do_map = ts.name not in complete_matches

                if do_map:
                    array_map=None
                    logger.info(f"Making Map for {ts.path}")
                    try:
                        array_map = map_maker.make_map_smurf(tunefile=ts.path)
                    except:
                        logger.warning(f"Map Maker Failed on {ts.path}")
                        continue

                    readout_ids, bands, channels = zip(*[(ch.name, ch.band, ch.channel) for ch in ts.channels])
                    bands = np.array(bands)
                    channels = np.array(channels)

                    for tune in array_map.tune_data:
                        duid = get_detector_id(array["name"], tune)                   

                        idx = np.where( np.all([bands == tune.smurf_band,
                            channels == tune.smurf_channel], axis=0))[0]
                        if len(idx) != 1:
                            logger.debug(f"Detector {duid} not found in Tuneset {ts.name}")
                            continue

                        readout_db.add_props(tbl_name, duid, 
                                             time_range=(int(ts.start.timestamp()), DetDb.ALWAYS[1]),
                                             readout_id = readout_ids[idx[0]],
                                             tuneset = ts.name)
                        
if __name__ == "__main__":
    main()