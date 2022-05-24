"""Build Matching Database between readout_id and detector_id
"""

import os
import sys
import yaml
import h5py
import logging
import numpy as np
import datetime as dt
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib import core
from sotodlib.io.load_smurf import G3tSmurf, TuneSets
from sotodlib.io.metadata import write_dataset, read_dataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file',help=
                        "Configuration File for running DetMap")
    parser.add_argument('--min-ctime', type=float, help=
                        "Minimum ctime to search for TuneSets")
    parser.add_argument('--max-ctime', type=float, help=
                        "Maximum ctime to search for TuneSets")
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse_args()

    configs = yaml.safe_load(open(args.config_file, "r"))

    SMURF = G3tSmurf(os.path.join(configs["data_prefix"], "timestreams"),
                     configs["g3tsmurf_db"],
                     meta_path=os.path.join(configs["data_prefix"], "smurf"))
    session = SMURF.Session()

    if os.path.exists(configs["read_db"]):
        logger.info(f'Mapping {configs["read_db"]} for the archive index.')
        db = core.metadata.ManifestDb(configs["read_db"])
        
    else:
        logger.info(f'Creating {configs["read_db"]} for the archive index.')
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('dets:detset')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(configs["read_db"], scheme=scheme)      
    
    array_names = [array["name"] for array in configs["arrays"]]
    det_info_group = ",".join(array_names)     

    for array in configs["arrays"]:
        ## Load Det Info
        rs = read_dataset(configs["det_info"], det_info_group)
        det_rs = core.metadata.merge_det_info(None, rs)
        det_info = core.metadata.loader.convert_det_info(det_rs, dets=det_rs["det_id"])
        det_info.restrict( 
            "dets", 
            det_info.dets.vals[det_info.wafer.array==array["name"]],
        )
        logger.info(
            f"Found {det_info.dets.count} detector_ids for Array {array['name']}"
        )

        ## Find TuneSets for Each Array
        tunesets = session.query(TuneSets).filter(
            TuneSets.stream_id==array["stream_id"]
        )
        if args.min_ctime is not None:
            tunesets.filter(
                TuneSets.start >= dt.datetime.utcfromtimestamp(args.min_ctime)
            )
        if args.max_ctime is not None:
            tunesets.filter(
                TuneSets.start <= dt.datetime.utcfromtimestamp(args.max_ctime)
             )
        tunesets = tunesets.all()
        logger.info(f"Found {len(tunesets)} TuneSets for Array {array['name']}")
        
        ## Run mapping on each TuneSet.
        for ts in tunesets:
            mapping = array["mapping"]      
            dest_dataset = f"{ts.stream_id}_{ts.name}_mapping_v{mapping['version']}"
            
            if dest_dataset in db.get_entries() and not args.override:
                logger.debug(f"Dataset {dest_dataset} already exists")
                continue
            
            ## Setup Mapping for Each Mapping Type
            map_maker = MapMaker(
                north_is_highband=array["north_is_highband"],
                array_name = array["name"],
                mapping_strategy = mapping["strategy"],
                dark_bias_lines=array["dark_bias_lines"],
                **mapping["params"]
            )

            array_map=None
            logger.info(f"Making Map for {ts.path}")
            try:
                array_map = map_maker.make_map_smurf(tunefile=ts.path)
            except:
                logger.warning(f"Map Maker Failed on {ts.path}")
                continue

            readout_ids, bands, channels = zip(*[
                    (ch.name, ch.band, ch.channel) for ch in ts.channels
                    ])
            bands = np.array(bands)
            channels = np.array(channels)

            rs_list = core.metadata.ResultSet(
                keys=["dets:det_id", 
                      "dets:readout_id"]
            )

            ## loop through detector IDs and find readout ID matches
            for tune in array_map:
                det_id = tune.detector_id
                if det_id is None:
                    ## resonators smurf found that array map doesn't think 
                    ## should exist (artifacts?)
                    continue
                i_did = np.where(det_info.dets.vals == det_id)[0]
                if len(i_did) != 1:
                    logger.warning(f"Map returned detector_id, {det_id}, "
                                "not in database, what happened?")
                    continue
                i_did = i_did[0]
                i_rid = np.where(np.all([
                        bands == tune.smurf_band,
                        channels == tune.smurf_channel], axis=0))[0]
                if len(i_rid) != 1:
                    logger.debug(f"Detector {det_id} not found in Tuneset {ts.name}")
                    ## If we want to track which detectors do not have readout_ids it
                    ## would go here, but has trouble with dets:readout_id loading in 
                    ## the current framework
                    continue
                i_rid = i_rid[0]
                #ch_info.readout_id[i_did] = readout_ids[i_rid]
                rs_list.append( {"dets:det_id": det_id,
                                 "dets:readout_id": readout_ids[i_rid]})

            for rid in readout_ids:
                if rid in rs_list["dets:readout_id"]:
                    continue
                rs_list.append( {"dets:det_id": "NO_MATCH",
                                 "dets:readout_id": rid})
        
            
            write_dataset(rs_list, configs["read_info"], dest_dataset, args.override)
            
            if not dest_dataset in db.get_entries():
                # Update the index.
                db_data = {'dets:detset': ts.name,
                           'dataset': dest_dataset}
                db.add_entry(db_data, configs["read_info"])



if __name__ == "__main__":
    main()

