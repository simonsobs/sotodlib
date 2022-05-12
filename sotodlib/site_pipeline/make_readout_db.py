import os
import sys
import yaml
import h5py
import logging
import numpy as np
import datetime as dt
from argparse import ArgumentParser

from detmap.makemap import MapMaker
from sotodlib.core import AxisManager
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

"""
h5 file group plan

stream_id/
    tunesets/
        v0_map
        v1_map
        ...
"""

def main(args=None):
    if args is None:
        args = parse_args()

    configs = yaml.safe_load(open(args.config_file, "r"))

    SMURF = G3tSmurf(os.path.join(configs["data_prefix"], "timestreams"),
                     configs["g3tsmurf_db"],
                     meta_path=os.path.join(configs["data_prefix"], "smurf"))
    session = SMURF.Session()
    h5_file = h5py.File(configs["channel_info"], "a")

    for array in configs["arrays"]:
        ## Load Det Info
        det_info = AxisManager.load(configs["det_info"])
        det_info.restrict( 
            "dets", 
            det_info.dets.vals[det_info.array==array["name"]],
        )
        logger.info(
            "Found {det_info.dets.count} detector_ids for Array {array['name'}"
        )

        ## Make sure stream_id exists in Channel Info
        if array["stream_id"] not in h5_file:
            h5_file.create_group(array["stream_id"])
        array_group = h5_file[ array["stream_id"] ]

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
            if ts.name not in array_group:
                array_group.create_group(ts.name)
            ts_group = array_group[ ts.name ]
            
            for mapping in array["mapping"]:
                map_key = f"{mapping['version']}_{mapping['strategy']}"
                if map_key in ts_group:
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
                
                ch_info = AxisManager( det_info.dets )
                ch_info.wrap_new("readout_id", ("dets",), dtype=object)

                for tune in array_map:
                    det_id = tune.detector_id
                    if det_id is None:
                        continue
                    i_did = np.where(ch_info.dets.vals == det_id)[0]
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
                        continue
                    i_rid = i_rid[0]
                    ch_info.readout_id[i_did] = readout_ids[i_rid]

                not_found = ch_info.readout_id == 0
                logger.info(f"{np.sum(not_found)} detectors were not found in "
                            f"tuneset {ts.name} for mapping {map_key}")

                ch_info.restrict( "dets", ch_info.dets.vals[~not_found])
                ch_info.readout_id = np.array(ch_info.readout_id, dtype='str')
                
                ts_group.create_group(map_key)
                ch_info.save( ts_group[map_key])
    h5_file.close()

if __name__ == "__main__":
    main()

