""" Script to import tuning and readout id channel mapping into a manifest
database for book loading. At present this just works in the configuration where
it has access to both level 2 and level 3 indexing. This is technically possible
with just level 3 data / indexing but requires some still non-existant tools. 

Configuration file required:

config = {
    'context': 'context.yaml',
    'archive': {
        'index': 'manifest.sqlite',
        'h5file': 'manifest.h5'
    },
    'g3tsmurf': g3tsmurf_hwp_config.yaml',
    'imprinter': imprinter.yaml,
}
"""

import os
import argparse
import yaml

from typing import Optional

from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_smurf import G3tSmurf, TuneSets
from sotodlib.io.imprinter import Imprinter
import sotodlib.site_pipeline.util as sp_util

default_logger = sp_util.init_logger("smurf_caldbs")

def main(config:str | dict, 
        overwrite:Optional[bool]=False,
        logger=None):
    smurf_detset_info(config, overwrite, logger)

def smurf_detset_info(config:str | dict, 
        overwrite:Optional[bool]=False,
        logger=None):
    """Write out the updates for the manifest database with information about
    the readout ids present inside each detset.
    """

    if logger is None:
        logger = default_logger

    if type(config) == str:
        config = yaml.safe_load(open(config, 'r'))

    SMURF = G3tSmurf.from_configs(config['g3tsmurf'])
    session = SMURF.Session()

    imprinter = Imprinter(config['imprinter'])
    ctx = core.Context(config['context'])


    if not os.path.exists(config['archive']['index']):
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('dets:detset')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb( 
            config['archive']['index'], 
            scheme=scheme
        )
    else:
        db = core.metadata.ManifestDb( 
            config['archive']['index'],
        )

    keys = [
        "dets:readout_id",
        "dets:smurf.band",
        "dets:smurf.channel",
        "dets:smurf.frequency",
        "dets:stream_id",
        "dets:wafer_slot",
        "dets:tel_tube",
    ]

    stream_maps = {}
    for tube in imprinter.tubes:
        for s, slot in enumerate(imprinter.tubes[tube]['slots']):
            stream_maps[slot] = (f'ws{s}', tube)

    c = ctx.obsfiledb.conn.execute('select distinct name from detsets')
    ctx_detsets = [r[0] for r in c]
    added_detsets = db.get_entries(['dataset'])['dataset']

    detsets = session.query(TuneSets).all()

    for ts in detsets:
        if ts.name not in ctx_detsets:
            logger.debug(f"{ts.name} not in obsfiledb, ignoring")
            continue
        if ts.name in added_detsets and not overwrite:
            continue
        
        det_rs = core.metadata.ResultSet(keys=keys)
        for channel in ts.channels:
            det_rs.append({
                'dets:readout_id': channel.name,
                'dets:smurf.band': channel.band,
                'dets:smurf.channel': channel.channel,
                'dets:smurf.frequency': channel.frequency,
                'dets:stream_id': ts.stream_id,
                'dets:wafer_slot': stream_maps[ts.stream_id][0],
                'dets:tel_tube':stream_maps[ts.stream_id][1],
            })
        write_dataset(
            det_rs, 
            config['archive']['h5file'], 
            ts.name, 
            overwrite,
        )
        # add new entries to database
        if ts.name not in added_detsets:
            db_data = {'dets:detset': ts.name,
                    'dataset': ts.name}
            db.add_entry(db_data, config['archive']['h5file'])

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help="configuration file"
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing entries'
    )
    return parser


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))

