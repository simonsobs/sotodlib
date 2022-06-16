"""This module updates the produced CalDbs from smurf metadata output.
Command line arguments can be used to select which CalDbs are updated,
and depending on the type of CalDb will either do so based on an obs_id or 
a time range. In addition, this will update the Metadata Indexes that 
describe how to load the proper dataset when loading observations or metadata
with the Context system.

"""

from argparse import ArgumentParser
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt

import sotodlib
from sotodlib.site_pipeline import util
from sotodlib.core import metadata

from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_smurf import G3tSmurf, Tunes, Observations

logger = util.init_logger(__name__, 'update-smurf-caldbs: ')

def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', help=
                        "Configuration File")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Pass multiple times to increase.")
    parser.add_argument('obs_id',help=
                        "Observation ID if necessary for updating a particular CalDb")
    parser.add_argument('-iv', '--iv', action='store_true',help=
                        "If present, update the IV CalDbs")
    parser.add_argument('-bgm', '--bias-group-map', action='store_true',help=
                        "If present, update the bias group map CalDbs")
    parser.add_argument('-bs', '--bias-steps', action='store_true',help=
                        "If present, update the bias step CalDbs")
    
    return parser

def _get_config(args):
    cfg = yaml.safe_load(open(args.config_file,'r'))
    for k in ['obs_id', 'verbose']:
        cfg[k] = getattr(args, k, None)
    cfg['_args'] = args
    return cfg

def _make_iv_caldb(cfg):
    # initialize SQLAlchemy Session from G3tSmurf Db
    SMURF = G3tSmurf(archive_path=cfg['g3tsmurf']['archive_path'],
                     meta_path=cfg['g3tsmurf']['meta_path'],
                     db_path=cfg['g3tsmurf']['db_path'])
    session = SMURF.Session()
   
    # load ManifestDb
    if os.path.exists(cfg['archive']['iv']['index']):
        db = metadata.ManifestDb(map_file=cfg['archive']['iv']['index'])
    else:
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')

        db = metadata.ManifestDb(scheme=scheme)
        db.to_file(cfg['archive']['iv']['index'])

    # initialize a ResultSet to hold the IV CalDb
    iv_caldb = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'R_n', 'p_sat', 'polarity'])

    o = session.query(Observations).filter(Observations.obs_id == cfg['obs_id']).one_or_none()

    for fname, stream_id, ctime, path in SMURF.search_metadata_files(max_ctime=o.timestamp,reverse=True):
        if 'iv' in fname:
            iva_fp = path
            break

    iva = np.load(iva_fp, allow_pickle=True).item()        
            
    # use G3tSmurf index to extract correct detector names
    t = iva['meta']['tunefile'].split('/')[-1]
    tun = session.query(Tunes).filter(Tunes.name == t).one_or_none()
    channels = [(c.name, c.band, c.channel) for c in tun.tuneset.channels]

    for cname in channels:
        name, band, channel = cname

        try:
            idx = np.where((iva['bands']==band)*(iva['channels']==channel))[0][0]
            R_n = iva['R_n'][idx]
            p_sat = iva['p_sat'][idx]
            polarity = iva['polarity'][idx]
        except IndexError:
            R_n = np.nan
            p_sat = np.nan
            polarity = np.nan

        iv_caldb.rows.append((name, band, channel, R_n, p_sat, polarity))
        
    # Get file and dataset from policy
    policy = util.ArchivePolicy.from_params(cfg['archive']['iv']['policy'])
    dest_file, dest_dataset = policy.get_dest(cfg['obs_id']+'_iv')
    db_data = {'obs:obs_id': cfg['obs_id'], 'dataset': dest_dataset}
    
    logger.info(f'Writing IV CalDb to {dest_file}.')
    write_dataset(iv_caldb, dest_file, dest_dataset, overwrite=True)
    
    # add an entry to the ManifestDb at the current obs_id
    logger.info(f'Updating Metadata Index {cfg["archive"]["iv"]["index"]}')
    db.add_entry(db_data,filename=dest_file,replace=True)

def _make_bgmap_caldb(cfg):
    # initialize SQLAlchemy Session from G3tSmurf Db
    SMURF = G3tSmurf(archive_path=cfg['g3tsmurf']['archive_path'],
                     meta_path=cfg['g3tsmurf']['meta_path'],
                     db_path=cfg['g3tsmurf']['db_path'])
    session = SMURF.Session()
   
    # load ManifestDb
    if os.path.exists(cfg['archive']['bgmap']['index']):
        db = metadata.ManifestDb(map_file=cfg['archive']['bgmap']['index'])
    else:
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')

        db = metadata.ManifestDb(scheme=scheme)
        db.to_file(cfg['archive']['bgmap']['index'])

    # initialize a ResultSet to hold the bgmap CalDb
    bgmap_caldb = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'bias_group'])

    o = session.query(Observations).filter(Observations.obs_id == cfg['obs_id']).one_or_none()

    for fname, stream_id, ctime, path in SMURF.search_metadata_files(max_ctime=o.timestamp,reverse=True):
        if 'bg_map' in fname:
            bgm_fp = path
            break

    bgm = np.load(bgm_fp, allow_pickle=True).item()        
            
    # use G3tSmurf index to extract correct detector names
    t = bgm['meta']['tunefile'].split('/')[-1]
    tun = session.query(Tunes).filter(Tunes.name == t).one_or_none()
    channels = [(c.name, c.band, c.channel) for c in tun.tuneset.channels]

    for cname in channels:
        name, band, channel = cname

        try:
            idx = np.where((bgm['bands']==band)*(bgm['channels']==channel))[0][0]
            bg_assign = bgm['bgmap'][idx]
        except IndexError:
            bg_assign = -1

        bgmap_caldb.rows.append((name, band, channel, bg_assign))
        
    # Get file and dataset from policy
    policy = util.ArchivePolicy.from_params(cfg['archive']['bgmap']['policy'])
    dest_file, dest_dataset = policy.get_dest(cfg['obs_id']+'_bgmap')
    db_data = {'obs:obs_id': cfg['obs_id'], 'dataset': dest_dataset}
    
    logger.info(f'Writing bias group map CalDb to {dest_file}.')
    write_dataset(bgmap_caldb, dest_file, dest_dataset, overwrite=True)
    
    # add an entry to the ManifestDb at the current obs_id
    logger.info(f'Updating Metadata Index {cfg["archive"]["bgmap"]["index"]}')
    db.add_entry(db_data,filename=dest_file,replace=True)

def _make_bias_step_caldb(cfg):
    # initialize SQLAlchemy Session from G3tSmurf Db
    SMURF = G3tSmurf(archive_path=cfg['g3tsmurf']['archive_path'],
                     meta_path=cfg['g3tsmurf']['meta_path'],
                     db_path=cfg['g3tsmurf']['db_path'])
    session = SMURF.Session()
   
    # load ManifestDb
    if os.path.exists(cfg['archive']['bias_steps']['index']):
        db = metadata.ManifestDb(map_file=cfg['archive']['bias_steps']['index'])
    else:
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')

        db = metadata.ManifestDb(scheme=scheme)
        db.to_file(cfg['archive']['bias_steps']['index'])

    # initialize a ResultSet to hold the IV CalDb
    bs_caldb = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'tau_eff', 'Si', 'R0'])

    o = session.query(Observations).filter(Observations.obs_id == cfg['obs_id']).one_or_none()

    for fname, stream_id, ctime, path in SMURF.search_metadata_files(max_ctime=o.timestamp,reverse=True):
        if 'bias_step' in fname:
            bsa_fp = path
            break

    bsa = np.load(bsa_fp, allow_pickle=True).item()        
            
    # use G3tSmurf index to extract correct detector names
    t = bsa['meta']['tunefile'].split('/')[-1]
    tun = session.query(Tunes).filter(Tunes.name == t).one_or_none()
    channels = [(c.name, c.band, c.channel) for c in tun.tuneset.channels]

    for cname in channels:
        name, band, channel = cname

        try:
            idx = np.where((bsa['bands']==band)*(bsa['channels']==channel))[0][0]
            R0 = bsa['R0'][idx]
            Si = bsa['Si'][idx]
            tau = bsa['tau_eff'][idx]
        except IndexError:
            R0 = np.nan
            Si = np.nan
            tau = np.nan

        bs_caldb.rows.append((name, band, channel, tau, Si, R0))
        
    # Get file and dataset from policy
    policy = util.ArchivePolicy.from_params(cfg['archive']['bias_steps']['policy'])
    dest_file, dest_dataset = policy.get_dest(cfg['obs_id']+'_bias_steps')
    db_data = {'obs:obs_id': cfg['obs_id'], 'dataset': dest_dataset}
    
    logger.info(f'Writing bias step CalDb to {dest_file}.')
    write_dataset(bs_caldb, dest_file, dest_dataset, overwrite=True)
    
    # add an entry to the ManifestDb at the current obs_id
    logger.info(f'Updating Metadata Index {cfg["archive"]["bias_steps"]["index"]}')
    db.add_entry(db_data,filename=dest_file,replace=True)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = _get_parser()
    config = _get_config(parser.parse_args(args))
    print(config['_args'].iv, config['_args'].bias_group_map)

    if config['verbose'] >= 1:
        logger.setLevel('INFO')
    if config['verbose'] >= 2:
        sotodlib.logger.setLevel('INFO')
    if config['verbose'] >= 3:
        sotodlib.logger.setLevel('DEBUG')
    
    if config['_args'].iv:
        logger.info(f'Creating IV CalDb for obs_id: {config["obs_id"]}')
        _make_iv_caldb(config)
    if config['_args'].bias_group_map:
        logger.info(f'Creating bias group map CalDb for obs_id: {config["obs_id"]}')
        _make_bgmap_caldb(config)
    if config['_args'].bias_steps:
        logger.info(f'Creating bias step CalDb for obs_id: {config["obs_id"]}')
        _make_bias_step_caldb(config)

    # Return something?
    return True
    