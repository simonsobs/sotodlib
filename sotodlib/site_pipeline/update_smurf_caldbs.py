import numpy as np
import yaml
import os
from sotodlib.core import metadata
from sotodlib.io import load_smurf as ls
from sotodlib.io.load_smurf import Tunes
from sotodlib.io.metadata import read_dataset, write_dataset
import argparse
import datetime as dt
from sotodlib.site_pipeline import util
import logging

logger = util.init_logger(__name__)

def update_times(db):

    update_entries = db.inspect(params={'obs:timestamp':3e9-1})
    
    if len(update_entries) == 1:
        logger.debug('Only one entry with 3e9 endtime')
        return
    update_entries = update_entries[0]

    
    start = update_entries['obs:timestamp'][0]
    
    c = db.conn.cursor()
    c.execute('select `obs:timestamp__lo` from map where map.id=?',(update_entries['_id']+1,))
    rows = c.fetchall()
    assert(len(rows)==1)
    
    stop = rows[0]['obs:timestamp__lo']
    
    update_params = {'_id':update_entries['_id'],'obs:timestamp':(start,stop)}
    
    db.update_entry(update_params)

def _update_bgmap_caldb(bgmaps, config):
    starts = []
    bgnames = []

    if os.path.exists(config['bgmap']['index']):
        db = metadata.ManifestDb(config['bgmap']['index'])
    else:
        scheme = metadata.ManifestScheme()
        scheme.add_range_match('obs:timestamp')
        scheme.add_data_field('dataset')

        db = metadata.ManifestDb(scheme=scheme)
        db.to_file(config['bgmap']['index'])

    for i, bgmap_fp in enumerate(bgmaps):
        bgmap_filename = os.path.split(bgmap_fp)[1]
        bgmap_name = bgmap_filename.split('.')[0]
        bgmap = np.load(bgmap_fp,allow_pickle=True).item()
        if bgmap['meta']['tunefile'] is None:
            logger.debug(f'{bgmap_fp} has no tunefile, skipping this.') 
            continue
        bgmap_caldb = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'bias_group'])

        t = bgmap['meta']['tunefile'].split('/')[-1]
        tun = session.query(Tunes).filter(Tunes.name == t).one_or_none()
        channels = [(c.name, c.band, c.channel) for c in tun.tuneset.channels]
        start = bgmap['meta']['action_timestamp']
        starts.append(start)
        bgnames.append(bgmap_name)

        for cname in channels:
            name, band, channel = cname

            try:
                idx = np.where((bgmap['bands']==band)*(bgmap['channels']==channel))[0][0]
                bg_assign = bgmap['bgmap'][idx]
            except IndexError:
                bg_assign = -1

            bgmap_caldb.rows.append((name, band, channel, bg_assign))
        
        write_dataset(bgmap_caldb, config['bgmap']['filename'], bgmap_name, overwrite=True)

    
    for i, bgn in enumerate(bgnames):
        
        if i == (len(bgnames)-1):
            start = starts[i]
            stop = 3e9
        else:
            start = starts[i]
            stop = starts[i+1]
    
        db_data = {'obs:timestamp': (start,stop), 'dataset': f'{bgn}'}
        db.add_entry(db_data,filename=config['bgmap']['filename'],replace=True)
    
    update_times(db)
    
    db.to_file(config['bgmap']['index'])

def _update_iv_caldb(ivs, config):

    starts = []
    ivnames = []

    if os.path.exists(config['iv']['index']):
        db = metadata.ManifestDb(config['iv']['index'])
    else:
        scheme = metadata.ManifestScheme()
        scheme.add_range_match('obs:timestamp')
        scheme.add_data_field('dataset')

        db = metadata.ManifestDb(scheme=scheme)
        db.to_file(config['iv']['index'])

    for i, iv_fp in enumerate(ivs):
        iv_filename = os.path.split(iv_fp)[1]
        iv_name = iv_filename.split('.')[0]
        iv = np.load(iv_fp,allow_pickle=True).item()
        if iv['meta']['tunefile'] is None:
            logger.debug(f'{iv_fp} has no tunefile, skipping this.') 
            continue
        iv_caldb = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'resp', 'R_n', 'P_sat'])

        t = iv['meta']['tunefile'].split('/')[-1]
        tun = session.query(Tunes).filter(Tunes.name == t).one_or_none()
        channels = [(c.name, c.band, c.channel) for c in tun.tuneset.channels]
        start = iv['meta']['action_timestamp']
        starts.append(start)
        ivnames.append(iv_name)

        for cname in channels:
            name, band, channel = cname

            try:
                idx = np.where((iv['bands']==band)*(iv['channels']==channel))[0][0]
                rfrac50idx = np.argmin(np.abs(iv['R'][idx]/iv['R_n'][idx] - 0.5 ))
                resp = iv['si'][idx][rfrac50idx]
                R_n = iv['R_n'][idx]
                p_sat = iv['p_sat'][idx]
            except IndexError:
                resp = np.nan
                R_n = np.nan
                p_sat = np.nan
            iv_caldb.rows.append((name, band, channel, resp, R_n, p_sat))
        
        write_dataset(iv_caldb, config['iv']['filename'], iv_name, overwrite=True)

    for i, ivn in enumerate(ivnames):
        
        if i == (len(ivnames)-1):
            start = starts[i]
            stop = 3e9
        else:
            start = starts[i]
            stop = starts[i+1]
    
        db_data = {'obs:timestamp': (start,stop), 'dataset': f'{ivn}'}
        db.add_entry(db_data,filename=config['iv']['filename'],replace=True)
        
    update_times(db)
    
    db.to_file(config['iv']['index'])

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="g3tsmurf db configuration file")
    parser.add_argument('--min-days', help="Days to subtract from now to set as minimum ctime",
                        default=2, type=float)
    parser.add_argument('--max-days', help="Days to subtract from now to set as maximum ctime",
                        default=0, type=float)
    parser.add_argument('--bgmap', help="Updates the bgmap caldbs",
                        action="store_true")
    parser.add_argument('--iv', help="Updates the iv caldbs",
                        action="store_true")
    parser.add_argument('--bias-steps', help="Updates the bias step caldbs",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    return parser

if __name__ == '__main__':

    parser = get_parser(parser=None)
    args = parser.parse_args()

    verbosity = args.verbosity

    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    # config = yaml.safe_load(open('/mnt/so1/users/jseibert/pipeline-dev/update-smurf-caldbs/config.yaml', "r"))
    config = yaml.safe_load(open(args.config, "r"))

    SMURF = ls.G3tSmurf.from_configs(config['g3tsmurf']['config_path'])
    session = SMURF.Session()

    book_dir = config['books']['book_dir']

    bs_list = []
    iv_list = []
    bgmap_list = []

    min_ctime = (dt.datetime.now() - dt.timedelta(days=args.min_days)).timestamp()
    max_ctime = (dt.datetime.now() - dt.timedelta(days=args.max_days)).timestamp()

    for root, dirs, files in os.walk(os.path.join(book_dir,'oper')):
        for f in files:
            if 'M_index' in f:
                book_index = yaml.safe_load(open(os.path.join(root,f),"r"))
                book_id = book_index['book_id']
                if (book_index['start_time'] < min_ctime) or (book_index['start_time'] > max_ctime):
                    continue
                try:
                    meta_types = list(book_index['meta_files'].keys())
                except KeyError:
                    logger.debug(f'No metadata files in {book_id}')
                    continue
                if 'bias_steps' in meta_types:
                    bs_list.append(os.path.join(root,book_index['meta_files']['bias_steps']))
                if 'bgmap' in meta_types:
                    bgmap_list.append(os.path.join(root,book_index['meta_files']['bgmap']))
                if 'iv' in meta_types:
                    iv_list.append(os.path.join(root,book_index['meta_files']['iv']))

    bgmap_list = sorted(bgmap_list)
    bs_list = sorted(bs_list)
    iv_list = sorted(iv_list)

    if args.bgmap:
        _update_bgmap_caldb(bgmap_list, config)
    if args.iv:
        _update_iv_caldb(iv_list, config)
    if args.bias_steps:
        logger.debug('bias_steps')



