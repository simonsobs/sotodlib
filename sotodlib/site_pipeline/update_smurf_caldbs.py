""" Script to import tuning and readout id channel mapping into a manifest
database for book loading. At present this just works in the configuration where
it has access to both level 2 and level 3 indexing. This is technically possible
with just level 3 data / indexing but requires some still non-existant tools. 

Configuration file required:

config = {
    'context': 'context.yaml',
    'archive': {
        'detset': {
            'index': 'manifest.sqlite',
            'h5file': 'manifest.h5'
        },
        'det_cal': {
            'index': 'det_cal.sqlite',
            'h5file': 'det_cal.h5
        },
    },
    'g3tsmurf': g3tsmurf_hwp_config.yaml',
    'imprinter': imprinter.yaml,
}
"""

import traceback
import os
import argparse
import yaml
from dataclasses import dataclass, astuple
import numpy as np
from tqdm.auto import tqdm

from typing import Optional, List

from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_smurf import G3tSmurf, TuneSets
from sotodlib.io.imprinter import Imprinter
import sotodlib.site_pipeline.util as sp_util

# stolen  from pysmurf, max bias volt / num_bits
DEFAULT_RTM_BIT_TO_VOLT = 10 / 2**19

default_logger = sp_util.init_logger("smurf_caldbs")

def main(config:str | dict, 
        overwrite:Optional[bool]=False,
        logger=None):
    smurf_detset_info(config, overwrite, logger)
    run_update_det_caldb(config, log=logger)

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
            config['archive']['detset']['index'], 
            scheme=scheme
        )
    else:
        db = core.metadata.ManifestDb( 
            config['archive']['detset']['index'],
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
            config['archive']['detset']['h5file'], 
            ts.name, 
            overwrite,
        )
        # add new entries to database
        if ts.name not in added_detsets:
            db_data = {'dets:detset': ts.name,
                    'dataset': ts.name}
            db.add_entry(db_data, config['archive']['detset']['h5file'])


def get_cal_obsids(ctx, obs_id, cal_type):
    """
    Returns set of obs-ids corresponding to the most recent calibration
    operations for a given obsid:

    Returns
    ----------
        obs_ids: dict
            Dict of obs_ids for each detset in specified operation
    """
    obs = ctx.obsdb.query(f"obs_id == '{obs_id}'")[0]
    detsets = ctx.obsfiledb.get_detsets(obs_id)
    min_ct = obs['start_time'] - 3600*24*7
    cal_all = ctx.obsdb.query(
        f"""
        start_time <= {obs['start_time']} and subtype=='{cal_type}'
        and start_time > {min_ct}
        """, sort=['start_time']
    )[::-1]

    obs_ids = {
        ds: None for ds in detsets
    }

    for o in cal_all:
        dsets = ctx.obsfiledb.get_files(o['obs_id']).keys()
        for ds in dsets:
            if ds in obs_ids:
                if obs_ids[ds] is None:
                    obs_ids[ds] = o['obs_id']

    return obs_ids

# Dtype for calibration set
cal_dtype = [
    ('dets:readout_id', '<U40'),
    ('dets:cal.r_tes', float),
    ('dets:cal.r_frac', float),
    ('dets:cal.p_bias', float),
    ('dets:cal.s_i', float),
    ('dets:cal.bg', int),
    ('dets:cal.bg_polarity', int),
    ('dets:cal.r_n', float),
    ('dets:cal.p_sat', float),
]

@dataclass
class CalInfo:
    # Fields must be ordered like cal_dtype!
    readout_id: str = ''

    # From bias steps
    r_tes: float = np.nan   # Ohm
    r_frac: float = np.nan
    p_bias: float = np.nan  # J
    s_i: float = np.nan     # 1/V

    # From IV
    bg: int = -1
    polarity: int = 1
    r_n: float = np.nan  
    p_sat: float = np.nan   # J


def _load_smurf_npy(obs_id, substr):
    """
    Loads npy file from Z_smurf archive of book.

    Args
    _____
    obs_id: str
        obs-id of book to load file from
    substr: str
        substring to use to find numpy file in Z_smurf
    """
    files = ctx.obsfiledb.get_files(obs_id)
    book_dir = os.path.dirname(list(files.values())[0][0][0])
    smurf_dir = os.path.join(book_dir, 'Z_smurf')
    for f in os.listdir(smurf_dir):
        if substr in f:
            fpath = os.path.join(smurf_dir, f)
            break
    else:
        raise FileNotFoundError("Could not find npy file")
    res = np.load(fpath, allow_pickle=True).item()
    return res


def get_cal_resset(ctx: core.Context, obs_id):
    """Returns calibration ResultSet for a given ObsId"""
    am = ctx.get_obs(obs_id, samples=(0, 1), ignore_missing=True)

    cals = [CalInfo(rid) for rid in am.det_info.readout_id]

    iv_obsids = get_cal_obsids(ctx, obs_id, 'iv')

    rtm_bit_to_volt = None
    ivas = {dset: None for dset in iv_obsids}
    for dset, oid in iv_obsids.items():
        if oid is not None:
            ivas[dset] = _load_smurf_npy(oid, 'iv')
            if rtm_bit_to_volt is None:
                rtm_bit_to_volt = ivas[dset]['meta']['rtm_bit_to_volt']

    bias_step_obsids = get_cal_obsids(ctx, obs_id, 'bias_steps')
    bsas = {dset: None for dset in bias_step_obsids}
    for dset, oid in bias_step_obsids.items():
        if oid is not None:
            bsas[dset] = _load_smurf_npy(oid, 'bias_step_analysis')
            if rtm_bit_to_volt is None:
                rtm_bit_to_volt = bsas[dset]['meta']['rtm_bit_to_volt']

    rtm_bit_to_volt = DEFAULT_RTM_BIT_TO_VOLT

    # Add IV info
    for i, cal in enumerate(cals):
        band = am.det_info.smurf.band[i]
        chan = am.det_info.smurf.channel[i]
        detset = am.det_info.detset[i]
        iva = ivas[detset]

        if iva is None: # No IV analysis for this detset
            continue

        ridx = np.where(
            (iva['bands'] == band) & (iva['channels'] == chan)
        )[0]
        if not ridx: # Channel doesn't exist in IV analysis
            continue

        ridx = ridx[0]
        cal.bg = iva['bgmap'][ridx]
        cal.polarity = iva['polarity'][ridx]
        cal.r_n = iva['R_n'][ridx]
        cal.p_sat = iva['p_sat'][ridx]
    
    obs_biases = dict(zip(am.bias_lines.vals, am.biases[:, 0] * 2*rtm_bit_to_volt))
    for i, cal in enumerate(cals):
        band = am.det_info.smurf.band[i]
        chan = am.det_info.smurf.channel[i]
        detset = am.det_info.detset[i]
        stream_id = am.det_info.stream_id[i]
        bg = cal.bg
        bsa = bsas[detset]

        if bsa is None or bg == -1:
            continue

        bl_label = f'{stream_id}_b{bg:0>2}'
        # If observation bias differs from bias-steps by more than 0.1 V,
        # don't include bias step calibration info
        if np.abs(obs_biases[bl_label] - bsa['Vbias'][bg]) > 0.1:
            continue

        ridx = np.where(
            (bsa['bands'] == band) & (bsa['channels'] == chan)
        )[0]
        if not ridx: # Channel doesn't exist in bias step analysis
            continue

        ridx = ridx[0]
        cal.r_tes = bsa['R0'][ridx]
        cal.r_frac = bsa['Rfrac'][ridx]
        cal.p_bias = bsa['Pj'][ridx]
        cal.s_i = bsa['Si'][ridx]

    rset = core.metadata.ResultSet.from_friend(np.array(
        [astuple(c) for c in cals], dtype=cal_dtype
    ))
    return rset


def get_obs_with_detsets(ctx, detset_idx):
    """Gets all observations with detset data"""
    db = core.metadata.ManifestDb(detset_idx)
    detsets = db.get_entries(['dataset'])['dataset']

    obs_ids = set()
    for dset in detsets:
        cur = ctx.obsfiledb.conn.execute(
            f"select distinct obs_id from files where detset='{dset}'"
        )
        obs_ids = obs_ids.union({r[0] for r in cur})
    return obs_ids


def update_det_caldb(ctx, idx_path, detset_idx, h5_path, log=None, 
                     show_pb=False, format_exc=False):
    if log is None:
        log = default_logger

    if not os.path.exists(idx_path):
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(scheme=scheme)
        db.to_file(idx_path)
    db = core.metadata.ManifestDb(idx_path)
    
    # detset_db = metadata.Manifest(detset_idx)
    existing_obsids = db.get_entries(['dataset'])['dataset']
    all_obsids = get_obs_with_detsets(ctx, detset_idx)
    remaining_obsids = \
        sorted(list(set(all_obsids) - set(existing_obsids)),
            key=(lambda s: s.split('_')[1]))
    
    log.info(f"{len(remaining_obsids)} bias step datasets to add.....")
    for obs_id in tqdm(remaining_obsids, disable=(not show_pb)):
        try:
            rset = get_cal_resset(ctx, obs_id)
        except Exception as e:
            log.error(f"Failed on {obs_id}: {e}")
            if format_exc:
                log.error(traceback.format_exc())
            continue
        
        log.info(f"Writing metadata for {obs_id}")
        write_dataset(rset, h5_path, obs_id, overwrite=True)
        db.add_entry({
            'obs:obs_id': obs_id,
            'dataset': obs_id,
        }, filename=h5_path,)


def run_update_det_caldb(config, log=None):
    ctx = core.Context(config['context'])
    update_det_caldb(
        ctx, 
        config['archive']['det_cal']['index'],
        config['archive']['detset']['index'],
        config['archive']['det_cal']['h5file'],
        show_pb=False,
        log=log
    )


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

