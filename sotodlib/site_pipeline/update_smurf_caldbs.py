""" 
Script to import tuning and readout id channel mapping, and detector calibration
information into manifest dbs for book loading.  At present this just works in
the configuration where it has access to both level 2 and level 3 indexing. This
is technically possible with just level 3 data / indexing but requires some
still non-existant tools. 

Configuration file required::

    config = {
        'archive': {
            'detset': {
                'index': 'detset.sqlite',
                'h5file': 'detset.h5',
                'context': 'context.yaml',
            },
            'det_cal': {
                'index': 'det_cal.sqlite',
                'h5file': 'det_cal.h5,
                'context': 'context.yaml',
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
import logging

from typing import Optional, Union

from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_smurf import G3tSmurf, TuneSets
from sotodlib.io.load_book import load_smurf_npy_data, get_cal_obsids
from sotodlib.io.imprinter import Imprinter
import sotodlib.site_pipeline.util as sp_util

# stolen  from pysmurf, max bias volt / num_bits
DEFAULT_RTM_BIT_TO_VOLT = 10 / 2**19
DEFAULT_pA_per_phi0 = 9e6

default_logger = logging.getLogger('smurf_caldbs')
if not default_logger.hasHandlers():
    sp_util.init_logger('smurf_caldbs')


def main(config: Union[str, dict], 
        overwrite:Optional[bool]=False,
        logger=None):
    smurf_detset_info(config, overwrite, logger)
    run_update_det_caldb(config, logger=logger, overwrite=overwrite)

def smurf_detset_info(config: Union[str, dict], 
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
    ctx = core.Context(config['archive']['detset']['context'])


    if not os.path.exists(config['archive']['detset']['index']):
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


# Dtype for calibration set
cal_dtype = [
    ('dets:readout_id', '<U40'),
    ('r_tes', float),
    ('r_frac', float),
    ('p_bias', float),
    ('s_i', float),
    ('phase_to_pW', float),
    ('bg', int),
    ('polarity', int),
    ('r_n', float),
    ('p_sat', float),
]

@dataclass
class CalInfo:
    """
    Class that contains detector calibration information that will go into the
    caldb.

    Attributes
    ------------
    readout_id: str
        Readout id of detector
    r_tes: float
        Detector resistance [ohms], determined through bias steps while the
        detector is biased
    r_frac: float
        Fractional resistance of TES, given by r_tes / r_n
    p_bias: float
        Bias power on the TES [J] computed using bias steps at the bias point
    s_i: float
        Current responsivity of the TES [1/V] computed using bias steps at the
        bias point
    phase_to_pW: float
        Phase to power conversion factor [pW/rad] computed using s_i,
        pA_per_phi0, and detector polarity
    bg: int
        Bias group of the detector. Taken from IV curve data, which contains
        bgmap data taken immediately prior to IV. This will be -1 if the
        detector is unassigned
    polarity: int
        Polarity of the detector response for a positive change in bias current
        while the detector is superconducting. This is needed to correct for
        detectors that have reversed response. 
    r_n: float
        Normal resistance of the TES [Ohms] calculated from IV curve data
    p_sat: float
        "saturation power" of the TES [J] calculated from IV curve data.
        This is defined  as the electrical bias power at which the TES
        resistance is 90% of the normal resistance.
    """
    # Fields must be ordered like cal_dtype!
    readout_id: str = ''

    # From bias steps
    r_tes: float = np.nan   # Ohm
    r_frac: float = np.nan
    p_bias: float = np.nan  # J
    s_i: float = np.nan     # 1/V
    phase_to_pW: float = np.nan  # pW/rad

    # From IV
    bg: int = -1
    polarity: int = 1
    r_n: float = np.nan  
    p_sat: float = np.nan   # J


def get_cal_resset(ctx: core.Context, obs_id, logger=None):
    """
    Returns calibration ResultSet for a given ObsId. This pulls IV and bias step
    data for each detset in the observation, and uses that to compute CalInfo
    for each detector in the observation
    """
    if logger is None:
        logger=default_logger

    am = ctx.get_obs(obs_id, samples=(0, 1), ignore_missing=True, no_signal=True)
    cals = [CalInfo(rid) for rid in am.det_info.readout_id]

    iv_obsids = get_cal_obsids(ctx, obs_id, 'iv')

    rtm_bit_to_volt = None
    pA_per_phi0 = None
    ivas = {dset: None for dset in iv_obsids}
    for dset, oid in iv_obsids.items():
        if oid is not None:
            ivas[dset] = load_smurf_npy_data(ctx, oid, 'iv')
            if rtm_bit_to_volt is None:
                rtm_bit_to_volt = ivas[dset]['meta']['rtm_bit_to_volt']
                pA_per_phi0 = ivas[dset]['meta']['pA_per_phi0']
        else:
            logger.debug("missing IV data for %s", dset)

    bias_step_obsids = get_cal_obsids(ctx, obs_id, 'bias_steps')
    bsas = {dset: None for dset in bias_step_obsids}
    for dset, oid in bias_step_obsids.items():
        if oid is not None:
            bsas[dset] = load_smurf_npy_data(ctx, oid, 'bias_step_analysis')
            if rtm_bit_to_volt is None:
                rtm_bit_to_volt = bsas[dset]['meta']['rtm_bit_to_volt']
                pA_per_phi0 = ivas[dset]['meta']['pA_per_phi0']
        else:
            logger.debug("missing bias step data for %s", dset)

    if rtm_bit_to_volt is None:
        rtm_bit_to_volt = DEFAULT_RTM_BIT_TO_VOLT
    if pA_per_phi0 is None:
        pA_per_phi0 = DEFAULT_pA_per_phi0

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
    bias_line_is_valid = {k: True for k in obs_biases.keys()}

    # check to see if biases have changed between bias steps and obs
    for bsa in bsas.values():
        if bsa is None:
            continue

        for bg, vb_bsa in enumerate(bsa['Vbias']):
            bl_label = f"{bsa['meta']['stream_id']}_b{bg:0>2}"
            if np.isnan(vb_bsa):
                bias_line_is_valid[bl_label] = False
                continue

            if np.abs(vb_bsa - obs_biases[bl_label]) > 0.1:
                logger.debug("bias step and obs biases don't match for %s", bl_label)
                bias_line_is_valid[bl_label] = False

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
        if not bias_line_is_valid[bl_label]:
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
        cal.phase_to_pW = pA_per_phi0 / (2*np.pi) / cal.s_i * cal.polarity

    rset = core.metadata.ResultSet.from_friend(np.array(
        [astuple(c) for c in cals], dtype=cal_dtype
    ))
    return rset


def get_obs_with_detsets(ctx, detset_idx):
    """Gets all observations with type 'obs' that have detset data"""
    db = core.metadata.ManifestDb(detset_idx)
    detsets = db.get_entries(['dataset'])['dataset']
    obs_ids = set()
    for dset in detsets:
        obs_ids = obs_ids.union(ctx.obsfiledb.get_obs_with_detset(dset))
    return obs_ids


def update_det_caldb(ctx, idx_path, detset_idx, h5_path, logger=None, 
                     show_pb=False, format_exc=False, overwrite=False):
    if logger is None:
        logger = default_logger

    if not os.path.exists(idx_path):
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(scheme=scheme)
        db.to_file(idx_path)
    db = core.metadata.ManifestDb(idx_path)
    
    # detset_db = metadata.Manifest(detset_idx)
    obsids_with_detsets = get_obs_with_detsets(ctx, detset_idx)
    obsids_with_obs = set(ctx.obsdb.query("type=='obs'")['obs_id'])

    remaining_obsids = obsids_with_detsets.intersection(obsids_with_obs)
    if not overwrite: 
        existing_obsids = set(db.get_entries(['dataset'])['dataset'])
        remaining_obsids = remaining_obsids - existing_obsids

    # Sort remaining obs_ids by timestamp
    remaining_obsids = sorted(remaining_obsids,
                              key=(lambda s: s.split('_')[1]))

    logger.info("%d datasets to add", len(remaining_obsids))
    for obs_id in tqdm(remaining_obsids, disable=(not show_pb)):
        try:
            rset = get_cal_resset(ctx, obs_id)
        except Exception as e:
            logger.error("Failed on %s: %s", obs_id, e)
            if format_exc:
                logger.error(traceback.format_exc())
            continue

        logger.info("Writing metadata for %s", obs_id)
        write_dataset(rset, h5_path, obs_id, overwrite=overwrite)
        db.add_entry({
            'obs:obs_id': obs_id,
            'dataset': obs_id,
        }, filename=h5_path, replace=overwrite)


def run_update_det_caldb(config_path, logger=None, format_exc=False, show_pb=False, overwrite=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ctx = core.Context(config['archive']['det_cal']['context'])
    update_det_caldb(
        ctx, 
        config['archive']['det_cal']['index'],
        config['archive']['detset']['index'],
        config['archive']['det_cal']['h5file'],
        show_pb=show_pb,
        logger=logger,
        format_exc=format_exc,
        overwrite=overwrite
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

