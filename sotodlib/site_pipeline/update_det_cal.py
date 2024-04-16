import traceback
import os
import argparse
import yaml
from dataclasses import dataclass, astuple, fields
import numpy as np
from tqdm.auto import tqdm
import logging

from typing import Optional, Union, Type, TypeVar, List

from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_smurf import G3tSmurf, TuneSets
from sotodlib.io.load_book import load_smurf_npy_data, get_cal_obsids
from sotodlib.io.imprinter import Imprinter
import sotodlib.site_pipeline.util as sp_util
from multiprocessing import Pool, Lock
import threading

# stolen  from pysmurf, max bias volt / num_bits
DEFAULT_RTM_BIT_TO_VOLT = 10 / 2**19
DEFAULT_pA_per_phi0 = 9e6

logger = logging.getLogger('det_cal')
if not logger.hasHandlers():
    sp_util.init_logger('det_cal')

@dataclass
class DetCalCfg:
    root_dir: str
    "Path to the root of the results directory"

    context_path: str
    "Path to the context file to use"

    detset_index: str
    "Path to index file containing smurf detsets"

    interm_data_dir: str = 'interm_data'
    "Directory to store per-observation results before adding to h5file"

    index_path: str = 'det_cal.sqlite'
    h5_path: str = 'det_cal.h5'
    failed_obsid_cache: str = 'failed_obsids.yaml'
    write_relpath: bool = True
    show_pb: bool = True

    def __post_init__(self):
        self.root_dir = os.path.expandvars(self.root_dir)
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root dir does not exist: {self.root_dir}")

        self.context_path = os.path.expandvars(self.context_path)
        self.detset_index = os.path.expandvars(self.detset_index)
        # self.context = core.Context(self.context_path)

        def parse_path(path):
            p = os.path.expandvars(path)
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
            return p

        self.index_path = parse_path(self.index_path)
        self.h5_path = parse_path(self.h5_path)
        self.failed_obsid_cache = parse_path(self.failed_obsid_cache)
        self.interm_data_dir = parse_path(self.interm_data_dir)
        self.setup()
    
    def setup(self):
        if not os.path.exists(self.interm_data_dir):
            os.makedirs(self.interm_data_dir)
        
        if not os.path.exists(self.index_path):
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('dataset')
            db = core.metadata.ManifestDb(scheme=scheme)
            db.to_file(self.index_path)


@dataclass
class CalInfo:
    """ Class that contains detector calibration information that will go into
    the caldb. """
    readout_id: str = ''
    "Readout ID of the detector"

    r_tes: float = np.nan
    """Detector resistance [ohms], determined through bias steps while the
    detector is biased"""

    r_frac: float = np.nan
    "Fractional resistance of TES, given by r_tes / r_n"

    p_bias: float = np.nan
    "Bias power on the TES [J] computed using bias steps at the bias point"

    s_i: float = np.nan
    """Current responsivity of the TES [1/V] computed using bias steps at the
    bias point"""

    phase_to_pW: float = np.nan
    """Phase to power conversion factor [pW/rad] computed using s_i,
    pA_per_phi0, and detector polarity"""

    v_bias: float = np.nan
    "Commanded bias voltage [V] on the bias line of the detector for the observation"

    tau_eff: float = np.nan
    "Effective thermal time constant [sec] of the detector, measured from bias steps"

    bg: int = -1
    """Bias group of the detector. Taken from IV curve data, which contains
    bgmap data taken immediately prior to IV. This will be -1 if the detector is
    unassigned"""

    polarity: int = 1
    """Polarity of the detector response for a positive change in bias current
    while the detector is superconducting. This is needed to correct for
    detectors that have reversed response. """

    r_n: float = np.nan  
    "Normal resistance of the TES [Ohms] calculated from IV curve data"

    p_sat: float = np.nan
    """"saturation power" of the TES [J] calculated from IV curve data.
    This is defined  as the electrical bias power at which the TES
    resistance is 90% of the normal resistance."""

    @classmethod
    def dtype(cls):
        """Returns ResultSet dtype for an item based on this class"""
        dtype = []
        for field in fields(cls):
            if field.name == 'readout_id':
                dt = ('dets:readout_id', '<U40')
            else:
                dt = (field.name, field.type)
            dtype.append(dt)
        return dtype


class MissingSmurfInfo(Exception):
    pass

def get_cal_resset(cfg: DetCalCfg, obs_id):
    """
    Returns calibration ResultSet for a given ObsId. This pulls IV and bias step
    data for each detset in the observation, and uses that to compute CalInfo
    for each detector in the observation
    """
    ctx = core.Context(cfg.context_path)
    am = ctx.get_obs(
            obs_id, samples=(0, 1), ignore_missing=True, no_signal=True,
            on_missing={'det_cal': 'skip'}
    )
    cals = [CalInfo(rid) for rid in am.det_info.readout_id]
    if 'smurf' not in am.det_info:
        raise MissingSmurfInfo(f"Missing smurf info for {obs_id}")

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
        cal.tau_eff = bsa['tau_eff'][ridx]
        if bg != -1:
            cal.v_bias = bsa['Vbias'][bg]
        cal.phase_to_pW = pA_per_phi0 / (2*np.pi) / cal.s_i * cal.polarity

    data= np.array(
        [astuple(c) for c in cals], dtype=CalInfo.dtype()
    )
    return data

@dataclass
class ProcessResult:
    obs_id: str
    success: bool = False
    resset: Optional[core.metadata.ResultSet] = None
    exception: Optional[Exception] = None

def work_func(cfg, obs_id) -> ProcessResult:
    result = ProcessResult(obs_id)
    try:
        result.resset = get_cal_resset(cfg, obs_id)
        result.success = True
    except Exception as e:
        result.success = False
        result.exception = e
    return result
    
def run_all(cfg: DetCalCfg, njobs=3, num_obs=None):
    ctx = core.Context(cfg.context_path)

    db = core.metadata.ManifestDb(cfg.index_path)
    obs_ids_all = set(ctx.obsdb.query('type=="obs"')['obs_id'])
    processed_obsids = set(db.get_entries(['dataset'])['dataset'])
    obs_ids = sorted(list(obs_ids_all- processed_obsids), reverse=True)
    if num_obs is not None:
        obs_ids = obs_ids[:num_obs]
    logger.info(f"Processing {len(obs_ids)} obsids...")

    pb = tqdm(total=len(obs_ids), desc="Processing obsids",)
    def callback(res: ProcessResult):
        if not res.success:
            logger.error(f"Error processing obs_id: {res.obs_id}")
            logger.error(traceback.format_exception(res.exception))
            pb.update()
            return

        rset = core.metadata.ResultSet.from_friend(res.resset)
        db = core.metadata.ManifestDb(cfg.index_path)
        print("Adding obs_id {} to dataset".format(res.obs_id))
        write_dataset(rset, cfg.h5_path, res.obs_id, overwrite=True)
        db.add_entry(
            {'obs:obs_id': obs_id, 'dataset': obs_id}, 
            filename=cfg.h5_path, replace=True
        )
        pb.update()

    with Pool(njobs) as pool:
        for obs_id in obs_ids:
            pool.apply_async(
                work_func, args=(cfg, obs_id), 
                callback=callback,
            )
        pool.close()
        pool.join()




def main(config: str):
    pass
