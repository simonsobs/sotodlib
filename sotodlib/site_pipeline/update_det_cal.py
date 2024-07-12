import traceback
import os
import yaml
from dataclasses import dataclass, astuple, fields
import numpy as np
from tqdm.auto import tqdm
import logging
import sys
from typing import Optional, Union, Dict, List

from sotodlib import core
from sotodlib.io.metadata import write_dataset
from sotodlib.io.load_book import get_cal_obsids
import sotodlib.site_pipeline.util as sp_util
import multiprocessing as mp
import sodetlib.tes_param_correction as tpc


# stolen  from pysmurf, max bias volt / num_bits
DEFAULT_RTM_BIT_TO_VOLT = 10 / 2**19
DEFAULT_pA_per_phi0 = 9e6

logger = logging.getLogger('det_cal')
if not logger.hasHandlers():
    sp_util.init_logger('det_cal')


def get_data_root(ctx: core.Context):
    "Get root data directory based on context file"
    c = ctx.obsfiledb.conn.execute('select name from files limit 1')
    res = [r[0] for r in c][0]
    # split out <data_root>/obs/<timecode>/<obsid>/fname
    for _ in range(4): 
        res = os.path.dirname(res)
    return res


@dataclass
class DetCalCfg:
    """
    Class for configuring the behavior of the det-cal update script.

    Args
    -------
    root_dir: str
        Path to the root of the results directory.
    context_path: str
        Path to the context file to use.
    raise_exceptions: bool
        If Exceptions should be raised in the get_cal_resset function.
    apply_cal_correction: bool
        If True, apply the RP calibration correction, and use corrected results
        for Rtes, Si, Pj, and loopgain when successful.
    index_path: str
        Path to the index file to use for the det_cal database.
    h5_path: str
        Path to the HDF5 file to use for the det_cal database.
    failed_file_cache: str
        Path to the yaml file that will store failed obsids.
    show_pb: bool
        If True, show progress bar in the run_update function.
    num_processes: int
        Number of parallel processes that should be used to process observations.
    num_obs: int
        Max number of observations to process per run_update call.
    log_level: str
        Logging level for the logger.
    param_correction_config: dict
        Configuration for the TES param correction. If None, default values are used,
        with ``default_nprocs`` set to ``num_processes``.
    cache_failed_obsids: bool
        If True, will cache failed obs-ids and avoid running them in the future.
    data_root: str
        Path to the root of the data directory. If None, it will be determined
        from the context object.
    """
    root_dir: str
    context_path: str
    data_root: Optional[str] = None
    raise_exceptions: bool = False
    apply_cal_correction: bool = True
    index_path: str = 'det_cal.sqlite'
    h5_path: str = 'det_cal.h5'
    cache_failed_obsids: bool = True
    failed_cache_file: str = 'failed_obsids.yaml'
    show_pb: bool = True

    nprocs_obs_info: int = 20
    nprocs_result_set: int = 80

    num_obs: Optional[int] = None
    log_level: str = 'DEBUG'
    param_correction_config: tpc.AnalysisCfg = None

    def __post_init__(self):
        self.root_dir = os.path.expandvars(self.root_dir)
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root dir does not exist: {self.root_dir}")

        self.context_path = os.path.expandvars(self.context_path)

        def parse_path(path):
            p = os.path.expandvars(path)
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
            return p

        self.index_path = parse_path(self.index_path)
        self.h5_path = parse_path(self.h5_path)
        self.failed_cache_file = parse_path(self.failed_cache_file)

        kw = {'show_pb': False, 'default_nprocs': self.num_processes}
        if self.param_correction_config is None:
            self.param_correction_config = tpc.AnalysisCfg(**kw)
        elif isinstance(self.param_correction_config, dict):
            kw.update(self.param_correction_config)
            self.param_correction_config = tpc.AnalysisCfg(**kw)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))
    
    def setup(self):
        """Create directories and databases if they don't exist"""
        if not os.path.exists(self.failed_cache_file):
            # If file doesn't exist yet, just create an empty one
            with open(self.failed_cache_file, 'w') as f:
                yaml.dump({}, f)

        if not os.path.exists(self.index_path):
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('dataset')
            db = core.metadata.ManifestDb(scheme=scheme)
            db.to_file(self.index_path)
        
        if self.data_root is None:
            ctx = core.Context(self.context_path)
            self.data_root = get_data_root(ctx)


@dataclass
class CalInfo:
    """ 
    Class that contains detector calibration information that will go into
    the caldb.

    Attributes
    ----------
    readout_id: str
        Readout ID of the detector.
    r_tes: float
        Detector resistance [ohms], determined through bias steps while the
        detector is biased.
    r_frac: float
        Fractional resistance of TES, given by r_tes / r_n.
    p_bias: float
        Bias power on the TES [J] computed using bias steps at the bias point.
    s_i: float
        Current responsivity of the TES [1/V] computed using bias steps at the
        bias point.
    phase_to_pW: float
        Phase to power conversion factor [pW/rad] computed using s_i,
        pA_per_phi0, and detector polarity.
    v_bias: float
        Commanded bias voltage [V] on the bias line of the detector for the observation.
    tau_eff: float
        Effective thermal time constant [sec] of the detector, measured from bias steps.
    loopgain: float
        Loopgain of the detector.
    tes_param_correction_success: bool
        True if TES parameter corrections were successfully applied.
    bg: int
        Bias group of the detector. Taken from IV curve data, which contains
        bgmap data taken immediately prior to IV. This will be -1 if the
        detector is unassigned.
    polarity: int
        Polarity of the detector response for a positive change in bias current
        while the detector is superconducting. This is needed to correct for
        detectors that have reversed response.
    r_n: float
        Normal resistance of the TES [Ohms] calculated from IV curve data.
    p_sat: float
        "saturation power" of the TES [J] calculated from IV curve data.
        This is defined  as the electrical bias power at which the TES
        resistance is 90% of the normal resistance.
    """
    readout_id: str = ''
    r_tes: float = np.nan
    r_frac: float = np.nan
    p_bias: float = np.nan
    s_i: float = np.nan
    phase_to_pW: float = np.nan
    v_bias: float = np.nan
    tau_eff: float = np.nan
    loopgain: float = np.nan
    tes_param_correction_success: bool = False
    bg: int = -1
    polarity: int = 1
    r_n: float = np.nan  
    p_sat: float = np.nan

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


@dataclass
class ObsInfo:
    obs_id: str
    am: core.AxisManager

    iv_obsids: Dict[str, str]
    bs_obsids: Dict[str, str]

    iva_files: Dict[str, str]
    bsa_files: Dict[str, str]

def get_obs_info(cfg: DetCalCfg, obs_id: str):
    ctx = core.Context(cfg.context_path)
    am = ctx.get_obs(
            obs_id, samples=(0, 1), ignore_missing=True, no_signal=True,
            on_missing={'det_cal': 'skip'}
    )

    if 'smurf' not in am.det_info:
        raise ValueError(f"Missing smurf info for {obs_id}")

    logger.debug(f"Getting cal obsids ({obs_id})")

    iv_obsids = get_cal_obsids(ctx, obs_id, 'iv')

    # Load in IVs
    logger.debug(f"Loading Bias step and IV data ({obs_id})")
    rtm_bit_to_volt = None
    pA_per_phi0 = None

    # Automatically determine paths based on data root instead of obsfiledb
    # because obsfiledb queries are slow on nersc.

    iva_files = {}
    bsa_files = {}
    for dset, oid in iv_obsids.items():
        if oid is not None:
            timecode = oid.split('_')[1][:5]
            zsmurf_dir = os.path.join(
                cfg.data_root, 'oper', timecode, oid, f'Z_smurf'
            )
            for f in os.listdir(zsmurf_dir):
                if 'iv' in f:
                    iva_files[dset] = os.path.join(zsmurf_dir, f)
                    break
            else:
                raise ValueError(f"IV data not found for in cal obs {oid}")
        else:
            logger.debug("missing IV data for %s", dset)

    # Load in bias steps
    bias_step_obsids = get_cal_obsids(ctx, obs_id, 'bias_steps')
    for dset, oid in bias_step_obsids.items():
        if oid is not None:
            timecode = oid.split('_')[1][:5]
            zsmurf_dir = os.path.join(
                cfg.data_root, 'oper', timecode, oid, f'Z_smurf'
            )
            for f in os.listdir(zsmurf_dir):
                if 'bias_step' in f:
                    bs_file = os.path.join(zsmurf_dir, f)
                    bsa_files[dset] = bs_file
                    break
            else:
                raise ValueError(f"Bias step data not found for in cal obs {oid}")
        else:
            logger.debug("missing bias step data for %s", dset)
    
    if rtm_bit_to_volt is None:
        rtm_bit_to_volt = DEFAULT_RTM_BIT_TO_VOLT
    if pA_per_phi0 is None:
        pA_per_phi0 = DEFAULT_pA_per_phi0
    
    return ObsInfo(
        obs_id=obs_id, am=am, iv_obsids=iv_obsids,
        bs_obsids=bias_step_obsids,
        iva_files=iva_files, bsa_files=bsa_files,
    )


@dataclass
class CalRessetResult:
    """
    Results object for the get_cal_resset function.
    """
    obs_info: ObsInfo
    success: bool = False
    traceback: Optional[str] = None
    fail_msg: Optional[str] = None

    correction_results: Optional[List[tpc.CorrectionResults]] = None
    result_set: Optional[np.ndarray] = None


def get_cal_resset(cfg: DetCalCfg, obs_info: ObsInfo) -> CalRessetResult:
    """
    Returns calibration ResultSet for a given ObsId. This pulls IV and bias step
    data for each detset in the observation, and uses that to compute CalInfo
    for each detector in the observation.
    """

    obs_id = obs_info.obs_id
    res = CalRessetResult(obs_info)
    logger.debug("Computing Result set for %s", obs_info.obs_id)

    # Need to reset logger here because this may be created new for spawned process
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, cfg.log_level.upper()))

    try:
        am = obs_info.am

        ivas = {
            dset: np.load(iva_file, allow_pickle=True).item()
            for dset, iva_file in obs_info.iva_files.items()
        }
        bsas = {
            dset: np.load(bsa_file, allow_pickle=True).item()
            for dset, bsa_file in obs_info.bsa_files.items()
        }

        iva = list(ivas.values())[0]
        rtm_bit_to_volt = iva['meta']['rtm_bit_to_volt']
        pA_per_phi0 = iva['meta']['pA_per_phi0']
        cals = [CalInfo(rid) for rid in am.det_info.readout_id]

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

        # Add TES corrected params
        correction_results: Dict[List[tpc.CorrectionResults]] = {}
        if cfg.apply_cal_correction:
            logger.debug("Applying TES param corrections (%s)", obs_id)
            for dset in bsas:
                # logger.debug(f"Applying correction for {dset}")
                rs = []
                for b, c in zip(ivas[dset]['bands'], ivas[dset]['channels']):
                    chdata = tpc.RpFitChanData.from_data(
                        ivas[dset], bsas[dset], b, c
                    )
                    rs.append(tpc.run_correction(chdata, cfg.param_correction_config))
                correction_results[dset] = rs
        res.correction_results = correction_results

        def find_correction_results(band, chan, dset):
            for r in correction_results[dset]:
                if r.chdata.band == band and r.chdata.channel == chan:
                    return r
            return None

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

            correction = find_correction_results(band, chan, detset)
            if correction is None:
                logger.warn("Unable to find correction result for %s %s %s (%s)", band, chan, detset, obs_id)
                use_correction = False
                cal.tes_param_correction_success = False
            else:
                use_correction = correction.success
                cal.tes_param_correction_success = correction.success

            ridx = ridx[0]
            cal.tau_eff = bsa['tau_eff'][ridx]
            if bg != -1:
                cal.v_bias = bsa['Vbias'][bg]

            if use_correction:
                cal.r_tes = correction.corrected_R0
                cal.r_frac = correction.corrected_R0 / cal.r_n
                cal.s_i = correction.corrected_Si
                cal.p_bias = correction.corrected_Pj
                cal.loopgain = correction.loopgain
            else:
                cal.r_tes = bsa['R0'][ridx]
                cal.r_frac = bsa['Rfrac'][ridx]
                cal.p_bias = bsa['Pj'][ridx]
                cal.s_i = bsa['Si'][ridx]
            cal.phase_to_pW = pA_per_phi0 / (2*np.pi) / cal.s_i * cal.polarity

        res.result_set = np.array(
            [astuple(c) for c in cals], dtype=CalInfo.dtype()
        )
        res.success = True
    except Exception as e:
        res.traceback = traceback.format_exc()
        res.fail_msg = str(e)
        if cfg.raise_exceptions:
            raise
    return res


def get_obsids_to_run(cfg: DetCalCfg) -> List:
    ctx = core.Context(cfg.context_path)
    # Find all obs_ids that have not been processed
    with open(cfg.failed_cache_file, 'r') as f:
        failed_cache = yaml.safe_load(f)
    failed_obsids = set(failed_cache.keys())

    db = core.metadata.ManifestDb(cfg.index_path)
    obs_ids_all = set(ctx.obsdb.query('type=="obs"')['obs_id'])
    processed_obsids = set(db.get_entries(['dataset'])['dataset'])
    obs_ids = sorted(list(obs_ids_all - processed_obsids - failed_obsids), reverse=True)
    if cfg.num_obs is not None:
        obs_ids = obs_ids[:cfg.num_obs]
    return obs_ids


def handle_result(result: CalRessetResult, cfg: DetCalCfg):
    obs_id = str(result.obs_info.obs_id)
    if not result.success:
        logger.error(f"Failed on obs_id: {obs_id}")
        logger.error(result.traceback)

        msg = result.fail_msg
        if msg is None:
            msg = 'unknown error'

        if cfg.cache_failed_obsids:
            logger.info(f"Adding {obs_id} to failed_file_cache")
            with open(cfg.failed_cache_file, 'r') as f:
                d = yaml.safe_load(f)
                d[obs_id] = msg
            with open(cfg.failed_cache_file, 'w') as f:
                yaml.dump(d, f)
            return
    rset = core.metadata.ResultSet.from_friend(result.result_set)
    logger.debug(f"Adding obs_id {obs_id} to dataset")
    write_dataset(rset, cfg.h5_path, obs_id, overwrite=True)
    db = core.metadata.ManifestDb(cfg.index_path)
    db.add_entry(
        {'obs:obs_id': obs_id, 'dataset': obs_id}, 
        filename=cfg.h5_path, replace=True
    )

def get_obs_info_or_none(cfg, obs_id):
    try:
        get_obs_info(cfg, obs_id)
    except:
        tb = traceback.format_exc()
        logger.error(tb)
        return None


def run_update(cfg: Union[DetCalCfg, str]):
    """
    Update the det_cal database with metadata from unprocessed observations.
    See the ``DetCalCfg`` class for run configuration options.

    Args:
    ------
    cfg: DetCalCfg or str
        DetCalCfg object or path to config yaml file
    """
    if isinstance(cfg, str):
        cfg = DetCalCfg.from_yaml(cfg)

    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, cfg.log_level.upper()))

    cfg.setup()
    obs_ids = get_obsids_to_run(cfg)

    logger.info(f"Processing {len(obs_ids)} obsids...")

    pb = tqdm(total=len(obs_ids), disable=(not cfg.show_pb))

    def callback(result: CalRessetResult):
        pb.update()
        handle_result(result, cfg)

    def errback(e):
        raise e
    
    # We split into multiple pools because:
    # - we don't want to overload sqlite files with too much concurrent access
    # - we want to be able to continue getting the next obs_info data while ressets are being computed
    pool1 = mp.get_context('fork').Pool(cfg.nprocs_obs_info)
    pool2 = mp.get_context('fork').Pool(cfg.nprocs_result_set)

    def get_obs_info_callback(obs_info):
        if obs_info is None:
            return

        a = pool2.apply_async(
            get_cal_resset, args=(cfg, obs_info), callback=callback, error_callback=errback
        )

    try:
        for obs_id in obs_ids:
            a = pool1.apply_async(
                get_obs_info_or_none, args=(cfg, obs_id), callback=get_obs_info_callback, 
                error_callback=errback
            )
        a.wait()
    finally:
        pool1.close()
        pool1.join()
        pool2.close()
        pool2.join()
    pb.close()
    logger.info("Finished updates")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Must provide a config file")
    run_update(sys.argv[1])
