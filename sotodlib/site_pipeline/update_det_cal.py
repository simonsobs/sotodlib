"""
This module is used to compute detector calibration parameters from sodetlib
data products.

The naive computation is described in the `sodetlib documentation. <https://sodetlib.readthedocs.io/en/latest/operations/bias_steps.html#in-transition>`_

Details about the RP and loopgain correction `can be found on our confluence. <https://simonsobs.atlassian.net/wiki/spaces/~5570586d07625a6be74c8780e4b96f6156f5e6/blog/2024/02/02/286228683/Nonlinear+TES+model+using+RP+curve>`_
"""

import traceback
import os
import yaml
from dataclasses import dataclass, astuple, fields
import numpy as np
from tqdm.auto import tqdm
import logging
from typing import Optional, Union, Dict, List, Any, Tuple, Literal
from queue import Queue
import argparse

from sotodlib import core
from sotodlib.io.metadata import write_dataset, ResultSet
from sotodlib.io.load_book import get_cal_obsids
import sotodlib.site_pipeline.util as sp_util
import multiprocessing as mp
import sodetlib.tes_param_correction as tpc
from sodetlib.operations.iv import IVAnalysis
from sodetlib.operations.bias_steps import BiasStepAnalysis


# stolen  from pysmurf, max bias volt / num_bits
DEFAULT_RTM_BIT_TO_VOLT = 10 / 2**19
DEFAULT_pA_per_phi0 = 9e6
TES_BIAS_COUNT = 12  # per detset / primary file group

# For converting bias group to bandpass.
BGS = {'lb': [0, 1, 4, 5, 8, 9], 'hb': [2, 3, 6, 7, 10, 11]}
BAND_STR = {'mf': {'lb': 'f090', 'hb': 'f150'},
            'uhf': {'lb': 'f220', 'hb': 'f280'},
            'lf': {'lb': 'f030', 'hb': 'f040'}}

logger = logging.getLogger("det_cal")
if not logger.hasHandlers():
    sp_util.init_logger("det_cal")


def get_data_root(ctx: core.Context) -> str:
    "Get root data directory based on context file"
    c = ctx.obsfiledb.conn.execute("select name from files limit 1")
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
    -------------
    root_dir: str
        Path to the root of the results directory.
    context_path: str
        Path to the context file to use.
    data_root: Optional[str]
        Root path of L3 data. If this is not specified, will automatically
        determine it based on the context.
    raise_exceptions: bool
        If Exceptions should be raised in the get_cal_resset function.
        Defaults to False.
    apply_cal_correction: bool
        If True, apply the RP calibration correction, and use corrected results
        for Rtes, Si, Pj, and loopgain when successful. Defaults to True.
    index_path: str
        Path to the index file to use for the det_cal database. Defaults to
        "det_cal.sqlite".
    h5_path: str
        Path to the HDF5 file to use for the det_cal database. Default to
        "det_cal.h5".
    cache_failed_obsids: bool
        If True, will cache failed obs-ids to avoid re-running them. Defaults to
        True.
    failed_file_cache: str
        Path to the yaml file that will store failed obsids. Defaults to
        "failed_obsids.yaml".
    show_pb: bool
        If True, show progress bar in the run_update function. Defaults to True.
    param_correction_config: dict
        Configuration for the TES param correction. If None, default values are used.
    run_method: str
        Must be "site" or "nersc". If "site", this function will not parallelize SQLite access, and will only parallelize the TES parameter correction. If "nersc", this will parallelize both SQLite access and the TES param correction, using ``nprocs_obs_info`` and ``nprocs_result_set`` processes respectively.
    nprocs_obs_info: int
        Number of processes to use to acquire observation info from the file system.
        Defaults to 1.
    nprocs_result_set: int
        Number of parallel processes that should to compute the TES parameters,
        and to run the TES parameter correction.
    num_obs: Optional[int]
        Max number of observations to process per run_update call. If not set,
        will run on all available observations.
    log_level: str
        Logging level for the logger.
    multiprocess_start_method: str
        Method to use to start child processes. Can be "spawn" or "fork".
    """

    def __init__(
        self,
        root_dir: str,
        context_path: str,
        *,
        data_root: Optional[str] = None,
        raise_exceptions: bool = False,
        apply_cal_correction: bool = True,
        index_path: str = "det_cal.sqlite",
        h5_path: str = "det_cal.h5",
        cache_failed_obsids: bool = True,
        failed_cache_file: str = "failed_obsids.yaml",
        show_pb: bool = True,
        param_correction_config: Union[Dict[str, Any], None, tpc.AnalysisCfg] = None,
        run_method: str = "site",
        nprocs_obs_info: int = 1,
        nprocs_result_set: int = 10,
        num_obs: Optional[int] = None,
        log_level: str = "DEBUG",
        multiprocess_start_method: Literal["spawn", "fork"] = "spawn"
    ) -> None:
        self.root_dir = root_dir
        self.context_path = os.path.expandvars(context_path)
        ctx = core.Context(self.context_path)
        if data_root is None:
            self.data_root = get_data_root(ctx)
        self.raise_exceptions = raise_exceptions
        self.apply_cal_correction = apply_cal_correction
        self.cache_failed_obsids = cache_failed_obsids
        self.show_pb = show_pb
        self.run_method = run_method

        if self.run_method not in ["site", "nersc"]:
            raise ValueError("run_method must be in: ['site', 'nersc']")

        self.nprocs_obs_info = nprocs_obs_info
        self.nprocs_result_set = nprocs_result_set
        self.num_obs = num_obs
        self.log_level = log_level
        self.multiprocess_start_method = multiprocess_start_method

        self.root_dir = os.path.expandvars(self.root_dir)
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root dir does not exist: {self.root_dir}")

        def parse_path(path: str) -> str:
            "Expand vars and make path absolute"
            p = os.path.expandvars(path)
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
            return p

        self.index_path = parse_path(index_path)
        self.h5_path = parse_path(h5_path)
        self.failed_cache_file = parse_path(failed_cache_file)

        kw = {"show_pb": False, "default_nprocs": self.nprocs_result_set}
        if param_correction_config is None:
            self.param_correction_config = tpc.AnalysisCfg(**kw)  # type: ignore
        elif isinstance(param_correction_config, dict):
            kw.update(param_correction_config)
            self.param_correction_config = tpc.AnalysisCfg(**kw)  # type: ignore
        else:
            self.param_correction_config = param_correction_config

        self.setup_files()

    @classmethod
    def from_yaml(cls, path) -> "DetCalCfg":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
            return cls(**d)

    def setup_files(self) -> None:
        """Create directories and databases if they don't exist"""
        if not os.path.exists(self.failed_cache_file):
            # If file doesn't exist yet, just create an empty one
            with open(self.failed_cache_file, "w") as f:
                yaml.dump({}, f)

        if not os.path.exists(self.index_path):
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match("obs:obs_id")
            scheme.add_data_field("dataset")
            db = core.metadata.ManifestDb(scheme=scheme)
            db.to_file(self.index_path)


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
        Bias power on the TES [W] computed using bias steps at the bias point.
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
        "saturation power" of the TES [W] calculated from IV curve data.
        This is defined  as the electrical bias power at which the TES
        resistance is 90% of the normal resistance.
    naive_r_tes: float
        Detector resistance [ohms]. This is based on the naive bias step
        estimation without any additional corrections.
    naive_r_frac: float
        Fractional resistance of TES, given by r_tes / r_n. This is based on the
        naive bias step estimation without any additional corrections.
    naive_p_bias: float
        Bias power on the TES [W] computed using bias steps at the bias point.
        This is based on the naive bias step estimation without any additional
        corrections.
    naive_s_i: float
        Current responsivity of the TES [1/V] computed using bias steps at the
        bias point. This is based on the naive bias step estimation without
        using any additional corrections.
    bandpass: str
        Detector bandpass, computed from bias group information.
    """

    readout_id: str = ""
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

    naive_r_tes: float = np.nan
    naive_r_frac: float = np.nan
    naive_p_bias: float = np.nan
    naive_s_i: float = np.nan
    bandpass: str = "NC"

    @classmethod
    def dtype(cls) -> List[Tuple[str, Any]]:
        """Returns ResultSet dtype for an item based on this class"""
        dtype = []
        for field in fields(cls):
            if field.name == "readout_id":
                dt: Tuple[str, Any] = ("dets:readout_id", "<U40")
            elif field_name == 'bandpass':
                # Our bandpass str is at max 4 characters
                dt: Tuple[str, Any] = ("bandpass", "<U4")
            else:
                dt = (field.name, field.type)
            dtype.append(dt)
        return dtype


@dataclass
class ObsInfo:
    """
    Class containing observation gathered from obsdbs and the file system
    required to compute calibration results.

    Attributes
    ------------
    obs_id: str
        Obs id.
    am: AxisManager
        AxisManager containing metadata for the given observation.
    iv_obsids: dict
        Dict mapping detset to iv obs-id.
    bs_obsids: dict
        Dict mapping detset to bias step obs-id.
    iva_files: dict
        Dict mapping detset to IV analysis file path.
    bsa_files: dict
        Dict mapping detset to bias step analysis file path.
    """

    obs_id: str
    am: core.AxisManager

    iv_obsids: Dict[str, str]
    bs_obsids: Dict[str, str]

    iva_files: Dict[str, str]
    bsa_files: Dict[str, str]


@dataclass
class ObsInfoResult:
    obs_id: str
    success: bool = False
    traceback: str = ""
    obs_info: Optional[ObsInfo] = None


def get_obs_info(cfg: DetCalCfg, obs_id: str) -> ObsInfoResult:
    res = ObsInfoResult(obs_id)

    try:
        ctx = core.Context(cfg.context_path)
        am = ctx.get_obs(
            obs_id,
            samples=(0, 1),
            ignore_missing=True,
            no_signal=True,
            on_missing={"det_cal": "skip"},
        )

        if "smurf" not in am.det_info:
            raise ValueError(f"Missing smurf info for {obs_id}")

        logger.debug(f"Getting cal obsids ({obs_id})")

        iv_obsids = get_cal_obsids(ctx, obs_id, "iv")

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
                timecode = oid.split("_")[1][:5]
                zsmurf_dir = os.path.join(
                    cfg.data_root, "oper", timecode, oid, f"Z_smurf"
                )
                for f in os.listdir(zsmurf_dir):
                    if "iv" in f:
                        iva_files[dset] = os.path.join(zsmurf_dir, f)
                        break
                else:
                    raise ValueError(f"IV data not found for in cal obs {oid}")
            else:
                logger.debug("missing IV data for %s", dset)

        if len(iva_files) == 0:
            raise ValueError(f"No IV data found for {obs_id}")

        # Load in bias steps
        bias_step_obsids = get_cal_obsids(ctx, obs_id, "bias_steps")
        for dset, oid in bias_step_obsids.items():
            if oid is not None:
                timecode = oid.split("_")[1][:5]
                zsmurf_dir = os.path.join(
                    cfg.data_root, "oper", timecode, oid, f"Z_smurf"
                )
                for f in os.listdir(zsmurf_dir):
                    if "bias_step" in f:
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

        res.obs_info = ObsInfo(
            obs_id=obs_id,
            am=am,
            iv_obsids=iv_obsids,
            bs_obsids=bias_step_obsids,
            iva_files=iva_files,
            bsa_files=bsa_files,
        )
        res.success = True
    except:
        res.traceback = traceback.format_exc()
        if cfg.raise_exceptions:
            raise
    return res


@dataclass
class CalRessetResult:
    """
    Results object for the get_cal_resset function.
    """

    obs_info: ObsInfo
    success: bool = False
    traceback: Optional[str] = None
    fail_msg: Optional[str] = None

    correction_results: Optional[Dict[str, List[tpc.CorrectionResults]]] = None
    result_set: Optional[np.ndarray] = None


def get_cal_resset(cfg: DetCalCfg, obs_info: ObsInfo, pool=None) -> CalRessetResult:
    """
    Returns calibration ResultSet for a given ObsId. This pulls IV and bias step
    data for each detset in the observation, and uses that to compute CalInfo
    for each detector in the observation.

    Args
    ------
    cfg: DetCalCfg
        DetCal configuration object.
    obs_info: ObsInfo
        ObsInfo object.
    pool: Optional[multiprocessing.Pool]
        If specified, will run TES param correction in parallel using processes
        from this pool.
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
            dset: IVAnalysis.load(iva_file)
            for dset, iva_file in obs_info.iva_files.items()
        }
        bsas = {
            dset: BiasStepAnalysis.load(bsa_file)
            for dset, bsa_file in obs_info.bsa_files.items()
        }

        if cfg.apply_cal_correction:
            for iva in ivas.values():
                # Run R_L correction if analysis version is old...
                if getattr(iva, "analysis_version", 0) == 0:
                    # This will edit IVA dicts in place
                    logger.debug("Recomputing IV analysis for %s", obs_id)
                    tpc.recompute_ivpars(iva, cfg.param_correction_config)

        iva = list(ivas.values())[0]
        rtm_bit_to_volt = iva.meta["rtm_bit_to_volt"]
        pA_per_phi0 = iva.meta["pA_per_phi0"]
        cals = [CalInfo(rid) for rid in am.det_info.readout_id]
        if len(cals) == 0:
            raise ValueError(f"No detectors found for {obs_id}")

        # Add IV info
        for i, cal in enumerate(cals):
            band = am.det_info.smurf.band[i]
            chan = am.det_info.smurf.channel[i]
            detset = am.det_info.detset[i]
            iva = ivas[detset]

            if iva is None:  # No IV analysis for this detset
                continue

            ridx = np.where((iva.bands == band) & (iva.channels == chan))[0]
            if not ridx:  # Channel doesn't exist in IV analysis
                continue

            ridx = ridx[0]
            cal.bg = iva.bgmap[ridx]
            cal.polarity = iva.polarity[ridx]
            cal.r_n = iva.R_n[ridx]  # type: ignore
            cal.p_sat = iva.p_sat[ridx]  # type: ignore

        obs_biases = dict(
            zip(am.bias_lines.vals, am.biases[:, 0] * 2 * rtm_bit_to_volt)
        )
        bias_line_is_valid = {k: True for k in obs_biases.keys()}

        # check to see if biases have changed between bias steps and obs
        for bsa in bsas.values():
            if bsa is None:
                continue

            for bg, vb_bsa in enumerate(bsa.Vbias):
                bl_label = f"{bsa.meta['stream_id']}_b{bg:0>2}"
                # Usually we can count on bias voltages of bias lines >= 12 to be
                # Nan, however we have seen cases where they're not, so we also
                # restrict by count.
                if np.isnan(vb_bsa) or bg >= TES_BIAS_COUNT:
                    bias_line_is_valid[bl_label] = False
                    continue

                if np.abs(vb_bsa - obs_biases[bl_label]) > 0.1:
                    logger.debug(
                        "bias step and obs biases don't match for %s", bl_label
                    )
                    bias_line_is_valid[bl_label] = False

        # Add TES corrected params
        correction_results: Dict[str, List[tpc.CorrectionResults]] = {}
        if cfg.apply_cal_correction:
            logger.debug("Applying TES param corrections (%s)", obs_id)
            for dset in bsas:
                # logger.debug(f"Applying correction for {dset}")
                rs = []
                if pool is None:
                    for b, c in zip(ivas[dset].bands, ivas[dset].channels):
                        chdata = tpc.RpFitChanData.from_data(
                            ivas[dset], bsas[dset], b, c
                        )
                        rs.append(
                            tpc.run_correction(chdata, cfg.param_correction_config)
                        )
                else:
                    rs = tpc.run_corrections_parallel(
                        ivas[dset], bsas[dset], cfg.param_correction_config, pool=pool
                    )
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

            bl_label = f"{stream_id}_b{bg:0>2}"
            if not bias_line_is_valid[bl_label]:
                continue

            ridx = np.where((bsa.bands == band) & (bsa.channels == chan))[0]
            if not ridx:  # Channel doesn't exist in bias step analysis
                continue

            if cfg.apply_cal_correction:
                correction = find_correction_results(band, chan, detset)
                if correction is None:
                    logger.warn(
                        "Unable to find correction result for %s %s %s (%s)",
                        band,
                        chan,
                        detset,
                        obs_id,
                    )
                    use_correction = False
                    cal.tes_param_correction_success = False
                else:
                    use_correction = correction.success
                    cal.tes_param_correction_success = correction.success
            else:
                use_correction = False

            ridx = ridx[0]
            cal.tau_eff = bsa.tau_eff[ridx]
            if bg != -1:
                cal.v_bias = bsa.Vbias[bg]

            if use_correction and correction.corrected_params is not None:
                cpars = correction.corrected_params
                cal.r_tes = cpars.corrected_R0
                cal.r_frac = cpars.corrected_R0 / cal.r_n
                cal.s_i = cpars.corrected_Si * 1e6
                cal.p_bias = cpars.corrected_Pj * 1e-12
                cal.loopgain = cpars.loopgain
            else:
                cal.r_tes = bsa.R0[ridx]
                cal.r_frac = bsa.Rfrac[ridx]
                cal.p_bias = bsa.Pj[ridx]
                cal.s_i = bsa.Si[ridx]

            # Save naive parameters even if we're using corrected version
            cal.naive_r_tes = bsa.R0[ridx]
            cal.naive_r_frac = bsa.Rfrac[ridx]
            cal.naive_s_i = bsa.Si[ridx]
            cal.naive_p_bias = bsa.Pj[ridx]

            if cal.s_i == 0:
                cal.phase_to_pW = np.nan
            else:
                cal.phase_to_pW = pA_per_phi0 / (2 * np.pi) / cal.s_i * cal.polarity

            # Add bandpass informaton from bias group
            if cal.bg in BGS['lb']:
                cal.bandpass = BAND_STR[tube_flavor]['lb']
            elif cal.bg in BGS['hb']:
                cal.bandpass = BAND_STR[tube_flavor]['hb']

        res.result_set = np.array([astuple(c) for c in cals], dtype=CalInfo.dtype())
        res.success = True
    except Exception as e:
        res.traceback = traceback.format_exc()
        res.fail_msg = res.traceback
        if cfg.raise_exceptions:
            raise e
    return res


def get_obsids_to_run(cfg: DetCalCfg) -> List[str]:
    """
    Returns list of obs-ids to process, based on the configuration object.
    This will included non-processed obs-ids that are not found in the fail cache,
    and will be limitted to cfg.num_obs.
    """
    ctx = core.Context(cfg.context_path)
    # Find all obs_ids that have not been processed
    with open(cfg.failed_cache_file, "r") as f:
        failed_cache = yaml.safe_load(f)

    if failed_cache is not None:
        failed_obsids = set(failed_cache.keys())
    else:
        failed_obsids = set()

    db = core.metadata.ManifestDb(cfg.index_path)
    obs_ids_all = set(ctx.obsdb.query('type=="obs"')["obs_id"])
    processed_obsids = set(db.get_entries(["dataset"])["dataset"])
    obs_ids = sorted(list(obs_ids_all - processed_obsids - failed_obsids), reverse=True)
    if cfg.num_obs is not None:
        obs_ids = obs_ids[: cfg.num_obs]
    return obs_ids


def add_to_failed_cache(cfg: DetCalCfg, obs_id: str, msg: str) -> None:
    if "KeyboardInterrupt" in msg:  # Don't cache keyboard interrupts
        return

    if cfg.cache_failed_obsids:
        logger.info(f"Adding {obs_id} to failed_file_cache")
        with open(cfg.failed_cache_file, "r") as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        d[str(obs_id)] = msg
        with open(cfg.failed_cache_file, "w") as f:
            yaml.dump(d, f)
        return


def handle_result(result: CalRessetResult, cfg: DetCalCfg) -> None:
    """
    Handles a CalRessetResult. If successful, this will add to the manifestdb,
    if not this will add to the failed cache if cfg.cache_failed_obsids is True.
    """
    obs_id = str(result.obs_info.obs_id)
    if not result.success:
        logger.error(f"Failed on obs_id: {obs_id}")
        logger.error(result.traceback)

        msg = result.fail_msg
        if msg is None:
            msg = "unknown error"
        add_to_failed_cache(cfg, obs_id, msg)
        return

    logger.info(f"Adding obs_id {obs_id} to dataset")
    rset = ResultSet.from_friend(result.result_set)
    write_dataset(rset, cfg.h5_path, obs_id, overwrite=True)
    db = core.metadata.ManifestDb(cfg.index_path)
    relpath = os.path.relpath(cfg.h5_path, start=os.path.dirname(cfg.index_path))
    db.add_entry(
        {"obs:obs_id": obs_id, "dataset": obs_id}, filename=relpath, replace=True
    )


def run_update_site(cfg: DetCalCfg) -> None:
    """
    Main run script for computing det-cal results at the site. This will
    loop over obs-ids and serially gather the ObsInfo from the filesystem and
    sqlite dbs, and then compute the calibration results. A processing pool
    of cfg.nprocs_result_set processes will be used to parallelize the TES
    correction computation. If you have lots of compute power, and are limitted
    by filesystem or sqlite access, consider using the 'nersc' update function.

    Args:
    ------
    cfg: DetCalCfg or str
        DetCalCfg object or path to config yaml file
    """
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, cfg.log_level.upper()))

    obs_ids = get_obsids_to_run(cfg)

    logger.info(f"Processing {len(obs_ids)} obsids...")

    mp.set_start_method(cfg.multiprocess_start_method)
    with mp.Pool(cfg.nprocs_result_set) as pool:
        for oid in tqdm(obs_ids, disable=(not cfg.show_pb)):
            res = get_obs_info(cfg, oid)
            if not res.success:
                logger.info(f"Could not get obs info for obs id: {oid}")
                logger.error(res.traceback)

            if res.obs_info is None:
                continue

            result_set = get_cal_resset(cfg, res.obs_info, pool=pool)
            handle_result(result_set, cfg)


def run_update_nersc(cfg: DetCalCfg) -> None:
    """
    Main run script for computing det-cal results. This does the same thing as
    ``run_update_site`` however instantiates two separate pools for gathering
    ObsInfo and computing the ResultSets. This is useful in situations where
    sqlite/filesystem access are bottlenecks (such as nersc) so that ObsInfo can
    be gathered in parallel, and this can be done while ResultSet computation is
    ongoing. Because concurrent sqlite access can be limitted, it is recommended
    to keep cfg.nprocs_obs_info low (<10), while ``cfg.nprocs_result_set`` can be
    set arbitrarily large as to use remaining available resources.

    Args:
    ------
    cfg: DetCalCfg or str
        DetCalCfg object or path to config yaml file
    """
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, cfg.log_level.upper()))

    obs_ids = get_obsids_to_run(cfg)
    # obs_ids = ['obs_1713962395_satp1_0000100']
    # obs_ids = ['obs_1713758716_satp1_1000000']
    logger.info(f"Processing {len(obs_ids)} obsids...")

    pb = tqdm(total=len(obs_ids), disable=(not cfg.show_pb))

    def callback(result: CalRessetResult):
        pb.update()
        handle_result(result, cfg)

    def errback(e):
        logger.info(e)
        raise e

    # We split into multiple pools because:
    # - we don't want to overload sqlite files with too much concurrent access
    # - we want to be able to continue getting the next obs_info data while ressets are being computed
    mp.set_start_method(cfg.multiprocess_start_method)
    pool1 = mp.Pool(cfg.nprocs_obs_info)
    pool2 = mp.Pool(cfg.nprocs_result_set)

    resset_async_results: Queue = Queue()
    obsinfo_async_results: Queue = Queue()

    def get_obs_info_callback(result: ObsInfoResult):
        if result.success:
            r = pool2.apply_async(
                get_cal_resset,
                args=(cfg, result.obs_info),
                callback=callback,
                error_callback=errback,
            )
            resset_async_results.put(r)
        else:
            pb.update()
            add_to_failed_cache(cfg, result.obs_id, result.traceback)
            logger.error(
                f"Failed to get obs_info for {result.obs_id}:\n{result.traceback}"
            )

    try:
        for obs_id in obs_ids:
            a = pool1.apply_async(
                get_obs_info,
                args=(cfg, obs_id),
                callback=get_obs_info_callback,
                error_callback=errback,
            )
            obsinfo_async_results.put(a)

        while not obsinfo_async_results.empty():
            obsinfo_async_results.get().wait()
        while not resset_async_results.empty():
            resset_async_results.get().wait()

    finally:
        pool1.terminate()
        pool1.join()
        pool2.terminate()
        pool2.join()
    pb.close()
    logger.info("Finished updates")


def main(config_file: str) -> None:
    """
    Run update function. This will chose the correct method to run based on
    ``cfg.run_method``.
    """
    cfg = DetCalCfg.from_yaml(config_file)

    if cfg.run_method == "site":
        run_update_site(cfg)
    elif cfg.run_method == "nersc":
        run_update_nersc(cfg)
    else:
        raise ValueError(f"Unknown run_method: {cfg.run_method}")


def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        p = argparse.ArgumentParser()
    else:
        p = parser
    p.add_argument(
        "config_file", type=str, help="yaml file with configuration for update script."
    )
    return p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(config_file=args.config_file)
