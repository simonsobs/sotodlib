"""
This module is used to compute anomalous delay of sampling
in the smurf readout mostly in smurf band0 and part of band1.

It is done by anlyzing the bias group mapping (bgmap) data.
"""
import traceback
import os
import yaml
import argparse
import logging
from tqdm.auto import tqdm, trange
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, Callable, Self
from numpy.typing import NDArray
import numpy as np
from scipy.optimize import leastsq
from sotodlib import core
from sotodlib.io import load_book
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.site_pipeline.utils.logging import init_logger as sp_init_logger
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("bgmap_delay")
if not logger.hasHandlers():
    sp_init_logger("bgmap_delay")

#######################################################################

@dataclass
class DataCfg:
    '''
    Configurations for loading data

    context_path: str
        Path to the context file to use. (required)
    metadata_list: "all" or list of str
        List of metadata labels to load. (default: ['smurf'])
    '''
    context_path: str
    metadata_list: Union[str, list[str]] = field(default_factory=lambda: ['smurf'])


@dataclass
class OutputCfg:
    '''
    Configurations for output

    root_dir: str
        Path to the root of the results directory.
    obs_index_path: str = "bgmap_delay.sqlite"
        Path to the index file to use for the database per obs.
    bgmap_index_path: str = "bgmapdb.sqlite"
        Path to the index file to use for bgmap database.
    h5_path: str = "bgmap_delay.h5"
        Path to the HDF5 file to use for the delay_cal database.
    h5_unix_digits: int = 4
        Number of digits of unixtime to be added to h5_path.
    cache_failed_obsids: bool = True
        If True, will cache failed obs-ids to avoid re-running them.
    failed_bgmap_file: str = "failed_bgmaps.yaml"
        Path to the yaml file that will store failed bgmaps.
    failed_obsid_file: str = "failed_obsids.yaml"
        Path to the yaml file that will store failed obsids.
    '''
    root_dir: str
    obs_index_path: str = "bgmap_delay.sqlite"
    bgmap_index_path: str = "bgmapdb.sqlite"
    h5_path: str = "delay_cal.h5"
    h5_unix_digits: int = 4
    cache_failed_obsids: bool = True
    failed_bgmap_file: str = "failed_bgmaps.yaml"
    failed_obsid_file: str = "failed_obsids.yaml"

    def __post_init__(self):
        """ Check path and make database """
        self.root_dir = os.path.expandvars(self.root_dir)
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root dir does not exist: {self.root_dir}")

        def parse_path(path: str) -> str:
            "Expand vars and make path absolute"
            p = os.path.expandvars(path)
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
            return p

        self.obs_index_path = parse_path(self.obs_index_path)
        self.bgmap_index_path = parse_path(self.bgmap_index_path)
        self.h5_path = parse_path(self.h5_path)
        self.failed_bgmap_file = parse_path(self.failed_bgmap_file)
        self.failed_obsid_file = parse_path(self.failed_obsid_file)

        self.setup_files()


    def setup_files(self) -> None:
        """Create directories and databases if they don't exist"""
        if self.cache_failed_obsids:
            failed_files  = [self.failed_bgmap_file,
                             self.failed_obsid_file]
            for failed_file in failed_files:
                if not os.path.exists(failed_file):
                    # If file doesn't exist yet, just create an empty one
                    with open(failed_file, "w") as f:
                        yaml.dump({}, f)

        index_paths = [self.obs_index_path,
                       self.bgmap_index_path]
        for index_path in index_paths:
            if not os.path.exists(index_path):
                scheme = core.metadata.ManifestScheme()
                scheme.add_exact_match("obs:obs_id")
                scheme.add_data_field("dataset")
                db = core.metadata.ManifestDb(scheme=scheme)
                db.to_file(index_path)
                pass

    def get_h5_path(self, obs_id:str) -> str:
        """ Return h5 file name based on cfg """
        h5_path = self.h5_path
        if self.h5_unix_digits:
            name, ext = os.path.splitext(self.h5_path)
            unixtime = obs_id.split('_')[1][:self.h5_unix_digits]
            h5_path = f"{name}_{unixtime}{ext}"
        return h5_path

    def add_to_failed_cache(
            self,
            obs_id: str,
            msg: str,
            which: Literal['bgmap', 'obs'],
    ) -> None:
        """ Add obs into failed cache """

        if which == 'bgmap':
            failed_file_cache = self.failed_bgmap_file
        elif which == 'obs':
            failed_file_cache = self.failed_obsid_file

        if "KeyboardInterrupt" in msg:  # Don't cache keyboard interrupts
            return

        # Transient errors of metadata loading. We will retry later.
        transient_errors = [
            'sotodlib.core.metadata.loader.LoaderError',
            'BlockingIOError',
        ]
        for err in transient_errors:
            if err in msg:
                logger.error(f"obs_id {obs_id} failed to load metadata {err}."
                             " Try again later")
                return

        if not self.cache_failed_obsids:
            return

        logger.info(f"Adding {obs_id} to failed_file_cache")
        with open(failed_file_cache, "r") as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        d[str(obs_id)] = msg
        with open(failed_file_cache, "w") as f:
            yaml.dump(d, f)
        return


@dataclass
class AnalysisCfg:
    '''
    Configurations for analysis

    poly_order: int = 1
        Order of polynomials to fit the response.
    box_sample_width: float = 1.5
        With of the box window in sample number.
    sc_amp_thresh: float = 0.1
        Bias step |dItes/dIbias-1| threshold above which
        TES should be considered not superconducting and flagged.
    tau_thresh: float = 0.0005
        Superconducting timeconstant threshold [sec] above which
        TES should be considered not superconducting and flagged.
    '''
    poly_order: int = 1
    box_sample_width: float = 1.5
    sc_amp_thresh: float = 0.1
    tau_thresh: float = 0.0005


@dataclass
class ProcessCfg:
    '''
    Configurations for processing

    multiprocess_start_method: Literal["spawn", "fork"] = "spawn"
        Method to use to start child processes.
    nprocs_channel: int = 4
        Number of parallel processes that should to compute the TES parameters,
    num_bgmap: Optional[int] = None
    num_obs: Optional[int] = None
        Max number of observations to process per run_update call. If not set,
        will run on all available observations.
    raise_exceptions: bool = True
        If Exceptions should be raised.
    log_level: str = "DEBUG"
        Logging level for the logger.
    show_pb: bool = True
        If True, show progress bar in the run_update function.
    '''
    multiprocess_start_method: Literal["spawn", "fork"] = "spawn"
    nprocs_channel: int = 4
    num_bgmap: Optional[int] = None
    num_obs: Optional[int] = None
    raise_exceptions: bool = True
    log_level: str = "DEBUG"
    show_pb: bool = True


@dataclass
class BgmapDelayCfg:
    '''
    All configurations

    data: DataCfg
    output: OutputCfg
    analysis: AnalysisCfg
    process: ProcessCfg
    '''
    data: DataCfg
    output: OutputCfg
    analysis: AnalysisCfg
    process: ProcessCfg

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """
        BgmapDelayCfg from yaml file
        """
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls(
            data=DataCfg(**d['data']),
            output=OutputCfg(**d['output']),
            analysis=AnalysisCfg(**d.get('analysis', {})),
            process=ProcessCfg(**d.get('process', {})))

    def get_obsids_to_run(
            self,
            which: Literal['bgmap', 'obs'],
    ) -> list[str]:
        """
        Returns list of obs-ids to process, based on the configuration object.
        This will included non-processed obs-ids that are not found in the fail cache,
        and will be limitted to num_process.

        which: 'bgmap' or 'obs'
        """
        if which == 'bgmap':
            failed_file = self.output.failed_bgmap_file
            index_path = self.output.bgmap_index_path
            query = "type=='oper' and subtype='bgmap'"
            num_process = self.process.num_bgmap
        elif which == 'obs':
            failed_file = self.output.failed_obsid_file
            index_path = self.output.obs_index_path
            query = "type=='obs'"
            num_process = self.process.num_obs
        else:
            raise ValueError(f'which is not bgmap or obs but {which}')

        # Find all obs_ids that have not been processed
        with open(failed_file, "r") as f:
            failed_cache = yaml.safe_load(f)

        if failed_cache is not None:
            failed_obsids = set(failed_cache.keys())
        else:
            failed_obsids = set()

        db = core.metadata.ManifestDb(index_path)
        processed_obsids = set(
            db.get_entries(["obs:obs_id"])["obs:obs_id"])

        ctx = core.Context(
            self.data.context_path,
            metadata_list=self.data.metadata_list)
        obs_ids_all = set(ctx.obsdb.query(query)["obs_id"])

        obs_ids = sorted(list(obs_ids_all - processed_obsids - failed_obsids), reverse=True)
        obs_ids = obs_ids[: num_process]
        return obs_ids



####################################################


@dataclass
class BiasStepAnalysis:
    """
    Minimum set of BiasStepAnalysis from sodetlib

    bands: NDArray[np.int64]
    channels: NDArray[np.int64]
    bgmap: NDArray[np.int64]
    dIbias: NDArray[np.float64]
    resp_times: NDArray[np.float64]
    mean_resp: NDArray[np.float64]
    pts_before_step: int = 20
    """
    bands: NDArray[np.int64]
    channels: NDArray[np.int64]
    bgmap: NDArray[np.int64]
    dIbias: NDArray[np.float64]
    resp_times: NDArray[np.float64]
    mean_resp: NDArray[np.float64]
    pts_before_step: int = 20 # samples before bias step at t=0

    @classmethod
    def load(cls, ctx:core.Context, obs_id:str) -> Self:
        """
        Find and load BiasStepAnalysis from context
        """
        # Note bgmap is almost same with bias_step but ~zero bias
        d = load_book.load_smurf_npy_data(ctx, obs_id, 'bias_step')
        use = ['bands', 'channels', 'bgmap',
               'dIbias', 'resp_times', 'mean_resp']
        d = {k:d[k] for k in use}
        for t in d['resp_times']:
            if np.isnan(t).any():
                continue
            d['pts_before_step'] = np.argmin(np.abs(t))
            break
        return cls(**d)


@dataclass
class ChannelData:
    """
    Data of single channel from BiasStepAnalysis
    used for fitting and indexing in output

    band: smurf band
    channel: smurf channel
    bg: bias group
    fsample: sampling frequency of bgmap data in Hz. Usually 4000.
    resp_times: time [sec] around the bias step at t=0.
    mean_resp: averaged channel response (current).
       Here, we normalize this by the step of the bias current.
       So that it should be ~1 if channel is superconducting.
    pts_before_step: samples before t = 0
    """
    band: int
    channel: int
    bg: int
    fsample: float # Hz
    resp_times: np.ndarray # sec
    mean_resp: np.ndarray # dItes/dIbias
    pts_before_step: int

    @classmethod
    def from_data(
            cls,
            bsa: BiasStepAnalysis,
            band: int,
            channel: int
    ) -> Self:
        """
        Get ChannelData from BiasStepAnalysis data
        by selecting band and channel.
        """
        bs_idx = np.where((bsa.channels == channel) & (bsa.bands == band))[0]
        if len(bs_idx) == 0:
            raise ValueError(
                f"Could not find band={band} channel={channel} in Bias Steps"
            )
        bs_idx = bs_idx[0]

        bg = bsa.bgmap[bs_idx]
        resp_times = bsa.resp_times[bg]
        fsample = 1. / np.diff(resp_times).mean()

        # Use normalized and offset-subtracted mean_resp
        mean_resp = bsa.mean_resp[bs_idx] / bsa.dIbias[bg]
        mean_resp -= mean_resp[:bsa.pts_before_step].mean()

        return cls(
            band=band,
            channel=channel,
            bg=bg,
            fsample=fsample,
            resp_times=resp_times,
            mean_resp=mean_resp,
            pts_before_step=bsa.pts_before_step,
        )


##############################

@dataclass
class ExponBoxParams:
    """
    Model of bgmap response as a convolution of
    exponential function and box window.

    Params:
    --------
    t0: float
        Time of rising after bias step
    amp: float
        Height of the pulse
    tau: float
        time constant of exponential function
    w: float
        width of the box window.
    poly: list[float]
        coefficient of polynomials: offset, slope, ...
    """
    t0: float
    amp: float
    tau: float
    w: float
    poly: list[float]

    def __init__(self,
                 params: NDArray[np.float64],
                 w: Optional[float] = None,
                 ) -> None:
        """
        Construct ExponBoxParams from numpy array.

        params: NDArray[np.float64]
            numpy array of [t0 amp tau (w) (poly0 ...)]
        w: Optional[float] = None
            If given used for w.
        """
        if w is None:
            self.t0, self.amp, self.tau, self.w = params[:4]
            self.poly = params[4:]
        else:
            self.t0, self.amp, self.tau = params[:3]
            self.poly = params[3:]
            self.w = w

    def model(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Expon*Box model function of time t
        """
        t0 = self.t0
        amp = self.amp
        tau = self.tau
        w = self.w
        poly = self.poly

        dx = t - t0
        y = np.zeros_like(dx)
        if w == 0.0:
            i0 = np.searchsorted(dx, 0)
            m = slice(i0, None)
            y[m] = 1.0 - np.exp(-dx[m]/tau)
        else:
            i0 = np.searchsorted(dx, 0)
            iw = np.searchsorted(dx, w)
            m = slice(i0, iw)
            y[m] = (dx[m] - tau * (1. - np.exp(-dx[m]/tau))) / w
            m = slice(iw, None)
            y[m] = (w - tau * np.exp(-dx[m]/tau) * (np.exp(w/tau) - 1.)) / w
        y = amp * y
        y += np.polyval(poly, t)
        return y

    def scale(self, t: float, y:float) -> Self:
        """ Change scale of time and data """
        new_params = np.hstack([
            self.t0 * t,
            self.amp * y,
            self.tau * t,
            self.w * t,
            self.poly * y])
        return ExponBoxParams(new_params)

    def asarray(self) -> NDArray[np.float64]:
        """ Return parameters as numpy array """
        return np.hstack([
            self.t0, self.amp, self.tau, self.w, self.poly])


@dataclass
class FitBounds:
    """
    Control bounds of w and tau which must be positive.
    Here we assue time unit is normalized to sampling index.
    """
    w_fix: Optional[float]
    tau_min: float=0.1
    tau_max: float=10.0

    def __post_init__(self):
        if self.w_fix is not None:
            assert self.w_fix >= 0, "w_fix < 0!"
        assert self.tau_min >= 0, "tau_min < 0!"
        assert self.tau_max > self.tau_min, "tau_max <= tau_min!"

    def check_bound(self, params: NDArray[np.float64]) -> None:
        """ Check parameter bounds and overwrite """
        # w
        if self.w_fix is None and params[3] < 0.0:
            params[3] = 0.0
        # tau
        if params[2] < self.tau_min:
            params[2] = self.tau_min
        if params[2] > self.tau_max:
            params[2] = self.tau_max


def fit_error_func(
        params: NDArray[np.float64],
        t: NDArray[np.float64],
        data: NDArray[np.float64],
        fit_bounds: FitBounds
) -> NDArray[np.float64]:
    """
    Error function called from leastsq
    """
    fit_bounds.check_bound(params)
    model = ExponBoxParams(params, fit_bounds.w_fix).model(t)
    return data - model


def fit_channel(
        chdata: ChannelData,
        cfg: AnalysisCfg,
) -> dict[str, ExponBoxParams]:
    """
    Core process to run fitting for each channel.
    We fit three models: expon, exponbox, exponboxw.
    They are different in the width of the box window.
    """
    # Normalize not to be very small value
    pts_before_step = chdata.pts_before_step
    t = chdata.resp_times * chdata.fsample
    std = chdata.mean_resp[:pts_before_step].std()
    y = chdata.mean_resp / std

    # Usual BiasStepAnalysis has 20 samples before t=0.
    y1 = y[:pts_before_step].mean()
    y2 = y[-pts_before_step:].mean()

    # Select model by changing w_fix parameter.
    fit_models = [
        ('expon', 0.0),
        ('exponbox', None),
        ('exponboxw', cfg.box_sample_width)]
    results = {}
    for model, w_fix in fit_models:
        fit_bounds = FitBounds(w_fix=w_fix)
        params0 = [0.1, y2 - y1, 2.0, 1.0, y1] + [0.0] * cfg.poly_order
        if w_fix is not None:
            params0.pop(3)
        res = leastsq(fit_error_func, params0, args=(t, y, fit_bounds))
        normalized_popt = ExponBoxParams(res[0], w_fix)
        popt = normalized_popt.scale(t=1./chdata.fsample, y=std)
        results[model] = popt
    return results


@dataclass
class ChannelResult:
    """
    Container for results of channel process

    chdata: ChannelData
        Info from BiasStepAnalysis
    results: Optional[dict[str, ExponBoxParams]]
        dict of ExponBoxParams for each model
    """
    chdata: ChannelData
    results: Optional[dict[str, ExponBoxParams]] = None
    success: bool = False
    traceback: Optional[str] = None


def run_channel(
        chdata: ChannelData,
        cfg: BgmapDelayCfg,
) -> ChannelResult:
    """
    Process for single channel to fit response

    chdata: ChannelData
        Data from BiasStepAnalysis for single channel
    """
    out = ChannelResult(chdata)
    try:
        if chdata.bg < 0:
            raise ValueError("Not assigned to any bias group")
        out.results = fit_channel(chdata, cfg.analysis)
        out.success = True
    except Exception:
        if cfg.process.raise_exceptions:
            raise
        else:
            out.traceback = traceback.format_exc()
            out.success = False
    return out


def run_channel_parallel(
        bsa: BiasStepAnalysis,
        cfg: BgmapDelayCfg,
        executor=None,
        as_completed_callable=None
) -> list[ChannelResult]:
    """
    Process for single bgmap to call run_channel in parallel

    bsa: BiasStepAnalysis
        bgmap data
    cfg.process.nprocs_channel: int
        Number of parallel processes
    executor, as_completed_callable
        from main or somewhere.
        If not given, created here.
    """
    # Error handler
    def errback(e):
        raise e

    # Create executor (optionally externally provided)
    if executor is None:
        nprocs = cfg.process.nprocs_channel
        executor = ProcessPoolExecutor(max_workers=nprocs)
        as_completed_callable = as_completed
        close_executor = True
    else:
        close_executor = False

    out = []

    nchans = len(bsa.channels)
    pb = trange(nchans, disable=(not cfg.process.show_pb))
    try:
        futures = []
        for idx in range(nchans):
            chdata = ChannelData.from_data(
                bsa, bsa.bands[idx], bsa.channels[idx]
            )
            future = executor.submit(run_channel, chdata, cfg)
            futures.append(future)

        for future in as_completed_callable(futures):
            try:
                res = future.result()
                out.append(res)
            except Exception as e:
                errback(e)

            futures.remove(future)
            pb.update(1)

        pb.close()

    finally:
        if close_executor:
            executor.shutdown(wait=True, cancel_futures=True)

    return out


def pack_into_aman(cfg:BgmapDelayCfg,
                   results:list[ChannelResult],
                   det_info:core.AxisManager,
) -> core.AxisManager:
    """
    Pack results and some info into AxisManager.

    Args:
    ------
    cfg.analysis.sc_amp_thresh: float
    cfg.analysis.tau_thresh: float
        Thresholds to find reasonable i.e. superconducting channels
    results: list[ChannelResult]
        List of ChannelResult from run_channel
    det_info: AxisManager
        meta.det_info including dets and smurf

    Return: AxisManager
        dets : LabelAxis
            Axis for channels. It should be readout_id.
        fsample: float(dets)
            Sampling rate of the bgmap data in Hz, usually 4000.
        popt_expon: AxisManager(t0[dets], amp[dets], tau[dets], w[dets], poly0[dets], ...)
            Best fit values for w=0 model.
        popt_exponbox: AxisManager(t0[dets], amp[dets], tau[dets], w[dets], poly0[dets], ...)
            Best fit values varying w.
        popt_exponbox: AxisManager(t0[dets], amp[dets], tau[dets], w[dets], poly0[dets], ...)
            Best fit values for w=cfg.analysis.box_sample_width.
    """
    models = ['expon', 'exponbox', 'exponboxw']

    # Define axis
    dets = det_info.dets
    params = (
        ['t0', 'amp', 'tau', 'w'] +
        [f'poly{i}' for i in range(cfg.analysis.poly_order + 1)])

    # Make AxisManager and initialize
    aman = core.AxisManager(dets)
    fsample = aman.wrap_new('fsample', (dets,), dtype='float64')
    fsample[:] = np.nan

    popts = {}
    for model in models:
        name = f'popt_{model}'
        popt_aman = popts[model] = core.AxisManager(dets)
        for param in params:
            popt_aman.wrap_new(param, (dets,), dtype='float64')
            popt_aman[param][:] = np.nan
        aman.wrap(name, popt_aman)


    # Fill results
    bands = det_info.smurf.band
    channels = det_info.smurf.channel
    db_idx_map = {
        (b, c): i
        for i, (b, c) in enumerate(zip(bands, channels))}
    for r in results:
        if not r.success:
            continue
        # Check index in dets
        band = r.chdata.band
        channel = r.chdata.channel
        bc = (band, channel)
        if not bc in db_idx_map:
            logger.debug(f"missing dets entry for {band},{channel}")
            continue
        idx = db_idx_map[bc]

        # Fill
        fsample[idx] = r.chdata.fsample
        for model in models:
            for p, v in zip(params, r.results[model].asarray()):
                popts[model][p][idx] = v

    return aman

def process_aman(cfg: BgmapDelayCfg,
                 aman: core.AxisManager,
                 det_info: core.AxisManager,
) -> core.AxisManager:
    """
    Process AxisManager to evaluate delay.

    Args:
    ------
    cfg.analysis.sc_amp_thresh: float
    cfg.analysis.tau_thresh: float
        Thresholds to find reasonable i.e. superconducting channels
    aman: AxisManager(
            fsample[dets],
            popt_expon*[dets],
            popt_exponbox*[dets],
            popt_exponboxw*[dets]
            dets:LabelAxis)
        From pack_into_aman
    det_info: AxisManager
        meta.det_info including dets and smurf

    Return: AxisManager
        dets:LabelAxis,
        fsample[dets],
        popt_expon*[dets],
        popt_exponbox*[dets],
        popt_exponboxw*[dets],
        fsample: float[dets]
            Sampling rate of the bgmap data in Hz, usually 4000.
            Here we round it and fill the value into nan channels.
        model: str
            Main model used
        mean_delay: float[dets]
            Median delay among channels [sec]
        channel_delay: float[dets]
            Relative delay from the mean for each channel [sec]
            The unit is quantized to sampling = 1 / fsample.
        flag: int(dets)
            Flag if the detector condition seems good.
            bit0: t0 < 0
            bit1: tau < tau_thresh
            bit2: | amp / median(amp) - 1 | >  sc_amp_thresh
            bit3: deviation of delay from mean > sample_error_threshold
            bit4: isolated channel
    """
    main_model = 'exponboxw'
    sample_error_threshold = 0.2

    # Channel delay should appear as a step in readout channel
    dets = det_info.dets
    band = det_info.smurf.band
    channel = det_info.smurf.channel
    bandchannel = band * 512 + channel
    sort_in_bc = np.argsort(bandchannel)
    sort_in_dets = np.argsort(sort_in_bc)

    ndets = dets.count
    idx = np.arange(ndets)

    aman.wrap_new('mean_delay', (dets,), dtype='float64')
    aman.wrap_new('channel_delay', (dets,), dtype='float64')
    aman.wrap_new('flag', (dets,), dtype='int64')

    
    fsample = np.round(np.nanmedian(aman.fsample))
    aman.fsample[:] = fsample
    popt = aman['popt_' + main_model]
    t0 = popt.t0[sort_in_bc]
    amp = popt.amp[sort_in_bc]
    tau = popt.tau[sort_in_bc]
    sample0 = t0 * fsample
    flag = np.zeros(ndets, dtype=np.int64)

    # Select reliable channel
    ok = t0 > 0
    bad = ~ok
    flag[bad] += 1
    if cfg.analysis.tau_thresh is not None:
        bad = (flag==0) * (tau > cfg.analysis.tau_thresh)
        flag[bad] += 1 << 1
    use = flag == 0
    if cfg.analysis.sc_amp_thresh is not None:
        ramp = amp / np.nanmedian(amp[use])
        bad = (flag==0) * (np.abs(ramp - 1.0) > cfg.analysis.sc_amp_thresh)
        flag[bad] += 1 << 2
    use = flag == 0
    offset = np.median(sample0[use] % 1)
    offset = np.nanmedian((sample0[use] - offset + 0.5) % 1 - 0.5 + offset)
    ok = np.abs((sample0 - offset + 0.5) % 1 - 0.5) <= sample_error_threshold
    bad = (flag==0) * (~ok)
    flag[bad] += 1 << 3
    use = flag == 0
    # Drop isolated channel
    x = np.round(sample0[use] - offset)
    for _ in range(3):
        ok = np.hstack([
            x[1] == x[0],
            (x[1:-1] == x[:-2]) + (x[1:-1] == x[2:]),
            x[-2] == x[-1]])
        if ok.all():
            break
        use[use] = ok
        if not use.any():
            break
        x = np.round(sample0[use] - offset)
    bad = (flag==0) * (~use)
    flag[bad] += 1 << 4
    channel_delay = np.round(np.interp(idx, idx[use], x)) / fsample

    # Sort back to dets
    flag = flag[sort_in_dets]
    channel_delay = channel_delay[sort_in_dets]
    
    # Put into aman
    aman.wrap('model', main_model)
    aman['mean_delay'][:] = offset / fsample
    aman['flag'][:] = flag
    aman['channel_delay'][:] = channel_delay
    return aman


##############################################################

@dataclass
class RunBgmapResult:
    """
    Container for results of bgmap process

    results: AxisManager
        Fit parameters and misc. for dets.
    """
    bgmap_id: str
    success: bool = False
    traceback: str = ""
    results: Optional[core.AxisManager] = None


def run_bgmap_process(
        cfg: BgmapDelayCfg,
        bgmap_id: str,
        executor=None,
        as_completed_callable=None
) -> RunBgmapResult:
    """
    Process for each bgmap.
    Load data and run fitting for each channel in parallel.

    Args:
    ------
    cfg.data: DataCfg
        Configurations for context. We need only 'smurf' metadata.
    cfg.analysis: AnalysisCfg
        Configurations for bgmap response analysis
    bgmap_id: str
        obs_id for the bgmap to process

    Return: RunBgmapResult
    """
    logger.debug("Computing Result set for %s", bgmap_id)

    # Need to reset logger here because this may be created new for spawned process
    log_level = cfg.process.log_level.upper()
    logger.setLevel(getattr(logging, log_level))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, log_level))

    res = RunBgmapResult(bgmap_id)
    try:
        # Load data
        ctx = core.Context(
            cfg.data.context_path,
            metadata_list=cfg.data.metadata_list)
        meta = ctx.get_meta(bgmap_id)
        bsa = BiasStepAnalysis.load(ctx, bgmap_id)

        # Fit bias step response for each readout channel
        logger.debug("Run delay fit (%s)", bgmap_id)
        rs = []
        if executor is None:
            for b, c in zip(bsa.bands, bsa.channels):
                chdata = ChannelData.from_data(bsa, b, c)
                r = run_channel(chdata, cfg)
                rs.append(r)
        else:
            rs = run_channel_parallel(
                bsa, cfg, executor=executor,
                as_completed_callable=as_completed_callable)

        # Gather results into AxisManager
        aman = pack_into_aman(cfg, rs, meta.det_info)
        # Analysis with AxisManager
        aman = process_aman(cfg, aman, meta.det_info)
        res.results = aman
        res.success = True
    except Exception as e:
        res.traceback = traceback.format_exc()
        if cfg.process.raise_exceptions:
            raise e
    return res


def handle_bgmap_result(result: RunBgmapResult, cfg: BgmapDelayCfg) -> None:
    """
    Store results into hdf5 and update bgmap database.

    result: RunBgmapResult
        Result from run_bgmap_process for single bgmap
    cfg.output.bgmap_index_path: str
        Path to output databse
    cfg.output.h5_path: str
        Path to the HDF5 file
    cfg.output.h5_unix_digits: int
        Number of digits of unixtime to be added to h5_path.
    cfg.output.cache_failed_obsids: bool
        If True, will cache failed obs-ids to avoid re-running them.
    cfg.output.failed_bgmap_file: str
        Path to the yaml file that will store failed bgmaps.
    """
    bgmap_id = str(result.bgmap_id)
    if not result.success:
        logger.error(f"Failed on bgmap_id: {bgmap_id}")
        logger.error(result.traceback)

        msg = result.traceback
        if msg is None:
            msg = "unknown error"
        cfg.output.add_to_failed_cache(bgmap_id, msg, 'bgmap')
        return

    logger.info(f"Adding bgmap_id {bgmap_id} to dataset")

    h5_path = cfg.output.get_h5_path(bgmap_id)
    aman = result.results
    aman.save(h5_path, bgmap_id, overwrite=True)

    index_path = cfg.output.bgmap_index_path
    db = core.metadata.ManifestDb(index_path)
    relpath = os.path.relpath(h5_path, start=os.path.dirname(index_path))
    db.add_entry(
        {"obs:obs_id": bgmap_id, "dataset": bgmap_id},
        filename=relpath, replace=True)


######################################################

@dataclass
class RunObsResult:
    """
    Container for results of obs process to get list of bgmap ids
    """
    obs_id: str
    success: bool = False
    traceback: str = ""
    bgmap_ids: Optional[list[str]] = None


def run_obs_process(cfg: BgmapDelayCfg, obs_id: str) -> RunObsResult:
    """
    Process for each observation to find bgmap ids.
    This will be processed in serial.
    """
    logger.debug("Computing Result set for %s", obs_id)

    # Need to reset logger here because this may be created new for spawned process
    log_level = cfg.process.log_level.upper()
    logger.setLevel(getattr(logging, log_level))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, log_level))

    res = RunObsResult(obs_id)
    try:
        ctx = core.Context(
            cfg.data.context_path,
            metadata_list=cfg.data.metadata_list)

        bgmap_dict = load_book.get_cal_obsids(ctx, obs_id, "bgmap")
        bgmap_ids = []
        for dset, oid in bgmap_dict.items():
            if oid is None:
                logger.debug("missing bgmap data for %s", dset)
                continue
            bgmap_ids.append(oid)
        res.bgmap_ids = bgmap_ids
        res.success = True
    except Exception as e:
        res.traceback = traceback.format_exc()
        if cfg.process.raise_exceptions:
            raise e
    return res


def handle_obs_result(result: RunObsResult, cfg: BgmapDelayCfg) -> None:
    """
    Update database for observations
    This does not check hdf5 file and dataset exists.

    result: RunObsResult
        Result from run_obs_process for single obs
    cfg.output.obs_index_path: str
        Path to output database
    """
    obs_id = str(result.obs_id)
    if not result.success:
        logger.error(f"Failed on obs_id: {obs_id}")
        logger.error(result.traceback)

        msg = result.traceback
        if msg is None:
            msg = "unknown error"
        cfg.output.add_to_failed_cache(obs_id, msg, 'obs')
        return

    logger.info(f"Adding obs_id {obs_id} to dataset")

    index_path = cfg.output.obs_index_path
    db = core.metadata.ManifestDb(index_path)
    for bgmap_id in result.bgmap_ids:
        h5_path = cfg.output.get_h5_path(bgmap_id)
        relpath = os.path.relpath(
            h5_path, start=os.path.dirname(index_path))
        db.add_entry(
            {"obs:obs_id": obs_id, "dataset": bgmap_id},
            filename=relpath, replace=True)


######################################################

def run_update_bgmap(
        cfg: BgmapDelayCfg,
        executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
        as_completed_callable: Callable) -> None:
    """
    Main run script for computing bgmap delay results.
    This will loop over bgmaps serially, and compute the delay
    in parallel among readout channels. The results will be stored
    into hdf5 file and bgmap database.

    Args:
    ------
    cfg.process.num_bgmap: int or None
        Max number of bgmaps to process per call.
        If None, will run on all available bgmaps.
    cfg.process.nprocs_channel: int
        Number of parallel processes
    """
    log_level = cfg.process.log_level.upper()
    logger.setLevel(getattr(logging, log_level))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, log_level))

    bgmap_ids = cfg.get_obsids_to_run('bgmap')
    logger.info(f"Processing {len(bgmap_ids)} bgmaps...")

    for oid in tqdm(bgmap_ids, disable=(not cfg.process.show_pb)):
        res = run_bgmap_process(cfg, oid, executor, as_completed_callable)
        handle_bgmap_result(res, cfg)

    return


def run_update_obs(cfg: BgmapDelayCfg) -> None:
    """
    Update another database for mapping from observation to bgmap.
    This is processed in serial.

    Args:
    ------
    cfg.process.num_obs: int or None
        Max number of observations to process per call.
        If None, will run on all available observations.
    """
    log_level = cfg.process.log_level.upper()
    logger.setLevel(getattr(logging, log_level))
    for ch in logger.handlers:
        ch.setLevel(getattr(logging, log_level))

    obs_ids = cfg.get_obsids_to_run('obs')

    logger.info(f"Processing {len(obs_ids)} obs...")
    for oid in tqdm(obs_ids, disable=(not cfg.process.show_pb)):
        res = run_obs_process(cfg, oid)
        handle_obs_result(res, cfg)

    return


def _main(
    cfg: BgmapDelayCfg,
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable) -> None:
    """
    Run update function. This is done in two steps:
    First, we process bgmap data for each ufm, and make its db.
    We will process it in parallel over channels.

    Next, we make another db to map observation and the bgmap.
    We will process this in serial since it should be quick.
    """

    run_update_bgmap(cfg, executor, as_completed_callable)
    run_update_obs(cfg)
    return


def main(config_file: str):
    """
    main function

    config_file: str
        yaml file for BgmapDelayCfg
    """
    cfg = BgmapDelayCfg.from_yaml(config_file)

    # We will process observations serially in single process
    # and process channels in the observation in parallel
    rank, executor, as_completed_callable = get_exec_env(
        cfg.process.nprocs_channel)
    if rank == 0:
        _main(cfg, executor, as_completed_callable)


def get_parser(
        parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """
    Construct argument parser
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str,
        help="yaml file with configuration for update script.")
    return parser

if __name__ == "__main__":
    main_launcher(main, get_parser)
