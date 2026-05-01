"""
Computation functions for stimulator calibration.
Measures per-detector gain (stm_gain) and time constant (stm_timeconstant)
using a rotating chopped thermal source.
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from scipy.optimize import curve_fit

from sotodlib import tod_ops
from sotodlib.io import hkdb

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    best_values: dict
    best_fit: np.ndarray
    success: bool = True


@dataclass
class stm_config:
    hkdb_cfg: dict
    # Gain obs parameters
    chopping_freq_gain: float = 6.0        # [Hz] chopping frequency during gain obs
    t_cut_gain: list = field(default_factory=lambda: [10, 70])  # [s] time window
    # Time-constant obs parameters
    chopping_freqs: dict = field(default_factory=lambda: {
        'f1': 6, 'f2': 15, 'f3': 33, 'f4': 63,
        'f5': 93, 'f6': 123, 'f7': 147,
    })
    t_cuts: dict = field(default_factory=lambda: {
        f_key: [53 + 20 * i, 63 + 20 * i]
        for i, f_key in enumerate(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
    })
    # Common parameters
    n_bins: int = 40
    cutoff_factor: float = 1.5
    hpf_cutoff: float = 1.0   # [Hz]
    r_frac_min: float = 0.2
    r_frac_max: float = 0.8


_HK_FIELDS = [
    'stimulator-blh.motor.*',
    'stimulator-ds378.relay.*',
    'stimulator-enc.stim_enc.*',
    'stimulator-enc.stim_enc_downsampled.*',
    'stimulator-pcr500ma.heater_source.*',
    'stimulator-thermo.temperatures.*',
]


def load_hk_data(hkdb_cfg: dict, t_start: float, t_end: float):
    """Load stimulator house-keeping data for a given time range."""
    lspec = hkdb.LoadSpec(hkdb_cfg, fields=_HK_FIELDS, start=t_start, end=t_end)
    return hkdb.load_hk(lspec)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_enc_t0(hkdata) -> np.ndarray:
    """Return encoder timestamps (UTC) at chopper blade index-pulse events."""
    state = np.array(hkdata.data['stimulator-enc.stim_enc.state'][1])
    t_enc = np.array(
        hkdata.data['stimulator-enc.stim_enc.timestamps_tai'][1]
    )[state == 0] - 37   # TAI → UTC offset
    return t_enc


def _compute_frac_timing(timestamps: np.ndarray, t_enc: np.ndarray) -> np.ndarray:
    """Map each TOD sample to a fractional phase within one chopper cycle [0, 1)."""
    i_enc = 0
    frac = []
    for t in timestamps:
        if i_enc >= len(t_enc) - 2:
            frac.append(np.nan)
            continue
        while i_enc < len(t_enc) - 2 and t_enc[i_enc + 1] < t:
            i_enc += 1
        dt = t_enc[i_enc + 1] - t_enc[i_enc]
        frac.append((t - t_enc[i_enc]) / dt if dt > 0 else np.nan)
    return np.array(frac)


def _apply_gain_filters(aman, cfg: stm_config) -> None:
    """IIR correction + bidirectional HPF + bidirectional LPF for gain obs."""
    iirc = tod_ops.filters.iir_filter(aman, invert=True)
    sig = tod_ops.fourier_filter(aman, iirc, signal_name='signal')
    aman.wrap('signal_hpf', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)

    hpf = tod_ops.filters.high_pass_sine2(cfg.hpf_cutoff)
    for _ in range(2):
        sig = tod_ops.fourier_filter(aman, hpf, signal_name='signal_hpf')
        sig = np.fliplr(sig)
        aman.wrap('signal_hpf', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)

    fc = cfg.cutoff_factor * cfg.chopping_freq_gain
    lpf = tod_ops.filters.low_pass_sine2(fc, fc / 5)
    for _ in range(2):
        sig = tod_ops.fourier_filter(aman, lpf, signal_name='signal_hpf')
        sig = np.fliplr(sig)
        aman.wrap('signal_lpf', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)


def _apply_tc_filters(aman, cfg: stm_config) -> None:
    """IIR correction + bidirectional HPF + per-frequency bidirectional LPF for tc obs."""
    iirc = tod_ops.filters.iir_filter(aman, invert=True)
    sig = tod_ops.fourier_filter(aman, iirc, signal_name='signal')
    aman.wrap('signal_iirc', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)
    aman.wrap('signal_hpf', sig.copy(), [(0, 'dets'), (1, 'samps')], overwrite=True)

    hpf = tod_ops.filters.high_pass_sine2(cfg.hpf_cutoff)
    for _ in range(2):
        sig = tod_ops.fourier_filter(aman, hpf, signal_name='signal_hpf')
        sig = np.fliplr(sig)
        aman.wrap('signal_hpf', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)

    for f_key, f_hz in cfg.chopping_freqs.items():
        fc = cfg.cutoff_factor * f_hz
        lpf = tod_ops.filters.low_pass_sine2(fc, fc / 5)
        for _ in range(2):
            sig = tod_ops.fourier_filter(aman, lpf, signal_name='signal_hpf')
            sig = np.fliplr(sig)
            aman.wrap(f'signal_lpf_{f_key}', sig, [(0, 'dets'), (1, 'samps')], overwrite=True)


def _bin_mean(
    signal_row: np.ndarray,
    frac_timing: np.ndarray,
    timestamps: np.ndarray,
    t0: float,
    t_cut: list,
    n_bins: int,
):
    """Phase-bin one detector's signal and return (bin centres, bin means)."""
    x = np.linspace(0, 1 - 1 / n_bins, n_bins) + 0.5 / n_bins
    y = np.full(n_bins, np.nan)
    cut_time = (t_cut[0] <= timestamps - t0) & (timestamps - t0 < t_cut[1])
    for i_bin in range(n_bins):
        cut_phase = (i_bin / n_bins <= frac_timing) & (frac_timing < (i_bin + 1) / n_bins)
        d = signal_row[cut_phase & cut_time]
        if len(d) > 0:
            y[i_bin] = np.mean(d)
    return x, y


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def _func_sines(t, a0, a1, a2, a3, a4, a5, a6, t0, t1, t2, t3, t4, t5, t6):
    return (a0 * np.sin(1 * (t - t0) * 2 * np.pi)
            + a1 * np.sin(2 * (t - t1) * 2 * np.pi)
            + a2 * np.sin(3 * (t - t2) * 2 * np.pi)
            + a3 * np.sin(4 * (t - t3) * 2 * np.pi)
            + a4 * np.sin(5 * (t - t4) * 2 * np.pi)
            + a5 * np.sin(6 * (t - t5) * 2 * np.pi)
            + a6 * np.sin(7 * (t - t6) * 2 * np.pi))


_SINES_P0 = [1e-3, 2e-4, 2e-4, 1e-4, 1e-4, 1e-5, 1e-5,
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
_SINES_BOUNDS = ([-np.inf] * 7 + [0.0] * 7, [np.inf] * 7 + [1.0] * 7)
_SINES_NAMES = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                't0', 't1', 't2', 't3', 't4', 't5', 't6']


def _fit_sines(x: np.ndarray, y: np.ndarray) -> FitResult:
    mask = np.isfinite(y)
    if mask.sum() < len(_SINES_P0):
        return FitResult({n: np.nan for n in _SINES_NAMES},
                         np.full_like(x, np.nan), success=False)
    try:
        popt, _ = curve_fit(_func_sines, x[mask], y[mask],
                            p0=_SINES_P0, bounds=_SINES_BOUNDS, maxfev=10000)
        return FitResult(dict(zip(_SINES_NAMES, popt)), _func_sines(x, *popt))
    except RuntimeError:
        return FitResult({n: np.nan for n in _SINES_NAMES},
                         np.full_like(x, np.nan), success=False)


def _func_response_amplitude(f, tau, a):
    return a / np.sqrt(1 + (2 * np.pi * f * tau) ** 2)


def _func_response_phase_with_dt(f, tau, theta_geo, dt):
    theta_dt = -dt * 2 * np.pi * f * (180 / np.pi)
    return np.arctan(-2 * np.pi * f * tau) * (180 / np.pi) + theta_geo + theta_dt


def _fit_response_amplitude(f_arr: np.ndarray, a0s: np.ndarray) -> FitResult:
    try:
        popt, _ = curve_fit(_func_response_amplitude, f_arr, a0s,
                            p0=[1e-3, 1e-3], bounds=([0, 0], [1, 1]), maxfev=5000)
        return FitResult({'tau': popt[0], 'a': popt[1]},
                         _func_response_amplitude(f_arr, *popt))
    except RuntimeError:
        return FitResult({'tau': np.nan, 'a': np.nan},
                         np.full_like(f_arr, np.nan), success=False)


def _fit_phase_no_dt(f_arr: np.ndarray, phase_y: np.ndarray) -> FitResult:
    """Phase fit with dt fixed at 0."""
    def func(f, theta_geo, tau):
        return _func_response_phase_with_dt(f, tau, theta_geo, 0.0)
    try:
        popt, _ = curve_fit(func, f_arr, phase_y, p0=[0.0, 1e-3],
                            bounds=([-90, 0], [90, 0.1]), maxfev=5000)
        return FitResult({'theta_geo': popt[0], 'tau': popt[1], 'dt': 0.0},
                         func(f_arr, *popt))
    except RuntimeError:
        return FitResult({'theta_geo': np.nan, 'tau': np.nan, 'dt': np.nan},
                         np.full_like(f_arr, np.nan), success=False)


def _fit_phase_fix_tau(f_arr: np.ndarray, phase_y: np.ndarray,
                       tau_fixed: float) -> FitResult:
    """Phase fit with tau fixed to the amplitude-fit value."""
    def func(f, theta_geo, dt):
        return _func_response_phase_with_dt(f, tau_fixed, theta_geo, dt)
    try:
        popt, _ = curve_fit(func, f_arr, phase_y, p0=[0.0, 0.125e-3],
                            bounds=([-90, -3e-3], [90, 3e-3]), maxfev=5000)
        return FitResult({'theta_geo': popt[0], 'tau': tau_fixed, 'dt': popt[1]},
                         func(f_arr, *popt))
    except RuntimeError:
        return FitResult({'theta_geo': np.nan, 'tau': tau_fixed, 'dt': np.nan},
                         np.full_like(f_arr, np.nan), success=False)


def _fit_phase_free(f_arr: np.ndarray, phase_y: np.ndarray) -> FitResult:
    """Phase fit with tau and dt both free."""
    def func(f, theta_geo, tau, dt):
        return _func_response_phase_with_dt(f, tau, theta_geo, dt)
    try:
        popt, _ = curve_fit(func, f_arr, phase_y, p0=[0.0, 1e-3, 0.125e-3],
                            bounds=([-90, 0, -3e-3], [90, 0.1, 3e-3]), maxfev=5000)
        return FitResult({'theta_geo': popt[0], 'tau': popt[1], 'dt': popt[2]},
                         func(f_arr, *popt))
    except RuntimeError:
        return FitResult({'theta_geo': np.nan, 'tau': np.nan, 'dt': np.nan},
                         np.full_like(f_arr, np.nan), success=False)


# ---------------------------------------------------------------------------
# Public calibration functions
# ---------------------------------------------------------------------------

def calc_gain(aman, hkdata, cfg: stm_config) -> None:
    """Compute per-detector gain from a fixed-frequency stimulator observation.

    Phase-bins the HPF- and (HPF+LPF)-filtered signal, fits the fundamental
    sine amplitude, and wraps the results back onto *aman*:

    - ``stm_gain`` (float[n_dets])  -- |a0| of the LPF fit
    - ``stm_gain_all`` (object[n_dets]) -- dict with 'HPF' and 'LPF' FitResult
    """
    t0 = aman.timestamps[0]
    t_enc = _get_enc_t0(hkdata)
    frac_timing = _compute_frac_timing(aman.timestamps, t_enc)
    aman.wrap('frac_timing', frac_timing, [(0, 'samps')], overwrite=True)

    _apply_gain_filters(aman, cfg)

    n_dets = aman.signal.shape[0]
    stm_gain = np.full(n_dets, np.nan)
    stm_gain_all = np.full(n_dets, None, dtype=object)

    for i_det in range(n_dets):
        x, y_hpf = _bin_mean(aman.signal_hpf[i_det], frac_timing,
                              aman.timestamps, t0, cfg.t_cut_gain, cfg.n_bins)
        x, y_lpf = _bin_mean(aman.signal_lpf[i_det], frac_timing,
                              aman.timestamps, t0, cfg.t_cut_gain, cfg.n_bins)
        fit_hpf = _fit_sines(x, y_hpf)
        fit_lpf = _fit_sines(x, y_lpf)
        stm_gain[i_det] = abs(fit_lpf.best_values.get('a0', np.nan))
        stm_gain_all[i_det] = {'HPF': fit_hpf, 'LPF': fit_lpf}

    aman.wrap('stm_gain', stm_gain, [(0, 'dets')], overwrite=True)
    aman.wrap('stm_gain_all', stm_gain_all, [(0, 'dets')], overwrite=True)


def calc_timeconstant(aman, hkdata, cfg: stm_config) -> None:
    """Compute per-detector time constant from a multi-frequency stimulator obs.

    For each chopping frequency, phase-bins the LPF-filtered signal and fits
    the fundamental sine. Then fits amplitude and phase vs frequency to extract
    the time constant τ. Wraps onto *aman*:

    - ``stm_timeconstant`` (float[n_dets])  -- τ from amplitude fit [s]
    - ``stm_timeconstant_all`` (object[n_dets]) -- dict with amplitude and
      phase FitResults (``phase__no_dt``, ``phase__fix_tau``, ``phase__free``)
    """
    t0 = aman.timestamps[0]
    t_enc = _get_enc_t0(hkdata)
    frac_timing = _compute_frac_timing(aman.timestamps, t_enc)
    aman.wrap('frac_timing', frac_timing, [(0, 'samps')], overwrite=True)

    _apply_tc_filters(aman, cfg)

    f_keys = list(cfg.chopping_freqs.keys())
    f_arr = np.array([cfg.chopping_freqs[k] for k in f_keys], dtype=float)

    n_dets = aman.signal.shape[0]
    stm_timeconstant = np.full(n_dets, np.nan)
    stm_timeconstant_all = np.full(n_dets, None, dtype=object)

    for i_det in range(n_dets):
        a0s, t0s = [], []
        for f_key in f_keys:
            x, y = _bin_mean(aman[f'signal_lpf_{f_key}'][i_det], frac_timing,
                             aman.timestamps, t0, cfg.t_cuts[f_key], cfg.n_bins)
            fit = _fit_sines(x, y)
            a0s.append(fit.best_values.get('a0', np.nan))
            t0s.append(fit.best_values.get('t0', np.nan))

        # Unwrap phase: handle sign flip and cycle ambiguity
        for i in range(len(t0s)):
            if np.isfinite(a0s[i]) and a0s[i] < 0 and np.isfinite(t0s[i]):
                t0s[i] -= 0.5
            if i > 0 and np.isfinite(t0s[i]) and np.isfinite(t0s[i - 1]):
                if t0s[i] < t0s[i - 1]:
                    t0s[i] += 1.0
        a0s = np.abs(a0s)

        amp_fit = _fit_response_amplitude(f_arr, np.array(a0s))
        phase_y = -np.array(t0s) * 360
        tau_amp = amp_fit.best_values.get('tau', np.nan)

        stm_timeconstant[i_det] = tau_amp
        stm_timeconstant_all[i_det] = {
            'amplitude': amp_fit,
            'phase__no_dt': _fit_phase_no_dt(f_arr, phase_y),
            'phase__fix_tau': _fit_phase_fix_tau(
                f_arr, phase_y,
                tau_amp if np.isfinite(tau_amp) else 1e-3,
            ),
            'phase__free': _fit_phase_free(f_arr, phase_y),
        }

    aman.wrap('stm_timeconstant', stm_timeconstant, [(0, 'dets')], overwrite=True)
    aman.wrap('stm_timeconstant_all', stm_timeconstant_all, [(0, 'dets')], overwrite=True)
