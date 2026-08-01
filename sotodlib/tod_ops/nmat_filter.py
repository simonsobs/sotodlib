"""Joint demodulated Q/U Fourier Nmat filtering."""

import logging

import numpy as np
from pixell import fft

logger = logging.getLogger(__name__)

_TW_BETA1_QUANTILES = {
    0.95: 0.98,
    0.99: 2.02,
    0.999: 3.27,
    "5sigma": 6.91,
}


def get_marchenko_pastur_threshold(n_samples, n_features,
                                   significance=0.999):
    """Return the MP upper edge with a Tracy--Widom finite-size cut.

    Parameters
    ----------
    n_samples : int or float
        Number of independent real samples used for the covariance.
    n_features : int or float
        Number of detector/component channels in the covariance.
    significance : {0.95, 0.99, 0.999, "5sigma"}
        One-sided Tracy--Widom beta=1 selection threshold.
    """
    if significance not in _TW_BETA1_QUANTILES:
        choices = list(_TW_BETA1_QUANTILES)
        raise ValueError(
            f"Unsupported MP significance {significance!r}; choose from {choices}"
        )
    if n_samples <= 0 or n_features <= 0:
        raise ValueError("n_samples and n_features must both be positive")

    gamma = float(n_features) / float(n_samples)
    sqrt_gamma = np.sqrt(gamma)
    lambda_plus = (1 + sqrt_gamma) ** 2
    delta = (
        float(n_samples) ** (-2 / 3)
        * (1 + sqrt_gamma) ** (4 / 3)
        * gamma ** (-1 / 6)
    )
    return lambda_plus + _TW_BETA1_QUANTILES[significance] * delta


def _build_freq_bins(freqs, fmin=None, fmax=None, bin_width_hz=None):
    """Build half-open Fourier-index bins, always excluding the DC mode."""
    if freqs.ndim != 1 or freqs.size < 3:
        raise ValueError("At least three rFFT frequencies are required")

    nyquist = freqs[-1]
    if fmin is None:
        fmin = freqs[1]
    if fmax is None:
        fmax = nyquist
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("fmin and fmax must be finite")
    if fmin < 0 or fmax <= fmin or fmax > nyquist:
        raise ValueError(
            f"Require 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz); "
            f"got ({fmin}, {fmax})"
        )
    if bin_width_hz is not None and (
            not np.isfinite(bin_width_hz) or bin_width_hz <= 0):
        raise ValueError("bin_width_hz must be positive")

    start = max(1, int(np.searchsorted(freqs, fmin, side="left")))
    stop = int(np.searchsorted(freqs, fmax, side="right"))
    if stop <= start:
        raise ValueError("The requested filter range contains no Fourier modes")
    if bin_width_hz is None:
        return [(start, stop)]

    bins = []
    i0 = start
    while i0 < stop:
        upper_hz = min(fmax, freqs[i0] + bin_width_hz)
        i1 = min(stop, int(np.searchsorted(freqs, upper_hz, side="left")))
        i1 = max(i0 + 1, i1)
        bins.append((i0, i1))
        i0 = i1
    return bins


def _build_profile_bins(freqs, fmin, fmax, n_bins=40, min_nfreq=20):
    """Build approximately logarithmic bins for frequency profiles."""
    if int(n_bins) < 1:
        raise ValueError("profile_n_bins must be positive")
    if int(min_nfreq) < 1:
        raise ValueError("profile_min_nfreq must be positive")

    start, stop = _build_freq_bins(
        freqs, fmin=fmin, fmax=fmax, bin_width_hz=None
    )[0]
    edges = np.rint(
        np.geomspace(start, stop, int(n_bins) + 1)
    ).astype(int)
    edges[0], edges[-1] = start, stop
    edges = np.unique(edges)

    merged = [start]
    for edge in edges[1:-1]:
        if edge - merged[-1] >= int(min_nfreq):
            merged.append(int(edge))
    merged.append(stop)
    if len(merged) > 2 and merged[-1] - merged[-2] < int(min_nfreq):
        del merged[-2]
    return list(zip(merged[:-1], merged[1:]))


def _resolve_signal(tod, signal, label):
    if isinstance(signal, str):
        arr = tod[signal]
    elif isinstance(signal, np.ndarray):
        arr = signal
    else:
        raise TypeError(f"{label} must be a field name or ndarray")
    expected = (tod.dets.count, tod.samps.count)
    if arr.shape != expected:
        raise ValueError(f"{label} has shape {arr.shape}; expected {expected}")
    if np.iscomplexobj(arr):
        raise ValueError(f"{label} must be real-valued")
    return arr


def apply_joint_qu_nmat_filter(
    tod,
    signal_Q="demodQ",
    signal_U="demodU",
    model_tod=None,
    fmin=0.0015,
    fmax=0.2,
    noise_band=(0.5, 1.75),
    bin_width_hz=0.2,
    mode_selection="mp_tw",
    mp_significance=0.999,
    n_modes=1,
    n_modes_max=None,
    singleness_max=None,
    operator="nmat",
    profile_n_bins=40,
    profile_min_nfreq=20,
    profile_diagonal_floor=0.05,
    psd_scale=1.0,
    psd_min_weight=0.0,
    in_place=False,
    wrap_Q=None,
    wrap_U=None,
    return_diagnostics=False,
    store_mode_diagnostics=False,
):
    """Apply a joint Q/U Fourier inverse-noise-covariance filter.

    Q and U are stacked into a single ``(2 * ndet, nfreq)`` matrix and
    whitened using ``noise_band``. Marchenko--Pastur plus Tracy--Widom
    thresholding selects statistically significant correlated spatial modes.

    With ``operator="nmat"``, the selected modes are combined and their
    frequency-dependent power E(f), together with the residual detector-
    diagonal power D(f), is measured in approximately logarithmic bins. The
    model ``N(f) = D(f) + V E(f) V.T`` is inverted with the Woodbury identity.
    Working in white-noise-normalized coordinates gives a dimensionless
    operator with approximately unit response to white noise. This matches the
    detector-vector noise operator used by the maximum-likelihood mapmaker,
    normalized for preprocessing.

    The legacy ``operator="mode_shrink"`` applies only correlated-mode
    shrinkage. For a selected whitened mode with power p its response is
    ``1 - psd_scale * (p - 1) / p``. In both modes, ``psd_scale`` blends from
    unchanged data (0) to the full operator (1). DC and frequencies outside
    ``[fmin, fmax]`` are unchanged.

    Parameters
    ----------
    tod : AxisManager
        Observation containing dets, samps, and timestamps.
    signal_Q, signal_U : str or ndarray
        Q and U fields (or arrays) with shape (ndet, nsamp).
    model_tod : AxisManager or None
        Optional real-data observation from which to estimate the Nmat. The
        resulting operator is applied to ``tod``. This is required for signal-
        only transfer simulations; when omitted, the model is estimated from
        ``tod`` itself. Field names in ``signal_Q`` and ``signal_U`` must exist
        in both AxisManagers when a separate model_tod is used.
    fmin, fmax : float
        Frequency range in Hz in which the filter is applied.
    noise_band : (float, float)
        Frequency range used to estimate independent channel power.
    bin_width_hz : float or None
        Mode-estimation bin width. None uses one bin over the full range.
    mode_selection : {"mp_tw", "fixed"}
        Select modes statistically or take the leading n_modes.
    mp_significance : {0.95, 0.99, 0.999, "5sigma"}
        Tracy--Widom significance for mode_selection="mp_tw".
    n_modes : int
        Number of modes used with mode_selection="fixed".
    n_modes_max : int or None
        Optional cap on selected modes per mode-estimation bin.
    singleness_max : float or None
        Reject modes whose largest absolute channel coupling exceeds this.
    operator : {"nmat", "mode_shrink"}
        Full frequency-dependent inverse covariance, or correlated-mode-only
        shrinkage.
    profile_n_bins : int
        Target number of logarithmic D(f), E(f) bins for operator="nmat".
    profile_min_nfreq : int
        Minimum number of complex Fourier samples in a profile bin.
    profile_diagonal_floor : float
        Lower bound on D(f), in white-noise-normalized power units.
    psd_scale : float
        Blend from unchanged data (0) to full inverse covariance (1).
    psd_min_weight : float
        Minimum subtraction weight for operator="mode_shrink".
    in_place : bool
        Overwrite input Q/U arrays when true.
    wrap_Q, wrap_U : str or None
        Optionally wrap filtered arrays into these TOD fields.
    return_diagnostics : bool
        Return mode and frequency-profile diagnostics.
    store_mode_diagnostics : bool
        Also retain per-bin eigenvalues/eigenvectors in diagnostics.

    Returns
    -------
    (filtered_Q, filtered_U), diagnostics_or_None
    """
    if mode_selection not in ("mp_tw", "fixed"):
        raise ValueError("mode_selection must be 'mp_tw' or 'fixed'")
    if operator not in ("nmat", "mode_shrink"):
        raise ValueError("operator must be 'nmat' or 'mode_shrink'")
    if not np.isfinite(psd_scale) or not 0 <= psd_scale <= 1:
        raise ValueError("psd_scale must be finite and in [0, 1]")
    if not 0 <= psd_min_weight <= 1:
        raise ValueError("psd_min_weight must be in [0, 1]")
    if n_modes_max is not None and int(n_modes_max) < 0:
        raise ValueError("n_modes_max must be non-negative")
    if mode_selection == "fixed" and int(n_modes) < 0:
        raise ValueError("n_modes must be non-negative")
    if singleness_max is not None and not 0 < singleness_max <= 1:
        raise ValueError("singleness_max must be in (0, 1]")
    if (not np.isfinite(profile_diagonal_floor)
            or profile_diagonal_floor <= 0):
        raise ValueError("profile_diagonal_floor must be positive")

    q = _resolve_signal(tod, signal_Q, "signal_Q")
    u = _resolve_signal(tod, signal_U, "signal_U")
    if model_tod is None:
        model_tod = tod
    if model_tod is not tod and (
            not isinstance(signal_Q, str) or not isinstance(signal_U, str)):
        raise ValueError(
            "signal_Q and signal_U must be field names with a separate model_tod"
        )
    model_q = _resolve_signal(model_tod, signal_Q, "model signal_Q")
    model_u = _resolve_signal(model_tod, signal_U, "model signal_U")
    if model_tod is not tod:
        if not np.array_equal(model_tod.dets.vals, tod.dets.vals):
            raise ValueError("model_tod and tod detector axes must match in order")
        if not np.array_equal(model_tod.timestamps, tod.timestamps):
            raise ValueError("model_tod and tod timestamps must match")
    if tod.samps.count < 4:
        raise ValueError("At least four samples are required")
    if "timestamps" not in tod._fields:
        raise ValueError("timestamps are required to define Fourier frequencies")
    dts = np.diff(tod.timestamps)
    if not np.all(np.isfinite(dts)) or not np.all(dts > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    dt = float(np.median(dts))

    ndet, nsamp = q.shape
    nfreq = nsamp // 2 + 1
    freqs = np.fft.rfftfreq(nsamp, dt)
    bins = _build_freq_bins(
        freqs, fmin=fmin, fmax=fmax, bin_width_hz=bin_width_hz
    )
    if operator == "nmat":
        profile_bins = _build_profile_bins(
            freqs,
            fmin=fmin,
            fmax=fmax,
            n_bins=profile_n_bins,
            min_nfreq=profile_min_nfreq,
        )
    else:
        profile_bins = []

    try:
        n0, n1 = noise_band
    except (TypeError, ValueError):
        raise ValueError("noise_band must contain exactly two frequencies")
    if not np.isfinite(n0) or not np.isfinite(n1) or n0 < 0 or n1 <= n0:
        raise ValueError("noise_band must satisfy 0 <= low < high")
    j0 = int(np.searchsorted(freqs, n0, side="left"))
    j1 = int(np.searchsorted(freqs, n1, side="right"))
    if j1 <= j0:
        raise ValueError(f"noise_band {noise_band} contains no Fourier modes")

    real_dtype = np.result_type(q.dtype, u.dtype)
    complex_dtype = np.complex64 if real_dtype.itemsize <= 4 else np.complex128
    ft = np.empty((2 * ndet, nfreq), dtype=complex_dtype)
    fft.rfft(q, ft[:ndet])
    fft.rfft(u, ft[ndet:])

    if model_tod is tod:
        model_ft = ft
    else:
        model_real_dtype = np.result_type(model_q.dtype, model_u.dtype)
        model_complex_dtype = (
            np.complex64 if model_real_dtype.itemsize <= 4 else np.complex128
        )
        model_ft = np.empty((2 * ndet, nfreq), dtype=model_complex_dtype)
        fft.rfft(model_q, model_ft[:ndet])
        fft.rfft(model_u, model_ft[ndet:])

    noise_power = np.mean(np.abs(model_ft[:, j0:j1]) ** 2, axis=1)
    noise_sigma = np.sqrt(noise_power)
    valid = np.isfinite(noise_sigma) & (noise_sigma > 0)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise ValueError("No Q/U channels have finite, non-zero noise-band power")
    inv_sigma = 1 / noise_sigma[valid]

    diag = {
        "bins": np.asarray(bins, dtype=np.int32),
        "freqs": freqs,
        "bin_centers_hz": np.asarray([
            np.mean(freqs[i0:i1]) for i0, i1 in bins
        ]),
        "operator": operator,
        "mode_selection": mode_selection,
        "selected_modes": [],
        "thresholds": [],
        "gammas": [],
        "mean_mode_weights": [],
        "valid_feature_indices": valid_idx,
        "noise_sigma": noise_sigma.reshape(2, ndet),
    }
    if operator == "nmat":
        diag.update({
            "profile_bins": np.asarray(profile_bins, dtype=np.int32),
            "profile_bin_centers_hz": np.asarray([
                np.mean(freqs[i0:i1]) for i0, i1 in profile_bins
            ]),
            "profile_mean_residual_power": [],
            "profile_mean_mode_power": [],
            "profile_mean_diagonal_response": [],
            "profile_power_response": [],
        })
    if store_mode_diagnostics:
        diag["evals_by_bin"] = []
        diag["evecs_by_bin"] = []
        diag["selected_indices_by_bin"] = []

    mode_vecs = []
    for i0, i1 in bins:
        nb = i1 - i0
        n_samples = 2 * nb
        if nb < 2:
            diag["selected_modes"].append(0)
            diag["thresholds"].append(np.nan)
            diag["gammas"].append(np.nan)
            diag["mean_mode_weights"].append(np.nan)
            if store_mode_diagnostics:
                diag["evals_by_bin"].append(np.empty(0))
                diag["evecs_by_bin"].append(
                    np.empty((valid_idx.size, 0), dtype=float)
                )
                diag["selected_indices_by_bin"].append(
                    np.empty(0, dtype=int)
                )
            continue

        # Work on only this narrow bin, avoiding another observation-sized FFT.
        model_xbin = np.array(model_ft[valid_idx, i0:i1], copy=True)
        model_xbin *= inv_sigma[:, None]
        cov = np.real(model_xbin.dot(np.conj(model_xbin.T))) / nb
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        keep = np.ones(evals.size, dtype=bool)
        if singleness_max is not None:
            keep &= np.max(np.abs(evecs), axis=0) < singleness_max

        gamma = valid_idx.size / n_samples
        if mode_selection == "mp_tw":
            threshold = get_marchenko_pastur_threshold(
                n_samples, valid_idx.size, significance=mp_significance
            )
            selected = np.flatnonzero((evals > threshold) & keep)
        else:
            threshold = np.nan
            selected = np.flatnonzero(keep)[:int(n_modes)]
        if n_modes_max is not None:
            selected = selected[:int(n_modes_max)]

        diag["selected_modes"].append(int(selected.size))
        diag["thresholds"].append(float(threshold))
        diag["gammas"].append(float(gamma))
        if store_mode_diagnostics:
            diag["evals_by_bin"].append(np.array(evals, copy=True))
            diag["evecs_by_bin"].append(np.array(evecs, copy=True))
            diag["selected_indices_by_bin"].append(
                np.array(selected, copy=True)
            )

        if selected.size:
            vecs = evecs[:, selected]
            model_amps = vecs.T.dot(model_xbin)
            mode_power = np.mean(np.abs(model_amps) ** 2, axis=1)
            excess = np.maximum(0, mode_power - 1)
            weights = excess / np.maximum(mode_power, np.finfo(float).tiny)
            weights = np.clip(psd_scale * weights, psd_min_weight, 1)
            diag["mean_mode_weights"].append(float(np.mean(weights)))
            mode_vecs.append(vecs)
        else:
            diag["mean_mode_weights"].append(np.nan)

        if operator == "mode_shrink" and selected.size:
            target_xbin = np.array(ft[valid_idx, i0:i1], copy=True)
            target_xbin *= inv_sigma[:, None]
            target_amps = vecs.T.dot(target_xbin)
            target_xbin -= vecs.dot(weights[:, None] * target_amps)
            target_xbin *= noise_sigma[valid, None]
            ft[valid_idx, i0:i1] = target_xbin

    if operator == "nmat":
        if mode_vecs:
            vecs = np.concatenate(mode_vecs, axis=1)
            pinv_vecs = np.linalg.pinv(vecs)
        else:
            vecs = np.empty((valid_idx.size, 0), dtype=float)
            pinv_vecs = np.empty((0, valid_idx.size), dtype=float)

        for i0, i1 in profile_bins:
            model_xbin = np.array(model_ft[valid_idx, i0:i1], copy=True)
            model_xbin *= inv_sigma[:, None]

            if vecs.shape[1]:
                model_amps = pinv_vecs.dot(model_xbin)
                residual = model_xbin - vecs.dot(model_amps)
                mode_power = np.mean(np.abs(model_amps) ** 2, axis=1)
            else:
                residual = model_xbin
                mode_power = np.empty(0)

            residual_power = np.mean(np.abs(residual) ** 2, axis=1)
            residual_power = np.where(
                np.isfinite(residual_power),
                residual_power,
                profile_diagonal_floor,
            )
            residual_power = np.maximum(
                residual_power, profile_diagonal_floor
            )
            inv_diagonal = 1 / residual_power
            target_xbin = np.array(ft[valid_idx, i0:i1], copy=True)
            target_xbin *= inv_sigma[:, None]
            filtered = inv_diagonal[:, None] * target_xbin

            if vecs.shape[1]:
                physical_modes = vecs * np.sqrt(
                    np.maximum(mode_power, np.finfo(float).tiny)
                )[None, :]
                invd_modes = inv_diagonal[:, None] * physical_modes
                core = (
                    np.eye(physical_modes.shape[1])
                    + physical_modes.T.dot(invd_modes)
                )
                rhs = invd_modes.T.dot(target_xbin)
                filtered -= invd_modes.dot(np.linalg.solve(core, rhs))

            filtered = target_xbin + psd_scale * (filtered - target_xbin)
            input_power = np.sum(np.abs(target_xbin) ** 2)
            output_power = np.sum(np.abs(filtered) ** 2)
            power_response = (
                output_power / input_power if input_power > 0 else np.nan
            )

            diag["profile_mean_residual_power"].append(
                float(np.mean(residual_power))
            )
            diag["profile_mean_mode_power"].append(
                float(np.mean(mode_power)) if mode_power.size else np.nan
            )
            diag["profile_mean_diagonal_response"].append(
                float(np.mean(inv_diagonal))
            )
            diag["profile_power_response"].append(float(power_response))

            filtered *= noise_sigma[valid, None]
            ft[valid_idx, i0:i1] = filtered

    if in_place:
        q_out, u_out = q, u
    else:
        q_out = np.empty_like(q)
        u_out = np.empty_like(u)
    fft.irfft(ft[:ndet], q_out, normalize="phys")
    fft.irfft(ft[ndet:], u_out, normalize="phys")

    for wrap, out in ((wrap_Q, q_out), (wrap_U, u_out)):
        if wrap is None:
            continue
        if wrap in tod._fields:
            tod.move(wrap, None)
        tod.wrap(wrap, out, [(0, "dets"), (1, "samps")])

    for key in ("selected_modes", "thresholds", "gammas",
                "mean_mode_weights", "profile_mean_residual_power",
                "profile_mean_mode_power", "profile_mean_diagonal_response",
                "profile_power_response"):
        if key in diag:
            diag[key] = np.asarray(diag[key])
    result = (q_out, u_out)
    if return_diagnostics:
        return result, diag
    return result, None
