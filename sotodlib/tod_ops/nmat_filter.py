"""Joint demodulated Q/U Fourier Nmat filtering.

The operator is fit and applied in two separate steps. ``fit_joint_qu_nmat_operator``
derives everything from a model TOD (whitening scale, correlated modes, and the
frequency profiles D(f) and E(f)) and returns it as an AxisManager, which can be
stored in a preprocess archive. ``apply_joint_qu_nmat_operator`` then applies a
stored operator to a target TOD. This allows the operator to be fit once on real
data and reapplied to signal-only simulations without redoing the real-data
analysis.
"""

import logging

import numpy as np
from pixell import fft

from .. import core

logger = logging.getLogger(__name__)

_TW_BETA1_QUANTILES = {
    0.95: 0.98,
    0.99: 2.02,
    0.999: 3.27,
    "5sigma": 6.91,
}


def get_marchenko_pastur_threshold(n_samples, n_features,
                                   significance=0.999, return_edge=False):
    """Return the MP upper edge with a Tracy--Widom finite-size cut.

    Parameters
    ----------
    n_samples : int or float
        Number of independent real samples used for the covariance.
    n_features : int or float
        Number of detector/component channels in the covariance.
    significance : {0.95, 0.99, 0.999, "5sigma"}
        One-sided Tracy--Widom beta=1 selection threshold.
    return_edge : bool
        If True, also return the asymptotic MP edge lambda_plus, which is
        useful to separate modes rejected by the finite-size term from those
        below the bulk edge itself.

    Returns
    -------
    threshold : float
        lambda_plus plus the Tracy--Widom correction.
    lambda_plus : float
        Only returned when ``return_edge`` is True.
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
    threshold = lambda_plus + _TW_BETA1_QUANTILES[significance] * delta
    if return_edge:
        return threshold, lambda_plus
    return threshold


def _build_freq_bins(freqs, fmin=None, fmax=None, bin_width_hz=None):
    """Build half-open mode-estimation bins, always excluding the DC mode."""
    nyquist = freqs[-1]
    if fmin is None:
        fmin = freqs[1]
    if fmax is None:
        fmax = nyquist
    if fmin < 0 or fmax <= fmin or fmax > nyquist:
        raise ValueError(
            f"Require 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz); "
            f"got ({fmin}, {fmax})"
        )

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
        # Absorb a short trailing remainder rather than emitting a bin too
        # narrow to estimate a covariance from.
        if stop - i1 < 2:
            i1 = stop
        i1 = max(i0 + 1, i1)
        bins.append((i0, i1))
        i0 = i1
    return bins


def _build_profile_bins(freqs, fmin, fmax, n_bins=40, min_nfreq=20):
    """Build approximately logarithmic bins for the D(f), E(f) profiles."""
    # Deferred: sotodlib.mapmaking imports tod_ops, so importing it at module
    # scope would be circular.
    from ..mapmaking.utils import makebins

    start, stop = _build_freq_bins(freqs, fmin=fmin, fmax=fmax)[0]
    # makebins drops the final edge when the whole range is shorter than nmin,
    # which would leave no bins at all, so handle that case directly.
    if stop - start < 2 * int(min_nfreq):
        return [(start, stop)]
    srate = 2 * freqs[-1]
    edges = np.geomspace(freqs[start], freqs[stop - 1], int(n_bins) + 1)
    # rfun=floor: the default (ceil) rounds the lowest edge up and so drops the
    # first Fourier mode, which carries a large share of the 1/f power.
    bins = makebins(edges, srate, freqs.size, nmin=int(min_nfreq), cap=False,
                    rfun=np.floor)
    bins = [[int(lo), int(hi)] for lo, hi in bins]
    # Guarantee the bins tile [start, stop) exactly, independent of rounding.
    bins[0][0], bins[-1][1] = start, stop
    return [(lo, hi) for lo, hi in bins if hi > lo]


def _resolve_signal(tod, signal):
    return tod[signal] if isinstance(signal, str) else signal


def _stack_rfft(q, u):
    """rFFT of the Q/U stack, shape (2 ndet, nfreq)."""
    ndet, nsamp = q.shape
    nfreq = nsamp // 2 + 1
    real_dtype = np.result_type(q.dtype, u.dtype)
    complex_dtype = np.complex64 if real_dtype.itemsize <= 4 else np.complex128
    ft = np.empty((2 * ndet, nfreq), dtype=complex_dtype)
    fft.rfft(q, ft[:ndet])
    fft.rfft(u, ft[ndet:])
    return ft


def _sample_dt(tod):
    return float(np.median(np.diff(tod.timestamps)))


def fit_joint_qu_nmat_operator(
    tod,
    signal_Q="demodQ",
    signal_U="demodU",
    fmin=None,
    fmax=None,
    noise_band=(0.5, 1.75),
    bin_width_hz=None,
    mode_selection="mp_tw",
    mp_significance=0.999,
    n_modes=1,
    n_modes_max=None,
    singleness_max=None,
    profile_n_bins=40,
    profile_min_nfreq=20,
    profile_diagonal_floor=0.05,
):
    """Fit a joint Q/U Fourier Nmat operator from a TOD.

    The demodulated Q and U detector streams are stacked into a single set of
    ``2 ndet`` channels and whitened by their power in ``noise_band``. Within
    each mode-estimation bin the whitened channel covariance is eigen-decomposed
    and thresholded against the Marchenko--Pastur bulk edge with a Tracy--Widom
    finite-size correction, so only modes statistically inconsistent with
    independent noise are retained.

    The retained modes V are then used to measure, in approximately logarithmic
    frequency bins, the mode power E(f) and the residual per-channel power
    D(f), giving the whitened noise model ``N(f) = D(f) + V E(f) V.T``.

    Parameters
    ----------
    tod : AxisManager
        Observation the operator is derived from. For transfer-function
        simulations this is the real-data observation.
    signal_Q, signal_U : str or ndarray
        Q and U fields (or arrays) with shape (ndet, nsamp).
    fmin, fmax : float
        Frequency range in Hz over which the operator is defined.
    noise_band : (float, float)
        Frequency range used to set the per-channel whitening scale.
    bin_width_hz : float or None
        Mode-estimation bin width. None uses one bin over the full range.
    mode_selection : {"mp_tw", "fixed"}
        Select modes statistically or take the leading ``n_modes``.
    mp_significance : {0.95, 0.99, 0.999, "5sigma"}
        Tracy--Widom significance for mode_selection="mp_tw".
    n_modes : int
        Number of modes used with mode_selection="fixed".
    n_modes_max : int or None
        Optional cap on selected modes per mode-estimation bin.
    singleness_max : float or None
        Reject modes whose largest absolute channel coupling exceeds this, i.e.
        modes carried by essentially a single detector rather than shared.
    profile_n_bins : int
        Target number of logarithmic D(f), E(f) bins.
    profile_min_nfreq : int
        Minimum number of complex Fourier samples in a profile bin.
    profile_diagonal_floor : float
        Lower bound on D(f), in white-noise-normalized power units.

    Returns
    -------
    operator : AxisManager
        Everything needed to apply the operator, on the ``dets``,
        ``nmat_modes``, ``nmat_profile`` and ``nmat_bins`` axes. See
        :func:`apply_joint_qu_nmat_operator`.
    """
    if mode_selection not in ("mp_tw", "fixed"):
        raise ValueError("mode_selection must be 'mp_tw' or 'fixed'")
    if singleness_max is not None and not 0 < singleness_max <= 1:
        raise ValueError("singleness_max must be in (0, 1]")
    if profile_diagonal_floor <= 0:
        raise ValueError("profile_diagonal_floor must be positive")

    q = _resolve_signal(tod, signal_Q)
    u = _resolve_signal(tod, signal_U)
    ndet, nsamp = q.shape
    dt = _sample_dt(tod)
    freqs = np.fft.rfftfreq(nsamp, dt)

    bins = _build_freq_bins(freqs, fmin=fmin, fmax=fmax, bin_width_hz=bin_width_hz)
    profile_bins = _build_profile_bins(
        freqs, fmin=fmin, fmax=fmax,
        n_bins=profile_n_bins, min_nfreq=profile_min_nfreq,
    )

    n0, n1 = noise_band
    if n0 < 0 or n1 <= n0:
        raise ValueError("noise_band must satisfy 0 <= low < high")
    j0 = int(np.searchsorted(freqs, n0, side="left"))
    j1 = int(np.searchsorted(freqs, n1, side="right"))
    if j1 <= j0:
        raise ValueError(f"noise_band {noise_band} contains no Fourier modes")

    ft = _stack_rfft(q, u)

    # Whitening scale: sigma, not sigma**2, since it is applied to the field.
    noise_sigma = np.sqrt(np.mean(np.abs(ft[:, j0:j1]) ** 2, axis=1))
    valid = np.isfinite(noise_sigma) & (noise_sigma > 0)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise ValueError("No Q/U channels have finite, non-zero noise-band power")
    inv_sigma = 1 / noise_sigma[valid]

    # One whitened copy of the band, reused by both the mode-estimation and the
    # profile loops below.
    band_lo = min(b[0] for b in bins + profile_bins)
    band_hi = max(b[1] for b in bins + profile_bins)
    whitened = np.array(ft[valid_idx, band_lo:band_hi], copy=True)
    whitened *= inv_sigma[:, None]

    def _slice(i0, i1):
        return whitened[:, i0 - band_lo:i1 - band_lo]

    nchan = valid_idx.size
    mode_vecs, mode_singleness = [], []
    stats = {k: [] for k in (
        "bin_lo_hz", "bin_hi_hz", "gamma", "threshold", "lambda_plus",
        "n_above_lambda_plus", "n_above_threshold", "n_failed_singleness",
        "n_capped_by_max", "n_selected",
    )}

    for i0, i1 in bins:
        nb = i1 - i0
        n_samples = 2 * nb  # a complex sample carries two real dof
        stats["bin_lo_hz"].append(float(freqs[i0]))
        stats["bin_hi_hz"].append(float(freqs[i1 - 1]))
        if nb < 2:
            for key, val in (("gamma", np.nan), ("threshold", np.nan),
                             ("lambda_plus", np.nan)):
                stats[key].append(val)
            for key in ("n_above_lambda_plus", "n_above_threshold",
                        "n_failed_singleness", "n_capped_by_max", "n_selected"):
                stats[key].append(0)
            continue

        xbin = _slice(i0, i1)
        # Normalized by nb, not by n_samples: the whitening sets
        # mean|x|**2 = 1 per complex sample, so this puts the diagonal at 1,
        # which is the scale the MP edge is defined on. gamma below still uses
        # the 2 nb real degrees of freedom.
        cov = np.real(xbin.dot(np.conj(xbin.T))) / nb
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]

        singleness = np.max(np.abs(evecs), axis=0)
        keep = np.ones(evals.size, dtype=bool)
        if singleness_max is not None:
            keep &= singleness < singleness_max

        gamma = nchan / n_samples
        if mode_selection == "mp_tw":
            threshold, lambda_plus = get_marchenko_pastur_threshold(
                n_samples, nchan, significance=mp_significance, return_edge=True
            )
            above = evals > threshold
        else:
            threshold, lambda_plus = np.nan, np.nan
            above = np.zeros(evals.size, dtype=bool)
            above[:int(n_modes)] = True
        selected = np.flatnonzero(above & keep)
        n_before_cap = selected.size
        if n_modes_max is not None:
            selected = selected[:int(n_modes_max)]

        stats["gamma"].append(float(gamma))
        stats["threshold"].append(float(threshold))
        stats["lambda_plus"].append(float(lambda_plus))
        stats["n_above_lambda_plus"].append(
            int(np.sum(evals > lambda_plus)) if np.isfinite(lambda_plus) else 0
        )
        stats["n_above_threshold"].append(int(np.sum(above)))
        stats["n_failed_singleness"].append(int(np.sum(above & ~keep)))
        stats["n_capped_by_max"].append(int(n_before_cap - selected.size))
        stats["n_selected"].append(int(selected.size))

        if selected.size:
            mode_vecs.append(evecs[:, selected])
            mode_singleness.append(singleness[selected])

    if mode_vecs:
        vecs = np.concatenate(mode_vecs, axis=1)
        singleness_sel = np.concatenate(mode_singleness)
    else:
        vecs = np.zeros((nchan, 0))
        singleness_sel = np.zeros(0)
    nmodes = vecs.shape[1]
    pinv_vecs = np.linalg.pinv(vecs) if nmodes else np.zeros((0, nchan))

    nprof = len(profile_bins)
    residual_power = np.zeros((nchan, nprof))
    mode_power = np.zeros((nprof, nmodes))
    for k, (i0, i1) in enumerate(profile_bins):
        xbin = _slice(i0, i1)
        if nmodes:
            amps = pinv_vecs.dot(xbin)
            resid = xbin - vecs.dot(amps)
            mode_power[k] = np.mean(np.abs(amps) ** 2, axis=1)
        else:
            resid = xbin
        rp = np.mean(np.abs(resid) ** 2, axis=1)
        rp = np.where(np.isfinite(rp), rp, profile_diagonal_floor)
        residual_power[:, k] = np.maximum(rp, profile_diagonal_floor)

    # --- pack into an AxisManager on the dets axis, so that the stored
    # operator follows detector restriction like any other preprocess product.
    op = core.AxisManager(
        tod.dets,
        core.IndexAxis("nmat_modes", nmodes),
        core.IndexAxis("nmat_profile", nprof),
        core.IndexAxis("nmat_bins", len(bins)),
    )

    def _split(arr_over_channels, fill=0.0):
        """Scatter a valid-channel array back to (dets,) Q and U halves."""
        full = np.full(2 * ndet, fill, dtype=float)
        full[valid_idx] = arr_over_channels
        return full[:ndet], full[ndet:]

    sig_q, sig_u = noise_sigma[:ndet], noise_sigma[ndet:]
    op.wrap("sigma_Q", sig_q, [(0, "dets")])
    op.wrap("sigma_U", sig_u, [(0, "dets")])
    op.wrap("valid_Q", valid[:ndet], [(0, "dets")])
    op.wrap("valid_U", valid[ndet:], [(0, "dets")])

    vecs_full = np.zeros((2 * ndet, nmodes))
    vecs_full[valid_idx] = vecs
    op.wrap("vecs_Q", vecs_full[:ndet], [(0, "dets"), (1, "nmat_modes")])
    op.wrap("vecs_U", vecs_full[ndet:], [(0, "dets"), (1, "nmat_modes")])

    rp_full = np.zeros((2 * ndet, nprof))
    rp_full[valid_idx] = residual_power
    op.wrap("residual_power_Q", rp_full[:ndet], [(0, "dets"), (1, "nmat_profile")])
    op.wrap("residual_power_U", rp_full[ndet:], [(0, "dets"), (1, "nmat_profile")])
    op.wrap("mode_power", mode_power, [(0, "nmat_profile"), (1, "nmat_modes")])
    op.wrap("mode_singleness", singleness_sel, [(0, "nmat_modes")])

    op.wrap("profile_lo_hz",
            np.array([freqs[i0] for i0, _ in profile_bins]), [(0, "nmat_profile")])
    op.wrap("profile_hi_hz",
            np.array([freqs[i1 - 1] for _, i1 in profile_bins]), [(0, "nmat_profile")])
    for key, val in stats.items():
        op.wrap(key, np.asarray(val), [(0, "nmat_bins")])

    op.wrap("fmin", float(freqs[band_lo]))
    op.wrap("fmax", float(freqs[band_hi - 1]))
    op.wrap("profile_diagonal_floor", float(profile_diagonal_floor))
    op.wrap("nsamp", int(nsamp))
    return op


def _operator_arrays(tod, operator):
    """Unpack a stored operator into stacked-channel arrays."""
    if not np.array_equal(np.asarray(operator.dets.vals),
                          np.asarray(tod.dets.vals)):
        raise ValueError(
            "operator and tod detector axes differ; the stored Nmat operator "
            "is only valid for the detector set it was fit on"
        )
    sigma = np.concatenate([operator.sigma_Q, operator.sigma_U])
    valid = np.concatenate([operator.valid_Q, operator.valid_U]).astype(bool)
    vecs = np.concatenate([operator.vecs_Q, operator.vecs_U], axis=0)
    resid = np.concatenate([operator.residual_power_Q,
                            operator.residual_power_U], axis=0)
    valid_idx = np.flatnonzero(valid)
    return sigma, valid_idx, vecs[valid_idx], resid[valid_idx]


def _filtered_ft(tod, operator, signal_Q, signal_U, psd_scale):
    """Return (ft, filtered_ft_view) with the operator applied in place on ft."""
    if not 0 <= psd_scale <= 1:
        raise ValueError("psd_scale must be in [0, 1]")

    q = _resolve_signal(tod, signal_Q)
    u = _resolve_signal(tod, signal_U)
    ndet, nsamp = q.shape
    freqs = np.fft.rfftfreq(nsamp, _sample_dt(tod))

    sigma, valid_idx, vecs, resid = _operator_arrays(tod, operator)
    inv_sigma = 1 / sigma[valid_idx]
    nmodes = vecs.shape[1]

    ft = _stack_rfft(q, u)
    lo_hz = np.asarray(operator.profile_lo_hz)
    hi_hz = np.asarray(operator.profile_hi_hz)

    for k in range(lo_hz.size):
        i0 = int(np.searchsorted(freqs, lo_hz[k], side="left"))
        i1 = int(np.searchsorted(freqs, hi_hz[k], side="right"))
        if i1 <= i0:
            continue
        xbin = np.array(ft[valid_idx, i0:i1], copy=True)
        xbin *= inv_sigma[:, None]

        inv_diagonal = 1 / resid[:, k]
        filtered = inv_diagonal[:, None] * xbin
        if nmodes:
            physical = vecs * np.sqrt(
                np.maximum(operator.mode_power[k], np.finfo(float).tiny)
            )[None, :]
            invd = inv_diagonal[:, None] * physical
            core_mat = np.eye(nmodes) + physical.T.dot(invd)
            filtered -= invd.dot(np.linalg.solve(core_mat, invd.T.dot(xbin)))

        filtered = xbin + psd_scale * (filtered - xbin)
        filtered *= sigma[valid_idx, None]
        ft[valid_idx, i0:i1] = filtered
    return ft, ndet


def apply_joint_qu_nmat_operator(
    tod,
    operator,
    signal_Q="demodQ",
    signal_U="demodU",
    psd_scale=1.0,
    in_place=True,
    wrap_Q=None,
    wrap_U=None,
):
    """Apply a stored joint Q/U Nmat operator to a TOD.

    Working in white-noise-normalized coordinates makes the operator
    dimensionless with approximately unit response to white noise, i.e. it
    applies ``Sigma N^-1`` rather than ``N^-1``, leaving the white
    inverse-variance weighting to the mapmaker. DC and frequencies outside the
    range the operator was fit over are unchanged.

    Parameters
    ----------
    tod : AxisManager
        Observation to filter.
    operator : AxisManager
        Output of :func:`fit_joint_qu_nmat_operator`.
    signal_Q, signal_U : str or ndarray
        Q and U fields (or arrays) with shape (ndet, nsamp).
    psd_scale : float
        Blend from unchanged data (0) to the full operator (1).
    in_place : bool
        Overwrite the input Q/U arrays when true.
    wrap_Q, wrap_U : str or None
        Optionally wrap the filtered arrays into these TOD fields.

    Returns
    -------
    (filtered_Q, filtered_U)
    """
    q = _resolve_signal(tod, signal_Q)
    u = _resolve_signal(tod, signal_U)
    ft, ndet = _filtered_ft(tod, operator, signal_Q, signal_U, psd_scale)

    if in_place:
        q_out, u_out = q, u
    else:
        q_out, u_out = np.empty_like(q), np.empty_like(u)
    fft.irfft(ft[:ndet], q_out, normalize=True)
    fft.irfft(ft[ndet:], u_out, normalize=True)

    for wrap, out in ((wrap_Q, q_out), (wrap_U, u_out)):
        if wrap is None:
            continue
        if wrap in tod._fields:
            tod.move(wrap, None)
        tod.wrap(wrap, out, [(0, "dets"), (1, "samps")])
    return q_out, u_out


def get_nmat_subtraction(tod, operator, signal_Q="demodQ", signal_U="demodU",
                         psd_scale=1.0):
    """Return what the Nmat filter removes from a TOD.

    Gives ``(Q - F Q, U - F U)`` in time domain, i.e. the component the filter
    subtracts, for overplotting against the input timestreams or comparing this
    filter with others in TOD space.

    Parameters
    ----------
    tod : AxisManager
        Observation to evaluate the operator on.
    operator : AxisManager
        Output of :func:`fit_joint_qu_nmat_operator`.
    signal_Q, signal_U : str or ndarray
        Q and U fields (or arrays) with shape (ndet, nsamp).
    psd_scale : float
        Blend from unchanged data (0) to the full operator (1).

    Returns
    -------
    (removed_Q, removed_U)
    """
    q = _resolve_signal(tod, signal_Q)
    u = _resolve_signal(tod, signal_U)
    filt_q, filt_u = apply_joint_qu_nmat_operator(
        tod, operator, signal_Q=signal_Q, signal_U=signal_U,
        psd_scale=psd_scale, in_place=False,
    )
    return q - filt_q, u - filt_u


def apply_joint_qu_nmat_filter(
    tod,
    signal_Q="demodQ",
    signal_U="demodU",
    model_tod=None,
    psd_scale=1.0,
    in_place=True,
    wrap_Q=None,
    wrap_U=None,
    return_operator=False,
    **fit_kwargs,
):
    """Fit and apply a joint Q/U Nmat operator in one call.

    Convenience wrapper around :func:`fit_joint_qu_nmat_operator` followed by
    :func:`apply_joint_qu_nmat_operator`. Prefer the two-step form when the
    operator should be stored and reused, e.g. fit on real data and reapplied
    to a signal-only simulation.

    Parameters
    ----------
    tod : AxisManager
        Observation to filter.
    signal_Q, signal_U : str or ndarray
        Q and U fields (or arrays) with shape (ndet, nsamp).
    model_tod : AxisManager or None
        Observation the operator is fit from. Defaults to ``tod`` itself.
    psd_scale : float
        Blend from unchanged data (0) to the full operator (1).
    in_place : bool
        Overwrite the input Q/U arrays when true.
    wrap_Q, wrap_U : str or None
        Optionally wrap the filtered arrays into these TOD fields.
    return_operator : bool
        Also return the fitted operator AxisManager.
    **fit_kwargs
        Passed to :func:`fit_joint_qu_nmat_operator`.

    Returns
    -------
    (filtered_Q, filtered_U), operator_or_None
    """
    if model_tod is None:
        model_tod = tod
    elif isinstance(signal_Q, str) and isinstance(signal_U, str):
        if not np.array_equal(model_tod.dets.vals, tod.dets.vals):
            raise ValueError("model_tod and tod detector axes must match in order")
    else:
        raise ValueError(
            "signal_Q and signal_U must be field names with a separate model_tod"
        )

    operator = fit_joint_qu_nmat_operator(
        model_tod, signal_Q=signal_Q, signal_U=signal_U, **fit_kwargs
    )
    out = apply_joint_qu_nmat_operator(
        tod, operator, signal_Q=signal_Q, signal_U=signal_U,
        psd_scale=psd_scale, in_place=in_place, wrap_Q=wrap_Q, wrap_U=wrap_U,
    )
    return out, (operator if return_operator else None)
