"""Tests for joint demodulated Q/U common-mode filtering."""

import numpy as np
from numpy.testing import assert_allclose

from sotodlib import core, tod_ops
from sotodlib.preprocess import Pipeline


def make_joint_tod(ndet=6, nsamp=4096, fsamp=20.0, seed=1234):
    rng = np.random.default_rng(seed)
    tod = core.AxisManager(
        core.LabelAxis("dets", [f"det{i}" for i in range(ndet)]),
        core.OffsetAxis("samps", nsamp),
    )
    tod.wrap(
        "timestamps",
        np.arange(nsamp) / fsamp,
        [(0, "samps")],
    )
    q = rng.normal(size=(ndet, nsamp)).astype(np.float32)
    u = rng.normal(size=(ndet, nsamp)).astype(np.float32)
    tod.wrap("demodQ", q, [(0, "dets"), (1, "samps")])
    tod.wrap("demodU", u, [(0, "dets"), (1, "samps")])
    return tod


def add_common_mode(tod, amplitude, freq_hz=0.1):
    """Inject a shared sinusoid with a spread of per-channel couplings."""
    ndet = tod.dets.count
    common = np.sin(2 * np.pi * freq_hz * tod.timestamps)
    couplings = np.linspace(0.7, 1.3, 2 * ndet)
    joint = np.vstack([tod.demodQ, tod.demodU])
    joint += amplitude * couplings[:, None] * common
    tod.demodQ[:] = joint[:ndet]
    tod.demodU[:] = joint[ndet:]
    return couplings


FIT_KWARGS = dict(
    fmin=0.05,
    fmax=0.15,
    noise_band=(0.5, 2.0),
    bin_width_hz=0.1,
    mp_significance=0.999,
    n_modes_max=4,
)


def test_joint_qu_nmat_filter():
    tod = make_joint_tod()
    couplings = add_common_mode(tod, amplitude=30)

    before_q = tod.demodQ.copy()
    before_u = tod.demodU.copy()
    (q_filt, u_filt), operator = tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
        tod, in_place=False, return_operator=True, **FIT_KWARGS
    )

    assert operator.n_selected.sum() > 0
    # in_place=False must leave the input untouched
    assert_allclose(tod.demodQ, before_q)
    assert_allclose(tod.demodU, before_u)

    nsamp = tod.samps.count
    fsamp = 1 / np.median(np.diff(tod.timestamps))
    freqs = np.fft.rfftfreq(nsamp, 1 / fsamp)
    signal_bin = np.argmin(np.abs(freqs - 0.1))
    high_bin = np.argmin(np.abs(freqs - 3.0))
    before_ft = np.fft.rfft(np.vstack([before_q, before_u]), axis=1)
    after_ft = np.fft.rfft(np.vstack([q_filt, u_filt]), axis=1)
    mode = couplings / np.linalg.norm(couplings)
    assert np.abs(mode.dot(after_ft[:, signal_bin])) < (
        np.abs(mode.dot(before_ft[:, signal_bin])) * 0.1
    )
    # outside [fmin, fmax] the data is untouched
    assert_allclose(
        after_ft[:, high_bin], before_ft[:, high_bin], rtol=2e-5, atol=2e-3
    )


def test_fit_and_apply_match_single_call():
    """The two-step form must reproduce the one-shot convenience wrapper."""
    tod = make_joint_tod()
    add_common_mode(tod, amplitude=30)

    (q_one, u_one), _ = tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
        tod, in_place=False, **FIT_KWARGS
    )
    operator = tod_ops.nmat_filter.fit_joint_qu_nmat_operator(tod, **FIT_KWARGS)
    q_two, u_two = tod_ops.nmat_filter.apply_joint_qu_nmat_operator(
        tod, operator, in_place=False
    )
    assert_allclose(q_one, q_two)
    assert_allclose(u_one, u_two)


def test_operator_fit_on_model_applies_to_other_tod():
    """Fit on one observation, apply to another without refitting."""
    model_tod = make_joint_tod(seed=11)
    target_tod = make_joint_tod(seed=12)
    couplings = add_common_mode(model_tod, amplitude=30)
    add_common_mode(target_tod, amplitude=10)

    model_before = np.vstack([model_tod.demodQ, model_tod.demodU]).copy()
    target_before = np.vstack([target_tod.demodQ, target_tod.demodU]).copy()

    operator = tod_ops.nmat_filter.fit_joint_qu_nmat_operator(
        model_tod, mode_selection="fixed", n_modes=1,
        profile_n_bins=1, profile_min_nfreq=10,
        fmin=0.05, fmax=0.15, noise_band=(0.5, 2.0), bin_width_hz=0.1,
    )
    q_filt, u_filt = tod_ops.nmat_filter.apply_joint_qu_nmat_operator(
        target_tod, operator, in_place=False
    )

    signal_bin = np.argmin(
        np.abs(np.fft.rfftfreq(target_tod.samps.count, 0.05) - 0.1)
    )
    before_ft = np.fft.rfft(target_before, axis=1)
    after_ft = np.fft.rfft(np.vstack([q_filt, u_filt]), axis=1)
    mode = couplings / np.linalg.norm(couplings)
    assert operator.n_selected.sum() == 1
    assert np.abs(mode.dot(after_ft[:, signal_bin])) < (
        np.abs(mode.dot(before_ft[:, signal_bin])) * 0.1
    )
    # fitting must not modify the observation it was derived from
    assert_allclose(
        np.vstack([model_tod.demodQ, model_tod.demodU]), model_before
    )


def test_operator_rejects_mismatched_dets():
    tod = make_joint_tod(ndet=6)
    operator = tod_ops.nmat_filter.fit_joint_qu_nmat_operator(tod, **FIT_KWARGS)
    other = make_joint_tod(ndet=5)
    try:
        tod_ops.nmat_filter.apply_joint_qu_nmat_operator(other, operator)
    except ValueError as err:
        assert "detector axes differ" in str(err)
    else:
        raise AssertionError("expected a ValueError on mismatched dets")


def test_get_nmat_subtraction():
    tod = make_joint_tod()
    add_common_mode(tod, amplitude=30)
    before_q = tod.demodQ.copy()
    before_u = tod.demodU.copy()

    operator = tod_ops.nmat_filter.fit_joint_qu_nmat_operator(tod, **FIT_KWARGS)
    q_filt, u_filt = tod_ops.nmat_filter.apply_joint_qu_nmat_operator(
        tod, operator, in_place=False
    )
    removed_q, removed_u = tod_ops.nmat_filter.get_nmat_subtraction(tod, operator)

    assert_allclose(removed_q, before_q - q_filt)
    assert_allclose(removed_u, before_u - u_filt)
    assert np.any(np.abs(removed_q) > 0)


def test_selection_stats_are_recorded():
    tod = make_joint_tod()
    add_common_mode(tod, amplitude=30)
    operator = tod_ops.nmat_filter.fit_joint_qu_nmat_operator(
        tod, singleness_max=0.55, **FIT_KWARGS
    )
    nbins = operator.nmat_bins.count
    for key in ("n_above_lambda_plus", "n_above_threshold",
                "n_failed_singleness", "n_capped_by_max", "n_selected"):
        assert operator[key].shape == (nbins,)
    # every selected mode is accounted for by the threshold it passed
    assert np.all(operator.n_selected <= operator.n_above_threshold)
    assert np.all(operator.n_above_threshold <= operator.n_above_lambda_plus)
    assert operator.mode_singleness.shape == (operator.nmat_modes.count,)


def test_joint_qu_nmat_pipeline_steps():
    """The model step stores the operator; the filter step reloads it."""
    tod = make_joint_tod(ndet=3, nsamp=1024)
    proc_aman = core.AxisManager(tod.dets, tod.samps)
    pipe = Pipeline([
        {
            "name": "joint_qu_nmat_model",
            "skip_on_sim": True,
            "calc": {
                "fmin": 0.1,
                "fmax": 1.0,
                "noise_band": [2.0, 8.0],
                "bin_width_hz": 0.5,
                "mode_selection": "fixed",
                "n_modes": 1,
            },
            "save": {"wrap_name": "nmat_qu"},
        },
        {
            "name": "joint_qu_nmat_filter",
            "skip_on_sim": False,
            "process": {"nmat_model": "nmat_qu", "psd_scale": 0},
        },
    ])

    tod, proc_aman = pipe[0].calc_and_save(tod, proc_aman)
    assert "nmat_qu" in proc_aman

    # psd_scale=0 is a no-op, so this isolates the reload path
    q_before = tod.demodQ.copy()
    u_before = tod.demodU.copy()
    tod, _ = pipe[1].process(tod, proc_aman, sim=True)
    assert_allclose(tod.demodQ, q_before, rtol=2e-5, atol=2e-5)
    assert_allclose(tod.demodU, u_before, rtol=2e-5, atol=2e-5)


def test_filter_step_errors_without_model():
    tod = make_joint_tod(ndet=3, nsamp=1024)
    proc_aman = core.AxisManager(tod.dets, tod.samps)
    pipe = Pipeline([{
        "name": "joint_qu_nmat_filter",
        "process": {"nmat_model": "nmat_qu"},
    }])
    try:
        pipe[0].process(tod, proc_aman, sim=True)
    except KeyError as err:
        assert "nmat_qu" in str(err)
    else:
        raise AssertionError("expected a KeyError when the model is missing")


def test_joint_qu_nmat_downweights_diagonal_red_noise():
    tod = make_joint_tod(ndet=6, nsamp=8192, seed=5)
    rng = np.random.default_rng(6)
    fsamp = 1 / np.median(np.diff(tod.timestamps))
    freqs = np.fft.rfftfreq(tod.samps.count, 1 / fsamp)
    red_shape = np.sqrt(
        1 + (0.2 / np.maximum(freqs, freqs[1])) ** 2
    )

    for signal_name in ("demodQ", "demodU"):
        ft = (
            rng.normal(size=(tod.dets.count, freqs.size))
            + 1j * rng.normal(size=(tod.dets.count, freqs.size))
        ) * red_shape
        ft[:, 0] = 0
        tod[signal_name][:] = np.fft.irfft(
            ft, n=tod.samps.count, axis=1
        )

    before = np.vstack([tod.demodQ.copy(), tod.demodU.copy()])
    (q_filt, u_filt), _ = tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
        tod,
        fmin=0.01,
        fmax=0.5,
        noise_band=(1.0, 3.0),
        bin_width_hz=0.49,
        mode_selection="fixed",
        n_modes=0,
        profile_n_bins=16,
        profile_min_nfreq=20,
        in_place=False,
    )
    after = np.vstack([q_filt, u_filt])

    before_ft = np.fft.rfft(before, axis=1)
    after_ft = np.fft.rfft(after, axis=1)
    before_power = np.mean(np.abs(before_ft) ** 2, axis=0)
    after_power = np.mean(np.abs(after_ft) ** 2, axis=0)
    low = (freqs > 0.02) & (freqs < 0.15)
    white = (freqs > 1.0) & (freqs < 3.0)
    before_ratio = np.median(before_power[low]) / np.median(before_power[white])
    after_ratio = np.median(after_power[low]) / np.median(after_power[white])

    assert before_ratio > 4
    assert after_ratio < before_ratio * 0.1
    assert_allclose(
        after_ft[:, freqs > 1.0],
        before_ft[:, freqs > 1.0],
        rtol=2e-5,
        atol=2e-3,
    )


def test_marchenko_pastur_threshold_validation():
    lambda_plus_expected = (1 + np.sqrt(0.1)) ** 2
    threshold = tod_ops.nmat_filter.get_marchenko_pastur_threshold(
        n_samples=1000,
        n_features=100,
        significance=0.999,
    )
    assert threshold > lambda_plus_expected

    threshold2, lambda_plus = tod_ops.nmat_filter.get_marchenko_pastur_threshold(
        n_samples=1000,
        n_features=100,
        significance=0.999,
        return_edge=True,
    )
    assert threshold2 == threshold
    assert_allclose(lambda_plus, lambda_plus_expected)
    assert threshold > lambda_plus
