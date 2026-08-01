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


def test_joint_qu_nmat_filter():
    tod = make_joint_tod()
    ndet = tod.dets.count
    common = np.sin(2 * np.pi * 0.1 * tod.timestamps)
    couplings = np.linspace(0.7, 1.3, 2 * ndet)
    joint = np.vstack([tod.demodQ, tod.demodU])
    joint += 30 * couplings[:, None] * common
    tod.demodQ[:] = joint[:ndet]
    tod.demodU[:] = joint[ndet:]

    before_q = tod.demodQ.copy()
    before_u = tod.demodU.copy()
    (q_filt, u_filt), diag = (
        tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
            tod,
            fmin=0.05,
            fmax=0.15,
            noise_band=(0.5, 2.0),
            bin_width_hz=0.1,
            mp_significance=0.999,
            n_modes_max=4,
            psd_scale=1,
            return_diagnostics=True,
        )
    )

    assert diag["selected_modes"].sum() > 0
    assert_allclose(tod.demodQ, before_q)
    assert_allclose(tod.demodU, before_u)
    assert_allclose(q_filt.mean(axis=1), before_q.mean(axis=1), atol=2e-5)
    assert_allclose(u_filt.mean(axis=1), before_u.mean(axis=1), atol=2e-5)

    nsamp = tod.samps.count
    fsamp = 1 / np.median(np.diff(tod.timestamps))
    freqs = np.fft.rfftfreq(nsamp, 1 / fsamp)
    signal_bin = np.argmin(np.abs(freqs - 0.1))
    high_bin = np.argmin(np.abs(freqs - 3.0))
    before_ft = np.fft.rfft(np.vstack([before_q, before_u]), axis=1)
    after_ft = np.fft.rfft(np.vstack([q_filt, u_filt]), axis=1)
    mode = couplings / np.linalg.norm(couplings)
    before_mode = np.abs(mode.dot(before_ft[:, signal_bin]))
    after_mode = np.abs(mode.dot(after_ft[:, signal_bin]))
    assert after_mode < before_mode * 0.1
    assert_allclose(
        after_ft[:, high_bin],
        before_ft[:, high_bin],
        rtol=2e-5,
        atol=2e-3,
    )


def test_joint_qu_nmat_filter_uses_separate_model_tod():
    model_tod = make_joint_tod(seed=11)
    target_tod = make_joint_tod(seed=12)
    ndet = model_tod.dets.count
    common = np.sin(2 * np.pi * 0.1 * model_tod.timestamps)
    couplings = np.linspace(0.7, 1.3, 2 * ndet)

    model_joint = np.vstack([model_tod.demodQ, model_tod.demodU])
    model_joint += 30 * couplings[:, None] * common
    model_tod.demodQ[:] = model_joint[:ndet]
    model_tod.demodU[:] = model_joint[ndet:]

    target_joint = np.vstack([target_tod.demodQ, target_tod.demodU])
    target_joint += 10 * couplings[:, None] * common
    target_tod.demodQ[:] = target_joint[:ndet]
    target_tod.demodU[:] = target_joint[ndet:]
    target_before = target_joint.copy()
    model_before = model_joint.copy()

    (q_filt, u_filt), diag = (
        tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
            target_tod,
            model_tod=model_tod,
            fmin=0.05,
            fmax=0.15,
            noise_band=(0.5, 2.0),
            bin_width_hz=0.1,
            mode_selection="fixed",
            n_modes=1,
            operator="nmat",
            profile_n_bins=1,
            profile_min_nfreq=10,
            psd_scale=1,
            return_diagnostics=True,
        )
    )

    signal_bin = np.argmin(
        np.abs(np.fft.rfftfreq(target_tod.samps.count, 0.05) - 0.1)
    )
    before_ft = np.fft.rfft(target_before, axis=1)
    after_ft = np.fft.rfft(np.vstack([q_filt, u_filt]), axis=1)
    mode = couplings / np.linalg.norm(couplings)
    assert diag["selected_modes"].sum() == 1
    assert np.abs(mode.dot(after_ft[:, signal_bin])) < (
        np.abs(mode.dot(before_ft[:, signal_bin])) * 0.1
    )
    assert_allclose(
        np.vstack([model_tod.demodQ, model_tod.demodU]), model_before
    )


def test_joint_qu_nmat_pipeline_step():
    tod = make_joint_tod(ndet=3, nsamp=1024)
    model_tod = make_joint_tod(ndet=3, nsamp=1024, seed=4321)
    proc_aman = core.AxisManager(tod.dets, tod.samps)
    pipe = Pipeline([{
        "name": "joint_qu_nmat_filter",
        "use_data_aman": True,
        "process": {
            "fmin": 0.1,
            "fmax": 1.0,
            "noise_band": [2.0, 8.0],
            "bin_width_hz": 0.5,
            "mode_selection": "fixed",
            "n_modes": 1,
            "psd_scale": 0,
        },
    }])
    q_before = tod.demodQ.copy()
    u_before = tod.demodU.copy()
    tod, _ = pipe[0].process(tod, proc_aman, sim=True, data_aman=model_tod)
    assert_allclose(tod.demodQ, q_before, rtol=2e-5, atol=2e-5)
    assert_allclose(tod.demodU, u_before, rtol=2e-5, atol=2e-5)



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
    (q_filt, u_filt), diag = (
        tod_ops.nmat_filter.apply_joint_qu_nmat_filter(
            tod,
            fmin=0.01,
            fmax=0.5,
            noise_band=(1.0, 3.0),
            bin_width_hz=0.49,
            mode_selection="fixed",
            n_modes=0,
            operator="nmat",
            profile_n_bins=16,
            profile_min_nfreq=20,
            psd_scale=1,
            return_diagnostics=True,
        )
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
    assert np.nanmin(diag["profile_power_response"]) < 0.1
    assert_allclose(
        after_ft[:, freqs > 1.0],
        before_ft[:, freqs > 1.0],
        rtol=2e-5,
        atol=2e-3,
    )


def test_marchenko_pastur_threshold_validation():
    threshold = tod_ops.nmat_filter.get_marchenko_pastur_threshold(
        n_samples=1000,
        n_features=100,
        significance=0.999,
    )
    assert threshold > (1 + np.sqrt(0.1)) ** 2
