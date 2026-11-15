import numpy as np
import pytest
import yaml

from sotodlib.site_pipeline.utils.depth1_utils import (
    DEPTH1MAPMAKER_DEFAULTS,
    create_mapmaker_config,
    gapfill_cuts,
    strip_samples,
)


@pytest.fixture
def file_config():
    mapmaker_config = {
        "query": "type == 'obs' and subtype == 'cmb' and tube_flavor == 'mf'",
        "context": "/so/metadata/lat/contexts/use_this.yaml",
        "area": "/so/site-pipeline/lat/maps/so_geometry_f090_lat.fits",
        "odir": "/so/site-pipeline/lat/maps/depth1/",
        "site": "so_lat",
        "mapcat_database_name": "/so/site-pipeline/lat/maps/depth1/mapcat.sqlite",
        "mapcat_depth_one_parent": "/so/site-pipeline/lat/maps/depth1/",
        "srcsamp": "/so/site-pipeline/lat/srcsamp/srcsamp_mask.fits",
        "comps": "T",
        "downsample": 2,
        "tiled": 1,
        "maxiter": 20,
        "verbose": True,
        "quiet": False,
        "tasks_per_group": 4,
        "unit": "K",
        "update_delay": 2,
    }
    return mapmaker_config


def test_defaults_only_raises_without_required_fields():
    with pytest.raises(KeyError):
        create_mapmaker_config()


def test_defaults():
    config = create_mapmaker_config(args={"area": "a", "context": "c"})
    for key, value in DEPTH1MAPMAKER_DEFAULTS.items():
        assert key in config
        assert config[key] == value
    assert config["area"] == "a"
    assert config["context"] == "c"


def test_config_file(file_config, tmp_path):
    file_config["preprocess_config"] = "/config/lat/preprocess_config_cmb_mf.yaml"

    fname = tmp_path / "config.yaml"
    fname.write_text(yaml.dump(file_config))

    config = create_mapmaker_config(config_file=str(fname))
    for key, value in file_config.items():
        assert key in config
        assert config[key] == value


def test_kwargs_config_file(file_config, tmp_path):
    fname = tmp_path / "config.yaml"
    fname.write_text(yaml.dump(file_config))

    config = create_mapmaker_config(
        config_file=str(fname), args={"preprocess_config": "file2.yaml"}
    )
    for key, value in file_config.items():
        assert key in config
        assert config[key] == value
    assert config["preprocess_config"] == "file2.yaml"


def test_config_file_override_kargs(file_config, tmp_path):
    file_config["preprocess_config"] = "file1.yaml"

    fname = tmp_path / "config.yaml"
    fname.write_text(yaml.dump(file_config))

    config = create_mapmaker_config(
        config_file=str(fname), args={"preprocess_config": "file2.yaml", "area": "new_area.fits"}
    )
    for key, value in file_config.items():
        if key in ["preprocess_config", "area"]:
            continue
        assert key in config
        assert config[key] == value
    assert config["preprocess_config"] == "file2.yaml"
    assert config["area"] == "new_area.fits"


def _strip_obs(sig, cut):
    import so3g
    from sotodlib import core

    ndet, nsamp = sig.shape
    obs = core.AxisManager(core.LabelAxis("dets", [f"d{i}" for i in range(ndet)]),
                           core.OffsetAxis("samps", nsamp))
    obs.wrap("signal", sig.astype(np.float32), [(0, "dets"), (1, "samps")])
    obs.wrap("flags", core.FlagManager.for_tod(obs))
    obs.flags.wrap("glitch_flags", so3g.proj.RangesMatrix.from_mask(cut),
                   [(0, "dets"), (1, "samps")])
    return obs


def test_strip_samples_and_gapfill():
    nsamp = 40
    x = np.arange(nsamp, dtype=float)
    base = 2.0 + 0.5 * x
    src = np.zeros(nsamp)
    src[10:15] = 10.0
    sig = np.stack([base + src, base + src])
    bright = np.zeros((2, nsamp), bool)
    bright[0, 9:16] = True  # inside the TOD
    bright[1, 0:4] = True   # at the TOD start: one end point
    bright[1, 37:] = True   # at the TOD end: one end point
    cut = bright.copy()
    cut[0, 16:18] = True    # glitch cut next to the strip
    obs = _strip_obs(sig, cut)

    s = strip_samples(obs, bright)
    assert s.det.size == bright.sum()
    d0 = s.det == 0
    np.testing.assert_array_equal(s.left[d0], 8)
    np.testing.assert_array_equal(s.right[d0], 18)
    np.testing.assert_allclose(s.resid[d0], src[9:16], atol=1e-4)
    d1 = s.det == 1
    start = d1 & (s.samp < 4)
    end = d1 & (s.samp >= 37)
    np.testing.assert_array_equal(s.left[start], 4)
    np.testing.assert_array_equal(s.right[start], 4)
    np.testing.assert_array_equal(s.left[end], 36)
    np.testing.assert_array_equal(s.right[end], 36)
    np.testing.assert_allclose(s.resid[start], base[0:4] - base[4], atol=1e-4)

    gapfill_cuts(obs)
    np.testing.assert_allclose(obs.signal[0], base, atol=1e-4)
    filled = obs.signal[1].copy()
    np.testing.assert_allclose(filled[0:4], base[4], atol=1e-4)
    np.testing.assert_allclose(filled[37:], base[36], atol=1e-4)
    # strip value = resid + fill at the end points gives back the data
    np.testing.assert_allclose(
        s.resid + (1 - s.frac) * obs.signal[s.det, s.left] + s.frac * obs.signal[s.det, s.right],
        sig[s.det, s.samp], atol=1e-4)
