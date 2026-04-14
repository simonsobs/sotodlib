import pytest
import yaml

from sotodlib.site_pipeline.utils.depth1_utils import (
    DEPTH1MAPMAKER_DEFAULTS,
    create_mapmaker_config,
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
