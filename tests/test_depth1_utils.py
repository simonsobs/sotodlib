import os
import tempfile
import unittest

import yaml

from sotodlib.site_pipeline.utils.depth1_utils import (
    DEPTH1MAPMAKER_DEFAULTS,
    create_mapmaker_config,
)


class TestCreateMapmakerConfig(unittest.TestCase):

    def test_defaults_only_raises_without_required_fields(self):
       with self.assertRaises(KeyError):
            create_mapmaker_config()

    def test_defaults(self):
        config = create_mapmaker_config(args={"area": "a", "context": "c"})
        for key, value in DEPTH1MAPMAKER_DEFAULTS.items():
            self.assertIn(key, config)
            self.assertEqual(config[key], value)

    def test_config_file(self):
        file_config = {"query": "type == 'obs' and subtype == 'cmb' and tube_flavor == 'mf'",
                       "context": "/so/metadata/lat/contexts/use_this.yaml",
                       "area": "/so/site-pipeline/lat/maps/so_geometry_f090_lat.fits",
                       "odir": "/so/site-pipeline/lat/maps/depth1/",
                       "site": "so_lat",
                       "preprocess_config": "/config/lat/preprocess_config_cmb_mf.yaml",
                        "mapcat_database_name": '/so/site-pipeline/lat/maps/depth1/mapcat.sqlite',
                        "mapcat_depth_one_parent": '/so/site-pipeline/lat/maps/depth1/',
                        "srcsamp": '/so/site-pipeline/lat/srcsamp/srcsamp_mask.fits',
                        "comps": 'T',
                        "downsample": 2,
                        "tiled": 1,
                        "maxiter": 20,
                        "verbose": True,
                        "quiet": False,
                        "tasks_per_group": 4,
                        "unit": 'K',
                        "update_delay": 2}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(file_config, f)
            fname = f.name
        try:
            config = create_mapmaker_config(config_file=fname)
            for key, value in file_config.items():
                self.assertIn(key, config)
                self.assertEqual(config[key], value)
        finally:
            os.unlink(fname)

    def test_kwargs_config_file(self):
        file_config = {"query": "type == 'obs' and subtype == 'cmb' and tube_flavor == 'mf'",
                "context": "/so/metadata/lat/contexts/use_this.yaml",
                "area": "/so/site-pipeline/lat/maps/so_geometry_f090_lat.fits",
                "odir": "/so/site-pipeline/lat/maps/depth1/",
                "site": "so_lat",
                "mapcat_database_name": '/so/site-pipeline/lat/maps/depth1/mapcat.sqlite',
                "mapcat_depth_one_parent": '/so/site-pipeline/lat/maps/depth1/',
                "srcsamp": '/so/site-pipeline/lat/srcsamp/srcsamp_mask.fits',
                "comps": 'T',
                "downsample": 2,
                "tiled": 1,
                "maxiter": 20,
                "verbose": True,
                "quiet": False,
                "tasks_per_group": 4,
                "unit": 'K',
                "update_delay": 2}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(file_config, f)
            fname = f.name
        try:
            config = create_mapmaker_config(config_file=fname, args={"preprocess_config": "file2.yaml"})
            for key, value in file_config.items():
                self.assertIn(key, config)
                self.assertEqual(config[key], value)
            self.assertEqual(config["preprocess_config"], "file2.yaml")
        finally:
            os.unlink(fname)

    def test_config_file_override_kargs(self):
        file_config = {"query": "type == 'obs' and subtype == 'cmb' and tube_flavor == 'mf'",
                "context": "/so/metadata/lat/contexts/use_this.yaml",
                "area": "/so/site-pipeline/lat/maps/so_geometry_f090_lat.fits",
                "odir": "/so/site-pipeline/lat/maps/depth1/",
                "site": "so_lat",
                "preprocess_config": "/config/lat/preprocess_config_cmb_mf.yaml",
                "mapcat_database_name": '/so/site-pipeline/lat/maps/depth1/mapcat.sqlite',
                "mapcat_depth_one_parent": '/so/site-pipeline/lat/maps/depth1/',
                "srcsamp": '/so/site-pipeline/lat/srcsamp/srcsamp_mask.fits',
                "comps": 'T',
                "downsample": 2,
                "tiled": 1,
                "maxiter": 20,
                "verbose": True,
                "quiet": False,
                "tasks_per_group": 4,
                "unit": 'K',
                "update_delay": 2,
                "preprocess_config": "file1.yaml"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(file_config, f)
            fname = f.name
        try:
            config = create_mapmaker_config(config_file=fname, args={"preprocess_config": "file2.yaml"})
            for key, value in file_config.items():
                if key == "preprocess_config":
                    continue
                self.assertIn(key, config)
                self.assertEqual(config[key], value)
            # args override config_file values
            self.assertEqual(config["preprocess_config"], "file2.yaml")
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
