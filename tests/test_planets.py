import os
import unittest
import pytest
import json
from datetime import datetime

from astropy.utils import data as au_data

from sotodlib.coords.planets import _get_astrometric


class TestPlanets(unittest.TestCase):

    def test_get_astrometric(self):

        timestamp = datetime.now().timestamp()
        with pytest.raises(RuntimeError):
            _get_astrometric(source_name="Jupiter", timestamp=timestamp)

        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {"de421.bsp": "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp"}
        )

        t = _get_astrometric(source_name="Jupiter", timestamp=timestamp)

        self.assertEqual(t.target, 5)

        # Returns a path to the downloaded data.
        astropy_data = au_data.download_file(
            "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp", cache=True
        )

        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {"de421.bsp": "file://" + astropy_data}
        )

        t = _get_astrometric(source_name="Jupiter", timestamp=timestamp)

        self.assertEqual(t.target, 5)
