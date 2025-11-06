import os
import unittest
import json
import shutil

from sotodlib.core import resources

ODD_NAME = 'planet47-ephemeris.txt'


class TestCoreResources(unittest.TestCase):

    def test_get_local_file(self):

        # This url should not be accessed, as long as download=False
        resources.RESOURCE_DEFAULTS[ODD_NAME] = \
            'ftp://simonsobs.org/sotodlib-ci.txt'

        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {"someotherfile": "file://somepath/otherfile"}
        )

        # Use a fake file for this, or else it might find a
        # user-cached copy.
        t = resources.get_local_file(ODD_NAME, cache=False, download=False)
        expected_path = "/tmp/" + ODD_NAME
        self.assertEqual(expected_path, t)

        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {ODD_NAME: "file://somepath/de421.bsp"}
        )
        t = resources.get_local_file(ODD_NAME)
        self.assertEqual("somepath/de421.bsp", t)

        with self.assertRaises(RuntimeError):
            resources.get_local_file("doesnotexist.file")
