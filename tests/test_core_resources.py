import os
import unittest
import json
import shutil
import pytest

from sotodlib.core.resources import get_local_file


class TestCoreResources(unittest.TestCase):

    def test_get_local_file(self):

        # t = get_local_file("de421.bsp", cache=True)
        # expected_path = os.path.join(
        #     os.path.expanduser("~"), ".sotodlib/filecache/de421.bsp"
        # )
        # self.assertEqual(expected_path, t)
        # shutil.rmtree(os.path.join(os.path.expanduser("~"), ".sotodlib/"))
        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {"someotherfile": "file://somepath/otherfile"}
        )
        t = get_local_file("de421.bsp", cache=False)
        expected_path = "/tmp/de421.bsp"
        self.assertEqual(expected_path, t)

        os.environ["SOTODLIB_RESOURCES"] = json.dumps(
            {"de421.bsp": "file://somepath/de421.bsp"}
        )
        t = get_local_file("de421.bsp")
        self.assertEqual("somepath/de421.bsp", t)

        with pytest.raises(RuntimeError):
            get_local_file("doesnotexist.file")
