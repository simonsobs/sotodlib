import unittest
from unittest.mock import MagicMock
from typing import cast

from sotodlib.core import AxisManager
from sotodlib.mapmaking.utilities import parameter_in_path, get_flags_from_path

class TestParameterInPath(unittest.TestCase):
    
    def setUp(self):
        # Mock objects for testing
        obj1 = MagicMock()
        obj1.__dict__ = {"c": 42}
        obj2 = MagicMock()
        obj2.__dict__ = {"b": obj1}
        obj3 = MagicMock()
        obj3.__dict__ = {"a": obj2, "x": 5}
        self.obj_with_dict = obj3

        flags = MagicMock(spec=AxisManager)
        flags._fields = {"glitch_flags": "some_value"}
        aman = MagicMock(spec=AxisManager)
        aman._fields = {"flags": flags,
                        "other_key": "another_value"}

        self.obj_with_fields = aman

    def test_parameter_exists_in_dict(self):
        self.assertTrue(parameter_in_path(self.obj_with_dict, 'a.b.c'))
        self.assertFalse(parameter_in_path(self.obj_with_dict, 'a.b.d'))
        self.assertTrue(parameter_in_path(self.obj_with_dict, 'x'))

    def test_parameter_exists_in_fields(self):
        self.assertTrue(parameter_in_path(self.obj_with_fields, 'flags.glitch_flags'))
        self.assertFalse(parameter_in_path(self.obj_with_fields, 'flags.non_existent'))

    def test_parameter_with_custom_separator(self):
        self.assertTrue(parameter_in_path(self.obj_with_dict, 'a/b/c', sep='/'))

class TestGetFlagsFromPath(unittest.TestCase):

    def setUp(self):
        # Mocking core.AxisManager
        self.axis_manager = cast(AxisManager, MagicMock())
        self.axis_manager.__getitem__.side_effect = lambda key: {
            'flags': {'glitch_flags': 'some_value'},
            'other_key': 'another_value'
        }[key]

    def test_get_flags_from_path(self):
        result = get_flags_from_path(self.axis_manager, 'flags.glitch_flags')
        self.assertEqual(result, 'some_value')

    def test_get_non_existent_flag(self):
        with self.assertRaises(KeyError):
            get_flags_from_path(self.axis_manager, 'flags.non_existent')

    def test_get_flags_with_custom_separator(self):
        result = get_flags_from_path(self.axis_manager, 'flags/glitch_flags', sep='/')
        self.assertEqual(result, 'some_value')

if __name__ == '__main__':
    unittest.main()
