import importlib
import unittest


class TestSitePipelineCli(unittest.TestCase):
    def test_full_parser(self):
        """Build the full so-site-pipeline parser. Python >= 3.14 validates
        help strings at add_argument time, so a stray '%' breaks everything."""
        try:
            cli = importlib.import_module('sotodlib.site_pipeline.cli')
        except ImportError as e:
            self.skipTest(f'optional dependency missing: {e}')
        cli.get_parser().format_help()


if __name__ == '__main__':
    unittest.main()
