# Copyright (c) 2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check that every process class registered with the preprocessing
Pipeline has documentation coverage in docs/preprocess.rst.

This guards against the failure mode where a new process is added to
sotodlib.preprocess.processes and registered on the pipeline, but never
added to the "Processing Modules" section of docs/preprocess.rst -- so
it's usable from a config file but invisible to anyone reading
https://sotodlib.readthedocs.io/en/latest/preprocess.html. An audit found
32 of 54 registered classes missing from the docs this way before this
test was added; see the "preproc_docs" repo's GAP_ANALYSIS.md for the
full write-up.
"""

import re
import unittest
from pathlib import Path

from sotodlib.preprocess import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_RST = REPO_ROOT / "docs" / "preprocess.rst"

# Matches lines like:
#   .. autoclass:: sotodlib.preprocess.processes.GlitchDetection
AUTOCLASS_RE = re.compile(
    r"^\.\.\s+autoclass::\s+sotodlib\.preprocess\.processes\.(\w+)\s*$",
    re.MULTILINE,
)


class TestPreprocessDocsCoverage(unittest.TestCase):
    """Cross-check the live Pipeline.PIPELINE registry against the
    ``.. autoclass::`` entries in docs/preprocess.rst.
    """

    def setUp(self):
        if not PREPROCESS_RST.exists():
            # docs/ isn't guaranteed to ship alongside every install of
            # the sotodlib package (e.g. a pip install from an sdist
            # without docs/), so skip rather than fail in that case --
            # this check only makes sense when run from a full checkout,
            # which is how it's run in CI.
            self.skipTest(
                f"{PREPROCESS_RST} not found -- skipping doc-coverage "
                "check (this test needs a full repo checkout, not just "
                "an installed sotodlib package)."
            )
        self.registered = {
            cls.__name__ for cls in Pipeline.PIPELINE.values()
        }
        self.documented = set(AUTOCLASS_RE.findall(PREPROCESS_RST.read_text()))

    def test_100_all_registered_processes_are_documented(self):
        missing = sorted(self.registered - self.documented)
        self.assertEqual(
            missing, [],
            "The following process classes are registered on "
            "Pipeline.PIPELINE (sotodlib/preprocess/processes.py) but "
            "have no '.. autoclass::' entry in docs/preprocess.rst. Add "
            "one under the 'Processing Modules' section (and a row in "
            f"the 'Process Step Glossary' table) for: {missing}"
        )

    def test_101_no_stale_autoclass_entries(self):
        stale = sorted(self.documented - self.registered)
        self.assertEqual(
            stale, [],
            "The following classes are referenced via '.. autoclass::' "
            "in docs/preprocess.rst but are no longer registered on "
            f"Pipeline.PIPELINE (renamed or removed?): {stale}"
        )


if __name__ == '__main__':
    unittest.main()
