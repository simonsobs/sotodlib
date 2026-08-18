# Copyright (c) 2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check that every process class registered with the preprocessing
Pipeline has documentation coverage in docs/preprocess.rst.

This guards against the failure mode where a new process is added to
sotodlib.preprocess.processes and registered on the pipeline, but never
added to the "Processing Modules" section of docs/preprocess.rst -- so
it's usable from a config file but invisible to anyone reading
https://sotodlib.readthedocs.io/en/latest/preprocess.html.

This only checks for missing/stale coverage of the registry against the
docs page -- it does not check that any of the RST itself is
well-formed. That's what the Sphinx build is for: it will fail on a broken
``.. autoclass::``/``.. autofunction::`` target or malformed docstring,
which is a class of bug this test can't see (e.g. it can't tell that a
docstring's own RST is broken -- only that some string referencing the
class exists somewhere on the page).
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

# Matches a double-backtick'd identifier, e.g. the ``FFTTrim`` cell of a
# "Process Step Glossary" table row.
DOUBLE_BACKTICK_RE = re.compile(r"``(\w+)``")


class TestPreprocessDocsCoverage(unittest.TestCase):
    """Cross-check the live Pipeline.PIPELINE registry against
    docs/preprocess.rst: every registered class needs both a
    ``.. autoclass::`` entry (its full reference section) and a row in
    one of the "Process Step Glossary" summary tables (its one-line
    quick-reference entry).
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
        rst_text = PREPROCESS_RST.read_text()
        self.documented = set(AUTOCLASS_RE.findall(rst_text))
        # Table rows aren't parsed structurally -- just check that each
        # class name appears somewhere as a double-backtick'd identifier
        # outside of its own ".. autoclass::" line, which is what a
        # "Class" column entry in a glossary table looks like.
        non_autoclass_text = "\n".join(
            line for line in rst_text.splitlines()
            if not line.lstrip().startswith(".. autoclass::")
        )
        self.in_tables = set(DOUBLE_BACKTICK_RE.findall(non_autoclass_text))

    def test_100_all_registered_processes_have_an_autoclass_entry(self):
        missing = sorted(self.registered - self.documented)
        self.assertEqual(
            missing, [],
            "The following process classes are registered on "
            "Pipeline.PIPELINE (sotodlib/preprocess/processes.py) but "
            "have no '.. autoclass::' entry in docs/preprocess.rst. Add "
            f"one under the 'Processing Modules' section for: {missing}"
        )

    def test_101_all_registered_processes_have_a_glossary_row(self):
        missing = sorted(self.registered - self.in_tables)
        self.assertEqual(
            missing, [],
            "The following process classes have no row in a 'Process "
            "Step Glossary' summary table in docs/preprocess.rst (no "
            "``ClassName`` appears outside an '.. autoclass::' line). "
            f"Add a table row for: {missing}"
        )


if __name__ == '__main__':
    unittest.main()
