# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Internal helper functions for unit tests.
"""

import os


def create_outdir(subdir=None):
    """Create the top level output directory and per-test subdir.

    Args:
        subdir (str): the sub directory for this test.

    Returns:
        str: full path to the test subdir if specified, else the top dir.

    """
    pwd = os.path.abspath(".")
    testdir = os.path.join(pwd, "sotodlib_test_output")
    retdir = testdir
    if subdir is not None:
        retdir = os.path.join(testdir, subdir)
    if not os.path.isdir(testdir):
        os.mkdir(testdir)
    if not os.path.isdir(retdir):
        os.mkdir(retdir)
    return retdir
