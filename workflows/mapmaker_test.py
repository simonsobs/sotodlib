#!/usr/bin/env python

"""Copyright (c) 2023 Simons Observatory.
Full license can be found in the top level "LICENSE" file.

The only reason this exists is for backwards compatibility
such that calling `mapmaker_test.py` will still work.
New users should either use `mapmaker_test`
or `python -m sotodlib.workflows.mapmaker_test`.
"""

from sotodlib.toast.workflows.mapmaker_test import cli


if __name__ == "__main__":
    cli()
