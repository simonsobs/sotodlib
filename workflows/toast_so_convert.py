#!/usr/bin/env python

"""Copyright (c) 2023 Simons Observatory.
Full license can be found in the top level "LICENSE" file.

The only reason this exists is for backwards compatibility
such that calling `toast_so_convert.py` will still work.
New users should either use `toast_so_convert`
or `python -m sotodlib.workflows.toast_so_convert`.
"""

from sotodlib.toast.workflows.toast_so_convert import cli


if __name__ == "__main__":
    cli()
