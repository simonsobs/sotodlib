#!/usr/bin/env python

"""Copyright (c) 2023 Simons Observatory.
Full license can be found in the top level "LICENSE" file.

The only reason this exists is for backwards compatibility
such that calling `get_wafer_offset.py` will still work.
New users should either use `get_wafer_offset`
or `python -m sotodlib.workflows.get_wafer_offset`.
"""

from sotodlib.toast.workflows.get_wafer_offset import main


if __name__ == "__main__":
    main()
