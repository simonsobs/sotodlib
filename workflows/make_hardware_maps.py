#!/usr/bin/env python

"""Copyright (c) 2023 Simons Observatory.
Full license can be found in the top level "LICENSE" file.

The only reason this exists is for backwards compatibility
such that calling `make_hardware_maps.py` will still work.
New users should either use `make_hardware_maps`
or `python -m sotodlib.workflows.make_hardware_maps`.
"""

from sotodlib.toast.workflows.make_hardware_maps import main


if __name__ == "__main__":
    main()
