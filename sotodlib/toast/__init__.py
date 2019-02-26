# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory TOAST interfaces.

This module contains classes and functions for working with the external
TOAST software tools.

"""

available = True
try:
    import toast
except:
    available = False
