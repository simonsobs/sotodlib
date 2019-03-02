# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware models for use in analysis.

This module contains code for simulating a hardware model and
dumping / loading hardware information to / from disk.

"""

# These are simply namespace imports for convenience.

from .config import (Hardware, get_example)

from .sim import (sim_wafer_detectors, sim_telescope_detectors)

from .vis import (plot_detectors, summary_text)
