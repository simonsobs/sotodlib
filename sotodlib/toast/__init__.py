# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory simulation and processing modules using TOAST.

"""

# On first import, set toast default names to use across all built-in operators.

import toast

toast.observation.set_default_values(
    {
        "times": "times",
        "shared_flags": "flags",
        "det_data": "signal",
        "det_flags": "flags",
        "hwp_angle": "hwp_angle",
        "azimuth": "azimuth",
        "elevation": "elevation",
        "roll": "roll",
        "boresight_azel": "boresight_azel",
        "boresight_radec": "boresight_radec",
        "position": "position",
        "velocity": "velocity",
        "pixels": "pixels",
        "weights": "weights",
        "quats": "quats",
        "quats_azel": "quats_azel",
        "corotator_angle": "corotator_angle",
        "boresight_angle": "boresight_angle",
    }
)

from .instrument import SOFocalplane, simulated_telescope
from .sim_focalplane import sim_telescope_detectors
