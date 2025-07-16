# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory ancillary data retrieval and repackaging routines.

"""
import logging
logger = logging.getLogger(__name__)


from . import utils

from . import apex, scan_props, so_hk, so_obs, pwv

ANCIL_ENGINES = {
    'apex-pwv': apex.ApexPwv,
    'toco-pwv': so_hk.TocoPwv,
    'hwp-stats': so_obs.HwpStats,
    'weather-station': so_hk.WeatherStation,
    'scan-props': scan_props.ScanProps,
    'pwv': pwv.PwvCombo,
}
