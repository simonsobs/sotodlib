# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory workflow functions.

"""

# Namespace imports

from .job import setup_job, reduction_group_size
from .data import (
    setup_load_data_context,
    load_data_context,
    setup_load_data_hdf5,
    load_data_hdf5,
    setup_load_data_books,
    load_data_books,
    setup_save_data_hdf5,
    save_data_hdf5,
    setup_save_data_books,
    save_data_books,
)
from .pointing import setup_pointing, select_pointing
from .proc_act import setup_act_responsivity_sign, act_responsivity_sign
from .proc_demodulation import setup_demodulate, demodulate
from .proc_filters import (
    setup_deconvolve_detector_timeconstant,
    deconvolve_detector_timeconstant,
    setup_filter_hwpss,
    filter_hwpss,
    setup_filter_ground,
    filter_ground,
    setup_filter_poly1d,
    filter_poly1d,
    setup_filter_poly2d,
    filter_poly2d,
    setup_filter_common_mode,
    filter_common_mode,
)
from .proc_flagging import (
    setup_simple_jumpcorrect,
    simple_jumpcorrect,
    setup_simple_deglitch,
    simple_deglitch,
    setup_flag_sso,
    flag_sso,
    setup_flag_diff_noise_outliers,
    flag_diff_noise_outliers,
    setup_flag_noise_outliers,
    flag_noise_outliers,
    setup_processing_mask,
    processing_mask,
)
from .proc_intervals import setup_az_intervals, create_az_intervals
from .proc_mapmaker_filterbin import setup_mapmaker_filterbin, mapmaker_filterbin
from .proc_mapmaker_madam import setup_mapmaker_madam, mapmaker_madam
from .proc_mapmaker_ml import setup_mapmaker_ml, mapmaker_ml
from .proc_mapmaker import setup_mapmaker, mapmaker
from .proc_noise_est import (
    setup_diff_noise_estimation,
    diff_noise_estimation,
    setup_noise_estimation,
    noise_estimation,
)
from .proc_characterize import (
    setup_raw_statistics,
    raw_statistics,
    setup_filtered_statistics,
    filtered_statistics,
    setup_hn_map,
    hn_map,
    setup_cadence_map,
    cadence_map,
    setup_crosslinking_map,
    crosslinking_map,
)
from .sat import setup_splits, splits
from .sim_observe import setup_simulate_observing, simulate_observing
from .sim_noise_model import setup_simple_noise_models, simple_noise_models
from .sim_sky import (
    setup_simulate_sky_map_signal,
    simulate_sky_map_signal,
    setup_simulate_conviqt_signal,
    simulate_conviqt_signal,
)
from .sim_sources import (
    setup_simulate_source_signal,
    simulate_source_signal,
    setup_simulate_sso_signal,
    simulate_sso_signal,
    setup_simulate_catalog_signal,
    simulate_catalog_signal,
)
from .sim_atm import setup_simulate_atmosphere_signal, simulate_atmosphere_signal
from .sim_calibrator import (
    setup_simulate_wiregrid_signal,
    simulate_wiregrid_signal,
    setup_simulate_stimulator_signal,
    simulate_stimulator_signal,
)
from .sim_optical_pickup import (
    setup_simulate_scan_synchronous_signal,
    simulate_scan_synchronous_signal,
    setup_simulate_hwpss_signal,
    simulate_hwpss_signal,
)
from .sim_detector_readout import (
    setup_simulate_detector_timeconstant,
    simulate_detector_timeconstant,
    setup_simulate_detector_noise,
    simulate_detector_noise,
    setup_simulate_readout_effects,
    simulate_readout_effects,
    setup_simulate_detector_yield,
    simulate_detector_yield,
    setup_simulate_mumux_crosstalk,
    simulate_mumux_crosstalk,
)
from .sim_gain_error import (
    setup_simulate_calibration_error,
    simulate_calibration_error,
)
