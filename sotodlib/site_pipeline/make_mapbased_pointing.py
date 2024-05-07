import os
import numpy as np
import yaml
import argparse

from sotodlib import core
from sotodlib import coords
from sotodlib import tod_ops
from sotodlib.tod_ops.filters import high_pass_sine2, low_pass_sine2, fourier_filter
from sotodlib.coords import map_based_pointing as mbp
from sotodlib.site_pipeline import update_pointing as up
from sotodlib.io.metadata import write_dataset

from sotodlib.site_pipeline import util
from sotodlib.preprocess import Pipeline
logger = util.init_logger(__name__, 'make_map_based_pointing: ')

def main_one_wafer(configs, obs_id, wafer_slot,
         sso_name=None, 
         single_det_maps_dir=None, map_based_result_dir=None, tod_based_result_dir=None,
         tune_by_tod=None, restrict_dets_for_debug=False):
    
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
        
    # Derive parameters from config file
    ctx = core.Context(configs.get('context_file'))
    if single_det_maps_dir is None:
        single_det_maps_dir = configs.get('single_det_maps_dir')
    if map_based_result_dir is None:
        map_based_result_dir = configs.get('map_based_result_dir')
    if tod_based_result_dir is None:
        tod_based_result_dir = configs.get('tod_based_result_dir')
    optics_config_fn = configs.get('optics_config_fn')
    xieta_bs_offset = configs.get('xieta_bs_offset', [0., 0.])
    wafer_mask_deg = configs.get('wafer_mask_deg', 8.)
    res_deg = configs.get('res_deg', 0.3)
    edge_avoidance_deg = configs.get('edge_avoidance_deg', 0.3)
    save_normal_roll = configs.get('save_normal_roll', True)
    save_force_zero_roll = configs.get('save_force_zero_roll', True)
    
    # parameters for tod tuning
    tune_by_tod = configs.get('tune_by_tod')
    if tune_by_tod:
        tod_ds_factor = configs.get('tod_ds_factor')
        tod_mask_deg = configs.get('tod_mask_deg')
        tod_fit_func_name = configs.get('tod_fit_func_name')
        tod_max_non_linear_order = configs.get('tod_max_non_linear_order')
        tod_fwhm_init_deg = configs.get('tod_fwhm_init_deg')
        tod_error_estimation_method = configs.get('tod_error_estimation_method')
        tod_flag_name_rms_calc = configs.get('tod_flag_name_rms_calc')
        tod_flag_rms_calc_exclusive = configs.get('tod_flag_rms_calc_exclusive')
    
    
    # If sso_name is not specified, get sso name from observation tags
    obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
    if sso_name is None:
        known_source_names = ['moon', 'jupiter']
        for _source_name in known_source_names:
            if _source_name in obs_tags:
                sso_name = _source_name
        if _source_name is None:
            raise ValueError('sso_name is not specified')
    
    # Load data
    logger.info('loading data')
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    if restrict_dets_for_debug is not False:
        meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
    tod = ctx.get_obs(meta)
    
    # tod processing
    logger.info('tod processing')
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    proc_aman, success = pipe.run(tod)
    
    # make single detecctor maps
    logger.info(f'Making single detector maps')    
    os.makedirs(single_det_maps_dir, exist_ok=True)
    map_hdf = os.path.join(single_det_maps_dir, f'{obs_id}_{wafer_slot}.hdf')
    mbp.make_wafer_centered_maps(tod, sso_name, optics_config_fn, map_hdf=map_hdf, 
                                 xieta_bs_offset=xieta_bs_offset,
                                 wafer_mask_deg=wafer_mask_deg, res_deg=res_deg)
    
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    # reconstruct pointing from single detector maps
    if save_normal_roll:
        logger.info(f'Saving map-based pointing results')
        
        fp_rset_map_based = mbp.get_xieta_from_maps(map_hdf, save=True,
                                                            output_dir=map_based_result_dir,
                                                            filename=result_filename,
                                                            force_zero_roll=False,
                                                            edge_avoidance = edge_avoidance_deg*coords.DEG)

        if tune_by_tod:
            logger.info(f'Making tod-based pointing results')
            up.wrap_fp_rset(tod, fp_rset_map_based)
            fp_rset_tod_based = up.update_xieta( tod,
                                                 sso_name=sso_name,
                                                 fp_hdf_file=None,
                                                 force_zero_roll=False,
                                                 pipe=None,
                                                 ds_factor=tod_ds_factor,
                                                 mask_deg=tod_mask_deg,
                                                 fit_func_name = tod_fit_func_name,
                                                 max_non_linear_order = tod_max_non_linear_order,
                                                 fwhm_init_deg = tod_fwhm_init_deg,
                                                 error_estimation_method=tod_error_estimation_method,
                                                 flag_name_rms_calc = tod_flag_name_rms_calc,
                                                 flag_rms_calc_exclusive = tod_flag_rms_calc_exclusive, 
                                                 )
            os.makedirs(tod_based_result_dir, exist_ok=True)
            write_dataset(fp_rset_tod_based, 
                          filename=os.path.join(tod_based_result_dir, f'focal_plane_{obs_id}_{wafer_slot}.hdf'),
                          address='focal_plane',
                          overwrite=True)
        
    if save_force_zero_roll:
        logger.info(f'Saving map-based pointing results (force-zero-roll)')
        map_based_result_dir_force_zero_roll = map_based_result_dir + '_force_zero_roll'
        fp_rset_map_based_force_zero_roll = mbp.get_xieta_from_maps(map_hdf, save=True,
                                                            output_dir=map_based_result_dir_force_zero_roll,
                                                            filename=result_filename,
                                                            force_zero_roll=True,
                                                            edge_avoidance = edge_avoidance_deg*coords.DEG)
        if tune_by_tod:
            logger.info(f'Making tod-based pointing results (force-zero-roll)')
            up.wrap_fp_rset(tod, fp_rset_map_based_force_zero_roll)
            tod_based_result_dir_force_zero_roll = tod_based_result_dir + '_force_zero_roll'
            fp_rset_tod_based_force_zero_roll = up.update_xieta( tod,
                                                 sso_name=sso_name,
                                                 fp_hdf_file=None,
                                                 force_zero_roll=False,
                                                 pipe=None,
                                                 ds_factor=tod_ds_factor,
                                                 mask_deg=tod_mask_deg,
                                                 fit_func_name = tod_fit_func_name,
                                                 max_non_linear_order = tod_max_non_linear_order,
                                                 fwhm_init_deg = tod_fwhm_init_deg,
                                                 error_estimation_method=tod_error_estimation_method,
                                                 flag_name_rms_calc = tod_flag_name_rms_calc,
                                                 flag_rms_calc_exclusive = tod_flag_rms_calc_exclusive, 
                                                 )
            os.makedirs(tod_based_result_dir_force_zero_roll, exist_ok=True)
            write_dataset(fp_rset_tod_based_force_zero_roll, 
                          filename=os.path.join(tod_based_result_dir_force_zero_roll, f'focal_plane_{obs_id}_{wafer_slot}.hdf'),
                          address='focal_plane',
                          overwrite=True)
    return

def main(configs, obs_id, wafer_slots,
         sso_name=None, 
         single_det_maps_dir=None, map_based_result_dir=None, tod_based_result_dir=None,
         tune_by_tod=None, hit_time_threshold=1200, hit_circle_r_deg=7.0,
         restrict_dets_for_debug=False):
    
    logger.info('get wafer_slots which hit the source because wafer_slots are not specified')    
    if wafer_slots is None:
        if type(configs) == str:
            configs = yaml.safe_load(open(configs, "r"))
        
        ctx = core.Context(configs.get('context_file'))
        optics_config_fn = configs.get('optics_config_fn')
        
        obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
        if sso_name is None:
            known_source_names = ['moon', 'jupiter']
            for _source_name in known_source_names:
                if _source_name in obs_tags:
                    sso_name = _source_name
            if _source_name is None:
                raise ValueError('sso_name is not specified')
        
        wafer_slots = []
        tod = ctx.get_obs(obs_id, dets=[])
        for ws in [f'ws{i}' for i in range(7)]:
            hit_time = mbp.get_rough_hit_time(tod, wafer_slot=ws, sso_name=sso_name, circle_r_deg=hit_circle_r_deg,
                                             optics_config_fn=optics_config_fn)
            logger.info(f'hit_time for {ws} is {hit_time:.1f} [sec]')
            if hit_time > hit_time_threshold:
                wafer_slots.append(ws)        
    assert np.all(np.array(wafer_slots, dtype='U2') == 'ws')
    
    logger.info(f'wafer_slots which pointing calculated: {wafer_slots}')
    for wafer_slot in wafer_slots:
        main_one_wafer(configs=configs,
                       obs_id=obs_id,
                       wafer_slot=wafer_slot,
                       sso_name=sso_name,
                       single_det_maps_dir=single_det_maps_dir,
                       map_based_result_dir=map_based_result_dir, 
                       tod_based_result_dir=tod_based_result_dir,
                       tune_by_tod=tune_by_tod,
                       restrict_dets_for_debug=restrict_dets_for_debug)

def get_parser():
    parser = argparse.ArgumentParser(description="Process TOD data and update pointing")
    parser.add_argument("configs", type=str, help="Path to the configuration file")
    parser.add_argument("obs_id", type=str, help="Observation id")
    parser.add_argument("--wafer_slots", nargs='*', default=None, help="Wafer slots to be processed")
    parser.add_argument("--sso_name", type=str, default=None, help="Name of solar system object (e.g., 'moon', 'jupiter')")
    parser.add_argument("--single_det_maps_dir", type=str, default=None, help="Directory to save single detector maps")
    parser.add_argument("--map_based_result_dir", type=str, default=None, help="Directory to save map-based pointing results")
    parser.add_argument("--tod_based_result_dir", type=str, default=None, help="Directory to save TOD-based pointing results")
    parser.add_argument("--hit_time_threshold", type=float, default=1200, 
                        help="Minimum hit time. If calculated wafer hit time is smaller than that, pointing calculation for that wafer is skipped")
    parser.add_argument("--hit_circle_r_deg", type=float, default=7.,
                        help="circle radius for wafer hit time calculation")
    parser.add_argument("--restrict_dets_for_debug", type=int, default=False)
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
