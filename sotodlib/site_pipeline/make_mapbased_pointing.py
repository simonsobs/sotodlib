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
    
def main(configs, obs_id, wafer_slot, 
         sso_name=None, optics_config_fn=None,
         single_det_maps_dir=None, map_based_result_dir=None, tod_based_result_dir=None,
         tune_by_tod=None, restrict_dets_for_debug=False):
    
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
        
    # Derive parameters from config file
    if optics_config_fn is None:
        optics_config_fn = configs.get('optics_config_fn')
    if single_det_maps_dir is None:
        single_det_maps_dir = configs.get('single_det_maps_dir')
    if map_based_result_dir is None:
        map_based_result_dir = configs.get('map_based_result_dir')
    if tod_based_result_dir is None:
        tod_based_result_dir = configs.get('tod_based_result_dir')
    
    xieta_bs_offset = configs.get('xieta_bs_offset', [0., 0.])
    wafer_mask_deg = configs.get('wafer_mask_deg', 8.)
    res_deg = configs.get('res_deg', 0.3)
    edge_avoidance_deg = configs.get('edge_avoidance_deg', 0.3)
    save_force_zero_roll = configs.get('save_force_zero_roll', True)
    
    tune_by_tod = configs.get('tune_by_tod')
    R2_threshold = configs.get('R2_threshold')
    ds_factor = configs.get('ds_factor')
    
    
    ctx = core.Context(configs.get('context_file'))
    # If sso_name is not specified, get sso name from observation tags
    obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
    if sso_name is None:
        if 'moon' in obs_tags:
            sso_name = 'moon'
        elif 'jupiter' in obs_tags:
            sso_name = 'jupiter'
        else:
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
    
    # reconstruct pointing from single detector maps
    logger.info(f'Saving map-based pointing results')
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    fp_rset_map_based = mbp.get_xieta_from_maps(map_hdf, save=True,
                                                        output_dir=map_based_result_dir,
                                                        filename=result_filename,
                                                        force_zero_roll=False,
                                                        edge_avoidance = edge_avoidance_deg*coords.DEG)
    
    if tune_by_tod:
        focal_plane = core.AxisManager(tod.dets)
        focal_plane.wrap('xi', fp_rset_map_based['xi'], [(0, 'dets')])
        focal_plane.wrap('eta', fp_rset_map_based['eta'], [(0, 'dets')])
        focal_plane.wrap('gamma', fp_rset_map_based['gamma'], [(0, 'dets')])
        is_low_R2 = fp_rset_map_based['R2'] < R2_threshold
        focal_plane.xi[is_low_R2] = np.nan
        focal_plane.eta[is_low_R2] = np.nan
        
        tod.focal_plane = focal_plane
        tod.flags.move(sso_name, None)
        logger.info(f'Making tod-based pointing results')
        fp_rset_tod_based = up.update_xieta(tod, sso_name, ds_factor=ds_factor, save=True, 
                                            result_dir=tod_based_result_dir, filename=result_filename)
        
    if save_force_zero_roll:
        logger.info(f'Saving map-based pointing results (force-zero-roll)')
        output_dir = map_based_result_dir + '_force_zero_roll'
        fp_rset_map_based_force_zero_roll = mbp.get_xieta_from_maps(map_hdf, save=True,
                                                            output_dir=output_dir,
                                                            filename=result_filename,
                                                            force_zero_roll=True,
                                                            edge_avoidance = edge_avoidance_deg*coords.DEG)            
    return

def get_parser():
    parser = argparse.ArgumentParser(description="Process TOD data and update pointing")
    parser.add_argument("configs", type=str, help="Path to the configuration file")
    parser.add_argument("obs_id", type=int, help="Observation ID")
    parser.add_argument("wafer_slot", type=int, help="Wafer slot number")
    parser.add_argument("--sso_name", type=str, default=None, help="Name of solar system object (e.g., 'moon', 'jupiter')")
    parser.add_argument("--optics_config_fn", type=str, default=None, help="Path to optics configuration file")
    parser.add_argument("--single_det_maps_dir", type=str, default=None, help="Directory to save single detector maps")
    parser.add_argument("--map_based_result_dir", type=str, default=None, help="Directory to save map-based pointing results")
    parser.add_argument("--tod_based_result_dir", type=str, default=None, help="Directory to save TOD-based pointing results")
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
