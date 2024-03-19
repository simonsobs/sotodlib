import os
import numpy as np
import yaml
import argparse

from sotodlib import core
from sotodlib import coords
from sotodlib import tod_ops
from sotodlib.tod_ops.filters import high_pass_sine2, low_pass_sine2, fourier_filter
from sotodlib.coords import mapbased_pointing as mbp
from sotodlib.site_pipeline import update_pointing as up
from sotodlib.io.metadata import write_dataset

from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'make_mapbased_pointing: ')

def filter_tod(tod, cutoff_high=0.01, cutoff_low=1.8):
    if cutoff_low is not None:
        tod.signal = fourier_filter(tod, filt_function=low_pass_sine2(cutoff=cutoff_low),)
    if cutoff_high is not None:
        tod.signal = fourier_filter(tod, filt_function=high_pass_sine2(cutoff=cutoff_high),)
    return

def tod_process(tod):
    tod_ops.detrend_tod(tod)
    tod_ops.apodize_cosine(tod, apodize_samps=2000)
    filter_tod(tod)
    tod.restrict('samps', (tod.samps.offset+2000, tod.samps.offset+tod.samps.count-2000))
    return
    
def main(ctx_file, obs_id, wafer_slot,
         sso_name, optics_config_fn,
         map_dir, mapbased_result_dir, todbased_result_dir,
         tune_by_tod=True, R2_threshold=0.3, restrict_dets=False):
    
    ctx = core.Context(ctx_file)
    meta = ctx.get_meta(obs_id)
    meta.restrict('dets', meta.dets.vals[meta.det_info.wafer_slot == wafer_slot])
    if restrict_dets:
        meta.restrict('dets', meta.dets.vals[:100])
    logger.info('loading data')
    tod = ctx.get_obs(meta)
    logger.info('tod processing')
    tod_process(tod)
    
    if not os.path.exists(map_dir):
        logger.info(f'Make a directory: f{map_dir}')
        os.makedirs(map_dir)
    
    logger.info(f'Making single detector maps')
    map_hdf = os.path.join(map_dir, f'{obs_id}_{wafer_slot}.hdf')
    mbp.make_wafer_centered_maps(tod, sso_name, optics_config_fn, map_hdf=map_hdf,)
    
    logger.info(f'Making map-based pointing results')
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    focal_plane_rset_mapbased = mbp.get_xieta_from_maps(map_hdf, 
                                                        save=True,
                                                        output_dir=mapbased_result_dir,
                                                        filename=result_filename,
                                                        force_zero_roll=False,
                                                        edge_avoidance=1.0*coords.DEG)
    
    if tune_by_tod:
        focal_plane = core.AxisManager(tod.dets)
        focal_plane.wrap('xi', focal_plane_rset_mapbased['xi'], [(0, 'dets')])
        focal_plane.wrap('eta', focal_plane_rset_mapbased['eta'], [(0, 'dets')])
        focal_plane.wrap('gamma', focal_plane_rset_mapbased['gamma'], [(0, 'dets')])
        is_low_R2 = focal_plane_rset_mapbased['R2'] < R2_threshold
        focal_plane.xi[is_low_R2] = np.nan
        focal_plane.eta[is_low_R2] = np.nan
        
        tod.focal_plane = focal_plane
        tod.flags.move(sso_name, None)
        logger.info(f'Making tod-based pointing results')
        focal_plane_rset_todbased = up.update_xieta(tod, sso_name, ds_factor=10,
                                                    save=True, 
                                                    result_dir=todbased_result_dir, 
                                                    filename=result_filename)
    return



def get_parser():
    parser = argparse.ArgumentParser(description="Process TOD data and update pointing")
    parser.add_argument("ctx_file", type=str, help="Path to the context file")
    parser.add_argument("obs_id", type=str, help="Observation ID")
    parser.add_argument("wafer_slot", type=int, help="Wafer slot number")
    parser.add_argument("sso_name", type=str, help="Name of Solar System Object (SSO)")
    parser.add_argument("optics_config_fn", type=str, help="Path to optics configuration file")
    parser.add_argument("map_dir", type=str, help="Directory to save map data")
    parser.add_argument("mapbased_result_dir", type=str, help="Directory to save map-based result")
    parser.add_argument("todbased_result_dir", type=str, help="Directory to save TOD-based result")
    parser.add_argument("--tune_by_tod", action="store_true", help="Whether to tune by TOD data")
    parser.add_argument("--R2_threshold", type=float, default=0.3,
                        help="Threshold for R2 value. If R2 of map-domain result is lower than the threshold,\
                        the tod-fitting for that detector is skipped.")
    parser.add_argument("--restrict_dets", action="store_true",
                        help="If specified, number of detectors are restricted to 100")
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
