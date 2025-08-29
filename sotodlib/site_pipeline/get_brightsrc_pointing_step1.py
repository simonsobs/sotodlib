import os
import numpy as np
import yaml
import argparse
import time
import glob
from joblib import Parallel, delayed

from sotodlib import core
from sotodlib import coords
from sotodlib import tod_ops
from sotodlib.coords import brightsrc_pointing as bsp
from sotodlib.io import metadata
from sotodlib.io.metadata import read_dataset, write_dataset

from sotodlib.site_pipeline import util
from sotodlib.preprocess import Pipeline
logger = util.init_logger(__name__, 'make_map_based_pointing: ')

def _get_sso_names_from_tags(ctx, obs_id, candidate_names=['moon', 'jupiter', 'mars']):
    obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
    sso_names = []
    for _name in candidate_names:
        if _name in obs_tags:
            sso_names.append(_name)
    if len(sso_names) == 0:
        raise NameError('Could not find sso_name from observation tags')
    else:
        return sso_names
    
def main_one_wafer(configs, obs_id, wafer_slot, sso_name=None,
                   restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
        
    # Derive parameters from config file
    # required parameters
    ctx = core.Context(configs.get('context_file'))
    single_det_maps_dir = configs.get('single_det_maps_dir')
    result_dir = configs.get('result_dir')
    optics_config_fn = configs.get('optics_config_fn')
    save_normal_roll = configs.get('save_normal_roll')
    save_force_zero_roll = configs.get('save_force_zero_roll')
    
    # optional parameters
    xieta_bs_offset = configs.get('xieta_bs_offset', [0., 0.])
    wafer_mask_deg = configs.get('wafer_mask_deg', 8.)
    res_deg = configs.get('res_deg', 0.3)
    edge_avoidance_deg = configs.get('edge_avoidance_deg', 0.3)
    
    if sso_name is None:
        logger.info('deriving sso_name from observation tag')
        obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
        sso_names = _get_sso_names_from_tags(ctx, obs_id)
        sso_name = sso_names[0]
        if len(sso_names) >= 2:
            logger.info(f'sso_names of {sso_names} are found from observation tags.' + 
                        f'Processing only {sso_name}')
            
    # Load data
    logger.info(f'loading meta data: {wafer_slot}')
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    logger.info(f'finished loading meta data: {wafer_slot}')
    try:
        meta.restrict('dets', meta.detcal.bg > -1)
    except:
        pass
    if restrict_dets_for_debug is not False:
        try:
            restrict_dets_for_debug = int(restrict_dets_for_debug)
            meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
        except ValueError:
            _testdets = restrict_dets_for_debug.split(',')
            restrict_list = [det.split('\'')[1].strip() for det in _testdets]
            meta.restrict('dets', restrict_list)
    logger.info(f'loading tod data: {wafer_slot}')
    tod = ctx.get_obs(meta)
    logger.info(f'finished loading tod data: {wafer_slot}')
    # tod processing
    logger.info(f'tod processing {wafer_slot}')
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    proc_aman, success = pipe.run(tod)
    logger.info(f'done with tod processing {wafer_slot}')
    # make single detecctor maps
    logger.info(f'Making single detector maps')    
    os.makedirs(single_det_maps_dir, exist_ok=True)
    map_hdf = os.path.join(single_det_maps_dir, f'{obs_id}_{wafer_slot}.hdf')
    bsp.make_wafer_centered_maps(tod, sso_name, optics_config_fn, map_hdf=map_hdf, 
                                 xieta_bs_offset=xieta_bs_offset,
                                 wafer_mask_deg=wafer_mask_deg, res_deg=res_deg)

    #next step
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    # reconstruct pointing from single detector maps
    if save_normal_roll:
        logger.info(f'Saving map-based pointing results')
        
        fp_rset_map_based = bsp.get_xieta_from_maps(map_hdf, save=True,
                                                    output_dir=result_dir,
                                                    filename=result_filename,
                                                    force_zero_roll=False,
                                                    edge_avoidance = edge_avoidance_deg*coords.DEG)
        
    if save_force_zero_roll:
        logger.info(f'Saving map-based pointing results (force-zero-roll)')
        result_dir_force_zero_roll = result_dir + '_force_zero_roll'
        fp_rset_map_based_force_zero_roll = bsp.get_xieta_from_maps(map_hdf, save=True,
                                                            output_dir=result_dir_force_zero_roll,
                                                            filename=result_filename,
                                                            force_zero_roll=True,
                                                            edge_avoidance = edge_avoidance_deg*coords.DEG)
    return

def main_one_wafer_dummy(configs, obs_id, wafer_slot, restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    single_det_maps_dir = configs.get('single_det_maps_dir')
    result_dir = configs.get('result_dir')
    save_normal_roll = configs.get('save_normal_roll', True)
    save_force_zero_roll = configs.get('save_force_zero_roll', True)
    
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    if restrict_dets_for_debug is not False:
        try:
            restrict_dets_for_debug = int(restrict_dets_for_debug)
            meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
        except ValueError:
            _testdets = restrict_dets_for_debug.split(',')
            restrict_list = [det.split('\'')[1].strip() for det in _testdets]
            meta.restrict('dets', restrict_list)
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    
    fp_rset_dummy_map_based = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 'R2'])
    for det in meta.dets.vals:
        fp_rset_dummy_map_based.rows.append((det, np.nan, np.nan, np.nan, np.nan))
        
    if save_normal_roll:
        os.makedirs(result_dir, exist_ok=True)
        write_dataset(fp_rset_dummy_map_based, 
                      filename=os.path.join(result_dir, result_filename),
                      address='focal_plane',
                      overwrite=True)
            
    if save_force_zero_roll:
        result_dir_force_zero_roll = result_dir + '_force_zero_roll'
        os.makedirs(result_dir_force_zero_roll, exist_ok=True)
        write_dataset(fp_rset_dummy_map_based, 
                      filename=os.path.join(result_dir_force_zero_roll, result_filename),
                      address='focal_plane',
                      overwrite=True)
    return

def combine_pointings(pointing_result_files):
    combined_dict = {}
    for file in pointing_result_files:
        rset = read_dataset(file, 'focal_plane')
        for row in rset[:]:
            if row['dets:readout_id'] not in combined_dict.keys():
                combined_dict[row['dets:readout_id']] = {}
                combined_dict[row['dets:readout_id']]['xi'] = row['xi']
                combined_dict[row['dets:readout_id']]['eta'] = row['eta']
                combined_dict[row['dets:readout_id']]['gamma'] = row['gamma']
                combined_dict[row['dets:readout_id']]['R2'] = row['R2']

    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 'R2'])
    
    for det, val in combined_dict.items():
        focal_plane.rows.append((det, val['xi'], val['eta'], val['gamma'], val['R2']))
    return focal_plane

def parallel_process_wafer_slot(configs, obs_id, wafer_slot, sso_name, restrict_dets_for_debug):
    logger.info(f'Processing {obs_id}, {wafer_slot}')
    main_one_wafer(configs=configs,
                   obs_id=obs_id,
                   wafer_slot=wafer_slot,
                   sso_name=sso_name,
                   restrict_dets_for_debug=restrict_dets_for_debug)


def main_one_obs(configs, obs_id, sso_name=None,
                 restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    optics_config_fn = configs.get('optics_config_fn')
    
    result_dir = configs.get('result_dir')
    save_normal_roll = configs.get('save_normal_roll')
    save_force_zero_roll = configs.get('save_force_zero_roll')
    
    hit_time_threshold = configs.get('hit_time_threshold', 600)
    hit_circle_r_deg = configs.get('hit_circle_r_deg', 7.0)
    
    if sso_name is None:
        logger.info('deriving sso_name from observation tag')
        obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
        sso_names = _get_sso_names_from_tags(ctx, obs_id)
        sso_name = sso_names[0]
        if len(sso_names) >= 2:
            logger.info(f'sso_names of {sso_names} are found from observation tags.' + 
                        f'Processing only {sso_name}')
    
    tod = ctx.get_obs(obs_id, no_signal=True)
    streamed_wafer_slots = ['ws{}'.format(index) for index, bit in enumerate(obs_id.split('_')[-1]) if bit == '1']
    processed_wafer_slots = []
    finished_wafer_slots = []
    skipped_wafer_slots = []
    check_dir = result_dir + '_force_zero_roll' if save_force_zero_roll else result_dir
    
    for ws in streamed_wafer_slots:
        hit_time = bsp.get_rough_hit_time(tod,
                                          wafer_slot=ws,
                                          sso_name=sso_name,
                                          circle_r_deg=hit_circle_r_deg,
                                          optics_config_fn=optics_config_fn)
        logger.info(f'hit_time for {ws} is {hit_time:.1f} [sec]')
        if hit_time >= hit_time_threshold:
            if os.path.exists(os.path.join(check_dir, f'focal_plane_{obs_id}_{ws}.hdf')):
                finished_wafer_slots.append(ws)
            else:
                processed_wafer_slots.append(ws)
        else:
            skipped_wafer_slots.append(ws)
    
    logger.info(f'Found saved data for these wafer_slots: {finished_wafer_slots}')
    logger.info(f'Will continue for these wafer_slots: {processed_wafer_slots}')

    if configs.get('parallel_job'):
        logger.info('Continuing with parallel job')
        try:
            n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        except: 
            n_jobs = -1    

        logger.info('Entering wafer pool')
        Parallel(n_jobs=n_jobs)(
            delayed(parallel_process_wafer_slot)(
                configs,
                obs_id,
                wafer_slot,
                sso_name,
                restrict_dets_for_debug,
            )
            for wafer_slot in processed_wafer_slots
        )
        logger.info('Exiting wafer pool')
    else:
        logger.info('Continuing with serial processing of wafers.')
        for wafer_slot in processed_wafer_slots:
            logger.info(f'Processing {obs_id}, {wafer_slot}')
            main_one_wafer(configs=configs,
                           obs_id=obs_id,
                           wafer_slot=wafer_slot,
                           sso_name=sso_name,
                           restrict_dets_for_debug=restrict_dets_for_debug)        
           
    logger.info(f'create dummy hdf for non-hitting wafer: {skipped_wafer_slots}')
    for wafer_slot in skipped_wafer_slots:
        main_one_wafer_dummy(configs=configs,
                       obs_id=obs_id,
                       wafer_slot=wafer_slot,
                       restrict_dets_for_debug=restrict_dets_for_debug)
    
    logger.info('making combined result')
    if save_normal_roll:
        pointing_result_files = glob.glob(os.path.join(result_dir, f'focal_plane_{obs_id}_ws[0-6].hdf'))
        fp_rset_full = combine_pointings(pointing_result_files)
        fp_rset_full_file = os.path.join(os.path.join(result_dir, f'focal_plane_{obs_id}_all.hdf'))
        write_dataset(fp_rset_full, filename=fp_rset_full_file,
                      address='focal_plane', overwrite=True)
        
        
    if save_force_zero_roll:
        result_dir_force_zero_roll = result_dir + '_force_zero_roll'
        pointing_result_files = glob.glob(os.path.join(result_dir_force_zero_roll, f'focal_plane_{obs_id}_ws[0-6].hdf'))
        fp_rset_full = combine_pointings(pointing_result_files)
        fp_rset_full_file = os.path.join(os.path.join(result_dir_force_zero_roll, f'focal_plane_{obs_id}_all.hdf'))
        write_dataset(fp_rset_full, filename=fp_rset_full_file,
              address='focal_plane', overwrite=True)
        
    logger.info(f'ta da! Finished with {obs_id}')     
    return
    
def main(configs, min_ctime=None, max_ctime=None, update_delay=None,
         obs_id=None, wafer_slot=None, sso_name=None, restrict_dets_for_debug=False):
    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) - update_delay*86400
        
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    
    if obs_id is None:
        query_text = configs.get('query_text', None)
        query_tags = configs.get('query_tags', None)
        tot_query = "and "
        if query_text is not None:
            tot_query += f"{query_text} and "
        if min_ctime is not None:
            tot_query += f"timestamp>={min_ctime} and "
        if max_ctime is not None:
            tot_query += f"timestamp<={max_ctime} and "
        tot_query = tot_query[4:-4]
        if tot_query == "":
            tot_query = "1"
            
        logger.info(f'tot_query: {tot_query}')
        obs_list= ctx.obsdb.query(tot_query, query_tags)

        for obs in obs_list:
            obs_id = obs['obs_id']
            logger.info(f'Processing {obs_id}')
            main_one_obs(configs=configs, obs_id=obs_id,
                        restrict_dets_for_debug=restrict_dets_for_debug)
    
    elif obs_id is not None:
        logger.info(f'Processing {obs_id}')
        if wafer_slot is None:
            main_one_obs(configs=configs, obs_id=obs_id, sso_name=sso_name,
                         restrict_dets_for_debug=restrict_dets_for_debug)
        else:
            main_one_wafer(configs=configs, obs_id=obs_id, wafer_slot=wafer_slot, sso_name=sso_name, 
                           restrict_dets_for_debug=restrict_dets_for_debug)

def get_parser():
    parser = argparse.ArgumentParser(description="Process TOD data and update pointing")
    parser.add_argument("configs", type=str, help="Path to the configuration file")
    parser.add_argument('--min_ctime', type=int, help="Minimum timestamp for the beginning of an observation list")
    parser.add_argument('--max_ctime', type=int, help="Maximum timestamp for the beginning of an observation list")
    parser.add_argument('--update-delay', type=int, help="Number of days (unit is days) in the past to start observation list.")
    parser.add_argument("--obs_id", type=str, 
                        help="Specific observation obs_id to process. If provided, overrides other filtering parameters.")
                         
    parser.add_argument("--wafer_slot", type=str, default=None, 
                        help="Wafer slot to be processed (e.g., 'ws0', 'ws3'). Valid only when obs_id is specified.")
                         
    parser.add_argument("--sso_name", type=str, default=None,
                        help="Name of solar system object (e.g., 'moon', 'jupiter'). If not specified, get sso_name from observation tags. "\
                       + "Valid only when obs_id is specified")                     
    parser.add_argument("--restrict_dets_for_debug", type=str, default=False)
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
