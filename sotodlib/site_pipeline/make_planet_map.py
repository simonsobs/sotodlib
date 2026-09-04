"""
This script makes an atmic planet map in instrument/planet-centered coordinate system based on a provided config file.
First make runlist for each observation/wafer/band, then parallelize the map making work for each observation/wafer/band.
Output is atomic planet maps and an assosiate database.
TOD fitting can be performed before map making if you need, and the results will be stored in a separate database.
Main function is sotodlib.mapmaking.planet_mapmaker.planet_mapmake_eachobs.

Example config file
---------------------
context_file: "/home/ys5857/workspace/script/planet_config/ys_planet_260212/contexts/satp3/use_this_local_251215.yaml" # path to the context file, same as 251215 for comparing this with 2025/12/15 result

base_dir: '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/test/2026/09/03'
save_dir: '{base_dir}/detcen_jupiter/'
errlogpath: "{save_dir}/err/satp3_detcen_jupiter_errlog.txt"
# database paths
dbpath: "{save_dir}/db/detcen_coadd_map.sqlite"
todfit_dbpath: "{save_dir}/db/todfit.sqlite"
single_dbpath: "{save_dir}/db/detcen_single_detector_map.sqlite"
overwrite: True # overwrite maps even if they already exist

query:
    start: '2024/7/22'
    end: '2026/1/1'
    tags: ['jupiter', 'taua']
    all_wafers: True # whether to include all wafers in obs or only the ones with tags. If True, it will include all wafers in obs regardless of tags. If False, it will only include wafers with tags.
    specific_wafers: [0, 1]  #or None # if all_wafers is False, then include only these particular wafers, replace 0,1 with any waferslot number or None

mapmaking:
    map_PCA: False # whether to map PCA modes instead of actual signal, if True, the PCA modes will be mapped and the aman.signal, aman.dsT, aman.demodQ, aman.demodU will be replaced by the PCA mode signal before making maps.
    inv_var:
      signal: 'demodQU' # Which signal to use for inverse variance: 'demodQ','demodU' or 'demodQU'.
    fittod:
      process: False # False for not performing the fit
      Tsignal: "signal" # name of Tsignal. Usually "dsT" or "signal", but you can use anthing if you make special Tsignal.
      dbpath: "{todfit_dbpath}"
      #flags: "planet_mapmaking"
      r_use: 1.0 # degrees, the radius of the circle used for process
      r_fit: 0.8 # degrees, the radius of the circle used for fit
      mask_res: 2 # arcmin, the resolution of the map for making flags
      detrend_eachscan: True # whether to detrend each set of scan
      npoly: 0 # order of polynomial for detrending
      deflection_model: True # whether to use pointing deflection model for todfit
      fitselection: False # Whether to apply selection based on todfit results. If True, it will apply selection based on todfit residuals. If False, it will not apply any selection.
      #threshold_path: '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2025/12/15/detcen_jupiter/todfit_selection_threshold.pkl'
    xieta_correction:
      process: False # True, apply xi/eta correciton based on todfit
    deflection_correction:
      process: True # True, apply pointing deflection correction
      wafer_base: True # True, apply wafer-base correction. False and process is True, apply pointing deflection correction for every single detector based on todfit
    map:
      coordinate: "detector_center" # "detector_center" or "boresight_center" or "planet_equatorial" or "planet_horizon"
      recv_coords: False # Whether to use receiver coordinates system, only works with "planet_horizon"
      source: "jupiter"
      res: 2/60 # degrees
      size: 5 # degrees
      proj: "car"
      flags: "planet_mapmaking" # define the flags used in mapmaking
      Tsignal: "signal" # name of Tsignal. Usually "dsT" or "signal", but you can use anthing if you make special Tsignal.
      recenter: False # whether to recenter output maps
      var_minr: 50 # arcmin, minimum radius for calculating map variance
      var_maxr: 72 # arcmin, maximum radius for calculating map variance
      save_dire: "{save_dir}"
      single_save: False # Save single map or not, currently only works with "detector_center" or "boresight_center" coordinate
      single_db: "{single_dbpath}" # database path for saving single maps
    bootstrap: # this is only for boresight_center/detector_center for now.
      process: False # True, apply bootstrap correction
      N_bootstrap: 10 # numeber of bootstrap realizations
      save_dire: "{base_dir}/detcen_jupiter/bootstrap/"
      save_map: False
      fit_map: False # Whether or not fitting the bootstrap coadded map
      fitthre: 1.2 # fitting radius threshold in deg
      sig_ran: 1.0 # signal range in deg. sig_ran < r < fitthre is used to estimate the background(i.e., offset)


process_pipe:
    - name: "pointing_model"
      process: True

    - name: "hwp_angle_model"
      process: True
      calc:
        on_sign_ambiguous: 'fail'
      save: True

    - name: "correct_iir_params"
      process: True

    - name: "smurfgaps_flags"
      calc:
        buffer: 200
        name: "smurfgaps"
        merge: True
      save: True
    
    More process steps can be added here if needed, following the same structure as above.
    See preprocess.py for more details on available process steps and their configurations 
    or configs/example_satp3_planet_map.yaml for an example configuration file.
"""


import yaml, os, time, datetime, traceback
from typing import Optional, Union, Callable
import numpy as np
from argparse import ArgumentParser
from typing import Optional, List, Callable

from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.site_pipeline.utils.pipeline import main_launcher
from sotodlib.mapmaking import planet_mapmaker
from sotodlib import core
from ..site_pipeline.utils import logging

def future_write_to_log(e, errlog):
    errmsg = f'{type(e)}: {e}'
    tb = ''.join(traceback.format_tb(e.__traceback__))
    f = open(errlog, 'a')
    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
    f.close()

def main(
    config_path: str,
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable) -> None:

    verbosity = 2

    # Load the configuration file
    configs = planet_mapmaker.read_configs(config_path)
    context = core.Context(configs["context_file"])
    logger = logging.init_logger("preprocess", verbosity=verbosity)
    logger.info(f'context_file: {context}')

    # make runlist
    logger.info(f'load obslist from queue written in yaml file')
    start = configs['query']['start']
    end = configs['query']['end']
    if isinstance(start, str):
        start = datetime.datetime.strptime(start, '%Y/%m/%d').replace(tzinfo=datetime.timezone.utc).timestamp()
    if isinstance(end, str):
        end = datetime.datetime.strptime(end, '%Y/%m/%d').replace(tzinfo=datetime.timezone.utc).timestamp()
    if not (isinstance(start, (float, int)) and isinstance(end, (float, int))):
        raise ValueError('start and end must be float or int or str following YYYY/MM/DD')
        
    obslist_all = context.obsdb.query(f'timestamp > {start} and timestamp < {end} and type="obs" and subtype="cal"')
    obslist = []
    for iobs in obslist_all:
        if context.obsdb.get(iobs['obs_id'], tags=True)['tags'][0] in configs['query']['tags']:
            obslist.append(iobs['obs_id'])
    del obslist_all

    # check errorlogpath is exist
    errlogdir = os.path.dirname(configs['errlogpath'])
    if not os.path.exists(errlogdir):
        os.makedirs(errlogdir, exist_ok=True)
        logger.info(f"Database dir does not exist, so created database dir: {errlogdir}")

    # Make runlist for each observation/wafer/band
    runlist = []
    for obs_id in obslist:
        # find targeted wafers in scan
        obs = context.obsdb.get(f'{obs_id}', tags=True)
        tags = obs['tags']
        subs = 'ws'
        if configs['query'].get('all_wafers', False):
            obs_wafers = obs['wafer_slots_list']
        else:
            specific_wafers = configs['query'].get('specific_wafers')
            if specific_wafers is not None:
                obs_wafers = [f'ws{i}' for i in specific_wafers]
            else:
                obs_wafers = [i for i in tags if subs in i]
        
        bands = configs['query'].get('bands', bands)
        for band in bands:
            for wafer in obs_wafers: 
                if configs['overwrite']:
                    irunlist = {'obs_id':obs_id, 'wafer_info': {'wafer_slot': wafer, 'wafer.bandpass': band}}
                    runlist.append(irunlist)
                else:
                    if not os.path.exists(configs['dbpath']):
                        irunlist = {'obs_id':obs_id, 'wafer_info': {'wafer_slot': wafer, 'wafer.bandpass': band}}
                        runlist.append(irunlist)
                    else:
                        idb = planet_mapmaker.get_db(configs['dbpath'], obs_id=obs_id, wafer=wafer, freq_channel=band)
                        if not bool(idb):
                            irunlist = {'obs_id':obs_id, 'wafer_info': {'wafer_slot': wafer, 'wafer.bandpass': band}}
                            runlist.append(irunlist)
                        else:
                            logger.info(f"Observation {obs_id}, wafer {wafer}, band {band} already processed. Skipping.")

    n_runs = len(runlist)
    logger.debug(f'Runlist: {runlist}')
    logger.info(f'Found {n_runs} observations to analyze')
    
    logger.debug('Parallelizing the map making work')
    future_to_rl = {executor.submit(planet_mapmaker.planet_mapmake_eachobs, config_path=config_path, obs_id = rl['obs_id'], 
                            wafer_info = rl['wafer_info'], verbosity = verbosity, debug = debug): rl for rl in runlist}
    futures = list(future_to_rl)

    n = 0
    logger.debug('Parallelization completed. Summarizing')
    for future in as_completed_callable(futures):
        rl = future_to_rl[future]
        try:
            n += 1
            logger.info(f'Processing results {n}/{n_runs}')
            dbinfo = future.result()
            planet_mapmaker.save_info(dbinfo, dbpath = configs['dbpath'])
            futures.remove(future)
            logger.info(f'Processing Finished correctly {n}/{n_runs}')
        except Exception as e:
            future_write_to_log(e, configs['errlogpath'], rl=rl)
            futures.remove(future)
            logger.info(f'Processing Failed somehow {n}/{n_runs}')
            continue
    

def cli_main(config_file: str, nprocs: int):
    rank, executor, as_completed_callable = get_exec_env(nprocs)
    if rank == 0:
        main(config_file, executor, as_completed_callable)


def get_parser(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        p = ArgumentParser()
    else:
        p = parser
    p.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    p.add_argument(
        "--nprocs", type=int, help="Number of processors to use."
        )
    return p

if __name__ == '__main__':
    main_launcher(cli_main, get_parser)