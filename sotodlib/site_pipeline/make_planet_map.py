# To make atmic planet map in detector-centered coordinate system
import yaml, os, time, datetime, traceback
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
    verbosity: int,
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable) -> None:

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
            obs_wafers = [f'ws{i}' for i in range(7)]
        else:
            specific_wafers = configs['query'].get('specific_wafers')
            if specific_wafers is not None:
                obs_wafers = [f'ws{i}' for i in specific_wafers]
            else:
                obs_wafers = [i for i in tags if subs in i]
        
        bands = configs['query'].get('bands', bands)
        for band in bands:
            # add function for overwrite
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
    main_launcher(main, get_parser)