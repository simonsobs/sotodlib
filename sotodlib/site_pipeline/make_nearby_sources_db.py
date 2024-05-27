import os
import yaml
import time
import argparse
import logging
from typing import Optional
import numpy as np
from sotodlib import core
from sotodlib.coords import optics
import sotodlib.coords.planets as planets

from sotodlib.site_pipeline import util
logger = util.init_logger('make_cosamp_hk', 'make_cosamp_hk: ')

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()    
    parser.add_argument(
        'config',
        help="configuration yaml file.")
    parser.add_argument(
        '-o', '--output_dir', action='store', default=None, type=str,
        help='output data directory, overwrite config output_dir')
    parser.add_argument(
        '--verbose', default=2, type=int,
        help="increase output verbosity. \
        0: Error, 1: Warning, 2: Info(default), 3: Debug")
    parser.add_argument(
        '--overwrite',
        help="If true, overwrites existing entries in the database",
        action='store_true',
    )
    parser.add_argument(
        '--query_text', type=str,
        help="Query text",
    )
    parser.add_argument(
        '--query_tags', nargs="*", type=str,
        help="Query tags",
    )
    parser.add_argument(
        '--min_ctime', type=int,
        help="Minimum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--max_ctime', type=int,
        help="Maximum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--obs_id', type=str,
        help="obs_id of particular observation if we want to run on just one",
    )
    parser.add_argument(
        '--update-delay',
        help="Number of days (unit is days) in the past to start observation list.",
        type=int
    )
    return parser

def main(config: str,
        output_dir: Optional[str] = None,
        verbose: Optional[int] = 2,
        overwrite: Optional[bool] = False,
        query_text: Optional[str] = None,
        query_tags: Optional[list] = None,
        min_ctime: Optional[float] = None,
        max_ctime: Optional[float] = None,
        obs_id: Optional[str] = None,
        update_delay: Optional[str] = None,
        ):
    """
    Args:
        config: str
            Path to the configuration file containing processing parameters.
        output_dir: str or None
            Directory where the processed data will be saved. \
            If not provided, it is extracted from the configuration file.
        verbose: str
            Verbosity level for logging. Default is 2 (INFO).
        overwrite: bool
            Whether to overwrite existing output files. Default is False.
        query_text: str or None
            Text-based query to filter observations. If not provided, it is extracted from the configuration file.
        query_tags: list or None
            Tags used for further filtering observations. If not provided, it is extracted from the configuration file.
        min_ctime: float or None
            Minimum timestamp of observations to be queried. If not provided, it is extracted from the configuration file.
        max_ctime: float or None
            Maximum timestamp of observations to be queried. If not provided, it is extracted from the configuration file.
        obs_id: str or None
            Specific observation obs_id to process. If provided, overrides other filtering parameters.
        update_delay: str or None
            Number of days (unit is days) in the past to start observation list.
    """
    # Set verbose
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.INFO)
    elif verbose == 3:
        logger.setLevel(logging.DEBUG)
    
    # load configuration file
    if type(config) == str:
        with open(config) as f:
            config = yaml.safe_load(f)
    
    # load context
    context_file = config['context_file']
    ctx = core.Context(context_file)
    
    if output_dir is None:
        if 'output_dir' in config.keys():
            output_dir = config['output_dir']
        else:
            raise ValueError('output_dir is not specified.')
    if query_text is None:
        if 'query_text' in config.keys():
            query_text = config['query_text']
    if query_tags is None:
        if 'query_tags' in config.keys():
            query_tags = config['query_tags']
    if min_ctime is None:
        if 'min_ctime' in config.keys():
            min_ctime = config['min_ctime']
    if max_ctime is None:
        if 'max_ctime' in config.keys():
            max_ctime = config['max_ctime']
    if update_delay is None:
        if 'update_delay' in config.keys():
            update_delay = config['update_delay']
    
    if (min_ctime is None) and (update_delay is not None):
        min_ctime = int(time.time()) - update_delay*86400
        logger.info(f'min_ctime: {min_ctime}')
    
    source_list = config.get('source_list', None)
    source_names = []
    for _s in source_list:
        if isinstance(_s, str):
            source_names.append(_s)
        elif len(_s) == 3:
            source_names.append(_s[0])
        else:
            raise ValueError('Invalid style of source')
    distance = config.get('distance', 1.)
    
    # Other configurations
    telescope = config.get('telescope', 'SAT')
    if telescope == 'SAT':
        wafer_slots = [f'ws{i}' for i in range(7)]
    elif telescope == 'LAT':
        wafer_slots = [f'ws{i}' for i in range(3)]
    else:
        raise NameError('Only "SAT" or "LAT" is supported.')
    optics_config = config.get('optics_config', None)
    
    # Load metadata sqlite
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    man_db_filename = os.path.join(output_dir, f'nearby_sources.sqlite')
    if os.path.exists(man_db_filename):
        logger.info(f"Mapping {man_db_filename} for the \
                    archive index.")
        man_db = core.metadata.ManifestDb(man_db_filename)
    else:
        logger.info(f"Creating {man_db_filename} for the \
                     archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        for source_name in source_names:
            scheme.add_data_field(source_name)
            for ws in wafer_slots:
                scheme.add_data_field(f'{source_name}_{ws}')
        man_db = core.metadata.ManifestDb(man_db_filename, scheme=scheme)
    
    # query observations
    if obs_id is not None:
        tot_query = f"obs_id=='{obs_id}'"
    else:
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
    
    # output cosampled data for each obs_id
    for obs in obs_list:
        obs_id = obs['obs_id']
        h5_filename = f"nearby_sources_{obs_id.split('_')[1][:4]}.h5"
        output_filename = os.path.join(output_dir, h5_filename)
        entry = {'obs:obs_id': obs_id, 'dataset': obs_id}
        streamed_wafer_slots  = ['ws{}'.format(index) for index, bit in enumerate(obs_id.split('_')[-1]) if bit == '1']
        try:
            for ws in wafer_slots:
                if ws not in streamed_wafer_slots:
                    for source_name in source_names:
                        entry[f'{source_name}_{ws}'] = 0                    
                else:
                    meta = ctx.get_meta(obs_id, dets={'wafer_slot': ws})
                    _aman = ctx.get_obs(obs_id, dets=[])
                    aman = core.AxisManager(meta.dets, _aman.samps)
                    aman.wrap('timestamps', _aman.timestamps, [(0, 'samps')])
                    aman.wrap('boresight', _aman.boresight)
                    
                    if not 'focal_plane' in list(meta._fields.keys()):
                        optics.get_focal_plane(aman, 
                                               x = np.zeros(aman.dets.count),
                                               y = np.zeros(aman.dets.count),
                                               wafer_slot=ws,
                                               config_path=optics_config)
                    else:
                        aman.wrap('focal_plane', meta.focal_plane)                        
                    flag_fp_isnan = (np.isnan(aman.focal_plane.xi)) | \
                                    (np.isnan(aman.focal_plane.eta)) | \
                                    (np.isnan(aman.focal_plane.gamma))
                    aman.restrict('dets', aman.dets.vals[~flag_fp_isnan])

                    nearby_sources = planets.get_nearby_sources(aman, source_list=source_list, distance=distance)
                    nearby_source_names = []
                    for _source in nearby_sources:
                        nearby_source_names.append(_source[0])
                    for source_name in source_names:
                        entry[f'{source_name}_{ws}'] = 1 if source_name in nearby_source_names else 0
            
            for source_name in source_names:
                hit_any = False
                for ws in wafer_slots:
                    if entry[f'{source_name}_{ws}'] == 1:
                        hit_any = True
                entry[f'{source_name}'] = 1 if hit_any else 0
            
            man_db.add_entry(entry, 
                     filename=output_filename, replace=overwrite)
            logger.info(f"saved: {obs_id}, {ws}")

        except Exception as e:
            logger.error(f"Exception '{e}' thrown while processing {obs_id}")
            continue

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
    