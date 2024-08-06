import os
import yaml
import time
import numpy as np
import argparse
import logging
from typing import Optional

from sotodlib import core
from sotodlib.io import hk_utils
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
    
    full_fields = config['fields']
    aliases = config['aliases']
    data_fields = [full_field.split('.')[1] for full_field in full_fields]
            
    save_mean = config.get('save_mean', False)
    save_median = config.get('save_median', False)
    save_rms = config.get('save_rms', False)
    save_ptp = config.get('save_ptp', False)
    
    min_valid_value = config.get('min_valid_value', None)
    max_valid_value = config.get('max_valid_value', None)
    max_valid_dvalue_dt = config.get('max_valid_dvalue_dt', None)
    if np.all(np.array([min_valid_value is None,
                        max_valid_value is None,
                        max_valid_dvalue_dt is None])):
        save_nan_fraction = False
    else:
        save_nan_fraction = True

    if (min_ctime is None) and (update_delay is not None):
        min_ctime = int(time.time()) - update_delay*86400
    logger.info(f'min_ctime: {min_ctime}')
    
    # Load metadata sqlite
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_prefix = config['output_prefix']
    man_db_filename = os.path.join(output_dir, f'{output_prefix}.sqlite')
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
           
        if save_mean:
            scheme.add_data_field(f'{output_prefix}_mean')
        if save_median:
            scheme.add_data_field(f'{output_prefix}_median')
        if save_rms:
            scheme.add_data_field(f'{output_prefix}_rms')
        if save_ptp:
            scheme.add_data_field(f'{output_prefix}_ptp')
        if save_nan_fraction:
            scheme.add_data_field(f'{output_prefix}_nan_fraction')
        
        man_db = core.metadata.ManifestDb(
            man_db_filename,
            scheme=scheme
        )
        
    
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
        try:
            meta = ctx.get_meta(obs_id)
            aman = ctx.get_obs(obs_id, dets=[])
        except Exception as e:
            logger.error(f"Exception \n '{e}' \n thrown while aman loading of {obs_id}. Skipped")
            continue
            
        try:
            _hkman = hk_utils.get_detcosamp_hkaman(aman, 
                                      fields = config['fields'],
                                      data_dir = config['input_dir'])
            dt_buffer = 60
            start = float(aman.timestamps[0] - dt_buffer)
            stop = float(aman.timestamps[-1] + dt_buffer)
            _data = hk_utils.sort_hkdata(start=start, stop=stop, fields=config['fields'],
                                         data_dir=config['input_dir'], alias=None)
            _hkman = hk_utils.make_hkaman(grouped_data=_data, alias_exists=False,
                                          det_cosampled=True, det_aman=aman)
            hkman = core.AxisManager(meta.dets, aman.samps)
            hkman.wrap('timestamps', aman.timestamps, [(0, 'samps')])
            
            num_different_field = 0
            for i, (data_field, alias) in enumerate(zip(data_fields, aliases)):
                if i > 0:
                    num_different_field += int(data_fields[i-1] != data_field)
                i_in_same_field = i - num_different_field
                hkman.wrap(alias, _hkman[data_field][data_field][i_in_same_field], [(0, 'samps')])
            # set values nan if it is invalid
            mask_invalid = np.zeros(hkman.samps.count, dtype=bool)
            if min_valid_value is not None:
                mask_invalid[hkman[aliases[0]] < min_valid_value] = True
            if max_valid_value is not None:
                mask_invalid[max_valid_value < hkman[aliases[0]]] = True
            if max_valid_dvalue_dt is not None:
                dt = np.median(np.diff(aman.timestamps))
                mask_invalid[ np.abs( np.diff(np.append(hkman[aliases[0]], hkman[aliases[0]][-1])) / dt) > max_valid_dvalue_dt ] = True
                mask_invalid[ np.abs( np.diff(np.append(hkman[aliases[0]][0],  hkman[aliases[0]])) / dt) > max_valid_dvalue_dt ] = True
            if np.any(mask_invalid):
                hkman[aliases[0]][mask_invalid] = np.nan
                
        except Exception as e:
            logger.warning(f"Exception \n '{e}' \n thrown while hk loading of {obs_id}. Making dummy hkaman")
            hkman = core.AxisManager(meta.dets, aman.samps)
            hkman.wrap('timestamps', aman.timestamps, [(0, 'samps')])
            num_different_field = 0
            for i, (data_field, alias) in enumerate(zip(data_fields, aliases)):
                if i > 0:
                    num_different_field += int(data_fields[i-1] != data_field)
                i_in_same_field = i - num_different_field
                hkman.wrap(alias, np.nan*np.zeros(hkman.samps.count), [(0, 'samps')])
            
        try:
            h5_filename = f"{output_prefix}_{obs_id.split('_')[1][:4]}.h5"
            output_filename = os.path.join(output_dir, h5_filename)
            hkman.save(dest=output_filename, group=obs_id, overwrite=overwrite)
            
            entry_dict = {'obs:obs_id': obs_id, 'dataset': obs_id}
            if save_mean:
                entry_dict[f'{output_prefix}_mean'] = np.nanmean(hkman[aliases[0]])
            if save_median:
                entry_dict[f'{output_prefix}_median'] = np.nanmedian(hkman[aliases[0]])
            if save_rms:
                entry_dict[f'{output_prefix}_rms'] = np.nanstd(hkman[aliases[0]])
            if save_ptp:
                entry_dict[f'{output_prefix}_ptp'] = np.nanmax(hkman[aliases[0]]) - np.nanmin(hkman[aliases[0]])
            if save_nan_fraction:
                entry_dict[f'{output_prefix}_nan_fraction'] = np.mean(np.isnan(hkman[aliases[0]]).astype('float'))
            man_db.add_entry(entry_dict, filename=output_filename, replace=overwrite)
            logger.info(f"saved: {obs_id}")
        except Exception as e:
            logger.error(f"Exception \n '{e}' \n thrown while saving of {obs_id}. Skipped")
            continue
            
            


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
    