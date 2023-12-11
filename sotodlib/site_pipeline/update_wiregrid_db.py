import argparse                                                                                                                                                                                                                                            
import os
import yaml

from sotodlib import core
import sotodlib.io.metadata as io_meta
import sotodlib.io.g3tsmurf_utils as utils
from sotodlib.site_pipeline import util
from sotodlib.wiregrid import angle_fitter as af

def get_parser() : 

    parser = argparse.ArgumentParser(description='Analyze wiregrid data')

    parser.add_argument(
        '-c', '--config_file', default=None, type=str, required=True,
        help="Configuration File")
    parser.add_argument(
        '-o', '--obs_id', default=None, type=str, required=True,
        help="obs_id")
    parser.add_argument(
        '-v', '--verbose', default=0, type=int, required=False,
        help="verbose of logger")
    parser.add_argument(
        '-ow', '--overwrite', default=True, type=bool, required=False,
        help="wheather overwrite db or not")

    return parser 


def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def main(config_file=None, obs_id=None, verbose=0, overwrite=True):

    # set logger
    logger = util.init_logger(__name__, 'wg_calibration: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    # load config file
    config = _get_config(config_file)

    # load context file
    ctx = core.Context(config['context_file'])

    # place of house keeping data
    hk_dir = config['hk_dir']

    # output file
    _output_h5 = config['archive']['policy']['filename']

    # wiregrid angle has a offset against the zero of encoder
    # offset should be given in config file
    _ang_offset = float(config['wg_angle_offset'])
    _axis_offset = float(config['wg_axis_offset'])

    groups = ctx.obsfiledb.get_detsets(obs_id)
    print(groups)
    for group in groups :
        print(group)def main(config_file=None, obs_id=None, verbose=0, overwrite=True):

    #parser = get_parser()
    #args = parser.parse_args()

    # set logger
    logger = util.init_logger(__name__, 'wg_calibration: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    # load config file
    config = _get_config(config_file)

    # load context file
    ctx = core.Context(config['context_file'])

    # place of house keeping data
    hk_dir = config['hk_dir']

    # output file
    _output_h5 = config['archive']['policy']['filename']

    # wiregrid angle has a offset against the zero of encoder
    # offset should be given in config file
    _ang_offset = float(config['wg_angle_offset'])
    _axis_offset = float(config['wg_axis_offset'])

    groups = ctx.obsfiledb.get_detsets(obs_id)
    for group in groups :

        aman = ctx.get_obs(obs_id, dets={"detset": group})
        wafern = aman.dets.vals[0].split("_")[2]
        output_h5 = _output_h5.replace("wafer",wafern)
        
        if os.path.exists(config['archive']['index']):
            logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
            db = core.metadata.ManifestDb(config['archive']['index'])
        else:
            logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
            scheme = core.metadata.ManifestScheme()
            scheme.add_exact_match('obs:obs_id')
            scheme.add_data_field('group')
            scheme.add_data_field('dataset')
            db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)
    
        fitting_results = core.metadata.ResultSet(
            #keys=["dets:readout_id","amplitude","amplitude_e","angle","angle_e","offset_q","offset_u","chi2"]
            keys=["dets:readout_id","cal",]
        )
        
        ### Wrap wg angle info 
        af._wrap_wg_angle_info(aman, config)
        
        
        ### Demodulate TOD
        utils.load_hwp_data(aman, config_file)
        af._wg_demod_tod(aman)
        
        ### Get wg info
        wg_info = af._get_wg_angle(aman,ang_offset=_ang_offset,axis_offset=_axis_offset)
    
        ### Fit 
        fitting_results = af.fit_angle(aman,fitting_results,wg_info)
        
        # Save outputs
        db_data = {'obs:obs_id': obs_id, 'dataset' : f'wg_fitting_{obs_id}', 'group' : group}
        db.add_entry(db_data, output_h5, f'{obs_id}',replace=True)
        db.to_file(config['archive']['index'])
        io_meta.write_dataset(fitting_results, output_h5, f'{obs_id}', overwrite=overwrite)


if __name__ == '__main__':
    util.main_launcher(main, get_parser)           
