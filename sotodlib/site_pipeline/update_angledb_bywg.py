import numpy as np
import matplotlib.pyplot as plt 
import yaml
import os 
import argparse

from sotodlib import core, tod_ops
import sotodlib.io.load_smurf as load_smurf
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.site_pipeline.util as sp_util
import sotodlib.io.metadata as io_meta
import sotodlib.hwp.hwp as hwp
from so3g.hk import load_range, HKArchiveScanner

from sotodlib.wiregrid.wiregrid_analysis import _get_config,wg_demod_tod,get_wg_angle,fit_angle


def get_parser() : 
    
     parser = argparse.ArgumentParser(description='Analyze wiregrid data')
     parser.add_argument(
        '-c', '--config_file', default=None, type=str, required=True,
        help="Configuration File")
     parser.add_argument(
        '-o', '--obs_id', default=None, type=str, required=False,
        help="obs_id")
     parser.add_argument(
        '-w', '--overwrite', default=True, type=bool, required=False,
        help="If you want overwrite db or not")
     parser.add_argument(
        '-s', '--min_ctime', default=None, type=int, required=False,
        help="start ctime")
     parser.add_argument(
        '-e', '--max_ctime', default=None, type=int, required=False,
        help="end ctime")

     return parser


if __name__ == '__main__':

    # Get argument
    parser = get_parser()
    args = parser.parse_args()

    config_file = args.config_file
    if args.obs_id is not None : obs_id = args.obs_id
    else : obs_id = None
    overwrite = args.overwrite
    if (args.min_ctime is not None) and (args.max_ctime is not None) : 
        min_ctime = args.min_ctime
        max_ctime = args.max_ctime

    verbose = 0
    # set logger
    logger = sp_util.init_logger(__name__, 'wg_calibration: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    # load config file
    config = _get_config(config_file)

    # load context file
    context = core.Context(config['context_file'])
    obsdb = context.obsdb

    if obs_id is not None :
        obs = obsdb.query(f'obs_id == "{obs_id}"')[0]
    else :
        tot_query = f"timestamp<={max_ctime} and timestamp>={min_ctime}"
        obs = obsdb.query(tot_query)[0]

    print(obs)

    # place of house keeping data
    hk_dir = config['hk_dir']
    output_h5 = config['archive']['policy']['filename']

    # axis manager
    meta = context.get_meta(obs_id)
    aman = context.get_obs(meta)
    print("aman generated")

    # wiregrid angle has a offset against the zero of encoder
    # offset should be given in config file
    _ang_offset = float(config['wg_angle_offset'])

    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = core.metadata.ManifestDb(config['archive']['index'])
    else:
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)


    fitting_results = core.metadata.ResultSet(
        keys=["dets:readout_id","amplitude","amplitude_e","angle","angle_e","offset_q","offset_u","chi2"]
    )

    ### Wrap wg angle info
    wg_max_enc = 52000.
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(float(aman.timestamps[0]), float(aman.timestamps[-1]), wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/wg_max_enc*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)
    print("wg_aman is wrapped")

    ### Demodulate TOD
    utils.load_hwp_data(aman, config_file)


    wg_demod_tod(aman)
    print("Demod done")
    num_angle, wg_info = get_wg_angle(aman,ang_offset=_ang_offset,debug=False)
    print("fitting now")
    fitting_results = fit_angle(aman,fitting_results,wg_info,debug=False,rotate_tau_wg=False, rotate_tau_tune=False)

    # Save outputs
    db_data = {'obs:obs_id': obs_id, 'dataset' : f'wg_fitting_{obs_id}'}
    db.add_entry(db_data, output_h5, f'{obs_id}',replace=True)
    db.to_file(config['archive']['index'])
    io_meta.write_dataset(fitting_results, output_h5, f'{obs_id}', overwrite=overwrite)
