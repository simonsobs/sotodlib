'''This module receives the fitted detector optical response 
from the analyze-bright-ptsrc after the detmap has been applied, 
and determines a 'flat field' for the detectors. The script can 
take one single scan or multiple scans. The final output is a 
set of per-detector calibration factors of order 1 associated 
with det_id, to be multiplied to the detector response. 
'''
import sys
import numpy as np
import pandas as pd
import os
import yaml
import h5py
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sotodlib.core import Context, metadata
import sotodlib.io.metadata as io_meta
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__)

def get_parser():

    parser = ArgumentParser()

    parser.add_argument('config_file', help=
        'Configuration file.')
 
    parser.add_argument('--obs_id', nargs='*', help=
         'Observations to summarize over.')

    parser.add_argument('--obs_id_file', help=
        'Path to a txt that stores obs_ids.')

    parser.add_argument('-w', '--wafer', help=
         'The wafer or wafers to run model on.')

    parser.add_argument('-q', '--obs_query', help = 
        'String to pass to ctx.obsdb.query()')

    parser.add_argument('-v', '--verbose', action= 'count',
        default = 0, help ='Pass multiple times to increase.')

    parser.add_argument('--make_plot', action = 'store_true',
        help = 'Plot response over time.')

    parser.add_argument('--save_plot', action = 'store_true',
        help = 'Save plots to output directory.')

    parser.add_argument('--overwrite', action = 'store_true',
        help = 'Overwrites existing entries.')


    return parser


def read_store_data(ctx, obs_ids, value, wafers, min_cut = None, max_cut = None):
    logger.info(f'Start data loading.')

    if not wafers:
        logger.info(f'Load all wafers.')
        find_wafer = True
    else: 
        logger.info(f'Load wafer {wafers}.')
        find_wafer = False

    # Store data in dict[wafer][band] = df['det_id','amps1','amps2',...]
    amp_dict = {}
    noise_dict = {}
    plot_dict = {}

    for obs_id in obs_ids:
        logger.info(f'Loading observation {obs_id}.')

        aman = ctx.get_meta(obs_id = obs_id)
        target = aman.obs_info.target
        timestamp = aman.obs_info.timestamp

        if find_wafer:
            wafers = set(aman.det_info.wafer_slot)

        for wafer in wafers:


            wafer_mask = (aman.det_info.wafer_slot== wafer)
            bands = aman.det_info.band[wafer_mask]
            det_id = aman.det_info.det_id[wafer_mask]
            amp = aman.ptsrc_params.amp[wafer_mask]
            fwhm1 = aman.ptsrc_params.fwhm_xi[wafer_mask]
            fwhm2 = aman.ptsrc_params.fwhm_eta[wafer_mask]

            if value =='Peak': 
                cal_val = amp
            elif value == 'Flux': 
                cal_val = amp*fwhm1*fwhm2
            else:
                logger.error('Wrong value specified: Peak or Flux')
                sys.exit(1)

            noise = cal_val/aman.ptsrc_params.snr[wafer_mask]


            for band in set(bands):
                plot_dict.setdefault(target,{}).setdefault(band,{}).setdefault(wafer,{})[obs_id] = timestamp

                mask = (bands == band)
                if min_cut:
                    mask = mask & (min_cut < cal_val)
                if max_cut:
                    mask = mask & (cal_val < max_cut)
                                   
                det_per_band = det_id[mask]
                cal_per_band = cal_val[mask]
                noise_per_band = noise[mask] 
                amp_dict[wafer][band] = amp_dict.setdefault(wafer,{}).setdefault(band,pd.DataFrame({'det_id':{}})).merge(
                    pd.DataFrame({'det_id':det_per_band, obs_id:cal_per_band}), how = 'outer')
                
                noise_dict[wafer][band] = noise_dict.setdefault(wafer,{}).setdefault(band,pd.DataFrame({'det_id':{}})).merge(
                    pd.DataFrame({'det_id':det_per_band, obs_id:noise_per_band}), how = 'outer')

    logger.info(f'Data loading completed for a total of {len(amp_dict)} wafers.')
    return amp_dict, noise_dict, plot_dict

def calibrate_data(amp_dict, noise_dict, method, 
    max_obs_residue = np.inf, max_error_ratio = np.inf, min_obs_per_det =1):

    logger.info("Start calibration. ")
    factor_dict = {}
    for wafer in amp_dict:
        for band,item in amp_dict[wafer].items():
            obs_ids = item.columns[1:].values
            num_obs = len(obs_ids)
            assert num_obs>0

            if num_obs==1: 
                logger.warning(f'{wafer} {band} has only 1 observation.')

            # Normalize all observations if multiple observations exist
            if num_obs >1: 
                logger.info(f'Normalizing {wafer} {band} over {num_obs} obsrvations')
                # Select the best observation: highest yield, highest median amp, lowest snr
                df_select = pd.DataFrame(columns = ['obs_id','yield','amp','noise'])

                for obs_id in obs_ids: 
                    df_select=pd.concat([df_select,pd.DataFrame([{'obs_id':obs_id,'yield':item[obs_id].count(),
                                            'amp':item[obs_id].median(), 
                                            'noise':noise_dict[wafer][band][obs_id].median()}])],
                                            ignore_index = True)

                best_obs=df_select.sort_values(by=['yield','amp','noise'],ascending=[False, False, True])['obs_id'][0]

                for obs_id in obs_ids:
                    if obs_id==best_obs: continue
                    fit_mask = (~np.isnan(item[obs_id]))&(~np.isnan(item[best_obs]))

                    x = item[obs_id][fit_mask].values
                    y = item[best_obs][fit_mask].values

                    # Scale all observations to be the same amplitude
                    a, _, _, _ = np.linalg.lstsq(np.transpose([x]), y,rcond=None)
                    mean_residue = np.mean(y - x*a)

                    # There should not be any intercepts
                    if mean_residue > max_obs_residue:
                        logger.warning(f'{obs_id}: {wafer} {band} fails normalization')
                        item[obs_id] = [np.nan]*len(item)
                    else:
                        item[obs_id] *= a
                        noise_dict[wafer][band][obs_id] *= a


            cal_val = amp_dict[wafer][band].iloc[:,1:]
            noise = noise_dict[wafer][band].iloc[:,1:]

            error = noise.values.flatten()

            if method == 'Mean':
                average = cal_val.mean(axis =1)
                if num_obs>1: error = cal_val.std(axis = 1)

            elif method == 'Median':
                average = cal_val.median(axis =1)
                if num_obs>1: 
                    # Using median absolute deviation
                    error = cal_val.subtract(cal_val.median(axis = 1), axis = 0).abs().median(axis =1)

            elif method == 'Weighted_average':
                weights = 1/(noise**2)
                average = (cal_val * weights).sum(axis = 1)/weights.sum(axis = 1)
                error = np.sqrt((1/weights).sum(axis = 1))
            else:
                logger.error('Wrong method specified: Median, Mean or Weighted_average')
                sys.exit(1)

            logger.info(f'{wafer} {band}: average response of {average.mean():.2e} and error {error.mean():.2e}.')
            mask = (error/average <= max_error_ratio) & (cal_val.count(axis = 1)>= min_obs_per_det)

            factor_dict.setdefault(wafer,{})[band] = np.array(list(zip(item['det_id'][mask].values,
                                                                       (average.median()/average)[mask].values,
                                                                       ((average.median()/ (average ** 2)) * error)[mask].values)),
                                                              dtype=[('dets:det_id', '<U30'), ('cal_relcal', '<f4'), ('error_relcal', '<f4')])
    return factor_dict

def make_plots(amp_dict, plot_dict, value, save_plot = False, output_dir = None):

    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        logger.info(f"Saving plots to {plot_dir}")
    
    for source in plot_dict:
        for b in plot_dict[source]:
            plt.figure()
            for w in sorted(plot_dict[source][b].keys()):
                item = plot_dict[source][b][w]
                obs_ids = list(item.keys())
                timestamps = [int(item[key]) for key in item]
                plt.scatter(timestamps, (amp_dict[w][b][obs_ids].mean().values), label = w, alpha = 0.5)
            plt.grid()
            plt.legend()
            plt.xlabel("Timestamp")
            plt.ylabel(value)
            plt.title(f"{source} scans : Mean {value} response for {b}")
            if save_plot: plt.savefig(os.path.join(plot_dir, f"{source}_{b}_det_relcal.png"))
            plt.show()


    for source in plot_dict:
        for b in plot_dict[source]:
            plt.figure()
            alpha = 1
            s = 50 
            for w in sorted(plot_dict[source][b].keys()):
                item = plot_dict[source][b][w]
                obs_ids = list(item.keys())
                responses = amp_dict[w][b][obs_ids].values
                relative_response = responses/np.nanmedian(responses, axis = 0)
                timestamps = np.array([int(item[key]) for key in item]* len(responses))
                plt.scatter(timestamps.flatten(), relative_response.flatten(), s= s, label = w, alpha = alpha)
                # Asjust alpha and dotsize to prevent overlapping
                alpha *=0.7
                s *= 0.7
            plt.grid()
            plt.legend()
            plt.yscale('log')
            plt.xlabel("Timestamp")
            plt.ylabel(f"det_response / median_wafer_response")
            plt.title(f"{source} scans\ndet_response / median_wafer_response for {b}")
            if save_plot: plt.savefig(os.path.join(plot_dir, f"{source}_{b}_mean_response.png"))
            plt.show()


def main(args=None):

    args = get_parser().parse_args(args)

    # Get cofig
    config = yaml.safe_load(open(args.config_file, 'r'))
    ctx = Context(config['context_file'])
    overwrite = args.overwrite
    value = config['value']
    method = config['method']
    output_dir = config['archive']['policy']['out_dir']
    verbose = args.verbose
    fname = os.path.join(output_dir, 'relcal_model.h5')

    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f'Calibrating with the {method} of {value}...')

    # Create ManifestDb
    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = metadata.ManifestDb(config['archive']['index'])
    else: 
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('dets:wafer_slot')
        scheme.add_exact_match('dets:band')
        scheme.add_data_field('dataset')
        scheme.add_data_field('config')
        db = metadata.ManifestDb(config['archive']['index'], scheme=scheme)

    min_cut = config.setdefault('min_cut', None)
    max_cut = config.setdefault('max_cut', None)
    max_obs_residue = config.setdefault('max_obs_residue', np.inf)
    max_error_ratio = config.setdefault('max_error_ratio', np.inf)
    min_obs_per_det = config.setdefault('min_obs_per_det', 1)

    # Get list of observations

    if args.obs_id_file: obs_id_file = args.obs_id_file
    elif 'obs_id_file' in config: obs_id_file = config['obs_id_file']
    else: obs_id_file = None

    if args.obs_id:
        logger.info(f'Loading obs_ids from inputs.')
        obs_ids = args.obs_id
        if isinstance(obs_ids, str): obs_ids = [obs_ids]

    elif args.obs_query: 
        obs_ids = ctx.obsdb.query(args.obs_query)['obs_id'].tolist()

    elif obs_id_file:
        logger.info(f'Loading obs_ids from file {obs_id_file}.')
        with open(obs_id_file,'r') as f:
            obs_ids = [line.strip() for line in f]
    else:
        logger.error('No obs_ids specified.')
        sys.exit(1)

    if len(obs_ids)== 0: 
        logger.error('No obs_ids found.')
        sys.exit(1)


    # Find all wafers if wafer is not specified
    if args.wafer:
        wafers = args.wafer
    elif 'wafer' in config: 
        wafers = config['wafer']
        if isinstance(wafers, str): wafers = [wafers]
    else:
        wafers = []

    # Load all the data and store relevant parts
    amp_dict, noise_dict, plot_dict = read_store_data(ctx, obs_ids, value, wafers,min_cut,max_cut)


    if args.make_plot: 
        make_plots(amp_dict, plot_dict, value, args.save_plot, output_dir)

    # Calculate relcal factor
    factor_dict = calibrate_data(amp_dict, noise_dict, method,
        max_obs_residue, max_error_ratio , min_obs_per_det)


    for wafer in factor_dict:
        for band in factor_dict[wafer]:
            dataset = f'relcal_{wafer}_{band}'
            logger.info(f"Writing to dataset {dataset}.")

            # Write to h5 and ManifestDb
            rs = metadata.ResultSet.from_friend(factor_dict[wafer][band])
            
            io_meta.write_dataset(rs, fname, dataset, overwrite = True) 

            # The obs_id used is stored in the attributes. 
            with h5py.File(fname, 'r+') as h:
                h[dataset].attrs['obs_id'] = obs_ids

            db.add_entry({'dets:wafer_slot':wafer, 'dets:band': band,'dataset': dataset, 'config': args.config_file}, 
                fname, replace = overwrite)


if __name__ == '__main__':
    main()