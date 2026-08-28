"""This module combines multiple measurements of detector time constants
from analyze-bright-ptsrc to produce a time constant model. For first 
light, the model is simply the average measured value. This script works 
with dets:det_id and assumes that channel map has been applied. The 
final result is a time constant indexed by wafer, with associated obs_id 
stored in the dataset attributes. There's an option to 
make this number per wafer, per biasline or per detector.
"""


import numpy as np
import os
import yaml
import h5py
import scipy
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sotodlib.core import Context, metadata
import sotodlib.io.metadata as io_meta
from sotodlib.site_pipeline import util


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('config_file', help=
        "Configuration file.")

    parser.add_argument('-obs_id', nargs='*', help=
        "Observations to summarize over.")

    parser.add_argument('--obs_id_file', help=
        'Path to a txt that stores obs_ids.')

    parser.add_argument('-w', '--wafers', help=
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


def make_plots(plot_dict, min_cut, max_cut, save_plot = False, output_dir = None):
    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        logger.info(f"Saving plots to {plot_dir}")

    val_range = { band: {'min' : min([plot_dict[band][wafer]['data'].iloc[:,1:].min().min() for wafer in plot_dict[band]]),
                         'max' : max([plot_dict[band][wafer]['data'].iloc[:,1:].max().max() for wafer in plot_dict[band]])}
                         for band in plot_dict}
    for band in val_range:
        if min_cut: val_range[band]['min'] = max(val_range[band]['min'], min_cut)
        if max_cut: val_range[band]['max'] = min(val_range[band]['max'], max_cut)

    for band in plot_dict:
        plt.figure()

        for wafer in sorted(plot_dict[band].keys()):
            plt.hist(plot_dict[band][wafer]['data'].iloc[:,1:].mean(axis = 1), 
                range = (val_range[band]['min'],val_range[band]['max']), bins = 100, label = f'{wafer} {band}', alpha = 0.5)
        plt.grid()
        plt.legend()
        plt.xlabel("Averaged timeconst (s)")
        plt.ylabel('Count')
        plt.title(f"Averaged timeconsts for each wafer at {band}")
        if save_plot: plt.savefig(os.path.join(plot_dir, f"{band}_timeconst_hist.png"))
        plt.show()

    for band in plot_dict:
        plt.figure()
        alpha = 1
        s = 50 
        for wafer in sorted(plot_dict[band].keys()):
            item = plot_dict[band][wafer]
            plt.scatter([item['pwv']]*len(item['data']),item['data'].iloc[:,1:].values,
                        s = s, alpha = alpha, label = f'{wafer} {band}')
            alpha *=0.7
            s *= 0.7
        plt.grid()
        plt.legend()
        plt.xlabel("PWV (mm)")
        plt.ylabel('Timeconst (s)')
        plt.ylim(val_range[band]['min'],val_range[band]['max'])
        plt.title(f"Detector timeconst over PWV")
        if save_plot: plt.savefig(os.path.join(plot_dir, f"{band}_timeconst_pwv.png"))
        plt.show()

    for band in plot_dict:
        plt.figure()
        alpha = 1
        s = 50 
        for wafer in sorted(plot_dict[band].keys()):
            item = plot_dict[band][wafer]
            plt.scatter([item['timestamp']]*len(item['data']),item['data'].iloc[:,1:].values,
                        s = s, alpha = alpha, label = f'{wafer} {band}')
            alpha *=0.7
            s *= 0.7
        plt.grid()
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel('Timeconst (s)')
        plt.ylim(val_range[band]['min'],val_range[band]['max'])
        plt.title(f"Detector timeconst over timestamp")
        if save_plot: plt.savefig(os.path.join(plot_dir, f"{band}_timeconst_timestamp.png"))
        plt.show()


def main(config_file=None, obs_id = None, obs_id_file = None, wafers = None,
    obs_query = None, verbose = 0, make_plot= True, save_plot = False, 
    overwrite = False, logger = None):

    import pandas as pd

    if logger is None:
        logger = util.init_logger(__name__)

    # Get cofig
    config = yaml.safe_load(open(config_file, 'r'))
    ctx = Context(config['context_file'])
    group_by = config.get('group_by','wafer')
    method = config.get('method', 'Mean')
    output_dir = config['archive']['policy']['out_dir']
    fname = os.path.join(output_dir, "timeconst.h5")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    min_cut = config.setdefault('min_cut', None)
    max_cut = config.setdefault('max_cut', None)
    max_z_score = config.setdefault('max_obs_residue', np.inf)
    min_yield = config.setdefault('max_error_ratio', 1)

    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')


    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = metadata.ManifestDb(config['archive']['index'])
    else: 
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('dets:wafer_slot')
        scheme.add_data_field('dataset')
        scheme.add_data_field('config')
        db = metadata.ManifestDb(config['archive']['index'], scheme=scheme)


    # Get observations

    if not obs_id_file: 
        if 'obs_id_file' in config: obs_id_file = config['obs_id_file']
        else: obs_id_file = None

    if obs_id:
        logger.info(f'Loading obs_ids from inputs.')
        if isinstance(obs_ids, str): obs_ids = [obs_ids]

    elif obs_query: 
        logger.info(f'Query Obsdb {obs_query}.')
        obs_ids = ctx.obsdb.query(obs_query)['obs_id'].tolist()

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

    logger.info(f"Found a total of {len(obs_ids)} observations.")


    # Find all wafers if wafer is not specified
    if not wafers:
        if 'wafer' in config: 
            wafers = config['wafer']
            if isinstance(wafers, str): wafers = [wafers]
        else:
            wafers = []

    if not wafers: 
        logger.info(f'No wafer specified. Load all wafers.')
        find_wafer = True
    else: find_wafer = False

    # Load and group data
    tau_dict ={}
    plot_dict = {}

    for obs_id in obs_ids:
        logger.info(f'Loading observation {obs_id}.')
        aman = ctx.get_meta(obs_id = obs_id)
        timestamp = aman.obs_info.timestamp
        pwv = aman.obs_info.pwv
        #pwv = abs(np.random.normal(1,1)) #Only for testing!

        if find_wafer:
            wafers = set(aman.det_info.wafer_slot)

        for wafer in wafers:

            wafer_mask = (aman.det_info.wafer_slot== wafer)
            tau = aman.ptsrc_params.tau[wafer_mask]

            range_mask = np.ones_like(tau)
            if min_cut: range_mask = np.logical_and(range_mask, (tau>min_cut))
            if max_cut: range_mask = np.logical_and(range_mask, (tau<max_cut))

            tau = tau[range_mask]
            det_id = aman.det_info.det_id[wafer_mask][range_mask]
            bias_line = aman.det_info.wafer.bias_line[wafer_mask][range_mask]
            bandpass = aman.det_info.band[wafer_mask][range_mask]
                
            for band in set(bandpass):
                mask = (bandpass ==band)

                logger.info(f'{wafer} {band} average timeconst of {np.nanmean(tau[mask]):.2e} and std {np.nanstd(tau[mask]):.2e}.')
                
                if make_plot:
                    plot_dict[band][wafer]['data'] = plot_dict.setdefault(band,{}).setdefault(wafer,{'data':pd.DataFrame({'det_id':{}})})['data'].merge(
                                                                pd.DataFrame({'det_id':det_id[mask], obs_id:tau[mask]}), how = 'outer')
                    plot_dict[band][wafer].setdefault('pwv',[]).append(pwv)
                    plot_dict[band][wafer].setdefault('timestamp',[]).append(timestamp)


            group_idx = {'dets':det_id,'biasline':bias_line,'wafer':bandpass}

            for i in range(len(det_id)):
                tau_dict.setdefault(wafer,{}).setdefault(group_idx[group_by][i],[])
                tau_dict[wafer][group_idx[group_by][i]].append(tau[i])  
            
    nan_entries = [(wafer, group) for group in tau_dict[wafer] for wafer in tau_dict if (np.isnan(tau_dict[wafer][group]).all())]
    
    if nan_entries:
        logger.warning(f"Certain {group_by} groups have no usable timeconst value. Outputing NaN.")

                    

    if make_plot:
        make_plots(plot_dict, min_cut, max_cut, save_plot, output_dir)

    # Take the average of all observation and write to file
    tau_model ={}
    tau_error ={}

    if not min_yield: min_yield=1

    for wafer in tau_dict:

        logger.info(f'Taking the {method} of each {group_by}.')
        if max_z_score and max_z_score>0:
            tau_dict[wafer] = {key: np.array(item)[abs(scipy.stats.zscore(item))< max_z_score] for key, item in tau_dict[wafer].items()}
        
        if method == 'Median':
            tau_model[wafer] = {key: np.nanmedian(item) if ((not np.isnan(item).all()) and (len(item)>min_yield)) else np.nan for key, item in tau_dict[wafer].items()}
            tau_error[wafer] = {key: np.nanmedian(abs(item - np.nanmedian(item))) if ((not np.isnan(item).all()) and (len(item)>min_yield)) else np.nan for key, item in tau_dict[wafer].items()}

        else:
            if  method != 'Mean':
                logger.warning('Wrong method specified! Must be Mean or Median.')
                logger.info('Using default method (Mean).')
            tau_model[wafer] = {key:np.nanmean(item) if ((not np.isnan(item).all()) and (len(item)>min_yield)) else np.nan for key, item in tau_dict[wafer].items()}
            tau_error[wafer] = {key:np.nanstd(item) if ((not np.isnan(item).all()) and (len(item)>min_yield)) else np.nan for key, item in tau_dict[wafer].items()}

        # For now, the dataset name includes wafer slot and group_by
        dataset = f'timeconst_{wafer}_{group_by}'
        logger.info(f"Writing to dataset {dataset}.")

        # Write to h5 and ManifestDb
        index_name = {'dets':'dets:det_id','biasline':'dets:wafer.bias_line','wafer':'dets:band'}

        tau_rs = metadata.ResultSet(keys=[index_name[group_by], 'timeconst', 'timeconst_error'])

        for idx in tau_model[wafer]:
            tau_rs.rows.append((idx, tau_model[wafer][idx], tau_error[wafer][idx]))

        io_meta.write_dataset(tau_rs, fname, dataset, overwrite = overwrite) 

        # The obs_id used is stored in the attributes. 
        with h5py.File(fname, 'r+') as h:
            h[dataset].attrs['obs_id'] = obs_ids

        db.add_entry({'dets:wafer_slot':wafer, 'dataset': dataset, 'config': config_file}, 
            fname, replace = overwrite)

    logger.info(f'Finished writing to {config["archive"]["index"]}.')

if __name__ == '__main__':
    util.main_launcher(main, get_parser)