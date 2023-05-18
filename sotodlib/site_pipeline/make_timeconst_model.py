"""This module combines multiple measurements of detector time constants
from analyze-bright-ptsrc to produce a time constant model. For first 
light, the model is simply the average measured value. This script works 
with dets:det_id and assumes that channel map has been applied. The 
final result is a time constant indexed by wafer, with associated obs_id 
stored in the dataset attributes. There's an option to 
make this number per wafer, per biasline or per detector.
"""


from argparse import ArgumentParser
import numpy as np
import os
import yaml
import h5py

from sotodlib.core import Context, metadata
import sotodlib.io.metadata as io_meta


def parse_args(args):

    parser = ArgumentParser()

    parser.add_argument('-c', '--config-file', help=
        "Configuration file.")

    parser.add_argument('--overwrite', default= False, help=
     "Overwrites existing entries in h5 (default: False)")
    
    # parser.add_argument('-w', '--wafer', help=
    #     "The wafer or wafers to run model on.")

    # parser.add_argument('-obs_id', nargs='*', help=
    #     "Observations to summarize over.")

    # parser.add_argument('-g', '--group_by' help=
    #     "dets, biasline or wafer.")

    args = parser.parse_args(args)

    return args


def main(args=None):

    # Will change this when in site_pipeline
    args = parse_args(args) 

    # Get cofig
    config = yaml.safe_load(open(args.config_file, 'r'))
    ctx = Context(config['context_file'])

    overwrite = args.overwrite
    group_by = config.get('group_by','wafer')
    output_dir = config['archive']['policy']['out_dir']
    fname = os.path.join(output_dir, "timeconst.h5")


    if os.path.exists(config['archive']['index']):
        db = metadata.ManifestDb(config['archive']['index'])
    else: 
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('dets:wafer_slot')
        scheme.add_data_field('dataset')
        scheme.add_data_field('config')
        db = metadata.ManifestDb(config['archive']['index'], scheme=scheme)


    # Get observations
    obs_ids= config['obs_id']
    if isinstance(obs_ids, str): obs_ids = [obs_ids]


    # Find all wafers if wafer is not specified
    if 'wafer' in config: 
        find_wafer = False
        wafers = config['wafer']
        if isinstance(wafers, str): wafers = [wafers]
    else:
        find_wafer = True

    # Load and group data
    tau_dict ={}

    for obs_id in obs_ids:
        aman = ctx.get_meta(obs_id = obs_id)

        if find_wafer:
            wafers = set(aman.det_info.wafer_slot)

        for wafer in wafers:
            wafer_mask = (aman.det_info.wafer_slot== wafer)
            det_id = aman.det_info.det_id[wafer_mask]
            bias_line = aman.det_info.wafer.bias_line[wafer_mask]
            bandpass = aman.det_info.band[wafer_mask]
            tau = aman.ptsrc_params.tau[wafer_mask]

            group_idx = {'dets':det_id,'biasline':bias_line,'wafer':bandpass}

            for i in range(len(det_id)):
                tau_dict.setdefault(wafer,{}).setdefault(group_idx[group_by][i],[]).append(tau[i])  

    # Take the average of all observation and write to file
    tau_model={}

    for wafer in tau_dict:

        tau_model[wafer] = {key:np.nanmean(item) for key,item in tau_dict[wafer].items()}

        # For now, the dataset name includes wafer slot and group_by
        dataset = f'timeconst_{wafer}_{group_by}'

        # Write to h5 and ManifestDb
        index_name = {'dets':'dets:det_id','biasline':'dets:wafer.bias_line','wafer':'dets:band'}

        tau_rs = metadata.ResultSet(keys=[index_name[group_by], 'tau'])

        for idx in tau_model[wafer]:
            tau_rs.rows.append((idx, tau_model[wafer][idx]))

        io_meta.write_dataset(tau_rs, fname, dataset, overwrite = overwrite) 

        # The obs_id used is stored in the attributes. 
        with h5py.File(fname, 'r+') as h:
            h[dataset].attrs['obs_id'] = obs_ids

        db.add_entry({'dets:wafer_slot':wafer, 'dataset': dataset, 'config': args.config_file}, 
            fname, replace = overwrite)


if __name__ == '__main__':
    main()