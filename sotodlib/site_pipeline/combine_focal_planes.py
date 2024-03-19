import os
import re
import glob
import numpy as np

from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset, read_dataset

from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'combine_focal_planes: ')

def combine_pointings(pointing_result_files, method='highest_R2', R2_threshold=0.3, 
                      save=False, output_dir=None, save_name=None):
    combined_dict = {}
    for file in pointing_result_files:
        rset = read_dataset(file, 'focal_plane')
        for row in rset[:]:
            if row['dets:readout_id'] not in combined_dict.keys():
                combined_dict[row['dets:readout_id']] = {}
                combined_dict[row['dets:readout_id']]['band'] = row['band']
                combined_dict[row['dets:readout_id']]['channel'] = row['channel']
                
                combined_dict[row['dets:readout_id']]['R2'] = np.atleast_1d([])
                combined_dict[row['dets:readout_id']]['xi'] = np.atleast_1d([])
                combined_dict[row['dets:readout_id']]['eta'] = np.atleast_1d([])
                combined_dict[row['dets:readout_id']]['gamma'] = np.atleast_1d([])
                
            combined_dict[row['dets:readout_id']]['R2'] = np.append(combined_dict[row['dets:readout_id']]['R2'], row['R2'])
            combined_dict[row['dets:readout_id']]['xi'] = np.append(combined_dict[row['dets:readout_id']]['xi'], row['xi'])
            combined_dict[row['dets:readout_id']]['eta'] = np.append(combined_dict[row['dets:readout_id']]['eta'], row['eta'])
            combined_dict[row['dets:readout_id']]['gamma'] = np.append(combined_dict[row['dets:readout_id']]['gamma'], row['gamma'])

    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'R2', 'xi', 'eta', 'gamma'])
    for det, val in combined_dict.items():
        band = int(val['band'])
        channel = int(val['channel'])
        
        mask = val['R2'] > R2_threshold
        if np.all(~mask):
                xi, eta, gamma, R2 = np.nan, np.nan, np.nan, np.nan
        else:
            if method == 'highest_R2':
                idx = np.argmax(val['R2'][mask])
                xi, eta, gamma, R2 = val['xi'][mask][idx], val['eta'][mask][idx], val['gamma'][mask][idx], val['R2'][mask][idx]
            elif method == 'mean':
                xi, eta, gamma = np.mean(val['xi'][mask]), np.mean(val['eta'][mask]), np.mean(val['gamma'][mask])
                R2 = np.nan
            elif method == 'median':
                xi, eta, gamma = np.median(val['xi'][mask]), np.median(val['eta'][mask]), np.median(val['gamma'][mask])
                R2 = np.nan
            else:
                raise ValueError('Not supported method. Supported methods are `highest_R2`, `mean` or `median`')
        focal_plane.rows.append((det, band, channel, R2, xi, eta, gamma))
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'combined_pointing_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_name is None:
            ctimes = np.atleast_1d([])
            wafer_slots = np.atleast_1d([])
            for file in pointing_result_files:
                filename = os.path.basename(file)
                match = re.search('\d{10}', filename)
                ctime = int(match.group(0) if match else None)
                match = re.search('ws\d{1}', filename)
                ws = match.group(0)
                ctimes = np.append(ctimes, ctime)
                wafer_slots = np.append(wafer_slots, ws)
            ctimes = ctimes.astype('int')
            wafer_slots = np.sort(np.unique(wafer_slots.astype('U3')))
            save_name = f'focal_plane_{ctimes.min()}_{ctimes.max()}_' + ''.join(wafer_slots) + '.hdf'
            
        write_dataset(focal_plane, os.path.join(output_dir, save_name), 'focal_plane', overwrite=True)
    return focal_plane

def combine_onewafer_results(pointing_dir, ws, output_dir, filename=None,
                            method='highest_R2', R2_threshold=0.3,):
    pointing_result_files = glob.glob(os.path.join(pointing_dir, f'focal_plane*{ws}.hdf'))
    if filename is None:
        filename = f'focal_plane_{ws}_combined.hdf'
    _ = combine_pointings(pointing_result_files, save=True, output_dir=output_dir, save_name=filename)
    return

def combine_allwafer_results(pointing_dir, output_dir, filename=None,
                            method='highest_R2', R2_threshold=0.3,):
    pointing_result_files = glob.glob(os.path.join(pointing_dir, 'focal_plane*.hdf'))
    if filename is None:
        filename = f'focal_plane_combined.hdf'
    _ = combine_pointings(pointing_result_files, save=True, output_dir=output_dir, save_name=filename)
    return

def make_detabase(focal_plane_file, db_file,):
    scheme = metadata.ManifestScheme().add_data_field('dataset')
    db = metadata.ManifestDb(scheme=scheme)
    db.add_entry({'dataset': 'focal_plane'}, filename=focal_plane_file)
    db.to_file(db_file)
    return
    
def main(pointing_dir, output_dir=None, method='highest_R2', R2_threshold=0.3,):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'combined_results')
    
    logger.info('Combining each wafer resluts')
    wafer_slots = [f'ws{i}' for i in range(7)]
    for ws in wafer_slots:
        combine_onewafer_results(pointing_dir=pointing_dir, ws=ws, 
                                 output_dir=output_dir, filename=None,
                                 method=method, R2_threshold=R2_threshold)
    
    logger.info('Combining all wafer resluts')
    combine_allwafer_results(pointing_dir=pointing_dir, output_dir=output_dir, filename='focal_plane_combined.hdf',
                             method=method, R2_threshold=R2_threshold)
    
    logger.info('Making a database')
    focal_plane_file = os.path.join(output_dir, 'focal_plane_combined.hdf')
    db_file = os.path.join(output_dir, 'focal_plane_combined.sqlite')
    make_detabase(focal_plane_file, db_file,)
    return
    
def get_parser():
    parser = argparse.ArgumentParser(description="Combine multiple result of pointing.")
    parser.add_argument('--pointing_dir', type=str, required=True, help='Directory containing pointing result files.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save combined results. Default is "combined_results".')
    parser.add_argument('--method', type=str, default='highest_R2', choices=['highest_R2', 'mean', 'median'], help='Combination method. Default is "highest_R2".')
    parser.add_argument('--R2_threshold', type=float, default=0.3, help='Threshold for R2 value. Default is 0.3.')
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
