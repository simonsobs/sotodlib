import glob
import math
import h5py
import yaml
import argparse 
import os 
import pandas as pd 
import pickle

import numpy as np 
import matplotlib.pyplot as plt 

from sotodlib import core, tod_ops
import sotodlib.io.load_smurf as load_smurf
from sotodlib.io.load_smurf import load_file, G3tSmurf, Observations, SmurfStatus
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.io.metadata as io_meta

plt.rcParams['font.family']='Serif'
plt.rcParams['mathtext.fontset']='stix'
plt.rcParams['font.size']=16
plt.rcParams['image.origin']='lower'

def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def load_csv(base_dir="/mnt/so1/shared/site-pipeline/data_pkg/ucsd-sat1/detmaps/"):
    
    exist_dirs = glob.glob(base_dir+"Mv*")
    
    det_info = {}
    for _dir in exist_dirs : 
        
        target = _dir + "/*_readout.csv"
        _latest_csv = sorted(glob.glob(target))

        if len(_latest_csv) == 0 : continue
        else : latest_csv = _latest_csv[-1]
        mv_num = latest_csv.replace(_dir,"").split("_")[0].replace("/Mv","")
    
        detmap_df = pd.read_csv(latest_csv)
        
        det_info[mv_num] = detmap_df[["smurf_band", "smurf_subband", "smurf_channel",  "angle_raw_deg", "det_x", "det_y", "bandpass", "is_optical"]]
        
    return det_info


def calc_tau_with_wg(dset1,dset2,del_hwp_speed, tau_results,  debug=False) : 

    #det_info = load_csv(base_dir="/homes/kkiuchi/so_test/detmap_output/")
    det_info = load_csv()


    rname1, amp_arr1, ang_arr1, chi_arr1 = [], [], [], []
    rname2, amp_arr2, ang_arr2, chi_arr2 = [], [], [], []
    tc_dict = {}

    for line in dset1 : 
        rname1.append(line[0])
        amp_arr1.append(line[1])
        ang_arr1.append(line[3])
        chi_arr1.append(line[7])

    for line in dset2 : 
        rname2.append(line[0])
        amp_arr2.append(line[1])
        ang_arr2.append(line[3])
        chi_arr2.append(line[7])    

    arr_tc90 = []
    arr_tc150 = []
    dict_tau = {}
    for i,rid in enumerate(rname2) : 
    
        if rid not in rname1 : continue

        str_rid = str(rid)
        _sp = str_rid.split("_")
        mv = _sp[2].replace("mv","")
        s_band = int(_sp[4])
        s_channel = int(_sp[5].replace("'",""))
        df = det_info[mv]

        bandpass = df[(df["smurf_band"]==s_band) & (df["smurf_channel"]==s_channel)]["bandpass"]
        #print(bandpass.size)
        if bandpass.size ==0 : continue
        #print(bandpass.iloc[0])

        try : 
            bandpass = int(bandpass.iloc[0])
        except : 
            continue 
        if math.isnan(bandpass) : continue

        index = rname1.index(rid)
        temp=np.rad2deg(ang_arr2[i])-np.rad2deg(ang_arr1[index])
        if temp<0. : temp += 180.
        elif temp>180. : temp -= 180.
        corrected = temp
        tau = corrected/4./180./del_hwp_speed
        tc_dict[rname1[index]] = tau
        if bandpass == 90 : arr_tc90.append(tau)
        elif bandpass == 150  : arr_tc150.append(tau)
        tau_results.rows.append((rid, tau))

    if debug : 
        print(len(arr_tc90), len(arr_tc150))
        plt.hist(arr_tc90, bins=300,range=(0.,0.03), color='red', alpha=0.6)
        plt.hist(arr_tc150, bins=300,range=(0.,0.03), color='blue', alpha=0.6)
        plt.xlabel("Time constant [sec]")
        plt.yscale("log")
        plt.grid()
        plt.show()

    return tc_dict, tau_results


def compare_to_tune(config_file, obs_id, tau_dict, tag, stream_id) : 

    SMURF = load_smurf.G3tSmurf.from_configs(config_file)
    session = SMURF.Session()   
    obs_list = session.query(Observations).filter(Observations.tag.like(tag), Observations.stream_id.like(stream_id)).all()
    obs_ids = [obs.obs_id for obs in obs_list]
    obs_start = [obs.start for obs in obs_list]
    obs_end = [obs.stop for obs in obs_list]
    aman = SMURF.load_data(obs_start[0], obs_end[-1], stream_id = stream_id) 

    
    ### Load data
    utils.load_hwp_data(aman, config_file)
    bias_step_file = utils.get_last_bias_step(obs_ids[0], SMURF)
    bias_step_obj = np.load(bias_step_file, allow_pickle=True).item()
    print(bias_step_obj.keys())

    return np.array(bias_step_obj["tau_eff"])


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()

    parser.add_argument('dbname', type=str)
    parser.add_argument('--obsid_1hz', type=str, default="obs_1676657812_sat1_1111101")
    parser.add_argument('--obsid_2hz', type=str, default="obs_1676624899_sat1_1111101")
    parser.add_argument('--del_hwp_speed', type=float, default=1.)
    parser.add_argument('--compare', type=bool, default=False)
    args = parser.parse_args()

    _file = h5py.File(args.dbname,"r")
    dset1 = _file[args.obsid_1hz] # 1 Hz
    dset2 = _file[args.obsid_2hz] # 2 Hz
    del_hwp_speed = args.del_hwp_speed


    tag = "wg_step15sec_wTravel"
    config_file = "/homes/atakeuchi/workspace/SO/wg_dev/sat1_p10r2/test_config.yaml"
   
    config = _get_config(config_file)
    arrays = config['arrays']

    output_h5 = config['archive']['policy']['filename']
    if os.path.exists(config['archive']['index']):
        db = core.metadata.ManifestDb(config['archive']['index'])
    else:
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)

    tau_results = core.metadata.ResultSet(
        keys=["dets:readout_id","tau_wg"]
    )

    tau_dict, tau_results = calc_tau_with_wg(dset1, dset2, del_hwp_speed, tau_results, debug=True)

    ### Save tau in dict format
    with open("./my_tau.pickle", "wb") as fout :
        pickle.dump(tau_dict, fout)

    ### Will be updated.
    if args.compare : 
        for i, array in enumerate(arrays) : 
            stream_id = array['stream_id']
            if i == 0 : 
                arr_tau_eff = compare_to_tune(config_file, args.obsid_2hz, tau_dict, "obs,stream,"+tag, stream_id)
            else : 
                temp = compare_to_tune(config_file, args.obsid_2hz, tau_dict, "obs,stream,"+tag, stream_id)
                arr_tau_eff = np.append(arr_tau_eff,temp)

        plt.hist(arr_tau_eff, bins=1000,range=(0.,0.03))
        plt.yscale("log")
        plt.show()

    """
    obs_id = args.obsid_2hz
    ### Update database with fitted tau
    db_data = {'obs:obs_id': obs_id, 'dataset' : f'time_const_{obs_id}'}
    db.add_entry(db_data, output_h5, f'{obs_id}',replace=True)
    db.to_file(config['archive']['index'])
    io_meta.write_dataset(tau_results, output_h5, f'time_const_{obs_id}', overwrite=True)
    """


