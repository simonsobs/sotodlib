import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import csv 
import yaml
import os 
import sqlite3
import argparse

from iminuit import Minuit, cost

from sotodlib import core, tod_ops
from sotodlib.tod_ops import filters, fft_ops, apodize, detrend # FFT modules
import sotodlib.io.load_smurf as load_smurf
from sotodlib.io.load_smurf import load_file, G3tSmurf, Observations, SmurfStatus
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.site_pipeline.util as sp_util
import sotodlib.io.metadata as io_meta
import sotodlib.hwp.hwp as hwp
from sotodlib.hwp import demod
from sotodlib.hwp.g3thwp import G3tHWP
from so3g.hk import load_range, HKArchiveScanner
import sodetlib 


def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def search_wg_obs(config_file='./test_config.yaml', keyword="wg_step") :

    config2 = _get_config(config_file)

    context_yaml = _get_config(config2['context_file'])
    dbfile = context_yaml['obsdb'].replace('{base_dir}',context_yaml['tags']['base_dir'])
    table = "tags"
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    obs_ids = cur.execute('select obs_id from %s '%(table)).fetchall()
    tags = cur.execute('select tag from %s '%(table)).fetchall()

    wg_obs_dict = {}
    wg_id_list = []


    for i, tag in enumerate(tags) :

        if tag[0] is None : continue

        if keyword in tag[0] :
            wg_obs_dict[obs_ids[i][0]] = tag[0]
            print(obs_ids[i][0], tag[0])

    return wg_obs_dict


def wg_tag2obsid(tag, config_file='./test_config.yaml', keyword="wg_step") :

    obs_dict = search_wg_obs(config_file=config_file, keyword=keyword)

    for key in obs_dict.keys() :
        if obs_dict[key] == tag :
            return key
    return None


def wg_obsid2tag(obsid, config_file='./test_config.yaml', keyword="wg_step") :

    obs_dict = search_wg_obs(config_file=config_file, keyword=keyword)

    for key in obs_dict.keys() :
        if key == obsid :
            return obs_dict[key]

    return None


def wg_init_setting(config_file, stream_id, tag, hk_dir) : 
   
    wg_max_enc = 52000.
    ### Declare axis manager 
    SMURF = load_smurf.G3tSmurf.from_configs(config_file)
    session = SMURF.Session()   
    obs_list = session.query(Observations).filter(Observations.tag.like(tag), Observations.stream_id.like(stream_id)).all()
    obs_ids = [obs.obs_id for obs in obs_list]
    obs_start = [obs.start for obs in obs_list]
    obs_end = [obs.stop for obs in obs_list]
    # I guess the number of obs_XXX component is just one. But adopt the last component as the end time just in case.
    aman = SMURF.load_data(obs_start[0], obs_end[-1], stream_id = stream_id) 

    ### Load data
    utils.load_hwp_data(aman, config_file)
    bias_step_file = utils.get_last_bias_step(obs_ids[0], SMURF)
    bias_step_obj = np.load(bias_step_file, allow_pickle=True).item()
    
    ### Load and wrap wiregrid house keeping data
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(obs_start[0], obs_end[-1], wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/wg_max_enc*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)
    
    return aman


def wg_init_setting_ctime(start, end, config_file, hk_dir) :

    ### Declare axis manager
    SMURF = load_smurf.G3tSmurf.from_configs(config_file)
    aman = SMURF.load_data(start, end)

    ### Load data
    utils.load_hwp_data(aman, config_file)
    bias_step_file = utils.get_last_bias_step(start, SMURF)
    bias_step_obj = np.load(bias_step_file, allow_pickle=True).item()

    ### Load and wrap wiregrid house keeping data
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(start, end, wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/wg_max_enc*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)

    return aman



def wg_demod_tod(aman) :

    ### If axis manager already has demomded components, skip the process
    if ("demodQ" in aman.keys()) or ("demodU" in aman.keys()) : return

    ### Demod
    ind=3
    fig = plt.figure()
    plt.plot(np.arange(len(aman["signal"][ind])), aman["signal"][ind], alpha=0.5)
    detrend.detrend_tod(aman,method='median')
    plt.plot(np.arange(len(aman["signal"][ind])), aman["signal"][ind], alpha=0.5)
    apodize.apodize_cosine(aman)
    plt.plot(np.arange(len(aman["signal"][ind])), aman["signal"][ind], alpha=0.5)
    hwp.demod_tod(aman,signal='signal')
    plt.plot(np.arange(len(aman.demodQ[ind])), aman.demodQ[ind], alpha=0.5)
    plt.grid(ls=":")
    plt.show()

    return

def get_moving_part() : 
    pass

### Note : Length of plateau and in/ouf time of wg (remove_len) are determined in hard coding.
###        You may have to check if the length is ok to analyze recent wg calibration.
def get_wg_angle(aman, threshold=0.00015, plateau_len=3*5000, remove_len= 30*5000, debug=False) :
    
    wg_angle =  aman["hkwg"]["wg_angle"]
    wg_timestamp = aman["hkwg"]["wg_timestamp"]
    
    ### Get static part
    diff_wgangle = np.diff(wg_angle)
    moving = (diff_wgangle>threshold).astype(np.int32)
    switch_indices = np.where(np.diff(moving) != 0)[0] + 1
    run_lengths = np.diff(switch_indices)
    
    #static_indices_start = switch_indices[:-1][(run_lengths>plateau_len) & (run_lengths<remove_len)]
    static_indices_start = switch_indices[:-1][run_lengths>plateau_len]
    static_indices_end = []
    #for _id,run_length in enumerate(run_lengths[(run_lengths>plateau_len) & (run_lengths<remove_len)]) : 
    #for _id,run_length in enumerate(run_lengths[(run_lengths>plateau_len)]) : 
    #    static_indices_end.append(static_indices_start[_id]+run_length-1)
    for _id,run_length in enumerate(run_lengths[(run_lengths>plateau_len)]) : 
        if run_length < remove_len : 
            static_indices_end.append(static_indices_start[_id]+run_length-1)
        else : 
            static_indices_end.append(static_indices_start[_id]+prev-1)
        prev = run_length
     
    ### Generate a list that contents timestamps of wg static start/end and angle of wg
    wg_info = []
    for _id in range(len(static_indices_start)) : 
        angle = np.mean(wg_angle[static_indices_start[_id]:static_indices_end[_id]])
        wg_info.append([wg_timestamp[static_indices_start[_id]],wg_timestamp[static_indices_end[_id]],angle])

    ### Temporary to add the last part
    length = len(static_indices_start)
    start_ts = static_indices_end[length-1] + 1 + static_indices_start[-1] - static_indices_end[-2]
    last_ts = static_indices_end[length-1] - static_indices_end[length-2] + start_ts 
    angle = np.mean(wg_angle[start_ts:last_ts])
    wg_info.append([wg_timestamp[start_ts], wg_timestamp[last_ts],angle])
    static_indices_start = np.append(static_indices_start,start_ts)
    static_indices_end.append(last_ts)


    if debug : 
        fig_check = plt.figure(figsize=[20,4])
        plt.plot(wg_timestamp[:-1][diff_wgangle>0]-1.676598e9,diff_wgangle[diff_wgangle>0])
        plt.plot(aman.timestamps-1.676598e9, aman.demodQ[0], alpha=0.5)
        for _id in range(len(static_indices_start)) : 
            #plt.vlines(wg_timestamp[static_indices_start[_id]]-1.676598e9,0.,0.00035,color="blue")
            #plt.vlines(wg_timestamp[static_indices_end[_id]]-1.676598e9,0.,0.00035,color="red")
            plt.vlines(wg_timestamp[static_indices_start[_id]]-1.676598e9,-0.06,0.,color="blue")
            plt.vlines(wg_timestamp[static_indices_end[_id]]-1.676598e9,-0.06,0.,color="red")
        plt.show()
    

    return len(static_indices_start),wg_info


def funcQ(x, amp, freq, phase, offsetQ):
    return amp * np.cos ( freq * x + 2.*phase) + offsetQ


def funcU(x, amp, freq, phase, offsetU):
    return amp * np.cos ( freq * x + 2.*phase + np.pi*0.5) + offsetU


def fit_angle(aman,fitting_results,wg_info,rotate_tau=False,debug=False):
    
    ### Calculate HWP rotation speed 
    timestamp = aman["timestamps"]
    dt = (timestamp[-1]-timestamp[0])
    #hwp rotation count
    rt_count = aman["hwp_angle"]*0.5/np.pi
    num_rot = np.sum((np.diff(rt_count)>0.9).astype(np.int32))
    hwp_speed = num_rot/dt
    dspeed = np.abs(hwp_speed - 2.)
    print(f"HWP rotation speed is {hwp_speed}")

    ### If you want to consider the effect of time constant, you need to make tau vector
    ### NOTE :::  Must be updated    
    t_const = []
    if rotate_tau : 
        import pickle
        with open('/data/atakeuchi/my_soenv/repos/230313/sotodlib/sotodlib/wiregrid/my_tau.pickle', 'rb') as fin :
            
            data = pickle.load(fin)

            for det in aman["det_info"]["readout_id"] :
                if det.encode() in data.keys() : 
                    t_const.append(data[det.encode()])
                else : 
                    t_const.append(0.)

        t_const = np.array(t_const)

    ### Get compoents of wg_info then calculate polarized angle
    each_result = []
    ang_mean,Q_mean,U_mean,Q_error,U_error=[],[],[],[],[]
    fig1 = plt.figure()
    for _id in range(len(wg_info)) : 
        wg_angle = wg_info[_id][2]
        t_selection = (aman.timestamps > wg_info[_id][0]) & (aman.timestamps < wg_info[_id][1])
           
        ### time constant correction
        Q = aman.demodQ[:,t_selection]
        U = aman.demodU[:,t_selection]
        
        if debug :
            fig = plt.figure()
            plt.plot(timestamp[t_selection],Q[0])
            plt.plot(timestamp[t_selection],U[0])
            plt.show()
         
        if rotate_tau : 
            exp = np.exp(1j*t_const*hwp_speed*2.*np.pi*4.).reshape((len(t_const), 1)) #  Need to update this later
        else : 
            exp = 1.
                
        rotated_tod = exp*(Q+1j*U)
        Q = np.real(rotated_tod)
        U = np.imag(rotated_tod)
        
        ### get sin curve
        ang_mean.append(wg_angle)
        Q_mean.append(np.mean(Q,axis=1))
        U_mean.append(np.mean(U,axis=1))
        Q_error.append(np.std(Q,axis=1))
        U_error.append(np.std(U,axis=1))

    
    ang_mean = np.array(ang_mean)
    Q_mean = np.array(Q_mean)
    U_mean = np.array(U_mean)
    Q_error = np.array(Q_error)
    U_error = np.array(U_error)
    
    for i in range(len(Q_mean[0])) : 

        if np.sum(np.isnan(Q_mean[:,i]).astype(np.int32)) > 0 : 
            print(aman["det_info"]["readout_id"][i])
            print(aman.dets.vals)
            continue 

        Qi,Ui = Q_mean[:,i],U_mean[:,i]
        Qi_e,Ui_e = Q_error[:,i],U_error[:,i]                  
            
        def fcn(amp,freq,phase,offsetQ,offsetU):
            model1 = amp * np.cos ( freq * ang_mean + 2.*phase) + offsetQ 
            model2 = amp * np.cos ( freq * ang_mean + 2.*phase + np.pi*0.5) + offsetU
            chi_squared1 =  ((model1-Qi)/Qi_e)*((model1-Qi)/Qi_e)
            chi_squared2 =  ((model2-Ui)/Ui_e)*((model2-Ui)/Ui_e)
            return np.sum(chi_squared1)+np.sum(chi_squared2)
            
        m = Minuit(fcn, amp=(np.max(Q_mean[:,i])-np.min(Q_mean[:,i]))*0.5, freq=hwp_speed, phase=0.5*np.pi, offsetQ=np.mean(Q_mean[:,i]), offsetU=np.mean(U_mean[:,i]))
        m.limits['amp'] = (0, None)
        m.limits['phase'] = (-np.pi,2.*np.pi)
        m.limits['freq'] = (2.,2.)
        m.migrad().migrad().hesse()

        _id = aman["det_info"]["readout_id"][i]
        _amp = m.values[0]
        _amp_e = m.errors[0]
        _ang = m.values[2]
        if _ang > np.pi : _ang -= np.pi
        elif _ang < 0. : _ang += np.pi
        _ang_e = m.errors[2]
        _off_q = m.values[3]
        _off_u = m.values[4]
        ndf = len(Qi) + len(Ui) - len(m.values)
        _chi2 = m.fval/ndf
           
        fitting_results.rows.append((_id,_amp,_amp_e,_ang,_ang_e,_off_q,_off_u,_chi2))
   
        if debug and (i%30 == 0) :
            print(_chi2,m.values[1])
            fig = plt.figure()
            angle = np.arange(628)*0.01
            plt.errorbar(ang_mean,Q_mean[:,i],yerr=Q_error[:,i],fmt='.',color='r')
            plt.errorbar(ang_mean,U_mean[:,i],yerr=U_error[:,i],fmt='.',color='b')
            plt.plot(angle, funcQ(angle, m.values[0], m.values[1], m.values[2], m.values[3]), ls="--", label="fittedQ",color='r')
            plt.plot(angle, funcU(angle, m.values[0], m.values[1], m.values[2], m.values[4]), ls="--", label="fittedU",color='b')
            plt.show()
            
    return fitting_results



def main(config_file='./test_config.yaml', query=None, obs_id="", overwrite=False, min_ctime=None, max_ctime=None,logger=None,):

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
    obs = obsdb.query(f'obs_id == "{obs_id}"')[0]

    # place of house keeping data
    hk_dir = config['hk_dir']

    aman = context.get_meta(obs_id)

    # Analysis settings
    tag = 'obs,stream,'+wg_obsid2tag(obsid=obs_id, config_file=config_file)
    arrays = config['arrays']
    output_h5 = config['archive']['policy']['filename']

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

    for array in arrays :
        stream_id = array['stream_id']
        aman = wg_init_setting(config_file=config_file,stream_id=stream_id,tag=tag,hk_dir=hk_dir)
        wg_demod_tod(aman)
        num_angle, wg_info = get_wg_angle(aman,debug=True)
        fitting_results = fit_angle(aman,fitting_results,wg_info,debug=False,rotate_tau=True)
        del aman

    # Save outputs
    db_data = {'obs:obs_id': obs_id, 'dataset' : f'wg_fitting_{obs_id}'}
    db.add_entry(db_data, output_h5, f'{obs_id}',replace=True)
    db.to_file(config['archive']['index'])
    io_meta.write_dataset(fitting_results, output_h5, f'{obs_id}', overwrite=overwrite)


if __name__ == '__main__' :

    #sp_util.main_launcher(main,get_parser)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('obs_id', type=str)
    parser.add_argument('--query', type=str, default="")
    parser.add_argument('--min_ctime', type=float, default=0.)
    parser.add_argument('--max_ctime', type=float, default=0.)
    args = parser.parse_args()
    main(config_file=args.config_file, obs_id=args.obs_id)

