import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import csv 

from iminuit import Minuit, cost

from sotodlib import core, tod_ops
from sotodlib.tod_ops import filters, fft_ops, apodize, detrend # FFT modules
import sotodlib.io.load_smurf as load_smurf
from sotodlib.io.load_smurf import load_file, G3tSmurf, Observations, SmurfStatus
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.hwp.hwp as hwp
from sotodlib.hwp import demod
from sotodlib.hwp.g3thwp import G3tHWP
from so3g.hk import load_range, HKArchiveScanner
import sodetlib 



def wg_init_setting(config_file, stream_id, tag, hk_dir) : 
    
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
    aman.wrap('si',bias_step_obj['Si'] , [(0, 'dets')])
    aman.wrap('tau_eff',bias_step_obj['tau_eff'] , [(0, 'dets')])
    
    ### Load and wrap wiregrid house keeping data
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(obs_start[0], obs_end[-1], wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/52000.*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)
    
    return aman



def wg_demod_tod(aman) : 
   
    ### If axis manager already has demomded components, skip the process 
    if ("demodQ" in aman.keys()) or ("demodU" in aman.keys()) : return
    
    ### Demod
    detrend.detrend_tod(aman,method='median')
    apodize.apodize_cosine(aman)
    hwp.demod_tod(aman,signal='signal')
    
    return 



def get_wg_angle(aman, threshold=0.00015, plateau_len=5000, debug=False) :
    
    ### Will we apply down sampling in the future?
    down_sample = 1
    num_split,rem  = len(aman["hkwg"]["wg_timestamp"])//down_sample, len(aman["hkwg"]["wg_timestamp"])%down_sample
    if rem == 0 : 
        _wg_angle = aman["hkwg"]["wg_angle"]/down_sample
        _wg_timestamp = aman["hkwg"]["wg_timestamp"]/down_sample
    else : 
        _wg_angle = aman["hkwg"]["wg_angle"][:-rem]/down_sample
        _wg_timestamp = aman["hkwg"]["wg_timestamp"][:-rem]/down_sample
        
    _wg_angle = _wg_angle.reshape((num_split,down_sample))
    _wg_timestamp = _wg_timestamp.reshape((num_split,down_sample))

    wg_angle = np.sum(_wg_angle,axis=1)
    wg_timestamp = np.sum(_wg_timestamp,axis=1)
    
    ### Get static part
    diff_wgangle = np.diff(wg_angle)
    moving = (diff_wgangle>threshold).astype(np.int32)
    switch_indices = np.where(np.diff(moving) != 0)[0] + 1
    run_lengths = np.diff(switch_indices)
    #print(switch_indices[:-1][run_lengths>plateau_len])
    
    static_indices_start = switch_indices[:-1][run_lengths>plateau_len/down_sample]
    static_indices_end = []
    for _id,run_length in enumerate(run_lengths[run_lengths>plateau_len/down_sample]) : 
        static_indices_end.append(static_indices_start[_id]+run_length-1)
     
    if debug : 
        plt.plot(wg_timestamp[:-1][diff_wgangle>0],diff_wgangle[diff_wgangle>0])
        for _id in range(len(static_indices_start)) : 
            plt.vlines(wg_timestamp[static_indices_start[_id]],0.,0.00035,color="red")
            plt.vlines(wg_timestamp[static_indices_end[_id]],0.,0.00035,color="red")
        plt.show()
    
    ### Generate a list that contents timestamps of wg static start/end and angle of wg
    wg_info = []
    for _id in range(len(static_indices_start)) : 
        angle = np.mean(wg_angle[static_indices_start[_id]:static_indices_end[_id]])
        wg_info.append([wg_timestamp[static_indices_start[_id]],wg_timestamp[static_indices_end[_id]],angle])
    
    return len(static_indices_start),wg_info



def wrap_det_info(aman,detmap_file,debug=False) : 
    detmap_df = pd.read_csv(detmap_file)
    
    sband = np.array(detmap_df['smurf_band'])
    schannel = np.array(detmap_df['smurf_channel'])
    idx_map = sodetlib.map_band_chans(aman.det_info.smurf.band, aman.det_info.smurf.channel, sband, schannel)
    idx_map = np.where(idx_map==-1, 0, idx_map)
    
    dmaps = core.AxisManager(aman.dets)
    isopt = np.array(detmap_df['is_optical'][idx_map], dtype=str)
    dmaps.wrap('angle_raw_deg', np.array(detmap_df['angle_raw_deg'][idx_map], dtype=float), [(0, 'dets')])
    dmaps.wrap('isopt', np.where(isopt=='True', True, False), [(0, 'dets')])
    dmaps.wrap('channel', np.array(detmap_df['smurf_channel'][idx_map], dtype=float), [(0, 'dets')])
    dmaps.wrap('band',  np.array(detmap_df['smurf_band'][idx_map], dtype=float), [(0, 'dets')])
    dmaps.wrap('subband', np.array(detmap_df['smurf_subband'][idx_map], dtype=float), [(0, 'dets')])
    bpass = np.array(detmap_df['bandpass'][idx_map].replace('NC','-1'), dtype=float)
    dmaps.wrap('bandpass', bpass, [(0, 'dets')])
    aman.wrap('detmap', dmaps)
       
    
        
def funcQ(x, amp, freq, phase, offsetQ):
    
    return amp * np.cos ( freq * x + 2.*phase) + offsetQ



def funcU(x, amp, freq, phase, offsetU):
    
    return amp * np.cos ( freq * x + 2.*phase + np.pi*0.5) + offsetU



def fit_angle(aman,wg_info,rotate_tau=True,output_file='./hoge.csv',debug=False):
    
    ### Calculate HWP rotation speed 
    timestamp = aman["timestamps"]
    dt = (timestamp[-1]-timestamp[0])
    #hwp rotation count
    rt_count = aman["hwp_angle"]*0.5/np.pi
    num_rot = np.sum((np.diff(rt_count)>0.9).astype(np.int32))
    hwp_speed = num_rot/dt
    print(f"HWP rotation speed is {hwp_speed}")
    
    ### Get compoents of wg_info then calculate polarized angle
    t_const = aman["tau_eff"]
    each_result = []
    ang_mean,Q_mean,U_mean,Q_error,U_error=[],[],[],[],[]
    fig1 = plt.figure()
    for _id in range(len(wg_info)) : 
        wg_angle = wg_info[_id][2]
        t_selection = (aman.timestamps > wg_info[_id][0]) & (aman.timestamps < wg_info[_id][1])
        
        
        
        ### time constant correction
        Q = aman.demodQ[:,t_selection]
        U = aman.demodU[:,t_selection]
        
        """
        if rotate_tau : 
            exp = np.exp(t_const*hwp_speed*2.*np.pi*4.).reshape((len(t_const), 1)) #  Check this later
        else : 
            exp = 1.
                
        rotated_tod = exp*(Q+1j*U)
        Q = np.real(rotated_tod)
        U = np.imag(rotated_tod)
        """
        
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
    
    ### Fitting with iMinuit
    fitted_phases,fitted_amps,fitted_offsetq,fitted_offsetu = ['phase'],['amp'],['offsetq'],['offsetu']
    error_phases,error_amps,error_offsetq,error_offsetu = ['phase_e'],['amp_e'],['offsetq_e'],['offsetu_u']
    channels,bands,subbands,bandpasses = ['channel'],['band'],['subband'],['bandpass']
    minuit_chi2 = ['chi2']
    for i in range(len(Q_mean[0])) : 
        
        if np.sum(np.isnan(Q_mean[:,i]).astype(np.int32)) > 0 : 
            print(f'{int(aman["detmap"]["channel"][i])}.{int(aman["detmap"]["band"][i])}.{int(aman["detmap"]["subband"][i])}_{int(np.nan_to_num(aman["detmap"]["bandpass"][i]))} is skipped')
            continue
        errs = Q_error[:,i]
        if np.sum((errs==0).astype(np.int32)>0) :
            print(f'{int(aman["detmap"]["channel"][i])}.{int(aman["detmap"]["band"][i])}.{int(aman["detmap"]["subband"][i])}_{int(np.nan_to_num(aman["detmap"]["bandpass"][i]))} is skipped')
            continue
        errs = U_error[:,i]
        if np.sum((errs==0).astype(np.int32)>0) : 
            print(f'{int(aman["detmap"]["channel"][i])}.{int(aman["detmap"]["band"][i])}.{int(aman["detmap"]["subband"][i])}_{int(np.nan_to_num(aman["detmap"]["bandpass"][i]))} is skipped')
            continue        
            
        chisquared = cost.LeastSquares(ang_mean, Q_mean[:,i], Q_error[:,i], funcQ) + cost.LeastSquares(ang_mean, U_mean[:,i], U_error[:,i], funcU)
        m = Minuit(chisquared, amp=(np.max(Q_mean[:,i])-np.min(Q_mean[:,i]))*0.5, freq=2., phase=0.5*np.pi, offsetQ=np.mean(Q_mean[:,i]), offsetU=np.mean(U_mean[:,i]))
        m.limits['amp'] = (0, None)
        m.limits['phase'] = (0.,np.pi)
        m.limits['freq'] = (2.,2.)
        m.simplex().migrad().migrad().hesse()

        fitted_amps.append(m.values[0])
        fitted_phases.append(m.values[2])
        fitted_offsetq.append(m.values[3])
        fitted_offsetu.append(m.values[4])
        error_amps.append(m.errors[0])
        error_phases.append(m.errors[2])
        error_offsetq.append(m.values[3])
        error_offsetu.append(m.values[4])
        minuit_chi2.append(m.fmin.reduced_chi2)
        channels.append(aman["detmap"]["channel"][i])
        bands.append(aman["detmap"]["band"][i])
        subbands.append(aman["detmap"]["subband"][i])
        bandpasses.append(aman["detmap"]["bandpass"][i])
        
        if debug and (i%30 == 0) :
            fig = plt.figure()
            angle = np.arange(628)*0.01
            plt.errorbar(ang_mean,Q_mean[:,i],yerr=Q_error[:,i],fmt='.',color='r')
            plt.errorbar(ang_mean,U_mean[:,i],yerr=U_error[:,i],fmt='.',color='b')
            plt.plot(angle, funcQ(angle, m.values[0], m.values[1], m.values[2], m.values[3]), ls="--", label="fittedQ",color='r')
            plt.plot(angle, funcU(angle, m.values[0], m.values[1], m.values[2], m.values[4]), ls="--", label="fittedU",color='b')
            plt.show()      
            
    np_fitted_phases = np.array(fitted_phases[1:])
    np_minuit_chi2 = np.array(minuit_chi2[1:])
    
    if debug : 
        fig = plt.figure()
        plt.hist(np.rad2deg(np_fitted_phases[np_minuit_chi2>0.]),bins=200, range=(0.,180.))
        plt.xlim(0.,180.)
        plt.ylim(0.,50.)
        plt.show()
        fig2 = plt.figure()
        plt.hist(np_minuit_chi2,bins=500,range=(0.,5.))
        plt.yscale("log")
        plt.show()    
    
    ### Output 
    rows = zip(channels,bands,subbands,bandpasses,fitted_amps,fitted_phases,fitted_offsetq,fitted_offsetu,error_amps,error_phases,error_offsetq,error_offsetu,minuit_chi2)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
            

            
def summary_result(input_file='./hoge.txt') : 
    df = pd.read_csv(input_file)
    data = df.values
    ### detector angle
    fig1 = plt.figure()
    plt.hist(data[:,5],bins=200)
    plt.title(df.columns.values[5])
    ### chi2
    fig2 = plt.figure()
    plt.hist(data[:,12],bins=200)
    plt.title(df.columns.values[12])
    plt.yscale("log")
    ### amplitude
    fig3 = plt.figure()
    plt.hist(data[:,4],bins=200)
    plt.title(df.columns.values[4])
    plt.yscale("log")
    plt.show()
    

    
def main() : 
    config_file = "/mnt/so1/shared/site-pipeline/data_pkg/p10r1/g3tsmurf_hwp_config.yaml"
    stream_id = 'ufm_mv14'
    tag = 'obs,stream,wg_step_wTravel1'
    hk_dir = "/mnt/so1/data/ucsd-sat1/hk/"
    detmap_file = '/homes/kkiuchi/so_test/detmap_output/Mv14_tunefile/Mv14_tunefile_1668022305_readout.csv'
    result_file = './test.csv'
    aman = wg_init_setting(config_file=config_file,stream_id=stream_id,tag=tag,hk_dir=hk_dir)
    wg_demod_tod(aman)
    num_angle, wg_info = get_wg_angle(aman,debug=False)
    wrap_det_info(aman,detmap_file,debug=False)
    fit_angle(aman,wg_info,output_file=result_file,debug=False)
    #summary_result(input_file=result_file)

    
    
if __name__ == '__main__':
    main() 