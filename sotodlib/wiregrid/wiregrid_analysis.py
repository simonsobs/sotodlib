import numpy as np
import matplotlib.pyplot as plt 
import yaml
import os 
import sqlite3
import argparse
import sys

from iminuit import Minuit, cost

from sotodlib import core, tod_ops
from sotodlib.tod_ops import filters, fft_ops, apodize, detrend # FFT modules
import sotodlib.io.load_smurf as load_smurf
from sotodlib.io.load_smurf import load_file, G3tSmurf, Observations, SmurfStatus
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.site_pipeline.util as sp_util
import sotodlib.io.metadata as io_meta
import sotodlib.hwp.hwp as hwp
from so3g.hk import load_range, HKArchiveScanner


def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def get_wg_angle(aman, ang_offset=312.63, threshold=0.00015, plateau_len=3*5000, remove_len= 30*5000, debug=False) :
### Note : Length of plateau and in/ouf time of wg (remove_len) are determined in "hard" coding.
###        You may have to check if the length is ok to analyze recent wg calibration.
###        Need to update.

    wg_angle =  aman["hkwg"]["wg_angle"] + np.deg2rad(87.) - np.deg2rad(ang_offset)
    wg_angle[wg_angle>2.*np.pi] = wg_angle[wg_angle>2.*np.pi] - 2.*np.pi
    wg_angle[wg_angle<0.] = wg_angle[wg_angle<0.] + 2.*np.pi
    wg_timestamp = aman["hkwg"]["wg_timestamp"]

    ### Get static part
    diff_wgangle = np.diff(wg_angle)
    if debug == True :

        fig = plt.figure()
        plt.plot(wg_angle)
        plt.show()
        fig = plt.figure()
        plt.plot(diff_wgangle)
        plt.show()
    moving = (diff_wgangle>threshold).astype(np.int32)
    switch_indices = np.where(np.diff(moving) != 0)[0] + 1
    run_lengths = np.diff(switch_indices)

    static_indices_start = switch_indices[:-1][run_lengths>plateau_len]
    static_indices_end = []
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


    ### Temporary to add the last part (16th step)
    length = len(static_indices_start)
    start_ts = static_indices_end[length-1] + 1 + static_indices_start[-1] - static_indices_end[-2]
    last_ts = static_indices_end[length-1] - static_indices_end[length-2] + start_ts
    angle = np.mean(wg_angle[start_ts:last_ts])
    wg_info.append([wg_timestamp[start_ts], wg_timestamp[last_ts],angle])
    static_indices_start = np.append(static_indices_start,start_ts)
    static_indices_end.append(last_ts)


    return len(static_indices_start),wg_info


def funcQ(x, amp, freq, phase, offsetQ):
    return amp * np.cos ( freq * x + 2.*phase) + offsetQ

def funcU(x, amp, freq, phase, offsetU):
    return amp * np.cos ( freq * x + 2.*phase + np.pi*0.5) + offsetU


def fit_angle(aman,fitting_results,wg_info,rotate_tau_wg=False,rotate_tau_tune=False,debug=False):
    
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
    ### NOTE :::  Must be updated. Should we make a db for timeconstant or use tuning file?   
    t_const = []
    if rotate_tau_wg :
        try : 
            import pickle
            with open('/temp/your_tau.pickle', 'rb') as fin :
            
                data = pickle.load(fin)

                for det in aman["det_info"]["readout_id"] :
                    if det.encode() in data.keys() : 
                        t_const.append(data[det.encode()])
                    else : 
                        t_const.append(0.)

            t_const = np.array(t_const)
        except : 
            print("No pickle file was found.")
            sys.exit()

    if rotate_tau_tune :
        if rotate_tau_wg : 
            print("Time constant info from wiregrid is not used.")
        try : 
            t_const = aman["tau_eff"]
        except : 
            print("loading tau_eff failed")
            sys.exit()

    ### Get compoents of wg_info then calculate polarized angle
    each_result = []
    ang_mean,Q_mean,U_mean,Q_error,U_error=[],[],[],[],[]
    for _id in range(len(wg_info)) : 
        wg_angle = wg_info[_id][2]
        t_selection = (aman.timestamps > wg_info[_id][0]) & (aman.timestamps < wg_info[_id][1])
           
        ### time constant correction
        Q = aman.demodQ[:,t_selection]
        U = aman.demodU[:,t_selection]
        
        if rotate_tau_tune or rotate_tau_wg : 
            exp = np.exp(1j*t_const*hwp_speed*2.*np.pi*4.).reshape((len(t_const), 1)) #  Need to update this later
        else : 
            exp = 1.
                
        rotated_tod = exp*(Q+1j*U)
        Q = np.real(rotated_tod)
        U = np.imag(rotated_tod)
        
        ### get sin curve
        # factor 56 is determined to make chi2 ~ 1
        ang_mean.append(wg_angle)
        Q_mean.append(np.mean(Q,axis=1))
        U_mean.append(np.mean(U,axis=1))
        Q_error.append(np.std(Q,axis=1)/np.sqrt(len(Q[0]))*np.sqrt(56.))
        U_error.append(np.std(U,axis=1)/np.sqrt(len(U[0]))*np.sqrt(56.))

    
    ang_mean = np.array(ang_mean)
    Q_mean = np.array(Q_mean)
    U_mean = np.array(U_mean)
    Q_error = np.array(Q_error)
    U_error = np.array(U_error)

    for i in range(len(Q_mean[0])) : 

        if (np.sum(np.isnan(Q_mean[:,i]).astype(np.int32)) > 0) or (np.sum(Q_error[:,i])==0) or (np.sum(U_error[:,i])==0): 
            print(aman["det_info"]["readout_id"][i])
            continue 

        Qi,Ui = Q_mean[:,i],U_mean[:,i]
        Qi_e,Ui_e = Q_error[:,i],U_error[:,i]                  
            
        def fcn(amp,freq,phase,offsetQ,offsetU):
            model1 = amp * np.cos ( freq * ang_mean + 2.*phase) + offsetQ 
            model2 = amp * np.cos ( freq * ang_mean + 2.*phase + np.pi*0.5) + offsetU
            chi_squared1 =  ((model1-Qi)/Qi_e)*((model1-Qi)/Qi_e)
            chi_squared2 =  ((model2-Ui)/Ui_e)*((model2-Ui)/Ui_e)
            return np.sum(chi_squared1)+np.sum(chi_squared2)
            
        m = Minuit(fcn, amp=(np.max(Q_mean[:,i])-np.min(Q_mean[:,i]))*0.5, freq=2., phase=0.5*np.pi, offsetQ=np.mean(Q_mean[:,i]), offsetU=np.mean(U_mean[:,i]))
        m.limits['amp'] = (0, None)
        m.limits['phase'] = (-np.pi,2.*np.pi)
        m.fixed['freq'] = True
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
   
        if debug and (i%100 == 0) :
            print(_chi2,m.values[1])
            fig = plt.figure()
            angle = np.arange(628)*0.01
            plt.errorbar(ang_mean,Q_mean[:,i],yerr=Q_error[:,i],fmt='.',color='r')
            plt.errorbar(ang_mean,U_mean[:,i],yerr=U_error[:,i],fmt='.',color='b')
            plt.plot(angle, funcQ(angle, m.values[0], m.values[1], m.values[2], m.values[3]), ls="--", label="fittedQ",color='r')
            plt.plot(angle, funcU(angle, m.values[0], m.values[1], m.values[2], m.values[4]), ls="--", label="fittedU",color='b')
            plt.show()
            
    return fitting_results


def wg_demod_tod(aman) :

    ### If axis manager already has demomded components, skip the process
    if ("demodQ" in aman.keys()) or ("demodU" in aman.keys()) : return

    ### Demod
    detrend.detrend_tod(aman,method='median')
    apodize.apodize_cosine(aman)
    hwp.demod_tod(aman,signal_name='signal')

    return
