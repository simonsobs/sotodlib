import numpy as np
import matplotlib.pyplot as plt 
import yaml
import os 
import argparse

from iminuit import Minuit, cost

from sotodlib import core, tod_ops
from sotodlib.tod_ops import apodize, detrend       
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.site_pipeline.util as sp_util
import sotodlib.io.metadata as io_meta
import sotodlib.hwp.hwp as hwp
from so3g.hk import load_range


def _get_wg_angle(aman, ang_offset=0., axis_offset=0., threshold=0.0001, step_length=10., fluctuaion= 0.2) :

    """
    Get the indices of wiregrid stationary periods and angle in the period.

    Parameters:
    - aman : AxisManager
    - ang_offset : float
        Optional. Offset of wiregrid encoder.
    - axis_offset : float
        Optional. Offset of wiregrid coordinate against the detector coordinate
    - threshold : float
        Optional. Threshold to judje if wiregrid encoder is moving.
    - steplength : float
        Optional. Length of each wiregrid rotation step in seconds.
    fluctuation : float
        Optional. The allowed fluctuation on steplength.

    Return:
    - wg_info :
        list that includes timestamps of start and end time and wiregrid angle
        of each step.
    """

    ### wiregrid info from housekeeping
    wg_angle =  aman["hkwg"]["wg_angle"] + np.deg2rad(axis_offset) - np.deg2rad(ang_offset)
    wg_angle[wg_angle>2.*np.pi] = wg_angle[wg_angle>2.*np.pi] - 2.*np.pi
    wg_angle[wg_angle<0.] = wg_angle[wg_angle<0.] + 2.*np.pi
    wg_timestamp = aman["hkwg"]["wg_timestamp"]
    wg_dt = np.mean(np.diff(wg_timestamp))

    ### Get stationary part
    diff_wgangle = np.diff(wg_angle)
    stationary = (diff_wgangle<threshold).astype(np.int32)
    ones = np.where(stationary)[0]
    diff = np.diff(ones)

    _min = step_length/wg_dt*(1.-fluctuaion)
    _max = step_length/wg_dt*(1.+fluctuaion)
    starts = np.insert(ones[np.where(diff != 1)[0]+1], 0, ones[0])
    ends = np.append(ones[np.where(diff != 1)[0] ], ones[-1])
    indices = list(zip(starts, ends))
    indices = np.array([(start, end) for start, end in indices if ((end - start + 1 >= _min) and (end - start + 1 < _max))])

    ### Generate a list that contents timestamps of
    ### wg stationary part {start/end) and angle of wg
    wg_info = []
    for _id in range(len(indices)) :
        angle = np.mean(wg_angle[indices[_id][0]:indices[_id][1]])
        wg_info.append([wg_timestamp[indices[_id][0]],wg_timestamp[indices[_id][1]],angle])

    ### add the last part (16th step)
    ## we cannot find when is the end of steppig rotation from hk information
    ## so length of previous step is used
    length = len(indices)
    start_ts = indices[length-1][0] + 1 + indices[-1][0] - indices[-2][1]
    last_ts = indices[length-1][1] - indices[length-2][1] + start_ts
    angle = np.mean(wg_angle[start_ts:last_ts])
    wg_info.append([wg_timestamp[start_ts], wg_timestamp[last_ts],angle])
    indices = np.append(indices,[start_ts,last_ts])

    return wg_info


def fit_angle(aman,fitting_results,wg_info):

    """
    Get the indices of wiregrid stationary periods and angle in the period.

    Parameters:
    - aman : AxisManager
    - fitting_result : metadata.ResultSet object
        Fitting result holder
    - wg_info : list
        output from _get_wg_angle()

    Return:
    - fitting_results :
        Fitting results
    """

    ### Get compoents of wg_info then calculate polarized angle
    timestamp = aman["timestamps"]
    ang_mean,Q_mean,U_mean,Q_error,U_error=[],[],[],[],[]
    for _id in range(len(wg_info)) :
        wg_angle = wg_info[_id][2]
        t_selection = (aman.timestamps > wg_info[_id][0]) & (aman.timestamps < wg_info[_id][1])

        ### time constant correction
        Q = aman.demodQ[:,t_selection]
        U = aman.demodU[:,t_selection]

        ### In the future, we may want to correct time constant effect
        """
        if rotate_tau :
            exp = np.exp(1j*t_const*hwp_speed*2.*np.pi*4.).reshape((len(t_const), 1)) #  Need to update this later
        else :
            exp = 1.

        rotated_tod = exp*(Q+1j*U)
        Q = np.real(rotated_tod)
        U = np.imag(rotated_tod)
        """

        ### get sin curve
        # factor 56 is determined using lab test to make chi2 ~ 1
        # need to revise with site environment
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
            _id = aman["det_info"]["readout_id"][i]
            print("invalid TOD : ",  _id )
            fitting_results.rows.append((_id,np.nan))
            continue

        Qi,Ui = Q_mean[:,i],U_mean[:,i]
        Qi_e,Ui_e = Q_error[:,i],U_error[:,i]

        def fcn(amp,phase,offsetQ,offsetU):
            model1 = amp * np.cos ( 2.*ang_mean + 2.*phase) + offsetQ
            model2 = amp * np.cos ( 2.*ang_mean + 2.*phase + np.pi*0.5) + offsetU
            chi_squared1 =  ((model1-Qi)/Qi_e)*((model1-Qi)/Qi_e)
            chi_squared2 =  ((model2-Ui)/Ui_e)*((model2-Ui)/Ui_e)
            return np.sum(chi_squared1)+np.sum(chi_squared2)

        m = Minuit(fcn, amp=(np.max(Q_mean[:,i])-np.min(Q_mean[:,i]))*0.5, phase=0.5*np.pi, offsetQ=np.mean(Q_mean[:,i]), offsetU=np.mean(U_mean[:,i]))
        m.limits['amp'] = (0, None)
        m.limits['phase'] = (-np.pi,2.*np.pi)
        m.migrad().migrad().hesse()

        _id = aman["det_info"]["readout_id"][i]
        _amp = m.values[0]
        _amp_e = m.errors[0]
        _ang = m.values[1]
        if _ang > np.pi : _ang -= np.pi
        elif _ang < 0. : _ang += np.pi
        _ang_e = m.errors[1]
        _off_q = m.values[2]
        _off_u = m.values[3]
        ndf = len(Qi) + len(Ui) - len(m.values)
        _chi2 = m.fval/ndf

        #fitting_results.rows.append((_id,_amp,_amp_e,_ang,_ang_e,_off_q,_off_u,_chi2))
        fitting_results.rows.append((_id,_amp))

    return fitting_results


def _wg_demod_tod(aman) :

    ### If axis manager already has demomded components, skip the process
    if ("demodQ" in aman.keys()) or ("demodU" in aman.keys()) : return

    ### Demod
    detrend.detrend_tod(aman,method='median')
    apodize.apodize_cosine(aman)
    hwp.demod_tod(aman,signal_name='signal')

    return


def _wrap_wg_angle_info(aman, config) :

    """
    Get the indices of wiregrid stationary periods and angle in the period.

    Parameters:
    - aman : AxisManager
    - fitting_result : metadata.ResultSet object
        Fitting result holder
    - wg_info : list
        output from _get_wg_angle()

    Return:
    - fitting_results :
        Fitting results
    """

    wg_max_enc = 52000. # This is constant determined by the hardware

    wg_fields_str = config["wg_fields"]
    wg_fields = []
    wg_fields.append(wg_fields_str)

    hk_dir = config["hk_dir"]
    hk_in = load_range(float(aman.timestamps[0]), float(aman.timestamps[-1]), wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in[wg_fields_str]

    wg_ang = wg_enc/wg_max_enc*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)

    return
