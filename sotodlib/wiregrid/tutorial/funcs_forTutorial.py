#!/usr/bin/python3

# import modules
# python standard
import os
import sys
import asyncio
import scipy
from scipy import odr
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle


import matplotlib.pyplot as plt

# SO specific
from sotodlib import core
from sotodlib.core import Context
from sotodlib.io import hk_utils
import sotodlib.hwp.hwp as hwp
from sotodlib.tod_ops import apodize, detrend
import sotodlib.io.load_smurf as ls
from sotodlib.io.load_smurf import Observations
from sotodlib.hwp.g3thwp import G3tHWP

wg_degpercount = 360/52000
stream_ids = ['ufm_mv19', 'ufm_mv18', 'ufm_mv22', 'ufm_mv29', 'ufm_mv7', 'ufm_mv9', 'ufm_mv15'] # e.g. MF1

def get_det_angle(aman, **kwargs):
    _config = kwargs['wg_config']
    _offset = kwargs['wg_offset']
    _stopped_time = kwargs['stopped_time']
    #
    # wrap the house-keeping data of the wire grid
    aman = wg_wrap_hk(aman, _config)
    #
    # restrict AxisManger within the operation range
    idx_wg_inside = _get_operation_range(aman)
    aman = aman.restrict('samps', (idx_wg_inside[0], idx_wg_inside[-1]), in_place=False)
    #
    # wrap the demodQ/demodU related with the steps of the wire grid
    # AxisManager here must have demodQ/demodU by the HWP
    aman = wg_wrap_QU(aman, _stopped_time)
    #
    # fit the QU signal created by the wire grid
    aman = wg_get_cfitres(aman)
    aman = wg_get_lfitres(aman)
    #
    # get the det angles
    det_angle_raw = 0.5*(aman.wg.lfitval[:,1] + aman.wg.lfitval[:,0]*(_offset))%180
    det_angle_raw_err = 0.5*aman.wg.lfiterr[:,1] # temporal value
    #
    aman.wg.wrap('det_angle_raw', det_angle_raw, [(0, 'dets')])
    aman.wg.wrap('det_angle_raw_err', det_angle_raw_err, [(0, 'dets')])
    #
    return aman

def obsid2datetime(obs_id):
    from datetime import datetime
    # parse obs_id and convert datetime to string at UTC
    #obs_id = 'obs_1705878834_satp1_1111111'
    obs_id = obs_id.split('_')
    obs_id = int(obs_id[1][0:10])
    date_op = datetime.utcfromtimestamp(obs_id).strftime('%Y%m%d_%H%M%S')[2:]
    return date_op

# get AxisManager with the specified obs_id and the stream_id
def get_aman(ctx, obs_id, stream_id):
    if stream_id == None:
        stream_id = stream_ids[0]
        print("Warning:Please define a stream_id like 'ufm_mv19'. Now loading from stream_id sets for MF1.")
        pass
    am0 = ctx.get_obs(obs_id, no_signal=True)
    dmask = np.isin(am0.det_info.stream_id, stream_id)
    dets = am0.dets.vals[dmask]
    am = ctx.get_obs(obs_id, dets=dets)
    return am

# wrap HouseKeeping data
def wg_wrap_hk(aman, wg_config):
    """
    """
    wgctx = Context(wg_config)
    metakeys = wgctx['metakey']
    wgfield = wgctx['wgfield']
    agent_names = wgctx['wg_agent_names']
    enc = agent_names['encoder']
    act = agent_names['actuator']
    #
    wgenc = hk_utils.get_detcosamp_hkaman(aman, alias=['enc_deg'], fields = [wgfield['encoder']], data_dir = metakeys['hk_dir'])
    wglsl1 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL1'], fields = [wgfield['LSL1']], data_dir = metakeys['hk_dir'])
    wglsl2 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL1'], fields = [wgfield['LSL2']], data_dir = metakeys['hk_dir'])
    wglsr1 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL1'], fields = [wgfield['LSR1']], data_dir = metakeys['hk_dir'])
    wglsr2 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL1'], fields = [wgfield['LSR2']], data_dir = metakeys['hk_dir'])
    #
    wg_ax = core.AxisManager(aman.samps, aman.dets)
    wg_ax.wrap('enc_deg', wgenc[enc][enc][0]*wg_degpercount, [(0,'samps')])
    wg_ax.wrap('LSL1', wglsl1[act][act][0], [(0,'samps')])
    wg_ax.wrap('LSL2', wglsl2[act][act][0], [(0,'samps')])
    wg_ax.wrap('LSR1', wglsr1[act][act][0], [(0,'samps')])
    wg_ax.wrap('LSR2', wglsr2[act][act][0], [(0,'samps')])
    aman.wrap('wg', wg_ax)
    return aman

def _get_operation_range(aman):
    idx_wg_inside = np.where(aman.wg.LSR2.astype(int) | aman.wg.LSL2.astype(int))[0]
    return idx_wg_inside

def plot_wg_op(obs_id, aman, idx_wg_inside, odir=None):
    """
    """
    title = obs_id + ":wg house-keeping characteristics"
    fig, ax1 = plt.subplots(figsize=(10,9), tight_layout=True)
    fig.suptitle(title)
    plt.rcParams['font.size'] = 12
    ax1.plot(aman.timestamps, aman.wg.enc_deg, alpha=0.4, lw=2, label="wg-encoder\n wire angle")
    ax1.set_xlabel("unixtime [sec]")
    ax1.set_ylabel("wire angle [deg]")
    #
    ax2 = ax1.twinx()
    ax2.plot(aman.timestamps, aman.wg.LSL1, lw=2, label="LS-outside-LEFT")
    ax2.plot(aman.timestamps, aman.wg.LSL2, lw=2, label="LS-inside-LEFT")
    ax2.plot(aman.timestamps, aman.wg.LSR1, lw=2, label="LS-outside-RIGHT")
    ax2.plot(aman.timestamps, aman.wg.LSR2, lw=2, label="LS-inside-RIGHT")
    ax2.set_ylabel("limitswitch status")
    ax2.set_yticks([0,1.], ['OFF','ON'])
    #
    ax2.axvspan(aman.timestamps[idx_wg_inside[0]], aman.timestamps[idx_wg_inside[-1]],
                color='k', alpha=.1, label='roi_in')
    #
    ax1.legend(loc='center')
    ax2.legend(loc='lower left')
    #
    date_op = obsid2datetime(obs_id)
    if odir is not None:
        os.makedirs(odir, exist_ok=True)
        save_path = odir+date_op+"_wghk.png"
        fig.savefig(save_path)
        plt.clf()
        plt.close()

# wrap_hwp_angle
def wrap_hwp_angle(aman, wg_config, hwp_config):
    """
    """
    wgctx = Context(wg_config)
    data_dir = wgctx['metakey']['hk_dir']
    gh = G3tHWP(config_file=hwp_config)
    hwp_data = gh.load_data(float(aman.timestamps[0])-1,
                            float(aman.timestamps[-1])+1,
                            data_dir=data_dir)
    solved = gh.analyze(hwp_data, mod2pi=False)
    #
    angle_resampled = scipy.interpolate.interp1d(solved['fast_time'], solved['angle'],
                                                 kind='linear', fill_value='extrapolate')(aman.timestamps)
    angle_resampled = np.mod(angle_resampled, 2*np.pi)
    aman.wrap('hwp_angle', angle_resampled, [(0, 'samps')])
    return aman

# Detect the motion of the wire grid
def _detect_action(count, flag=0):
    """
    """
    if flag == 0:
        vecbool = count[1:]-count[:-1] != 0  # moving
        pass
    else:
        vecbool = count[1:]-count[:-1] == 0  # static
    lvec = len(vecbool)
    inv_vec = vecbool[::-1]
    left = np.maximum.accumulate(
        list(map(lambda x: 0 if vecbool[x] == True else x, np.arange(lvec)))
    )
    right = lvec - 1 - np.maximum.accumulate(
        list(map(lambda x: 0 if inv_vec[x] == True else x, np.arange(lvec)))
    )[::-1]
    return right-left

# Define the timestamp and the wire angle in each step
def _detect_steps(aman, stopped_time=10, thresholds=None):
    """
    """
    if thresholds is None:
        thresholds = (5, 300) # upper bound of the static, lower bound of its diff
        pass
    cdiff0 = np.where(_detect_action(aman.wg.enc_deg/wg_degpercount) < thresholds[0], 1, 0)
    cdiff1 = np.where(_detect_action(cdiff0, flag=1) > thresholds[1], 0, 1)
    cdiff2 = cdiff1[1:] - cdiff1[:-1]
    step_end = np.where(cdiff2 == 1)[0]
    step_start = np.where(cdiff2 == -1)[0]
    #
    wg_av = []
    wg_std = []
    step_size = np.ceil(stopped_time/np.average(np.diff(aman.timestamps))).astype(int)
    step_end = np.append(step_end, step_end[-1]+step_size)
    for _i in range(len(step_start)):
        wg_av.append(np.average(aman.wg.enc_deg[step_start[_i]:step_end[_i+1]]))
        wg_std.append(np.std(aman.wg.enc_deg[step_start[_i]:step_end[_i+1]]))
        pass
    ts_step_start = aman.timestamps[step_start]
    ts_step_end = aman.timestamps[step_end]
    wg_av = np.array(wg_av)
    wg_std = np.array(wg_std)
    #
    return (ts_step_start, ts_step_end), (wg_av, wg_std)

# Wrap the QU response during the operation of the Grid Loader
def wg_wrap_QU(aman, stopped_time, thresholds=None):
    """
    """
    ts_step, wg_deg = _detect_steps(aman, stopped_time, thresholds)
    ts_instep = []
    step_Q = []
    step_U = []
    step_Qerr = []
    step_Uerr = []
    for _i in range(len(wg_deg[0])):
        instep = np.where(ts_step[0][_i] < aman.timestamps, True, False)
        instep = np.where(aman.timestamps < ts_step[1][_i+1], instep, False)
        #
        ts_instep.append(np.average(aman.timestamps[instep]))
        step_Q.append(np.average(aman.demodQ, axis=1, weights=instep))
        step_U.append(np.average(aman.demodU, axis=1, weights=instep))
        step_Qerr.append(np.std(aman.demodQ[:, instep == True], axis=1))
        step_Uerr.append(np.std(aman.demodU[:, instep == True], axis=1))
        pass
    ts_instep = np.array(ts_instep)
    step_Q = np.array(step_Q)
    step_U = np.array(step_U)
    step_Qerr = np.array(step_Qerr)
    step_Uerr = np.array(step_Uerr)
    #
    aman.wg.wrap('flag_step_start', ts_step[0])
    aman.wg.wrap('flag_step_stop', ts_step[1])
    aman.wg.wrap('enc_ang_deg', wg_deg[0])
    aman.wg.wrap('enc_ang_std', wg_deg[1])
    aman.wg.wrap('ts_instep', ts_instep)
    aman.wg.wrap('Q', step_Q, [(1, 'dets')])
    aman.wg.wrap('U', step_U, [(1, 'dets')])
    aman.wg.wrap('Qerr', step_Qerr, [(1, 'dets')])
    aman.wg.wrap('Uerr', step_Uerr, [(1, 'dets')])
    return aman

def _get_initial_param_circle(x):
    if len(np.shape(x)) == 2:
        A = np.average(x[0])
        B = np.average(x[1])
        C = np.average(x[0]**2+x[1]**2)
        params = np.array([A, B, C])
        return params
    elif len(np.shape(x)) == 3:
        (A, B) = np.average(x, axis=2)
        C = np.average(x[0]**2+x[1]**2, axis=1)
        params = np.array([A, B, C])
        return params
    else:
        print("This input vector is not valid shape.")
        return False

def _circle_resid(params, x):
    # (x-A)^2 + (y-B)^2 - C = 0
    if len(np.shape(x)) == 2:
        real = x[0]
        imag = x[1]
        A = params[0]
        B = params[1]
        C = params[2]
        return (real - A)**2 + (imag - B)**2 - C
    elif len(np.shape(x)) == 3:
        real = x[0]
        imag = x[1]
        N, M = np.shape(real)
        A = params[0]
        B = params[1]
        C = params[2]
        return (real2 - np.diag(A))**2 + (imag2 - np.diag(B))**2 - np.reshape(np.tile(C, M), (M, N)).T

def _comp_plane_fit(obs_data, std_data, fitfunc, param0):
    if len(np.shape(obs_data)) == 2:
        mdr = odr.Model(fitfunc, implicit=True)
        mydata = odr.RealData(obs_data, y=1, sx=std_data)
        myodr = odr.ODR(mydata, mdr, beta0=param0)
        myoutput = myodr.run()
        return myoutput
    elif len(np.shape(obs_data)) == 3:
        alloutput = []
        for i in range(np.shape(obs_data)[1]):
            mdr = odr.Model(fitfunc, implicit=True)
            mydata = odr.RealData(obs_data[:, i, :], y=1, sx=std_data[:, i, :])
            myodr = odr.ODR(mydata, mdr, beta0=param0[:, i])
            myoutput = myodr.run()
            alloutput.append(myoutput)
        return alloutput
    else:
        print("This input vector is not valid shape.")
        return False

def _linear1d(x, a, b):
    return (a*x + b)%360

def wg_get_cfitres(aman):
    obs_data = []
    obs_std = []
    fit_results = []
    cfitval = []
    cfitcov = []
    cfitresvar = []
    for _i in range(np.shape(aman.det_info)[0]):
        obs_data.append(np.array([aman.wg.Q[:, _i], aman.wg.U[:, _i]]))
        obs_std = np.array([aman.wg.Qerr[:, _i], aman.wg.Uerr[:, _i]])
        params_init = _get_initial_param_circle(obs_data[-1])
        fit_results.append(
            _comp_plane_fit(obs_data[-1], obs_std, _circle_resid, params_init))
        cfitval.append(
            np.array([fit_results[-1].beta[0], fit_results[-1].beta[1], np.sqrt(fit_results[-1].beta[2])]))
        cfitcov.append(fit_results[-1].cov_beta)
        cfitresvar.append(fit_results[-1].res_var)
        pass
    aman.wg.wrap('cfitval', np.array(cfitval), [(0, 'dets')])
    aman.wg.wrap('cfitcov', np.array(cfitcov), [(0, 'dets')])
    aman.wg.wrap('cfitresvar', np.array(cfitresvar), [(0, 'dets')])
    return aman

def wg_get_lfitres(aman):
    lfitval = []
    lfiterr = []
    lfitchi2 = []
    for _i in range(np.shape(aman.det_info)[0]):
        _ang_demodQU = np.rad2deg(np.arctan2(aman.wg.U[:,_i] - aman.wg.cfitval[_i][1], aman.wg.Q[:,_i] - aman.wg.cfitval[_i][0]))
        yerr_temp = np.sqrt(aman.wg.Qerr[:,_i]**2 + aman.wg.Uerr[:,_i]**2) # this is a temporal value, NOT correct
        c = cost.LeastSquares((aman.wg.enc_ang_deg+90)%180-90, _ang_demodQU%360, yerr_temp, _linear1d)
        m = Minuit(c, a=2, b=(_ang_demodQU%360)[np.argsort(aman.wg.enc_ang_deg%180)[0]] -2*min(aman.wg.enc_ang_deg%180))
        m.limits['a'] = (-2.1,2.1)
        m.limits['b'] = (-180,480)
        m.migrad()
        lfitval.append([m.values[0], m.values[1]%360])
        lfiterr.append([m.errors[0], m.errors[1]])
        lfitchi2.append(m.fmin.reduced_chi2)
        pass
    aman.wg.wrap('lfitval', np.array(lfitval), [(0, 'dets')])
    aman.wg.wrap('lfiterr', np.array(lfiterr), [(0, 'dets')])
    aman.wg.wrap('lfitchi2', np.array(lfitchi2), [(0, 'dets')])
    return aman

def plot_wg_charac(aman, detid, odir=None):
    _i = detid
    fig, axarr = plt.subplots(1,3,figsize=(24,8),tight_layout=True)
    ax = axarr.ravel()
    fig.suptitle(f'test:{_i:04d}')
    #
    # plot QU response against time
    #
    ax[0].plot(aman.timestamps, aman.demodQ[_i], label="demodQ")
    ax[0].plot(aman.timestamps, aman.demodU[_i], label="demodU")
    for _j in range(len(aman.wg.flag_step_start)):
        ax[0].axvline(aman.wg.flag_step_start[_j], lw=1, ls=':', color='lime', label='step_start' if _j==0 else None)
        ax[0].axvline(aman.wg.flag_step_stop[_j+1], lw=1, ls='-.', color='g', label='step_stop' if _j==0 else None)
    hans, labs = ax[0].get_legend_handles_labels()
    l1 = ax[0].legend(handles=hans[:2], labels=labs[:2],loc='upper right')
    l2 = ax[0].legend(handles=hans[2:], labels=labs[2:],loc='upper left')
    ax[0].add_artist(l1)
    #
    # plot the circle in the QU plane
    #
    ax[1].errorbar(aman.wg.Q[:,_i], aman.wg.U[:,_i], xerr=aman.wg.Qerr[:,_i],
                yerr=aman.wg.Uerr[:,_i], capsize=1, fmt='ob', markersize=5)
    circle = Circle(xy=(aman.wg.cfitval[_i][0], aman.wg.cfitval[_i][1]),
                    radius=aman.wg.cfitval[_i][2], fc="none", ec='r')
    ax[1].scatter(aman.wg.cfitval[_i][0], aman.wg.cfitval[_i][1], marker='+', \
               label=f"cx={aman.wg.cfitval[_i][0]:.5f},\n cy={aman.wg.cfitval[_i][1]:.5f},\n cr={np.sqrt(aman.wg.cfitval[_i][2]):.5f}")
    ax[1].add_artist(circle)
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlabel('demodQ [a.u.]')
    ax[1].set_ylabel('demodU [a.u.]')
    ax[1].grid()
    ax[1].legend(loc='upper right')
    #
    # plot the wangle-detQU correspondence
    #
    QU_ang = np.rad2deg(np.arctan2(aman.wg.U[:,_i] - aman.wg.cfitval[_i][1], aman.wg.Q[:,_i] - aman.wg.cfitval[_i][0]))
    ax[2].scatter((aman.wg.enc_ang_deg+90)%180-90, QU_ang%360)
    yerr_temp = np.sqrt(aman.wg.Qerr[:,_i]**2 + aman.wg.Uerr[:,_i]**2)# this is not correct
    ax[2].errorbar((aman.wg.enc_ang_deg+90)%180-90, QU_ang%360, xerr=aman.wg.enc_ang_std, yerr=yerr_temp, fmt='ob', markersize=5,
                   label=f'a={aman.wg.lfitval[_i][0]:.3f},\n b={aman.wg.lfitval[_i][1]:.3f},\n chi2={aman.wg.lfitchi2[_i]:.3f}')
    ax[2].plot(np.linspace(-90,90,100), _linear1d(np.linspace(-90,90,100), aman.wg.lfitval[_i][0], aman.wg.lfitval[_i][1]), color='r')
    ax[2].set_xlabel('wire angle [deg]')
    ax[2].set_ylabel('pol-respons angle in QU-plane [deg]')
    ax[2].legend(loc='upper right')
    ax[2].set_xticks(np.linspace(-90,90,7), np.linspace(-90,90,7))
    ax[2].set_yticks(np.linspace(0,360,9), np.linspace(0,360,9))
    ax[2].grid()
    if odir is not None:
        save_path = odir+f'det{_i:04d}.png'
        fig.savefig(save_path)
        plt.clf()
        plt.close()
        pass

def main():
    print("This is the module for the calibration using the Sparse Wire Grid.\n \
        Please excute from sotodlib.wiregrid.calibrate import get_det_angle ")
    pass

if __name__=='__main__':
    main()
