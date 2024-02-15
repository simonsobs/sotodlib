#!/usr/bin/python3

# import modules
# python standard
import os
import sys
import scipy
from scipy import odr
import numpy as np
from iminuit import cost, Minuit
from matplotlib.patches import Ellipse, Circle

# SO specific
from sotodlib import core
from sotodlib.core import Context
from sotodlib.io import hk_utils
import sotodlib.io.load_smurf as ls
from sotodlib.io.load_smurf import Observations

wg_degpercount = 360/52000

def get_det_angle(aman, wg_ctx, wg_offset, stopped_time):
    """
    Calibrate TODs and get det_angle_raw by using the wire grid.
    Before using this code, your axis manager must have two field implemented by HWP, i.e., demodQ and demodU

    Parameters:
    - aman : AxisManager
    - wg_ctx : yaml file for the settings to load the wire grid meta data
        e.g. wg_ctx = '/so/home/hnakata/scratch/metadata/context/2401/2401_wg_satp1.yaml'
    - wg_offset : the offset angle of wires defined by the hardware of the Grid Loader
    - stopped_time : the stopped time of your target operation
    Return:
    - aman : wraped with the calibration data of the wire grid. Fields are explained in each method.
    """
    _config = wg_ctx
    _offset = wg_offset
    _stopped_time = stopped_time
    #
    # wrap the house-keeping data of the wire grid
    aman = wg_wrap_hk(aman, _config)
    #
    # restrict AxisManger within the operation range
    idx_wg_inside = _get_operation_range(aman)
    aman = aman.restrict('samps', (idx_wg_inside[0], idx_wg_inside[-1]))
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

# wrap HouseKeeping data
def wg_wrap_hk(aman, wg_ctx):
    """
    Wrap the house-keeping data about the wire grid operation.

    Parameters:
    - aman : AxisManager
    - wg_ctx : same as the definition in the get_det_angle

    Return:
    - aman : AxisManager
        This includes fields, which are related with the wire grid hardware.
        enc_deg : Encoder counts in degree of the rotation
        LSL1 : Status of the limit switch LEFT 1 of the actuator
        LSL2 : Status of the limit switch LEFT 2 of the actuator
        LSR1 : Status of the limit switch RIGHT 1 of the actuator
        LSR2 : Status of the limit switch RIGHT 2 of the actuator
    """
    wgctx = Context(wg_ctx)
    metakeys = wgctx['metakey']
    wgfield = wgctx['wgfield']
    agent_names = wgctx['wg_agent_names']
    enc = agent_names['encoder']
    act = agent_names['actuator']
    #
    wgenc = hk_utils.get_detcosamp_hkaman(aman, alias=['enc_deg'], fields = [wgfield['encoder']], data_dir = metakeys['hk_dir'])
    wglsl1 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL1'], fields = [wgfield['LSL1']], data_dir = metakeys['hk_dir'])
    wglsl2 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchL2'], fields = [wgfield['LSL2']], data_dir = metakeys['hk_dir'])
    wglsr1 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchR1'], fields = [wgfield['LSR1']], data_dir = metakeys['hk_dir'])
    wglsr2 = hk_utils.get_detcosamp_hkaman(aman, alias=['limitswitchR2'], fields = [wgfield['LSR2']], data_dir = metakeys['hk_dir'])
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
    """
    Define the range of the wire grid operation by the limit switches
    """
    idx_wg_inside = np.where(aman.wg.LSR2.astype(int) | aman.wg.LSL2.astype(int))[0]
    return idx_wg_inside

# Detect the motion of the wire grid
def _detect_action(count, flag=0):
    """
    Detect the rotation of the grid.

    Parameters:
    - count : encoder actual counts
    - flag : flag to choose moving or static

    Return:
    - counts in the same status related with the flag
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
    Detect all steps of the stepwise operation

    Parameters:
    - aman : AxisManager
    - stopped_time : the stopped time of your target operation
    - threshold : the thresholds on the encoder counts.
        the first element is the upper bound for the static state,
        and the second element is the lower bound for the difference of the first

    Return:
    - (ts_step_start, ts_step_stop) : time stamps of the start/stop for each step
    - (ang_av, ang_std) : average angles and its standard deviations for each step
    """
    if thresholds is None:
        thresholds = (5, 300) # upper bound of the static, lower bound of its diff
        pass
    _cdiff0 = np.where(_detect_action(aman.wg.enc_deg/wg_degpercount) < thresholds[0], 1, 0)
    _cdiff1 = np.where(_detect_action(_cdiff0, flag=1) > thresholds[1], 0, 1)
    _cdiff2 = _cdiff1[1:] - _cdiff1[:-1]
    step_stop = np.where(_cdiff2 == 1)[0]
    step_start = np.where(_cdiff2 == -1)[0]
    #
    ang_av = []
    ang_std = []
    step_size = np.ceil(stopped_time/np.average(np.diff(aman.timestamps))).astype(int)
    step_stop = np.append(step_stop, step_stop[-1]+step_size)
    for _i in range(len(step_start)):
        ang_av.append(np.average(aman.wg.enc_deg[step_start[_i]:step_stop[_i+1]]))
        ang_std.append(np.std(aman.wg.enc_deg[step_start[_i]:step_stop[_i+1]]))
        pass
    ts_step_start = aman.timestamps[step_start]
    ts_step_stop = aman.timestamps[step_stop]
    ang_av = np.array(ang_av)
    ang_std = np.array(ang_std)
    #
    return (ts_step_start, ts_step_stop), (ang_av, ang_std)

# Wrap the QU response during the operation of the Grid Loader
def wg_wrap_QU(aman, stopped_time, thresholds=None):
    """
    Wrap QU signal by the wire grid.

    Parameters:
    - aman : AxisManager
    - stopped_time : the stopped time of your target operation
    - threshold : the thresholds on the encoder counts.
        the first element is the upper bound for the static state,
        and the second element is the lower bound for the difference of the first

    Return:
    - aman : AxisManger
        This includes the characterics of the wire grid operation and the Q/U signal
        related with it.
        flag_step_start : the start time stamps for the steps
        flag_step_stop : the stop time stamps for the steps
        enc_ang_deg : encouder counts in degreee for each step
        enc_ang_std : the standard deviations of the encoder counts for steps
        Q, U : Q (and U) signal for steps
        Qerr, Uerr : the standard deviations of Q (and U) signal for steps
    """
    if hasattr(aman, 'demodQ') == False:
        print("This AxisManager does not have demodQ/demodU.")
        print("Please call this method after you have demodulated the signal.")
        return False
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
    """
    Get the initial parameters as inputs of the circle fitting in Q-U plane

    Paramter:
    - x : assuming ndarray of Q and U signal with 16 steps in nominal, np.array([Q(16), U(16)])

    Return:
    - params : the initial paramters for the circle fit
    """
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
    """
    Fit funciton

    Parameter:
    - params : the input parameters
    - x : data set

    Return:
    - residual between the data point and the function under the input paramters
    """
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
    """
    Fit the data with a circle in a complex plane

    Parameters:
    - obs_data : ndarray of Q and U signal with 16 steps in nominal, np.array([Q(16), U(16)])
    - std_data : ndarray of the standard deviations of Q and U, np.array([Qerr(16), Uerr(16)])
    - fitfunc : a function used for this fitting, basically assumed the function of _circle_resid
    - param0 : the initial paramer set

    Return:
    - fit result determined by the implicit fitting of scipy.odr
    """
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
    """
    one dimention linear funciton mod by 360 [degee]
    """
    return (a*x + b)%360

def wg_get_cfitres(aman):
    """
    Get the results by the circle fitting about the responce against the wires in Q+iU plane

    Parameters:
    - aman : AxisManager

    Return:
    - aman : AxisManager
        This includes fit results for all the inpput detectors
        cfitval : Estimated parameter values by fitting
        cfitcov : covariance matrix of the estimated parameters
        cfitresvar : Residual variance
    """
    obs_data = []
    obs_std = []
    fit_results = []
    cfitval = []
    cfitcov = []
    cfitresvar = []
    for _i in range(aman.dets.count):
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
    """
    Get the results by the linear fitting, which is comparing the reference counts of the wire grid
    and the angle of the respose in Q+iU plane

    Parameters:
    - aman : AxisManager

    Return:
    - aman : AxisManager
        This includes fit results for all the inpput detectors
        lfitval : Estimated parameter values by fitting
        lfitcov : covariance matrix of the estimated parameters
        lfitresvar : Residual variance
    """
    lfitval = []
    lfiterr = []
    lfitchi2 = []
    for _i in range(aman.dets.count):
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

def main():
    print("This is the module for the calibration using the Sparse Wire Grid.\n \
        Please excute from sotodlib.wiregrid.calibrate import get_det_angle ")
    pass

if __name__=='__main__':
    main()
