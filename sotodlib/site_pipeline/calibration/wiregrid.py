# fundamental functions for the det_gamma calibration by the wire grid
# Please contact hnakata_JP(zhongtianjiaxin@gmail.com) if you have questions.

# import modules
# python standard
import numpy as np
from scipy import odr
from matplotlib.patches import Ellipse, Circle

# SO specific
from sotodlib import core
import sotodlib.io.load_smurf as ls
import sotodlib.hwp.hwp as hwp
from sotodlib.hwp.g3thwp import G3tHWP
# for house-keeping data
from sotodlib.io import hk_utils
from sotodlib.site_pipeline import util

wg_counts2rad = 2*np.pi/52000
wg_offset_satp1 = np.deg2rad(12.13) # SAT1, MF1
wg_offset_satp2 = np.deg2rad(9.473) # SAT2, UHF
wg_offset_satp3 = np.deg2rad(11.21) # TSAT, MF2

logger = util.init_logger('wiregrid', 'wiregrid: ')

# Wrap house-keeping data
def _wrap_wg_hk(tod, telescope=None, hk_dir=None):
    """
    Wrap the house-keeping data about the wire grid operation.

    Parameters
    ----------
        tod : AxisManager
        telescope : telescope type, e.g. satp1, satp3, etc.
            this parameter will basically be filled by the obs_info wrapped in the axismanager.

    Returns
    -------
        tod : AxisManager
            This includes fields, which are related with the wire grid hardware.
            enc_rad_raw : wires' direction read by encoder in radian (raw data from the encoder)
            LSL1 : ON/OFF status of the limit switch LEFT 1 (outside) of the actuator
            LSL2 : ON/OFF status of the limit switch LEFT 2 (inside) of the actuator
            LSR1 : ON/OFF status of the limit switch RIGHT 1 (outside) of the actuator
            LSR2 : ON/OFF status of the limit switch RIGHT 2 (inside) of the actuator

    """
    if telescope is None:
        try:
            telescope = tod.obs_info.telescope
        except:
            print("Cannot load the telescope info. Please assign your target telescope")
            logger.info("Telescope assignment was failed in site_pipeline.calibrate.wiregrid._wrap_wg_hk")
    #
    alias_dict = {
        'enc_rad_raw': telescope + '.wg-encoder.feeds.wgencoder_full.reference_count',
        'limitswitchL1': telescope + '.wg-actuator.feeds.wgactuator.limitswitch_LSL1',
        'limitswitchL2': telescope + '.wg-actuator.feeds.wgactuator.limitswitch_LSL2',
        'limitswitchR1': telescope + '.wg-actuator.feeds.wgactuator.limitswitch_LSR1',
        'limitswitchR2': telescope + '.wg-actuator.feeds.wgactuator.limitswitch_LSR2',
        }
    if hk_dir is None: hk_dir = '/so/level2-daq/' + telescope + '/hk'
    #
    _gl_aman = hk_utils.get_detcosamp_hkaman(tod,
                                     alias = [_i for _i in alias_dict.keys()],
                                     fields = [_j for _j in alias_dict.values()],
                                     data_dir = hk_dir)
    _gl_data = {}
    for _i in alias_dict.keys():
        if _i == 'enc_rad_raw':
            _gl_data[_i] = _gl_aman['wg-encoder']['wg-encoder'][0]
        else:
            _gl_data[_i] = _gl_aman.restrict('hklabels_wg-actuator', [_i], in_place=False)['wg-actuator']['wg-actuator'][0]
    #
    _ax_wg = core.AxisManager(tod.samps, tod.dets)
    _ax_wg.wrap('enc_rad_raw', _gl_data['enc_rad_raw'] * wg_counts2rad, [(0,'samps')])
    _ax_wg.wrap('LSL1', _gl_data['limitswitchL1'], [(0,'samps')])
    _ax_wg.wrap('LSL2', _gl_data['limitswitchL2'], [(0,'samps')])
    _ax_wg.wrap('LSR1', _gl_data['limitswitchR1'], [(0,'samps')])
    _ax_wg.wrap('LSR2', _gl_data['limitswitchR2'], [(0,'samps')])
    tod.wrap('wg', _ax_wg)
    return tod

def _get_operation_range(tod):
    """
    Define the range of the wire grid operation by the limit switches
    """
    idx_wg_inside = np.where(tod.wg.LSR2.astype(int) | tod.wg.LSL2.astype(int))[0]
    return idx_wg_inside

# Correct wires' direction for each telescope
def correct_wg_angle(tod, telescope=None, restrict=True):
    """
    Correct offset of wires' direction by the mechanical design and hardware testing. This function is still under construction.

    Parameters
    ----------
        tod : AxisManager
        telescope : telescope type, e.g. satp1, satp3, etc.
            this parameter will basically be filled by the obs_info wrapped in the axismanager.
        restrict : bool (default, True)
            this parameter restricts the sample of the axismanger by the operation range of the wire grid.

    Returns
    -------
        (tod, idx_wg_inside) : Tuple
            tod is an AxisManager, and idx_wg_inside is the flags that indicates the opration range of the wire grid.
    """
    if telescope is None:
        try:
            telescope = tod.obs_info.telescope
        except:
            print("Cannot load the telescope info. Please assign your target telescope")
            logger.info("Telescope assignment was failed in site_pipeline.calibrate.wiregrid.correct_wg_angle")
    tod = _wrap_wg_hk(tod)
    idx_wg_inside = _get_operation_range(tod)
    if restrict: tod.restrict('samps', (idx_wg_inside[0], idx_wg_inside[-1]), in_place=True)
    #
    if telescope == 'satp1':
        wg_offset = wg_offset_satp1
    elif telescope == 'satp2':
        wg_offset = wg_offset_satp2
    elif telescope == 'satp3':
        wg_offset = wg_offset_satp3
    else:
        logger.warning(f"No matched telescope name of {telescope} for wire grid offset value, wg_offset")
    tod.wg.wrap_new('enc_rad', dtype='float32', shape=('dets', 'samps'))
    tod.wg.enc_rad = -  tod.wg.enc_rad_raw + wg_offset
    return (tod, idx_wg_inside)

# Detect the motion of the wire grid
def _detect_action(count, flag=0):
    """
    Detect the rotation of the grid.

    Parameters
    ----------
        count : int
            encoder actual counts
        flag : int
            flag to choose moving or static

    Returns
    -------
        count: int
            counts in the same status related with the flag
    """
    if flag == 0:
        vecbool = count[1:]-count[:-1] != 0  # moving
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
def _detect_steps(tod, stopped_time=10, thresholds=None):
    """
    Detect all steps of the stepwise operation

    Parameters
    ----------
        tod : AxisManager
        stopped_time : int
            the stopped time of your target operation
        threshold : Tuple
            the thresholds on the encoder counts.
            the first element is the upper bound for the static state,
            and the second element is the lower bound for the difference of the first

    Returns
    -------
        (ts_step_start, ts_step_stop) : time stamps of the start/stop for each step
        (angle_mean, angle_std) : average angles and its standard deviations for each step

    """
    if thresholds is None:
        thresholds = (10, 300) # upper bound of the static, lower bound of its diff
    cdiff0 = np.where(_detect_action(tod.wg.enc_rad/wg_counts2rad) < thresholds[0], 1, 0)
    cdiff1 = np.where(_detect_action(cdiff0, flag=1) > thresholds[1], 0, 1)
    cdiff2 = cdiff1[1:] - cdiff1[:-1]
    step_end = np.where(cdiff2 == 1)[0]
    step_start = np.where(cdiff2 == -1)[0]
    #
    angle_mean = []
    angle_std = []
    step_size = np.ceil(stopped_time/np.average(np.diff(tod.timestamps))).astype(int)
    step_end = np.append(step_end, step_end[-1]+step_size)
    for _i, _ in enumerate(step_start):
        angle_mean.append(np.average(tod.wg.enc_rad[step_start[_i]:step_end[_i+1]]))
        angle_std.append(np.std(tod.wg.enc_rad[step_start[_i]:step_end[_i+1]]))
    ts_step_start = tod.timestamps[step_start]
    ts_step_end = tod.timestamps[step_end]
    angle_mean = np.array(angle_mean)
    angle_std = np.array(angle_std)
    #
    return (ts_step_start, ts_step_end), (angle_mean, angle_std)

# Wrap the QU response during the operation of the Grid Loader
def wrap_qu_cal(tod, stopped_time, thresholds=None):
    """
    Wrap QU signal by the wire grid. This method is based on the demodulation by HWP.
    Users have to apply some HWP process before calling this.

    Parameters
    ----------
        tod : AxisManager
        stopped_time : int
            the stopped time of your target operation
        threshold : tuple
            the thresholds on the encoder counts.
            the first element is the upper bound for the static state,
            and the second element is the lower bound for the difference of the first

    Returns
    -------
        tod : AxisManger
            This includes the characterics of the wire grid operation and the Q/U signal
            related with it.
            wg.flag_step_start : the start time stamps for the steps
            wg.flag_step_stop : the stop time stamps for the steps
            wg.theta_wire_rad : wires' direction in radian for each step
            wg.theta_wire_std : the standard deviations of theta_wire for each step
            wg.Q, wg.U : Q (and U) signal by wires
            wg.Qerr, wg.Uerr : the standard deviations of Q (and U) signal

    """
    if hasattr(tod, 'demodQ') is False:
        print("This AxisManager does not have demodQ/demodU. Please call this method after you have demodulated the signal.")
        logger.info("Attribution check of demod signal was failed in site_pipeline.calibrate.wiregrid.wrap_QUcal")
        return False
    ts_step, wg_rad = _detect_steps(tod, stopped_time, thresholds)
    ts_instep = []
    step_Q = []
    step_U = []
    step_Qerr = []
    step_Uerr = []
    for _i, _ in enumerate(wg_rad[0]):
        instep = np.where(ts_step[0][_i] < tod.timestamps, True, False)
        instep = np.where(tod.timestamps < ts_step[1][_i+1], instep, False)
        #
        ts_instep.append(np.average(tod.timestamps[instep]))
        step_Q.append(np.average(tod.demodQ, axis=1, weights=instep))
        step_U.append(np.average(tod.demodU, axis=1, weights=instep))
        step_Qerr.append(np.std(tod.demodQ[:, instep == True], axis=1))
        step_Uerr.append(np.std(tod.demodU[:, instep == True], axis=1))
    ts_instep = np.array(ts_instep)
    step_Q = np.array(step_Q)
    step_U = np.array(step_U)
    step_Qerr = np.array(step_Qerr)
    step_Uerr = np.array(step_Uerr)
    #
    if 'cal_data' in dir(tod.wg): tod.wg.move('cal_data', None)
    _QUcal_ax = core.AxisManager(tod.dets)
    _QUcal_ax.wrap('flag_step_start', ts_step[0])
    _QUcal_ax.wrap('flag_step_stop', ts_step[1])
    _QUcal_ax.wrap('theta_wire_rad', wg_rad[0])
    _QUcal_ax.wrap('theta_wire_std', wg_rad[1])
    _QUcal_ax.wrap('ts_instep', ts_instep)
    _QUcal_ax.wrap('Q', step_Q, [(1, 'dets')])
    _QUcal_ax.wrap('U', step_U, [(1, 'dets')])
    _QUcal_ax.wrap('Qerr', step_Qerr, [(1, 'dets')])
    _QUcal_ax.wrap('Uerr', step_Uerr, [(1, 'dets')])
    tod.wg.wrap('cal_data', _QUcal_ax)
    return tod


### Circle fitting fucntions from here ###
def _get_initial_param_circle(x):
    """
    Get the initial parameters as inputs for the circle fitting in Q-U plane

    Parameters
    ----------
        x : assuming ndarray of Q and U signal with 16 steps in nominal, np.array([Q(16), U(16)])

    Returns
    -------
        params : the initial paramters for the circle fit
    """
    if len(np.shape(x)) == 2:
        A = np.average(x[0])
        B = np.average(x[1])
        C = np.average(x[0]**2+x[1]**2)
        params = np.array([A, B, C])
        return params
    else:
        print("This input vector is not valid shape.")
        return False

def _circle_resid(params, x):
    """
    Fit funciton

    Parameters
    ----------
        params : the input parameters
        x : data set

    Returns
    -------
        residual :  between the data point and the function under the input paramters
    """
    # (x-A)^2 + (y-B)^2 - C = 0
    if len(np.shape(x)) == 2:
        real = x[0]
        imag = x[1]
        A = params[0]
        B = params[1]
        C = params[2]
        return (real - A)**2 + (imag - B)**2 - C

def _comp_plane_fit(obs_data, std_data, fitfunc, param0):
    """
    Fit the data with a circle in a complex plane

    Parameters
    ----------
        obs_data : ndarray of Q and U signal with 16 steps in nominal, np.array([Q(16), U(16)])
        std_data : ndarray of the standard deviations of Q and U, np.array([Qerr(16), Uerr(16)])
        fitfunc : a function used for this fitting, basically assumed the function of _circle_resid
        param0 : the initial paramer set

    Returns
    -------
        fit result : determined by the implicit fitting of scipy.odr
    """
    if len(np.shape(obs_data)) == 2:
        mdr = odr.Model(fitfunc, implicit=True)
        mydata = odr.RealData(obs_data, y=1, sx=std_data)
        myodr = odr.ODR(mydata, mdr, beta0=param0)
        myoutput = myodr.run()
        return myoutput
    else:
        print("This input vector is not valid shape.")
        return False

def fit_with_circle(tod):
    """
    Get the results by the circle fitting about the responce against the wires in Q+iU plane

    Parameters
    ----------
        tod : AxisManager

    Returns
    -------
        tod : AxisManager
            This includes fit results for all the inpput detectors
            cx0 : Estimated x-offset value
            cy0 : Estimated y-offset value
            cr : Estimated radius vaule
            covariance : covariance matrix of the estimated parameters
            residual_var : Residual variance

    """
    _cal_data = tod.wg.cal_data
    _obs_data = []
    _obs_std = []
    _fit_results = []
    _cfitval = []
    _cfitcov = []
    _cfitresvar = []
    for _i in range(tod.dets.count):
        _obs_data.append(np.array([_cal_data.Q[:, _i], _cal_data.U[:, _i]]))
        _obs_std = np.array([_cal_data.Qerr[:, _i], _cal_data.Uerr[:, _i]])
        _params_init = _get_initial_param_circle(_obs_data[-1])
        _fit_results.append(
            _comp_plane_fit(_obs_data[-1], _obs_std, _circle_resid, _params_init))
        _cfitval.append(
            np.array([_fit_results[-1].beta[0], _fit_results[-1].beta[1], np.sqrt(_fit_results[-1].beta[2])]))
        _cfitcov.append(_fit_results[-1].cov_beta)
        _cfitresvar.append(_fit_results[-1].res_var)
    if 'cfit_result' in dir(tod.wg): tod.wg.move('cfit_result', None)
    _fit_ax = core.AxisManager(tod.dets)
    _fit_ax.wrap('cx0', np.array(_cfitval)[:,0], [(0, 'dets')])
    _fit_ax.wrap('cy0', np.array(_cfitval)[:,1], [(0, 'dets')])
    _fit_ax.wrap('cr', np.array(_cfitval)[:,2], [(0, 'dets')])
    _fit_ax.wrap('covariance', np.array(_cfitcov), [(0, 'dets')])
    _fit_ax.wrap('residual_var', np.array(_cfitresvar), [(0, 'dets')])
    tod.wg.wrap('cfit_result', _fit_ax)
    return tod
### Circle fitting fucntions to here ###

def get_cal_gamma(tod, wrap_aman=False, remove_cal_data=False):
    """
    Calibrate detectors' polarization response angle by wire grid.


    Parameters
    ----------
        tod : AxisManager
        wrap_aman : (default) False
        remove_cal_data : (defalut) False

    Returns
    -------
        (det_angle, det_angle_err) : polarization response angle of detectors in radian, which has the shape of (dets, wire's step)
        (bg_amp, bg_theta) : The amplitude and the direction of the background polarization not about the wires' signal.
        tod : AxisManager
            which has calibrated angles(tod.wg.gamma_cal). Only returned this by wrap_aman==True

    """
    _cal_data = tod.wg.cal_data
    _cfit_result = tod.wg.cfit_result
    _det_angle = []
    _det_angle_err = []
    for _i in range(16):
        Qcal = _cal_data.Q.T[:,_i] - _cfit_result.cx0
        Ucal = _cal_data.U.T[:,_i] - _cfit_result.cy0
        _atan_sig = np.arctan2(Ucal, Qcal)%(2*np.pi)
        #
        _det_angle.append(+ (0.5*(_atan_sig[:]) - _cal_data.theta_wire_rad[_i,np.newaxis]%(2*np.pi))) # need to be corrected
        _det_angle_err.append(np.sqrt(_cal_data.Uerr.T[:,_i]**2 + _cal_data.Qerr.T[:,_i]**2))
    _det_angle = np.array(_det_angle).T
    _det_angle_err = np.array(_det_angle_err).T
    _cal_amp = _cfit_result.cr
    _bg_theta = 0.5*np.arctan2(_cfit_result.cy0, _cfit_result.cx0)%np.pi - np.nanmean(_det_angle%np.pi, axis=1)
    _bg_amp = np.sqrt(_cfit_result.cx0**2 + _cfit_result.cy0**2)
    if remove_cal_data: tod.move('wg', None)
    if wrap_aman:
        if 'gamma_cal' in dir(tod): tod.move('gamma_cal', None)
        _gamma_ax = core.AxisManager(tod.dets)
        _gamma_ax.wrap('gamma_raw', _det_angle, [(0, 'dets')])
        _gamma_ax.wrap('gamma_raw_err', _det_angle_err, [(0, 'dets')])
        _gamma_ax.wrap('wires_relative_power', _cal_amp, [(0, 'dets')])
        _gamma_ax.wrap('gamma', np.nanmean(_det_angle%np.pi, axis=1), [(0, 'dets')])
        _gamma_ax.wrap('gamma_err', np.nanmean(_det_angle_err, axis=1)/np.sqrt(np.shape(_det_angle_err)[1]), [(0, 'dets')])
        _gamma_ax.wrap('background_pol_rad', _bg_theta, [(0, 'dets')])
        _gamma_ax.wrap('background_pol_relative_power', _bg_amp, [(0, 'dets')])
        _gamma_ax.wrap('theta_det_instr', 0.5*np.pi - np.nanmean(_det_angle%np.pi, axis=1), [(0, 'dets')]) # instumental angle of dets
        tod.wrap('gamma_cal', _gamma_ax)
        return tod
    else:
        return (_det_angle, _det_angle_err), (_bg_amp, _bg_theta)
