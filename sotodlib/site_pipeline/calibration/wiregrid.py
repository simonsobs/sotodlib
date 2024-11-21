# fundamental functions for the det_gamma calibration by the wire grid
# Please contact hnakata_JP(zhongtianjiaxin@gmail.com) if you have questions.

# import modules
# python standard
import os
import numpy as np
import scipy
from scipy import odr
from scipy.stats import mode
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse, Circle

# SO specific
from so3g.hk import load_range
from sotodlib import core
import sotodlib.io.load_smurf as ls
import sotodlib.hwp.hwp as hwp
from sotodlib.hwp.g3thwp import G3tHWP
# for house-keeping data
from sotodlib.io import hk_utils
from sotodlib.site_pipeline import util

WG_COUNTS2RAD = 2*np.pi/52000
WG_OFFSET_SATP1 = np.deg2rad(12.13) # SAT1, MF1
WG_OFFSET_SATP2 = np.deg2rad(9.473) # SAT2, UHF
WG_OFFSET_SATP3 = np.deg2rad(11.21) # TSAT, MF2

logger = util.init_logger('wiregrid', 'wiregrid: ')

def interpolate_hk(hk_data):
    """
    Simple function to get an interpolation function with a given house-keeping data loaded by so3g.hk.load_range.
    This is specified to the wire grid calibration not to extrapolate the real data.

    Parameters
    ----------
        hk_data : data
            house-keeping data as the output of so3g.hk.load_range

    Returns
    -------
        interp_func : function
            a function to interpolate the house-keeping data

    """
    interp_func = {}
    for key, val in hk_data.items():
        interp_func[key] = interp1d(hk_data[key][0], hk_data[key][1], bounds_error=False)
    return interp_func


# interp_func is the dictionary of functions to interpolate house-keeping data
def cosamnple_hk(tod, interp_func, is_merge=False):
    """
    Simple function to co-sampling house-keeping data along the timestamps in tod.
    This is specified to the wire grid calibration not to extrapolate the real data.

    Parameters
    ----------
        tod : AxisManager
        interp_func : function
            output of the interpolate_hk
        is_marge : bool (default, False)
            if merge the result into tod or not

    Returns
    -------
        hk_aman : AxisManager

    """
    hk_aman = core.AxisManager(tod.samps)
    for key in interp_func.keys():
        _cosamp_data = interp_func[key](tod.timestamps)
        hk_aman.wrap(key, _cosamp_data, [(0, 'samps')])
    if is_merge:
        tod.wrap('hk_data', hk_aman)
    return hk_aman

# Wrap house-keeping data
def wrap_wg_hk(tod, hk_dir, ts_margin=1, wg_encoder_fields=None, wg_actuator_fields=None):
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

                - enc_rad_raw : wires' direction read by encoder in radian (raw data from the encoder):
                - LSL1 : ON/OFF status of the limit switch LEFT 1 (outside) of the actuator
                - LSL2 : ON/OFF status of the limit switch LEFT 2 (inside) of the actuator
                - LSR1 : ON/OFF status of the limit switch RIGHT 1 (outside) of the actuator
                - LSR2 : ON/OFF status of the limit switch RIGHT 2 (inside) of the actuator

    """
    tod_start = float(tod.obs_info.start_time) - ts_margin
    tod_stop =  float(tod.obs_info.stop_time) + ts_margin
    _tel = tod.obs_info.telescope

    if hk_dir is None:
        print("Please specify the directory of the house-keeping data.")
        return False
    if wg_encoder_fields is None:
        wg_encoder_fields = {
            'enc_rad_raw': _tel + '.wg-encoder.feeds.wgencoder_full.reference_count',
        }
    if wg_actuator_fields is None:
        wg_actuator_fields = {
            'LSL1': _tel + '.wg-actuator.feeds.wgactuator.limitswitch_LSL1',
            'LSL2': _tel + '.wg-actuator.feeds.wgactuator.limitswitch_LSL2',
            'LSR1': _tel + '.wg-actuator.feeds.wgactuator.limitswitch_LSR1',
            'LSR2': _tel + '.wg-actuator.feeds.wgactuator.limitswitch_LSR2',
        }

    try:
        # for encoder data
        _data_enc = load_range(
            tod_start, tod_stop, data_dir=hk_dir,
            fields=list(wg_encoder_fields.values()),
            alias=list(wg_encoder_fields.keys())
            )
        if len(_data_enc[list(_data_enc.keys())[0]][0]) == 0:
            raise ValueError("There is no available encoder data with this tod.")
        _ifunc = interpolate_hk(_data_enc)
        wg_enc_aman = cosamnple_hk(tod, interp_func=_ifunc, is_merge=False)

        # for actuator data
        _data_act = load_range(
            tod_start, tod_stop, data_dir=hk_dir,
            fields=list(wg_actuator_fields.values()),
            alias=list(wg_actuator_fields.keys())
            )
        assert len(_data_act[list(_data_act.keys())[0]][0]) > 1
        _ifunc = interpolate_hk(_data_act)
        wg_act_aman = cosamnple_hk(tod, interp_func=_ifunc, is_merge=False)
    except AssertionError:
        logger.warning("The actuator data is not correctly stored in house-keeping. \
                        \n inside/outside status is not certificated so far.")
        wg_act_aman = None

    # wrap house-keeping data to the tod
    _ax_wg = core.AxisManager(tod.samps)
    _ax_instrument = core.AxisManager(tod.samps)
    _ax_wg.wrap('enc_rad_raw', wg_enc_aman['enc_rad_raw'] * WG_COUNTS2RAD, [(0, 'samps')])
    if wg_act_aman is not None:
        _ax_wg.wrap('LSL1', wg_act_aman['LSL1'], [(0,'samps')])
        _ax_wg.wrap('LSL2', wg_act_aman['LSL2'], [(0,'samps')])
        _ax_wg.wrap('LSR1', wg_act_aman['LSR1'], [(0,'samps')])
        _ax_wg.wrap('LSR2', wg_act_aman['LSR2'], [(0,'samps')])
    _ax_instrument.wrap('instrument', _ax_wg)
    tod.wrap('wg', _ax_instrument)

    # tod, which has the wire grid data with samps axis
    return tod


# Wrap house-keeping data with hkdb
def wrap_wg_hkdb(tod, hkdb_config, ts_margin=1):
    # wrap function will be replaced by this in the near future
    pass


# Correct wires' direction for each telescope
def correct_wg_angle(tod):
    """
    Correct offset of wires' direction by the mechanical design and hardware testing. Users can use this first, but developers need to implement offsets of other SATs as well.

    Parameters
    ----------
        tod : AxisManager
        telescope : telescope type, e.g. satp1, satp3, etc.
            this parameter will basically be filled by the obs_info wrapped in the axismanager.
        restrict : bool (default, True)
            this parameter restricts the sample of the axismanger by the operation range of the wire grid.

    """
    _tel = tod.obs_info.telescope
    if _tel == 'satp1':
        wg_offset = WG_OFFSET_SATP1
    elif _tel == 'satp2':
        wg_offset = WG_OFFSET_SATP2
    elif _tel == 'satp3':
        wg_offset = WG_OFFSET_SATP3
    else:
        logger.warning(f"No matched telescope name of {_tel} for wire grid offset value, wg_offset.\n \
                        the encoder offset is specified to zero. The hardware offset must remain.")
        wg_offset = 0
    enc_rad = (-tod.wg.instrument.enc_rad_raw + wg_offset)%(2*np.pi)
    tod.wg.instrument.wrap('enc_rad', enc_rad, [(0, 'samps')])
    return True


# Detect the motion of the wire grid
def _detect_motion(count, flag=0):
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
def _detect_steps(tod, stopped_time=None, steps_thresholds=None):
    """
    Detect all steps of the stepwise operation

    Parameters
    ----------
        tod : AxisManager
        stopped_time : int (default, 10 sec)
            the stopped time of your target calibration run
        steps_thresholds : Tuple
            the thresholds on the encoder counts.
            the first element is the upper bound for the static state,
            and the second element is the lower bound for the difference of the first

    Returns
    -------
        idx_steps_start, idx_steps_stop : sample index of the start/stop for each step

    """
    if stopped_time is None:
        stopped_time = 10

    if steps_thresholds is None:
        steps_thresholds = (10, 300) # upper bound of the static, lower bound of its diff
    cdiff0 = np.where(_detect_motion(tod.wg.instrument.enc_rad/WG_COUNTS2RAD) < steps_thresholds[0], 1, 0)
    cdiff1 = np.where(_detect_motion(cdiff0, flag=1) > steps_thresholds[1], 0, 1)
    cdiff2 = cdiff1[1:] - cdiff1[:-1]
    idx_step_stop = np.where(cdiff2 == 1)[0]
    idx_step_start = np.where(cdiff2 == -1)[0]

    step_size = np.average(idx_step_stop[1:] - idx_step_start[:-1]).astype(int)
    idx_step_stop = np.append(idx_step_stop, idx_step_start[-1]+step_size)

    return idx_step_start, idx_step_stop


def _get_operation_range(tod, stopped_time=None, ls_margin=None, is_restrict=True, remove_trembling=True, tremble_threshold=1.):
    """
    Define the range of the wire grid operation by the limit switches

    Parameters
    ----------
        tod : AxisManager
        stopped_time : int
            see also _detect_steps
        ls_margin : int (default, 2000 samples)
            the offsets to define the operation range of the wire grid calibration.
        is_restrict : bool (default, True)
            whether restrict TODs by the opration range or not
        remove_trembling : bool (default, True)
            whether remove the steps in the encoder data under the motor-motion threshold, tremble_threshold
        tremble_threshold : float (default, 1 deg)
            the threshold to remove the slight vibration in the steps of the wire grid rotation

    Returns
    -------
        idx_steps_start, idx_steps_stop : sample index of the start/stop for each step

    """

    if ('LSR2' in tod.wg.instrument) or ('LSL2' in tod.wg.instrument):
        if ls_margin is None:
            ls_margin = 2000
        _r2_nan = np.isnan(tod.wg.instrument.LSR2)
        _l2_nan = np.isnan(tod.wg.instrument.LSL2)
        _ls_nan = _r2_nan & _l2_nan
        ls_state = np.where(_ls_nan, 0, \
                      tod.wg.instrument.LSR2.astype(int) | tod.wg.instrument.LSL2.astype(int))
        _idx_temp = np.where(ls_state == 1)[0]
        idx_wg_oper = np.arange(
            _idx_temp[0] - ls_margin if 0 < _idx_temp[0] - ls_margin else 0,
            _idx_temp[-1] + ls_margin
            )

    else:
        logger.warning("Define the oparation range by the encoder data instead of limit switches. \
                        \n This module doesn't guarantee that the wire grid was inside.")
        # detect typical motion of the calibrator just in case
        try:
            _idx_step_start, _ = _detect_steps(tod, stopped_time=stopped_time, steps_thresholds=None)
        except IndexError:
            logger.error("The rotation couldn't be detected in this obs.")
        _ts_margin = 30
        _before_motion = np.where(tod.timestamps < tod.timestamps[_idx_step_start][0] - _ts_margin)[0]
        _after_motion  = np.where(tod.timestamps[_idx_step_start][-1] + _ts_margin < tod.timestamps)[0]
        idx_cal_start = max(_before_motion) if len(_before_motion) != 0 else 0
        idx_cal_stop = min(_after_motion) if len(_after_motion) != 0 else -1
        idx_wg_oper = np.arange(idx_cal_start, idx_cal_stop)

    if is_restrict:
        tod.restrict('samps', (idx_wg_oper[0], idx_wg_oper[-1]), in_place=True)

    # detect typical motion of the calibrator
    idx_steps_start, idx_steps_stop = _detect_steps(tod, stopped_time=stopped_time, steps_thresholds=None)

    # remove the rattling in the detected steps with the threshold of tremble_threshold
    if remove_trembling:
        _theta = tod.wg.instrument.enc_rad
        _idx_start = idx_steps_start
        _idx_stop = idx_steps_stop[1:]

        _fluc_start = np.where(np.abs(np.diff(_theta[_idx_start])) < np.deg2rad(tremble_threshold))[0]
        _fluc_stop = np.where(np.abs(np.diff(_theta[_idx_start])) < np.deg2rad(tremble_threshold))[0]
        assert _fluc_start.all() == _fluc_stop.all()

        idx_start_real = np.delete(_idx_start, _fluc_start+1)
        idx_stop_real = np.delete(_idx_stop, _fluc_stop)
        return idx_start_real, idx_stop_real

    return idx_steps_start, idx_steps_stop[1:]


def initialize_wire_grid(tod, stopped_time=None, ls_margin=None, is_restrict=True, remove_trembling=True):
    """
    Including the information of the wire grid operation. Steps are as follows:

        - step 0 : to wrap the house-keeping data(rotation encoder, limit switches),
        - step 1 : to correct the rotation angle based on the hardware design of the calibrator,
        - step 2 : to get opration range of a single calibration run,
        - step 3 : to wrap the stepwise rotation data into the tod (AxisManager) as tod.wg.cal_data.

    Parameters
    ----------
        tod : AxisManager,
            the others are same as the _get_operation_range

    Returns
    -------
        tod : AxisManger
            including the opration information of the target calibration

    """
    # wrap house-keepind data of the wire grid
    wrap_wg_hk(tod)

    # correct wires' angle for each telescope
    correct_wg_angle(tod)

    # get the opration range of the calibration
    idx_steps_start, idx_steps_stop  = _get_operation_range(tod, stopped_time=stopped_time, ls_margin=ls_margin, is_restrict=is_restrict, remove_trembling=True)

    if 'cal_data' in tod.wg._fields.keys():
        tod.wg.move('cal_data', None)
    _cal_data = core.AxisManager(tod.samps, tod.dets, core.IndexAxis('wg_steps'))
    _cal_data.wrap('idx_steps_start', idx_steps_start, [(0, 'wg_steps')])
    _cal_data.wrap('idx_steps_stop',  idx_steps_stop,  [(0, 'wg_steps')])
    tod.wg.wrap('cal_data', _cal_data)

    #return _tod, idx_steps, theta_wire
    return tod


# Wrap the QU response during the operation of the Grid Loader
def wrap_qu_cal(tod):
    """
    Wrap QU signal by the wire grid. This method is based on the demodulation by HWP.
    Users have to apply some HWP process before calling this.

    Parameters
    ----------
        tod : AxisManager

    Returns
    -------
        tod : AxisManger
            This includes the characterics of the wire grid operation and the Q/U signal
            related with it.

    Notes
    -----
        return includes fields as follows:

            - wg.cal_data.theta_wire_rad is the average direction of wires in each step
            - wg.cal_data.theta_wire_std is the standard deviation of the direction of wires in each step
            - wg.cal_data.ts_step_mid is the time stamps in the middle of each step
            - wg.cal_data.Q, wg.cal_data.U : Q (and U) signal by wires
            - wg.cal_data.Qerr, wg.cal_data.Uerr : the standard deviations of Q (and U) signal

    """
    if 'demodQ' in tod._fields.keys():
        _theta_wire = tod.wg.instrument.enc_rad
        _idx_start = tod.wg.cal_data.idx_steps_start
        _idx_stop = tod.wg.cal_data.idx_steps_stop

        ts_step_mid = []
        theta_wire_av = []
        theta_wire_std = []
        step_Q = []
        step_U = []
        step_Qerr = []
        step_Uerr = []
        for _i, _ in enumerate(_idx_start):
            _idx_frame = np.arange(tod.samps.count)
            instep = np.where(_idx_start[_i] < _idx_frame, True, False)
            instep = np.where(_idx_frame < _idx_stop[_i], instep, False)
            ts_step_mid.append(np.average(tod.timestamps[instep]))
            theta_wire_av.append(np.average(_theta_wire[instep]))
            theta_wire_std.append(np.std(_theta_wire[instep]))
            step_Q.append(np.average(tod.demodQ, axis=1, weights=instep))
            step_U.append(np.average(tod.demodU, axis=1, weights=instep))
            step_Qerr.append(np.std(tod.demodQ[:, instep == True], axis=1))
            step_Uerr.append(np.std(tod.demodU[:, instep == True], axis=1))
        ts_step_mid = np.array(ts_step_mid)
        theta_wire_av = np.array(theta_wire_av)
        theta_wire_std = np.array(theta_wire_std)
        step_Q = np.array(step_Q)
        step_U = np.array(step_U)
        step_Qerr = np.array(step_Qerr)
        step_Uerr = np.array(step_Uerr)
        #
        #_ax_qu_cal = core.AxisManager(tod.dets, tod.wg.wg_steps)
        tod.wg.cal_data.wrap('theta_wire_rad', theta_wire_av , [(0, 'wg_steps')])
        tod.wg.cal_data.wrap('theta_wire_std', theta_wire_std, [(0, 'wg_steps')])
        tod.wg.cal_data.wrap('ts_step_mid',    ts_step_mid,    [(0, 'wg_steps')])
        tod.wg.cal_data.wrap('Q',              step_Q,         [(0, 'wg_steps'), (1, 'dets')])
        tod.wg.cal_data.wrap('U',              step_U,         [(0, 'wg_steps'), (1, 'dets')])
        tod.wg.cal_data.wrap('Qerr',           step_Qerr,      [(0, 'wg_steps'), (1, 'dets')])
        tod.wg.cal_data.wrap('Uerr',           step_Uerr,      [(0, 'wg_steps'), (1, 'dets')])
        return tod
    else:
        print("This AxisManager does not have demodQ/demodU. Please call hwp.demod_tod before using this method.")
        logger.error("Attribution check of demod signal was failed in site_pipeline.calibrate.wiregrid.wrap_QUcal")
        return False

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
        A = np.nanmean(x[0])
        B = np.nanmean(x[1])
        C = np.nanmean(x[0]**2+x[1]**2)
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
        fit_results : list
            the result of the circle fitting of the wires' signal in the Q/U plane.

    Notes
    -----
        With this process, tod will include the several parameters of the fittings as tod.wg.cfit_result:

            - cx0(_err) : the estimated x-offset value(, and its fit err).
            - cy0(_err) : the estimated y-offset value(, and its fit err).
            - cr(_err) : Estimated radius vaule(, and its fit err).
            - covariance : covariance matrix of the estimated parameters.
            - residual_var : Residual variance.
            - is_normally_stopped : the status of how fits end.

    """
    _cal_data = tod.wg.cal_data
    fit_results = []
    _cfitval = []
    _cfiterr = []
    _cfitcov = []
    _cfitresvar = []
    _is_normally_stopped = []
    for _i in range(tod.dets.count):
        _obs_data = np.array([_cal_data.Q[:, _i], _cal_data.U[:, _i]])
        _obs_std = np.array([_cal_data.Qerr[:, _i], _cal_data.Uerr[:, _i]])
        _params_init = _get_initial_param_circle(_obs_data)
        fit_results.append(
            _comp_plane_fit(_obs_data, _obs_std, _circle_resid, _params_init))
        _cfitval.append(
            np.array([
                fit_results[-1].beta[0],
                fit_results[-1].beta[1],
                np.sqrt(fit_results[-1].beta[2])]))
        _cfiterr.append(
            np.array([
                fit_results[-1].sd_beta[0],
                fit_results[-1].sd_beta[1],
                fit_results[-1].sd_beta[2]]))
        _cfitcov.append(fit_results[-1].cov_beta)
        _cfitresvar.append(fit_results[-1].res_var)
        _is_normally_stopped.append(True if fit_results[-1].stopreason[0] == 'Parameter convergence' else False)
    if 'cfit_result' in tod.wg._fields.keys():
        tod.wg.move('cfit_result', None)
    _ax_cfit = core.AxisManager(tod.dets)
    _ax_cfit.wrap('cx0',                 np.array(_cfitval)[:,0],        [(0, 'dets')])
    _ax_cfit.wrap('cy0',                 np.array(_cfitval)[:,1],        [(0, 'dets')])
    _ax_cfit.wrap('cr',                  np.array(_cfitval)[:,2],        [(0, 'dets')])
    _ax_cfit.wrap('cx0_err',             np.array(_cfiterr)[:,0],        [(0, 'dets')])
    _ax_cfit.wrap('cy0_err',             np.array(_cfiterr)[:,1],        [(0, 'dets')])
    _ax_cfit.wrap('cr_err',              np.array(_cfiterr)[:,2],        [(0, 'dets')])
    _ax_cfit.wrap('covariance',          np.array(_cfitcov),             [(0, 'dets')])
    _ax_cfit.wrap('residual_var',        np.array(_cfitresvar),          [(0, 'dets')])
    _ax_cfit.wrap('is_normally_stopped', np.array(_is_normally_stopped), [(0, 'dets')])
    tod.wg.wrap('cfit_result', _ax_cfit)
    return fit_results
### Circle fitting fucntions to here ###


def _ignore_outlier_angle(estimated_angle, num_bins=None, gap_size=None):
    """
    A function to reduce the effects by the outlier to the averaging of the calibration

    Parameters
    ----------
        estimated_angle : ndarray
            raw fit results of the polatization angle estimation
        num_bins : int
            number of bins to get the majority of the estimated angle
        gap_size : float
            a shift of the outlier binning

    Returns
    -------
        valid_angle : ndarray
            a set of the validate angle limited to the majority
    """
    if num_bins is None:
        num_bins = 18
    if gap_size is None:
        gap_size = np.deg2rad(5.)
    _low_bins = np.digitize(estimated_angle%np.pi, bins=np.linspace(-gap_size,np.pi-gap_size,num_bins))
    _high_bins = np.digitize(estimated_angle%np.pi, bins=np.linspace(0,np.pi+gap_size,num_bins))
    _is_likely = np.where(
        mode(_low_bins, axis=1)[1] < mode(_high_bins, axis=1)[1], True, False
        )
    _digitized = np.array([_high_bins[_i] if _j else _low_bins[_i] for _i, _j in enumerate(_is_likely)])
    _valid_steps = _digitized.T == mode(_digitized, axis=1)[0]
    valid_angle = np.where(_valid_steps.T==True, estimated_angle, np.nan)
    return valid_angle


def get_cal_gamma(tod, wrap_aman=True, remove_cal_data=False, num_bins=None, gap_size=None):
    """
    Calibrate detectors' polarization response angle by wire grid.

    Parameters
    ----------
        tod : AxisManager
        wrap_aman : bool (default, True)
        remove_cal_data : bool (defalut) False
        num_bins : int
            see _ignore_outlier_angle
        gap_size : float
            see _ignore_outlier_angle

    Returns
    -------
        tod (or _ax_gamma) : AxisManager
            which includes the calibrated angle of gamma in the sky coordinate, etc.

    Notes
    -----
        gamma_raw(_err) is the raw output of the calibration for each wire grid step.
        wires_relative_power is the signal intensity of the wires, which will relatively change depending on the ambient temperature.
        gamma(_err) is the main result of the calibration using Sparse Wire Grid.
        background_pol_rad or background_pol_relative_power is the Q/U-plane offsets of the detectors signal respect to the wires' reflection.
        theta_det_instr is gamma translated to the instrumental angle printed on the silicon wafer.

    """
    _cal_data = tod.wg.cal_data
    _cfit_result = tod.wg.cfit_result
    _det_angle = []
    _det_angle_err = []
    for _i in range(tod.wg.wg_steps.count):
        # remove the offset of the circle
        Qcal = _cal_data.Q.T[:,_i] - _cfit_result.cx0
        Ucal = _cal_data.U.T[:,_i] - _cfit_result.cy0
        _atan_sig = np.arctan2(Ucal, Qcal)%(2*np.pi)

        # 0.5*(2*theta_det + 2*theta_wire) - theta_wire
        _det_angle.append(+ (0.5*(_atan_sig[:]) - _cal_data.theta_wire_rad[_i,np.newaxis]%(2*np.pi))) # need to be corrected
        _det_angle_err.append(np.sqrt(_cal_data.Uerr.T[:,_i]**2 + _cal_data.Qerr.T[:,_i]**2)) # need to be corrected
    _det_angle = np.array(_det_angle).T
    _det_angle_err = np.array(_det_angle_err).T

    # pick up common modes for each estimation
    _valid_angle = _ignore_outlier_angle(_det_angle, num_bins=num_bins, gap_size=gap_size)

    # calibrated gamma
    gamma = np.nanmean(np.unwrap(_valid_angle, period=np.pi), axis=1)%np.pi
    gamma_err = np.nanmean(_det_angle_err, axis=1)/np.sqrt(np.shape(_det_angle_err)[1])

    # back ground polarization
    _bg_theta = 0.5*np.arctan2(_cfit_result.cy0, _cfit_result.cx0)%np.pi - gamma
    _bg_amp = np.sqrt(_cfit_result.cx0**2 + _cfit_result.cy0**2)

    _ax_gamma = core.AxisManager(tod.dets, tod.wg.wg_steps)
    _ax_gamma.wrap('gamma_raw',                     _det_angle%np.pi,  [(0, 'dets'), (1, 'wg_steps')])
    _ax_gamma.wrap('gamma_raw_err',                 _det_angle_err,    [(0, 'dets'), (1, 'wg_steps')])
    _ax_gamma.wrap('wires_relative_power',          _cfit_result.cr,   [(0, 'dets')])
    _ax_gamma.wrap('gamma',                         gamma,             [(0, 'dets')])
    _ax_gamma.wrap('gamma_err',                     gamma_err,         [(0, 'dets')])
    _ax_gamma.wrap('background_pol_rad',            _bg_theta,         [(0, 'dets')])
    _ax_gamma.wrap('background_pol_relative_power', _bg_amp,           [(0, 'dets')])
    _ax_gamma.wrap('theta_det_instr',               0.5*np.pi - gamma, [(0, 'dets')]) # instumental angle of dets
    if remove_cal_data:
        tod.move('wg', None)
    if wrap_aman:
        if 'gamma_cal' in tod._fields.keys():
            tod.move('gamma_cal', None)
        tod.wrap('gamma_cal', _ax_gamma)
        return tod
    else:
        return _ax_gamma


## function for time constant measurement
def divide_tod(tod, forward_margin=None, backward_margin=None):
    """
    a function to divide the single time constant calibration run into the two range, forward rotating HWP and reversely rotating HWP

    Parameters
    ----------
        tod : AxisManager
            a fully-scale tod of the time constant calbiration run
        forward/backward_margin : int
            how much steps back from the flip of the hwp rotation

    Returns
    -------
        tod1 : AxisManager
            tod, which corresponds to 1st hwp-direction range
        tod2 : AxisManager
            tod, which corresponds to 2nd hwp-direction range

    """
    if forward_margin is None:
        forward_margin = 60000
    if backward_margin is None:
        backward_margin = 2000
    _flip_idx1 = np.where(tod.hwp_solution.quad_1[1:] - tod.hwp_solution.quad_1[:-1])[0]
    _flip_idx2 = np.where(tod.hwp_solution.quad_2[1:] - tod.hwp_solution.quad_2[:-1])[0]
    try:
        assert _flip_idx1 == _flip_idx2
        flip_at = _flip_idx1
    except AssertionError:
        flip_at = int(0.5*(_flip_idx1 + _flip_idx2))
        logger.warning("The flip indexes of the quadature(direction of rotation) data are different. The average is assigned.")
    tod1 = tod.restrict('samps', (tod.samps.offset, flip_at - forward_margin), in_place=False)
    tod2 = tod.restrict('samps', (tod.samps.offset + flip_at + backward_margin, -1), in_place=False)
    return tod1, tod2


def binning_data(ref_data, target_data, num_bin=100):
    """
    This function returns the binned data and its error along with the binning of the reference data.

    Parameters
    ----------
        ref_data: ndarray
            the reference data to bin the target data
        target_data: ndarray
            the target data to be binned
        num_bin: int
            the number of bins to divide the reference data

    Returns
    -------
        binned: ndarray
            the binned data
        binned_err: ndarray
            the error of the binned data
    """
    _hist = np.histogram(ref_data, bins=num_bin)
    _bins = np.digitize(ref_data, _hist[1][1:], right=True)
    if len(np.shape(target_data)) == 1:
        _binned = []
        _binned_err = []
        for _j in range(num_bin):
            _sig_abin = target_data[_bins == _j]
            _binned.append(np.mean(_sig_abin))
            _binned_err.append(np.std(_sig_abin))
        _binned = np.array(_binned)
        _binned_err = np.array(_binned_err)
    elif len(np.shape(target_data)) == 2:
        _binned = []
        _binned_err = []
        for _j in range(num_bin):
            _sig_abin = target_data[:,_bins == _j]
            _binned.append(np.mean(_sig_abin, axis=1))
            _binned_err.append(np.std(_sig_abin, axis=1))
        _binned = np.array(_binned).T
        _binned_err = np.array(_binned_err).T
    return _binned, _binned_err


def _linear_model(params, xval, yval, yerr):
    a, b = params[0], params[1]
    # mathematicall, we want to keep this formula as 2 * tau * 2*np.pi * hwp_speed_hz
    model = 2*a*2*np.pi*xval + b
    chi = (yval - model) / yerr
    return chi


def _fit_time_const(ref_hwp_speed, normalized_angle, angle_err):
    """
    A function to fit how the estimated angle changes related to the hwp speed

    Parameters
    ----------
        ref_hwp_speed : ndarray
            the speed of HWP as a reference
        normalized_angle : ndarray
            the corresponding angle to the reference hwp speed
        angle_err : ndarray
            the error of the angle estimation

    Returns
    -------
        fres : fit results
            gradient of the slope and the y-section
        ferr : fit error
        fchi2 : chi square of the fitting

    """
    _x = ref_hwp_speed
    _y = normalized_angle
    _yerr = angle_err
    fres = []
    ferr = []
    fchi2 = []
    for _i in range(np.shape(_y)[0]):
        iparams = np.array([1e-3, -2e-2])
        bounds = ([0, -2*np.pi], [1e-1, 2*np.pi])
        _res = least_squares(_linear_model, x0=iparams, bounds=bounds, \
                                args=(_x, _y[_i], _yerr[_i]))
        # calculate fit error
        _J = _res.jac
        _cov = np.linalg.inv(_J.T.dot(_J))
        _err = np.sqrt(np.diag(_cov))

        fres.append([_res.x[0], _res.x[1]])
        ferr.append([_err[0], _err[1]])
        fchi2.append(np.mean(_linear_model(_res.x, _x, _y[_i], _yerr[_i])**2))
    return np.array(fres), np.array(ferr), np.array(fchi2)


def get_time_const(tod, hwp_direction="forward", slice_bin=(20,-20), angle_offsets=None, is_wrap=True):
    """
    function to get the values of time constants measured by HWP/wire grid.

    Parameters
    ----------
        tod1 : AxisManager
            tod for the 1st range in which the speed of HWP is decreasing to 0 Hz
        tod2 : AxisManager
            tod for the 2nd range in which the speed of HWP is increasing to 2 Hz
        hwp_sign : int (default, -1)
            the sign of hwp rotation. the default value is determined by the SATP1 configuration
        slice0 :
            the slice to cut tod1 before the fitting
        slice1 :
            the slice to cut tod2 before the fitting
        is_wrap : bool (defult True)
            whether this function wraps the result into both tods or not

    Returns
    -------
        if ``is_wrap`` is False, then will return fit results for each. Otherwise, return is the AxisManager.

    """
    # be careful about the convention of the hwp direction, hwp_sign below is defined by the satp1 configuration
    if hwp_direction == "forward":
        hwp_sign = 1
    elif hwp_direction == "reverse":
        hwp_sign = -1

    _hwp_spped = hwp_sign * np.gradient(np.unwrap(tod.hwp_angle, period=2*np.pi)) / np.gradient(tod.timestamps) / (2*np.pi)

    _raw_angle = 0.5*np.arctan2(tod.demodU, tod.demodQ)
    bin_hwp_speed, bin_hwp_speed_err = binning_data(tod.timestamps, _hwp_spped)
    bin_angle, bin_angle_err = binning_data(tod.timestamps, _raw_angle)

    sl0, sl1 = slice_bin
    if angle_offsets is None:
        angle_offsets = bin_angle[:,0]
    _ref = bin_hwp_speed[sl0:sl1]
    _data = hwp_sign * bin_angle[:,sl0:sl1] - hwp_sign * angle_offsets[:,np.newaxis]
    _err = bin_angle_err[:,sl0:sl1]

    fres, ferr, fchi2 = _fit_time_const(_ref, _data, _err)

    if is_wrap:
        _ax_tc = core.AxisManager(tod.dets)
        _ax_tc.wrap("tau", fres[:,0],[(0,'dets')])
        tod.wg.wrap("time_consts", _ax_tc)
        return True, angle_offsets
    else:
        return (fres, ferr, fchi2), angle_offsets
