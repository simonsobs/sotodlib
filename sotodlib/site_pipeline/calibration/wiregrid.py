# fundamental functions for the det_gamma calibration by the wire grid
# Please contact hnakata_JP(zhongtianjiaxin@gmail.com) if you have questions.

# import python standard libraries
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# SO specific
from so3g.hk import load_range
from sotodlib import core
from sotodlib.io import hkdb
from sotodlib.io import hk_utils
from sotodlib.site_pipeline.utils.logging import init_logger
logger = init_logger('wiregrid', 'wiregrid: ')


@dataclass
class wg_config:
    hk_root: str
    db_file: str
    site: bool
    aliases: dict
    wg_count: int
    wg_offset: float
    telescope: str
    timestamp_margin: float = 1.0


def load_data(config: wg_config, start_time: float, stop_time: float) -> dict:
    """
    Load wire grid house-keeping data based on the provided configuration in site-pipeline-configs.
    start_time and stop_time are used to load the data within the specified range.

    Parameters
    ----------
        config : wg_config
            configuration for loading the wire grid house-keeping data
        start_time : float
            start time in unixtime
        stop_time : float
            stop time in unixtime

    Returns
    -------
        wg_data_dict : dict
            dictionary including the raw house-keeping data and corrected encoder angle about the wire grid operation.

    Notes
    -----
        We assume wg_cfg is a yaml file, which has the following structure.

        - hk_root. path to house-keeping data directory, e.g. ``'../data/satp1/hk/'``
        - db_file.  path to sqlite database file, e.g. ``'./hkdb-satp1.db'``
        - site: bool. If True, load housekeeping data from L2 hk data at site, if False load it from L3 hkdb. 
        - aliases:
            - enc_count : ``'wg-encoder.wgencoder_full.reference_count'``
            - LSL1 : ``'wg-actuator.wgactuator.limitswitch_LSL1'``
            - LSL2 : ``'wg-actuator.wgactuator.limitswitch_LSL2'``
            - LSR1 : ``'wg-actuator.wgactuator.limitswitch_LSR1'``
            - LSR2 : ``'wg-actuator.wgactuator.limitswitch_LSR2'``
            - angleX : ``'wg-tilt-sensor.wgtiltsensor.angleX'``
            - angleY : ``'wg-tilt-sensor.wgtiltsensor.angleY'``
            - tempX : ``'wg-tilt-sensor.wgtiltsensor.temperatureX'``
            - tempY : ``'wg-tilt-sensor.wgtiltsensor.temperatureY'``
            - temp_rotator: ``'wg-labjack.sensors_downsampled.AIN0C'``
            - temp_rotation_motor: ``'wg-labjack.sensors_downsampled.AIN1C'``
            - temp_elec_plate: ``'wg-labjack.sensors_downsampled.AIN2C'``
        - wg_count: magnetic scale counts per one lap (52,000 for SATp1~SATp3)
        - wg_offset: correction for the encoder offset, in degrees 
            (12.13 deg for SATp1, 9.473 deg for SATp2, 11.21 deg for SATp3)
        - telescope: telescope name in lowercase letters, e.g. 'satp1'
        - timestamp_margin: Time buffer added before and after the time range of observation, in seconds. Defaults to 1. This margin compensates for imperfect timestamp synchronization between detectors' data and housekeeping data.

    """
    if config.site:
        logger.info("Loading the wire grid house-keeping data with hk_dir.")
        wg_data_dict = _load_l2_data(config, start_time, stop_time)
    else:
        logger.info("Loading the wire grid house-keeping data with hkdb.")
        wg_data_dict = _load_l3_data(config, start_time, stop_time)

    wg_data_dict['enc_rad'] = (
    wg_data_dict['enc_count'][0], _correct_wg_angle(wg_data_dict['enc_count'][1], config)
        )

    return wg_data_dict

def _load_l2_data(config: wg_config, start_time: float, stop_time: float) -> dict:
    """
    Load wire grid house-keeping data with so3g.hk.load_range method.

    Parameters
    ----------
        config : wg_config
            configuration for loading the wire grid house-keeping data
        start_time : float
            start time in unixtime
        stop_time : float
            stop time in unixtime

    Returns
    -------
        raw_data_dict : dict
            dictionary including the raw house-keeping data about the wire grid operation.
    """

    aliases = config.aliases.keys()
    fields = []
    for field in config.aliases.values():
        first_field, second_field = field.split('.', 1)
        fields.append(config.telescope + '.' + first_field + '.feeds.' + second_field)
    
    _start = float(start_time - config.timestamp_margin)  # margin of 1 second
    _stop = float(stop_time + config.timestamp_margin)
    
    _data = load_range(
        _start, _stop, data_dir=config.hk_root,
        fields=fields,
        alias=aliases,
    )

    if len(_data) == 0:
        logger.error("No available wire grid data with this tod.")
        raise ValueError("No available wire grid data with this tod.")

    if 'enc_count' not in _data:
        logger.error("No available ENCODER data with this tod.")
        raise ValueError("No available ENCODER data with this tod.")

    if 'angleX' not in _data or \
        'angleY' not in _data or \
        'tempX' not in _data or \
        'tempY' not in _data:
        logger.warning("Lack of tilt sensor data.")

    if 'LSL1' not in _data or \
        'LSL2' not in _data or \
        'LSR1' not in _data or \
        'LSR2' not in _data:
        logger.warning("The actuator data is not correctly stored in house-keeping. \
                        \n inside/outside status is not certificated so far.")
    raw_data_dict = _data
    
    return raw_data_dict

def _load_l3_data(config: wg_config, start_time: float, stop_time: float) -> dict:
    '''
    Load wire grid house-keeping data with sotodlib.io.load_hk method.

    Parameters
    ----------
        config : wg_config
            configuration for loading the wire grid house-keeping data
        start_time : float
            start time in unixtime
        stop_time : float
            stop time in unixtime

    Returns
    -------
        raw_data_dict : dict
            dictionary including the raw house-keeping data about the wire grid operation.
    '''
    hkdb_config = {k: getattr(config, k) for k in ['hk_root', 'db_file','aliases']}

    cfg = hkdb.HkConfig.from_dict(hkdb_config)
    lspec = hkdb.LoadSpec(
        cfg=cfg, start=start_time, end=stop_time,
        fields=list(cfg.aliases.keys())
    )
    raw_data = hkdb.load_hk(lspec)
    _fields = list(raw_data.data.keys())
    _swapped_aliases = {v: k for k, v in config.aliases.items()}
    _aliases = [_swapped_aliases[f] for f in _fields]
    raw_data_dict = {
        alias: raw_data.data[field] for alias, field in zip(_aliases, _fields)
    }
    
    return raw_data_dict

# Wrap house-keeping data
def wrap_wg_hk(tod, raw_data_dict, merge=True):
    """
    Wrap the house-keeping data about the wire grid operation.

    Parameters
    ----------
        tod : AxisManager
        raw_data_dict : dict
            dictionary including the raw house-keeping data about the wire grid operation.
        merge : bool (default, True)
            whether merge the house-keeping data into tod or not

    Returns
    -------
        tod : AxisManager
            This includes fields, which are related with the wire grid hardware.

                - enc_count : wires' direction read in raw encoder count (raw data from the encoder)
                - enc_rad : wires' direction read by encoder in radian (corrected with encoder count and hardware offset)
                - LSL1 : ON/OFF status of the limit switch LEFT 1 (outside) of the actuator
                - LSL2 : ON/OFF status of the limit switch LEFT 2 (inside) of the actuator
                - LSR1 : ON/OFF status of the limit switch RIGHT 1 (outside) of the actuator
                - LSR2 : ON/OFF status of the limit switch RIGHT 2 (inside) of the actuator
                - angleX : tilt angle X of the tilt sensor in degree
                - angleY : tilt angle Y of the tilt sensor in degree
                - tempX : temperature sensor X of the tilt sensor in Celsius
                - tempY : temperature sensor Y of the tilt sensor in Celsius
                - temp_rotator: temperature sensor of the rotator in Celsius
                - temp_rotation_motor: temperature sensor of the rotation motor in Celsius
                - temp_elec_plate: temperature sensor of the electric plate in Celsius

    """
    if not hasattr(tod, 'timestamps'):
        logger.warning("TOD does not have timestamps, and it should be only meta data. \
                        \n Calculating TOD samplings under 5 ms samples. \
                        \n The sampling of this wrapped data is not correct.")
        _start = tod.obs_info.start_time
        _stop = tod.obs_info.stop_time
        timestamps = np.linspace(_start, _stop, tod.samps.count)
    else:
        timestamps = tod.timestamps

    interp_func = {}
    for _k, _v in raw_data_dict.items():
        interp_func[_k] = interp1d(_v[0], _v[1], bounds_error=False)

    hk_aman = core.AxisManager(core.OffsetAxis('samps', count=len(timestamps)))
    for _f in interp_func.keys():
        _cosamp_data = interp_func[_f](timestamps)
        hk_aman.wrap(_f, _cosamp_data, [(0, 'samps')])
    if merge:
        _ax_instrument = core.AxisManager(tod.samps)
        _ax_instrument.wrap('instrument', hk_aman)
        tod.wrap('wg', _ax_instrument)
        return tod
    return hk_aman

def _correct_wg_angle(enc_count, config: wg_config):
    '''
    Correct offset of wires' direction with value confirmed in the hardware testing.

    Parameters
    ----------
        enc_count : np.ndarray
            encoder actual counts
        config : wg_config
    Returns
    -------
        _in_radians_w_offset : np.ndarray
            wires' direction in radian (corrected with both encoder count and hardware offset) 
    '''
    if not isinstance(config, wg_config):
        raise TypeError("config should be an instance of wg_config dataclass.")
    else:
        _in_radians = enc_count * 2 * np.pi / config.wg_count
        _in_radians_w_offset = (- _in_radians + np.deg2rad(config.wg_offset))%(2*np.pi) 

    return _in_radians_w_offset

# Detect the motion of the wire grid
def _detect_motion(count, flag=0):
    """
    Detect the rotation of the grid.

    Parameters
    ----------
        count : int
            encoder actual counts
        flag : int (default, 0)
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
def _detect_steps(tod, steps_thresholds=(10, 300)):
    """
    Detect all steps of the stepwise operation

    Parameters
    ----------
        tod : AxisManager
        steps_thresholds : Tuple (default, (10, 300))
            the thresholds on the encoder counts.
            the first element is the upper bound for the static state,
            and the second element is the lower bound for the difference of the first

    Returns
    -------
        idx_steps_start, idx_steps_stop : sample index of the start/stop for each step

    """
    cdiff0 = np.where(_detect_motion(tod.wg.instrument.enc_count) < steps_thresholds[0], 1, 0)
    cdiff1 = np.where(_detect_motion(cdiff0, flag=1) > steps_thresholds[1], 0, 1)
    cdiff2 = cdiff1[1:] - cdiff1[:-1]
    idx_step_stop = np.where(cdiff2 == 1)[0]
    idx_step_start = np.where(cdiff2 == -1)[0]

    if len(idx_step_start) == len(idx_step_stop):
        logger.info(
            "The count of start index is equal to that of stop index."
        )
        step_size = np.average(idx_step_stop[1:] - idx_step_start[:-1]).astype(int)
        idx_step_stop = np.append(idx_step_stop, idx_step_start[-1]+step_size)

    elif len(idx_step_start) == len(idx_step_stop) + 1:
        logger.warning(
            "The count of start index is one greater than that of stop index. \
            \n Trying to fix the index by the average step size."
            )
        if idx_step_start[0] < idx_step_stop[0]:
            idx_step_start = idx_step_start[1:]
        # the other case has to be implemented in the future
        step_size = np.average(idx_step_stop[1:] - idx_step_start[:-1]).astype(int)
        idx_step_stop = np.append(idx_step_stop, idx_step_start[-1]+step_size)

    elif len(idx_step_start) + 1 == len(idx_step_stop):
        if idx_step_start[0] < idx_step_stop[0]:
            logger.warning(
                "The count of stop index is one less than that of start index. \
                \n Pass the original index to the next step."
            )
        else:
            logger.warning("_detect_steps: Future subject. Not corrected yet.")

    return idx_step_start, idx_step_stop

# Find the nominal operation range of the wire grid
def find_operation_range(tod, steps_thresholds=(10, 300), ls_margin=2000, is_restrict=True, \
                        remove_trembling=True, tremble_threshold=1.):
    """
    Define the range of the wire grid operation by the limit switches

    Parameters
    ----------
        tod : AxisManager
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
        _r2_nan = np.isnan(tod.wg.instrument.LSR2)
        _l2_nan = np.isnan(tod.wg.instrument.LSL2)
        _ls_nan = _r2_nan & _l2_nan
        ls_state = np.where(
            _ls_nan, 0,
            tod.wg.instrument.LSR2.astype(int) | tod.wg.instrument.LSL2.astype(int)
        )
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
            _idx_step_start, _ = _detect_steps(tod, steps_thresholds=steps_thresholds)
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
    idx_steps_start, idx_steps_stop = _detect_steps(tod, steps_thresholds=steps_thresholds)

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

# Wrap the data specified from the operation range of the wire grid
def calc_calibration_data_set(tod, idx_steps_start, idx_steps_stop):
    """
    Interpret the Q_cal, U_cal data from the wrapped calibration data.
    This method is based on both of demodQ and demodU, which are calculated by the HWP demodulation process.
    Users have to apply some HWP process before calling this.

    Parameters
    ----------
        tod : AxisManager,
            the others are same as the _get_operation_range
        idx_steps_start : np.ndarray
            sample index of the start of each step
        idx_steps_stop : np.ndarray
            sample index of the stop of each step

    Returns
    -------
        axes : AxisManger
        return includes fields as follows:

            - wg.cal_data.theta_wire_rad is the average direction of wires in each step
            - wg.cal_data.theta_wire_std is the standard deviation of the direction of wires in each step
            - wg.cal_data.ts_step_mid is the time stamps in the middle of each step
            - wg.cal_data.Q, wg.cal_data.U : Q (and U) signal by wires
            - wg.cal_data.Qerr, wg.cal_data.Uerr : the standard deviations of Q (and U) signal
    """
    _theta_wire = tod.wg.instrument.enc_rad

    ts_step_mid = []
    theta_wire_av = []
    theta_wire_std = []
    step_Q = []
    step_U = []
    step_Qerr = []
    step_Uerr = []
    for _i, _ in enumerate(idx_steps_start):
        _idx_frame = np.arange(tod.samps.count)
        instep = np.where(idx_steps_start[_i] < _idx_frame, True, False)
        instep = np.where(_idx_frame < idx_steps_stop[_i], instep, False)
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

    # wrap the data for each step
    if 'cal_data' in tod.wg._fields.keys():
        tod.wg.move('cal_data', None)
    ax = core.AxisManager(tod.samps, tod.dets, core.IndexAxis('wg_steps'))
    ax.wrap('idx_steps_start', idx_steps_start, [(0, 'wg_steps')])
    ax.wrap('idx_steps_stop',  idx_steps_stop,  [(0, 'wg_steps')])
    ax.wrap('theta_wire_rad', theta_wire_av , [(0, 'wg_steps')])
    ax.wrap('theta_wire_std', theta_wire_std, [(0, 'wg_steps')])
    ax.wrap('ts_step_mid',    ts_step_mid,    [(0, 'wg_steps')])
    ax.wrap('Q',              step_Q,         [(0, 'wg_steps'), (1, 'dets')])
    ax.wrap('U',              step_U,         [(0, 'wg_steps'), (1, 'dets')])
    ax.wrap('Qerr',           step_Qerr,      [(0, 'wg_steps'), (1, 'dets')])
    ax.wrap('Uerr',           step_Uerr,      [(0, 'wg_steps'), (1, 'dets')])
    tod.wg.wrap('cal_data', ax)
    return ax

### Circle fitting functions start from here ###
def _circle_model(params, x, y, xerr, yerr):
    cx, cy, r = params
    dx = x - cx
    dy = y - cy
    rho = np.sqrt(dx**2 + dy**2) + 1e-15 # to avoid zero division
    model = rho - r
    sigma_m2 = (dx/rho)**2 * xerr**2 + (dy/rho)**2 * yerr**2
    return model / np.sqrt(sigma_m2)

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
            - is_success : the status of how fits end.

    """
    _cal = tod.wg.cal_data
    fit_results = []
    params_val = []
    params_err = []
    covs = []
    res_vars = []
    is_success = []

    for _i in range(tod.dets.count):
        _obs_data = np.array([_cal.Q[:, _i], _cal.U[:, _i]])
        _obs_std = np.array([_cal.Qerr[:, _i], _cal.Uerr[:, _i]])

        # initial parameters
        cx0 = np.nanmean(_obs_data[0])
        cy0 = np.nanmean(_obs_data[1])
        cr = np.sqrt(np.nanmean(_obs_data[0]**2 + _obs_data[1]**2))
        _init_params = np.array([cx0, cy0, cr])
        _bounds = ([-2*cr, -2*cr, 0], [2*cr, 2*cr, 10*cr])

        # fit the data
        out = least_squares(
            _circle_model, _init_params, bounds=_bounds,
            args=(_obs_data[0], _obs_data[1], _obs_std[0], _obs_std[1])
        )
        fit_results.append(out)

        # Jacobian matrix and covariance matrix
        J = out.jac
        RSS = float(out.fun @ out.fun)
        sigma2_hat = RSS / (out.fun.size - len(_init_params))
        cov = np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))

        params_val.append(out.x)
        params_err.append(se)
        covs.append(cov)
        res_vars.append(sigma2_hat)
        is_success.append(out.success)

    ax = core.AxisManager(tod.dets)
    pvals = np.array(params_val)  # shape (ndet, 3)
    perrs = np.array(params_err)  # shape (ndet, 3)
    ax.wrap('cx0',                 pvals[:,0],          [(0, 'dets')])
    ax.wrap('cy0',                 pvals[:,1],          [(0, 'dets')])
    ax.wrap('cr',                  pvals[:,2],          [(0, 'dets')])
    ax.wrap('cx0_err',             perrs[:,0],          [(0, 'dets')])
    ax.wrap('cy0_err',             perrs[:,1],          [(0, 'dets')])
    ax.wrap('cr_err',              perrs[:,2],          [(0, 'dets')])
    ax.wrap('covariance',          np.array(covs),      [(0, 'dets')])
    ax.wrap('residual_var',        np.array(res_vars),  [(0, 'dets')])
    ax.wrap('is_success',          np.array(is_success),[(0, 'dets')])
    tod.wg.wrap('cfit_result', ax)
    return fit_results
### Circle fitting functions to here ###

### Elliptical fitting functions start from here ###
def _ellipse_model(params, x, y, xerr, yerr):
    A, B, a, b, theta = params

    # Translate to center
    dx = x - A
    dy = y - B

    # Rotate
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    # Implicit ellipse function F(x,y)=0
    m = (x_rot / a)**2 + (y_rot / b)**2 - 1.0

    # Gradients wrt observed x,y (needed for error propagation)
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    Fx = 2.0 * (x_rot * inv_a2 * cos_t - y_rot * inv_b2 * sin_t)
    Fy = 2.0 * (x_rot * inv_a2 * sin_t + y_rot * inv_b2 * cos_t)

    # Error propagation: Var[m] ≈ Fx^2 * xerr^2 + Fy^2 * yerr^2
    sigma_m2 = Fx**2 * (xerr**2) + Fy**2 * (yerr**2)

    # Avoid division by zero (e.g., at exact boundary), add tiny floor
    eps = 1e-15
    chi = m / np.sqrt(sigma_m2 + eps)   # shape: (N,)

    return chi

def fit_with_ellipse(tod):
    """
    Get the results by the ellipse fitting about the responce against the wires in Q+iU plane

    Parameters
    ----------
        tod : AxisManager

    Returns
    -------
        efit_results : list
            the result of the ellipse fitting of the wires' signal in the Q/U plane.
    Notes
    -----
        With this process, tod will include the several parameters of the fittings as tod.wg.cfit_result:

            - ex0(_err) : the estimated x-offset value(, and its fit err).
            - ey0(_err) : the estimated y-offset value(, and its fit err).
            - ea(_err) : Estimated major axis vaule(, and its fit err).
            - eb(_err) : Estimated minor axis vaule(, and its fit err).
            - etheta(_err) : Estimated rotation angle vaule(, and its fit err).
            - covariance : covariance matrix of the estimated parameters.
            - residual_var : Residual variance.
            - is_success : the status of how fits end. 
    """

    _cal = tod.wg.cal_data
    fit_results = []
    params_val = []
    params_err = []
    covs       = []
    res_vars   = []
    is_success = []

    for i in range(tod.dets.count):
        obs = np.array([_cal.Q[:, i], _cal.U[:, i]])
        err = np.array([_cal.Qerr[:, i], _cal.Uerr[:, i]])

        # initial parameters
        A0 = np.nanmean(obs[0])
        B0 = np.nanmean(obs[1])
        a0 = max(np.sqrt(obs[0]**2 + obs[1]**2))
        b0 = min(np.sqrt(obs[0]**2 + obs[1]**2))
        theta0 = 0.0
        _init_params = np.array([A0, B0, a0, b0, theta0])
        _bounds = ([-2*a0, -2*b0, 0, 0, -np.pi/2], [2*a0, 2*b0, 10*a0, 10*b0, np.pi/2])

        # fit the data
        out = least_squares(
            _ellipse_model, _init_params, bounds=_bounds,
            args=(obs[0], obs[1], err[0], err[1])
        )
        fit_results.append(out)

        J = out.jac
        cov = np.linalg.inv(J.T @ J)
        params_val.append(out.x)
        params_err.append(np.sqrt(np.diag(cov)) * np.sqrt(out.fun.size - len(_init_params)))
        covs.append(cov)
        res_vars.append(np.sum(out.fun**2) / (len(out.fun) - len(_init_params)))
        is_success.append(out.success)

    # wrap results into the axis manager
    ax = core.AxisManager(tod.dets)
    vals = np.array(params_val)  # shape (ndet, 5)
    errs = np.array(params_err)  # shape (ndet, 5)
    ax.wrap('ex0',      vals[:,0], [(0, 'dets')])
    ax.wrap('ey0',      vals[:,1], [(0, 'dets')])
    ax.wrap('ea',       vals[:,2], [(0, 'dets')])
    ax.wrap('eb',       vals[:,3], [(0, 'dets')])
    ax.wrap('etheta',   vals[:,4], [(0, 'dets')])
    ax.wrap('ex0_err',  errs[:,0], [(0, 'dets')])
    ax.wrap('ey0_err',  errs[:,1], [(0, 'dets')])
    ax.wrap('ea_err',   errs[:,2], [(0, 'dets')])
    ax.wrap('eb_err',   errs[:,3], [(0, 'dets')])
    ax.wrap('etheta_err',errs[:,4], [(0, 'dets')])
    ax.wrap('covariance', np.array(covs),       [(0, 'dets')])
    ax.wrap('residual_var', np.array(res_vars), [(0, 'dets')])
    ax.wrap('is_success',   np.array(is_success),  [(0, 'dets')])
    tod.wg.wrap('efit_result', ax)
    return fit_results
### Elliptical fitting functions to here ###

# Get and wrap the calibration angle by the wire grid
def get_cal_gamma(tod, merge=True, remove_cal_data=False):
    """
    Calibrate detector polarization angle with a circle model

    Parameters
    ----------
        tod : AxisManager
        merge : bool (default, True)
            whether merge the calibration results into tod or not
        remove_cal_data : bool (default, False)
            whether remove the intermediate product for the wire grid calibration in tod or not

    Returns
    -------
        ax : AxisManager
            which includes the calibrated angle of gamma in the sky coordinate, etc.

    Notes
    -----
        - gamma_raw(_err) is the raw output of the calibration for each wire grid step.
        - wires_relative_power is the signal intensity of the wires, which will relatively change depending on the ambient temperature.
        - gamma(_err) is the main result of the calibration using Sparse Wire Grid.
        - background_pol_rad or background_pol_relative_power is the Q/U-plane offsets of the detectors signal respect to the wires' reflection.
        - theta_det_instr is gamma translated to the instrumental angle printed on the silicon wafer.

    """
    _cd = tod.wg.cal_data
    _cfr = tod.wg.cfit_result
    _det_angle = []
    _det_angle_err = []
    for _i in range(tod.wg.wg_steps.count):

        # remove the offset of the circle
        Qcal = _cd.Q.T[:,_i] - _cfr.cx0
        Ucal = _cd.U.T[:,_i] - _cfr.cy0
        _atan_sig = np.arctan2(Ucal, Qcal)%(2*np.pi)

        # 0.5*(2*theta_det + 2*theta_wire) - theta_wire
        _det_angle.append(
            (0.5*(_atan_sig[:]) - _cd.theta_wire_rad[_i,np.newaxis]%(2*np.pi))
        )
        _det_angle_err.append(
            np.sqrt(
                (_cd.Uerr.T[:,_i]**2 + _cd.Qerr.T[:,_i]**2) * 0.5
                + (_cfr.cy0_err[_i]**2 + _cfr.cx0_err[_i]**2) * 0.5 / _cfr.cr[_i]
            )
        )

    _det_angle = np.unwrap(np.array(_det_angle).T, period=np.pi)
    _det_angle_err = np.array(_det_angle_err).T

    # calibrated gamma
    gamma = np.nanmean(_det_angle, axis=1)%np.pi
    gamma_err = np.nanmean(_det_angle_err, axis=1)/np.sqrt(np.shape(_det_angle_err)[1])

    # back ground polarization
    _bg_theta = (0.5*np.arctan2(_cfr.cy0, _cfr.cx0) - gamma)%np.pi
    _bg_amp = np.sqrt(_cfr.cx0**2 + _cfr.cy0**2)

    ax = core.AxisManager(tod.dets, tod.wg.wg_steps)
    ax.wrap('gamma_raw',                     _det_angle,        [(0, 'dets'), (1, 'wg_steps')])
    ax.wrap('gamma_raw_err',                 _det_angle_err,    [(0, 'dets'), (1, 'wg_steps')])
    ax.wrap('wires_relative_power',          _cfr.cr,           [(0, 'dets')])
    ax.wrap('gamma',                         gamma,             [(0, 'dets')])
    ax.wrap('gamma_err',                     gamma_err,         [(0, 'dets')])
    ax.wrap('background_pol_rad',            _bg_theta,         [(0, 'dets')])
    ax.wrap('background_pol_relative_power', _bg_amp,           [(0, 'dets')])
    ax.wrap('theta_det_instr',               0.5*np.pi - gamma, [(0, 'dets')]) # instumental angle of dets
    if remove_cal_data:
        tod.move('wg', None)
    if merge:
        tod.wg.wrap('gamma_cal', ax)
    return ax

def get_ecal_gamma(tod):
    """
    Calibrate dtector polarization angle with an ellipse model.
    See also get_cal_gamma.

    Parameters
    ----------
        tod : AxisManager
    Returns
    -------
        ax : AxisManager
            which includes the calibrated angle of gamma by ellipse model in the sky coordinate, etc.
    """
    _cal = tod.wg.cal_data
    _efr = tod.wg.efit_result

    ang = []
    for _i in range(tod.wg.wg_steps.count):
        Qraw = (_cal.Q.T[:,_i] - _efr.ex0)
        Uraw = (_cal.U.T[:,_i] - _efr.ey0)

        Qrot = + Qraw * np.cos(_efr.etheta) + Uraw * np.sin(_efr.etheta)
        Urot = - Qraw * np.sin(_efr.etheta) + Uraw * np.cos(_efr.etheta)
        Qmod = Qrot /_efr.ea
        Umod = Urot /_efr.eb

        Qcal = Qmod * np.cos(_efr.etheta) - Umod * np.sin(_efr.etheta)
        Ucal = Qmod * np.sin(_efr.etheta) + Umod * np.cos(_efr.etheta)

        atan = np.arctan2(Ucal, Qcal)
        atan = atan%(2*np.pi)
        ang.append(
            0.5*atan[:] - _cal.theta_wire_rad[_i, np.newaxis]%(2*np.pi)
        )
    det_angle = np.unwrap(np.array(ang).T, period=np.pi)
    R = np.abs(np.nanmean(np.exp(1j*det_angle), axis=1))
    std_err = np.sqrt(-2*np.log(R))
    det_angle_err = np.repeat(np.atleast_2d(std_err), tod.wg.wg_steps.count, axis=0).T

    # gamma calibrated with ellipse model
    gamma = np.nanmean(det_angle, axis=1)%np.pi
    gamma_err = std_err

    ax = core.AxisManager(tod.dets, tod.wg.wg_steps)
    ax.wrap('gamma_raw',        det_angle%np.pi,   [(0, 'dets'), (1, 'wg_steps')])
    ax.wrap('gamma_raw_err',    det_angle_err,     [(0, 'dets'), (1, 'wg_steps')])
    ax.wrap('theta_det_instr',  0.5*np.pi - gamma, [(0, 'dets')])
    ax.wrap('gamma',            gamma,             [(0, 'dets')])
    ax.wrap('gamma_err',        gamma_err,        [(0, 'dets')])
    tod.wg.wrap('gamma_ecal', ax)
    return ax