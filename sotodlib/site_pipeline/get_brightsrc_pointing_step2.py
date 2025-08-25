import os
import numpy as np
import yaml
import h5py
import argparse
import time
import glob
from tqdm import tqdm
from joblib import Parallel, delayed

from scipy.optimize import curve_fit
from sotodlib.core import metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.coords import brightsrc_pointing as bsp
from sotodlib import core
from sotodlib import coords
from sotodlib import tod_ops
import so3g
from so3g.proj import quat
import sotodlib.coords.planets as planets
from sotodlib.site_pipeline import util
from sotodlib.preprocess import Pipeline
logger = util.init_logger(__name__, 'update_pointing: ')

def _get_sso_names_from_tags(ctx, obs_id, candidate_names=['moon', 'jupiter']):
    obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
    sso_names = []
    for _name in candidate_names:
        if _name in obs_tags:
            sso_names.append(_name)
    if len(sso_names) == 0:
        raise NameError('Could not find sso_name from observation tags')
    else:
        return sso_names
    
def gaussian2d_nonlin(xieta, xi0, eta0, fwhm_xi, fwhm_eta, phi, a, nonlin_coeffs):
    """ An Gaussian beam model with non-linear response
    Args
    ------
    xi, eta: cordinates in the detector's system
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)
    a: float
        amplitude of the Gaussian beam model
    nonlin_coeffs: float
        Coefficient of non-linear term normalized by linear term (from 2nd term). 
        The order is ascending.
    Ouput:
    ------
    Model at xieta
    """
    xi, eta = xieta
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    lin_gauss = np.exp(xi_coef + eta_coef)
    polycoeffs_discending = np.hstack([nonlin_coeffs[::-1], [1, 0]])
    return a * np.poly1d(polycoeffs_discending)(lin_gauss)

def wrapper_gaussian2d_nonlin(xieta, xi0, eta0, fwhm_xi, fwhm_eta, phi, a, *args):
    """
    A wrapper for `gaussian2d_nonlin`
    """
    nonlin_coeffs = np.array(args)
    return  gaussian2d_nonlin(xieta, xi0, eta0, fwhm_xi, fwhm_eta, phi, a, nonlin_coeffs)

def wrap_fp_rset(tod, fp_rset):
    tod.restrict('dets', tod.dets.vals[np.isin(tod.dets.vals, fp_rset['dets:readout_id'])])
    focal_plane = core.AxisManager(tod.dets)
    focal_plane.wrap_new('xi', shape=('dets', ))
    focal_plane.wrap_new('eta', shape=('dets', ))
    focal_plane.wrap_new('gamma', shape=('dets', ))

    for di, det in enumerate(tod.dets.vals):
        di_rset = np.where(fp_rset['dets:readout_id'] == det)[0][0]
        focal_plane.xi[di] = fp_rset['xi'][di_rset]
        focal_plane.eta[di] = fp_rset['eta'][di_rset]
        focal_plane.gamma[di] = fp_rset['gamma'][di_rset]

    if 'focal_plane' in tod._fields.keys():
        tod.move('focal_plane', None)
    tod.wrap('focal_plane', focal_plane)
    return
    
def wrap_fp_from_hdf(tod, fp_hdf_file, data_set='focal_plane'):
    fp_rset = read_dataset(fp_hdf_file, data_set)
    wrap_fp_rset(tod, fp_rset)
    return
    
    
def update_xieta(tod, 
                 sso_name=None,
                 fp_hdf_file=None,
                 force_zero_roll=False,
                 pipe=None,
                 ds_factor=10, 
                 mask_deg=3,
                 fit_func_name = 'gaussian2d_nonlin',
                 max_non_linear_order = 1,
                 fwhm_init_deg = 0.5,
                 error_estimation_method='force_one_redchi2',
                 flag_name_rms_calc = 'source',
                 flag_rms_calc_exclusive = True, 
                 ):
    """
    Update xieta parameters for each detector by TOD fitting of a point source observation.

    Parameters:
    - tod : 
        an Axismanager object
    - sso_name (str): 
        Name of the Solar System Object (SSO).
    - fp_hdf_file (str or None): 
        Path to the HDF file containing focal plane information. Default is None.
        If None, tod.focal_plane is used for focal plane information.    
    - force_zero_roll (bool): 
        Flag indicating whether to force the roll to be zero. Default is False.
        If True, input and output focal plane information assumes force_zero_roll condition.
    - pipe (Pipeline or None): 
        Preprocessing pipeline to be applied to the TOD. Default is None, which
        do not apply any processing.
    - ds_factor (int): 
        Downsampling factor for fitting. Default is 10.
    - mask_deg (float): 
        Mask radius in degrees for source flagging. Default is 3.
    - fit_func_name (str): 
        Name of the fitting function. Default is 'gaussian2d_nonlin'. 'gaussian2d_nonlin' is only supported.
    - max_non_linear_order (int): 
        Maximum non-linear order for fitting function. Default is 1. If you want to use simple gaussian set it to be 1. 
        Higher order is for the case that detector response is distorted by non-point-like source or too-strogng source, such as the Moon.
    - fwhm_init_deg (float):
        Initial guess for full width at half maximum in degrees. Default is 0.5.
    - error_estimation_method (str):
        Method for error estimation. Default is 'force_one_redchi2'.  'force_one_redchi2' and 'rms_from_data' are supported. 
        If 'rms_from_data', errorbar of each data point is set by root-mean-square of the data points flaged by 'flag_name_rms_calc', 
        and errorbar of xi,eta is set from the fit covariance matrix. If 'force_one_redchi2', the errorbar of (xi,eta) is equivalent the case 
        if the error bar of each data point is set as the reduced chi-square is equal to unity.
    - flag_name_rms_calc (str): 
        Name of the flag used for RMS calculation. Default is 'source'.
    - flag_rms_calc_exclusive (bool):
        Flag indicating whether the RMS calculation is exclusive to the flag. Default is True.

    Returns:
    - focal_plane (ResultSet): ResultSet containing updated xieta parameters for each detector.
    """
    # if focal_plane result is specified, use the information as a prior
    if fp_hdf_file is not None:
        wrap_fp_from_hdf(tod, fp_hdf_file)
    
    # set dets without focal_plane info to have (xi, eta, gamma) = (0, 0, 0), just to avoid error
    xieta_isnan = (np.isnan(tod.focal_plane.xi)) | (np.isnan(tod.focal_plane.eta))
    gamma_isnan = np.isnan(tod.focal_plane.gamma)
    tod.focal_plane.xi[xieta_isnan] = 0.
    tod.focal_plane.eta[xieta_isnan] = 0.
    tod.focal_plane.gamma[gamma_isnan] = 0.
    
    # If input focal_plane is a result with `force_zero_roll`, set the roll to be zero
    # Original value is stored to `roll_original`
    if force_zero_roll:
        if 'roll_original' in tod.boresight._fields.keys():
            pass
        else:
            tod.boresight.wrap('roll_original', tod.boresight.roll, [(0, 'samps')])
            tod.boresight.roll *= 0.
    
    # compute source flags
    if 'source' in tod.flags._fields.keys():
        tod.flags.move('source', None)
    coords.planets.compute_source_flags(tod, 
                                        center_on=sso_name,
                                        max_pix=1e10, 
                                        wrap='source', 
                                        mask={'shape':'circle', 'xyr':[0.,0.,mask_deg]})
    
    # restrict data to duration when at least one detector hit the source
    summed_flag = np.sum(tod.flags['source'].mask()[~xieta_isnan], axis=0).astype('bool')
    idx_hit = np.where(summed_flag)[0]
    idx_first, idx_last = idx_hit[0], idx_hit[-1]
    tod.restrict('samps', (tod.samps.offset+idx_first, tod.samps.offset+idx_last))
    
    # run preprocess pipeline if provided
    if pipe is not None:
        proc_aman, success = pipe.run(tod)
    
    # get rms of flagged region for later error estimation
    if flag_rms_calc_exclusive:
        mask_for_rms_calc = tod.flags[flag_name_rms_calc].mask()
    else:
        mask_for_rms_calc = ~tod.flags[flag_name_rms_calc].mask()
    rms = np.ma.std(np.ma.masked_array(tod.signal, mask_for_rms_calc), axis=1).data
    if 'rms' in tod._fields.keys():
        tod.move('rms', None)
    tod.wrap('rms', rms, [(0, 'dets')])
    
    # use downsampled data for faster fitting
    mask_ds = slice(None, None, ds_factor)
    ts_ds = tod.timestamps[mask_ds]
    q_bore = so3g.proj.CelestialSightLine.az_el(ts_ds, tod.boresight.az[mask_ds], 
                                                tod.boresight.el[mask_ds], weather="typical").Q
    q_bore_roll = quat.rotation_iso(0, 0, np.median(tod.boresight.roll))
    sig_ds = tod.signal[:, mask_ds]
    source_flags_ds = tod.flags['source'].mask()[:, mask_ds]
    
    # fit each detector data
    xieta_dict = {}
    for di, det in enumerate(tqdm(tod.dets.vals)):
        mask_di = source_flags_ds[di]
        if np.any([xieta_isnan[di], np.all(mask_di==False), tod.rms[di]==0.]):
            xieta_dict[det] = {'xi': np.nan, 'eta':  np.nan, 'xi_err': np.nan, 'eta_err': np.nan,
                               'R2': np.nan, 'redchi2': np.nan}
        else:
            ts = ts_ds[mask_di]
            d1_unix = np.median(ts)
            xieta_det = np.array([tod.focal_plane.xi[di], tod.focal_plane.eta[di]])

            q_det = so3g.proj.quat.rotation_xieta(xieta_det[0], xieta_det[1])
            planet = planets.SlowSource.for_named_source(sso_name, d1_unix * 1.)
            ra0, dec0 = planet.pos(d1_unix)
            q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)
            q_total = ~q_det * ~q_bore_roll * ~q_bore * q_obj

            xi_src, eta_src, _ = quat.decompose_xieta(q_total)
            xieta_src = np.array([xi_src, eta_src])
            xieta_src = xieta_src[:, mask_di]
            sig = sig_ds[di][mask_di]            
            ptp_val = np.ptp(np.percentile(sig, [0.1, 99.9]))
            
            if fit_func_name == 'gaussian2d_nonlin':
                p0 = np.array([0., 0., fwhm_init_deg*coords.DEG, fwhm_init_deg*coords.DEG, 0., ptp_val])
                bounds = np.array(
                    [[-np.inf, -np.inf, fwhm_init_deg*coords.DEG/5., fwhm_init_deg*coords.DEG/5., -np.pi, 0.1*ptp_val],
                     [np.inf, np.inf, fwhm_init_deg*coords.DEG*5, fwhm_init_deg*coords.DEG*5, np.pi, 10*ptp_val]]
                              )
                if max_non_linear_order >= 2:
                    p0 = np.append(p0, np.zeros(max_non_linear_order-1))
                    bounds = np.hstack([bounds,
                                       np.vstack([[-np.inf * np.ones(max_non_linear_order-1),
                                                  np.inf * np.ones(max_non_linear_order-1)]])
                                        ])
                fit_func = wrapper_gaussian2d_nonlin
            else:
                raise NameError("Unsupported name for 'fit_func_name'")
            
            try:
                popt, pcov = curve_fit(fit_func, xdata=xieta_src, ydata=sig, sigma=tod.rms[di]*np.ones_like(sig), 
                                       p0=p0, bounds=bounds, absolute_sigma=True)

                chi2 = np.sum(((fit_func(xieta_src, *popt) - sig)/tod.rms[di])**2)
                redchi2 = chi2 / (np.prod(xieta_src.shape) - popt.shape[0])
                R2 = 1. - np.sum((fit_func(xieta_src, *popt) - sig)**2) / np.sum((sig - sig.mean())**2)
                xi_opt, eta_opt = popt[0], popt[1]

                if error_estimation_method == 'rms_from_data':
                    xi_err, eta_err = np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])                
                elif error_estimation_method == 'force_one_redchi2':
                    # The error of (xi, eta) is equivalent the case if the error bar of each data point is set 
                    # as the reduced chi-square is equal to unity.
                    xi_err, eta_err = np.sqrt(pcov[0,0] * redchi2), np.sqrt(pcov[1,1] * redchi2)
                    redchi2 = 1.
                else:
                    raise NameError("Unsupported name for 'error_estimation_method'")

                xieta_det += np.array([xi_opt, eta_opt])
                xieta_dict[det] = {'xi': xieta_det[0], 'eta': xieta_det[1], 'xi_err': xi_err, 'eta_err': eta_err,
                                   'R2': R2, 'redchi2': redchi2}
            except RuntimeError:
                xieta_dict[det] = {'xi': np.nan, 'eta':  np.nan, 'xi_err': np.nan, 'eta_err': np.nan,
                                   'R2': np.nan, 'redchi2': np.nan}
            
    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 'xi_err', 'eta_err', 'R2', 'redchi2'])
    for det in tod.dets.vals:
        focal_plane.rows.append((det, xieta_dict[det]['xi'], xieta_dict[det]['eta'], 0.,
                                 xieta_dict[det]['xi_err'], xieta_dict[det]['eta_err'], 
                                 xieta_dict[det]['R2'], xieta_dict[det]['redchi2'],
                                ))

    return focal_plane

def main_one_wafer(configs, obs_id, wafer_slot, sso_name=None, 
                   restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    
    # Derive parameters from config file
    ctx = core.Context(configs.get('context_file'))
    
    # get prior
    fp_hdf_file = configs.get('fp_hdf_file', None)
    fp_hdf_dir = configs.get('fp_hdf_dir', None)
    if fp_hdf_file is None:
        if fp_hdf_dir is not None:
            fp_hdf_file = os.path.join(fp_hdf_dir, f'focal_plane_{obs_id}_{wafer_slot}.hdf')
            if not os.path.exists(fp_hdf_file):
                fp_hdf_file = None
    
    result_dir = configs.get('result_dir')
    force_zero_roll = configs.get('force_zero_roll', True)
    if force_zero_roll:
        result_dir = result_dir + '_force_zero_roll'
    
    # get sso_name if it is not specified
    if sso_name is None:
        logger.info('deriving sso_name from observation tag')
        obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
        sso_names = _get_sso_names_from_tags(ctx, obs_id)
        sso_name = sso_names[0]
        if len(sso_names) >= 2:
            logger.info(f'sso_names of {sso_names} are found from observation tags.' + 
                        f'Processing only {sso_name}')
    
    # construct pipeline from configs
    pipe = Pipeline(configs["process_pipe"], logger=logger)
    for pipe_component in pipe:
        if pipe_component.name == 'source_flags':
            pipe_component.calc_cfgs['center_on'] = sso_name
    
    # Other parameters
    force_zero_roll = configs.get('force_zero_roll')
    ds_factor = configs.get('ds_factor', 20)
    mask_deg = configs.get('mask_deg', 3.0)
    fit_func_name = configs.get('fit_func_name', 'gaussian2d_nonlin')
    max_non_linear_order = configs.get('max_non_linear_order', 2)
    fwhm_init_deg = configs.get('fwhm_init_deg', 0.5)
    error_estimation_method = configs.get('error_estimation_method', 'force_one_redchi2')
    flag_name_rms_calc = configs.get('flag_name_rms_calc', 'source')
    flag_rms_calc_exclusive = configs.get('flag_rms_calc_exclusive', True)
    
     
    # Load data
    logger.info('loading data')
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    if restrict_dets_for_debug is not False:
        try:
            restrict_dets_for_debug = int(restrict_dets_for_debug)
            meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
        except ValueError:
            _testdets = restrict_dets_for_debug.split(',')
            restrict_list = [det.split('\'')[1].strip() for det in _testdets]
            meta.restrict('dets', restrict_list)
            
    tod = ctx.get_obs(meta)
    
    # get pointing
    focal_plane_rset = update_xieta( tod, 
                                     sso_name=sso_name,
                                     fp_hdf_file=fp_hdf_file,
                                     force_zero_roll=force_zero_roll,
                                     pipe=pipe,
                                     ds_factor=ds_factor,
                                     mask_deg=mask_deg,
                                     fit_func_name = fit_func_name,
                                     max_non_linear_order = max_non_linear_order,
                                     fwhm_init_deg = fwhm_init_deg,
                                     error_estimation_method=error_estimation_method,
                                     flag_name_rms_calc = flag_name_rms_calc,
                                     flag_rms_calc_exclusive = flag_rms_calc_exclusive, 
                                     )
    
    os.makedirs(result_dir, exist_ok=True)
    write_dataset(focal_plane_rset, 
                  filename=os.path.join(result_dir, f'focal_plane_{obs_id}_{wafer_slot}.hdf'),
                  address='focal_plane',
                  overwrite=True)
        
    return

def main_one_wafer_dummy(configs, obs_id, wafer_slot, restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    result_dir = configs.get('result_dir')
    force_zero_roll = configs.get('force_zero_roll', True)
    if force_zero_roll:
        result_dir = result_dir + '_force_zero_roll'
    
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    if restrict_dets_for_debug is not False:
        try:
            restrict_dets_for_debug = int(restrict_dets_for_debug)
            meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
        except ValueError:
            _testdets = restrict_dets_for_debug.split(',')
            restrict_list = [det.split('\'')[1].strip() for det in _testdets]
            meta.restrict('dets', restrict_list)
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    
    fp_rset_dummy = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 
                                             'xi_err', 'eta_err', 'R2', 'redchi2'])
    for det in meta.dets.vals:
        fp_rset_dummy.rows.append((det, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        
    os.makedirs(result_dir, exist_ok=True)
    write_dataset(fp_rset_dummy, 
                  filename=os.path.join(result_dir, result_filename),
                  address='focal_plane',
                  overwrite=True)
    return

def combine_pointings(pointing_result_files):
    combined_dict = {}
    for file in pointing_result_files:
        rset = read_dataset(file, 'focal_plane')
        for row in rset[:]:
            if row['dets:readout_id'] not in combined_dict.keys():
                combined_dict[row['dets:readout_id']] = {}
                combined_dict[row['dets:readout_id']]['xi'] = row['xi']
                combined_dict[row['dets:readout_id']]['eta'] = row['eta']
                combined_dict[row['dets:readout_id']]['gamma'] = row['gamma']
                combined_dict[row['dets:readout_id']]['xi_err'] = row['xi_err']
                combined_dict[row['dets:readout_id']]['eta_err'] = row['eta_err']
                combined_dict[row['dets:readout_id']]['R2'] = row['R2']
                combined_dict[row['dets:readout_id']]['redchi2'] = row['redchi2']

    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 'xi_err', 'eta_err', 'R2', 'redchi2'])
    
    for det, val in combined_dict.items():
        focal_plane.rows.append((det, val['xi'], val['eta'], val['gamma'], val['xi_err'], val['eta_err'], val['R2'], val['redchi2']))
    return focal_plane

def parallel_process_wafer_slot(configs, obs_id, wafer_slot, sso_name, restrict_dets_for_debug):
    logger.info(f'Processing {obs_id}, {wafer_slot}')
    main_one_wafer(configs=configs,
                   obs_id=obs_id,
                   wafer_slot=wafer_slot,
                   sso_name=sso_name,
                   restrict_dets_for_debug=restrict_dets_for_debug)

def main_one_obs(configs, obs_id, sso_name=None,
                restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    result_dir = configs.get('result_dir')
    force_zero_roll = configs.get('force_zero_roll', True)
    if force_zero_roll:
        result_dir = result_dir + '_force_zero_roll'
    optics_config_fn = configs.get('optics_config_fn')
    
    hit_time_threshold = configs.get('hit_time_threshold', 600)
    hit_circle_r_deg = configs.get('hit_circle_r_deg', 7.0)
    
    if sso_name is None:
        logger.info('deriving sso_name from observation tag')
        obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
        sso_names = _get_sso_names_from_tags(ctx, obs_id)
        sso_name = sso_names[0]
        if len(sso_names) >= 2:
            logger.info(f'sso_names of {sso_names} are found from observation tags.' + 
                        f'Processing only {sso_name}')
    
    tod = ctx.get_obs(obs_id, dets=[])
    streamed_wafer_slots = ['ws{}'.format(index) for index, bit in enumerate(obs_id.split('_')[-1]) if bit == '1']
    processed_wafer_slots = []
    finished_wafer_slots = []
    skipped_wafer_slots = []
    check_dir = result_dir + '_force_zero_roll' if force_zero_roll else result_dir

    for ws in streamed_wafer_slots:
        hit_time = bsp.get_rough_hit_time(tod,
                                          wafer_slot=ws,
                                          sso_name=sso_name,
                                          circle_r_deg=hit_circle_r_deg,
                                          optics_config_fn=optics_config_fn)
        logger.info(f'hit_time for {ws} is {hit_time:.1f} [sec]')
        if hit_time >= hit_time_threshold:
            if os.path.exists(os.path.join(check_dir, f'focal_plane_{obs_id}_{ws}.hdf')):
                finished_wafer_slots.append(ws)
            else:
                processed_wafer_slots.append(ws)
        else:
            skipped_wafer_slots.append(ws)

    logger.info(f'Found saved data for these wafer_slots: {finished_wafer_slots}')
    logger.info(f'Will continue for these wafer_slots: {processed_wafer_slots}')

    if configs.get('parallel_job'):
        logger.info('Continuing with parallel job')
        try:
            n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        except: 
            n_jobs = -1
        Parallel(n_jobs=n_jobs)(
            delayed(parallel_process_wafer_slot)(
                configs,
                obs_id, 
                wafer_slot, 
                sso_name, 
                restrict_dets_for_debug
            ) 
            for wafer_slot in processed_wafer_slots
        )
    else:
        logger.info('Continuing with serial processing of wafers.')
        for wafer_slot in processed_wafer_slots:
            logger.info(f'Processing {obs_id}, {wafer_slot}')
            main_one_wafer(configs=configs,
                           obs_id=obs_id,
                           wafer_slot=wafer_slot,
                           sso_name=sso_name,
                           restrict_dets_for_debug=restrict_dets_for_debug)            

    logger.info(f'create dummy hdf for non-hitting wafer: {skipped_wafer_slots}')
    for wafer_slot in skipped_wafer_slots:
        main_one_wafer_dummy(configs=configs,
                       obs_id=obs_id,
                       wafer_slot=wafer_slot,
                       restrict_dets_for_debug=restrict_dets_for_debug)
    
    logger.info('making combined result')
    pointing_result_files = glob.glob(os.path.join(result_dir, f'focal_plane_{obs_id}_ws[0-6].hdf'))
    fp_rset_full = combine_pointings(pointing_result_files)
    fp_rset_full_file = os.path.join(os.path.join(result_dir, f'focal_plane_{obs_id}_all.hdf'))
    write_dataset(fp_rset_full, filename=fp_rset_full_file,
                  address='focal_plane', overwrite=True)
    logger.info(f'ta da! Finsihed with {obs_id}')
        
def main(configs, min_ctime=None, max_ctime=None, update_delay=None,
         obs_id=None, wafer_slot=None, sso_name=None, restrict_dets_for_debug=False):
    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) - update_delay*86400
        
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs.get('context_file'))
    
    if obs_id is None:
        query_text = configs.get('query_text', None)
        query_tags = configs.get('query_tags', None)
        tot_query = "and "
        if query_text is not None:
            tot_query += f"{query_text} and "
        if min_ctime is not None:
            tot_query += f"timestamp>={min_ctime} and "
        if max_ctime is not None:
            tot_query += f"timestamp<={max_ctime} and "
        tot_query = tot_query[4:-4]
        if tot_query == "":
            tot_query = "1"
            
        logger.info(f'tot_query: {tot_query}')
        obs_list= ctx.obsdb.query(tot_query, query_tags)

        for obs in obs_list:
            obs_id = obs['obs_id']
            logger.info(f'Processing {obs_id}')
            main_one_obs(configs=configs, obs_id=obs_id,
                        restrict_dets_for_debug=restrict_dets_for_debug)
    
    elif obs_id is not None:
        logger.info(f'Processing {obs_id}')
        if wafer_slot is None:
            main_one_obs(configs=configs, obs_id=obs_id, sso_name=sso_name,
                         restrict_dets_for_debug=restrict_dets_for_debug)
        else:
            main_one_wafer(configs=configs, obs_id=obs_id, wafer_slot=wafer_slot, sso_name=sso_name, 
                           restrict_dets_for_debug=restrict_dets_for_debug)  
    return
    
    
def get_parser():
    parser = argparse.ArgumentParser(description="Get updated result of pointings with tod-based results")
    parser.add_argument("configs", type=str, help="Path to the configuration file")
    parser.add_argument('--min_ctime', type=int, help="Minimum timestamp for the beginning of an observation list")
    parser.add_argument('--max_ctime', type=int, help="Maximum timestamp for the beginning of an observation list")
    parser.add_argument('--update-delay', type=int, help="Number of days (unit is days) in the past to start observation list.")                         
    parser.add_argument("--obs_id", type=str, 
                        help="Specific observation obs_id to process. If provided, overrides other filtering parameters.")
                         
    parser.add_argument("--wafer_slot", type=str, default=None, 
                        help="Wafer slot to be processed (e.g., 'ws0', 'ws3'). Valid only when obs_id is specified.")
                         
    parser.add_argument("--sso_name", type=str, default=None,
                        help="Name of solar system object (e.g., 'moon', 'jupiter'). If not specified, get sso_name from observation tags. "\
                       + "Valid only when obs_id is specified")                     
    parser.add_argument("--restrict_dets_for_debug", type=int, default=False)
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
