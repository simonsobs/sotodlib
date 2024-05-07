import os
import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy.optimize import curve_fit
from sotodlib.core import metadata
from sotodlib.io.metadata import read_dataset, write_dataset

from sotodlib import core
from sotodlib import coords
from sotodlib import tod_ops
import so3g
from so3g.proj import quat
import sotodlib.coords.planets as planets

from sotodlib.tod_ops import pca
from so3g.proj import Ranges, RangesMatrix
from pixell import enmap, enplot
from sotodlib.tod_ops.filters import high_pass_sine2, low_pass_sine2, fourier_filter

from sotodlib.site_pipeline import util
logger = util.init_logger(__name__, 'update_pointing: ')

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
    tod.restrict('dets', tod.dets.vals[np.in1d(tod.dets.vals, fp_rset['dets:readout_id'])])
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
                 sso_name='moon',
                 fp_hdf_file=None,
                 save_dir=None,
                 pipe=None,
                 force_zero_roll=False,
                 ds_factor=10, 
                 mask_deg=3,
                 fit_func_name = 'gaussian2d_nonlin',
                 max_non_linear_order = 1,
                 fwhm_init_deg = 0.5,
                 error_estimation_method='force_one_redchi2', # rms_from_data
                 flag_name_rms_calc = 'source',
                 flag_rms_calc_exclusive = True, 
                 save=True, ):
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
    - save_dir (str or None):
        Directory where the updated data will be saved. Required if save is True.
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
        Method for error estimation. Default is 'rms_from_data'. 'rms_from_data' and 'force_one_redchi2' are supported. 
        If 'rms_from_data', errorbar of each data point is set by root-mean-square of the data points flaged by 'flag_name_rms_calc', 
        and errorbar of xi,eta is set from the fit covariance matrix. If 'force_one_redchi2', the errorbar of (xi,eta) is equivalent the case 
        if the error bar of each data point is set as the reduced chi-square is equal to unity.
    - flag_name_rms_calc (str): 
        Name of the flag used for RMS calculation. Default is 'source'.
    - flag_rms_calc_exclusive (bool):
        Flag indicating whether the RMS calculation is exclusive to the flag. Default is True.
    - save (bool):
        Flag indicating whether to save the updated focal plane data. Default is True.

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
                    [[-np.inf, -np.inf, 0.1*fwhm_init_deg*coords.DEG, 0.1*fwhm_init_deg*coords.DEG, -np.pi, 0.1*ptp_val],
                     [np.inf, np.inf, 10*fwhm_init_deg*coords.DEG, 10*fwhm_init_deg*coords.DEG, np.pi, 10*ptp_val]]
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

def main(configs, obs_id, wafer_slot, sso_name=None,
         fp_hdf_file=None, save_dir=None, restrict_dets_for_debug=False):
    if type(configs) == str:
        configs = yaml.safe_load(open(configs, "r"))
    
    # Derive parameters from config file
    ctx = core.Context(configs.get('context_file'))
    if fp_hdf_file is None:
        fp_hdf_file = configs.get('fp_hdf_file', None)
    if save_dir is None:
        save_dir = configs.get('save_dir', None)
    
    # get sso_name if it is not specified
    obs_tags = ctx.obsdb.get(obs_id, tags=True)['tags']
    if sso_name is None:
        known_source_names = ['moon', 'jupiter']
        for _source_name in known_source_names:
            if _source_name in obs_tags:
                sso_name = _source_name
        if _source_name is None:
            raise ValueError('sso_name is not specified')
    
    # construct pipeline from configs
    pipe = Pipeline(configs["process_pipe"])
    for pipe_component in pipe:
        if pipe_component.name == 'compute_source_flags':
            pipe_component.process_cfgs['center_on'] = sso_name
    
    # Other parameters
    force_zero_roll = configs.get('force_zero_roll')
    ds_factor = configs.get('ds_factor')
    mask_deg = configs.get('mask_deg')
    fit_func_name = configs.get('fit_func_name')
    max_non_linear_order = configs.get('max_non_linear_order')
    fwhm_init_deg = configs.get('fwhm_init_deg')
    error_estimation_method = configs.get('error_estimation_method')
    flag_name_rms_calc = configls.get('flag_name_rms_calc')
    flag_rms_calc_exclusive = configls.get('flag_rms_calc_exclusive')
    
     
    # Load data
    logger.info('loading data')
    meta = ctx.get_meta(obs_id, dets={'wafer_slot': wafer_slot})
    if restrict_dets_for_debug is not False:
        meta.restrict('dets', meta.dets.vals[:restrict_dets_for_debug])
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
    
    os.makedirs(save_dir, exist_ok=True)
    write_dataset(focal_plane_rset, 
                  filename=os.path.join(save_dir, f'focal_plane_{obs_id}_{wafer_slot}.hdf'),
                  address='focal_plane',
                  overwrite=True)
        
    return

def get_parser():
    parser = argparse.ArgumentParser(description="Get updated result of pointings with tod-based results")
    parser.add_argument("ctx_file", type=str, help="Path to the context file.")
    parser.add_argument("obs_id", type=str, help="Observation ID.")
    parser.add_argument("wafer_slot", type=int, help="Wafer slot number.")
    parser.add_argument("sso_name", type=str,  default=None, help="Name of the Solar System Object (SSO).")
    parser.add_argument("save_dir", type=str, help="Directory to save the result.")
    parser.add_argument("restrict_dets_for_debug", action="store_true", help="Flag to restrict the number of detectors.")
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
