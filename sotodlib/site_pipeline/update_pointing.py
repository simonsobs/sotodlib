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

def _gaussian2d(xieta, xi0, eta0, fwhm_xi, fwhm_eta, phi, a):
    """Simulate a time stream with an Gaussian beam model
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

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi, eta = xieta
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data

def _gaussian2d_nonlin(xieta, xi0, eta0, fwhm_xi, fwhm_eta, phi, a, b2,):
    """Simulate a time stream with an Gaussian beam model with non-linear response
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
    b2: float
        coefficient of 2nd-order term

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi, eta = xieta
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    _y = np.exp(xi_coef + eta_coef)
    sim_data = a * (_y + b2*_y**2)
    return sim_data

# def filter_tod(tod, cutoff_high=0.01, cutoff_low=1.8):
#     if cutoff_low is not None:
#         tod.signal = fourier_filter(tod, filt_function=low_pass_sine2(cutoff=cutoff_low),)
#     if cutoff_high is not None:
#         tod.signal = fourier_filter(tod, filt_function=high_pass_sine2(cutoff=cutoff_high),)
#     return

# def tod_process(tod):
#     tod_ops.detrend_tod(tod)
#     tod_ops.apodize_cosine(tod, apodize_samps=2000)
#     filter_tod(tod)
#     tod.restrict('samps', (tod.samps.offset+2000, tod.samps.offset+tod.samps.count-2000))
#     return

def wrap_fp_from_hdf(tod, fp_hdf_file, data_set='focal_plane'):
    fp_rset = read_dataset(fp_hdf_file, data_set)
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
    
    
def update_xieta(tod, 
                 sso_name='moon',
                 fp_hdf_file=None,
                 input_force_zero_roll=False,
                 pipe=None,
                 ds_factor=10, 
                 mask_deg=3,
                 fit_func_name = '_gaussian2d_nonlin',
                 fwhm_init_deg = 0.5,
                 error_estimation_method='force_one_redchi2', # rms_from_data
                 flag_name_rms_calc = 'source',
                 flag_rms_calc_exclusive = True, 
                 save=False, result_dir=None, filename=None):
    """
    Update xieta parameters for each detector by tod fitting of a point source observation

    Parameters:
    - tod : an Axismanager object
    - sso_name (str): Name of the Solar System Object (SSO).
    - ds_factor (int): Downsampling factor for processing TOD.
    - fwhm (float): Full width at half maximum of the Gaussian model.
    - save (bool): Flag indicating whether to save the updated focal plane data.
    - result_dir (str): Directory where the updated data will be saved.
    - filename (str): Name of the file to save the updated data.

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
    if input_force_zero_roll:
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
            
            if fit_func_name == '_gaussian2d':
                p0 = (0., 0., fwhm_init_deg*coords.DEG, fwhm_init_deg*coords.DEG, 0., ptp_val)
                fit_func = _gaussian2d
            elif fit_func_name == '_gaussian2d_nonlin':
                p0 = (0., 0., fwhm_init_deg*coords.DEG, fwhm_init_deg*coords.DEG, 0., ptp_val, -0.1,)
                fit_func = _gaussian2d_nonlin
            
            popt, pcov = curve_fit(fit_func, xdata=xieta_src, ydata=sig, p0=p0, sigma=tod.rms[di]*np.ones_like(sig),
                                   absolute_sigma=True, maxfev=int(1e5))
            
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
            
    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'xi', 'eta', 'gamma', 'xi_err', 'eta_err', 'R2', 'redchi2'])
    for det in tod.dets.vals:
        focal_plane.rows.append((det, xieta_dict[det]['xi'], xieta_dict[det]['eta'], 0.,
                                 xieta_dict[det]['xi_err'], xieta_dict[det]['eta_err'], 
                                 xieta_dict[det]['R2'], xieta_dict[det]['redchi2'],
                                ))
    if save:
        assert result_dir is not None
        assert filename is not None
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        write_dataset(focal_plane, 
                      filename=os.path.join(result_dir, filename),
                      address='focal_plane',
                      overwrite=True)
    return focal_plane

def main(ctx_file, obs_id, wafer_slot, sso_name, result_dir, 
         ds_factor=10, fwhm = 1.*coords.DEG, restrict_dets=False):
    ctx = core.Context(ctx_file)
    meta = ctx.get_meta(obs_id)
    meta.restrict('dets', meta.dets.vals[meta.det_info.wafer_slot == wafer_slot])
    if restrict_dets:
        meta.restrict('dets', meta.dets.vals[:100])
        
    logger.info('loading data')
    tod = ctx.get_obs(meta)
    logger.info('tod processing')
    tod_process(tod)
    
    if not os.path.exists(result_dir):
        logger.info(f'Make a directory: f{result_dir}')
        os.makedirs(result_dir)
    
    result_filename = f'focal_plane_{obs_id}_{wafer_slot}.hdf'
    focal_plane_rset = update_xieta(tod=tod, sso_name=sso_name, ds_factor=ds_factor, fwhm=fwhm,
                                    save=True, result_dir=result_dir, filename=result_filename)
    return

def get_parser():
    parser = argparse.ArgumentParser(description="Get updated result of pointings with tod-based results")
    parser.add_argument("ctx_file", type=str, help="Path to the context file.")
    parser.add_argument("obs_id", type=str, help="Observation ID.")
    parser.add_argument("wafer_slot", type=int, help="Wafer slot number.")
    parser.add_argument("sso_name", type=str, help="Name of the Solar System Object (SSO).")
    parser.add_argument("result_dir", type=str, help="Directory to save the result.")
    parser.add_argument("--ds_factor", type=int, default=10, help="Downsampling factor for TOD processing.")
    parser.add_argument("--fwhm", type=float, default=1.0, help="Full width at half maximum of the Gaussian model.")
    parser.add_argument("--restrict_dets", action="store_true", help="Flag to restrict the number of detectors.")
    return parser

if __name__ == '__main__':
    util.main_launcher(main, get_parser)
