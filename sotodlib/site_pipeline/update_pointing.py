import os
import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy.optimize import minimize
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset

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

def _gaussian2d(xi, eta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
    """Simulate a time stream with an Gaussian beam model
    Args
    ------
    xi, eta: cordinates in the detector's system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data

def filter_tod(tod, cutoff_high=0.01, cutoff_low=1.8):
    if cutoff_low is not None:
        tod.signal = fourier_filter(tod, filt_function=low_pass_sine2(cutoff=cutoff_low),)
    if cutoff_high is not None:
        tod.signal = fourier_filter(tod, filt_function=high_pass_sine2(cutoff=cutoff_high),)
    return

def tod_process(tod):
    tod_ops.detrend_tod(tod)
    tod_ops.apodize_cosine(tod, apodize_samps=2000)
    filter_tod(tod)
    tod.restrict('samps', (tod.samps.offset+2000, tod.samps.offset+tod.samps.count-2000))
    return

def update_xieta(tod, sso_name='moon', ds_factor=10, fwhm = 1.*coords.DEG,
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
    mask_ds = slice(None, None, ds_factor)
    
    fp_isnan = np.isnan(tod.focal_plane.xi)
    if np.any(fp_isnan):
        tod.focal_plane.xi[fp_isnan] = 0.
        tod.focal_plane.eta[fp_isnan] = 0.
        tod.focal_plane.gamma[fp_isnan] = 0.
    
    coords.planets.compute_source_flags(tod, center_on=sso_name, max_pix=100000000,
                                wrap=sso_name, mask={'shape':'circle', 'xyr':[0,0,3]})
    
    summed_flag = np.sum(tod.flags[sso_name].mask()[~fp_isnan], axis=0).astype('bool')
    idx_hit = np.where(summed_flag)[0]
    idx_first, idx_last = idx_hit[0], idx_hit[-1]
    tod.restrict('samps', (tod.samps.offset+idx_first, tod.samps.offset+idx_last))
    csl = so3g.proj.CelestialSightLine.az_el(tod.timestamps[mask_ds], tod.boresight.az[mask_ds], 
                                             tod.boresight.el[mask_ds], weather="typical")
    q_bore = csl.Q
    
    ts_ds = tod.timestamps[mask_ds]
    sig_ds = tod.signal[:, mask_ds]
    source_flags_ds = tod.flags[sso_name].mask()[:, mask_ds]
    xieta_dict = {}
    for di, det in enumerate(tqdm(tod.dets.vals)):
        if fp_isnan[di]:
            xieta_dict[det] = {'xi':np.nan, 'eta':np.nan, 'R2':np.nan}
        else:
            mask_di = source_flags_ds[di]
            ts = ts_ds[mask_di]

            xieta_det = np.array([tod.focal_plane.xi[di], tod.focal_plane.eta[di]])
            q_det = so3g.proj.quat.rotation_xieta(xieta_det[0], xieta_det[1])
            d1_unix = np.median(ts)
            planet = planets.SlowSource.for_named_source(sso_name, d1_unix * 1.)
            ra0, dec0 = planet.pos(d1_unix)
            q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)
            q_total = ~q_det * ~q_bore * q_obj

            xi_src, eta_src, _ = quat.decompose_xieta(q_total)
            xieta_src = np.array([xi_src, eta_src])
            xieta_src = xieta_src[:, mask_di]


            
            sig = sig_ds[di][mask_di]
            amp = np.ptp(sig)
            def fit_func(xi0, eta0):
                model_tod = _gaussian2d(xieta_src[0], xieta_src[1], amp, xi0, eta0, fwhm, fwhm, 0)
                residual = sig - model_tod
                return np.sum(residual ** 2)
            
            res = minimize(lambda x: fit_func(*x), [0, 0])
            R2 = 1 - res.fun/np.sum((sig - np.mean(sig))**2)
            
            if np.rad2deg(np.sqrt(np.sum(res.x**2))) > 1.0:
                xieta_dict[det] = {'xi':np.nan, 'eta':np.nan, 'R2':np.nan}
            else:
                xieta_det += res.x
                xieta_dict[det] = {'xi':xieta_det[0], 'eta':xieta_det[1], 'R2':R2}
                
    focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'R2', 'xi', 'eta', 'gamma'])
    for det in tod.dets.vals:
        band = int(det.split('_')[-2])
        channel = int(det.split('_')[-1])
        focal_plane.rows.append((det, band, channel, xieta_dict[det]['R2'],
                                xieta_dict[det]['xi'], xieta_dict[det]['eta'], 0.))
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
