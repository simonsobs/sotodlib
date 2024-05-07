import os
import re
from tqdm import tqdm
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

from sotodlib import core
from sotodlib import coords
from sotodlib.coords import optics
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset, read_dataset

from so3g.proj import quat
from pixell import enmap
import h5py
from scipy.ndimage import maximum_filter

def get_planet_trajectry(tod, planet, _split=20, return_model=False):
    """
    Generate the trajectory of a given planet over a specified time range.

    Parameters:
        tod : An axis manager
        planet (str): The name of the planet for which to generate the trajectory.
        _split (int, optional): Number of points to interpolate the trajectory. Defaults to 20.
        return_model (bool, optional): If True, returns interpolation functions of az and el. Defaults to False.

    Returns:
        If return_model is True:
            tuple: Tuple containing interpolation functions for azimuth and elevation.
        If return_model is False:
            array: Array of quaternions representing trajectry of the planet at each timestamp.
    """
    timestamps_sparse = np.linspace(tod.timestamps[0], tod.timestamps[-1], _split)
    
    planet_az_sparse = np.zeros_like(timestamps_sparse)
    planet_el_sparse = np.zeros_like(timestamps_sparse)
    for i, timestamp in enumerate(timestamps_sparse):
        az, el, _ = coords.planets.get_source_azel(planet, timestamp)
        planet_az_sparse[i] = az
        planet_el_sparse[i] = el
    planet_az_func = interpolate.interp1d(timestamps_sparse, planet_az_sparse, kind="quadratic", fill_value='extrapolate')
    planet_el_func = interpolate.interp1d(timestamps_sparse, planet_el_sparse, kind="quadratic", fill_value='extrapolate')
    if return_model:
        return planet_az_func, planet_el_func
    else:
        planet_az = planet_az_func(tod.timestamps)
        planet_el = planet_el_func(tod.timestamps)
        q_planet = quat.rotation_lonlat(planet_az, planet_el)
        return q_planet

def get_wafer_centered_sight(tod=None, planet=None, q_planet=None, q_bs=None, q_wafer=None):
    """
    Calculate the sightline vector from the focal plane, centered on the wafer, to a planet.

    Parameters:
        tod : An axis manager
        planet (str): The name of the planet to calculate the sightline vector.
        q_planet (optional): Quaternion representing the trajectry of the planet. 
            If None, it will be computed using get_planet_trajectory. Defaults to None.
        q_bs (optional): Quaternion representing the trajectry of the boresight.
            If None, it will be computed using the current boresight angles from tod. Defaults to None.
        q_wafer (optional): Quaternion representing the center of wafer to the center of boresight.
            If None, it will be computed using the median of the focal plane xi and eta from tod.focal_plane. 
            Defaults to None.

    Returns:
        Sightline vector for the planet trajectry centered on the center of the wafer.
    """
    if q_planet is None:
        q_planet = get_planet_trajectry(tod, planet)
    if q_bs is None:
        q_bs = quat.rotation_lonlat(tod.boresight.az, tod.boresight.el)
    if q_wafer is None:
        q_wafer = quat.rotation_xieta(np.nanmedian(tod.focal_plane.xi), 
                                     np.nanmedian(tod.focal_plane.eta))
        
    xi_wafer, eta_wafer, _ = quat.decompose_xieta(q_wafer)
    q_wafer_f = quat.rotation_xieta(-xi_wafer, eta_wafer)
    z_to_x = quat.rotation_lonlat(0, 0)
    sight = z_to_x * ~(q_bs * q_wafer_f) * q_planet
    return sight

def get_wafer_xieta(wafer_slot, optics_config_fn, xieta_bs_offset=(0., 0.), 
                    roll_bs_offset=0., tod=None, wrap_to_tod=True,):
    """
    Calculate the xi and eta coordinates for a given wafer slot on the focal plane.

    Parameters:
        wafer_slot (str): The slot identifier of the wafer.
        optics_config_fn (str): File name containing the optics configuration.
        xieta_bs_offset (tuple): Offset in xieta coordinates for the focal plane, default is (0., 0.).
        roll_bs_offset (float): Boresight roll offset. Default is 0
        tod (TimeOrderedData): TOD object to which focal plane infomation that all detectors have uniform pointing at center of the wafer is wrapped.        
        wrap_to_tod (bool): If True, wrap the calculated xi and eta coordinates to the Time-Ordered Data (TOD), default is True.

    Returns:
        tuple: A tuple containing the calculated xi and eta coordinates for the specified wafer slot.
    """
    
    optics_config = optics.load_ufm_to_fp_config(optics_config_fn)['SAT']
    wafer_x, wafer_y = optics_config[wafer_slot]['dx'], optics_config[wafer_slot]['dy']
    wafer_r = np.sqrt(wafer_x**2 + wafer_y**2)
    wafer_theta = np.arctan2(wafer_y, wafer_x)

    fp_to_sky = optics.sat_to_sky(optics.SAT_X, optics.SAT_LON)
    lon = fp_to_sky(wafer_r)
    
    q1 = quat.rotation_iso(lon, 0)
    q2 = quat.rotation_iso(0, 0, np.pi/2 - wafer_theta + roll_bs_offset)
    q3 = quat.rotation_xieta(xieta_bs_offset[0], xieta_bs_offset[1])
    q = q3 * q2 * q1
    
    xi_wafer, eta_wafer, _ = quat.decompose_xieta(q)
    if wrap_to_tod:
        if tod is None:
            raise ValueError('tod is not provided.')
        if 'focal_plane' in tod._fields.keys():
            tod.move('focal_plane', None)
        focal_plane = core.AxisManager(tod.dets)
        focal_plane.wrap('xi', np.ones(tod.dets.count, dtype='float32') * xi_wafer, [(0, 'dets')])
        focal_plane.wrap('eta', np.ones(tod.dets.count, dtype='float32') * eta_wafer, [(0, 'dets')])
        focal_plane.wrap('gamma', np.zeros(tod.dets.count, dtype='float32'), [(0, 'dets')])
        tod.wrap('focal_plane', focal_plane)
        
        # set boresight roll to zero
        tod.boresight.wrap('roll_original', tod.boresight.roll, [(0, 'samps')])
        tod.boresight.roll *= 0.
        
    return xi_wafer, eta_wafer


def get_rough_hit_time(tod, wafer_slot, sso_name, circle_r_deg=7.,optics_config_fn=None):
    """
    Estimate the rough hit time for a axismanager, wafer_slot, and sso_name.

    Parameters:
        tod : An AxisManager object
        wafer_slot (str): Identifier for the wafer slot.
        sso_name (str): Name of the Solar System Object (e.g., 'moon', 'jupiter').
        circle_r_deg (float, optional): Radius in degrees defining the circular region around the wafer center. 
                                        Defaults to 7 degrees.

    Returns:
        float: Estimated rough hit time within the circular region around the wafer center.
    """
    q_bs = quat.rotation_lonlat(tod.boresight.az, tod.boresight.el)
    q_planet = get_planet_trajectry(tod, sso_name)
    xi_wafer, eta_wafer = get_wafer_xieta(wafer_slot, optics_config_fn=optics_config_fn, 
                                            roll_bs_offset=np.median(tod.boresight.roll), wrap_to_tod=False)
    q_wafer = quat.rotation_xieta(xi_wafer, eta_wafer)
    
    q_wafer_centered = get_wafer_centered_sight(q_planet=q_planet, q_bs=q_bs, q_wafer=q_wafer)
    x_to_z = ~quat.rotation_lonlat(0, 0)
    xi_wafer_centered, eta_wafer_centered, _ = quat.decompose_xieta(x_to_z * q_wafer_centered)
    r_wafer_centered = np.sqrt(xi_wafer_centered**2 + eta_wafer_centered**2)
    hit_time = (tod.timestamps[-1] - tod.timestamps[0]) * np.mean(np.rad2deg(r_wafer_centered) < circle_r_deg)
    return hit_time


def make_wafer_centered_maps(tod, sso_name, optics_config_fn, map_hdf, 
                           xieta_bs_offset=(0., 0.), roll_bs_offset=None,
                           signal='signal', wafer_mask_deg=8., res_deg=0.3, cuts=None,):
    """
    Generate boresight-centered maps from Time-Ordered Data (TOD) for each individual detector.

    Parameters:
        tod : an axismanager object
        sso_name (str): Name of the planet for which the trajectory is calculated.
        optics_config_fn (str): File name containing the optics configuration.
        map_hdf (str): Path to the HDF5 file where the maps will be saved.
        xieta_bs_offset (tuple): Offset in xieta coordinates for the boresight, default is (0., 0.).
        roll_bs_offset (float): Offset in roll angle for the boresight, default is None.
        signal (str): Name of the signal to be used, default is 'signal'.
        wcs_kernel (ndarray): WCS kernel for mapping, default is None.
        res_deg (float): Resolution of the map in degrees, default is 0.3 degrees.
        cuts (tuple): Cuts to be applied to the map, default is None.

    Returns:
        None
    """    
    q_planet = get_planet_trajectry(tod, sso_name)
    q_bs = quat.rotation_lonlat(tod.boresight.az, tod.boresight.el)
    
    if roll_bs_offset is None:
        roll_bs_offset = np.mean(tod.boresight.roll)
        
    # wafer
    if np.unique(tod.det_info.wafer_slot).shape[0] > 1:
        raise ValueError('tod include detectors from more than one wafer')
    wafer_slot = tod.det_info.wafer_slot[0]
    xi_wafer, eta_wafer = get_wafer_xieta(wafer_slot=wafer_slot, 
                    xieta_bs_offset=xieta_bs_offset, 
                    roll_bs_offset=roll_bs_offset,
                    tod=tod,
                    optics_config_fn=optics_config_fn,
                    wrap_to_tod=True)
    
    coords.planets.compute_source_flags(tod, center_on=sso_name, max_pix=100000000,
                                wrap='source', mask={'shape':'circle', 'xyr':[0., 0., wafer_mask_deg]})
    
    
    
    q_wafer = quat.rotation_xieta(xi_wafer, eta_wafer)
    sight = get_wafer_centered_sight(tod, sso_name, q_planet, q_bs, q_wafer)
    xi0 = tod.focal_plane.xi[0]
    eta0 = tod.focal_plane.eta[0]
    xi_bs_offset, eta_bs_offset = xieta_bs_offset    
    tod.focal_plane.xi *= 0.
    tod.focal_plane.eta *= 0.
    tod.boresight.roll *= 0.
    
    
    box = np.deg2rad([[-wafer_mask_deg, -wafer_mask_deg], [wafer_mask_deg, wafer_mask_deg]])
    geom = enmap.geometry(pos=box, res=res_deg*coords.DEG)
    if cuts is None:
        cuts = ~tod.flags['source']
    P = coords.P.for_tod(tod=tod, geom=geom, comps='T', cuts=cuts, sight=sight, threads=False)
    for di, det in enumerate(tqdm(tod.dets.vals)):
        det_weights = np.zeros(tod.dets.count, dtype='float32')
        det_weights[di] = 1.
        mT_weighted = P.to_map(tod=tod, signal=signal, comps='T', det_weights=det_weights)
        wT = P.to_weights(tod, signal=signal, comps='T', det_weights=det_weights)
        mT = P.remove_weights(signal_map=mT_weighted, weights_map=wT, comps='T')[0]
        
        enmap.write_hdf(map_hdf, mT, address=det,
                        extra={'xi0': xi0, 'eta0': eta0, 
                               'xi_bs_offset': xi_bs_offset, 'eta_bs_offset': eta_bs_offset, 'roll_bs_offset': roll_bs_offset})
    return

def detect_peak_xieta(mT, filter_size=None):
    """
    Detects the peak in a given pixcell map and converts it to ξ and η coordinates.

    Parameters:
    - mT (enmap.ndmap): a map object
    - filter_size (int, optional): Size of the filter window for peak detection.
                                   If not provided, it's calculated as a fraction 
                                   of the minimum dimension of mT.

    Returns:
    - xi_peak (float): xi coordinate of the peak.
    - eta_peak (float): eta coordinate of the peak.
    - ra_peak (float): ra coordinate of the peak.
    - dec_peak (float): dec coordinate of the peak.
    - peak_i (int): Row index of the peak.
    - peak_j (int): Column index of the peak.
    """
    if filter_size is None:
        filter_size = int(np.min(mT.shape)//10)
    local_max = maximum_filter(mT, footprint=np.ones((filter_size, filter_size)), 
                               mode='constant', cval=np.nan)
    peak_i, peak_j = np.where(mT == np.nanmax(local_max))
    peak_i = int(np.median(peak_i))
    peak_j = int(np.median(peak_j))
    dec_grid, ra_grid = mT.posmap()
    
    ra_peak = ra_grid[peak_i][peak_j]
    dec_peak = dec_grid[peak_i][peak_j]
    xi_peak, eta_peak = _radec2xieta(ra_peak, dec_peak)
    return xi_peak, eta_peak, ra_peak, dec_peak, peak_i, peak_j

def get_center_of_mass(x, y, z, 
                       circle_mask={'x0':0, 'y0':0, 'r_circle':3.0*coords.DEG},
                       percentile_mask = {'q': 50}):
    """
    Calculates the center of mass of a dataset within specified masks.

    Parameters:
    - x (ndarray): Array of x-coordinates.
    - y (ndarray): Array of y-coordinates.
    - z (ndarray): Array of data values corresponding to the coordinates.
    - circle_mask (dict, optional): Parameters defining circular mask.
                                    Should contain keys 'x0', 'y0', and 'r_circle'.
                                    Defaults to a circle centered at (0, 0) with radius 3.0 degrees.
    - percentile_mask (dict, optional): Parameters defining percentile mask.
                                        Should contain key 'q' representing the percentile threshold.
                                        Defaults to the 50th percentile.

    Returns:
    - x_center (float): x-coordinate of the center of mass.
    - y_center (float): y-coordinate of the center of mass.
    """
    mask = ~np.isnan(z)
    if circle_mask is not None:
        x0, y0 = circle_mask['x0'], circle_mask['y0']
        r_circle = circle_mask['r_circle']
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        mask = np.logical_and(mask, r<r_circle)
        
    if percentile_mask is not None:
        q = percentile_mask['q']
        mask = np.logical_and(mask, z>np.nanpercentile(z[mask], q))
    
    _x = x[mask]
    _y = y[mask]
    _z = z[mask]
    
    total_mass = np.nansum(_z)
    x_center = np.nansum(_x * _z) / total_mass
    y_center = np.nansum(_y * _z) / total_mass
    return x_center, y_center

def get_edgemap(mT, edge_avoidance=1*coords.DEG, edge_check='nan'):
    """
    Generates an edge map for a given map, marking regions near the edges where data is potentially unreliable.

    Parameters:
    - mT (enmap.ndmap): a map object
    - edge_avoidance (float, optional): Size of the edge avoidance region, defaults to 1 degree.
    - edge_check (str, optional): Method for checking edges. Should be one of {'nan', 'zero'}.
                                  'nan': Checks for NaN values at edges.
                                  'zero': Checks for zero values at edges.
                                  Defaults to 'nan'.

    Returns:
    - edge_map (enmap.ndmap): 2D boolean array representing the edge map, where True indicates regions near the edges.
    """
    if edge_check not in ('nan', 'zero'):
        raise ValueError('only `nan` or `zero` is supported')
    
    edge_map = enmap.zeros(mT.shape, mT.wcs)
    edge_margin_size = int(edge_avoidance/np.mean(mT.pixshape()))
    
    for i, row in enumerate(mT):
        if edge_check == 'nan':
            nonzero_idxes = np.where(~np.isnan(row))[0]
        elif edge_check == 'zero':
            nonzero_idxes = np.where(row != 0)[0]
        if len(nonzero_idxes>0):
            edge_map[i, :nonzero_idxes[0] + edge_margin_size] = True
            edge_map[i, nonzero_idxes[-1] - edge_margin_size:] = True
        else:
            edge_map[i, :] = True
            
    for j, col in enumerate(mT.T):
        if edge_check == 'nan':
            nonzero_idxes = np.where(~np.isnan(col))[0]
        elif edge_check == 'zero':
            nonzero_idxes = np.where(col != 0)[0]
        if len(nonzero_idxes>0):
            edge_map[:nonzero_idxes[0] + edge_margin_size, j] = True
            edge_map[nonzero_idxes[-1] - edge_margin_size:, j] = True
        else:
            edge_map[:, j] = True
    return edge_map


    
def map_to_xieta(mT, edge_avoidance=1.0*coords.DEG, edge_check='nan',
                 r_tune_circle=1.0*coords.DEG, q_tune=50, 
                 r_fit_circle=3.0*coords.DEG, beam_sigma_init=0.5*coords.DEG, ):
    """
    Derive (xi,eta) coordinate of a peak from a given map and calculates the coefficient of determination (R^2) 
    as a measure of how well the data fits a Gaussian model around the peak.

    Parameters:
    - mT (enmap.ndmap): a map object.
    - edge_avoidance (float, optional): Size of the edge avoidance region, defaults to 1 degree.
    - edge_check (str, optional): Method for checking edges. Should be one of {'nan', 'zero'}. Defaults to 'nan'.
    - r_tune_circle (float, optional): Radius of the circle used for tuning the peak position, specified in radians. Defaults to 1 degree.
    - q_tune (int, optional): Percentile threshold used for tuning the peak position. Defaults to 50.
    - r_fit_circle (float, optional): Radius of the circle used for fitting the Gaussian model, specified in radians. Defaults to 3 degrees.
    - beam_sigma_init (float, optional): Initial guess for the sigma parameter of the Gaussian beam, specified in radians. Defaults to 0.5 degree.

    Returns:
    - xi_det (float): xi coordinate of the detected peak.
    - eta_det (float): eta coordinate of the detected peak.
    - R2_det (float): Coefficient of determination (R^2) indicating the goodness of fit of the data around the peak.
                      If no valid peak is detected or if fitting fails, returns NaN.
    """
    if np.all(np.isnan(mT)):
        xi_det, eta_det, R2_det = np.nan, np.nan, np.nan
        
    else:
        xi_peak, eta_peak, ra_peak, dec_peak, peak_i, peak_j = detect_peak_xieta(mT)
        if edge_avoidance > 0.:
            edge_map = get_edgemap(mT, edge_avoidance=edge_avoidance, edge_check=edge_check)
            edge_valid = not edge_map[peak_i, peak_j]
        else:
            edge_valid = True
        
        if edge_valid:
            dec_flat, ra_flat = mT.posmap()
            dec_flat, ra_flat = dec_flat.flatten(), ra_flat.flatten()
            xi_flat, eta_flat = _radec2xieta(ra_flat, dec_flat)

            circle_mask = {'x0':xi_peak, 'y0':eta_peak, 'r_circle':r_tune_circle}
            percentile_mask = {'q': q_tune}
            xi_peak, eta_peak = get_center_of_mass(xi_flat, eta_flat, mT.flatten(), 
                                                  circle_mask=circle_mask, percentile_mask=percentile_mask)

            # check R2(=coefficient of determination)
            r = np.sqrt((xi_flat - xi_peak)**2 + (eta_flat - eta_peak)**2)
            z = mT.flatten()
            mask_fit = np.logical_and(~np.isnan(z), r<r_fit_circle)
            _r = r[mask_fit]
            _z = z[mask_fit]
            
            if _r.shape[0] == 0:
                xi_det, eta_det, R2_det = np.nan, np.nan, np.nan
            else:
                # Fit for R2 calculation (=coefficient of determination)
                popt, pcov = curve_fit(_gauss1d, _r, _z,
                                       p0=[np.ptp(_z), beam_sigma_init, np.percentile(_z, 1)],
                                       bounds = ((-np.inf, beam_sigma_init/5, -np.inf),
                                                  (np.inf, beam_sigma_init*5, np.inf),),
                                       max_nfev = 1000000)
                R2 = 1 - np.sum((_z - _gauss1d(_r, *popt))**2)/np.sum((_z - np.mean(_z))**2)
                xi_det, eta_det, R2_det = -xi_peak, eta_peak, R2
        else:
            xi_det, eta_det, R2_det = np.nan, np.nan, np.nan
    return xi_det, eta_det, R2_det

def get_xieta_from_maps(map_hdf_file, 
                        edge_avoidance=1.0*coords.DEG, edge_check='nan',
                        r_tune_circle=1.0*coords.DEG, q_tune=50,
                        save=False, output_dir=None, filename=None, force_zero_roll=False):
    """
    Process a set of maps stored in an HDF5 file to calculate ξ and η coordinates for each detector.

    Parameters:
    - map_hdf_file (str): Path to the HDF5 file containing the maps.
    - edge_avoidance (float, optional): Size of the edge avoidance region, specified in radians. Defaults to 1 degree.
    - edge_check (str, optional): Method for checking edges. Should be one of {'nan', 'zero'}. Defaults to 'nan'.
    - r_tune_circle (float, optional): Radius of the circle used for tuning the peak position, specified in radians. Defaults to 1 degree.
    - q_tune (int, optional): Percentile threshold used for tuning the peak position. Defaults to 50.
    - save (bool, optional): Whether to save the results to an output file. Defaults to False.
    - output_dir (str, optional): Directory path to save the output file. If not provided, saves in the current directory. Defaults to None.
    - filename (str, optional): Name of the output file. If not provided, a default name is generated. Defaults to None.
    - force_zero_roll (bool, optional): Whether to force zero roll, pretend as if roll of boresight is zero,
        before calculating xi and eta coordinates. Defaults to False.

    Returns:
    - focal_plane (ResultSet): ResultSet containing the results for each detector, including readout ID, band, 
                                        channel, R2 value, xi coordinate, eta coordinate, and gamma value.
    """
    _input_file = h5py.File(map_hdf_file, 'r')
    
    with _input_file as ifile:
        keys = list(ifile.keys())
        dets = [key for key in keys if isinstance(ifile[key], h5py.Group)]
        xieta_dict = {}
        for di, det in enumerate(tqdm(dets)):
            mT = enmap.read_hdf(ifile[det])
            mT[mT==0.] = np.nan
            xi, eta, R2 = map_to_xieta(mT, edge_avoidance=edge_avoidance, edge_check=edge_check,
                                       r_tune_circle=r_tune_circle, q_tune=q_tune)
            
            xi0 = ifile[det]['xi0'][...].item()
            eta0 = ifile[det]['eta0'][...].item()
            xi, eta = _add_xieta((xi0, eta0), (xi, eta))
            if force_zero_roll:
                xieta_dict[det] = {'xi':xi, 'eta':eta, 'R2':R2}
            else:
                xi_bs_offset = ifile[det]['xi_bs_offset'][...].item()
                eta_bs_offset = ifile[det]['eta_bs_offset'][...].item()
                roll_bs_offset = ifile[det]['roll_bs_offset'][...].item()
                
                q1 = quat.rotation_xieta(xi, eta)
                q2 = quat.rotation_xieta(xi_bs_offset, eta_bs_offset)
                q3 = quat.rotation_iso(0, 0, -roll_bs_offset) #
                q = q3 * ~q2 * q1
                xieta = quat.decompose_xieta(q)
                xi, eta = xieta[0], xieta[1]
                xieta_dict[det] = {'xi':xi, 'eta':eta, 'R2':R2}
    if save:
        if output_dir is None:
            if force_zero_roll:
                output_dir = os.path.join(os.getcwd(), 'pointing_results_force_zero_roll')
            else:
                output_dir = os.path.join(os.getcwd(), 'pointing_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if filename is None:
            filename = 'focal_plane_' + os.path.splitext(os.path.basename(map_hdf_file))[0] + '.hdf'
        output_file = os.path.join(output_dir, filename)
        
        focal_plane = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'R2', 'xi', 'eta', 'gamma'])
        for det in dets:
            band = int(det.split('_')[-2])
            channel = int(det.split('_')[-1])
            focal_plane.rows.append((det, band, channel, xieta_dict[det]['R2'],
                                     xieta_dict[det]['xi'], xieta_dict[det]['eta'], 0.))
        write_dataset(focal_plane, output_file, 'focal_plane', overwrite=True)
    return focal_plane

def _radec2xieta(ra, dec, ra0=0, dec0=0):
    q0 = quat.rotation_lonlat(lon=ra0, lat=dec0)
    q = quat.rotation_lonlat(lon=ra, lat=dec)
    xi, eta, _ = quat.decompose_xieta(~q0 * q)
    return xi, eta

def _gauss1d(x, peak, sigma, base):
    return peak * np.exp( - x**2 / (2*sigma**2)) + base

def _add_xieta(xieta1, xieta2):
    xi1, eta1 = xieta1
    xi2, eta2 = xieta2
    
    q1 = quat.rotation_xieta(xi1, eta1)
    q2 = quat.rotation_xieta(xi2, eta2)
    
    q_add = q2*q1
    xi_add, eta_add, _ = quat.decompose_xieta(q_add)
    return xi_add, eta_add

