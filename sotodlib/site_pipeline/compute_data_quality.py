import numpy as np
import yaml
import os
import warnings
from sotodlib import coords
import argparse
import scipy
from pixell import enmap
from sotodlib.site_pipeline.utils.pipeline import main_launcher
import importlib
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# sys.path.append('/home/gs8865/work/scripts/pwg-scripts/flp/planet_mapmaker')
# from planet_mapmaking import planet_mapmake_eachobs

# running the planet mapmaker will produce the maps and database, and store it in a location specified by the paths in the config file --- which paths?

def expand(obj, vars):
    """
    Function to expand the variables in the configuration file.
    Args:
        obj: Object to expand.
        vars: Variables to expand.
    Returns:
        obj: Expanded object.
    """
    if isinstance(obj, str):
        try:
            return obj.format(**vars)
        except (KeyError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: expand(v, vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v, vars) for v in obj]
    return obj
    
def read_configs(config_path):
    """Function to read the configuration file with expanding variables.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        cfg (dict): Config file after expanding.
    """
    configs = yaml.safe_load(open(config_path, "r"))
    cfg = configs.copy()
    for _ in range(3):
        cfg = expand(cfg, cfg)
    return cfg

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Fits elliptical gaussian: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    Args:
        xy (2D array): map coordinates
        amplitude (float): Amplitude of Peak
        xo (float): X0 center
        yo (float): Y0 center
        sigma_x (float): variance in X
        sigma_y (float): variance in Y
        theta (float): rotation parameter
        offset (float): A0 + Aexp(...); A0 is the offset.

    Returns:
        numpy.array(dtype=float): Flattened array of 2D gaussian given inputs.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def calc_rad_profile(map, binsize=0.5, normalize=True, positive_only=False):
    """Fucntion to calculate the radial profile of an input beam map.
    Assumes that the beam center is at the center of the map. 

    Args:
        map (ndmap): 2D ndmap to profile
        binsize (float, optional): radial bin size
        normalize (bool, optional): Normalizes profile to center pixel
    Returns:
        rad_prof (dictionary): Keys of 'radius', 'profile', and 'bin_stdev'
    """
    rad_prof = {'radius':[], 'profile':[],'bin_stdev':[]}    
    rad = np.array([])
    amps = np.array([])
    stdevs = np.array([])
    
    # setup
    pos = map.posmap()
    ra = pos[1]
    dec= pos[0]    
    r = np.sqrt(ra**2 + dec**2) # radians
    r = (r/coords.DEG) * 60 # arcminutes
    r_max = np.nanmax(r)
    r_bins= np.arange(0, r_max, binsize)
    
    # take radial average
    if positive_only:
        for i in range(len(r_bins)-1):
            idx = (r > r_bins[i]) & (r < r_bins[i+1]) & (map >= 0)
            # idx = (r > r_bins[i]) & (r < r_bins[i+1])
            ravg = np.nanmean(map[idx])
            vars = np.nanstd(map[idx])
            
            rad = np.append(rad,(r_bins[i] + r_bins[i+1])/2)
            amps = np.append(amps,ravg)
            stdevs = np.append(stdevs, vars)
    else:
        for i in range(len(r_bins)-1):
            # idx = (r > r_bins[i]) & (r < r_bins[i+1]) & (map >= 0)
            idx = (r > r_bins[i]) & (r < r_bins[i+1])
            ravg = np.nanmean(map[idx])
            vars = np.nanstd(map[idx])
            
            rad = np.append(rad,(r_bins[i] + r_bins[i+1])/2)
            amps = np.append(amps,ravg)
            stdevs = np.append(stdevs, vars)
    
    
    #take peak for normalization as maximum pixel value within 2 arcmin of r=0, so as to avoid degeneracy with FWHM in fitting
    peakidx = r < 2
    peak = np.nanmean(map[peakidx])
    if normalize:
        amps /= peak
        amps= np.append(np.array([1]),amps)
        rad = np.append(np.array([0]), rad)
        stdevs = np.append(np.array([stdevs[0]]), stdevs)
        
    rad_prof['radius'] = rad
    rad_prof['profile']   = amps
    rad_prof['bin_stdev']= stdevs

    return rad_prof

def fit_gaussian_to_beam(maps):
    imap = maps[0]
    pos = maps.posmap()
    ra  = pos[1]
    dec = pos[0]
    max_pix = imap.max()
    Tp0 = [max_pix,0,0,0.003,0.003,0.0,0]
    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, [ra, dec], imap.ravel(),Tp0)
    
    return popt, pcov

def map_selection(band, obs_id, solved_maps=None, rad_fwhm_max = None, rad_fwhm_min = None):
    """Determines if provided map contributes to co-add.
    Based on FWHM.

    Args:
        obs_id (str): Check if obs_id has peculiarity not covered by basic cuts
        band (str): 'f090' or 'f150'
        solved_maps (ndmap): Map to check beam properties
        rad_fwhm_max (float) : max fwhm [arcmin] for map selection
        rad_fwhm_min (float) : min fwhm [arcmin] for map selection
    Returns:
        True/False: bool
    """
    if rad_fwhm_max is None:
        if band == 'f090':
            rad_fwhm_max = 31
        else:
            rad_fwhm_max = 22

    if rad_fwhm_min is None:            
        if band == 'f090':
            rad_fwhm_min = 20
        else:
            rad_fwhm_min = 15
    try:
        rad_prof = calc_rad_profile(solved_maps[0])  # solved_maps[0] is imap
    except Exception as e:
        print(f'{obs_id} failed fit. Skipping.')
        return False, None
    try:
        profile = 10*np.log10(rad_prof['profile'])
        radius = rad_prof['radius']
        _3db = np.nanargmin(np.abs(profile+3)) #finding minimum distance to -3 dB
        fwhm = radius[_3db]*2        
        
    except Exception as e:
        print(e)
        print(f'{obs_id} does not have fwhm near -3db.')    
        return False, None
    
    if fwhm > rad_fwhm_max or fwhm < rad_fwhm_min:
        print(f'{obs_id} has poor fwhm fit: {fwhm}') 
        return False, None 
    if obs_id == 'obs_1724567567_satp1_1111111' or obs_id == 'obs_1724553289_satp1_1111111':
        print(f'{obs_id} skipped! -- seeing double!')
        return False, None
    if obs_id == 'obs_1732418203_satp3_1111111':
        print(f'{obs_id} is stripey (by eye). Skipping.')
        return False, None   
    return True, rad_prof
    
def calc_map_quality(maps, band, metric='resid', fwhm_max=None, fwhm_min=None, mask_rad=None):
    """Iterate through stored maps, and Calculate map quality based on fitted
    2D Gaussian. 

    Args:
        maps: list of solved maps
        metric (str, optional): Map quality metric Residuals or R2. Defaults to 'resid'.
        fwhm_max (float, optional): Max fitted fwhm in arcminutes. Defaults to None.
        fwhm_min (float, optional): Min fitted fwhm in arcminutes. Defaults to None.
        mask_rad (float, optional): Radius [arcmin] to compare fit to observed. Defaults to None.
    """

    size = len(maps)
    q_scores = np.empty(size)
    popts = np.empty([size, 7])
    if fwhm_max is None:
        if band == 'f090':
            fwhm_max = 31
        else:
            fwhm_max = 22
            
    if fwhm_min is None:
        if band == 'f090':
            fwhm_min = 20
        else:
            fwhm_min = 15
            
    if mask_rad is None:
        if band == 'f090':
            mask_rad = 29
        else:
            mask_rad = 20
            
    for idx, m in enumerate(maps):
        filename = os.path.basename(m)
        # wafer = filename.split("_")[-1].split(".")[0]
        obs_id = filename.rsplit("_", 1)[0]
        solved_maps = enmap.read_fits(m)

        success, rad_prof = map_selection(band, obs_id, solved_maps=solved_maps,
                                              rad_fwhm_max=fwhm_max, rad_fwhm_min=fwhm_min)
        
        if success:
            try:  
                popt, _, = fit_gaussian_to_beam(solved_maps)
            except Exception as e:
                print(f'{obs_id} failed 2D Gauss fit: {e}')
                quality = np.inf
                popt= np.full(7,np.nan)
                q_scores[idx] = quality
                popts[idx] = popt
                continue
            
            # print(f'{obs_id}, success: {success}')
            pos = solved_maps[0].posmap()
            ra = pos[1]
            dec = pos[0]
            x_vals = np.rad2deg(ra)*60
            y_vals = np.rad2deg(dec)*60
            r = np.sqrt(x_vals**2 + y_vals**2)
                  
            twoD_data = twoD_Gaussian([ra,dec], *popt)
            twoD_data = twoD_data.reshape(*solved_maps[0].shape) 

            mask = r<mask_rad
            masked_map = solved_maps[0][mask]
            masked_fit = twoD_data[mask]
            if metric == 'resid':
                masked_map /= popt[0]
                masked_fit /= popt[0]
                residuals = masked_map - masked_fit
                quality = np.nanstd(residuals)
                
        else:
            quality=np.inf
            popt=np.full(7,np.nan)
            
        q_scores[idx] = quality
        popts[idx] = popt
    
    return q_scores, popts


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='path to config file')
    parser.add_argument('--freq_channel', type=str, help='frequency channel of observation')
    parser.add_argument('--wafer', type=str, help='wafer of observation')
    parser.add_argument('--obs_id', type=str, default=None, help='obs id to be mapped')
    # parser.add_argument('--nproc', type=int, default=2, help='Number of processes to use')
    parser.add_argument('--verbosity', type=int, default=2, help='Number for logger verbosity')

    return parser

def main(config_file: str, wafer: str, freq_channel: str, obs_id: str, verbosity: int):
    configs = read_configs(config_file)
    
    repo_root = Path(configs["mapmaker_path"])
    module_path = repo_root / "planet_mapmaking.py"
    
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("planet_mapmaking", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    ws = wafer
    band = freq_channel
    dets={'wafer_slot': ws, 'wafer.bandpass': band, }

    module.planet_mapmake_eachobs(config_file, obs_id, dets, verbosity)
    
    map_path =  configs['mapmaking']['map']['save_dire']   

    maps_dire_band = os.path.join(map_path, band, "map")
    matching_files = [os.path.join(maps_dire_band, f) for f in os.listdir(maps_dire_band) if f.startswith(obs_id) and f.endswith(".fits")]

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No map found for obs_id={obs_id}")
    map_file = matching_files[0]
    maps = [map_file]    
    
    qscores, popts = calc_map_quality(maps, band)
    threshold_quality = 0.05
    if qscores[0] > threshold_quality:
        print( f'Observation {obs_id} quality is not good enough '
           f'(RMSE={qscores[0]:.4f})')
    else: 
        print(f'{obs_id} is good')
    
# if __name__ == '__main__':
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args.config_file, args.wafer, args.freq_channel, args.obs_id, args.verbosity)

if __name__ == '__main__':
    main_launcher(main, get_parser)