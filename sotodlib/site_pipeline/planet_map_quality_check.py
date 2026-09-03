import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy
import yaml
from pixell import enmap
from sotodlib import coords, core
from sotodlib.io.metadata import ResultSet, write_dataset
from sotodlib.site_pipeline.utils import logging
from sotodlib.site_pipeline.utils.obsdb import get_obslist
from sotodlib.site_pipeline.utils.pipeline import main_launcher
import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)

# running the planet mapmaker will produce the maps and database, and store it in a location specified by the paths in the config file 

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
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
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
    
    # setup
    pos = map.posmap()
    ra = pos[1]
    dec= pos[0]    
    r = np.sqrt(ra**2 + dec**2) # radians
    r = (r/coords.DEG) * 60 # arcminutes
    r_max = np.nanmax(r)
    r_bins= np.arange(0, r_max + binsize, binsize)

    rad = np.empty(len(r_bins)-1)
    amps = np.empty(len(r_bins)-1)
    stdevs = np.empty(len(r_bins)-1)

    for i in range(len(r_bins)-1):
        idx = (r > r_bins[i]) & (r < r_bins[i+1])
        
        if positive_only:
            idx &= (map >= 0)
    
        values = map[idx]
        values = values[np.isfinite(values)]
    
        if len(values) > 0:
            amps[i] = np.mean(values)
            stdevs[i] = np.std(values)
        else:
            amps[i] = np.nan
            stdevs[i] = np.nan
    
        rad[i] = (r_bins[i] + r_bins[i+1])/2
    
    # take peak for normalization as the mean pixel value within 2 arcmin of r=0, so as to avoid degeneracy with FWHM in fitting
    peakidx = r < 2
    peak_values = map[peakidx]
    peak_values = peak_values[np.isfinite(peak_values)]
    
    if len(peak_values) > 0:
        peak = np.mean(peak_values)
    else:
        peak = np.nan
        
    if normalize:
        if np.isfinite(peak) and peak != 0:
            amps /= peak
        else:
            amps[:] = np.nan
    
        amps = np.append(np.array([1]), amps)
        rad = np.append(np.array([0]), rad)
        stdevs = np.append(np.array([stdevs[0]]), stdevs)
        
    rad_prof['radius'] = rad
    rad_prof['profile']   = amps
    rad_prof['bin_stdev']= stdevs

    return rad_prof
    
def fit_gaussian_to_beam(maps, init_params):
    imap = maps[0]
    pos = maps.posmap()
    ra  = pos[1]
    dec = pos[0]
    max_pix = imap.max()
    Tp0 = [max_pix] + list(init_params)
    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, [ra, dec], imap.ravel(),Tp0)
    
    return popt, pcov

def get_errors(e):
    errmsg = f'{type(e)}: {e}'
    tb = ''.join(traceback.format_tb(e.__traceback__))

    return errmsg, tb

def map_selection(band, obs_id, configs, solved_maps=None, verbosity = 3):
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
    logger = logging.init_logger("compute_data_quality", verbosity=verbosity)
    
    rad_fwhm_max = configs['query']['fwhm_max'][band]
    rad_fwhm_min = configs['query']['fwhm_min'][band]

    try:
        rad_prof = calc_rad_profile(solved_maps[0])  # solved_maps[0] is imap
    except Exception as e:
        # logger.info(f"{obs_id} failed fit. Skipping.")
        errmsg, tb = get_errors(e)
        logger.error(f" FWHM fit failed for {obs_id}: \n{errmsg}\n{tb}")
        return False, None
    try:
        profile = rad_prof['profile']
        radius = rad_prof['radius']
    
        valid = np.isfinite(profile) & (profile > 0)
    
        if not np.any(valid):
            raise ValueError("No valid positive beam profile values")
    
        target_3db = 10**(-3/10)  # 10log10(B) = -3
        _3db = np.argmin(np.abs(profile[valid] - target_3db))
        fwhm = radius[valid][_3db] * 2
    except Exception as e:
        errmsg, tb = get_errors(e)
        logger.error(f"No FWHM near -3dB for {obs_id}: \n{errmsg}\n{tb}")
        return False, None
    
    if fwhm > rad_fwhm_max or fwhm < rad_fwhm_min:
        logger.info(f"{obs_id} has poor fwhm fit: {fwhm}")
        return False, None 
    return True, rad_prof
    
def calc_map_quality(maps, band, configs, metric='resid', verbosity = 3):
    """Iterate through stored maps, and Calculate map quality based on fitted
    2D Gaussian. 

    Args:
        maps: list of solved maps
        metric (str, optional): Map quality metric Residuals or R2. Defaults to 'resid'.
        fwhm_max (float, optional): Max fitted fwhm in arcminutes. Defaults to None.
        fwhm_min (float, optional): Min fitted fwhm in arcminutes. Defaults to None.
        mask_rad (float, optional): Radius [arcmin] to compare fit to observed. Defaults to None.
    """
    n_params = 7
    size = len(maps)
    q_scores = np.empty(size)
    popts = np.empty([size, n_params])
    logger = logging.init_logger("compute_data_quality", verbosity=verbosity)

    mask_rad = configs['query']['mask_rad'][band]
            
    for idx, m in enumerate(maps):
        filename = os.path.basename(m)
        # wafer = filename.split("_")[-1].split(".")[0]
        obs_id = filename.rsplit("_", 1)[0]
        solved_maps = enmap.read_fits(m)

        success, _ = map_selection(band, obs_id, configs, solved_maps=solved_maps)
        
        if success:
            try:  
                popt, _, = fit_gaussian_to_beam(solved_maps, init_params = configs['query']['init_gauss_params'])
            except Exception as e:
                errmsg, tb = get_errors(e)
                logger.error(f" 2D Gauss fit failed for {obs_id}: \n{errmsg}\n{tb}")
                # logger.info(f"{obs_id} failed 2D Gauss fit: {e}")
                quality = np.inf
                popt= np.full(n_params, np.nan)
                q_scores[idx] = quality
                popts[idx] = popt
                continue
            
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
                quality = np.sqrt(np.nanmean(residuals**2))  # true RMSE
                
        else:
            quality=np.inf
            popt=np.full(n_params, np.nan)
            
        q_scores[idx] = quality
        popts[idx] = popt
    
    return q_scores, popts

def add_to_failed_cache(obs_id, band, ws, msg, failed_cache_file, logger):
    """Cache one failed (obs_id, band, wafer) combination."""
    # logger = logging.init_logger("compute_data_quality")

    if msg is None:
        msg = "unknown error"
    msg = str(msg)

    # Transient metadata errors should be retried rather than cached.
    transient_errors = [
        "sotodlib.core.metadata.loader.LoaderError",
        "BlockingIOError",
    ]
    for err in transient_errors:
        if err in msg:
            logger.error(
                f"obs_id {obs_id} failed to load metadata {err}. Try again later"
            )
            return False

    cache_key = f"{obs_id}_{band}_{ws}"
    logger.info(f"Adding {cache_key} to failed_file_cache")

    if os.path.exists(failed_cache_file):
        with open(failed_cache_file, "r") as f:
            d = yaml.safe_load(f) or {}
    else:
        d = {}

    d[cache_key] = msg

    with open(failed_cache_file, "w") as f:
        yaml.safe_dump(d, f)

    return True


def get_quality_db(index_path):
    """
    The index key is (obs:obs_id, band, wafer).  The HDF5 dataset address is
    stored explicitly so every tuple can point to its own dataset.
    """
    required_fields = {"obs:obs_id", "band", "wafer", "success", "rmse", "dataset"}

    if os.path.exists(index_path):
        db = core.metadata.ManifestDb(index_path)
        scheme_rs = db.scheme.as_resultset()
        existing_fields = set(scheme_rs["field"])
        missing = required_fields - existing_fields
        if missing:
            raise RuntimeError(
                f"Existing manifest {index_path} has the old/incompatible schema. "
                f"Missing fields: {sorted(missing)}. Move/delete the old index and "
                "rerun so a tuple-aware manifest can be created."
            )
        return db

    scheme = core.metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_exact_match("band")
    scheme.add_exact_match("wafer")
    scheme.add_data_field("success")
    scheme.add_data_field("rmse")
    scheme.add_data_field("dataset")
    return core.metadata.ManifestDb(index_path, scheme=scheme)


def handle_result(obs_id, rmse, wafer, band, success, h5_path, h5_unix_digits, index_path, failed_cache_file, logger, msg=None, popts=None):
    """Store/cache exactly one (obs_id, band, wafer) result.

    Deliberately rejects a wafer list so one RMSE can never accidentally be
    copied to several wafers.
    """
    if not isinstance(wafer, str):
        raise TypeError(
            "handle_result expects one wafer string, e.g. 'ws3'. "
            "Call it once per wafer so RMSE values cannot be duplicated."
        )

    if not success:
        if msg is None:
            msg = "unknown error"
        return add_to_failed_cache(obs_id, band, wafer, msg, failed_cache_file, logger)

    obsid_result = np.array(
        [(obs_id, rmse, wafer, band, True, *popts)],
        dtype=[
            ("obs_id", "U50"), ("rmse", "f8"), ("wafer_slot", "U10"),
            ("wafer.bandpass", "U10"), ("success", "?"),
            ("amp", "f8"), ("x0", "f8"), ("y0", "f8"), ("sigma_x", "f8"),
            ("sigma_y", "f8"), ("theta", "f8"), ("offset", "f8")])
    rset = ResultSet.from_friend(obsid_result)

    if h5_unix_digits:
        name, ext = os.path.splitext(h5_path)
        obs_parts = str(obs_id).split("_")
        if len(obs_parts) < 2:
            raise ValueError(
                f"Cannot extract unix time from obs_id={obs_id!r} for h5_unix_digits"
            )
        unixtime = obs_parts[1][:h5_unix_digits]
        h5_path = f"{name}_{unixtime}{ext}"

    dataset_name = f"{obs_id}_{band}_{wafer}"
    write_dataset(rset, h5_path, dataset_name, overwrite=True)

    db = get_quality_db(index_path)
    relpath = os.path.relpath(h5_path, start=os.path.dirname(index_path))
    db.add_entry({"obs:obs_id": obs_id, "band": band, "wafer": wafer,
            "success": True, "rmse": float(rmse), "dataset": dataset_name},
        filename=relpath, replace=True, )
    return True


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='path to config file')
    parser.add_argument('--query', type=str, default=None, help='query string for obsdb')
    parser.add_argument('--update_delay', type=int, default=None, help='number of days to subtract from the current time to set the minimum ctime')
    parser.add_argument('--planet_tags', type=str, nargs='+', default=[], help='planet names to be tagged')
    parser.add_argument('--verbosity', type=int, default=2, help='Number for logger verbosity')
    parser.add_argument('--h5_path', type=str, default='quality_check.h5', help='Path to store hdf5 files')
    parser.add_argument('--h5_unix_digits', type=int, default=4, help='unix digits to store with h5 path')
    parser.add_argument('--index_path', type=str, default='quality_check.sqlite', help='Path to store db')
    parser.add_argument('--failed_cache_file', type=str, default='failed_obsids.yaml', help='Path to store failed obsids')
    parser.add_argument('--overwrite_map', type=bool, help='Re-run mapmaker even if map already exists')

    return parser

def main(config_file: str, query: str, update_delay: int, verbosity: int, h5_path: str, 
         h5_unix_digits: int, index_path: str, failed_cache_file: str, planet_tags: list[str] | None = None, overwrite_map: bool = False):

    configs = read_configs(config_file)
    context = core.Context(configs["context_file"])
    map_path =  configs['mapmaking']['map']['save_dire']   
    bands = configs['query'].get('bands')
    if not bands:
        raise ValueError("No bands configured at configs['query']['bands']")
    
    repo_root = Path(configs["mapmaker_path"])
    module_path = repo_root / "planet_mapmaking.py"
    
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("planet_mapmaking", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger = logging.init_logger("compute_data_quality", verbosity=verbosity)
    
    obsid_list = get_obslist(context=context, query=query, update_delay=update_delay, tags=planet_tags, planet_obs=True)
    obs_ids = list(dict.fromkeys( obs["obs_id"] for obs in obsid_list))

    if os.path.exists(failed_cache_file):
        with open(failed_cache_file, "r") as f:
          d = yaml.safe_load(f) or {}
    else:
        d = {}

    failed_obsids = set(d.keys())

    for obs_id in obs_ids:
        for band in bands:
            try:
                obs = context.obsdb.get(str(obs_id), tags=True)
                tags = obs["tags"]
                wafers = [tag for tag in tags if tag.startswith("ws")]

                if not wafers:
                    raise ValueError(
                        f"No wafer tags found for obs_id={obs_id}. Available tags: {tags}"
                    )

            except Exception as e:
                # At this point we do not reliably know the wafer(s), so do not
                # fabricate tuple-level cache entries.
                errmsg, tb = get_errors(e)
                logger.error(
                    f"Could not determine wafers for {obs_id} in {band}:\n{errmsg}\n{tb}"
                )
                continue

            # Process each wafer independently.  This guarantees that each
            # (obs_id, band, wafer) has its own map-quality calculation and
            # that a failure in one wafer does not affect the others.
            for ws in wafers:
                cache_key_name = f"{obs_id}_{band}_{ws}"

                if cache_key_name in failed_obsids:
                    logger.info(
                        f"{obs_id} is known to be bad for {band} and {ws}. "
                        "Skipping this wafer."
                    )
                    continue

                try:
                    logger.info(f"Processing obs_id={obs_id}, band={band}, wafer={ws}")
                    dets = {"wafer_slot": [ws], "wafer.bandpass": band}
                    map_file = os.path.join(map_path, band, "map", f"{obs_id}_{ws}.fits")
                
                    if (not os.path.exists(map_file)) or overwrite_map:
                        if overwrite_map and os.path.exists(map_file):
                            logger.info(f"Overwriting existing map for obs_id={obs_id}, band={band}, wafer={ws}")
                
                        module.planet_mapmake_eachobs(config_file, obs_id, dets, verbosity)
                
                    else:
                        logger.info(f"Map already exists for obs_id={obs_id}, band={band}, wafer={ws}. Using existing map.")
                
                    if not os.path.exists(map_file):
                        raise FileNotFoundError(f"No map found for obs_id={obs_id}, band={band}, wafer={ws}: {map_file}")
                
                    qscores, popts = calc_map_quality([map_file], band, configs)
                    quality_score = float(qscores[0])

                    threshold_quality = configs['query']['threshold']

                    if (not np.isfinite(quality_score)) or (quality_score > threshold_quality):
                        logger.info(f"Bad map: obs_id={obs_id}, band={band}, wafer={ws}, "
                            f"RMSE={quality_score}")

                        cached = handle_result(obs_id, quality_score, ws, band, False, h5_path=h5_path,
                            h5_unix_digits=h5_unix_digits, index_path=index_path, failed_cache_file=failed_cache_file,
                            logger=logger, msg=f"Large RMSE: {quality_score}")
                        if cached:
                            failed_obsids.add(cache_key_name)

                    else:
                        logger.info(f"Good map: obs_id={obs_id}, band={band}, wafer={ws}, "
                            f"RMSE={quality_score:.6g}")

                        handle_result(obs_id, quality_score, ws, band, True, h5_path=h5_path,
                            h5_unix_digits=h5_unix_digits, index_path=index_path, failed_cache_file=failed_cache_file,
                            logger=logger, msg=None, popts=popts[0])
                        

                except Exception as e:
                    errmsg, tb = get_errors(e)
                    logger.error(f"Processing failed for obs_id={obs_id}, band={band}, "
                        f"wafer={ws}:\n{errmsg}\n{tb}")
                    cached = handle_result(obs_id, np.nan, ws, band, False, h5_path=h5_path,
                        h5_unix_digits=h5_unix_digits, index_path=index_path, failed_cache_file=failed_cache_file,
                        logger=logger, msg=errmsg)
                    if cached:
                        failed_obsids.add(cache_key_name)
                    continue

if __name__ == '__main__':
    main_launcher(main, get_parser)