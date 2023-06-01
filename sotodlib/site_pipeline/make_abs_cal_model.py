import os
import numpy as np
import yaml

from pixell import enmap
from sotodlib import core
from sotodlib.calibration import planet_ref
from . import util


def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', help=
                        "Configuration file.")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Pass multiple times to increase.")
    parser.add_argument('obs_id',help=
                        "Observation for which to make source map.")
    parser.add_argument('--test', action='store_true', help=
                        "Reduce detector count for quick tests.")

    return parser

def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def get_peak_and_Omega(imap):
    # This is temporal. Eventually, use the result from analyze-beam-map
    from scipy.optimize import curve_fit
    import numpy as np
    
    imap_flatten = imap.flatten()
    dec, ra = imap.posmap().reshape([2, imap.posmap().shape[1]*imap.posmap().shape[2]])
    decra = (dec, ra)
    def gaussian(decra, peak, sigma, dec0, ra0, offset):
        dec, ra = decra
        dtheta = np.sqrt((dec-dec0)**2 + (ra-ra0)**2)
        return offset + peak * np.exp(- dtheta**2 / 2 / sigma**2)

    peak0 = np.ptp(imap_flatten)
    sigma0 = np.deg2rad(10/60) # 10 arcmin

    # Perform the fitting
    p0 = [peak0, sigma0, 0, 0, 0]  # Initial guess for the parameters
    popt, pcov = curve_fit(gaussian, decra, imap_flatten, p0=p0)
    peak_fit, sigma_fit = popt[0], popt[1]
    Omega_fit = 2 * np.pi * sigma_fit**2
    return peak_fit, Omega_fit

def main(config_file=None, obs_id=None, groups=None, verbose=0):
    # set logger
    logger = util.init_logger(__name__, 'make_abs_cal_model: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')
        
    # load config file
    config = _get_config(config_file)
    # load context file
    ctx = core.Context(config['context_file'])
    # load uncal-beammap archive
    uncalmap_db = core.metadata.ManifestDb.from_file(config['uncalmap_archive'])
    
    # get obs
    obsdb = ctx.obsdb
    obs = obsdb.query(f'obs_id == "{obs_id}"')[0]
    
    # load infomation of uncal-beammap
    uncalmap_info_all_groups = uncalmap_db.match({'obs:obs_id': obs_id}, multi=True)
    
    #if groups is None:
    #    groups = [uncalmap_info['group'] for uncalmap_info in uncalmap_info_all_groups]
        
    for uncalmap_info in uncalmap_info_all_groups:
        group = uncalmap_info['group']
        if group in groups:
            uncalmap_file = os.path.join(os.path.dirname(config['uncalmap_archive']),
                                         uncalmap_info['filename'])
            uncalmap = enmap.read_map(uncalmap_file)
            
            peak_fit, Omega_fit = get_peak_and_Omega(uncalmap[0])
            planet_vals_dict = planet_ref.fiducial_models[obs['target']]
            
            expected_Trj_Omega = planet_ref.get_expected_Trj_Omega(planet=target, 
                                                   bandpass_name=uncalmap_info['split'],
                                                   timestamp=obs['timestamp'] + obs['duration']/2.)
            expected_Trj = expected_Trj_Omega / Omega_fit
            gain_factor = expected_Trj / peak_fit
            print(gain_factor)
    
    return
            
if __name__ == '__main__':
    util.main_launcher(main, get_parser)