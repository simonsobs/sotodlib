import os
import numpy as np
import sotodlib
from astropy import units
au = units.au.si.to(units.m)


# fiducial models of planets
# R_eq, R_pole, Omega_ref, and d_ref are comes from Table2 of Weiland et al. (2010)
# SEVEN-YEAR WMAP OBSERVATIONS: PLANETS AND CELESTIAL CALIBRATION SOURCES
mars_fiducial = {
                'R_eq': 3396e3,
                'R_pole': 3376e3,
                'Omega_ref': 7.153e-10,
                'd_ref': 1.5 * au,
                'Trj': {'f090': None,
                        'f150': None}
                }
jupiter_fiducial = {
                'R_eq': 71492e3,
                'R_pole': 66854e3,
                'd_ref': 5.2 * au,
                'Omega_ref': 2.481e-8,
                'Trj': {'f090': 166.73, # from ESA1 * SO bandpass
                        'f150': 166.81}
                }
saturn_fiducial =  {
                'R_eq': 60268e3,
                'R_pole': 54364e3,
                'd_ref': 9.5 * au,
                'Omega_ref': 5.096e-9,
                'Trj': {'f090': None,
                        'f150': None}
                }
uranus_fiducial =  {
                'R_eq': 25559e3,
                'R_pole': 24973e3,
                'Omega_ref': 2.482e-10,
                'd_ref': 19. * au,
                'Trj': {'f090': 130.76, # from ESA4 * SO bandpass
                        'f150': 104.10}
                }
neptune_fiducial = {
                'R_eq': 24764e3,
                'R_pole': 24341e3,
                'Omega_ref': 1.006e-10,
                'd_ref': 29. * au,
                'Trj': {'f090': 121.77, # from ESA4 * SO bandpass
                        'f150': 108.10}
                }

fiducial_models = {
    'mars': mars_fiducial,
    'jupiter': jupiter_fiducial,
    'saturn': saturn_fiducial,
    'uranus': uranus_fiducial,
    'neptune': neptune_fiducial
                    }

def distance_correction_factor(planet):
    R_eq = fiducial_models[planet]['R_eq']
    R_pole = fiducial_models[planet]['R_pole']
    A_ref = np.pi * R_pole * R_eq
    
    Dw = 0 # this should be provided as argument later
    R_proj_pole = R_pole * np.sqrt(1 - np.sin(Dw)**2 * (1 - (R_pole/R_eq)**2) )
    A_proj_disk = np.pi * R_proj_pole * R_eq
    
    f_A = A_proj_disk / A_ref
    return f_A

def disk_oblateness_correction_factor():
    return

def calc_model_temperature(bandpass_name, bandpass_suffix,
                          planet_name, model_name):
    """
    Args:
        bandpass_name (str): Name of SO's bandpass (e.g. 'f090', 'f150')
        bandpass_suffix (str): Suffix of the bandpass file (e.g., 'design', 'measured').
        planet_name (str): Name of the planet (e.g. 'uranus', 'neptune')
        model_name (str): Name of the planet model file without the extension.
    Returns:
    """
    bandpass_dir = os.path.join(os.path.dirname(sotodlib.__file__), 'calibration', 'bandpass')
    bandpass_file = os.path.join(bandpass_dir, f'{bandpass_name}_{bandpass_suffix}.txt')
    bandpass_data = np.loadtxt(bandpass_file, comments='#')
    freq_GHz_bandpass = bandpass_data[:, 0]
    trans_bandpass = bandpass_data[:, 1]
    
    planet_model_dir = os.path.join(os.path.dirname(sotodlib.__file__), 'calibration', 'planet_models')
    planet_model_file = os.path.join(planet_model_dir, planet_name, f'{planet_name}_{model_name}.txt')
    planet_model_data = np.loadtxt(planet_model_file, comments='#')
    freq_GHz_planet = planet_model_data[:, 0]
    Trj_planet = planet_model_data[:, 1]
    
    Trj_planet_interp = np.interp(freq_GHz_bandpass, freq_GHz_planet, Trj_planet)
    
    
    integral_numer = np.trapz(trans_bandpass * Trj_planet_interp, freq_GHz_bandpass)
    integral_denom = np.trapz(trans_bandpass, freq_GHz_bandpass)
    band_averaged_Trj = integral_numer / integral_denom
    return band_averaged_Trj