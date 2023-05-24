import os
import numpy as np
import sotodlib


fiducial_models_rj = {
                 'uranus' : {'f090':130.76, 'f150':104.10}, #from ESA4
                 'neptune': {'f090':121.77, 'f150':108.10}  #from ESA4
                }

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